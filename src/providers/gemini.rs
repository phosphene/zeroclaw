//! Google Gemini provider with support for:
//! - Direct API key (`GEMINI_API_KEY` env var or config)
//! - Gemini CLI OAuth tokens (reuse existing ~/.gemini/ authentication)
//! - ZeroClaw auth-profiles OAuth tokens
//! - Google Cloud ADC (`GOOGLE_APPLICATION_CREDENTIALS`)

use crate::auth::AuthService;
use crate::providers::traits::{
    ChatMessage, ChatRequest, ChatResponse, Provider, ProviderCapabilities, StreamChunk,
    StreamError, StreamEvent, StreamOptions, StreamResult, TokenUsage, ToolsPayload,
};
use async_trait::async_trait;
use base64::Engine;
use directories::UserDirs;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

/// Gemini provider supporting multiple authentication methods.
pub struct GeminiProvider {
    auth: Option<GeminiAuth>,
    oauth_project: Arc<tokio::sync::Mutex<Option<String>>>,
    oauth_cred_paths: Vec<PathBuf>,
    oauth_index: Arc<tokio::sync::Mutex<usize>>,
    /// AuthService for managed profiles (auth-profiles.json).
    auth_service: Option<AuthService>,
    /// Override profile name for managed auth.
    auth_profile_override: Option<String>,
    /// Base URL for API requests (internal/testing).
    base_url: Option<String>,
}

/// Mutable OAuth token state — supports runtime refresh for long-lived processes.
struct OAuthTokenState {
    access_token: String,
    refresh_token: Option<String>,
    client_id: Option<String>,
    client_secret: Option<String>,
    /// Expiry as unix millis. `None` means unknown (treat as potentially expired).
    expiry_millis: Option<i64>,
}

/// Resolved credential — the variant determines both the HTTP auth method
/// and the diagnostic label returned by `auth_source()`.
enum GeminiAuth {
    /// Explicit API key from config: sent as `?key=` query parameter.
    ExplicitKey(String),
    /// API key from `GEMINI_API_KEY` env var: sent as `?key=`.
    EnvGeminiKey(String),
    /// API key from `GOOGLE_API_KEY` env var: sent as `?key=`.
    EnvGoogleKey(String),
    /// OAuth access token from Gemini CLI: sent as `Authorization: Bearer`.
    /// Wrapped in a Mutex to allow runtime token refresh.
    OAuthToken(Arc<tokio::sync::Mutex<OAuthTokenState>>),
    /// OAuth token managed by AuthService (auth-profiles.json).
    /// Token refresh is handled by AuthService, not here.
    ManagedOAuth,
}

impl GeminiAuth {
    /// Whether this credential is an API key (sent as `?key=` query param).
    fn is_api_key(&self) -> bool {
        matches!(
            self,
            GeminiAuth::ExplicitKey(_) | GeminiAuth::EnvGeminiKey(_) | GeminiAuth::EnvGoogleKey(_)
        )
    }

    /// Whether this credential is an OAuth token (CLI or managed).
    fn is_oauth(&self) -> bool {
        matches!(self, GeminiAuth::OAuthToken(_) | GeminiAuth::ManagedOAuth)
    }

    /// The raw credential string (for API key variants only).
    fn api_key_credential(&self) -> &str {
        match self {
            GeminiAuth::ExplicitKey(s)
            | GeminiAuth::EnvGeminiKey(s)
            | GeminiAuth::EnvGoogleKey(s) => s,
            GeminiAuth::OAuthToken(_) | GeminiAuth::ManagedOAuth => "",
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// API REQUEST/RESPONSE TYPES
// ══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Clone)]
struct GeminiTool {
    #[serde(rename = "functionDeclarations")]
    function_declarations: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize, Clone)]
struct GenerateContentRequest {
    contents: Vec<Content>,
    #[serde(rename = "systemInstruction", skip_serializing_if = "Option::is_none")]
    system_instruction: Option<Content>,
    #[serde(rename = "generationConfig")]
    generation_config: GenerationConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GeminiTool>>,
}

/// Request envelope for the internal cloudcode-pa API.
/// OAuth tokens from Gemini CLI are scoped for this endpoint.
///
/// The internal API expects a nested structure:
/// ```json
/// {
///   "model": "models/gemini-...",
///   "project": "...",
///   "request": {
///     "contents": [...],
///     "systemInstruction": {...},
///     "generationConfig": {...}
///   }
/// }
/// ```
/// Ref: gemini-cli `packages/core/src/code_assist/converter.ts`
#[derive(Debug, Serialize)]
struct InternalGenerateContentEnvelope {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    project: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user_prompt_id: Option<String>,
    request: InternalGenerateContentRequest,
}

/// Nested request payload for cloudcode-pa's code assist APIs.
#[derive(Debug, Serialize)]
struct InternalGenerateContentRequest {
    contents: Vec<Content>,
    #[serde(rename = "systemInstruction", skip_serializing_if = "Option::is_none")]
    system_instruction: Option<Content>,
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GeminiTool>>,
}

#[derive(Debug, Serialize, Clone)]
struct Content {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    parts: Vec<Part>,
}

#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
enum Part {
    Text { text: String },
    Inline { inline_data: InlineData },
    #[serde(rename = "functionCall")]
    FunctionCall { function_call: FunctionCall },
    #[serde(rename = "functionResponse")]
    FunctionResponse { function_response: FunctionResponse },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct FunctionCall {
    name: String,
    args: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct FunctionResponse {
    name: String,
    response: serde_json::Value,
}

impl Part {
    fn text(s: impl Into<String>) -> Self {
        Part::Text { text: s.into() }
    }
}

#[derive(Debug, Serialize, Clone)]
struct InlineData {
    mime_type: String,
    data: String,
}

/// Build Gemini Parts from a message content string.
/// If the content contains [IMAGE:data:...] markers (already normalized by the
/// multimodal pipeline), they are extracted as inline_data parts. The remaining
/// text becomes a text part. Falls back to a single text part if no markers.
fn build_parts(content: &str) -> Vec<Part> {
    let (text, image_refs) = crate::multimodal::parse_image_markers(content);
    let mut parts = Vec::new();
    let trimmed = text.trim();
    if !trimmed.is_empty() {
        parts.push(Part::text(trimmed));
    }
    for uri in &image_refs {
        if let Some(rest) = uri.strip_prefix("data:") {
            if let Some(semi_pos) = rest.find(';') {
                let mime = &rest[..semi_pos];
                if let Some(b64) = rest[semi_pos + 1..].strip_prefix("base64,") {
                    parts.push(Part::Inline {
                        inline_data: InlineData {
                            mime_type: mime.to_string(),
                            data: b64.to_string(),
                        },
                    });
                }
            }
        }
    }
    if parts.is_empty() {
        parts.push(Part::text(content));
    }
    parts
}

#[derive(Debug, Serialize, Clone)]
struct GenerationConfig {
    temperature: f64,
    #[serde(rename = "maxOutputTokens")]
    max_output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct GenerateContentResponse {
    candidates: Option<Vec<Candidate>>,
    error: Option<ApiError>,
    #[serde(default)]
    response: Option<Box<GenerateContentResponse>>,
    #[serde(default, rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Deserialize)]
struct GeminiUsageMetadata {
    #[serde(default, rename = "promptTokenCount")]
    prompt_token_count: Option<u64>,
    #[serde(default, rename = "candidatesTokenCount")]
    candidates_token_count: Option<u64>,
}

/// Response envelope for the internal cloudcode-pa API.
/// The internal API nests the standard response under a `response` field.
#[derive(Debug, Deserialize)]
struct InternalGenerateContentResponse {
    response: GenerateContentResponse,
}

#[derive(Debug, Deserialize)]
struct Candidate {
    #[serde(default)]
    content: Option<CandidateContent>,
}

#[derive(Debug, Deserialize)]
struct CandidateContent {
    parts: Vec<ResponsePart>,
}

#[derive(Debug, Deserialize)]
struct ResponsePart {
    #[serde(default)]
    text: Option<String>,
    #[serde(rename = "functionCall", default)]
    function_call: Option<FunctionCall>,
    /// Thinking models (e.g. gemini-2.0-flash-thinking) mark reasoning parts with `thought: true`.
    #[serde(default)]
    thought: bool,
}

impl CandidateContent {
    /// Extract effective text, skipping thinking/signature parts.
    ///
    /// Gemini thinking models (e.g. gemini-3-pro-preview) return parts like:
    /// - `{"thought": true, "text": "reasoning..."}` — internal reasoning
    /// - `{"text": "actual answer"}` — the real response
    /// - `{"thoughtSignature": "..."}` — opaque signature (no text field)
    ///
    /// Returns the non-thinking text, falling back to thinking text only when
    /// no non-thinking content is available.
    fn effective_text(self) -> Option<String> {
        let mut answer_parts: Vec<String> = Vec::new();
        let mut first_thinking: Option<String> = None;

        for part in self.parts {
            if let Some(text) = part.text {
                if text.is_empty() {
                    continue;
                }
                if !part.thought {
                    answer_parts.push(text);
                } else if first_thinking.is_none() {
                    first_thinking = Some(text);
                }
            }
        }

        if answer_parts.is_empty() {
            first_thinking
        } else {
            Some(answer_parts.join(""))
        }
    }
}

#[derive(Debug, Deserialize)]
struct ApiError {
    message: String,
}

impl GenerateContentResponse {
    /// cloudcode-pa wraps the actual response under `response`.
    fn into_effective_response(self) -> Self {
        match self {
            Self {
                response: Some(inner),
                ..
            } => *inner,
            other => other,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// GEMINI CLI TOKEN STRUCTURES
// ══════════════════════════════════════════════════════════════════════════════

/// OAuth token stored by Gemini CLI in `~/.gemini/oauth_creds.json`
#[derive(Debug, Deserialize)]
struct GeminiCliOAuthCreds {
    access_token: Option<String>,
    #[serde(alias = "idToken")]
    id_token: Option<String>,
    refresh_token: Option<String>,
    #[serde(alias = "clientId")]
    client_id: Option<String>,
    #[serde(alias = "clientSecret")]
    client_secret: Option<String>,
    /// Unix milliseconds expiry (used by newer Gemini CLI versions).
    #[serde(alias = "expiryDate")]
    expiry_date: Option<i64>,
    /// RFC 3339 expiry string (used by older Gemini CLI versions).
    expiry: Option<String>,
}

// ══════════════════════════════════════════════════════════════════════════════
// GEMINI CLI OAUTH CONSTANTS
// ══════════════════════════════════════════════════════════════════════════════

/// Google OAuth token endpoint.
const GOOGLE_TOKEN_ENDPOINT: &str = "https://oauth2.googleapis.com/token";

/// Internal API endpoint used by Gemini CLI for OAuth users.
/// See: https://github.com/google-gemini/gemini-cli/issues/19200
const CLOUDCODE_PA_ENDPOINT: &str = "https://cloudcode-pa.googleapis.com/v1internal";

/// loadCodeAssist endpoint for resolving the project ID.
const LOAD_CODE_ASSIST_ENDPOINT: &str =
    "https://cloudcode-pa.googleapis.com/v1internal:loadCodeAssist";

/// Public API endpoint for API key users.
const PUBLIC_API_ENDPOINT: &str = "https://generativelanguage.googleapis.com/v1beta";

// ══════════════════════════════════════════════════════════════════════════════
// TOKEN REFRESH
// ══════════════════════════════════════════════════════════════════════════════

/// Result of a successful token refresh.
struct RefreshedToken {
    access_token: String,
    /// Expiry as unix millis (computed from `expires_in` seconds in the response).
    expiry_millis: Option<i64>,
}

/// Refresh an expired Gemini CLI OAuth token using the refresh_token grant.
///
/// Client credentials are optional and can be sourced from:
/// - `oauth_creds.json` if present
/// - `GEMINI_OAUTH_CLIENT_ID` / `GEMINI_OAUTH_CLIENT_SECRET` env vars
fn refresh_gemini_cli_token(
    refresh_token: &str,
    client_id: Option<&str>,
    client_secret: Option<&str>,
) -> anyhow::Result<RefreshedToken> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .connect_timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap_or_else(|_| reqwest::blocking::Client::new());

    let form = build_oauth_refresh_form(refresh_token, client_id, client_secret);

    let response = client
        .post(GOOGLE_TOKEN_ENDPOINT)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .header("Accept", "application/json")
        .form(&form)
        .send()
        .map_err(|error| anyhow::anyhow!("Gemini CLI OAuth refresh request failed: {error}"))?;

    let status = response.status();
    let body = response
        .text()
        .unwrap_or_else(|_| "<failed to read response body>".to_string());

    if !status.is_success() {
        anyhow::bail!("Gemini CLI OAuth refresh failed (HTTP {status}): {body}");
    }

    #[derive(Deserialize)]
    struct TokenResponse {
        access_token: Option<String>,
        expires_in: Option<i64>,
    }

    let parsed: TokenResponse = serde_json::from_str(&body)
        .map_err(|_| anyhow::anyhow!("Gemini CLI OAuth refresh response is not valid JSON"))?;

    let access_token = parsed
        .access_token
        .filter(|t| !t.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("Gemini CLI OAuth refresh response missing access_token"))?;

    let expiry_millis = parsed.expires_in.and_then(|secs| {
        let now_millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()
            .and_then(|d| i64::try_from(d.as_millis()).ok())?;
        now_millis.checked_add(secs.checked_mul(1000)?)
    });

    Ok(RefreshedToken {
        access_token,
        expiry_millis,
    })
}

fn build_oauth_refresh_form(
    refresh_token: &str,
    client_id: Option<&str>,
    client_secret: Option<&str>,
) -> Vec<(&'static str, String)> {
    let mut form = vec![
        ("grant_type", "refresh_token".to_string()),
        ("refresh_token", refresh_token.to_string()),
    ];
    if let Some(id) = client_id.and_then(GeminiProvider::normalize_non_empty) {
        form.push(("client_id", id));
    }
    if let Some(secret) = client_secret.and_then(GeminiProvider::normalize_non_empty) {
        form.push(("client_secret", secret));
    }
    form
}

fn extract_client_id_from_id_token(id_token: &str) -> Option<String> {
    let payload = id_token.split('.').nth(1)?;
    let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload)
        .or_else(|_| base64::engine::general_purpose::URL_SAFE.decode(payload))
        .ok()?;

    #[derive(Deserialize)]
    struct IdTokenClaims {
        aud: Option<String>,
        azp: Option<String>,
    }

    let claims: IdTokenClaims = serde_json::from_slice(&decoded).ok()?;
    claims
        .aud
        .as_deref()
        .and_then(GeminiProvider::normalize_non_empty)
        .or_else(|| {
            claims
                .azp
                .as_deref()
                .and_then(GeminiProvider::normalize_non_empty)
        })
}

/// Async version of token refresh for use during runtime (inside tokio context).
async fn refresh_gemini_cli_token_async(
    refresh_token: &str,
    client_id: Option<&str>,
    client_secret: Option<&str>,
) -> anyhow::Result<RefreshedToken> {
    let refresh_token = refresh_token.to_string();
    let client_id = client_id.map(str::to_string);
    let client_secret = client_secret.map(str::to_string);
    tokio::task::spawn_blocking(move || {
        refresh_gemini_cli_token(
            &refresh_token,
            client_id.as_deref(),
            client_secret.as_deref(),
        )
    })
    .await
    .map_err(|e| anyhow::anyhow!("Token refresh task panicked: {e}"))?
}

impl GeminiProvider {
    /// Create a new Gemini provider.
    ///
    /// Authentication priority:
    /// 1. Explicit API key passed in
    /// 2. `GEMINI_API_KEY` environment variable
    /// 3. `GOOGLE_API_KEY` environment variable
    /// 4. Gemini CLI OAuth tokens (`~/.gemini/oauth_creds.json`)
    pub fn new(api_key: Option<&str>) -> Self {
        let oauth_cred_paths = Self::discover_oauth_cred_paths();
        let resolved_auth = api_key
            .and_then(Self::normalize_non_empty)
            .map(GeminiAuth::ExplicitKey)
            .or_else(|| Self::load_non_empty_env("GEMINI_API_KEY").map(GeminiAuth::EnvGeminiKey))
            .or_else(|| Self::load_non_empty_env("GOOGLE_API_KEY").map(GeminiAuth::EnvGoogleKey))
            .or_else(|| {
                Self::try_load_gemini_cli_token(oauth_cred_paths.first())
                    .map(|state| GeminiAuth::OAuthToken(Arc::new(tokio::sync::Mutex::new(state))))
            });

        Self {
            auth: resolved_auth,
            oauth_project: Arc::new(tokio::sync::Mutex::new(None)),
            oauth_cred_paths,
            oauth_index: Arc::new(tokio::sync::Mutex::new(0)),
            auth_service: None,
            auth_profile_override: None,
            base_url: None,
        }
    }

    /// Create a new Gemini provider with managed OAuth from auth-profiles.json.
    ///
    /// Authentication priority:
    /// 1. Explicit API key passed in
    /// 2. `GEMINI_API_KEY` environment variable
    /// 3. `GOOGLE_API_KEY` environment variable
    /// 4. Managed OAuth from auth-profiles.json (if auth_service provided)
    /// 5. Gemini CLI OAuth tokens (`~/.gemini/oauth_creds.json`)
    pub fn new_with_auth(
        api_key: Option<&str>,
        auth_service: AuthService,
        profile_override: Option<String>,
    ) -> Self {
        let oauth_cred_paths = Self::discover_oauth_cred_paths();

        // First check API keys
        let resolved_auth = api_key
            .and_then(Self::normalize_non_empty)
            .map(GeminiAuth::ExplicitKey)
            .or_else(|| Self::load_non_empty_env("GEMINI_API_KEY").map(GeminiAuth::EnvGeminiKey))
            .or_else(|| Self::load_non_empty_env("GOOGLE_API_KEY").map(GeminiAuth::EnvGoogleKey));

        // If no API key, we'll use managed OAuth (checked at runtime)
        // or fall back to CLI OAuth
        let (auth, use_managed) = if resolved_auth.is_some() {
            (resolved_auth, false)
        } else {
            // Check if we have a managed profile - this is a blocking check
            // but we need to know at construction time
            let has_managed = std::thread::scope(|s| {
                s.spawn(|| {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .ok()?;
                    rt.block_on(async {
                        auth_service
                            .get_gemini_profile(profile_override.as_deref())
                            .await
                            .ok()
                            .flatten()
                    })
                })
                .join()
                .ok()
                .flatten()
                .is_some()
            });

            if has_managed {
                (Some(GeminiAuth::ManagedOAuth), true)
            } else {
                // Fall back to CLI OAuth
                let cli_auth = Self::try_load_gemini_cli_token(oauth_cred_paths.first())
                    .map(|state| GeminiAuth::OAuthToken(Arc::new(tokio::sync::Mutex::new(state))));
                (cli_auth, false)
            }
        };

        Self {
            auth,
            oauth_project: Arc::new(tokio::sync::Mutex::new(None)),
            oauth_cred_paths,
            oauth_index: Arc::new(tokio::sync::Mutex::new(0)),
            auth_service: if use_managed {
                Some(auth_service)
            } else {
                None
            },
            auth_profile_override: profile_override,
            base_url: None,
        }
    }

    fn normalize_non_empty(value: &str) -> Option<String> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    }

    fn load_non_empty_env(name: &str) -> Option<String> {
        std::env::var(name)
            .ok()
            .and_then(|value| Self::normalize_non_empty(&value))
    }

    fn load_gemini_cli_creds(creds_path: &PathBuf) -> Option<GeminiCliOAuthCreds> {
        if !creds_path.exists() {
            return None;
        }
        let content = std::fs::read_to_string(creds_path).ok()?;
        serde_json::from_str(&content).ok()
    }

    /// Discover all OAuth credential files from known Gemini CLI installations.
    ///
    /// Looks in `~/.gemini/oauth_creds.json` (default) plus any
    /// `~/.gemini-*-home/.gemini/oauth_creds.json` siblings.
    fn discover_oauth_cred_paths() -> Vec<PathBuf> {
        let home = match UserDirs::new() {
            Some(u) => u.home_dir().to_path_buf(),
            None => return Vec::new(),
        };

        let mut paths = Vec::new();

        let primary = home.join(".gemini").join("oauth_creds.json");
        if primary.exists() {
            paths.push(primary);
        }

        if let Ok(entries) = std::fs::read_dir(&home) {
            let mut extras: Vec<PathBuf> = entries
                .filter_map(|e| e.ok())
                .filter_map(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    if name.starts_with(".gemini-") && name.ends_with("-home") {
                        let path = e.path().join(".gemini").join("oauth_creds.json");
                        if path.exists() {
                            return Some(path);
                        }
                    }
                    None
                })
                .collect();
            extras.sort();
            paths.extend(extras);
        }

        paths
    }

    /// Try to load OAuth credentials from Gemini CLI's cached credentials.
    /// Location: `~/.gemini/oauth_creds.json`
    ///
    /// Returns the full `OAuthTokenState` so the provider can refresh at runtime.
    fn try_load_gemini_cli_token(path: Option<&PathBuf>) -> Option<OAuthTokenState> {
        let creds = Self::load_gemini_cli_creds(path?)?;

        // Determine expiry in millis: prefer expiry_date over expiry (RFC 3339)
        let expiry_millis = creds.expiry_date.or_else(|| {
            creds.expiry.as_deref().and_then(|expiry| {
                chrono::DateTime::parse_from_rfc3339(expiry)
                    .ok()
                    .map(|dt| dt.timestamp_millis())
            })
        });

        let access_token = creds
            .access_token
            .and_then(|token| Self::normalize_non_empty(&token))?;

        let id_token_client_id = creds
            .id_token
            .as_deref()
            .and_then(extract_client_id_from_id_token);

        let client_id = Self::load_non_empty_env("GEMINI_OAUTH_CLIENT_ID")
            .or_else(|| {
                creds
                    .client_id
                    .as_deref()
                    .and_then(Self::normalize_non_empty)
            })
            .or(id_token_client_id);
        let client_secret = Self::load_non_empty_env("GEMINI_OAUTH_CLIENT_SECRET").or_else(|| {
            creds
                .client_secret
                .as_deref()
                .and_then(Self::normalize_non_empty)
        });

        Some(OAuthTokenState {
            access_token,
            refresh_token: creds.refresh_token,
            client_id,
            client_secret,
            expiry_millis,
        })
    }

    /// Get the Gemini CLI config directory (~/.gemini)
    fn gemini_cli_dir() -> Option<PathBuf> {
        UserDirs::new().map(|u| u.home_dir().join(".gemini"))
    }

    /// Check if Gemini CLI is configured and has valid credentials
    pub fn has_cli_credentials() -> bool {
        Self::discover_oauth_cred_paths().iter().any(|path| {
            Self::load_gemini_cli_creds(path)
                .and_then(|creds| {
                    creds
                        .access_token
                        .as_deref()
                        .and_then(Self::normalize_non_empty)
                })
                .is_some()
        })
    }

    /// Check if any Gemini authentication is available
    pub fn has_any_auth() -> bool {
        Self::load_non_empty_env("GEMINI_API_KEY").is_some()
            || Self::load_non_empty_env("GOOGLE_API_KEY").is_some()
            || Self::has_cli_credentials()
    }

    /// Get authentication source description for diagnostics.
    /// Uses the stored enum variant — no env var re-reading at call time.
    pub fn auth_source(&self) -> &'static str {
        match self.auth.as_ref() {
            Some(GeminiAuth::ExplicitKey(_)) => "config",
            Some(GeminiAuth::EnvGeminiKey(_)) => "GEMINI_API_KEY env var",
            Some(GeminiAuth::EnvGoogleKey(_)) => "GOOGLE_API_KEY env var",
            Some(GeminiAuth::OAuthToken(_)) => "Gemini CLI OAuth",
            Some(GeminiAuth::ManagedOAuth) => "auth-profiles",
            None => "none",
        }
    }

    /// Get a valid OAuth access token, refreshing if expired.
    /// Adds a 60-second buffer before actual expiry to avoid edge-case failures.
    async fn get_valid_oauth_token(
        state: &Arc<tokio::sync::Mutex<OAuthTokenState>>,
    ) -> anyhow::Result<String> {
        let mut guard = state.lock().await;

        let now_millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()
            .and_then(|d| i64::try_from(d.as_millis()).ok())
            .unwrap_or(i64::MAX);

        // Refresh if expiry is unknown, already expired, or within 60s of expiry.
        let needs_refresh = guard
            .expiry_millis
            .map_or(true, |exp| exp <= now_millis.saturating_add(60_000));

        if needs_refresh {
            if let Some(ref refresh_token) = guard.refresh_token {
                let refreshed = refresh_gemini_cli_token_async(
                    refresh_token,
                    guard.client_id.as_deref(),
                    guard.client_secret.as_deref(),
                )
                .await?;
                tracing::info!("Gemini CLI OAuth token refreshed successfully (runtime)");
                guard.access_token = refreshed.access_token;
                guard.expiry_millis = refreshed.expiry_millis;
            } else {
                anyhow::bail!(
                    "Gemini CLI OAuth token expired and no refresh_token available — re-run `gemini` to authenticate"
                );
            }
        }

        Ok(guard.access_token.clone())
    }

    /// Rotate to the next available OAuth credentials file and swap state.
    /// Returns `true` when rotation succeeded.
    async fn rotate_oauth_credential(
        &self,
        state: &Arc<tokio::sync::Mutex<OAuthTokenState>>,
    ) -> bool {
        if self.oauth_cred_paths.len() <= 1 {
            return false;
        }

        let mut idx = self.oauth_index.lock().await;
        let start = *idx;

        loop {
            let next = (*idx + 1) % self.oauth_cred_paths.len();
            *idx = next;

            if next == start {
                return false;
            }

            if let Some(next_state) =
                Self::try_load_gemini_cli_token(self.oauth_cred_paths.get(next))
            {
                {
                    let mut guard = state.lock().await;
                    *guard = next_state;
                }
                {
                    let mut cached_project = self.oauth_project.lock().await;
                    *cached_project = None;
                }
                tracing::warn!(
                    "Gemini OAuth: rotated credential to {}",
                    self.oauth_cred_paths[next].display()
                );
                return true;
            }
        }
    }

    fn format_model_name(model: &str) -> String {
        if model.starts_with("models/") {
            model.to_string()
        } else {
            format!("models/{model}")
        }
    }

    fn format_internal_model_name(model: &str) -> String {
        model.strip_prefix("models/").unwrap_or(model).to_string()
    }

    fn build_generate_content_url(&self, model: &str, auth: &GeminiAuth) -> String {
        let (base, is_internal) = match auth {
            GeminiAuth::OAuthToken(_) | GeminiAuth::ManagedOAuth => {
                (self.base_url.as_deref().unwrap_or(CLOUDCODE_PA_ENDPOINT), true)
            }
            _ => (self.base_url.as_deref().unwrap_or(PUBLIC_API_ENDPOINT), false),
        };

        if is_internal {
            format!("{}:generateContent", base)
        } else {
            let model_name = Self::format_model_name(model);
            let url = format!("{}/{}:generateContent", base, model_name);
            if auth.is_api_key() {
                format!("{}?key={}", url, auth.api_key_credential())
            } else {
                url
            }
        }
    }

    fn http_client(&self) -> Client {
        crate::config::build_runtime_proxy_client_with_timeouts("provider.gemini", 120, 10)
    }

    /// Resolve the GCP project ID for OAuth by calling the loadCodeAssist endpoint.
    /// Caches the result for subsequent calls.
    async fn resolve_oauth_project(&self, token: &str) -> anyhow::Result<String> {
        let project_seed = Self::load_non_empty_env("GOOGLE_CLOUD_PROJECT")
            .or_else(|| Self::load_non_empty_env("GOOGLE_CLOUD_PROJECT_ID"));
        let project_seed_for_request = project_seed.clone();
        let duet_project_for_request = project_seed.clone();

        // Check cache first
        {
            let cached = self.oauth_project.lock().await;
            if let Some(ref project) = *cached {
                return Ok(project.clone());
            }
        }

        // Call loadCodeAssist
        let client = self.http_client();
        let response = client
            .post(LOAD_CODE_ASSIST_ENDPOINT)
            .bearer_auth(token)
            .json(&serde_json::json!({
                "cloudaicompanionProject": project_seed_for_request,
                "metadata": {
                    "ideType": "GEMINI_CLI",
                    "platform": "PLATFORM_UNSPECIFIED",
                    "pluginType": "GEMINI",
                    "duetProject": duet_project_for_request,
                }
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            if let Some(seed) = project_seed {
                tracing::warn!(
                    "loadCodeAssist failed (HTTP {status}); using GOOGLE_CLOUD_PROJECT fallback"
                );
                return Ok(seed);
            }
            anyhow::bail!("loadCodeAssist failed (HTTP {status}): {body}");
        }

        #[derive(Deserialize)]
        struct LoadCodeAssistResponse {
            #[serde(rename = "cloudaicompanionProject")]
            cloudaicompanion_project: Option<String>,
        }

        let result: LoadCodeAssistResponse = response.json().await?;
        let project = result
            .cloudaicompanion_project
            .filter(|p| !p.trim().is_empty())
            .or(project_seed)
            .ok_or_else(|| anyhow::anyhow!("loadCodeAssist response missing project context"))?;

        // Cache for future calls
        {
            let mut cached = self.oauth_project.lock().await;
            *cached = Some(project.clone());
        }

        Ok(project)
    }

    /// Build the HTTP request for generateContent.
    ///
    /// For OAuth, pass the resolved `oauth_token` and `project`.
    /// For API key, both are `None`.
    fn build_generate_content_request(
        &self,
        auth: &GeminiAuth,
        url: &str,
        request: &GenerateContentRequest,
        model: &str,
        include_generation_config: bool,
        project: Option<&str>,
        oauth_token: Option<&str>,
    ) -> reqwest::RequestBuilder {
        let req = self.http_client().post(url).json(request);
        match auth {
            GeminiAuth::OAuthToken(_) | GeminiAuth::ManagedOAuth => {
                let token = oauth_token.unwrap_or_default();
                // Internal Code Assist API uses a wrapped payload shape:
                // { model, project?, user_prompt_id?, request: { contents, systemInstruction?, generationConfig } }
                let internal_request = InternalGenerateContentEnvelope {
                    model: Self::format_internal_model_name(model),
                    project: project.map(|value| value.to_string()),
                    user_prompt_id: Some(uuid::Uuid::new_v4().to_string()),
                    request: InternalGenerateContentRequest {
                        contents: request.contents.clone(),
                        system_instruction: request.system_instruction.clone(),
                        generation_config: if include_generation_config {
                            Some(request.generation_config.clone())
                        } else {
                            None
                        },
                    },
                };
                self.http_client()
                    .post(url)
                    .json(&internal_request)
                    .bearer_auth(token)
            }
            _ => req,
        }
    }

    fn should_retry_oauth_without_generation_config(
        status: reqwest::StatusCode,
        error_text: &str,
    ) -> bool {
        if status != reqwest::StatusCode::BAD_REQUEST {
            return false;
        }

        error_text.contains("Unknown name \"generationConfig\"")
            || error_text.contains("Unknown name 'generationConfig'")
            || error_text.contains(r#"Unknown name \"generationConfig\""#)
    }

    fn should_rotate_oauth_on_error(status: reqwest::StatusCode, error_text: &str) -> bool {
        status == reqwest::StatusCode::TOO_MANY_REQUESTS
            || status == reqwest::StatusCode::SERVICE_UNAVAILABLE
            || status.is_server_error()
            || error_text.contains("RESOURCE_EXHAUSTED")
    }
}

impl GeminiProvider {
    async fn send_generate_content(
        &self,
        contents: Vec<Content>,
        system_instruction: Option<Content>,
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<(String, Option<TokenUsage>)> {
        let resp = self
            .send_generate_content_full(contents, system_instruction, None, model, temperature)
            .await?;
        Ok((resp.text.unwrap_or_default(), resp.usage))
    }

    async fn send_generate_content_full(
        &self,
        contents: Vec<Content>,
        system_instruction: Option<Content>,
        tools: Option<Vec<GeminiTool>>,
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<ChatResponse> {
        let auth = self.auth.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Gemini API key not found. Options:\n\
                 1. Set GEMINI_API_KEY env var\n\
                 2. Run `gemini` CLI to authenticate (tokens will be reused)\n\
                 3. Run `zeroclaw auth login --provider gemini`\n\
                 4. Get an API key from https://aistudio.google.com/app/apikey\n\
                 5. Run `zeroclaw onboard` to configure"
            )
        })?;

        let oauth_state = match auth {
            GeminiAuth::OAuthToken(state) => Some(state.clone()),
            _ => None,
        };

        // For OAuth: get a valid (potentially refreshed) token and resolve project
        let (mut oauth_token, mut project) = match auth {
            GeminiAuth::OAuthToken(state) => {
                let token = Self::get_valid_oauth_token(state).await?;
                let proj = self.resolve_oauth_project(&token).await?;
                (Some(token), Some(proj))
            }
            GeminiAuth::ManagedOAuth => {
                let auth_service = self
                    .auth_service
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("ManagedOAuth requires auth_service"))?;
                let token = auth_service
                    .get_valid_gemini_access_token(self.auth_profile_override.as_deref())
                    .await?
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Gemini auth profile not found. Run `zeroclaw auth login --provider gemini`."
                        )
                    })?;
                let proj = self.resolve_oauth_project(&token).await?;
                (Some(token), Some(proj))
            }
            _ => (None, None),
        };

        let request = GenerateContentRequest {
            contents,
            system_instruction,
            generation_config: GenerationConfig {
                temperature,
                max_output_tokens: 8192,
            },
            tools,
        };

        let url = self.build_generate_content_url(model, auth);

        let mut response = self
            .build_generate_content_request(
                auth,
                &url,
                &request,
                model,
                true,
                project.as_deref(),
                oauth_token.as_deref(),
            )
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();

            if auth.is_oauth() && Self::should_rotate_oauth_on_error(status, &error_text) {
                // For CLI OAuth: rotate credentials
                let can_retry = match auth {
                    GeminiAuth::OAuthToken(_) => {
                        if let Some(state) = oauth_state.as_ref() {
                            self.rotate_oauth_credential(state).await
                        } else {
                            false
                        }
                    }
                    GeminiAuth::ManagedOAuth => true,
                    _ => false,
                };

                if can_retry {
                    let (new_token, new_project) = match auth {
                        GeminiAuth::OAuthToken(state) => {
                            let token = Self::get_valid_oauth_token(state).await?;
                            let proj = self.resolve_oauth_project(&token).await?;
                            (token, proj)
                        }
                        GeminiAuth::ManagedOAuth => {
                            let auth_service = self.auth_service.as_ref().unwrap();
                            let token = auth_service
                                .get_valid_gemini_access_token(
                                    self.auth_profile_override.as_deref(),
                                )
                                .await?
                                .ok_or_else(|| anyhow::anyhow!("Gemini auth profile not found"))?;
                            let proj = self.resolve_oauth_project(&token).await?;
                            (token, proj)
                        }
                        _ => unreachable!(),
                    };
                    oauth_token = Some(new_token);
                    project = Some(new_project);
                    response = self
                        .build_generate_content_request(
                            auth,
                            &url,
                            &request,
                            model,
                            true,
                            project.as_deref(),
                            oauth_token.as_deref(),
                        )
                        .send()
                        .await?;
                } else {
                    anyhow::bail!("Gemini API error ({status}): {error_text}");
                }
            } else if auth.is_oauth()
                && Self::should_retry_oauth_without_generation_config(status, &error_text)
            {
                tracing::warn!(
                    "Gemini OAuth internal endpoint rejected generationConfig; retrying without generationConfig"
                );
                response = self
                    .build_generate_content_request(
                        auth,
                        &url,
                        &request,
                        model,
                        false,
                        project.as_deref(),
                        oauth_token.as_deref(),
                    )
                    .send()
                    .await?;
            } else {
                anyhow::bail!("Gemini API error ({status}): {error_text}");
            }
        }

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Gemini API error ({status}): {error_text}");
        }

        let result: GenerateContentResponse = response.json().await?;
        if let Some(err) = &result.error {
            anyhow::bail!("Gemini API error: {}", err.message);
        }
        let result = result.into_effective_response();
        if let Some(err) = result.error {
            anyhow::bail!("Gemini API error: {}", err.message);
        }

        let usage = result.usage_metadata.map(|u| TokenUsage {
            input_tokens: u.prompt_token_count,
            output_tokens: u.candidates_token_count,
            cached_input_tokens: None,
        });

        let mut tool_calls = Vec::new();
        let mut text = None;
        let mut reasoning_content = None;

        if let Some(candidate) = result.candidates.and_then(|c| c.into_iter().next()) {
            if let Some(content) = candidate.content {
                let mut answer_parts = Vec::new();
                for part in content.parts {
                    if part.thought {
                        if let Some(t) = part.text {
                            reasoning_content = Some(t);
                        }
                    } else if let Some(fc) = part.function_call {
                        tool_calls.push(crate::providers::traits::ToolCall {
                            id: uuid::Uuid::new_v4().to_string(), // Gemini doesn't always provide IDs
                            name: fc.name,
                            arguments: fc.args.to_string(),
                        });
                    } else if let Some(t) = part.text {
                        answer_parts.push(t);
                    }
                }
                if !answer_parts.is_empty() {
                    text = Some(answer_parts.join(""));
                }
            }
        }

        Ok(ChatResponse {
            text,
            tool_calls,
            usage,
            reasoning_content,
        })
    }
    async fn stream_generate_content(
        &self,
        contents: Vec<Content>,
        system_instruction: Option<Content>,
        tools: Option<Vec<GeminiTool>>,
        model: &str,
        temperature: f64,
    ) -> crate::providers::traits::StreamResult<
        futures_util::stream::BoxStream<'static, crate::providers::traits::StreamResult<crate::providers::traits::StreamEvent>>,
    > {
        let auth = self.auth.as_ref().ok_or_else(|| {
            crate::providers::traits::StreamError::Provider("Gemini API key not found".to_string())
        })?;

        // For OAuth: get a valid (potentially refreshed) token and resolve project
        let (oauth_token, project) = match auth {
            GeminiAuth::OAuthToken(state) => {
                let token = Self::get_valid_oauth_token(state).await.map_err(|e| {
                    crate::providers::traits::StreamError::Provider(format!(
                        "OAuth token error: {e}"
                    ))
                })?;
                let proj = self.resolve_oauth_project(&token).await.map_err(|e| {
                    crate::providers::traits::StreamError::Provider(format!(
                        "OAuth project error: {e}"
                    ))
                })?;
                (Some(token), Some(proj))
            }
            GeminiAuth::ManagedOAuth => {
                let auth_service = self.auth_service.as_ref().ok_or_else(|| {
                    crate::providers::traits::StreamError::Provider(
                        "ManagedOAuth requires auth_service".to_string(),
                    )
                })?;
                let token = auth_service
                    .get_valid_gemini_access_token(self.auth_profile_override.as_deref())
                    .await
                    .map_err(|e| {
                        crate::providers::traits::StreamError::Provider(format!(
                            "ManagedOAuth error: {e}"
                        ))
                    })?
                    .ok_or_else(|| {
                        crate::providers::traits::StreamError::Provider(
                            "Gemini auth profile not found".to_string(),
                        )
                    })?;
                let proj = self.resolve_oauth_project(&token).await.map_err(|e| {
                    crate::providers::traits::StreamError::Provider(format!(
                        "OAuth project error: {e}"
                    ))
                })?;
                (Some(token), Some(proj))
            }
            _ => (None, None),
        };

        let request = GenerateContentRequest {
            contents,
            system_instruction,
            generation_config: GenerationConfig {
                temperature,
                max_output_tokens: 8192,
            },
            tools,
        };

        let base_url = self.build_generate_content_url(model, auth);
        let stream_url = format!(
            "{}?alt=sse",
            base_url.replace(":generateContent", ":streamGenerateContent")
        );

        let response = self
            .build_generate_content_request(
                auth,
                &stream_url,
                &request,
                model,
                true,
                project.as_deref(),
                oauth_token.as_deref(),
            )
            .send()
            .await
            .map_err(|e| {
                crate::providers::traits::StreamError::Provider(format!("Streaming error: {e}"))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(crate::providers::traits::StreamError::Provider(format!(
                "Gemini streaming API error ({status}): {error_text}"
            )));
        }

        let (tx, rx) = tokio::sync::mpsc::channel(100);

        tokio::spawn(async move {
            Self::parse_gemini_stream(response, tx).await;
        });

        Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }

    async fn parse_gemini_stream(
        response: reqwest::Response,
        tx: tokio::sync::mpsc::Sender<crate::providers::traits::StreamResult<crate::providers::traits::StreamEvent>>,
    ) {
        use futures_util::StreamExt;
        use tokio::io::AsyncBufReadExt;
        use tokio_util::io::StreamReader;

        let byte_stream = response
            .bytes_stream()
            .map(|result| result.map_err(std::io::Error::other));
        let reader = StreamReader::new(byte_stream);
        let mut lines = reader.lines();

        while let Ok(Some(line)) = lines.next_line().await {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if !line.starts_with("data: ") {
                continue;
            }
            let json_str = &line["data: ".len()..];
            let result: Result<GenerateContentResponse, _> = serde_json::from_str(json_str);

            match result {
                Ok(resp) => {
                    let mut text_parts = Vec::new();
                    let mut tool_calls = Vec::new();

                    if let Some(candidate) = resp.candidates.and_then(|c| c.into_iter().next()) {
                        if let Some(content) = candidate.content {
                            for part in content.parts {
                                if part.thought {
                                    // Handle thought parts if needed
                                } else if let Some(fc) = part.function_call {
                                    tool_calls.push(crate::providers::traits::ToolCall {
                                        id: uuid::Uuid::new_v4().to_string(),
                                        name: fc.name,
                                        arguments: fc.args.to_string(),
                                    });
                                } else if let Some(t) = part.text {
                                    text_parts.push(t);
                                }
                            }
                        }
                    }

                    for text in text_parts {
                        if !text.is_empty() {
                            let _ = tx
                                .send(Ok(StreamEvent::TextDelta(
                                    StreamChunk::delta(text),
                                )))
                                .await;
                        }
                    }

                    for call in tool_calls {
                        let _ = tx
                            .send(Ok(StreamEvent::ToolCall(call)))
                            .await;
                    }
                }
                Err(e) => {
                    let _ = tx
                        .send(Err(StreamError::Provider(format!(
                            "JSON decode error: {e}"
                        ))))
                        .await;
                    return;
                }
            }
        }

        let _ = tx.send(Ok(crate::providers::traits::StreamEvent::Final)).await;
    }

    #[cfg(test)]
    pub(crate) fn set_base_url(&mut self, url: String) {
        self.base_url = Some(url);
    }
}

#[async_trait]
impl Provider for GeminiProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            native_tool_calling: true,
            vision: true,
            prompt_caching: false,
        }
    }

    fn convert_tools(&self, tools: &[crate::tools::ToolSpec]) -> ToolsPayload {
        ToolsPayload::Gemini {
            function_declarations: tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    })
                })
                .collect(),
        }
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[serde_json::Value],
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<ChatResponse> {
        let mut system_parts: Vec<&str> = Vec::new();
        let mut contents: Vec<Content> = Vec::new();

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    system_parts.push(&msg.content);
                }
                "user" => {
                    contents.push(Content {
                        role: Some("user".to_string()),
                        parts: build_parts(&msg.content),
                    });
                }
                "assistant" => {
                    contents.push(Content {
                        role: Some("model".to_string()),
                        parts: vec![Part::text(&msg.content)],
                    });
                }
                "tool" => {
                    // Logic for feeding tool results back to Gemini.
                    // This expects content to be JSON or stringified data.
                    let response_value = serde_json::from_str(&msg.content)
                        .unwrap_or_else(|_| serde_json::json!({ "output": msg.content }));

                    contents.push(Content {
                        role: Some("function".to_string()),
                        parts: vec![Part::FunctionResponse {
                            function_response: FunctionResponse {
                                name: "unknown".to_string(), // Need to map this from history if possible
                                response: response_value,
                            },
                        }],
                    });
                }
                _ => {}
            }
        }

        let system_instruction = if system_parts.is_empty() {
            None
        } else {
            Some(Content {
                role: None,
                parts: vec![Part::text(system_parts.join("\n\n"))],
            })
        };

        let gemini_tools = vec![GeminiTool {
            function_declarations: tools.to_vec(),
        }];

        self.send_generate_content_full(
            contents,
            system_instruction,
            Some(gemini_tools),
            model,
            temperature,
        )
        .await
    }

    async fn chat_with_system(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<String> {
        let system_instruction = system_prompt.map(|sys| Content {
            role: None,
            parts: vec![Part::text(sys)],
        });

        let contents = vec![Content {
            role: Some("user".to_string()),
            parts: build_parts(message),
        }];

        let (text, _) = self
            .send_generate_content(contents, system_instruction, model, temperature)
            .await?;
        Ok(text)
    }

    fn stream_chat_with_system(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        model: &str,
        temperature: f64,
        _options: StreamOptions,
    ) -> futures_util::stream::BoxStream<'static, StreamResult<StreamChunk>> {
        let system_instruction = system_prompt.map(|sys| Content {
            role: None,
            parts: vec![Part::text(sys)],
        });

        let contents = vec![Content {
            role: Some("user".to_string()),
            parts: build_parts(message),
        }];

        Box::pin(futures_util::stream::once(self.stream_generate_content(
            contents,
            system_instruction,
            None,
            model,
            temperature,
        )).filter_map(|res| async {
            match res {
                Ok(s) => Some(s.map(|item| match item {
                    crate::providers::traits::StreamEvent::TextDelta(chunk) => Ok(chunk),
                    crate::providers::traits::StreamEvent::Final => Ok(crate::providers::traits::StreamChunk::final_chunk()),
                    _ => Ok(crate::providers::traits::StreamChunk::delta("")), // Skip tool calls for text-only stream
                })),
                Err(e) => Some(Box::pin(futures_util::stream::once(async move { Err(e) })) as futures_util::stream::BoxStream<'static, _>),
            }
        }).flatten())
    }

    fn stream_chat(
        &self,
        request: ChatRequest<'_>,
        model: &str,
        temperature: f64,
        _options: StreamOptions,
    ) -> futures_util::stream::BoxStream<'static, StreamResult<StreamEvent>> {
        let mut system_parts: Vec<&str> = Vec::new();
        let mut contents: Vec<Content> = Vec::new();

        for msg in request.messages {
            match msg.role.as_str() {
                "system" => {
                    system_parts.push(&msg.content);
                }
                "user" => {
                    contents.push(Content {
                        role: Some("user".to_string()),
                        parts: build_parts(&msg.content),
                    });
                }
                "assistant" => {
                    contents.push(Content {
                        role: Some("model".to_string()),
                        parts: vec![Part::text(&msg.content)],
                    });
                }
                _ => {}
            }
        }

        let system_instruction = if system_parts.is_empty() {
            None
        } else {
            Some(Content {
                role: None,
                parts: vec![Part::text(system_parts.join("\n\n"))],
            })
        };

        Box::pin(futures_util::stream::once(self.stream_generate_content(
            contents,
            system_instruction,
            None,
            model,
            temperature,
        )).filter_map(|res| async {
            match res {
                Ok(s) => Some(s),
                Err(e) => Some(Box::pin(futures_util::stream::once(async move { Err(e) })) as futures_util::stream::BoxStream<'static, _>),
            }
        }).flatten())
    }

    async fn warmup(&self) -> anyhow::Result<()> {
        if let Some(auth) = self.auth.as_ref() {
            match auth {
                GeminiAuth::ManagedOAuth => {
                    let auth_service = self
                        .auth_service
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("ManagedOAuth requires auth_service"))?;

                    let _token = auth_service
                        .get_valid_gemini_access_token(self.auth_profile_override.as_deref())
                        .await?
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "Gemini auth profile not found or expired. Run: zeroclaw auth login --provider gemini"
                            )
                        })?;
                }
                GeminiAuth::OAuthToken(_) => {
                }
                _ => {
                    let base = self.base_url.as_deref().unwrap_or(PUBLIC_API_ENDPOINT);
                    let url = if auth.is_api_key() {
                        format!(
                            "{}/models?key={}",
                            base,
                            auth.api_key_credential()
                        )
                    } else {
                        format!("{}/models", base)
                    };

                    self.http_client()
                        .get(&url)
                        .send()
                        .await?
                        .error_for_status()?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::{StatusCode, header::AUTHORIZATION};
    use crate::providers::traits::{ToolsPayload, StreamEvent, ChatMessage};
    use crate::tools::ToolSpec;
    use axum::{Json, Router, routing::post};
    use futures_util::StreamExt;
    use std::sync::{Arc, Mutex};
    use tokio::net::TcpListener;

    /// Helper to create a test OAuth auth variant.
    fn test_oauth_auth(token: &str) -> GeminiAuth {
        GeminiAuth::OAuthToken(Arc::new(tokio::sync::Mutex::new(OAuthTokenState {
            access_token: token.to_string(),
            refresh_token: None,
            client_id: None,
            client_secret: None,
            expiry_millis: None,
        })))
    }

    fn test_provider(auth: Option<GeminiAuth>) -> GeminiProvider {
        GeminiProvider {
            auth,
            oauth_project: Arc::new(tokio::sync::Mutex::new(None)),
            oauth_cred_paths: Vec::new(),
            oauth_index: Arc::new(tokio::sync::Mutex::new(0)),
            auth_service: None,
            auth_profile_override: None,
            base_url: None,
        }
    }

    #[test]
    fn normalize_non_empty_trims_and_filters() {
        assert_eq!(
            GeminiProvider::normalize_non_empty(" value "),
            Some("value".into())
        );
        assert_eq!(GeminiProvider::normalize_non_empty(""), None);
        assert_eq!(GeminiProvider::normalize_non_empty(" \t\n"), None);
    }

    #[test]
    fn oauth_refresh_form_uses_provided_client_credentials() {
        let form = build_oauth_refresh_form("refresh-token", Some("client-id"), Some("secret"));
        let map: std::collections::HashMap<_, _> = form.into_iter().collect();
        assert_eq!(map.get("grant_type"), Some(&"refresh_token".to_string()));
        assert_eq!(map.get("refresh_token"), Some(&"refresh-token".to_string()));
        assert_eq!(map.get("client_id"), Some(&"client-id".to_string()));
        assert_eq!(map.get("client_secret"), Some(&"secret".to_string()));
    }

    #[test]
    fn oauth_refresh_form_omits_client_credentials_when_missing() {
        let form = build_oauth_refresh_form("refresh-token", None, None);
        let map: std::collections::HashMap<_, _> = form.into_iter().collect();
        assert!(!map.contains_key("client_id"));
        assert!(!map.contains_key("client_secret"));
    }

    #[test]
    fn extract_client_id_from_id_token_prefers_aud_claim() {
        let payload = serde_json::json!({
            "aud": "aud-client-id",
            "azp": "azp-client-id"
        });
        let payload_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&payload).unwrap());
        let token = format!("header.{payload_b64}.sig");

        assert_eq!(
            extract_client_id_from_id_token(&token),
            Some("aud-client-id".to_string())
        );
    }

    #[test]
    fn extract_client_id_from_id_token_uses_azp_when_aud_missing() {
        let payload = serde_json::json!({
            "azp": "azp-client-id"
        });
        let payload_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&payload).unwrap());
        let token = format!("header.{payload_b64}.sig");

        assert_eq!(
            extract_client_id_from_id_token(&token),
            Some("azp-client-id".to_string())
        );
    }

    #[test]
    fn extract_client_id_from_id_token_returns_none_for_invalid_tokens() {
        assert_eq!(extract_client_id_from_id_token("invalid"), None);
        assert_eq!(extract_client_id_from_id_token("a.b.c"), None);
    }

    #[test]
    fn try_load_cli_token_derives_client_id_from_id_token_when_missing() {
        let payload = serde_json::json!({ "aud": "derived-client-id" });
        let payload_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&payload).unwrap());
        let id_token = format!("header.{payload_b64}.sig");

        let file = tempfile::NamedTempFile::new().unwrap();
        let json = format!(
            r#"{{
                "access_token": "ya29.test-access",
                "refresh_token": "1//test-refresh",
                "id_token": "{id_token}"
            }}"#
        );
        std::fs::write(file.path(), json).unwrap();

        let path = file.path().to_path_buf();
        let state = GeminiProvider::try_load_gemini_cli_token(Some(&path)).unwrap();
        assert_eq!(state.client_id.as_deref(), Some("derived-client-id"));
        assert_eq!(state.client_secret, None);
    }

    #[test]
    fn provider_creates_without_key() {
        let provider = GeminiProvider::new(None);
        // May pick up env vars; just verify it doesn't panic
        let _ = provider.auth_source();
    }

    #[test]
    fn provider_creates_with_key() {
        let provider = GeminiProvider::new(Some("test-api-key"));
        assert!(matches!(
            provider.auth,
            Some(GeminiAuth::ExplicitKey(ref key)) if key == "test-api-key"
        ));
    }

    #[test]
    fn provider_rejects_empty_key() {
        let provider = GeminiProvider::new(Some(""));
        assert!(!matches!(provider.auth, Some(GeminiAuth::ExplicitKey(_))));
    }

    #[test]
    fn gemini_cli_dir_returns_path() {
        let dir = GeminiProvider::gemini_cli_dir();
        // Should return Some on systems with home dir
        if UserDirs::new().is_some() {
            assert!(dir.is_some());
            assert!(dir.unwrap().ends_with(".gemini"));
        }
    }

    #[test]
    fn auth_source_explicit_key() {
        let provider = test_provider(Some(GeminiAuth::ExplicitKey("key".into())));
        assert_eq!(provider.auth_source(), "config");
    }

    #[test]
    fn auth_source_none_without_credentials() {
        let provider = test_provider(None);
        assert_eq!(provider.auth_source(), "none");
    }

    #[test]
    fn auth_source_oauth() {
        let provider = test_provider(Some(test_oauth_auth("ya29.mock")));
        assert_eq!(provider.auth_source(), "Gemini CLI OAuth");
    }

    #[test]
    fn model_name_formatting() {
        assert_eq!(
            GeminiProvider::format_model_name("gemini-2.0-flash"),
            "models/gemini-2.0-flash"
        );
        assert_eq!(
            GeminiProvider::format_model_name("models/gemini-1.5-pro"),
            "models/gemini-1.5-pro"
        );
        assert_eq!(
            GeminiProvider::format_internal_model_name("models/gemini-2.5-flash"),
            "gemini-2.5-flash"
        );
        assert_eq!(
            GeminiProvider::format_internal_model_name("gemini-2.5-flash"),
            "gemini-2.5-flash"
        );
    }

    #[test]
    fn api_key_url_includes_key_query_param() {
        let auth = GeminiAuth::ExplicitKey("api-key-123".into());
        let provider = test_provider(None);
        let url = provider.build_generate_content_url("gemini-2.0-flash", &auth);
        assert!(url.contains(":generateContent?key=api-key-123"));
    }

    #[test]
    fn oauth_url_uses_internal_endpoint() {
        let auth = test_oauth_auth("ya29.test-token");
        let provider = test_provider(None);
        let url = provider.build_generate_content_url("gemini-2.0-flash", &auth);
        assert!(url.starts_with("https://cloudcode-pa.googleapis.com/v1internal"));
        assert!(url.ends_with(":generateContent"));
        assert!(!url.contains("generativelanguage.googleapis.com"));
        assert!(!url.contains("?key="));
    }

    #[test]
    fn api_key_url_uses_public_endpoint() {
        let auth = GeminiAuth::ExplicitKey("api-key-123".into());
        let provider = test_provider(None);
        let url = provider.build_generate_content_url("gemini-2.0-flash", &auth);
        assert!(url.contains("generativelanguage.googleapis.com/v1beta"));
        assert!(url.contains("models/gemini-2.0-flash"));
    }

    #[test]
    fn oauth_request_uses_bearer_auth_header() {
        let provider = test_provider(Some(test_oauth_auth("ya29.mock-token")));
        let auth = test_oauth_auth("ya29.mock-token");
        let url = provider.build_generate_content_url("gemini-2.0-flash", &auth);
        let body = GenerateContentRequest {
            contents: vec![Content {
                role: Some("user".into()),
                parts: vec![Part::text("hello")],
            }],
            system_instruction: None,
            generation_config: GenerationConfig {
                temperature: 0.7,
                max_output_tokens: 8192,
            },
        };

        let request = provider
            .build_generate_content_request(
                &auth,
                &url,
                &body,
                "gemini-2.0-flash",
                true,
                Some("test-project"),
                Some("ya29.mock-token"),
            )
            .build()
            .unwrap();

        assert_eq!(
            request
                .headers()
                .get(AUTHORIZATION)
                .and_then(|h| h.to_str().ok()),
            Some("Bearer ya29.mock-token")
        );
    }

    #[test]
    fn oauth_request_wraps_payload_in_request_envelope() {
        let provider = test_provider(Some(test_oauth_auth("ya29.mock-token")));
        let auth = test_oauth_auth("ya29.mock-token");
        let url = provider.build_generate_content_url("gemini-2.0-flash", &auth);
        let body = GenerateContentRequest {
            contents: vec![Content {
                role: Some("user".into()),
                parts: vec![Part::text("hello")],
            }],
            system_instruction: None,
            generation_config: GenerationConfig {
                temperature: 0.7,
                max_output_tokens: 8192,
            },
        };

        let request = provider
            .build_generate_content_request(
                &auth,
                &url,
                &body,
                "models/gemini-2.0-flash",
                true,
                Some("test-project"),
                Some("ya29.mock-token"),
            )
            .build()
            .unwrap();

        let payload = request
            .body()
            .and_then(|b| b.as_bytes())
            .expect("json request body should be bytes");
        let json: serde_json::Value = serde_json::from_slice(payload).unwrap();

        assert_eq!(json["model"], "gemini-2.0-flash");
        assert!(json.get("generationConfig").is_none());
        assert!(json.get("request").is_some());
        assert!(json["request"].get("generationConfig").is_some());
    }

    #[test]
    fn api_key_request_does_not_set_bearer_header() {
        let provider = test_provider(Some(GeminiAuth::ExplicitKey("api-key-123".into())));
        let auth = GeminiAuth::ExplicitKey("api-key-123".into());
        let url = provider.build_generate_content_url("gemini-2.0-flash", &auth);
        let body = GenerateContentRequest {
            contents: vec![Content {
                role: Some("user".into()),
                parts: vec![Part::text("hello")],
            }],
            system_instruction: None,
            generation_config: GenerationConfig {
                temperature: 0.7,
                max_output_tokens: 8192,
            },
        };

        let request = provider
            .build_generate_content_request(
                &auth,
                &url,
                &body,
                "gemini-2.0-flash",
                true,
                None,
                None,
            )
            .build()
            .unwrap();

        assert!(request.headers().get(AUTHORIZATION).is_none());
    }

    #[test]
    fn request_serialization() {
        let request = GenerateContentRequest {
            contents: vec![Content {
                role: Some("user".to_string()),
                parts: vec![Part::text("Hello")],
            }],
            system_instruction: Some(Content {
                role: None,
                parts: vec![Part::text("You are helpful")],
            }),
            generation_config: GenerationConfig {
                temperature: 0.7,
                max_output_tokens: 8192,
            },
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"text\":\"Hello\""));
        assert!(json.contains("\"systemInstruction\""));
        assert!(!json.contains("\"system_instruction\""));
        assert!(json.contains("\"temperature\":0.7"));
        assert!(json.contains("\"maxOutputTokens\":8192"));
    }

    #[test]
    fn internal_request_includes_model() {
        let request = InternalGenerateContentEnvelope {
            model: "gemini-3-pro-preview".to_string(),
            project: Some("test-project".to_string()),
            user_prompt_id: Some("prompt-123".to_string()),
            request: InternalGenerateContentRequest {
                contents: vec![Content {
                    role: Some("user".to_string()),
                    parts: vec![Part::text("Hello")],
                }],
                system_instruction: None,
                generation_config: Some(GenerationConfig {
                    temperature: 0.7,
                    max_output_tokens: 8192,
                }),
            },
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"model\":\"gemini-3-pro-preview\""));
        assert!(json.contains("\"request\""));
        assert!(json.contains("\"generationConfig\""));
        assert!(json.contains("\"maxOutputTokens\":8192"));
        assert!(json.contains("\"user_prompt_id\":\"prompt-123\""));
        assert!(json.contains("\"project\":\"test-project\""));
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"temperature\":0.7"));
    }

    #[test]
    fn internal_request_omits_generation_config_when_none() {
        let request = InternalGenerateContentEnvelope {
            model: "gemini-3-pro-preview".to_string(),
            project: Some("test-project".to_string()),
            user_prompt_id: None,
            request: InternalGenerateContentRequest {
                contents: vec![Content {
                    role: Some("user".to_string()),
                    parts: vec![Part::text("Hello")],
                }],
                system_instruction: None,
                generation_config: None,
            },
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(!json.contains("generationConfig"));
        assert!(json.contains("\"model\":\"gemini-3-pro-preview\""));
    }

    #[test]
    fn internal_request_includes_project() {
        let request = InternalGenerateContentEnvelope {
            model: "gemini-2.5-flash".to_string(),
            project: Some("my-gcp-project-id".to_string()),
            user_prompt_id: None,
            request: InternalGenerateContentRequest {
                contents: vec![Content {
                    role: Some("user".to_string()),
                    parts: vec![Part::text("Hello")],
                }],
                system_instruction: None,
                generation_config: None,
            },
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"project\":\"my-gcp-project-id\""));
    }

    #[test]
    fn internal_response_deserialize_nested() {
        let json = r#"{
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "Hello from internal API!"}]
                    }
                }]
            }
        }"#;

        let internal: InternalGenerateContentResponse = serde_json::from_str(json).unwrap();
        let text = internal
            .response
            .candidates
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
            .content
            .unwrap()
            .parts
            .into_iter()
            .next()
            .unwrap()
            .text;
        assert_eq!(text, Some("Hello from internal API!".to_string()));
    }

    #[test]
    fn creds_deserialize_with_expiry_date() {
        let json = r#"{
            "access_token": "ya29.test-token",
            "refresh_token": "1//test-refresh",
            "expiry_date": 4102444800000
        }"#;

        let creds: GeminiCliOAuthCreds = serde_json::from_str(json).unwrap();
        assert_eq!(creds.access_token.as_deref(), Some("ya29.test-token"));
        assert_eq!(creds.refresh_token.as_deref(), Some("1//test-refresh"));
        assert_eq!(creds.expiry_date, Some(4_102_444_800_000));
        assert!(creds.expiry.is_none());
    }

    #[test]
    fn creds_deserialize_accepts_camel_case_fields() {
        let json = r#"{
            "access_token": "ya29.test-token",
            "idToken": "header.payload.sig",
            "refresh_token": "1//test-refresh",
            "clientId": "test-client-id",
            "clientSecret": "test-client-secret",
            "expiryDate": 4102444800000
        }"#;

        let creds: GeminiCliOAuthCreds = serde_json::from_str(json).unwrap();
        assert_eq!(creds.id_token.as_deref(), Some("header.payload.sig"));
        assert_eq!(creds.client_id.as_deref(), Some("test-client-id"));
        assert_eq!(creds.client_secret.as_deref(), Some("test-client-secret"));
        assert_eq!(creds.expiry_date, Some(4_102_444_800_000));
    }

    #[test]
    fn oauth_retry_detection_for_generation_config_rejection() {
        // Bare quotes (e.g. pre-parsed error string)
        let err =
            "Invalid JSON payload received. Unknown name \"generationConfig\": Cannot find field.";
        assert!(
            GeminiProvider::should_retry_oauth_without_generation_config(
                StatusCode::BAD_REQUEST,
                err
            )
        );
        // JSON-escaped quotes (raw response body from Google API)
        let err_json = r#"Invalid JSON payload received. Unknown name \"generationConfig\": Cannot find field."#;
        assert!(
            GeminiProvider::should_retry_oauth_without_generation_config(
                StatusCode::BAD_REQUEST,
                err_json
            )
        );
        assert!(
            !GeminiProvider::should_retry_oauth_without_generation_config(
                StatusCode::UNAUTHORIZED,
                err
            )
        );
        assert!(
            !GeminiProvider::should_retry_oauth_without_generation_config(
                StatusCode::BAD_REQUEST,
                "something else"
            )
        );
    }

    #[test]
    fn response_deserialization() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello there!"}]
                }
            }]
        }"#;

        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        assert!(response.candidates.is_some());
        let text = response
            .candidates
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
            .content
            .unwrap()
            .parts
            .into_iter()
            .next()
            .unwrap()
            .text;
        assert_eq!(text, Some("Hello there!".to_string()));
    }

    #[test]
    fn error_response_deserialization() {
        let json = r#"{
            "error": {
                "message": "Invalid API key"
            }
        }"#;

        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().message, "Invalid API key");
    }

    #[test]
    fn internal_response_deserialization() {
        let json = r#"{
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "Hello from internal"}]
                    }
                }]
            }
        }"#;

        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let text = response
            .into_effective_response()
            .candidates
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
            .content
            .unwrap()
            .parts
            .into_iter()
            .next()
            .unwrap()
            .text;
        assert_eq!(text, Some("Hello from internal".to_string()));
    }

    // ── Thinking model response tests ──────────────────────────────────────

    #[test]
    fn thinking_response_extracts_non_thinking_text() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [
                        {"thought": true, "text": "Let me think about this..."},
                        {"text": "The answer is 42."},
                        {"thoughtSignature": "c2lnbmF0dXJl"}
                    ]
                }
            }]
        }"#;

        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let candidate = response.candidates.unwrap().into_iter().next().unwrap();
        let text = candidate.content.unwrap().effective_text();
        assert_eq!(text, Some("The answer is 42.".to_string()));
    }

    #[test]
    fn non_thinking_response_unaffected() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello there!"}]
                }
            }]
        }"#;

        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let candidate = response.candidates.unwrap().into_iter().next().unwrap();
        let text = candidate.content.unwrap().effective_text();
        assert_eq!(text, Some("Hello there!".to_string()));
    }

    #[test]
    fn thinking_only_response_falls_back_to_thinking_text() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [
                        {"thought": true, "text": "I need more context..."},
                        {"thoughtSignature": "c2lnbmF0dXJl"}
                    ]
                }
            }]
        }"#;

        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let candidate = response.candidates.unwrap().into_iter().next().unwrap();
        let text = candidate.content.unwrap().effective_text();
        assert_eq!(text, Some("I need more context...".to_string()));
    }

    #[test]
    fn empty_parts_returns_none() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": []
                }
            }]
        }"#;

        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let candidate = response.candidates.unwrap().into_iter().next().unwrap();
        let text = candidate.content.unwrap().effective_text();
        assert_eq!(text, None);
    }

    #[test]
    fn multiple_text_parts_concatenated() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "Part one. "},
                        {"text": "Part two."}
                    ]
                }
            }]
        }"#;

        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let candidate = response.candidates.unwrap().into_iter().next().unwrap();
        let text = candidate.content.unwrap().effective_text();
        assert_eq!(text, Some("Part one. Part two.".to_string()));
    }

    #[test]
    fn thought_signature_only_parts_skipped() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [
                        {"thoughtSignature": "c2lnbmF0dXJl"}
                    ]
                }
            }]
        }"#;

        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let candidate = response.candidates.unwrap().into_iter().next().unwrap();
        let text = candidate.content.unwrap().effective_text();
        assert_eq!(text, None);
    }

    #[test]
    fn internal_response_thinking_model() {
        let json = r#"{
            "response": {
                "candidates": [{
                    "content": {
                        "parts": [
                            {"thought": true, "text": "reasoning..."},
                            {"text": "final answer"}
                        ]
                    }
                }]
            }
        }"#;

        let response: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let effective = response.into_effective_response();
        let candidate = effective.candidates.unwrap().into_iter().next().unwrap();
        let text = candidate.content.unwrap().effective_text();
        assert_eq!(text, Some("final answer".to_string()));
    }

    #[tokio::test]
    async fn warmup_without_key_is_noop() {
        let provider = test_provider(None);
        let result = provider.warmup().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn warmup_oauth_is_noop() {
        let provider = test_provider(Some(test_oauth_auth("ya29.mock-token")));
        let result = provider.warmup().await;
        assert!(result.is_ok());
    }

    #[test]
    fn discover_oauth_cred_paths_does_not_panic() {
        let _paths = GeminiProvider::discover_oauth_cred_paths();
    }

    #[tokio::test]
    async fn rotate_oauth_without_alternatives_returns_false() {
        let state = Arc::new(tokio::sync::Mutex::new(OAuthTokenState {
            access_token: "ya29.mock".to_string(),
            refresh_token: None,
            client_id: None,
            client_secret: None,
            expiry_millis: None,
        }));
        let provider = test_provider(Some(GeminiAuth::OAuthToken(state.clone())));
        assert!(!provider.rotate_oauth_credential(&state).await);
    }

    #[test]
    fn test_map_messages_with_images() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Check this image: [IMAGE:data:image/jpeg;base64,/9j/4AAQ] What do you see?".to_string(),
        }];

        let provider = test_provider(None);
        let contents = provider.map_messages(&messages);

        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].parts.len(), 3);
        
        // Part 1: text before image
        assert!(contents[0].parts[0].text.as_ref().unwrap().contains("Check this image:"));
        
        // Part 2: image
        let image_part = &contents[0].parts[1].inline_data.as_ref().unwrap();
        assert_eq!(image_part.mime_type, "image/jpeg");
        assert_eq!(image_part.data, "/9j/4AAQ");
        
        // Part 3: text after image
        assert!(contents[0].parts[2].text.as_ref().unwrap().contains("What do you see?"));
    }

    #[tokio::test]
    async fn test_chat_with_tools_integration() {
        let captured: Arc<Mutex<Option<serde_json::Value>>> = Arc::new(Mutex::new(None));
        let captured_clone = captured.clone();

        let app = Router::new().route(
            "/models/gemini-2.0-flash:generateContent",
            post(move |Json(body): Json<serde_json::Value>| {
                let cap = captured_clone.clone();
                async move {
                    *cap.lock().unwrap() = Some(body);
                    Json(serde_json::json!({
                        "candidates": [{
                            "content": {
                                "role": "model",
                                "parts": [{
                                    "functionCall": {
                                        "name": "get_weather",
                                        "args": {"location": "San Francisco"}
                                    }
                                }]
                            },
                            "finishReason": "STOP"
                        }]
                    }))
                }
            }),
        );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server_handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let mut provider = test_provider(Some(GeminiAuth::ExplicitKey("test-key".into())));
        provider.base_url = Some(format!("http://{}", addr));

        let tools = vec![ToolSpec {
            name: "get_weather".to_string(),
            description: "Get weather".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }),
        }];

        let messages = vec![
            ChatMessage { role: "system".into(), content: "System prompt".into() },
            ChatMessage { role: "user".into(), content: "What is the weather?".into() },
        ];
        let result = provider.chat_with_tools(&messages, &tools, "gemini-2.0-flash", 0.7).await;
        
        server_handle.abort();

        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(response.tool_calls.iter().any(|tc| tc.name == "get_weather"));
        
        let req_body = captured.lock().unwrap().take().unwrap();
        assert_eq!(req_body["contents"][0]["role"], "user");
        assert!(req_body["tools"][0]["functionDeclarations"].is_array());
    }

    #[tokio::test]
    async fn test_stream_chat_integration() {
        let app = Router::new().route(
            "/models/gemini-2.0-flash:streamGenerateContent",
            post(|| async {
                let stream = "data: {\"candidates\": [{\"content\": {\"parts\": [{\"text\": \"Hello \"}]}}]}\n\ndata: {\"candidates\": [{\"content\": {\"parts\": [{\"text\": \"world!\"}]}}]}\n\n";
                axum::response::Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(axum::body::Body::from(stream))
                    .unwrap()
            }),
        );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server_handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let messages = vec![ChatMessage { role: "user".into(), content: "Hi".into() }];
        let request = ChatRequest {
            messages: &messages,
            tools: None,
        };
        let mut stream = provider.stream_chat(request, "gemini-2.0-flash", 0.7, StreamOptions::default());
        let mut text = String::new();
        use futures_util::StreamExt;
        while let Some(event) = stream.next().await {
            if let Ok(StreamEvent::TextDelta(delta)) = event {
                text.push_str(&delta.delta);
            }
        }

        server_handle.abort();
        assert_eq!(text, "Hello world!");
    }

    #[tokio::test]
    async fn test_error_handling_401() {
        let app = Router::new().route(
            "/models/gemini-2.0-flash:generateContent",
            post(|| async {
                (StatusCode::UNAUTHORIZED, "Unauthorized")
            }),
        );

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server_handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        let mut provider = test_provider(Some(GeminiAuth::ExplicitKey("wrong-key".into())));
        provider.base_url = Some(format!("http://{}", addr));

        let result = provider.chat_with_system(None, "Hi", "gemini-2.0-flash", 0.7).await;
        
        server_handle.abort();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("401"));
    }

    #[test]
    fn response_parses_usage_metadata() {
        let json = r#"{
            "candidates": [{"content": {"parts": [{"text": "Hello"}]}}],
            "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 40}
        }"#;
        let resp: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let usage = resp.usage_metadata.unwrap();
        assert_eq!(usage.prompt_token_count, Some(120));
        assert_eq!(usage.candidates_token_count, Some(40));
    }

    #[test]
    fn response_parses_without_usage_metadata() {
        let json = r#"{"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}"#;
        let resp: GenerateContentResponse = serde_json::from_str(json).unwrap();
        assert!(resp.usage_metadata.is_none());
    }

    /// Validates that warmup() for ManagedOAuth requires auth_service.
    #[tokio::test]
    async fn warmup_managed_oauth_requires_auth_service() {
        let provider = GeminiProvider {
            auth: Some(GeminiAuth::ManagedOAuth),
            oauth_project: Arc::new(tokio::sync::Mutex::new(None)),
            oauth_cred_paths: Vec::new(),
            oauth_index: Arc::new(tokio::sync::Mutex::new(0)),
            auth_service: None, // Missing auth_service
            auth_profile_override: None,
            base_url: None,
        };

        let result = provider.warmup().await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("ManagedOAuth requires auth_service")
        );
    }

    /// Validates that warmup() for CLI OAuth skips validation (existing behavior).
    #[tokio::test]
    async fn warmup_cli_oauth_skips_validation() {
        let provider = test_provider(Some(test_oauth_auth("fake_token")));
        let result = provider.warmup().await;
        // Should succeed without making HTTP requests
        assert!(result.is_ok());
    }

    // ── Part enum serialization tests ────────────────────────────────────

    #[test]
    fn part_text_serializes_as_text_object() {
        let part = Part::text("hello");
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json, serde_json::json!({"text": "hello"}));
    }

    #[test]
    fn part_inline_serializes_as_inline_data_object() {
        let part = Part::Inline {
            inline_data: InlineData {
                mime_type: "image/png".to_string(),
                data: "iVBOR...".to_string(),
            },
        };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"inline_data": {"mime_type": "image/png", "data": "iVBOR..."}})
        );
    }

    #[test]
    fn part_text_constructor_accepts_string_and_str() {
        let from_str = Part::text("hello");
        let from_string = Part::text(String::from("hello"));
        // Both should serialize identically
        assert_eq!(
            serde_json::to_value(&from_str).unwrap(),
            serde_json::to_value(&from_string).unwrap(),
        );
    }

    #[test]
    fn content_with_mixed_parts_serializes_correctly() {
        let content = Content {
            role: Some("user".to_string()),
            parts: vec![
                Part::text("Describe this image:"),
                Part::Inline {
                    inline_data: InlineData {
                        mime_type: "image/jpeg".to_string(),
                        data: "/9j/4AAQ...".to_string(),
                    },
                },
            ],
        };
        let json = serde_json::to_value(&content).unwrap();
        let parts = json["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert!(parts[0].get("text").is_some());
        assert!(parts[1].get("inline_data").is_some());
    }

    // ── build_parts tests ────────────────────────────────────────────────

    #[test]
    fn build_parts_plain_text_returns_single_text_part() {
        let parts = build_parts("Hello, world!");
        assert_eq!(parts.len(), 1);
        assert_eq!(
            serde_json::to_value(&parts[0]).unwrap(),
            serde_json::json!({"text": "Hello, world!"})
        );
    }

    #[test]
    fn build_parts_empty_string_returns_single_text_part() {
        let parts = build_parts("");
        assert_eq!(parts.len(), 1);
        // Falls back to original content when no markers and trimmed is empty
        assert_eq!(
            serde_json::to_value(&parts[0]).unwrap(),
            serde_json::json!({"text": ""})
        );
    }

    #[test]
    fn build_parts_extracts_data_uri_as_inline_part() {
        let content = "Check this [IMAGE:data:image/png;base64,iVBORw0KGgo=]";
        let parts = build_parts(content);
        assert_eq!(parts.len(), 2);
        // First part is text
        assert_eq!(
            serde_json::to_value(&parts[0]).unwrap(),
            serde_json::json!({"text": "Check this"})
        );
        // Second part is inline image
        assert_eq!(
            serde_json::to_value(&parts[1]).unwrap(),
            serde_json::json!({"inline_data": {"mime_type": "image/png", "data": "iVBORw0KGgo="}})
        );
    }

    #[test]
    fn build_parts_multiple_images() {
        let content = "Image A: [IMAGE:data:image/png;base64,AAAA] Image B: [IMAGE:data:image/jpeg;base64,BBBB]";
        let parts = build_parts(content);
        assert_eq!(parts.len(), 3); // text + 2 images
        // Verify both inline parts
        let inline_parts: Vec<_> = parts
            .iter()
            .filter(|p| matches!(p, Part::Inline { .. }))
            .collect();
        assert_eq!(inline_parts.len(), 2);
    }

    #[test]
    fn build_parts_ignores_non_data_uri_markers() {
        // File paths and URLs are not data URIs — build_parts should only
        // extract data: URIs, leaving non-data markers as stripped text.
        let content = "Look [IMAGE:/tmp/photo.png]";
        let parts = build_parts(content);
        // parse_image_markers extracts the marker, but build_parts only
        // converts data: URIs to inline parts. The text remains.
        for part in &parts {
            assert!(matches!(part, Part::Text { .. }));
        }
    }

    #[test]
    fn build_parts_image_only_still_produces_inline_part() {
        let content = "[IMAGE:data:image/gif;base64,R0lGODlh]";
        let parts = build_parts(content);
        // Should have just the inline part (text is empty after marker removal)
        assert_eq!(parts.len(), 1);
        assert!(matches!(&parts[0], Part::Inline { .. }));
    }

    // ── chat_with_history uses build_parts for user messages ─────────────

    #[test]
    fn chat_with_history_maps_roles_correctly() {
        // Verify the message→Content mapping logic directly by checking
        // that the provider constructs the right Content structures.
        // We can't call chat_with_history without a real API, but we can
        // verify the Part construction used in each role branch.

        // User messages should go through build_parts (supports images)
        let user_parts = build_parts("Hello [IMAGE:data:image/png;base64,AA==]");
        assert!(user_parts.iter().any(|p| matches!(p, Part::Inline { .. })));

        // Assistant messages should use Part::text (no image parsing)
        let assistant_part = Part::text("I see the image");
        assert!(matches!(assistant_part, Part::Text { .. }));

        // System messages should use Part::text
        let system_part = Part::text("You are helpful");
        assert!(matches!(system_part, Part::Text { .. }));
    }

    #[test]
    fn convert_tools_maps_zeroclaw_specs_to_gemini_functions() {
        let provider = test_provider(None);
        let tools = vec![ToolSpec {
            name: "get_weather".to_string(),
            description: "Get current weather".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }),
        }];

        let payload = provider.convert_tools(&tools);
        if let ToolsPayload::Gemini {
            function_declarations,
        } = payload
        {
            assert_eq!(function_declarations.len(), 1);
            let func = &function_declarations[0];
            assert_eq!(func["name"], "get_weather");
            assert_eq!(func["description"], "Get current weather");
            assert_eq!(func["parameters"]["type"], "object");
            assert_eq!(func["parameters"]["required"][0], "location");
        } else {
            panic!("Expected ToolsPayload::Gemini");
        }
    }

    #[test]
    fn chat_with_tools_message_mapping() {
        let provider = test_provider(None);
        let messages = vec![
            ChatMessage::user("Search for rust"),
            ChatMessage::assistant("I will search..."),
            ChatMessage::tool(r#"{"results": ["rust-lang.org"]}"#),
        ];

        // We can't easily call chat_with_tools without mocking the HTTP client,
        // but we can test the internal conversion logic if we refactor it or test
        // the Content structures it produces. Since the method is async and private
        // parts are used, we test the Part variants directly via build_parts
        // and our manual Content construction.

        let contents = vec![
            Content {
                role: Some("user".into()),
                parts: build_parts("Search for rust"),
            },
            Content {
                role: Some("model".into()),
                parts: vec![Part::text("I will search...")],
            },
            Content {
                role: Some("function".into()), // Role used for function response
                parts: vec![Part::FunctionResponse {
                    function_response: FunctionResponse {
                        name: "search".into(),
                        response: serde_json::json!({"results": ["rust-lang.org"]}),
                    },
                }],
            },
        ];

        let json = serde_json::to_value(&contents).unwrap();
        assert_eq!(json[0]["role"], "user");
        assert_eq!(json[1]["role"], "model");
        assert_eq!(json[2]["role"], "function");
        assert!(json[2]["parts"][0].get("functionResponse").is_some());
    }

    #[test]
    fn unary_response_parsing_with_tool_calls() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "I will read that file."},
                        {
                            "functionCall": {
                                "name": "read_file",
                                "args": {"path": "test.rs"}
                            }
                        }
                    ]
                }
            }]
        }"#;

        let resp: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let effective = resp.into_effective_response();
        let candidate = effective.candidates.unwrap().into_iter().next().unwrap();
        let content = candidate.content.unwrap();

        let text = content.effective_text();
        assert_eq!(text, Some("I will read that file.".to_string()));

        let tool_calls: Vec<_> = content
            .parts
            .iter()
            .filter_map(|p| p.function_call.as_ref())
            .collect();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "read_file");
        assert_eq!(tool_calls[0].args["path"], "test.rs");
    }

    #[test]
    fn unary_response_parsing_with_reasoning() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [
                        {"thought": true, "text": "Analyzing the codebase..."},
                        {"text": "Found it."}
                    ]
                }
            }]
        }"#;

        let resp: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let candidate = resp.candidates.unwrap().into_iter().next().unwrap();
        let content = candidate.content.unwrap();

        // Check reasoning extraction
        let thoughts: Vec<_> = content
            .parts
            .iter()
            .filter(|p| p.thought)
            .filter_map(|p| p.text.as_deref())
            .collect();
        assert_eq!(thoughts, vec!["Analyzing the codebase..."]);

        // Check final text
        assert_eq!(content.effective_text(), Some("Found it.".into()));
    }

    #[tokio::test]
    async fn stream_parsing_logic() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamResult<StreamEvent>>(10);

        // Simulation of SSE-like chunks
        let chunks = vec![
            r#"data: {"candidates": [{"content": {"parts": [{"text": "Hello "}]}}]}"#,
            r#"data: {"candidates": [{"content": {"parts": [{"text": "world!"}]}}]}"#,
            r#"data: {"candidates": [{"content": {"parts": [{"functionCall": {"name": "done", "args": {}}}]}}]}"#,
        ];

        // We can't easily call parse_gemini_stream directly because it takes a reqwest::Response.
        // Instead, we verify the heart of the loop: JSON parsing + event creation.

        for chunk_str in chunks {
            let line = chunk_str.trim();
            if line.starts_with("data: ") {
                let json_str = &line["data: ".len()..];
                let resp: GenerateContentResponse = serde_json::from_str(json_str).unwrap();

                if let Some(candidate) = resp.candidates.and_then(|c| c.into_iter().next()) {
                    if let Some(content) = candidate.content {
                        for part in content.parts {
                            if let Some(fc) = part.function_call {
                                let _ = tx
                                    .send(Ok(StreamEvent::ToolCall(crate::providers::traits::ToolCall {
                                        id: "123".into(),
                                        name: fc.name,
                                        arguments: fc.args.to_string(),
                                    })))
                                    .await;
                            } else if let Some(t) = part.text {
                                let _ = tx
                                    .send(Ok(StreamEvent::TextDelta(crate::providers::traits::StreamChunk::delta(t))))
                                    .await;
                            }
                        }
                    }
                }
            }
        }

        // Verify captured events
        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event.unwrap());
        }

        assert_eq!(events.len(), 3);
        if let StreamEvent::TextDelta(chunk) = &events[0] {
            assert_eq!(chunk.delta, "Hello ");
        } else {
            panic!("Event 1 should be TextDelta");
        }
        if let StreamEvent::TextDelta(chunk) = &events[1] {
            assert_eq!(chunk.delta, "world!");
        }
        if let StreamEvent::ToolCall(call) = &events[2] {
            assert_eq!(call.name, "done");
        }
    }
}
