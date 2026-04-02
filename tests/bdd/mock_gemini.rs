use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::net::SocketAddr;

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    candidates: Vec<Candidate>,
    usage_metadata: Option<UsageMetadata>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct Candidate {
    content: Content,
    finish_reason: String,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
struct Content {
    role: String,
    parts: Vec<Part>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
enum Part {
    Text { text: String },
    FunctionCall { name: String, args: serde_json::Value },
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct UsageMetadata {
    prompt_token_count: u32,
    candidates_token_count: u32,
    total_token_count: u32,
}

#[derive(Default)]
struct AppState {
    next_responses: Vec<GeminiResponse>,
}

#[tokio::main]
async fn main() {
    let state = Arc::new(Mutex::new(AppState::default()));

    let app = Router::new()
        .route("/mock/next_response", post(add_response))
        .route("/v1beta/models/:model:generateContent", post(handle_generate))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    println!("Mock Gemini listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn add_response(
    State(state): State<Arc<Mutex<AppState>>>,
    Json(response): Json<GeminiResponse>,
) {
    let mut state = state.lock().unwrap();
    state.next_responses.push(response);
    println!("Added mock response. Total pending: {}", state.next_responses.len());
}

async fn handle_generate(
    State(state): State<Arc<Mutex<AppState>>>,
) -> Json<GeminiResponse> {
    let mut state = state.lock().unwrap();
    if state.next_responses.is_empty() {
        // Return a default error-like response or empty
        println!("No mock responses programmed! Returning empty.");
        return Json(GeminiResponse {
            candidates: vec![Candidate {
                content: Content {
                    role: "model".to_string(),
                    parts: vec![Part::Text { text: "No mock response configured".to_string() }],
                },
                finish_reason: "STOP".to_string(),
            }],
            usage_metadata: None,
        });
    }
    let resp = state.next_responses.remove(0);
    println!("Returning mock response. Remaining: {}", state.next_responses.len());
    Json(resp)
}
