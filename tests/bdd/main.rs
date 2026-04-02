use cucumber::{gherkin::Step, given, then, when, World};
use reqwest::Client;
use serde_json::json;
use std::time::Duration;

#[derive(Debug, World)]
pub struct ZeroClawWorld {
    client: Client,
    last_response: Option<String>,
}

impl Default for ZeroClawWorld {
    fn default() -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap(),
            last_response: None,
        }
    }
}

#[given("a configured ZeroClaw instance using Gemini")]
async fn configured_zeroclaw(_world: &mut ZeroClawWorld) {
    // In Docker Compose, ZeroClaw is already running and configured via env vars.
    // We just verify it's healthy.
    let client = Client::new();
    let mut retry = 0;
    while retry < 10 {
        if let Ok(resp) = client.get("http://zeroclaw-app:42617/health").send().await {
            if resp.status().is_success() {
                return;
            }
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
        retry += 1;
    }
    panic!("ZeroClaw did not become healthy in time");
}

#[given(expr = "Gemini mock is programmed to return {string}")]
async fn program_mock_text(world: &mut ZeroClawWorld, response_text: String) {
    let mock_resp = json!({
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{ "text": response_text }]
                },
                "finishReason": "STOP"
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 10,
            "totalTokenCount": 20
        }
    });

    world.client
        .post("http://gemini-mock:3000/mock/next_response")
        .json(&mock_resp)
        .send()
        .await
        .expect("Failed to program gemini-mock");
}

#[given(expr = "Gemini mock is programmed to request tool {string} with args {string}")]
async fn program_mock_tool(world: &mut ZeroClawWorld, tool_name: String, args: String) {
    let args_json: serde_json::Value = serde_json::from_str(&args).expect("Invalid JSON args");
    let mock_resp = json!({
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": tool_name,
                            "args": args_json
                        }
                    }]
                },
                "finishReason": "STOP"
            }
        ]
    });

    world.client
        .post("http://gemini-mock:3000/mock/next_response")
        .json(&mock_resp)
        .send()
        .await
        .expect("Failed to program gemini-mock for tool");
}

#[given(expr = "Gemini mock is programmed to then return {string}")]
async fn program_mock_second_response(world: &mut ZeroClawWorld, response_text: String) {
    // Same as program_mock_text, but it appends to the queue in mock_gemini.rs
    program_text(world, response_text).await;
}

// Internal helper to avoid duplication if needed, but for simplicity I'll just use the steps.
async fn program_text(world: &mut ZeroClawWorld, text: String) {
    let mock_resp = json!({
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{ "text": text }]
                },
                "finishReason": "STOP"
            }
        ]
    });
    world.client
        .post("http://gemini-mock:3000/mock/next_response")
        .json(&mock_resp)
        .send()
        .await
        .unwrap();
}

#[when(expr = "I send the message {string}")]
async fn send_message(world: &mut ZeroClawWorld, message: String) {
    let resp = world.client
        .post("http://zeroclaw-app:42617/webhook")
        .json(&json!({ "message": message }))
        .send()
        .await
        .expect("Failed to send message to ZeroClaw");

    let status = resp.status();
    let body = resp.text().await.expect("Failed to read response body");
    
    if !status.is_success() {
        panic!("ZeroClaw returned error status {}: {}", status, body);
    }

    // ZeroClaw /webhook seems to return JSON. Let's parse it if possible, or just use as string.
    world.last_response = Some(body);
}

#[then(expr = "I should receive the response {string}")]
async fn check_response(world: &mut ZeroClawWorld, expected: String) {
    let actual = world.last_response.as_ref().expect("No response received");
    // Depending on what /webhook returns, we might need to parse.
    // Assuming it returns something like {"response": "..."} or just text.
    if !actual.contains(&expected) {
        panic!("Expected response to contain '{}', but got: '{}'", expected, actual);
    }
}

#[tokio::main]
async fn main() {
    ZeroClawWorld::run("tests/bdd/features").await;
}
