//! QuiPu-KV Dashboard — library entry point.
//!
//! Module layout:
//! - `platform`  — Layman Agent Platform v1 schema types
//! - `scanner`   — walks command-hub structure and parses YAML files
//! - `transport` — watches the command-hub root and emits trellis updates
//! - `trellis`   — folds a snapshot into HUD-ready projection tiles

pub mod platform;
pub mod scanner;
pub mod transport;
pub mod trellis;

use chrono::Utc;
use std::path::PathBuf;
use tauri::{Manager, State};
use trellis::Trellis;

pub use scanner::default_command_hub;

/// Shared application state.
pub struct AppState {
    pub command_hub_root: PathBuf,
}

/// Tauri command: rebuild and return the current trellis on demand.
/// Used by the frontend for initial hydration if it missed the event,
/// or for an explicit refresh from a UI gesture.
#[tauri::command]
fn get_trellis(state: State<'_, AppState>) -> Result<Trellis, String> {
    let snap = scanner::scan(&state.command_hub_root).map_err(|e| e.to_string())?;
    Ok(Trellis::from_snapshot(&snap, Utc::now()))
}

/// Tauri command: return the configured command-hub root path.
#[tauri::command]
fn get_command_hub_path(state: State<'_, AppState>) -> String {
    state.command_hub_root.to_string_lossy().to_string()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let command_hub_root = scanner::default_command_hub();
    tracing::info!("command-hub root: {}", command_hub_root.display());

    let state = AppState {
        command_hub_root: command_hub_root.clone(),
    };

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(state)
        .setup(move |app| {
            let handle = app.handle().clone();
            transport::spawn_watcher(handle, command_hub_root.clone())?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![get_trellis, get_command_hub_path])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
