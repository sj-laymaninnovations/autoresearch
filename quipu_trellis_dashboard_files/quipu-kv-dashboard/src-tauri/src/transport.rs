//! Transport: watches the command-hub folder and emits trellis updates.
//!
//! On any filesystem change beneath the command-hub root, the transport
//! debounces, re-scans the folder, folds the snapshot into a fresh trellis,
//! and emits it to the Tauri frontend as a single `quipu://trellis/update`
//! event. The frontend re-renders against the full trellis — simple,
//! correct, and matches the platform's polling semantics.

use crate::scanner;
use crate::trellis::Trellis;
use anyhow::{Context, Result};
use chrono::Utc;
use notify::{Config, Event as FsEvent, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter};

pub const EVENT_TRELLIS_UPDATE: &str = "quipu://trellis/update";
pub const EVENT_HYDRATED: &str = "quipu://trellis/hydrated";

/// Debounce interval for filesystem events. Box Sync and editors often
/// generate bursts of events; we coalesce them into one re-scan.
const DEBOUNCE_MS: u64 = 250;

/// Spawn a background watcher thread on `command_hub_root`.
/// Emits an initial trellis after a synchronous scan, then continues to
/// emit updated trellises on debounced filesystem changes.
pub fn spawn_watcher(app: AppHandle, command_hub_root: PathBuf) -> Result<()> {
    // Initial hydration: scan once and emit.
    let initial = build_trellis(&command_hub_root);
    if let Err(e) = app.emit(EVENT_TRELLIS_UPDATE, &initial) {
        tracing::warn!("initial trellis emit failed: {}", e);
    }
    if let Err(e) = app.emit(EVENT_HYDRATED, ()) {
        tracing::warn!("hydrated emit failed: {}", e);
    }

    thread::spawn(move || {
        if let Err(e) = watch_loop(app, command_hub_root) {
            tracing::error!("watcher loop terminated: {}", e);
        }
    });

    Ok(())
}

fn build_trellis(root: &std::path::Path) -> Trellis {
    let snap = match scanner::scan(root) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("scanner failed: {}", e);
            scanner::Snapshot::default()
        }
    };
    Trellis::from_snapshot(&snap, Utc::now())
}

fn watch_loop(app: AppHandle, root: PathBuf) -> Result<()> {
    let (tx, rx) = mpsc::channel::<notify::Result<FsEvent>>();
    // Use a closure so we don't depend on notify's optional channel features.
    let mut watcher: RecommendedWatcher = Watcher::new(
        move |res: notify::Result<FsEvent>| {
            let _ = tx.send(res);
        },
        Config::default(),
    )
    .context("constructing watcher")?;

    if root.exists() {
        watcher
            .watch(&root, RecursiveMode::Recursive)
            .with_context(|| format!("watching {}", root.display()))?;
    } else {
        tracing::warn!(
            "command-hub root does not exist yet; will not start watcher: {}",
            root.display()
        );
        // Park forever; the user should restart after creating the folder.
        // Alternative: poll for existence. Deferred for simplicity.
        loop {
            std::thread::sleep(Duration::from_secs(60));
        }
    }

    let mut last_change: Option<Instant> = None;
    loop {
        // Use recv_timeout so we can fire debounced re-scans even when the
        // channel is idle after a burst of events.
        match rx.recv_timeout(Duration::from_millis(DEBOUNCE_MS)) {
            Ok(Ok(_event)) => {
                last_change = Some(Instant::now());
            }
            Ok(Err(e)) => {
                tracing::warn!("watcher error: {}", e);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if let Some(ts) = last_change {
                    if ts.elapsed() >= Duration::from_millis(DEBOUNCE_MS) {
                        last_change = None;
                        let trellis = build_trellis(&root);
                        if let Err(e) = app.emit(EVENT_TRELLIS_UPDATE, &trellis) {
                            tracing::warn!("trellis emit failed: {}", e);
                        }
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                tracing::error!("watcher channel disconnected");
                break;
            }
        }
    }

    Ok(())
}
