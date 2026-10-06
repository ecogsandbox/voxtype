//! Audio capture module
//!
//! Provides audio recording capabilities using cpal, which works with
//! PipeWire, PulseAudio, and ALSA backends.

pub mod cpal_capture;
pub mod devices;
pub mod dual_capture;
#[cfg(feature = "onnx-common")]
pub mod enhance;
pub mod feedback;
pub mod levels;
pub mod media;
pub mod resampler;
pub mod speaker;
#[cfg(target_os = "macos")]
pub mod voice_processing;

pub use dual_capture::{AudioSourceType, DualCapture, DualSamples, SourcedSample};

use crate::config::AudioConfig;
use crate::error::AudioError;
use tokio::sync::mpsc;

/// Trait for audio capture implementations
#[async_trait::async_trait]
pub trait AudioCapture: Send + Sync {
    /// Backend selected for this recording, after start has succeeded.
    fn backend_name(&self) -> &'static str {
        "cpal"
    }

    /// A terminal capture failure. The daemon finishes with captured samples.
    fn has_failed(&self) -> bool {
        false
    }

    /// Why native capture fell back or ended early, if applicable.
    fn failure_reason(&self) -> Option<String> {
        None
    }

    /// Allocate capture resources without starting microphone input.
    async fn prepare(&mut self) {}

    /// Start capturing audio
    /// Returns a channel receiver for audio chunks (f32 samples, mono, 16kHz)
    async fn start(&mut self) -> Result<mpsc::Receiver<Vec<f32>>, AudioError>;

    /// Stop capturing and return all recorded samples
    async fn stop(&mut self) -> Result<Vec<f32>, AudioError>;

    /// Get current samples without stopping (for continuous recording modes)
    /// This drains the internal buffer and returns samples collected since the last call.
    /// Returns an empty Vec if not yet started or already stopped.
    async fn get_samples(&mut self) -> Vec<f32>;
}

/// Factory function to create audio capture
pub fn create_capture(config: &AudioConfig) -> Result<Box<dyn AudioCapture>, AudioError> {
    #[cfg(target_os = "macos")]
    if config.voice_processing {
        return Ok(Box::new(voice_processing::VoiceProcessingCapture::new(
            config,
        )?));
    }
    Ok(Box::new(cpal_capture::CpalCapture::new(config)?))
}

/// Observed dictation capture state, separate from the stable recording-state file.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct CaptureStatus {
    pub pid: u32,
    pub backend: String,
    pub active: bool,
    #[serde(default)]
    pub voice_processing: bool,
    pub error: Option<String>,
}

pub fn publish_capture_status(capture: &dyn AudioCapture, active: bool) {
    write_capture_status(&CaptureStatus {
        pid: std::process::id(),
        backend: capture.backend_name().into(),
        active: active && !capture.has_failed(),
        voice_processing: capture.backend_name() == "VoiceProcessingIO",
        error: capture.failure_reason(),
    });
}

fn write_capture_status(status: &CaptureStatus) {
    let path = crate::config::Config::runtime_dir().join("capture.json");
    let result = (|| -> std::io::Result<()> {
        std::fs::create_dir_all(path.parent().unwrap())?;
        let temp = path.with_extension("json.tmp");
        std::fs::write(&temp, serde_json::to_vec(status)?)?;
        std::fs::rename(temp, path)
    })();
    if let Err(error) = result {
        tracing::warn!(%error, "Could not publish capture state");
    }
}

pub fn capture_status() -> Option<CaptureStatus> {
    let pid = crate::daemon_status::read_pid_if_alive()? as u32;
    let data = std::fs::read(crate::config::Config::runtime_dir().join("capture.json")).ok()?;
    let status: CaptureStatus = serde_json::from_slice(&data).ok()?;
    (status.pid == pid).then_some(status)
}

pub fn mark_capture_stopped() {
    if let Some(mut status) = capture_status() {
        status.active = false;
        write_capture_status(&status);
    }
}
