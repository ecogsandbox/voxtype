//! Audio capture and feedback configuration.

use serde::{Deserialize, Serialize};

/// Audio capture configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AudioConfig {
    /// PipeWire/PulseAudio device name, or "default"
    #[serde(default = "default_audio_device")]
    pub device: String,

    /// Sample rate in Hz (whisper expects 16000)
    #[serde(default = "default_audio_sample_rate")]
    pub sample_rate: u32,

    /// Run the GTCRN speech enhancer on dictation audio before transcription
    /// (noise and speaker bleed removal, 16 kHz). Model: voxtype setup enhancer.
    #[serde(default)]
    pub enhance: bool,

    /// Frozen voiceprint gate for batch dictation.
    #[serde(default)]
    pub speaker_filter: SpeakerFilterConfig,

    /// Maximum recording duration in seconds (safety limit)
    #[serde(default = "default_audio_max_duration_secs")]
    pub max_duration_secs: u32,

    /// Pause MPRIS media players during recording and resume on stop
    #[serde(default)]
    pub pause_media: bool,

    /// MPRIS player bus-name suffixes to skip when pausing. Matched against
    /// the part after `org.mpris.MediaPlayer2.` either exactly or as a
    /// `<entry>.<instance>` prefix (e.g. `"chromium"` matches
    /// `chromium.instance123`). Useful for ignoring browsers whose MPRIS
    /// status is unreliable, or background players you never want paused.
    #[serde(default)]
    pub pause_media_ignored_players: Vec<String>,

    /// Lower active media stream volume during recording and restore on stop
    #[serde(default)]
    pub duck_media: bool,

    /// Fraction of its current amplitude a ducked stream keeps, in percent
    /// (50 = half the amplitude, -6 dB). Converted internally to PulseAudio's
    /// cubic percentage scale.
    #[serde(default = "default_duck_media_volume_percent")]
    pub duck_media_volume_percent: u8,

    /// Fade duration in milliseconds for the ducking ramp (0 = instant)
    #[serde(default = "default_duck_media_fade_ms")]
    pub duck_media_fade_ms: u32,

    /// Audio feedback settings
    #[serde(default)]
    pub feedback: AudioFeedbackConfig,

    /// Auto-stop an external-trigger (`voxtype record start`, e.g. from a
    /// wake-word integration) recording after this many seconds of
    /// silence. Unset (default) disables it — the recording only ends on
    /// an explicit `record stop`/`record toggle`, same as before this
    /// option existed. Never applies to hotkey-driven push-to-talk or
    /// toggle recordings, which already have an explicit user-driven stop;
    /// only sessions started via `record start` (SIGUSR1) arm this.
    #[serde(default)]
    pub external_trigger_silence_timeout_secs: Option<f32>,

    /// Peak level (dBFS) at or above which a frame counts as speech for
    /// `external_trigger_silence_timeout_secs`. Only meaningful when that
    /// option is set. -20 dBFS sits clearly above the ambient noise floor
    /// of typical laptop mics (measured here at ~-30 dBFS median) while
    /// still catching normal dictation speech; raise it (toward 0) on a
    /// noisier mic, lower it (more negative) if soft speech isn't resetting
    /// the silence timer. A single frame only counts as speech once a short
    /// burst sustains it (see `SPEECH_BURST_FRAMES`), so the exact value is
    /// not spike-sensitive.
    #[serde(default = "default_external_trigger_speech_threshold_dbfs")]
    pub external_trigger_speech_threshold_dbfs: f32,

    /// Shell command run whenever an external-trigger (`record start` /
    /// SIGUSR1) recording ends, for *any* reason: an explicit `record
    /// stop`/`record toggle`, `external_trigger_silence_timeout_secs`
    /// firing, or the `max_duration_secs` hard cap. Never fires for
    /// hotkey-driven push-to-talk/toggle recordings.
    ///
    /// External-trigger integrations (a wake-word daemon, a voice
    /// assistant plugin) typically start recording and then wait for
    /// *themselves* to be told to stop it — but silence-timeout means
    /// voxtype can now end the session on its own, with no way for the
    /// caller to know unless something tells it. Point this at whatever
    /// re-signals your integration (for OmaPilot's wake-word plugin:
    /// `omarchy-shell -q io.github.spencerbull.omapilot voiceToggle`,
    /// the same command a second wake word would send).
    #[serde(default)]
    pub external_trigger_stop_command: Option<String>,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            device: default_audio_device(),
            sample_rate: default_audio_sample_rate(),
            enhance: false,
            speaker_filter: SpeakerFilterConfig::default(),
            max_duration_secs: default_audio_max_duration_secs(),
            pause_media: false,
            pause_media_ignored_players: Vec::new(),
            duck_media: false,
            duck_media_volume_percent: default_duck_media_volume_percent(),
            duck_media_fade_ms: default_duck_media_fade_ms(),
            feedback: AudioFeedbackConfig::default(),
            external_trigger_silence_timeout_secs: None,
            external_trigger_speech_threshold_dbfs: default_external_trigger_speech_threshold_dbfs(
            ),
            external_trigger_stop_command: None,
        }
    }
}

/// Speaker filtering uses a frozen ECAPA voiceprint, never online adaptation.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct SpeakerFilterConfig {
    pub enabled: bool,
    pub voiceprint: String,
    /// Cosine similarity cutoff. Calibrate with `voxtype voiceprint test`.
    pub threshold: f32,
    /// Speech shorter than this bypasses the gate (at least 1.5 seconds).
    pub min_speech_secs: f32,
}

impl Default for SpeakerFilterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            voiceprint: String::new(),
            threshold: 0.5,
            min_speech_secs: 1.5,
        }
    }
}

impl SpeakerFilterConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.threshold.is_finite() && (-1.0..=1.0).contains(&self.threshold),
            "audio.speaker_filter.threshold must be a finite cosine similarity between -1 and 1"
        );
        anyhow::ensure!(
            self.min_speech_secs.is_finite() && self.min_speech_secs >= 1.5,
            "audio.speaker_filter.min_speech_secs must be finite and at least 1.5"
        );
        Ok(())
    }
}

fn default_audio_device() -> String {
    "default".to_string()
}

fn default_audio_sample_rate() -> u32 {
    16000
}

fn default_audio_max_duration_secs() -> u32 {
    60
}

/// 34 rather than 70 because the value now means amplitude directly. Before
/// the cube-root correction the configured percentage was applied to
/// PulseAudio's own cubic scale, so the shipped default of 70 actually left
/// 0.70^3 = 0.343 of the amplitude. 34 reproduces that same audible depth
/// under the corrected meaning; keeping 70 would have quietly turned a -9.3 dB
/// duck into a -3.1 dB one for everyone on defaults.
fn default_duck_media_volume_percent() -> u8 {
    34
}

fn default_duck_media_fade_ms() -> u32 {
    150
}

fn default_external_trigger_speech_threshold_dbfs() -> f32 {
    -20.0
}

/// Audio feedback configuration for sound cues
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AudioFeedbackConfig {
    /// Enable audio feedback sounds
    #[serde(default)]
    pub enabled: bool,

    /// Sound theme: "default", "subtle", "mechanical", or path to custom theme directory
    #[serde(default = "default_sound_theme")]
    pub theme: String,

    /// Volume level (0.0 to 1.0)
    #[serde(default = "default_volume")]
    pub volume: f32,
}

fn default_sound_theme() -> String {
    "default".to_string()
}

fn default_volume() -> f32 {
    0.7
}

impl Default for AudioFeedbackConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            theme: default_sound_theme(),
            volume: default_volume(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::config::Config;

    #[test]
    fn test_speaker_filter_config() {
        for text in ["", "[audio]", "[audio.speaker_filter]"] {
            let config: Config = toml::from_str(text).unwrap();
            let filter = config.audio.speaker_filter;
            assert!(!filter.enabled);
            assert_eq!(filter.voiceprint, "");
            assert_eq!(filter.threshold, 0.5);
            assert_eq!(filter.min_speech_secs, 1.5);
        }
        let config: Config = toml::from_str(
            r#"
            [audio.speaker_filter]
            enabled = true
            voiceprint = "/tmp/voice.json"
            threshold = 0.72
            min_speech_secs = 2.0
        "#,
        )
        .unwrap();
        let mut filter = config.audio.speaker_filter;
        assert!(filter.enabled);
        assert_eq!(filter.voiceprint, "/tmp/voice.json");
        assert_eq!(filter.threshold, 0.72);
        assert_eq!(filter.min_speech_secs, 2.0);
        assert!(filter.validate().is_ok());
        filter.threshold = f32::NAN;
        assert!(filter.validate().is_err());
        filter.threshold = 0.5;
        filter.min_speech_secs = 1.0;
        assert!(filter.validate().is_err());
    }

    #[test]
    fn test_audio_enhance() {
        assert!(!Config::default().audio.enhance);
        let config: Config = toml::from_str("[audio]").unwrap();
        assert!(!config.audio.enhance);
        let config: Config = toml::from_str("[audio]\nenhance = false").unwrap();
        assert!(!config.audio.enhance);
        let config: Config = toml::from_str("[audio]\nenhance = true").unwrap();
        assert!(config.audio.enhance);
    }
}
