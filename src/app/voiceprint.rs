//! Enrollment and calibration share dictation's preprocessing and ECAPA windows.

use voxtype::{cli::VoiceprintAction, config::Config};

pub(crate) fn run(config: &Config, action: VoiceprintAction) -> anyhow::Result<()> {
    #[cfg(not(feature = "ml-diarization"))]
    {
        let _ = (config, action);
        anyhow::bail!("Voiceprints require a build with ml-diarization");
    }
    #[cfg(feature = "ml-diarization")]
    {
        use voxtype::audio::speaker::{self, Voiceprint, SAMPLE_RATE};

        config.audio.speaker_filter.validate()?;
        let vad = voxtype::vad::create_vad(config)?;
        let prepare = |path: &std::path::Path| -> anyhow::Result<Vec<f32>> {
            let samples = super::transcribe_file::enhance_audio(
                config,
                super::transcribe_file::read_audio(path)?,
            );
            let regions = match &vad {
                Some(vad) => vad.detect(&samples)?.speech_regions,
                None => vec![0..samples.len()],
            };
            Ok(speaker::speech_audio(&samples, &regions))
        };
        match action {
            VoiceprintAction::Enroll { wav, out } => {
                let clips = wav
                    .iter()
                    .map(|path| prepare(path))
                    .collect::<anyhow::Result<Vec<_>>>()?;
                let speech_len = clips.iter().map(Vec::len).sum::<usize>();
                let speech_secs = speech_len as f64 / SAMPLE_RATE as f64;
                anyhow::ensure!(
                    speech_len >= 10 * SAMPLE_RATE,
                    "Enrollment needs at least 10 s of speech; found {:.3} s. Record more speech or check your VAD settings.",
                    (speech_secs * 1000.0).floor() / 1000.0
                );
                let path = voxtype::setup::model::ensure_ecapa_model()
                    .ok_or_else(|| anyhow::anyhow!("Could not install the ECAPA model"))?;
                let model = speaker::load_embedding_model(&path)?;
                let mut embeddings = Vec::new();
                let speech: Vec<f32> = clips.into_iter().flatten().collect();
                for window in speaker::windows(speech.len()) {
                    embeddings.push(
                        model
                            .extract_embedding(&speech[window])
                            .map_err(anyhow::Error::msg)?,
                    );
                }
                let count = embeddings.len();
                Voiceprint::from_embeddings(embeddings)?.save(&out)?;
                println!(
                    "Enrolled {:.2} s of speech, {} windows. Saved {}",
                    speech_secs,
                    count,
                    out.display()
                );
            }
            VoiceprintAction::Test {
                wav,
                voiceprint,
                threshold,
            } => {
                let mut settings = config.audio.speaker_filter.clone();
                if let Some(threshold) = threshold {
                    settings.threshold = threshold;
                }
                settings.validate()?;
                let voiceprint = Voiceprint::load(&voiceprint)?;
                let clip = prepare(&wav)?;
                let windows = speaker::windows(clip.len());
                anyhow::ensure!(
                    !windows.is_empty(),
                    "No complete 1.5 s speech windows; dictation would bypass the speaker gate"
                );
                let model = speaker::load_embedding_model(&voxtype::setup::model::ecapa_model_path())?;
                let mut accepted = 0;
                println!("Window times refer to VAD-positive audio with silence removed.");
                for window in &windows {
                    let similarity = voiceprint.similarity(
                        model
                            .extract_embedding(&clip[window.clone()])
                            .map_err(anyhow::Error::msg)?,
                    )?;
                    let keep = similarity >= settings.threshold;
                    accepted += usize::from(keep);
                    println!(
                        "{:.3}-{:.3}s similarity={:.6} {}",
                        window.start as f32 / SAMPLE_RATE as f32,
                        window.end as f32 / SAMPLE_RATE as f32,
                        similarity,
                        if keep { "keep" } else { "drop" }
                    );
                }
                println!(
                    "Share >= {:.4}: {}/{} ({:.2}%)",
                    settings.threshold,
                    accepted,
                    windows.len(),
                    accepted as f64 / windows.len() as f64 * 100.0
                );
            }
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "ml-diarization"))]
mod tests {
    use super::*;

    #[test]
    fn enrollment_rejects_under_ten_seconds_before_loading_ecapa() {
        let dir = tempfile::tempdir().unwrap();
        let wav = dir.path().join("short.wav");
        let out = dir.path().join("voiceprint.json");
        let mut writer = hound::WavWriter::create(
            &wav,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for _ in 0..159_999 {
            writer.write_sample(1000i16).unwrap();
        }
        writer.finalize().unwrap();
        let error = run(
            &Config::default(),
            VoiceprintAction::Enroll {
                wav: vec![wav],
                out: out.clone(),
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("at least 10 s of speech"));
        assert!(!out.exists());
    }
}
