//! `voxtype transcribe <file>` — one-shot transcription of an audio file.

use std::path::PathBuf;
use voxtype::{config, transcribe, vad};

/// Transcribe an audio file
pub(crate) fn transcribe_file(config: &config::Config, path: &PathBuf) -> anyhow::Result<()> {
    println!("Loading audio file: {:?}", path);
    let final_samples = read_audio(path)?;

    println!(
        "Processing {} samples ({:.2}s)...",
        final_samples.len(),
        final_samples.len() as f32 / 16000.0
    );

    let final_samples = enhance_audio(config, final_samples);
    let filter = voxtype::audio::speaker::SpeakerFilter::load(&config.audio.speaker_filter);
    let mut speech_regions = vec![0..final_samples.len()];

    // Run VAD if enabled
    if let Ok(Some(vad)) = vad::create_vad(config) {
        match vad.detect(&final_samples) {
            Ok(result) => {
                speech_regions = result.speech_regions;
                println!(
                    "VAD: {:.2}s speech ({:.1}% of audio)",
                    result.speech_duration_secs,
                    result.speech_ratio * 100.0
                );
                if !result.has_speech {
                    println!("No speech detected, skipping transcription.");
                    return Ok(());
                }
            }
            Err(e) => {
                eprintln!("VAD warning: {}", e);
                // Continue with transcription if VAD fails
            }
        }
    }

    let final_samples = match filter {
        Some(filter) => filter.filter(&final_samples, &speech_regions)?,
        None => final_samples,
    };
    if final_samples.is_empty() {
        return Ok(());
    }

    // Create transcriber and transcribe
    let transcriber = transcribe::create_transcriber(config)?;
    let text = transcriber.transcribe(&final_samples)?;

    println!("\n{}", process_transcript(config, &text));
    Ok(())
}

/// Shared WAV conversion for transcription, enrollment and calibration.
pub(crate) fn read_audio(path: &std::path::Path) -> anyhow::Result<Vec<f32>> {
    use hound::WavReader;

    let reader = WavReader::open(path)?;
    let spec = reader.spec();

    println!(
        "Audio format: {} Hz, {} channel(s), {:?}",
        spec.sample_rate, spec.channels, spec.sample_format
    );

    // Convert samples to f32 mono at 16kHz
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = 2.0_f32.powi(spec.bits_per_sample as i32 - 1);
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|s| s as f32 / max_val))
                .collect::<Result<_, _>>()?
        }
        hound::SampleFormat::Float => reader.into_samples::<f32>().collect::<Result<_, _>>()?,
    };

    // Mix to mono if stereo
    let mono_samples: Vec<f32> = if spec.channels > 1 {
        samples
            .chunks(spec.channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
            .collect()
    } else {
        samples
    };

    // Resample to 16kHz if needed
    let final_samples = if spec.sample_rate != 16000 {
        println!("Resampling from {} Hz to 16000 Hz...", spec.sample_rate);
        voxtype::audio::resampler::resample_buffer(&mono_samples, spec.sample_rate, 16000)
    } else {
        mono_samples
    };

    anyhow::ensure!(
        final_samples.iter().all(|s| s.is_finite()),
        "WAV contains non-finite samples"
    );
    Ok(final_samples)
}

/// Enrollment and calibration use the same configured enhancement as dictation.
pub(crate) fn enhance_audio(config: &config::Config, samples: Vec<f32>) -> Vec<f32> {
    #[cfg(feature = "onnx-common")]
    let samples =
        match voxtype::audio::enhance::GtcrnEnhancer::load_for_dictation(config.audio.enhance) {
            Some(enhancer) => enhancer.enhance_dictation(samples, 16000),
            None => samples,
        };
    #[cfg(not(feature = "onnx-common"))]
    if config.audio.enhance {
        tracing::warn!("Speech enhancement requires a build with ONNX support, continuing without");
    }

    samples
}

/// Apply the configured text pipeline (replacements, spoken punctuation,
/// filler filtering) to the finished transcript — the same treatment the
/// daemon gives a batch dictation before output. `voxtype transcribe`
/// shipped without this, so a config that worked for dictation silently
/// did nothing here (#581).
fn process_transcript(config: &config::Config, text: &str) -> String {
    voxtype::text::TextProcessor::new_for_language(&config.text, config.active_language())
        .process(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_conversion_resamples_and_downmixes() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let mut writer = hound::WavWriter::create(
            file.path(),
            hound::WavSpec {
                channels: 2,
                sample_rate: 48000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for _ in 0..48000 {
            writer.write_sample(8000i16).unwrap();
            writer.write_sample(-8000i16).unwrap();
        }
        writer.finalize().unwrap();
        let samples = read_audio(file.path()).unwrap();
        assert_eq!(samples.len(), 16000);
        assert!(samples.iter().all(|s| s.abs() < 1e-6));
    }

    /// #581: `voxtype transcribe <file>` must honor [text] replacements and
    /// spoken punctuation like the daemon does.
    #[test]
    fn transcript_gets_replacements_and_spoken_punctuation() {
        let mut config = config::Config::default();
        config
            .text
            .replacements
            .insert("vox type".to_string(), "voxtype".to_string());
        config.text.spoken_punctuation = true;

        assert_eq!(
            process_transcript(&config, "I use vox type period"),
            "I use voxtype."
        );
    }
}
