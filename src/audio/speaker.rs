//! Frozen ECAPA voiceprints and the shared batch dictation speaker gate.

use crate::config::SpeakerFilterConfig;
use anyhow::{ensure, Context};
use serde::{Deserialize, Serialize};
use std::ops::Range;
use std::path::Path;

pub const SAMPLE_RATE: usize = 16_000;
pub const WINDOW: usize = SAMPLE_RATE * 3 / 2;
pub const HOP: usize = SAMPLE_RATE / 2;
const JOIN_SILENCE: usize = SAMPLE_RATE / 10;

#[derive(Debug, Serialize, Deserialize)]
pub struct Voiceprint {
    pub model: String,
    pub dim: usize,
    pub embedding: Vec<f32>,
}

fn normalize(embedding: &mut [f32]) -> anyhow::Result<()> {
    let norm = embedding
        .iter()
        .map(|&x| (x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    ensure!(
        norm.is_finite() && norm > 0.0,
        "Invalid speaker embedding: expected a finite, nonzero vector"
    );
    for value in embedding {
        *value = (*value as f64 / norm) as f32;
    }
    Ok(())
}

impl Voiceprint {
    /// Normalize each window before averaging, then normalize the frozen mean.
    pub fn from_embeddings(embeddings: Vec<Vec<f32>>) -> anyhow::Result<Self> {
        let dim = embeddings
            .first()
            .context("No complete 1.5 s speech windows")?
            .len();
        let mut mean = vec![0.0; dim];
        let count = embeddings.len() as f32;
        for mut embedding in embeddings {
            ensure!(
                embedding.len() == dim,
                "Speaker embedding dimension mismatch"
            );
            normalize(&mut embedding)?;
            for (sum, value) in mean.iter_mut().zip(embedding) {
                *sum += value / count;
            }
        }
        normalize(&mut mean)?;
        Ok(Self {
            model: "ecapa_tdnn".into(),
            dim,
            embedding: mean,
        })
    }

    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let mut voiceprint: Self = serde_json::from_reader(
            std::fs::File::open(path)
                .with_context(|| format!("Cannot open voiceprint {}", path.display()))?,
        )
        .context("Invalid voiceprint JSON")?;
        ensure!(
            voiceprint.model == "ecapa_tdnn",
            "Voiceprint model must be ecapa_tdnn"
        );
        ensure!(
            voiceprint.dim > 0 && voiceprint.dim == voiceprint.embedding.len(),
            "Invalid voiceprint dimension"
        );
        normalize(&mut voiceprint.embedding)?;
        Ok(voiceprint)
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        serde_json::to_writer(std::fs::File::create(path)?, self)?;
        Ok(())
    }

    pub fn similarity(&self, mut embedding: Vec<f32>) -> anyhow::Result<f32> {
        ensure!(
            embedding.len() == self.dim,
            "Voiceprint and ECAPA embedding dimensions differ; re-enroll your voice"
        );
        normalize(&mut embedding)?;
        // The voiceprint is normalized both on creation and on load.
        Ok(self
            .embedding
            .iter()
            .zip(embedding)
            .map(|(a, b)| a * b)
            .sum::<f32>()
            .clamp(-1.0, 1.0))
    }
}

/// Full 1.5 s windows with a 0.5 s hop. Anchor one final full window at
/// the end when necessary so a partial hop never discards the last word.
pub fn windows(len: usize) -> Vec<Range<usize>> {
    if len < WINDOW {
        return Vec::new();
    }
    let mut ranges: Vec<_> = (0..=len - WINDOW)
        .step_by(HOP)
        .map(|s| s..s + WINDOW)
        .collect();
    if ranges.last().unwrap().end < len {
        ranges.push(len - WINDOW..len);
    }
    ranges
}

/// Union ordered coverage, without duplicating samples in overlapping windows.
fn merge_regions(regions: impl IntoIterator<Item = Range<usize>>) -> Vec<Range<usize>> {
    let mut merged: Vec<Range<usize>> = Vec::new();
    for region in regions {
        if let Some(last) = merged.last_mut().filter(|r| r.end >= region.start) {
            last.end = last.end.max(region.end);
        } else {
            merged.push(region);
        }
    }
    merged
}

/// Remove VAD-negative audio before windowing. No silence is fed to ECAPA
/// between regions. Source offsets are retained separately for the output join.
pub fn speech_audio(samples: &[f32], regions: &[Range<usize>]) -> Vec<f32> {
    regions
        .iter()
        .flat_map(|r| samples[r.clone()].iter().copied())
        .collect()
}

/// Pure gate with injected embeddings, also used by the ONNX-backed gate.
/// Regions must be ordered, disjoint sample ranges from the configured VAD.
pub fn gate_with(
    samples: &[f32],
    regions: &[Range<usize>],
    config: &SpeakerFilterConfig,
    voiceprint: &Voiceprint,
    mut embed: impl FnMut(&[f32]) -> anyhow::Result<Vec<f32>>,
) -> anyhow::Result<Vec<f32>> {
    config.validate()?;
    let speech_len: usize = regions.iter().map(|r| r.len()).sum();
    if (speech_len as f64) < config.min_speech_secs as f64 * SAMPLE_RATE as f64 {
        return Ok(samples.to_vec());
    }
    let speech = speech_audio(samples, regions);
    let mut accepted = Vec::new();
    for window in windows(speech.len()) {
        if voiceprint.similarity(embed(&speech[window.clone()])?)? >= config.threshold {
            accepted.push(window);
        }
    }
    let coverage = merge_regions(accepted);
    // Map coverage on the compact speech timeline back to the original audio.
    let mut kept = Vec::new();
    let mut offset = 0;
    for region in regions {
        for window in &coverage {
            let start = window.start.max(offset);
            let end = window.end.min(offset + region.len());
            if start < end {
                kept.push(region.start + start - offset..region.start + end - offset);
            }
        }
        offset += region.len();
    }
    let mut output = Vec::new();
    for region in merge_regions(kept) {
        if !output.is_empty() {
            output.resize(output.len() + JOIN_SILENCE, 0.0);
        }
        output.extend_from_slice(&samples[region]);
    }
    Ok(output)
}

/// Reuse meeting mode's raw [1, N] f32, mono 16 kHz ECAPA input contract.
/// Waveform normalization is deliberately absent; only embeddings are normalized.
#[cfg(feature = "ml-diarization")]
pub fn load_embedding_model(path: &Path) -> anyhow::Result<crate::meeting::diarization::ml::MlDiarizer> {
    let mut model = crate::meeting::diarization::ml::MlDiarizer::new(
        &crate::meeting::diarization::DiarizationConfig {
            model_path: Some(path.to_string_lossy().into_owned()),
            ..Default::default()
        },
    );
    model.load_model().map_err(anyhow::Error::msg)?;
    Ok(model)
}

pub struct SpeakerFilter {
    config: SpeakerFilterConfig,
    voiceprint: Voiceprint,
    #[cfg(feature = "ml-diarization")]
    model: crate::meeting::diarization::ml::MlDiarizer,
}

impl SpeakerFilter {
    /// Load once at startup. A missing file disables the gate with one warning.
    /// This path never calls the downloader, including on subsequent dictations.
    pub fn load(config: &SpeakerFilterConfig) -> Option<Self> {
        if !config.enabled {
            return None;
        }
        match Self::try_load(config) {
            Ok(filter) => Some(filter),
            Err(e) => {
                tracing::warn!("speaker filter disabled: {:#}", e);
                None
            }
        }
    }

    fn try_load(config: &SpeakerFilterConfig) -> anyhow::Result<Self> {
        config.validate()?;
        #[cfg(feature = "ml-diarization")]
        {
            let voiceprint = Voiceprint::load(Path::new(&config.voiceprint))?;
            let path = crate::setup::model::ecapa_model_path();
            ensure!(
                path.is_file(),
                "ECAPA model missing at {}; run voxtype voiceprint enroll first",
                path.display()
            );
            Ok(Self {
                config: config.clone(),
                voiceprint,
                model: load_embedding_model(&path)?,
            })
        }
        #[cfg(not(feature = "ml-diarization"))]
        anyhow::bail!("requires a build with ml-diarization")
    }

    pub fn filter(&self, samples: &[f32], regions: &[Range<usize>]) -> anyhow::Result<Vec<f32>> {
        let start = std::time::Instant::now();
        let result = gate_with(samples, regions, &self.config, &self.voiceprint, |window| {
            #[cfg(feature = "ml-diarization")]
            {
                self.model
                    .extract_embedding(window)
                    .map_err(anyhow::Error::msg)
            }
            #[cfg(not(feature = "ml-diarization"))]
            {
                let _ = window;
                anyhow::bail!("requires a build with ml-diarization")
            }
        });
        tracing::debug!(
            "speaker filter: gate time {:.3}s",
            start.elapsed().as_secs_f64()
        );
        if matches!(&result, Ok(samples) if samples.is_empty()) {
            tracing::info!("speaker filter: no matching voice");
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voiceprint() -> Voiceprint {
        Voiceprint::from_embeddings(vec![vec![2.0, 0.0]]).unwrap()
    }

    #[test]
    fn window_hop_and_tail_coverage() {
        assert!(windows(WINDOW - 1).is_empty());
        assert_eq!(windows(WINDOW), vec![0..WINDOW]);
        assert_eq!(
            windows(WINDOW + HOP),
            vec![0..WINDOW, HOP..HOP + WINDOW]
        );
        let ranges = windows(WINDOW + HOP + 1);
        assert_eq!(ranges.last(), Some(&(HOP + 1..WINDOW + HOP + 1)));
        assert_eq!(merge_regions(ranges), vec![0..WINDOW + HOP + 1]);
        assert_eq!(
            merge_regions([0..10, 5..20, 20..25, 30..35]),
            vec![0..25, 30..35]
        );
    }

    #[test]
    fn voiceprint_json_round_trip_and_normalized_mean() {
        let print = Voiceprint::from_embeddings(vec![vec![10.0, 0.0], vec![0.0, 2.0]]).unwrap();
        let file = tempfile::NamedTempFile::new().unwrap();
        print.save(file.path()).unwrap();
        let loaded = Voiceprint::load(file.path()).unwrap();
        assert_eq!(loaded.model, "ecapa_tdnn");
        assert_eq!(loaded.dim, 2);
        assert!((loaded.embedding[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        assert!((loaded.similarity(vec![3.0, 3.0]).unwrap() - 1.0).abs() < 1e-6);
        assert!(loaded.similarity(vec![0.0, 0.0]).is_err());
        assert!(loaded.similarity(vec![1.0]).is_err());
        for json in [
            r#"{"model":"wrong","dim":2,"embedding":[1,0]}"#,
            r#"{"model":"ecapa_tdnn","dim":3,"embedding":[1,0]}"#,
            r#"{"model":"ecapa_tdnn","dim":2,"embedding":[0,0]}"#,
        ] {
            std::fs::write(file.path(), json).unwrap();
            assert!(Voiceprint::load(file.path()).is_err());
        }
    }

    #[test]
    fn gate_keep_drop_join_and_short_speech_bypass() {
        let config = SpeakerFilterConfig::default();
        let samples = vec![0.25; SAMPLE_RATE * 5];
        let mut index = 0;
        let output = gate_with(
            &samples,
            &[0..samples.len()],
            &config,
            &voiceprint(),
            |_| {
                let keep = index == 0 || index == 7;
                index += 1;
                Ok(if keep { vec![4.0, 0.0] } else { vec![0.0, 1.0] })
            },
        )
        .unwrap();
        assert_eq!(index, 8);
        assert_eq!(output.len(), 2 * WINDOW + JOIN_SILENCE);
        assert_eq!(&output[..WINDOW], &samples[..WINDOW]);
        assert!(output[WINDOW..WINDOW + JOIN_SILENCE].iter().all(|&x| x == 0.0));
        assert!(output[WINDOW + JOIN_SILENCE..].iter().all(|&x| x == 0.25));
        let dropped = gate_with(
            &samples,
            &[0..samples.len()],
            &config,
            &voiceprint(),
            |_| Ok(vec![0.0, 1.0]),
        )
        .unwrap();
        assert!(dropped.is_empty());
        let kept = gate_with(
            &samples,
            &[0..samples.len()],
            &config,
            &voiceprint(),
            |_| Ok(vec![1.0, 0.0]),
        )
        .unwrap();
        assert_eq!(kept, samples); // overlapping windows do not duplicate audio
        let short = gate_with(
            &samples,
            &[0..WINDOW - 1],
            &config,
            &voiceprint(),
            |_| panic!("short speech must not be embedded"),
        )
        .unwrap();
        assert_eq!(short, samples); // bypass depends on speech, not buffer length
    }

    #[test]
    fn gate_maps_compact_speech_back_to_source_regions() {
        let samples = vec![0.5; SAMPLE_RATE * 4];
        let regions = [0..SAMPLE_RATE, SAMPLE_RATE * 3..SAMPLE_RATE * 4];
        let kept = gate_with(
            &samples,
            &regions,
            &SpeakerFilterConfig::default(),
            &voiceprint(),
            |window| {
                assert_eq!(window.len(), WINDOW);
                Ok(vec![1.0, 0.0])
            },
        )
        .unwrap();
        assert_eq!(kept.len(), SAMPLE_RATE * 2 + JOIN_SILENCE);
        assert!(kept[SAMPLE_RATE..SAMPLE_RATE + JOIN_SILENCE].iter().all(|&s| s == 0.0));
    }
}
