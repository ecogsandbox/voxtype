//! macOS dictation capture with Apple's echo cancellation and noise suppression.
//!
//! The AudioUnit lives on a worker, like cpal's stream. Startup is acknowledged
//! before returning the sample receiver, so every setup failure can use cpal.

use super::{cpal_capture::CpalCapture, resampler::StreamResampler, AudioCapture};
use crate::{config::AudioConfig, error::AudioError};
use coreaudio::audio_unit::{
    audio_format::LinearPcmFlags, macos_helpers, render_callback, AudioUnit, Element, IOType,
    SampleFormat, Scope, StreamFormat,
};
use coreaudio::sys;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};
use std::{thread, time::Duration};
use tokio::sync::{mpsc, oneshot};

// The factory creates a new capture for every recording. Remember a terminal
// native failure across those instances, until the daemon is restarted.
static NATIVE_FAILURE: AtomicI32 = AtomicI32::new(0);

pub struct VoiceProcessingCapture {
    config: AudioConfig,
    fallback: CpalCapture,
    using_fallback: bool,
    startup_error: Option<String>,
    render_error: Arc<AtomicI32>,
    commands: Option<std::sync::mpsc::Sender<Command>>,
    worker: Option<thread::JoinHandle<()>>,
}

enum Command {
    Stop(oneshot::Sender<Vec<f32>>),
    GetSamples(oneshot::Sender<Vec<f32>>),
}

impl VoiceProcessingCapture {
    pub fn new(config: &AudioConfig) -> Result<Self, AudioError> {
        Ok(Self {
            config: config.clone(),
            fallback: CpalCapture::new(config)?,
            using_fallback: false,
            startup_error: None,
            render_error: Arc::new(AtomicI32::new(0)),
            commands: None,
            worker: None,
        })
    }

    fn join_worker(&mut self) {
        self.commands.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[async_trait::async_trait]
impl AudioCapture for VoiceProcessingCapture {
    fn backend_name(&self) -> &'static str {
        if self.using_fallback {
            "cpal"
        } else {
            "VoiceProcessingIO"
        }
    }

    fn has_failed(&self) -> bool {
        !self.using_fallback && self.render_error.load(Ordering::Acquire) != 0
    }

    fn failure_reason(&self) -> Option<String> {
        let status = self.render_error.load(Ordering::Acquire);
        if status != 0 {
            Some(format!("VoiceProcessingIO render failed (OSStatus {status}); using cpal on subsequent recordings until restart"))
        } else {
            self.startup_error.clone()
        }
    }

    async fn start(&mut self) -> Result<mpsc::Receiver<Vec<f32>>, AudioError> {
        if self.commands.is_some() || self.using_fallback {
            return Err(AudioError::StreamError("Capture already started".into()));
        }
        self.render_error.store(0, Ordering::Release);
        self.startup_error = None;
        let previous_failure = NATIVE_FAILURE.load(Ordering::Acquire);
        if previous_failure != 0 {
            self.startup_error = Some(format!(
                "VoiceProcessingIO previously failed (OSStatus {previous_failure}); using cpal until restart"
            ));
        } else {
            let (chunks_tx, chunks_rx) = mpsc::channel(64);
            let (commands_tx, commands_rx) = std::sync::mpsc::channel();
            let (ready_tx, ready_rx) = oneshot::channel();
            let device = self.config.device.clone();
            let render_error = self.render_error.clone();
            self.commands = Some(commands_tx);
            self.worker = Some(thread::spawn(move || {
                capture_worker(device, chunks_tx, commands_rx, ready_tx, render_error);
            }));
            match ready_rx.await {
                Ok(Ok(())) => return Ok(chunks_rx),
                Ok(Err(error)) => self.startup_error = Some(error),
                Err(error) => self.startup_error = Some(error.to_string()),
            }
            self.join_worker();
        }
        tracing::warn!(error = ?self.startup_error, "VoiceProcessingIO unavailable; falling back to cpal");
        let receiver = self.fallback.start().await?;
        self.using_fallback = true;
        Ok(receiver)
    }

    async fn stop(&mut self) -> Result<Vec<f32>, AudioError> {
        if self.using_fallback {
            self.using_fallback = false;
            return self.fallback.stop().await;
        }
        let mut samples = Vec::new();
        if let Some(commands) = self.commands.take() {
            let (tx, rx) = oneshot::channel();
            if commands.send(Command::Stop(tx)).is_ok() {
                samples = rx.await.map_err(|e| AudioError::StreamError(e.to_string()))?;
            }
        }
        self.join_worker();
        if self.has_failed() {
            // A user stop can race the daemon's failure poll.
            super::publish_capture_status(self);
        }
        if samples.is_empty() {
            Err(AudioError::EmptyRecording)
        } else {
            Ok(samples)
        }
    }

    async fn get_samples(&mut self) -> Vec<f32> {
        if self.using_fallback {
            return self.fallback.get_samples().await;
        }
        if let Some(commands) = &self.commands {
            let (tx, rx) = oneshot::channel();
            if commands.send(Command::GetSamples(tx)).is_ok() {
                return rx.await.unwrap_or_default();
            }
        }
        Vec::new()
    }
}

impl Drop for VoiceProcessingCapture {
    fn drop(&mut self) {
        self.join_worker();
    }
}

/// Match cpal's selection precedence, while retaining the CoreAudio device ID.
fn match_device(devices: &[(u32, String)], requested: &str) -> Option<u32> {
    let lower = requested.to_lowercase();
    devices
        .iter()
        .find(|(_, name)| name == requested)
        .or_else(|| devices.iter().find(|(_, name)| name.to_lowercase() == lower))
        .or_else(|| {
            devices
                .iter()
                .find(|(_, name)| name.to_lowercase().contains(&lower))
        })
        .map(|(id, _)| *id)
}

fn input_device(requested: &str) -> Result<u32, String> {
    if requested == "default" {
        return macos_helpers::get_default_device_id(true)
            .filter(|id| *id != 0)
            .ok_or_else(|| "No default input device".into());
    }
    let mut devices = Vec::new();
    for id in macos_helpers::get_audio_device_ids().map_err(|e| e.to_string())? {
        if macos_helpers::get_audio_device_supports_scope(id, Scope::Input)
            .map_err(|e| e.to_string())?
        {
            if let Ok(name) = macos_helpers::get_device_name(id) {
                devices.push((id, name));
            }
        }
    }
    match_device(&devices, requested).ok_or_else(|| format!("Input device not found: {requested}"))
}

fn client_format(rate: f64) -> Result<StreamFormat, String> {
    // rubato takes integral Hz; never silently reinterpret an invalid rate.
    if !rate.is_finite() || rate < 1.0 || rate > u32::MAX as f64 || rate.fract() != 0.0 {
        return Err(format!("Invalid hardware sample rate: {rate}"));
    }
    Ok(StreamFormat {
        sample_rate: rate,
        sample_format: SampleFormat::F32,
        flags: LinearPcmFlags::IS_FLOAT | LinearPcmFlags::IS_PACKED,
        channels: 1,
    })
}

struct CaptureBuffer {
    samples: Vec<f32>,
    resampler: StreamResampler,
}

/// coreaudio-rs returns AudioUnitRender errors before invoking our typed input
/// closure. Wrap its installed callback so those errors reach the worker too.
struct CheckedCallback {
    inner: sys::AURenderCallbackStruct,
    error: Arc<AtomicI32>,
}

unsafe extern "C" fn checked_input(
    context: *mut std::ffi::c_void,
    flags: *mut sys::AudioUnitRenderActionFlags,
    timestamp: *const sys::AudioTimeStamp,
    bus: u32,
    frames: u32,
    data: *mut sys::AudioBufferList,
) -> sys::OSStatus {
    // SAFETY: the boxed context stays alive until after AudioUnit disposal.
    // CoreAudio supplies the remaining arguments to its own input callback.
    let context = unsafe { &*(context as *const CheckedCallback) };
    let previous = context.error.load(Ordering::Acquire);
    if previous != 0 {
        return previous;
    }
    let status = unsafe {
        context.inner.inputProc.unwrap()(
            context.inner.inputProcRefCon,
            flags,
            timestamp,
            bus,
            frames,
            data,
        )
    };
    if status != 0 {
        context.error.store(status, Ordering::Release);
    }
    status
}

struct Session {
    // Drop order matters: dispose the unit before freeing its callback context.
    unit: AudioUnit,
    _callback: Box<CheckedCallback>,
    buffer: Arc<Mutex<CaptureBuffer>>,
}

fn start_unit(
    device: u32,
    output_enabled: bool,
    chunks: mpsc::Sender<Vec<f32>>,
    error: Arc<AtomicI32>,
) -> Result<Session, String> {
    let output_device = if output_enabled {
        Some(
            macos_helpers::get_default_device_id(false)
                .filter(|id| *id != 0)
                .ok_or_else(|| "No default output device for voice processing".to_string())?,
        )
    } else {
        None
    };
    // Keep callback storage alive on all error paths, including start failure.
    let mut callback = Box::new(CheckedCallback {
        inner: sys::AURenderCallbackStruct {
            inputProc: None,
            inputProcRefCon: std::ptr::null_mut(),
        },
        error,
    });
    let mut unit = AudioUnit::new(IOType::VoiceProcessingIO).map_err(|e| e.to_string())?;
    let configure = |unit: &mut AudioUnit| -> Result<StreamFormat, coreaudio::Error> {
        // AudioUnit::new initializes immediately. Reconfigure uninitialized.
        unit.uninitialize()?;
        unit.set_property(
            sys::kAudioOutputUnitProperty_EnableIO,
            Scope::Input,
            Element::Input,
            Some(&1u32),
        )?;
        unit.set_property(
            sys::kAudioOutputUnitProperty_EnableIO,
            Scope::Output,
            Element::Output,
            Some(&u32::from(output_enabled)),
        )?;
        unit.set_property(
            sys::kAudioOutputUnitProperty_CurrentDevice,
            Scope::Global,
            Element::Input,
            Some(&device),
        )?;
        if let Some(output_device) = output_device {
            unit.set_property(
                sys::kAudioOutputUnitProperty_CurrentDevice,
                Scope::Global,
                Element::Output,
                Some(&output_device),
            )?;
        }
        let hardware = unit.get_property(
            sys::kAudioUnitProperty_StreamFormat,
            Scope::Input,
            Element::Input,
        )?;
        StreamFormat::from_asbd(hardware)
    };
    let hardware = configure(&mut unit).map_err(|e| e.to_string())?;
    let format = client_format(hardware.sample_rate)?;
    unit.set_property(
        sys::kAudioUnitProperty_StreamFormat,
        Scope::Output,
        Element::Input,
        Some(&format.to_asbd()),
    )
    .map_err(|e| e.to_string())?;
    if output_enabled {
        unit.set_property(
            sys::kAudioUnitProperty_StreamFormat,
            Scope::Input,
            Element::Output,
            Some(&format.to_asbd()),
        )
        .map_err(|e| e.to_string())?;
        unit.set_render_callback(
            |args: render_callback::Args<render_callback::data::Interleaved<f32>>| {
                args.data.buffer.fill(0.0);
                Ok(())
            },
        )
        .map_err(|e| e.to_string())?;
    }
    let buffer = Arc::new(Mutex::new(CaptureBuffer {
        samples: Vec::new(),
        resampler: StreamResampler::new(format.sample_rate as u32, 16_000)?,
    }));
    let callback_buffer = buffer.clone();
    unit.set_input_callback(
        move |args: render_callback::Args<render_callback::data::Interleaved<f32>>| {
            let mut buffer = callback_buffer.lock().map_err(|_| ())?;
            let converted = buffer.resampler.push(args.data.buffer);
            buffer.samples.extend_from_slice(&converted);
            let _ = chunks.try_send(converted);
            Ok(())
        },
    )
    .map_err(|e| e.to_string())?;
    callback.inner = unit
        .get_property(
            sys::kAudioOutputUnitProperty_SetInputCallback,
            Scope::Global,
            Element::Output,
        )
        .map_err(|e| e.to_string())?;
    if callback.inner.inputProc.is_none() {
        return Err("Audio unit has no input callback".into());
    }
    let checked = sys::AURenderCallbackStruct {
        inputProc: Some(checked_input),
        inputProcRefCon: (&mut *callback as *mut CheckedCallback).cast(),
    };
    unit.set_property(
        sys::kAudioOutputUnitProperty_SetInputCallback,
        Scope::Global,
        Element::Output,
        Some(&checked),
    )
    .map_err(|e| e.to_string())?;
    unit.initialize().map_err(|e| e.to_string())?;
    // Confirm the format after initialization instead of assuming a default.
    let actual: sys::AudioStreamBasicDescription = unit
        .get_property(
            sys::kAudioUnitProperty_StreamFormat,
            Scope::Output,
            Element::Input,
        )
        .map_err(|e| e.to_string())?;
    let expected = format.to_asbd();
    if actual.mSampleRate != expected.mSampleRate
        || actual.mChannelsPerFrame != 1
        || actual.mFormatID != expected.mFormatID
        || actual.mFormatFlags != expected.mFormatFlags
        || actual.mBytesPerFrame != 4
        || actual.mBitsPerChannel != 32
    {
        return Err("VoiceProcessingIO did not accept mono f32 client format".into());
    }
    unit.start().map_err(|e| e.to_string())?;
    tracing::info!(
        device,
        sample_rate = format.sample_rate,
        output_enabled,
        "VoiceProcessingIO started: mono f32, resampled to 16000 Hz"
    );
    Ok(Session {
        unit,
        _callback: callback,
        buffer,
    })
}

fn capture_worker(
    device: String,
    chunks: mpsc::Sender<Vec<f32>>,
    commands: std::sync::mpsc::Receiver<Command>,
    ready: oneshot::Sender<Result<(), String>>,
    error: Arc<AtomicI32>,
) {
    let started = input_device(&device).and_then(|id| {
        start_unit(id, false, chunks.clone(), error.clone()).or_else(|first| {
            tracing::debug!(%first, "Retrying VoiceProcessingIO with silent output");
            error.store(0, Ordering::Release);
            start_unit(id, true, chunks, error.clone())
                .map_err(|second| format!("input-only: {first}; silent output: {second}"))
        })
    });
    let session = match started {
        Ok(session) => session,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    let buffer = session.buffer.clone();
    let mut session = Some(session);
    if ready.send(Ok(())).is_err() {
        return;
    }
    loop {
        if error.load(Ordering::Acquire) != 0 && session.is_some() {
            NATIVE_FAILURE.store(error.load(Ordering::Acquire), Ordering::Release);
            tracing::warn!(
                status = error.load(Ordering::Acquire),
                "VoiceProcessingIO render failed; preserving audio and using cpal until restart"
            );
            stop_session(&mut session, &buffer);
        }
        match commands.recv_timeout(Duration::from_millis(20)) {
            Ok(Command::Stop(reply)) => {
                stop_session(&mut session, &buffer);
                let _ = reply.send(std::mem::take(&mut buffer.lock().unwrap().samples));
                break;
            }
            Ok(Command::GetSamples(reply)) => {
                let _ = reply.send(std::mem::take(&mut buffer.lock().unwrap().samples));
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    // Dispose before checking once more: a callback can fail while Stop or
    // channel closure is being handled, after the loop's initial check.
    stop_session(&mut session, &buffer);
    let status = error.load(Ordering::Acquire);
    if status != 0 {
        NATIVE_FAILURE.store(status, Ordering::Release);
        tracing::warn!(
            status,
            "VoiceProcessingIO recording ended early; using cpal until restart"
        );
    }
}

fn stop_session(session: &mut Option<Session>, buffer: &Mutex<CaptureBuffer>) {
    if let Some(mut session) = session.take() {
        if let Err(error) = session.unit.stop() {
            tracing::warn!(%error, "Could not stop VoiceProcessingIO cleanly");
        }
        drop(session);
        let mut buffer = buffer.lock().unwrap();
        let tail = buffer.resampler.flush();
        buffer.samples.extend(tail);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_failure_is_observable_and_stops_forwarding() {
        unsafe extern "C" fn render(
            context: *mut std::ffi::c_void,
            _: *mut sys::AudioUnitRenderActionFlags,
            _: *const sys::AudioTimeStamp,
            _: u32,
            _: u32,
            _: *mut sys::AudioBufferList,
        ) -> sys::OSStatus {
            // This fake callback never touches audio hardware or data pointers.
            let calls = unsafe { &mut *(context as *mut usize) };
            *calls += 1;
            if *calls == 1 {
                0
            } else {
                -10863
            }
        }

        let error = Arc::new(AtomicI32::new(0));
        let mut calls = 0usize;
        let mut callback = CheckedCallback {
            inner: sys::AURenderCallbackStruct {
                inputProc: Some(render),
                inputProcRefCon: (&mut calls as *mut usize).cast(),
            },
            error: error.clone(),
        };
        for expected in [0, -10863, -10863] {
            // The fake render ignores these arguments. Only the valid context
            // pointer is dereferenced by the trampoline and fake callback.
            let status = unsafe {
                checked_input(
                    (&mut callback as *mut CheckedCallback).cast(),
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    1,
                    160,
                    std::ptr::null_mut(),
                )
            };
            assert_eq!(status, expected);
            assert_eq!(error.load(Ordering::Acquire), expected);
        }
        assert_eq!(calls, 2);
    }

    #[test]
    fn device_matching_preserves_cpal_precedence() {
        let devices = vec![
            (11, "USB Microphone".into()),
            (22, "Microphone".into()),
            (33, "MICROPHONE".into()),
        ];
        assert_eq!(match_device(&devices, "MICROPHONE"), Some(33));
        assert_eq!(match_device(&devices, "microphone"), Some(22));
        assert_eq!(match_device(&devices, "usb"), Some(11));
        assert_eq!(match_device(&devices, "absent"), None);
        assert_eq!(match_device(&[], "default"), None);
    }

    #[test]
    fn format_is_mono_float_at_hardware_rate() {
        for rate in [16_000.0, 44_100.0, 48_000.0] {
            let asbd = client_format(rate).unwrap().to_asbd();
            assert_eq!(asbd.mSampleRate, rate);
            assert_eq!(asbd.mFormatID, sys::kAudioFormatLinearPCM);
            assert_eq!(asbd.mChannelsPerFrame, 1);
            assert_eq!(asbd.mBitsPerChannel, 32);
            assert_eq!(asbd.mBytesPerFrame, 4);
            assert_eq!(asbd.mFramesPerPacket, 1);
            assert_ne!(asbd.mFormatFlags & sys::kAudioFormatFlagIsFloat, 0);
        }
        for rate in [0.0, -1.0, f64::NAN, f64::INFINITY, 44_100.5] {
            assert!(client_format(rate).is_err());
        }
    }

    #[test]
    fn hardware_rate_conversion_flushes_short_recordings() {
        for rate in [16_000, 44_100, 48_000] {
            let mut resampler = StreamResampler::new(rate, 16_000).unwrap();
            let mut samples = resampler.push(&vec![0.25; (rate / 100) as usize]);
            samples.extend(resampler.flush());
            assert_eq!(samples.len(), 160);
            assert!(samples.iter().all(|sample| sample.is_finite()));
        }
    }
}
