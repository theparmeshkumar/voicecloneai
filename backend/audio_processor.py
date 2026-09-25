import os
import numpy as np
import soundfile as sf
import librosa
from scipy import signal
from pydub import AudioSegment
from faster_whisper import WhisperModel

# Ensure MKL / OpenMP compatibility
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"

# Ensure ffmpeg from Winget is in PATH if available
FFMPEG_EXE = r"C:\Users\pksar\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe"
FFMPEG_DIR = os.path.dirname(FFMPEG_EXE)
if os.path.exists(FFMPEG_DIR) and FFMPEG_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

if os.path.exists(FFMPEG_EXE):
    AudioSegment.converter = FFMPEG_EXE

# Lazy-loaded Whisper model instance
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        try:
            # Multi-lingual tiny model for accurate language and accent detection
            _whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=2)
        except Exception:
            _whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8", cpu_threads=2)
    return _whisper_model

def load_and_normalize_audio(file_path: str, target_sr: int = 24000) -> tuple[np.ndarray, int]:
    """
    Loads any audio file (WAV, MP3, M4A, OGG, WebM, FLAC), converts to mono,
    resamples to target_sr, trims long silence, and normalizes amplitude.
    """
    try:
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    except Exception:
        seg = AudioSegment.from_file(file_path)
        seg = seg.set_channels(1).set_frame_rate(target_sr)
        samples = np.array(seg.get_array_of_samples()).astype(np.float32)
        max_val = float(1 << (8 * seg.sample_width - 1))
        y = samples / max_val
        sr = target_sr

    trimmed, _ = librosa.effects.trim(y, top_db=25, frame_length=1024, hop_length=256)
    if len(trimmed) > target_sr * 0.5:
        y = trimmed

    peak = np.max(np.abs(y))
    if peak > 1e-5:
        y = y / peak * 0.95

    return y, sr

def compute_spectral_envelope(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    Extracts a smoothed spectral envelope representing the speaker's vocal tract transfer function.
    """
    if len(y) < n_fft:
        return np.ones(n_fft // 2 + 1, dtype=np.float32)
    
    # Compute magnitude spectrogram
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    # Frame energy weighting to focus on voiced speech frames
    frame_energy = np.sum(D, axis=0)
    thresh = np.percentile(frame_energy, 40)
    voiced_frames = D[:, frame_energy >= thresh]
    if voiced_frames.shape[1] > 0:
        mean_spec = np.mean(voiced_frames, axis=1) + 1e-7
    else:
        mean_spec = np.mean(D, axis=1) + 1e-7

    # Smooth the spectral curve (kernel size ~ 35 bins) to capture formant peaks without harmonic spikes
    kernel_size = 35
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(mean_spec, kernel, mode="same")
    envelope = np.maximum(smoothed, 1e-6)
    return envelope.astype(np.float32)

def extract_acoustic_profile(y: np.ndarray, sr: int) -> dict:
    """
    Performs deep acoustic feature extraction on the reference voice sample.
    Extracts fundamental pitch (F0), vocal register, formants, timbre metrics,
    spectral envelope, and studio radar attributes.
    """
    duration = float(len(y) / sr)

    # 1. Fundamental Frequency (F0) & Pitch Statistics
    # Use 20 seconds slice for robust pitch estimation
    slice_len = min(len(y), int(sr * 25))
    y_slice = y[:slice_len]

    try:
        f0, voiced_flag, _ = librosa.pyin(y_slice, fmin=65, fmax=400, sr=sr)
        valid_f0 = f0[voiced_flag & ~np.isnan(f0)]
    except Exception:
        valid_f0 = np.array([])

    if len(valid_f0) < 15:
        # Fallback to yin if pyin is sparse
        y_8k = librosa.resample(y_slice, orig_sr=sr, target_sr=8000)
        f0_yin = librosa.yin(y_8k, fmin=65, fmax=400, sr=8000, frame_length=1024, hop_length=256)
        rms = librosa.feature.rms(y=y_8k, frame_length=1024, hop_length=256)[0]
        voiced_mask = (rms > np.median(rms) * 0.3) & (f0_yin > 70) & (f0_yin < 380)
        valid_f0 = f0_yin[voiced_mask]

    if len(valid_f0) > 10:
        mean_f0 = float(np.mean(valid_f0))
        median_f0 = float(np.median(valid_f0))
        min_f0 = float(np.percentile(valid_f0, 5))
        max_f0 = float(np.percentile(valid_f0, 95))
        std_f0 = float(np.std(valid_f0))
    else:
        mean_f0 = 145.0
        median_f0 = 145.0
        min_f0 = 95.0
        max_f0 = 210.0
        std_f0 = 25.0

    # Vocal Register Classification
    if median_f0 < 115:
        vocal_register = "Bass"
        inferred_gender = "Male (Deep)"
    elif median_f0 < 150:
        vocal_register = "Baritone"
        inferred_gender = "Male"
    elif median_f0 < 185:
        vocal_register = "Tenor"
        inferred_gender = "Male / Neutral"
    elif median_f0 < 225:
        vocal_register = "Contralto / Alto"
        inferred_gender = "Female / Warm"
    elif median_f0 < 280:
        vocal_register = "Mezzo-Soprano"
        inferred_gender = "Female"
    else:
        vocal_register = "Soprano"
        inferred_gender = "Female (Bright)"

    # 2. Timbre & Spectral Characteristics
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = float(np.mean(spectral_centroid))

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    mean_rolloff = float(np.mean(spectral_rolloff))

    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
    mean_flatness = float(np.mean(spectral_flatness))

    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    mean_contrast = float(np.mean(spectral_contrast))

    # 3. Formant Estimation via Linear Predictive Coding (LPC)
    y_16k = librosa.resample(y[:min(len(y), sr*15)], orig_sr=sr, target_sr=16000)
    y_pre = np.append(y_16k[0], y_16k[1:] - 0.97 * y_16k[:-1])
    try:
        a = librosa.lpc(y_pre, order=18)
        roots = [r for r in np.roots(a) if np.imag(r) >= 0]
        angles = np.arctan2(np.imag(roots), np.real(roots))
        formants = sorted(angles * (16000.0 / (2.0 * np.pi)))
        valid_formants = [f for f in formants if 200 < f < 4000]
        f1 = float(valid_formants[0]) if len(valid_formants) > 0 else 520.0
        f2 = float(valid_formants[1]) if len(valid_formants) > 1 else 1550.0
        f3 = float(valid_formants[2]) if len(valid_formants) > 2 else 2600.0
    except Exception:
        f1, f2, f3 = 520.0, 1550.0, 2600.0

    # 4. Rhythm, Pace & Syllable Tempo
    onset_env = librosa.onset.onset_strength(y=y[:min(len(y), sr*30)], sr=sr)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
    speech_tempo = float(tempo[0]) if len(tempo) > 0 else 120.0

    # 5. Studio Radar Attributes (Normalized 0 to 100)
    radar_pitch = int(np.clip((median_f0 - 70) / (320 - 70) * 100, 5, 95))
    radar_brightness = int(np.clip((mean_centroid - 800) / (3500 - 800) * 100, 10, 95))
    radar_warmth = int(np.clip(100 - (radar_brightness * 0.7 + (f1 / 1000.0) * 30), 10, 95))
    radar_resonance = int(np.clip((mean_contrast / 35.0) * 80 + 20, 15, 95))
    radar_articulation = int(np.clip((mean_rolloff / 6000.0) * 70 + (speech_tempo / 150.0) * 30, 20, 95))
    radar_stability = int(np.clip(100 - (std_f0 / max(mean_f0, 1.0)) * 150, 15, 95))

    # Descriptive Tone Signature
    descriptors = []
    if radar_warmth > 60:
        descriptors.append("Warm")
    elif radar_brightness > 65:
        descriptors.append("Crisp & Bright")
    else:
        descriptors.append("Balanced")

    if radar_resonance > 60:
        descriptors.append("Resonant")
    if radar_pitch < 35:
        descriptors.append("Deep")
    elif radar_pitch > 65:
        descriptors.append("High-Register")

    tone_description = " • ".join(descriptors) if descriptors else "Natural & Expressive"

    return {
        "duration_seconds": round(duration, 2),
        "sample_rate": sr,
        "pitch": {
            "mean_f0_hz": round(mean_f0, 1),
            "median_f0_hz": round(median_f0, 1),
            "min_f0_hz": round(min_f0, 1),
            "max_f0_hz": round(max_f0, 1),
            "std_f0_hz": round(std_f0, 1),
            "vocal_register": vocal_register,
            "inferred_gender": inferred_gender,
        },
        "formants": {
            "f1_hz": round(f1, 1),
            "f2_hz": round(f2, 1),
            "f3_hz": round(f3, 1),
        },
        "timbre": {
            "spectral_centroid_hz": round(mean_centroid, 1),
            "spectral_rolloff_hz": round(mean_rolloff, 1),
            "spectral_flatness": round(mean_flatness, 4),
            "tone_description": tone_description,
        },
        "pacing": {
            "estimated_tempo_bpm": round(speech_tempo, 1),
            "words_per_minute": int(speech_tempo * 1.1),
        },
        "radar": {
            "Pitch": radar_pitch,
            "Brightness": radar_brightness,
            "Warmth": radar_warmth,
            "Resonance": radar_resonance,
            "Articulation": radar_articulation,
            "Stability": radar_stability,
        }
    }

def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribes audio into text with timestamps and detected language using Faster-Whisper.
    """
    model = get_whisper_model()
    segments, info = model.transcribe(audio_path, beam_size=1)
    transcript_segments = []
    full_text = []
    for s in segments:
        full_text.append(s.text.strip())
        transcript_segments.append({
            "start": round(s.start, 2),
            "end": round(s.end, 2),
            "text": s.text.strip()
        })
    return {
        "text": " ".join(full_text).strip(),
        "language": info.language,
        "language_probability": round(info.language_probability, 2),
        "segments": transcript_segments
    }

def apply_timbre_morph(
    synth_audio: np.ndarray,
    ref_profile: dict,
    ref_env: np.ndarray | None = None,
    sr: int = 24000,
    mix: float = 0.65,
    pitch_shift_semitones: float = 0.0
) -> np.ndarray:
    """
    Applies deep acoustic vocal tract cross-synthesis and pitch retargeting:
    1. Pitch Retargeting: Shifts the fundamental frequency towards the reference speaker's median pitch.
    2. Spectral Envelope Transfer: Shapes the synthesized voice with the reference speaker's exact vocal tract formant envelope.
    3. Formant Resonance Tuning: Enhances natural warmth and vocal tract color.
    """
    if len(synth_audio) < 256:
        return synth_audio

    filtered = synth_audio.copy()
    target_pitch = ref_profile.get("pitch", {}).get("median_f0_hz", 145.0)

    # 1. Pitch Retargeting
    # Calculate difference between synth pitch and target pitch
    try:
        # Estimate median pitch of synth audio
        synth_slice = filtered[:min(len(filtered), sr * 10)]
        f0_synth, v_flag, _ = librosa.pyin(synth_slice, fmin=65, fmax=400, sr=sr)
        valid_synth_f0 = f0_synth[v_flag & ~np.isnan(f0_synth)]
        
        auto_shift = 0.0
        if len(valid_synth_f0) > 8 and target_pitch > 50:
            synth_median = float(np.median(valid_synth_f0))
            if synth_median > 50:
                auto_shift = 12.0 * np.log2(target_pitch / synth_median)
                # Keep within reasonable retargeting range (-7 to +7 semitones)
                auto_shift = float(np.clip(auto_shift, -7.0, 7.0))
        
        total_shift = auto_shift + pitch_shift_semitones
        if abs(total_shift) >= 0.35:
            filtered = librosa.effects.pitch_shift(filtered, sr=sr, n_steps=total_shift)
    except Exception as e:
        print(f"[AcousticMorpher] Pitch shift notice: {e}")

    # 2. Spectral Envelope Cross-Synthesis (Vocal Tract Resonance Matching)
    if ref_env is not None and len(ref_env) > 10 and mix > 0.05:
        try:
            current_env = compute_spectral_envelope(filtered, sr)
            if len(current_env) == len(ref_env):
                gain = ref_env / np.maximum(current_env, 1e-6)
                # Limit dynamic range to prevent whistling or harsh cuts
                gain = np.clip(gain, 0.25, 4.0)
                gain = gain / np.mean(gain)
                
                # Blend with user mix factor
                applied_gain = (1.0 - mix) + mix * gain

                # Frequency-domain STFT shaping
                D = librosa.stft(filtered, n_fft=2048, hop_length=512)
                D_morphed = D * applied_gain[:, np.newaxis]
                filtered = librosa.istft(D_morphed, hop_length=512, length=len(filtered))
        except Exception as e:
            print(f"[AcousticMorpher] Spectral envelope notice: {e}")

    # 3. Parametric EQ Formant Polish
    target_f1 = ref_profile.get("formants", {}).get("f1_hz", 520.0)
    target_f2 = ref_profile.get("formants", {}).get("f2_hz", 1550.0)
    warmth = ref_profile.get("radar", {}).get("Warmth", 50)
    brightness = ref_profile.get("radar", {}).get("Brightness", 50)

    try:
        nyquist = sr / 2.0
        # Warmth low shelf
        low_gain_db = (warmth - 50.0) * 0.06
        if abs(low_gain_db) > 0.5:
            low_freq = np.clip(target_pitch * 1.1, 75.0, 260.0)
            gain_lin = 10.0 ** (low_gain_db / 20.0)
            b, a = signal.iirfilter(2, low_freq / nyquist, btype="lowpass", ftype="butter")
            low_comp = signal.lfilter(b, a, filtered)
            filtered = filtered + (gain_lin - 1.0) * low_comp * (mix * 0.5)

        # High brightness shelf
        high_gain_db = (brightness - 50.0) * 0.06
        if abs(high_gain_db) > 0.5:
            high_freq = np.clip(target_f2 * 1.4, 2000.0, 7500.0)
            gain_lin = 10.0 ** (high_gain_db / 20.0)
            b, a = signal.iirfilter(2, high_freq / nyquist, btype="highpass", ftype="butter")
            high_comp = signal.lfilter(b, a, filtered)
            filtered = filtered + (gain_lin - 1.0) * high_comp * (mix * 0.5)
    except Exception:
        pass

    # Normalize output amplitude
    peak = np.max(np.abs(filtered))
    if peak > 1e-5:
        filtered = filtered / peak * 0.95

    return filtered.astype(np.float32)
