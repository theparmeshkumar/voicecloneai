import os
import time
import uuid
import asyncio
import numpy as np
import soundfile as sf
import librosa
from kokoro_onnx import Kokoro
import edge_tts
import requests

from backend.audio_processor import (
    load_and_normalize_audio,
    extract_acoustic_profile,
    compute_spectral_envelope,
    apply_timbre_morph,
    transcribe_audio
)
from backend.storage import get_settings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "kokoro-v1.0.onnx")
VOICES_PATH = os.path.join(MODELS_DIR, "voices-v1.0.bin")
PROFILES_DIR = os.path.join(BASE_DIR, "data", "profiles")
OUTPUTS_DIR = os.path.join(BASE_DIR, "data", "outputs")

os.makedirs(PROFILES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Accent & Language Voice Mapping Table
VOICE_MAP = {
    "Indian English / Hindi": {
        "Male": {
            "kokoro_anchors": ["hm_omega", "hm_psi", "am_adam"],
            "kokoro_lang": "en-us",
            "edge_voice": "en-IN-PrabhatNeural",
            "edge_fallback": "hi-IN-MadhurNeural"
        },
        "Female": {
            "kokoro_anchors": ["hf_alpha", "hf_beta", "af_heart"],
            "kokoro_lang": "en-us",
            "edge_voice": "en-IN-NeerjaNeural",
            "edge_fallback": "hi-IN-SwaraNeural"
        }
    },
    "American English": {
        "Male": {
            "kokoro_anchors": ["am_adam", "am_michael", "am_eric", "am_liam"],
            "kokoro_lang": "en-us",
            "edge_voice": "en-US-GuyNeural",
            "edge_fallback": "en-US-ChristopherNeural"
        },
        "Female": {
            "kokoro_anchors": ["af_heart", "af_bella", "af_nicole", "af_sarah"],
            "kokoro_lang": "en-us",
            "edge_voice": "en-US-JennyNeural",
            "edge_fallback": "en-US-AvaNeural"
        }
    },
    "British English": {
        "Male": {
            "kokoro_anchors": ["bm_daniel", "bm_george", "bm_lewis", "bm_fable"],
            "kokoro_lang": "en-gb",
            "edge_voice": "en-GB-RyanNeural",
            "edge_fallback": "en-GB-ThomasNeural"
        },
        "Female": {
            "kokoro_anchors": ["bf_emma", "bf_alice", "bf_isabella", "bf_lily"],
            "kokoro_lang": "en-gb",
            "edge_voice": "en-GB-SoniaNeural",
            "edge_fallback": "en-GB-MaisieNeural"
        }
    },
    "Global English / Neutral": {
        "Male": {
            "kokoro_anchors": ["am_adam", "bm_daniel", "am_echo"],
            "kokoro_lang": "en-us",
            "edge_voice": "en-US-GuyNeural",
            "edge_fallback": "en-GB-RyanNeural"
        },
        "Female": {
            "kokoro_anchors": ["af_heart", "bf_emma", "af_sky"],
            "kokoro_lang": "en-us",
            "edge_voice": "en-US-JennyNeural",
            "edge_fallback": "en-GB-SoniaNeural"
        }
    }
}

class VoiceClonerEngine:
    def __init__(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VOICES_PATH):
            print("[VoiceCloner] Kokoro model files missing. Auto-downloading...")
            try:
                from download_models import main as download_all_models
                download_all_models()
            except Exception as e:
                print(f"[VoiceCloner] Warning: Could not auto-download model files: {e}")

        print(f"[VoiceCloner] Initializing Kokoro ONNX model from {MODEL_PATH}...")
        self.kokoro = Kokoro(MODEL_PATH, VOICES_PATH)
        self.available_voices = self.kokoro.get_voices()
        print(f"[VoiceCloner] Loaded {len(self.available_voices)} Kokoro base voice styles.")

    def detect_accent_and_language(self, audio_path: str) -> tuple[str, str]:
        """
        Transcribes a slice of audio to detect the primary language and accent.
        Returns: (accent_label, lang_code)
        """
        try:
            res = transcribe_audio(audio_path)
            lang = res.get("language", "en")
            # If language is Hindi or contains Indic markers
            if lang in ["hi", "mr", "ta", "te", "bn", "gu", "kn", "pa"]:
                return "Indian English / Hindi", lang
            elif lang == "en":
                # Check text for typical Indian English patterns or keywords if any
                text = res.get("text", "").lower()
                indian_keywords = ["sir", "dear", "delighted", "career", "institute", "student", "shree", "ji"]
                score = sum(1 for kw in indian_keywords if kw in text)
                if score >= 2:
                    return "Indian English / Hindi", "en"
                return "American English", "en"
            else:
                return "Global English / Neutral", lang
        except Exception:
            return "American English", "en"

    def analyze_and_create_clone(
        self,
        audio_file_path: str,
        clone_name: str,
        description: str = "",
        gender_override: str | None = None,
        accent_override: str | None = None
    ) -> dict:
        """
        Analyzes reference voice sample:
        1. Normalizes and trims silence.
        2. Performs deep acoustic feature extraction (robust F0, formants, timbre).
        3. Computes and saves the speaker's vocal tract Smoothed Spectral Envelope.
        4. Detects language/accent (or uses override).
        5. Computes tailored composite style vector.
        6. Saves profile, audio, and envelope to disk.
        """
        clone_id = str(uuid.uuid4())[:8]
        if not clone_name.strip():
            clone_name = f"Cloned Voice #{clone_id}"

        # 1. Load and normalize audio (target 24kHz)
        y, sr = load_and_normalize_audio(audio_file_path, target_sr=24000)

        # Save standardized reference WAV
        ref_wav_filename = f"{clone_id}_ref.wav"
        ref_wav_path = os.path.join(PROFILES_DIR, ref_wav_filename)
        sf.write(ref_wav_path, y, sr)

        # 2. Extract acoustic profile
        profile = extract_acoustic_profile(y, sr)
        median_pitch = profile["pitch"]["median_f0_hz"]

        # 3. Detect language & accent
        detected_accent, detected_lang = self.detect_accent_and_language(ref_wav_path)
        active_accent = accent_override if accent_override else detected_accent

        # 4. Gender determination
        if gender_override in ["Male", "Female"]:
            active_gender = gender_override
        else:
            active_gender = "Female" if median_pitch >= 170.0 else "Male"

        # 5. Extract and save the reference Smoothed Spectral Envelope
        env = compute_spectral_envelope(y[:min(len(y), sr * 30)], sr)
        env_filename = f"{clone_id}_env.npy"
        env_path = os.path.join(PROFILES_DIR, env_filename)
        np.save(env_path, env)

        # 6. Retrieve voice mapping configuration
        accent_cfg = VOICE_MAP.get(active_accent, VOICE_MAP["American English"])
        gender_cfg = accent_cfg.get(active_gender, accent_cfg["Male"])
        candidate_anchors = gender_cfg["kokoro_anchors"]
        valid_anchors = [v for v in candidate_anchors if v in self.available_voices]
        if not valid_anchors:
            valid_anchors = ["af_heart"] if active_gender == "Female" else ["am_adam"]

        # Composite Kokoro style vector
        style_vectors = [self.kokoro.get_voice_style(v) for v in valid_anchors]
        weights = np.ones(len(valid_anchors)) / len(valid_anchors)
        cloned_style = np.zeros_like(style_vectors[0])
        for w, vec in zip(weights, style_vectors):
            cloned_style += w * vec

        style_filename = f"{clone_id}_style.npy"
        style_path = os.path.join(PROFILES_DIR, style_filename)
        np.save(style_path, cloned_style)

        # Build complete clone profile
        clone_data = {
            "id": clone_id,
            "name": clone_name,
            "description": description or f"{active_accent} • {active_gender} • {profile['timbre']['tone_description']}",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ref_audio_file": ref_wav_filename,
            "style_file": style_filename,
            "env_file": env_filename,
            "sample_duration": profile["duration_seconds"],
            "gender": active_gender,
            "accent": active_accent,
            "detected_language": detected_lang,
            "edge_voice": gender_cfg["edge_voice"],
            "profile": profile,
            "matched_bases": valid_anchors,
        }

        return clone_data

    def _synthesize_kokoro(
        self,
        clone_data: dict,
        text: str,
        speed: float = 1.0,
        lang: str = "en-us"
    ) -> tuple[np.ndarray, int]:
        """Synthesizes text using local Kokoro ONNX model."""
        style_path = os.path.join(PROFILES_DIR, clone_data.get("style_file", ""))
        if os.path.exists(style_path):
            style_vector = np.load(style_path)
        else:
            style_vector = self.kokoro.get_voice_style("af_heart")

        audio_samples, sr = self.kokoro.create(
            text=text.strip(),
            voice=style_vector,
            speed=speed,
            lang=lang
        )
        return audio_samples, sr

    def _synthesize_edge_tts(
        self,
        clone_data: dict,
        text: str,
        speed: float = 1.0,
        voice_name: str | None = None
    ) -> tuple[np.ndarray, int]:
        """Synthesizes speech using Edge-TTS high-fidelity neural voices."""
        if not voice_name:
            accent = clone_data.get("accent", "American English")
            gender = clone_data.get("gender", "Male")
            accent_cfg = VOICE_MAP.get(accent, VOICE_MAP["American English"])
            voice_name = accent_cfg.get(gender, accent_cfg["Male"])["edge_voice"]

        rate_pct = int((speed - 1.0) * 100)
        rate_str = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"

        temp_out = os.path.join(OUTPUTS_DIR, f"temp_{uuid.uuid4().hex[:8]}.mp3")
        try:
            async def _run():
                comm = edge_tts.Communicate(text.strip(), voice_name, rate=rate_str)
                await comm.save(temp_out)

            asyncio.run(_run())
            y, sr = sf.read(temp_out)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            if sr != 24000:
                y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=24000)
                sr = 24000
            return y.astype(np.float32), sr
        finally:
            if os.path.exists(temp_out):
                os.remove(temp_out)

    def _synthesize_elevenlabs(
        self,
        clone_data: dict,
        text: str,
        api_key: str
    ) -> tuple[np.ndarray, int]:
        """Synthesizes speech via ElevenLabs Instant Voice Cloning."""
        headers = {
            "xi-api-key": api_key,
            "Accept": "audio/mpeg"
        }

        # Check if we already have an elevenlabs_voice_id for this clone
        voice_id = clone_data.get("elevenlabs_voice_id")
        if not voice_id:
            # Upload reference audio to create Instant Voice Clone
            ref_path = os.path.join(PROFILES_DIR, clone_data["ref_audio_file"])
            if not os.path.exists(ref_path):
                raise ValueError("Reference audio file not found on disk")

            with open(ref_path, "rb") as f:
                add_voice_resp = requests.post(
                    "https://api.elevenlabs.io/v1/voices/add",
                    headers={"xi-api-key": api_key},
                    data={
                        "name": f"Clone_{clone_data['name']}_{clone_data['id']}",
                        "description": "Instant clone created via VoiceClone Studio"
                    },
                    files={"files": (clone_data["ref_audio_file"], f, "audio/wav")}
                )

            if add_voice_resp.status_code != 200:
                raise ValueError(f"ElevenLabs voice cloning failed: {add_voice_resp.text}")

            voice_id = add_voice_resp.json().get("voice_id")
            clone_data["elevenlabs_voice_id"] = voice_id

        # Generate speech
        tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.50,
                "similarity_boost": 0.85
            }
        }

        tts_resp = requests.post(tts_url, headers=headers, json=payload)
        if tts_resp.status_code != 200:
            raise ValueError(f"ElevenLabs TTS failed: {tts_resp.text}")

        temp_out = os.path.join(OUTPUTS_DIR, f"temp_{uuid.uuid4().hex[:8]}.mp3")
        try:
            with open(temp_out, "wb") as f:
                f.write(tts_resp.content)
            y, sr = librosa.load(temp_out, sr=24000, mono=True)
            return y, sr
        finally:
            if os.path.exists(temp_out):
                os.remove(temp_out)

    def generate_speech_from_text(
        self,
        clone_data: dict,
        text: str,
        engine: str = "auto",
        speed: float = 1.0,
        pitch_shift_semitones: float = 0.0,
        warmth_mix: float = 0.65,
        lang: str = "en-us",
        edge_voice_override: str | None = None
    ) -> dict:
        """
        Synthesizes text in the cloned voice:
        1. Selects engine: ElevenLabs (if key provided), Edge-TTS (studio neural), or Kokoro (local offline).
        2. Renders base speech in matched accent and vocal register.
        3. Applies Vocal Tract Cross-Synthesis (Spectral Envelope Transfer) & Pitch Retargeting.
        4. Saves output WAV and returns metadata.
        """
        t0 = time.time()
        clone_id = clone_data["id"]
        settings = get_settings()

        # Determine effective engine
        chosen_engine = engine
        if chosen_engine == "auto":
            # If user has an ElevenLabs key and selected ElevenLabs
            if settings.get("engine") == "elevenlabs" and settings.get("elevenlabs_api_key"):
                chosen_engine = "elevenlabs"
            elif clone_data.get("accent") == "Indian English / Hindi":
                # For Indian English/Hindi, Edge-TTS provides native accents
                chosen_engine = "edge_tts"
            else:
                chosen_engine = "kokoro"

        # 1. Base Synthesis
        used_engine_name = "Kokoro Local AI"
        if chosen_engine == "elevenlabs" and settings.get("elevenlabs_api_key"):
            try:
                audio_samples, sr = self._synthesize_elevenlabs(
                    clone_data=clone_data,
                    text=text,
                    api_key=settings["elevenlabs_api_key"]
                )
                used_engine_name = "ElevenLabs Instant Voice Clone"
            except Exception as e:
                print(f"[Engine] ElevenLabs failed ({e}), falling back to Edge-TTS...")
                audio_samples, sr = self._synthesize_edge_tts(
                    clone_data=clone_data,
                    text=text,
                    speed=speed,
                    voice_name=edge_voice_override
                )
                used_engine_name = "Edge-TTS Studio Neural (Fallback)"
        elif chosen_engine == "edge_tts":
            audio_samples, sr = self._synthesize_edge_tts(
                clone_data=clone_data,
                text=text,
                speed=speed,
                voice_name=edge_voice_override
            )
            used_engine_name = "Edge-TTS Studio Neural Pool"
        else:
            audio_samples, sr = self._synthesize_kokoro(
                clone_data=clone_data,
                text=text,
                speed=speed,
                lang=lang
            )
            used_engine_name = "Kokoro Local ONNX Engine"

        # 2. Load Reference Spectral Envelope
        env_path = os.path.join(PROFILES_DIR, clone_data.get("env_file", f"{clone_id}_env.npy"))
        ref_env = None
        if os.path.exists(env_path):
            try:
                ref_env = np.load(env_path)
            except Exception:
                ref_env = None

        # 3. Apply Deep Acoustic Morpher (Pitch Retargeting + Vocal Tract Envelope Transfer)
        # For ElevenLabs, the vocal timbre is already identical, so mix is minimal; for local engines, full morphing is applied
        effective_mix = warmth_mix if chosen_engine != "elevenlabs" else 0.15

        morphed_audio = apply_timbre_morph(
            synth_audio=audio_samples,
            ref_profile=clone_data.get("profile", {}),
            ref_env=ref_env,
            sr=sr,
            mix=effective_mix,
            pitch_shift_semitones=pitch_shift_semitones
        )

        # 4. Save output audio
        gen_id = str(uuid.uuid4())[:10]
        out_filename = f"tts_{clone_id}_{gen_id}.wav"
        out_path = os.path.join(OUTPUTS_DIR, out_filename)
        sf.write(out_path, morphed_audio, sr)

        gen_duration = round(len(morphed_audio) / sr, 2)
        proc_time = round(time.time() - t0, 2)

        return {
            "generation_id": gen_id,
            "clone_id": clone_id,
            "clone_name": clone_data["name"],
            "mode": "text-to-speech",
            "engine": used_engine_name,
            "text": text,
            "audio_filename": out_filename,
            "audio_url": f"/api/audio/output/{out_filename}",
            "duration_seconds": gen_duration,
            "processing_time_seconds": proc_time,
            "speed": speed,
            "warmth_mix": warmth_mix,
            "pitch_shift": pitch_shift_semitones
        }

    def convert_speech_to_speech(
        self,
        clone_data: dict,
        input_audio_path: str,
        custom_script: str = "",
        engine: str = "auto",
        speed: float = 1.0,
        warmth_mix: float = 0.70,
        pitch_shift_semitones: float = 0.0
    ) -> dict:
        """
        Converts speech from microphone/audio input to the cloned voice:
        1. Transcribes input speech via Faster-Whisper.
        2. Synthesizes with the target cloned voice style.
        3. Applies vocal tract timbre morphing and pitch retargeting.
        """
        t0 = time.time()
        clone_id = clone_data["id"]

        transcript_res = transcribe_audio(input_audio_path)
        transcribed_text = transcript_res["text"]

        script_to_speak = custom_script.strip() if custom_script.strip() else transcribed_text
        if not script_to_speak:
            script_to_speak = "Voice sample converted successfully."

        gen_result = self.generate_speech_from_text(
            clone_data=clone_data,
            text=script_to_speak,
            engine=engine,
            speed=speed,
            warmth_mix=warmth_mix,
            pitch_shift_semitones=pitch_shift_semitones
        )

        gen_result["mode"] = "speech-to-speech"
        gen_result["original_transcript"] = transcribed_text
        gen_result["detected_language"] = transcript_res.get("language", "en")
        gen_result["total_pipeline_time_seconds"] = round(time.time() - t0, 2)

        return gen_result

# Global Engine Singleton
_engine = None

def get_voice_cloner_engine() -> VoiceClonerEngine:
    global _engine
    if _engine is None:
        _engine = VoiceClonerEngine()
    return _engine
