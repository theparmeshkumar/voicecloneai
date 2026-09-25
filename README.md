# VoiceClone AI Studio 🎙️

A local, neural acoustic voice cloning and speech synthesis web application running **100% offline on your CPU** without any paid cloud APIs.

---

## 🌟 Key Features

1. **Voice Sample Ingestion (1 to 3 Minutes)**
   - Upload any audio format (`.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`, `.webm`) or record your voice live in the browser using your microphone.
   - Real-time Web Audio API recording meter and waveform visualizer.

2. **Acoustic Profiling & True Voice Cloning**
   - **Language & Dialect Detection**: Uses Whisper to automatically detect speaker language and regional accent (e.g. Indian English / Hindi, American English, British English).
   - **Pitch Analysis ($F_0$) & Retargeting**: Uses probabilistic YIN (`librosa.pyin`) to extract fundamental frequency contours, vocal register, and semitone offsets.
   - **Vocal Tract Formants & Spectral Envelope**: Linear Predictive Coding (LPC) and smooth STFT spectral envelope cross-synthesis transfer the reference speaker's throat, nasal, and mouth formant coloring onto synthesized audio.
   - **Multi-Engine Neural Synthesis**:
     - **Auto-Optimal Hybrid**: Automatically pairs the speaker's detected dialect and gender with high-fidelity neural voices (such as `en-IN-PrabhatNeural` for Indian male or `en-IN-NeerjaNeural` for Indian female) and applies spectral formant morphing.
     - **Edge-TTS Studio Pool**: Over 300 studio-grade regional neural voices for accurate accents.
     - **Kokoro-ONNX**: 100% offline local CPU engine with style vector interpolation.
     - **ElevenLabs 1:1 Cloud Cloning**: Optional cloud fidelity if you add your API key in Settings.

3. **Dual-Mode Generation Studio**
   - **Mode A: Text Script Input (TTS)**
     - Enter any script or choose from presets (*Podcast Intro, Product Keynote, Audio Story, Documentary, Assistant*).
     - Live word counter and duration estimation.
     - Controls for speaking speed (0.70x to 1.50x), semitone pitch retargeting (-8st to +8st), and vocal tract warmth morphing.
   - **Mode B: Speech-to-Speech / Direct Voice Recording (STS)**
     - Record directly with your microphone or upload any source audio.
     - Local **Faster-Whisper** automatically transcribes your spoken words in ~1 second.
     - Review and tweak the recognized words before synthesis.
     - The engine re-synthesizes the speech matching the target cloned voice!

4. **Master Studio Player & Comparison**
   - Interactive HTML5 Canvas waveform player with time scrubbing and playhead.
   - **Side-by-side comparison**: Instantly switch between the *Generated Cloned Speech* and the *Original Reference Voice*.
   - Speed multiplier (0.75x, 1.0x, 1.25x, 1.5x) and lossless WAV download.

5. **Voice Library, Trait Editing & Settings**
   - Save multiple cloned voices with custom names, gender, and regional dialect tags.
   - Quick "Change ✎" modal to adjust speaker gender or accent at any time.
   - Global Settings modal to set default synthesis engine or optional ElevenLabs key.

---

## 🚀 Quick Start

### 1. Launch with Batch Script (Windows)
Double-click `start.bat` or run in PowerShell:
```powershell
.\start.bat
```

### 2. Or Launch with Python
```powershell
.venv\Scripts\python.exe run.py
```
This automatically starts the server and opens `http://localhost:8000` in your web browser.

---

## 🛠️ Technology Stack

- **Neural TTS Engines**:
  - Edge-TTS Studio Neural Pool (Multi-dialect regional neural models).
  - Kokoro-ONNX (82M parameter SOTA open-source model running on CPU).
  - Optional ElevenLabs API integration.
- **Acoustic Cross-Synthesis & DSP**:
  - `librosa.pyin` harmonic fundamental pitch tracking.
  - Frequency-domain spectral envelope transfer ($H(f) = (E_{\text{ref}} / E_{\text{synth}})^\alpha$).
  - Phase-vocoder pitch retargeting.
- **Speech-to-Text**: [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 int8 quantized CPU inference).
- **Backend**: FastAPI, Uvicorn, SoundFile, Librosa, SciPy, Pydub.
- **Frontend**: Vanilla CSS Glassmorphism design system, Web Audio API, HTML5 Canvas, modern JavaScript modules.

---

## 📁 Project Structure

```
AudioClonning/
├── backend/
│   ├── app.py             # FastAPI REST & audio streaming endpoints
│   ├── cloner.py          # Kokoro engine, style vector synthesis & generation
│   ├── audio_processor.py # Pitch, LPC formants, Whisper STT, timbre morph
│   └── storage.py         # JSON persistence for profiles and history
├── frontend/
│   ├── index.html         # Studio interface
│   ├── css/style.css      # Glassmorphic dark theme stylesheet
│   └── js/
│       ├── app.js         # Core UI coordinator
│       ├── audio_player.js# Waveform canvas player
│       ├── audio_recorder.js# Mic recorder with Web Audio API
│       └── visualizer.js  # 6-axis acoustic radar chart
├── models/
│   ├── kokoro-v1.0.onnx   # Neural TTS model weights
│   └── voices-v1.0.bin    # 54 base voice style embeddings
├── data/
│   ├── profiles/          # Saved cloned voice profiles & reference audio
│   └── outputs/           # Synthesized cloned speech output files
├── run.py                 # Launcher script
├── start.bat              # Windows double-click shortcut
└── requirements.txt       # Python dependencies
```
