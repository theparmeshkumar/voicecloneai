import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.cloner import get_voice_cloner_engine, PROFILES_DIR, OUTPUTS_DIR, VOICE_MAP
from backend.storage import (
    list_profiles,
    get_profile,
    save_profile,
    delete_profile,
    list_history,
    add_history_entry,
    get_settings,
    save_settings
)
from backend.audio_processor import transcribe_audio

app = FastAPI(title="VoiceClone Studio API", version="2.0.0")

# Enable CORS for local dev / browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event to warm up engine
@app.on_event("startup")
def startup_event():
    print("[Server] Initializing voice cloning engine...")
    get_voice_cloner_engine()
    print("[Server] Voice cloning engine ready!")

@app.get("/api/health")
def health_check():
    engine = get_voice_cloner_engine()
    return {
        "status": "online",
        "engine": "VoiceClone Studio Multi-Engine",
        "available_base_voices": len(engine.available_voices),
        "supported_accents": list(VOICE_MAP.keys())
    }

@app.get("/api/settings")
def read_settings():
    return get_settings()

class SettingsUpdate(BaseModel):
    engine: str = "auto"
    elevenlabs_api_key: str = ""

@app.post("/api/settings")
def update_settings(cfg: SettingsUpdate):
    updated = save_settings(cfg.dict())
    return updated

@app.get("/api/profiles")
def get_all_profiles():
    return list_profiles()

@app.get("/api/profiles/{clone_id}")
def get_single_profile(clone_id: str):
    profile = get_profile(clone_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile

class ProfileUpdateRequest(BaseModel):
    name: str | None = None
    gender: str | None = None
    accent: str | None = None

@app.post("/api/profiles/{clone_id}/update")
def update_existing_profile(clone_id: str, req: ProfileUpdateRequest):
    profile = get_profile(clone_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    if req.name is not None and req.name.strip():
        profile["name"] = req.name.strip()

    gender_changed = req.gender in ["Male", "Female"] and req.gender != profile.get("gender")
    accent_changed = req.accent in VOICE_MAP and req.accent != profile.get("accent")

    if gender_changed or accent_changed:
        if req.gender in ["Male", "Female"]:
            profile["gender"] = req.gender
        if req.accent in VOICE_MAP:
            profile["accent"] = req.accent

        # Recalculate anchors and style vector
        engine = get_voice_cloner_engine()
        accent_cfg = VOICE_MAP.get(profile["accent"], VOICE_MAP["American English"])
        gender_cfg = accent_cfg.get(profile["gender"], accent_cfg["Male"])
        candidate_anchors = gender_cfg["kokoro_anchors"]
        valid_anchors = [v for v in candidate_anchors if v in engine.available_voices] or ["am_adam"]

        import numpy as np
        style_vectors = [engine.kokoro.get_voice_style(v) for v in valid_anchors]
        weights = np.ones(len(valid_anchors)) / len(valid_anchors)
        cloned_style = np.zeros_like(style_vectors[0])
        for w, vec in zip(weights, style_vectors):
            cloned_style += w * vec

        style_path = os.path.join(PROFILES_DIR, profile["style_file"])
        np.save(style_path, cloned_style)
        profile["matched_bases"] = valid_anchors
        profile["edge_voice"] = gender_cfg["edge_voice"]

    save_profile(profile)
    return profile

@app.delete("/api/profiles/{clone_id}")
def remove_profile(clone_id: str):
    success = delete_profile(clone_id)
    if not success:
        raise HTTPException(status_code=404, detail="Profile not found")
    return {"status": "deleted", "id": clone_id}

@app.post("/api/clone/upload")
async def create_clone_from_audio(
    audio_file: UploadFile = File(...),
    name: str = Form(""),
    description: str = Form(""),
    gender: str = Form(""),
    accent: str = Form("")
):
    """
    Step 1: Upload a 1 to 3 minute sample voice.
    Analyzes the voice, extracts pitch/formants/envelope, and creates a tailored cloned voice profile.
    """
    engine = get_voice_cloner_engine()
    temp_dir = tempfile.mkdtemp()
    temp_audio_path = os.path.join(temp_dir, audio_file.filename or "sample.wav")
    
    try:
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        clone_data = engine.analyze_and_create_clone(
            audio_file_path=temp_audio_path,
            clone_name=name,
            description=description,
            gender_override=gender if gender in ["Male", "Female"] else None,
            accent_override=accent if accent in VOICE_MAP else None
        )
        
        # Save to persistent storage
        save_profile(clone_data)
        return clone_data
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to clone voice: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

class TTSRequest(BaseModel):
    clone_id: str
    text: str
    engine: str = "auto"
    speed: float = 1.0
    pitch_shift: float = 0.0
    warmth_mix: float = 0.65
    lang: str = "en-us"
    edge_voice: str | None = None

@app.post("/api/generate/tts")
def generate_tts(req: TTSRequest):
    """
    Step 2 (Option A): Generate speech from text script using cloned voice.
    """
    profile = get_profile(req.clone_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Cloned voice profile not found")
    
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text script cannot be empty")

    engine = get_voice_cloner_engine()
    try:
        result = engine.generate_speech_from_text(
            clone_data=profile,
            text=req.text,
            engine=req.engine,
            speed=req.speed,
            pitch_shift_semitones=req.pitch_shift,
            warmth_mix=req.warmth_mix,
            lang=req.lang,
            edge_voice_override=req.edge_voice
        )
        add_history_entry(result)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/api/transcribe")
async def transcribe_only(audio_file: UploadFile = File(...)):
    """
    Transcribes audio snippet via Faster-Whisper to allow user review before generation.
    """
    temp_dir = tempfile.mkdtemp()
    temp_audio_path = os.path.join(temp_dir, audio_file.filename or "input.wav")
    try:
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        return transcribe_audio(temp_audio_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/api/generate/sts")
async def generate_sts(
    clone_id: str = Form(...),
    audio_file: UploadFile = File(...),
    custom_script: str = Form(""),
    engine: str = Form("auto"),
    speed: float = Form(1.0),
    pitch_shift: float = Form(0.0),
    warmth_mix: float = Form(0.70)
):
    """
    Step 2 (Option B): Speech-to-Speech conversion.
    Takes user's recorded or uploaded audio, transcribes, and renders in cloned voice.
    """
    profile = get_profile(clone_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Cloned voice profile not found")

    engine_inst = get_voice_cloner_engine()
    temp_dir = tempfile.mkdtemp()
    temp_audio_path = os.path.join(temp_dir, audio_file.filename or "input.wav")
    try:
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        result = engine_inst.convert_speech_to_speech(
            clone_data=profile,
            input_audio_path=temp_audio_path,
            custom_script=custom_script,
            engine=engine,
            speed=speed,
            warmth_mix=warmth_mix,
            pitch_shift_semitones=pitch_shift
        )
        add_history_entry(result)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Speech-to-Speech failed: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.get("/api/history")
def get_generation_history():
    return list_history()

# Audio streaming endpoints
@app.get("/api/audio/profile/{filename}")
def stream_profile_audio(filename: str):
    file_path = os.path.join(PROFILES_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/wav")

@app.get("/api/audio/output/{filename}")
def stream_output_audio(filename: str):
    file_path = os.path.join(OUTPUTS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/wav")

# Serve Frontend static assets
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
