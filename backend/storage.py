import os
import json
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES_FILE = os.path.join(BASE_DIR, "data", "profiles", "profiles.json")
HISTORY_FILE = os.path.join(BASE_DIR, "data", "outputs", "history.json")

def _load_json(file_path: str, default=None):
    if default is None:
        default = []
    if not os.path.exists(file_path):
        return default
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _save_json(file_path: str, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def list_profiles() -> list[dict]:
    return _load_json(PROFILES_FILE, [])

def get_profile(clone_id: str) -> dict | None:
    profiles = list_profiles()
    for p in profiles:
        if p["id"] == clone_id:
            return p
    return None

def save_profile(profile_data: dict) -> dict:
    profiles = list_profiles()
    # Check if already exists
    for i, p in enumerate(profiles):
        if p["id"] == profile_data["id"]:
            profiles[i] = profile_data
            _save_json(PROFILES_FILE, profiles)
            return profile_data
    profiles.insert(0, profile_data)
    _save_json(PROFILES_FILE, profiles)
    return profile_data

def delete_profile(clone_id: str) -> bool:
    profiles = list_profiles()
    new_profiles = [p for p in profiles if p["id"] != clone_id]
    if len(new_profiles) != len(profiles):
        _save_json(PROFILES_FILE, new_profiles)
        return True
    return False

def list_history() -> list[dict]:
    return _load_json(HISTORY_FILE, [])

def add_history_entry(entry: dict):
    history = list_history()
    entry["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    history.insert(0, entry)
    # Keep last 50 generations
    _save_json(HISTORY_FILE, history[:50])
    return entry

SETTINGS_FILE = os.path.join(BASE_DIR, "data", "settings.json")

def get_settings() -> dict:
    default_settings = {
        "engine": "auto",
        "elevenlabs_api_key": ""
    }
    return _load_json(SETTINGS_FILE, default_settings)

def save_settings(settings: dict) -> dict:
    _save_json(SETTINGS_FILE, settings)
    return settings

