import os
import sys
import webbrowser
import threading
import time
import uvicorn

def open_browser():
    time.sleep(1.5)
    print("\n[VoiceClone Studio] Launching browser at http://localhost:8000 ...")
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    print("=" * 65)
    print("        🎙️ VOICECLONE AI STUDIO • NEURAL ACOUSTIC SYNTHESIS")
    print("=" * 65)
    print("Engine: Kokoro-ONNX + Faster-Whisper + Librosa Acoustic Profiler")
    print("Mode:   Offline Local Inference on CPU")
    print("URL:    http://localhost:8000")
    print("=" * 65)

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, log_level="info")
