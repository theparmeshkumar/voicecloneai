import os
import urllib.request
import sys

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

def download_file(url, target_path):
    print(f"Downloading {os.path.basename(target_path)} from:\n  {url}")
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            sys.stdout.write(f"\rProgress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
            sys.stdout.flush()
        else:
            sys.stdout.write(f"\rDownloaded: {downloaded / (1024*1024):.1f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, target_path, reporthook=reporthook)
    print("\nDownload complete!")

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "kokoro-v1.0.onnx")
    voices_path = os.path.join(MODELS_DIR, "voices-v1.0.bin")

    if not os.path.exists(model_path):
        download_file(MODEL_URL, model_path)
    else:
        print(f"kokoro-v1.0.onnx already exists at: {model_path}")

    if not os.path.exists(voices_path):
        download_file(VOICES_URL, voices_path)
    else:
        print(f"voices-v1.0.bin already exists at: {voices_path}")

    # Pre-cache Faster-Whisper model for zero-latency first audio request
    try:
        print("Pre-caching Faster-Whisper tiny model...")
        from faster_whisper import WhisperModel
        WhisperModel("tiny", device="cpu", compute_type="int8")
        print("Faster-Whisper model cached successfully!")
    except Exception as e:
        print(f"Faster-Whisper pre-cache note: {e}")

if __name__ == "__main__":
    main()

