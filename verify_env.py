# verify_env.py
import os, shutil, sys, platform, importlib
print("=== Env basics ===")
print("Python:", sys.version)
print("Platform:", platform.platform())
print("PATH has FFmpeg:", shutil.which("ffmpeg"))

cuda_dirs = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin",
    r"C:\Program Files\NVIDIA\CUDNN\v9.13\bin\12.9",
]
for p in cuda_dirs:
    print(f"On PATH? {p} ->", "YES" if p in os.environ.get("PATH","") else "NO")

def try_import(name):
    try:
        m = importlib.import_module(name)
        print(f"Import OK: {name} ({getattr(m, '__version__', 'no __version__')})")
        return m
    except Exception as e:
        print(f"Import FAIL: {name} -> {e}")
        return None

print("\n=== Imports ===")
yt_dlp = try_import("yt_dlp")
tqdm = try_import("tqdm")
fw = try_import("faster_whisper")
ct2 = try_import("ctranslate2")

# CTranslate2 CUDA introspection (if available)
if ct2:
    try:
        has_cuda = getattr(ct2, "contains_cuda", lambda: None)()
        dev_count = getattr(ct2, "get_cuda_device_count", lambda: None)()
        cuda_ver = getattr(ct2, "get_cuda_version", lambda: None)()
        print(f"\nCTranslate2 CUDA? {has_cuda}, device_count={dev_count}, cuda_version={cuda_ver}")
    except Exception as e:
        print("CTranslate2 CUDA probe error:", e)

# Minimal GPU check: load a tiny Whisper model on CUDA and release.
if fw:
    from faster_whisper import WhisperModel
    try:
        print("\nLoading tiny model on CUDA (smoke test)...")
        model = WhisperModel("tiny", device="cuda", compute_type="float16")
        print("Loaded tiny model on CUDA successfully.")
        del model
    except Exception as e:
        print("Failed to load tiny on CUDA:", e)

# Optional: full model check (set env var VERIFY_FULL=1 to try large-v3)
if fw and os.environ.get("VERIFY_FULL", "") == "1":
    from faster_whisper import WhisperModel
    try:
        print("\nLoading configured model on CUDA:", os.environ.get("WHISPER_MODEL", "large-v3"))
        model_name = os.environ.get("WHISPER_MODEL", "large-v3")
        model = WhisperModel(model_name, device="cuda", compute_type="float16")
        print("Loaded", model_name, "on CUDA successfully.")
        del model
    except Exception as e:
        print("Failed to load configured model on CUDA:", e)

print("\n=== Done ===")
