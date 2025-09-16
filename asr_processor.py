from __future__ import annotations
import os, re, csv, shutil, logging, subprocess
from pathlib import Path
from typing import Dict, List, Optional
from datetime import timedelta

import pandas as pd
from tqdm import tqdm

import config

# yt-dlp for download & metadata
from yt_dlp import YoutubeDL
# faster-whisper for GPU Whisper
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------- small utils ----------
def _add_to_path(path: str) -> None:
    if path and os.path.isdir(path) and path not in os.environ.get("PATH",""):
        os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]

def _inject_tool_paths():
    # Optional: add CUDA/cuDNN/FFmpeg to PATH on Windows
    for p in getattr(config, "CUDA_BIN_DIRS", []):
        _add_to_path(p)
    _add_to_path(getattr(config, "FFMPEG_DIR", ""))

def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_").lower()

def ensure_caption_dirs(college: str) -> Dict[str, Path]:
    base = Path(config.CAPTIONS_DIR) / slugify(college)
    raw = base / "raw"
    (base / "clean").mkdir(parents=True, exist_ok=True)
    (base / "pure").mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    return {"base": base, "raw": raw}

def _existing_vtts(raw_dir: Path) -> set[str]:
    return {p.stem for p in raw_dir.glob("*.vtt")}

def _append_asr_failed(row: Dict[str, str], reason: str, details: str = "") -> None:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    write_header = not os.path.exists(config.ASR_FAILED_FILE)
    with open(config.ASR_FAILED_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["college", "videoId", "url", "title", "failure_reason", "error_details"])
        w.writerow([row.get("college",""), row.get("videoId",""), row.get("url",""),
                    row.get("title",""), reason, details])

# ---------- yt-dlp helpers ----------
def _probe_video_info(url: str) -> dict:
    """
    Probe video metadata without downloading. If yt-dlp reports an upcoming live,
    return a dict that signals it so callers can skip gracefully.
    """
    try:
        with YoutubeDL({"skip_download": True, "quiet": True, "no_warnings": True}) as ydl:
            return ydl.extract_info(url, download=False)
    except Exception as e:
        msg = str(e) or ""
        # Common yt-dlp message for scheduled lives/premieres
        if "This live event will begin" in msg or "Premieres in" in msg or "is not yet available" in msg:
            return {"live_status": "upcoming"}
        logger.warning("yt-dlp probe failed for %s: %s", url, e)
        return {}

def _duration_minutes(info: dict) -> Optional[float]:
    dur = info.get("duration")
    if dur is None:
        return None
    try:
        return float(dur) / 60.0
    except Exception:
        return None

def download_video_audio(url: str, out_dir: Path, video_id: str) -> Optional[Path]:
    """
    Download bestaudio to a temp file (m4a/webm), then convert to WAV 16k mono via ffmpeg.
    Returns the WAV path or None on failure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / video_id
    # First stage: download bestaudio without postprocessors
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(base) + ".%(ext)s",
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "ignoreerrors": False,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            _ = ydl.extract_info(url, download=True)
        # Find the downloaded file by matching base.*
        dl_candidates = list(out_dir.glob(video_id + ".*"))
        # Prefer common audio extensions
        dl_candidates.sort(key=lambda p: (p.suffix not in {".m4a", ".webm", ".mp3", ".opus"}, len(p.suffix)))
        if not dl_candidates:
            logger.error("Could not locate downloaded audio for %s", url)
            return None
        src = dl_candidates[0]
        wav = base.with_suffix(".wav")

        # Convert to 16k mono WAV
        cmd = ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(wav)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0 or not wav.exists():
            logger.error("ffmpeg convert failed for %s: %s", src, proc.stderr[-400:])
            return None

        # Clean up source container (optional)
        try:
            src.unlink(missing_ok=True)
        except Exception:
            pass

        return wav
    except Exception as e:
        logger.error("Audio download failed for %s: %s", url, e)
        return None

# ---------- VTT writer ----------
def _sec_to_vtt_ts(sec: float) -> str:
    # hh:mm:ss.mmm
    td = timedelta(seconds=sec)
    total_ms = int(td.total_seconds() * 1000)
    hours = total_ms // 3600000
    rem = total_ms % 3600000
    mins = rem // 60000
    rem = rem % 60000
    secs = rem // 1000
    ms = rem % 1000
    return f"{hours:02d}:{mins:02d}:{secs:02d}.{ms:03d}"

def _write_vtt(segments, out_vtt: Path) -> None:
    with out_vtt.open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            start = _sec_to_vtt_ts(seg.start)
            end = _sec_to_vtt_ts(seg.end)
            text = (seg.text or "").strip()
            if not text:
                continue
            f.write(f"{start} --> {end}\n{text}\n\n")

# ---------- Whisper driver ----------
_model_singleton = None

def _get_model():
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = WhisperModel(
            model_size_or_path=config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
    return _model_singleton

def transcribe_audio_to_vtt(audio_path: Path, out_vtt: Path) -> bool:
    """
    Transcribe with faster-whisper.
    Strategy:
      - First pass: language='en', vad_filter=True (cleaner)
      - If zero segments, fallback pass: language='en', vad_filter=False
    """
    try:
        model = _get_model()

        def _run(vad_on: bool):
            seg_iter, _info = model.transcribe(
                str(audio_path),
                language="en",
                beam_size=getattr(config, "WHISPER_BEAM_SIZE", 1),
                vad_filter=vad_on,
                condition_on_previous_text=False,
            )
            return list(seg_iter)

        # Pass 1: with VAD
        segs = _run(vad_on=True)
        if not segs:
            # Pass 2: fallback without VAD
            logger.info("No segments with VAD; retrying without VAD for %s", audio_path.name)
            segs = _run(vad_on=False)
            if not segs:
                return False

        _write_vtt(segs, out_vtt)
        return out_vtt.exists()
    except Exception as e:
        logger.error("Transcription failed for %s: %s", audio_path, e)
        return False

def cleanup_temp_files(temp_dir: Path) -> None:
    try:
        if temp_dir.exists() and getattr(config, "CLEANUP_TEMP_FILES", True):
            shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass

# ---------- Orchestrator ----------
def process_failed_captions() -> None:
    """
    1) Reads data/caption_Failed.csv
    2) Skips any video whose captions/{school}/raw/{videoId}.vtt already exists (if ASR_ONLY_MISSING)
    3) Downloads audio (yt-dlp) -> temp_audio/{videoId}.wav (16k mono)
    4) Runs faster-whisper and writes captions/{school}/raw/{videoId}.vtt
    5) Logs failures to data/asr_failed.csv
    """
    
    # Start with a clean failures file this run (optional)
    try:
        if os.path.exists(config.ASR_FAILED_FILE):
            os.remove(config.ASR_FAILED_FILE)
    except Exception:
        pass
    
    if not os.path.exists(config.CAPTION_FAILED_FILE):
        logger.info("No caption_Failed.csv found at %s; nothing to process.", config.CAPTION_FAILED_FILE)
        return

    _inject_tool_paths()
    os.makedirs(config.TEMP_AUDIO_DIR, exist_ok=True)

    df = pd.read_csv(config.CAPTION_FAILED_FILE, dtype=str).fillna("")
    rows = df.to_dict(orient="records")
    if not rows:
        logger.info("caption_Failed.csv is empty; nothing to process.")
        return

    # Group by college for idempotent raw-folder check
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        grouped.setdefault(r["college"], []).append(r)

    total_done = total_skipped = total_failed = 0

    group_iter = grouped.items()
    if getattr(config, "ENABLE_PROGRESS_BARS", True):
        group_iter = tqdm(list(grouped.items()), desc="ASR colleges", unit="college")

    for college, items in group_iter:
        dirs = ensure_caption_dirs(college)
        raw_dir = dirs["raw"]
        existing = _existing_vtts(raw_dir)

        item_iter = items
        if getattr(config, "ENABLE_PROGRESS_BARS", True):
            item_iter = tqdm(items, desc=f"{college}", unit="video", leave=False)

        for r in item_iter:
            vid = r.get("videoId", "")
            url = r.get("url") or f"https://www.youtube.com/watch?v={vid}"
            out_vtt = raw_dir / f"{vid}.vtt"

            # Idempotent skip
            if getattr(config, "ASR_ONLY_MISSING", True) and (vid in existing or out_vtt.exists()):
                total_skipped += 1
                continue

            # ---- NEW: probe and skip upcoming lives BEFORE download ----
            info = _probe_video_info(url)
            live_status = (info or {}).get("live_status")
            if live_status in {"is_upcoming", "upcoming"}:
                _append_asr_failed(r, "upcoming_live", "live event not started")
                total_failed += 1
                continue

            # Length guard (avoid mega-long streams)
            dur_min = _duration_minutes(info)
            if dur_min is not None and dur_min > getattr(config, "MAX_VIDEO_LENGTH_MINUTES", 240):
                _append_asr_failed(r, "too_long", f"{dur_min:.1f} minutes")
                total_failed += 1
                continue

            # Download audio (or use pre-downloaded)
            temp_dir = Path(config.TEMP_AUDIO_DIR)
            audio_path: Optional[Path] = None
            if getattr(config, "ENABLE_ASR_DOWNLOAD", True):
                audio_path = download_video_audio(url, temp_dir, vid)
                if not audio_path or not audio_path.exists():
                    _append_asr_failed(r, "download_failed", "audio missing")
                    total_failed += 1
                    if getattr(config, "CLEANUP_TEMP_FILES", True):
                        cleanup_temp_files(temp_dir)
                    continue
            else:
                candidate = temp_dir / f"{vid}.wav"
                audio_path = candidate if candidate.exists() else None
                if audio_path is None:
                    _append_asr_failed(r, "audio_missing", "ENABLE_ASR_DOWNLOAD=False and no temp wav found")
                    total_failed += 1
                    continue

            # Transcribe
            ok = True
            if getattr(config, "ENABLE_ASR_TRANSCRIBE", True):
                ok = transcribe_audio_to_vtt(audio_path, out_vtt)
            else:
                ok = False  # explicitly disabled

            if ok and out_vtt.exists():
                total_done += 1
                existing.add(vid)
            else:
                _append_asr_failed(r, "transcription_failed")
                total_failed += 1

            if getattr(config, "CLEANUP_TEMP_FILES", True):
                try:
                    audio_path and audio_path.exists() and audio_path.unlink()
                except Exception:
                    pass

    logger.info("ASR finished. New VTTs: %d | Skipped existing: %d | Failed: %d",
                total_done, total_skipped, total_failed)
