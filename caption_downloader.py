"""
Module responsible for downloading captions for a list of YouTube videos.

Given the results of `video_collection.py`, this module attempts to
download either professional or automatic English captions for each video
using the `yt-dlp` library. Captions are stored in
`captions/{slugified_school_name}/raw/` as VTT files named exactly
`<VIDEOID>.vtt` (language suffixes like `.en` are stripped after download).

Videos without available captions are recorded in `data/caption_Failed.csv`
with a reason for failure.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Set, Optional
from pathlib import Path

import pandas as pd
from yt_dlp import YoutubeDL

import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ----------------------------
# Helpers
# ----------------------------
def slugify(value: str) -> str:
    """Convert a string to a filesystem-safe slug."""
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_").lower()


def ensure_caption_directories(school_slug: str) -> Dict[str, str]:
    """Ensure that raw/clean/pure subdirectories exist for a school."""
    base_dir = os.path.join(config.CAPTIONS_DIR, school_slug)
    raw_dir = os.path.join(base_dir, "raw")
    clean_dir = os.path.join(base_dir, "clean")
    pure_dir = os.path.join(base_dir, "pure")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(pure_dir, exist_ok=True)
    return {"raw": raw_dir, "clean": clean_dir, "pure": pure_dir}


def _existing_vtt_ids(raw_dir: str) -> Set[str]:
    """
    Return a set of videoIds that already have a VTT in raw/.
    Treat files like VIDEOID.vtt and VIDEOID.en.vtt (or en-US/en-GB) as the same VIDEOID.
    """
    ids: Set[str] = set()
    p = Path(raw_dir)
    if not p.exists():
        return ids
    for f in p.glob("*.vtt"):
        base = f.name[:-4]  # strip .vtt
        # If there's a language suffix (e.g., '.en' or '.en-US'), drop it.
        vid = base.split(".", 1)[0]
        if vid:
            ids.add(vid)
    return ids


def _append_failed(row: Dict[str, str], reason: str) -> None:
    """Append a single failure row to caption_Failed.csv (keeps history)."""
    from csv import writer
    os.makedirs(config.DATA_DIR, exist_ok=True)
    write_header = not os.path.exists(config.CAPTION_FAILED_FILE)
    with open(config.CAPTION_FAILED_FILE, "a", newline="", encoding="utf-8") as f:
        w = writer(f)
        if write_header:
            w.writerow(["college", "videoId", "url", "title", "failure_reason"])
        w.writerow([
            row.get("college", ""),
            row.get("videoId", ""),
            row.get("url", ""),
            row.get("title", ""),
            reason,
        ])


def _find_any_caption_variant(raw_dir: str, video_id: str) -> Optional[Path]:
    """
    Return the first found path among:
      VIDEOID.vtt, VIDEOID.en.vtt, VIDEOID.en-US.vtt, VIDEOID.en-GB.vtt, or any VIDEOID.*.vtt
    """
    folder = Path(raw_dir)
    direct = folder / f"{video_id}.vtt"
    if direct.exists():
        return direct

    # Prefer common English tags first
    for suffix in ("en", "en-US", "en-GB"):
        cand = folder / f"{video_id}.{suffix}.vtt"
        if cand.exists():
            return cand

    # Fall back to any VIDEOID.*.vtt
    for cand in folder.glob(f"{video_id}.*.vtt"):
        return cand

    return None


def _normalize_vtt_filename(raw_dir: str, video_id: str) -> Optional[Path]:
    """
    Ensure the subtitle lives at raw/VIDEOID.vtt.
    If we downloaded VIDEOID.en.vtt (or other variant), rename it to VIDEOID.vtt.
    If VIDEOID.vtt already exists, remove the variant.
    Returns the final path if present, else None.
    """
    folder = Path(raw_dir)
    final_path = folder / f"{video_id}.vtt"
    if final_path.exists():
        # Clean up any extra variants
        for cand in folder.glob(f"{video_id}.*.vtt"):
            if cand != final_path:
                try:
                    cand.unlink(missing_ok=True)
                except Exception:
                    pass
        return final_path

    variant = _find_any_caption_variant(raw_dir, video_id)
    if not variant:
        return None

    try:
        variant.rename(final_path)
    except FileExistsError:
        # Someone created it in the meantime; just remove variant
        try:
            variant.unlink(missing_ok=True)
        except Exception:
            pass
    except Exception as e:
        logger.warning("Could not rename %s -> %s: %s", variant.name, final_path.name, e)
        return variant if variant.exists() else None

    return final_path if final_path.exists() else None


# ----------------------------
# Download logic
# ----------------------------
def download_caption_for_video(video_url: str, output_base: str, prefer_manual: bool = True) -> bool:
    """
    Attempt to download captions for a single video.

    Parameters
    ----------
    video_url : str
        YouTube watch URL.
    output_base : str
        Base path without extension (e.g., raw/<VIDEOID>).
        yt-dlp will write VIDEOID.en.vtt (or similar); we will rename to VIDEOID.vtt afterward.
    prefer_manual : bool
        If True, try manual captions before auto captions.

    Returns
    -------
    bool
        True if at least one caption file was downloaded.
    """
    try:
        # Probe availability first
        with YoutubeDL({"skip_download": True, "quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(video_url, download=False)

        manual_langs = list(info.get("subtitles", {}) or {})
        auto_langs = list(info.get("automatic_captions", {}) or {})

        # Common options shared by both paths; set a subtitle-specific template to keep things tidy.
        # NOTE: yt-dlp will still add language code before .vtt; we normalize afterwards.
        base_opts = {
            "skip_download": True,
            "subtitlesformat": "vtt",
            "outtmpl": {
                # subtitles are saved using this pattern
                "subtitle": output_base + ".%(ext)s",
            },
            "quiet": True,
            "no_warnings": True,
            "retries": 3,
            "socket_timeout": 20,
        }

        # Prefer manual
        if prefer_manual and any(lang.startswith("en") for lang in manual_langs):
            ydl_opts = {
                **base_opts,
                "writesubtitles": True,
                "subtitleslangs": ["en", "en-US", "en-GB"],
            }
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            return True

        # Fallback to auto
        if any(lang.startswith("en") for lang in auto_langs):
            ydl_opts = {
                **base_opts,
                "writeautomaticsub": True,
                "subtitleslangs": ["en", "en-US", "en-GB"],
            }
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            return True

    except Exception as e:
        logger.error("Caption download error for %s: %s", video_url, e)

    return False


def download_captions() -> None:
    """
    Idempotent caption download for all videos in the video list CSV.

    - Creates captions/{school}/{raw,clean,pure}/ if missing
    - Skips any video whose {raw}/{videoId}.vtt already exists (or any VIDEOID.*.vtt)
    - Tries manual captions first, then auto captions
    - Renames any downloaded VIDEOID.*.vtt to VIDEOID.vtt
    - Appends failures to caption_Failed.csv
    """
    logger.info("Starting caption download…")
    config.ensure_directories()

    if not os.path.exists(config.VIDEO_LIST_FILE):
        logger.error("Video list file not found at %s. Run video collection first.", config.VIDEO_LIST_FILE)
        return

    video_df = pd.read_csv(config.VIDEO_LIST_FILE, dtype=str).fillna("")
    # Group rows by college so we can build a per-school stopper set
    by_college: Dict[str, List[Dict[str, str]]] = {}
    for _, r in video_df.iterrows():
        row = {
            "college": r["college"],
            "videoId": r["videoId"],
            "url": r["url"],
            "title": r["title"],
        }
        by_college.setdefault(row["college"], []).append(row)

    total_skipped = 0
    total_downloaded = 0
    total_failed = 0

    for college, rows in by_college.items():
        school_slug = slugify(college)
        dirs = ensure_caption_directories(school_slug)
        raw_dir = dirs["raw"]

        existing = _existing_vtt_ids(raw_dir)

        for row in rows:
            vid = row["videoId"]
            if not vid:
                _append_failed(row, "missing_videoId")
                total_failed += 1
                continue

            # STOPPER: if any variant exists, consider it done
            if vid in existing or _find_any_caption_variant(raw_dir, vid):
                logger.info("Skip existing captions for %s (%s)", row["title"], vid)
                total_skipped += 1
                continue

            url = row["url"] or f"https://www.youtube.com/watch?v={vid}"
            out_base = os.path.join(raw_dir, vid)

            logger.info("Downloading captions for %s (%s)…", row["title"], vid)
            ok = download_caption_for_video(url, out_base, prefer_manual=True)

            # Normalize filename to VIDEOID.vtt regardless of language suffix
            final_path = _normalize_vtt_filename(raw_dir, vid) if ok else None

            if ok and final_path and final_path.exists():
                existing.add(vid)
                total_downloaded += 1
                logger.info("Downloaded captions → %s", final_path.name)
            else:
                _append_failed(row, "No captions available or download failed")
                total_failed += 1
                logger.warning("No captions available for %s (%s)", row["title"], vid)

    logger.info(
        "Caption download complete. Downloaded: %d, Skipped existing: %d, Failed: %d",
        total_downloaded, total_skipped, total_failed
    )
