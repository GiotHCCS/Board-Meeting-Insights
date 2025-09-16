"""
Module responsible for downloading captions for a list of YouTube videos.

Given the results of `video_collection.py`, this module attempts to
download either professional or automatic English captions for each video
using the `yt-dlp` library.  Captions are stored in
`captions/{slugified_school_name}/raw/` as VTT files.  Videos without
available captions are recorded in `data/caption_Failed.csv` with a
reason for failure.

Only the raw caption files are downloaded in Step 1.  Subsequent stages
of the pipeline will clean and normalise these transcripts for further
processing.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Set
from pathlib import Path

import pandas as pd
from yt_dlp import YoutubeDL

import config


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def slugify(value: str) -> str:
    """Convert a string to a filesystem-safe slug.

    Replaces non-alphanumeric characters with underscores and collapses
    multiple underscores.  Leading/trailing underscores are stripped.
    """
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_").lower()


def ensure_caption_directories(school_slug: str) -> Dict[str, str]:
    """Ensure that raw/clean/pure subdirectories exist for a school.

    Returns a dictionary mapping folder types (raw, clean, pure) to their
    absolute paths.
    """
    base_dir = os.path.join(config.CAPTIONS_DIR, school_slug)
    raw_dir = os.path.join(base_dir, "raw")
    clean_dir = os.path.join(base_dir, "clean")
    pure_dir = os.path.join(base_dir, "pure")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(pure_dir, exist_ok=True)
    return {"raw": raw_dir, "clean": clean_dir, "pure": pure_dir}


def _existing_vtt_ids(raw_dir: str) -> Set[str]:
    """Return a set of videoIds that already have a VTT in raw/."""
    path = Path(raw_dir)
    if not path.exists():
        return set()
    return {p.stem for p in path.glob("*.vtt")}


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


def download_caption_for_video(video_url: str, output_path: str, prefer_manual: bool = True) -> bool:
    """Attempt to download captions for a single video.

    Parameters
    ----------
    video_url : str
        The YouTube watch URL for the video.
    output_path : str
        Path template where the caption file will be written (without extension).
        The final file will be `{output_path}.vtt`.
    prefer_manual : bool, optional
        If True, try manually uploaded subtitles before automatic captions.

    Returns
    -------
    bool
        True if a caption was downloaded, False otherwise.
    """
    try:
        # Inspect availability first
        with YoutubeDL({"skip_download": True, "quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(video_url, download=False)

        manual_langs = list(info.get("subtitles", {}).keys()) if info.get("subtitles") else []
        auto_langs = list(info.get("automatic_captions", {}).keys()) if info.get("automatic_captions") else []

        # Prefer manual captions
        if prefer_manual and any(lang.startswith("en") for lang in manual_langs):
            ydl_opts = {
                "writesubtitles": True,
                "subtitleslangs": ["en", "en-US", "en-GB"],
                "skip_download": True,
                "subtitlesformat": "vtt",
                "outtmpl": output_path + ".%(ext)s",
                "quiet": True,
                "no_warnings": True,
                "retries": 3,
                "socket_timeout": 20,
            }
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            return True

        # Fallback to automatic captions
        if any(lang.startswith("en") for lang in auto_langs):
            ydl_opts = {
                "writeautomaticsub": True,
                "subtitleslangs": ["en", "en-US", "en-GB"],
                "skip_download": True,
                "subtitlesformat": "vtt",
                "outtmpl": output_path + ".%(ext)s",
                "quiet": True,
                "no_warnings": True,
                "retries": 3,
                "socket_timeout": 20,
            }
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            return True

    except Exception as e:
        logger.error("Caption download error for %s: %s", video_url, e)

    return False


def download_captions() -> None:
    """Idempotent caption download for all videos in the video list CSV.

    - Creates captions/{school}/{raw,clean,pure}/ if missing
    - Skips any video whose {raw}/{videoId}.vtt already exists
    - Tries manual captions first, then auto captions
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

            out_base = os.path.join(raw_dir, vid)
            out_vtt = Path(out_base + ".vtt")

            # --- STOPPER: skip if VTT exists ---
            if vid in existing or out_vtt.exists():
                logger.info("Skip existing captions for %s (%s)", row["title"], vid)
                total_skipped += 1
                continue

            url = row["url"] or f"https://www.youtube.com/watch?v={vid}"
            logger.info("Downloading captions for %s (%s)…", row["title"], vid)
            ok = download_caption_for_video(url, out_base, prefer_manual=True)

            if ok and out_vtt.exists():
                existing.add(vid)
                total_downloaded += 1
                logger.info("Downloaded captions for %s", vid)
            else:
                _append_failed(row, "No captions available or download failed")
                total_failed += 1
                logger.warning("No captions available for %s (%s)", row["title"], vid)

    logger.info(
        "Caption download complete. Downloaded: %d, Skipped existing: %d, Failed: %d",
        total_downloaded, total_skipped, total_failed
    )
