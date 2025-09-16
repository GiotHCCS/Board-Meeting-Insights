"""
Entry point for the community college board meeting pipeline (Step 1).

This script orchestrates the three primary tasks of Step 1:

1. Initialising the project structure and data files.
2. Discovering the most relevant YouTube channel for each configured college.
3. Collecting recent board meeting videos from those channels.
4. Downloading available captions for the collected videos.

Each stage can be enabled or disabled via flags in `config.py`.  The
pipeline logs its progress and writes outputs to the `data/` and
`captions/` directories.
"""

import logging
import os
from datetime import datetime

import config
from youtube_discovery import discover_channels
from video_collection import collect_board_videos
from caption_downloader import download_captions
from asr_processor import process_failed_captions
from transcript_cleaner import process_all_transcripts
from vector_database import create_vector_database
from rag_interface import initialize_rag_system, interactive_chat

def initialize_project_structure() -> None:
    """Create necessary directories and CSV files on first run."""
    config.ensure_directories()
    # Initialise CSVs with headers if they don't exist
    if not os.path.exists(config.SCHOOL_IDS_FILE):
        with open(config.SCHOOL_IDS_FILE, "w", encoding="utf-8") as f:
            f.write("school_name,website,youtube_channel_id,youtube_channel_name,confidence_score\n")
    if not os.path.exists(config.VIDEO_LIST_FILE):
        with open(config.VIDEO_LIST_FILE, "w", encoding="utf-8") as f:
            f.write("college,channelId,videoId,url,title,publishedAt,relevance_score\n")
    if not os.path.exists(config.CAPTION_FAILED_FILE):
        with open(config.CAPTION_FAILED_FILE, "w", encoding="utf-8") as f:
            f.write("college,videoId,url,title,failure_reason\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logger = logging.getLogger("pipeline")
    logger.info("=== Community College Board Meeting Pipeline - Step 1 ===")
    start_time = datetime.now()
    # Step 0: Initialize directories and CSVs
    if getattr(config, "ENABLE_SETUP", True):
        logger.info("Initialising project structure…")
        initialize_project_structure()
    # Step 1: Discover YouTube channels
    if getattr(config, "ENABLE_CHANNEL_DISCOVERY", True):
        discover_channels()
    # Step 2: Collect board meeting videos
    if getattr(config, "ENABLE_VIDEO_COLLECTION", True):
        collect_board_videos()
    # Step 3: Download captions
    if getattr(config, "ENABLE_CAPTION_DOWNLOAD", True):
        download_captions()
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info("Step 1 complete in %s. Ready for Step 2 (ASR processing).", duration)

    # Step 2: ASR processing (generate VTTs for videos without captions)
    if getattr(config, "ENABLE_ASR_PROCESSING", False):
        print("\n=== Step 2: Processing videos without captions (ASR) ===")
        process_failed_captions()
    
    # Step 3: Transcript cleaning (NEW)
    if getattr(config, "ENABLE_TRANSCRIPT_CLEANING", False):
        print("\n=== Step 3: Converting transcripts to clean/pure formats ===")
        process_all_transcripts()

    # Step 4: Vector DB
    if getattr(config, "ENABLE_VECTOR_DB_CREATION", False):
        print("\n=== Step 4: Creating vector database ===")
        create_vector_database()

    # Step 5: RAG
    if getattr(config, "ENABLE_RAG_SYSTEM", False):
        print("\n=== Step 5: Starting RAG query system ===")
        initialize_rag_system()
        interactive_chat()


if __name__ == "__main__":
    main()