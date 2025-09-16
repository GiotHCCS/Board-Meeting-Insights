"""
Configuration module for the community college board meeting pipeline.

This file centralises all tunable settings and file paths used by the
application.  Adjust these values to customise the behaviour of the
discovery, video collection and caption downloading steps.

The `COLLEGES` list can be expanded to include additional institutions.
Each entry must contain a `name` and its corresponding `website`.  The
weighted channel discovery logic in `youtube_discovery.py` relies on
these fields to score candidate channels.

commands to run full program:
cd C:\Projects\BM_Full_Pipeline
uv run python main.py

command to run llm with existing data:
uv run python -c "from rag_interface import interactive_chat; interactive_chat()"
"""

import os

# Project root and derived paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
CAPTIONS_DIR: str = os.path.join(PROJECT_ROOT, "captions")

# YouTube API credentials and token storage
# `client_secret.json` must be provided by the user and placed in the project root.
YOUTUBE_API_KEY_FILE: str = "client_secret.json"
TOKEN_FILE: str = "token.json"

# OAuth scopes required for read‑only YouTube access
YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]

# List of colleges to search for.  Each entry contains the official
# institution name and its primary website.  The website is used to
# validate channels by matching against the channel description.
COLLEGES = [
    {"name": "Houston City College", "website": "https://www.hccs.edu"},
    {"name": "Lone Star College", "website": "https://www.lonestar.edu/"},
    {"name": "Central Texas College", "website": "https://www.ctcd.edu/"},
    {"name": "El Paso Community College", "website": "https://www.epcc.edu/"},
    {"name": "MT. San Antonio College", "website": "https://www.mtsac.edu/"},
]

# Search parameters for video discovery
MAX_VIDEOS_PER_CHANNEL: int = 3  # maximum number of videos to fetch per channel
SEARCH_CUTOFF_MONTHS: int = 6  # how many months back to search for videos
MIN_RELEVANCE_SCORE: int = 2  # minimum score a video must achieve to be kept

# File locations for data outputs
SCHOOL_IDS_FILE: str = os.path.join(DATA_DIR, "school_IDs.csv")
VIDEO_LIST_FILE: str = os.path.join(DATA_DIR, "video_List.csv")
CAPTION_FAILED_FILE: str = os.path.join(DATA_DIR, "caption_Failed.csv")

# Whisper / GPU
WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "cuda"               # "cuda" | "cpu"
WHISPER_COMPUTE_TYPE = "float16"      # "float16" on RTX 5090
WHISPER_BEAM_SIZE = 1

# ---------- Source toggles (Step 1 video discovery) ----------
ENABLE_SOURCE_UPLOADS = True        # don’t call channel uploads
ENABLE_SOURCE_CHANNEL_SEARCH = False # don’t call channel search
ENABLE_SOURCE_PLAYLISTS = True      # don’t call playlist fetches

# ---------- Step control flags ----------
ENABLE_SETUP = True                   # still fine, just ensures dirs exist
ENABLE_CHANNEL_DISCOVERY = False      # don’t run YouTube channel discovery
ENABLE_VIDEO_COLLECTION = True       # don’t run video fetching/scoring
ENABLE_CAPTION_DOWNLOAD = True       # don’t call yt-dlp for captions

EXCLUDE_LIVE_AND_UPCOMING = True      # Exclude live or scheduled/upcoming streams from video list
EXCLUDE_NON_PUBLIC = True            # (Optional) also drop non-public uploads


# ---------- ASR Step (Step 2) ----------
ENABLE_ASR_PROCESSING = True          # run Whisper ASR
ASR_ONLY_MISSING = True               # skip if raw/{videoId}.vtt already exists
ENABLE_ASR_DOWNLOAD = True            # True = download audio with yt-dlp
                                      # False = only use local temp_audio/{videoId}.wav
ENABLE_ASR_TRANSCRIBE = True          # run Whisper
ENABLE_PROGRESS_BARS = True
LOG_ASR_RESULTS = True

# Limits & temp
ASR_BATCH_SIZE = 3
MAX_VIDEO_LENGTH_MINUTES = 240
TEMP_AUDIO_DIR = os.path.join(PROJECT_ROOT, "temp_audio")
CLEANUP_TEMP_FILES = True

# Clean timestamp formatting for Step 3 ("auto" | "hh" | "mm")
CLEAN_TIME_STYLE = "auto"        # auto = HH if video ≥1h, else MM
CLEAN_TIME_INCLUDE_MS = False    # True to show .mmm
CLEAN_MAX_SPAN_SECONDS = None    # e.g., 15 to clamp each line's span

# Files
ASR_FAILED_FILE = os.path.join(DATA_DIR, "asr_failed.csv")

# ---------- Step 3: Transcript Cleaning ----------
ENABLE_TRANSCRIPT_CLEANING = True

# Processing options
REPROCESS_EXISTING_FILES = False     # Skip if clean & pure already exist
WORDS_PER_CLEAN_SEGMENT = 5
PURE_TEXT_PARAGRAPH_LENGTH = 60

# Text cleaning options
EXPAND_CONTRACTIONS = True
NORMALIZE_NUMBERS = True             # spell out 0–20, keep larger as digits
REMOVE_FILLER_WORDS = True           # remove repetitive filler sequences
COLLAPSE_REPETITIONS = True          # collapse repeated tokens ("the the" -> "the")

# Validation
ENABLE_OUTPUT_VALIDATION = True
MIN_PURE_TEXT_LENGTH = 100           # min words in pure text
MAX_PROCESSING_TIME_PER_FILE = 30    # seconds (soft indicator, not a hard timeout)

# Files
CLEANING_FAILED_FILE = os.path.join(DATA_DIR, "cleaning_failed.csv")

# ---------- Steps 4–5: Vector DB + RAG ----------
ENABLE_VECTOR_DB_CREATION = True
ENABLE_RAG_SYSTEM = True

# ChromaDB
CHROMADB_PERSIST_DIR = os.path.join(PROJECT_ROOT, "chromadb_storage")
COLLECTION_NAME = "board_meeting_transcripts"
REBUILD_VECTOR_DB = False  # True = drop & rebuild collection

# Embeddings / LLM via Ollama
OLLAMA_HOST = "http://127.0.0.1:11434"
EMBEDDING_MODEL = "nomic-embed-text"   # via Ollama
EMBEDDING_DIMENSIONS = 768             # nomic-embed-text dims
LLM_MODEL = "llama3.1:8b"              # generation model via Ollama

# Chunking
CHUNK_SIZE = 1000          # characters
CHUNK_OVERLAP = 200        # characters
MIN_CHUNK_SIZE = 300
MAX_CHUNK_SIZE = 2000
CHUNK_BY_SENTENCES = True
PRESERVE_PARAGRAPH_BREAKS = True

# Embedding batches
EMBEDDING_BATCH_SIZE = 50

# Metadata / correlation
ENABLE_METADATA_FILTERING = True
CHUNK_TIMESTAMP_TOPK = 5
CHUNK_TIMESTAMP_MIN_RATIO = 70  # fuzzy match threshold [0..100]

# RAG Retrieval
RAG_RETRIEVAL_SETTINGS = {
    "max_chunks": 8,
    "similarity_threshold": 0.0,     # cosine: lower is okay, we filter post-hoc
    "enable_reranking": True,
    "enable_temporal_boosting": True,
    "enable_school_filtering": False,
}

# RAG Generation
RAG_GENERATION_SETTINGS = {
    "temperature": 0.3,
    "max_response_tokens": 800,
    "enable_citations": True,
    "citation_style": "detailed",  # "simple" | "detailed"
    "enable_timestamp_references": True,
    "confidence_threshold": 0.6,
}

# RAG chunk diversity
RAG_RETRIEVAL_SETTINGS.update({
    "prefer_source_diversity": True,  # try to cover multiple videos first
    "max_per_video": 1,               # at most 1 chunk per video in the first pass
    "candidate_multiplier": 4,        # how big a pool to pull before diversifying
})

RAG_PROMPT_TEMPLATES = {
    "system_prompt": (
        "You are an expert analyst of community college board meeting transcripts.\n"
        "You will receive EXCERPTS from MULTIPLE different meetings, dates, and times.\n"
        "Each excerpt is an independent source - DO NOT assume continuity between sources.\n\n"
        
        "CORE RULES:\n"
        "- NEVER merge claims across sources unless explicitly noting they're from different meetings\n"
        "- If sources conflict, present each separately with clear attribution\n"
        "- Include inline citation [S#] after EVERY factual claim, number, date, or quote\n"
        "- Only use information present in sources; explicitly state what's missing\n"
        "- Prefer structured answers with bullets when covering multiple sources\n\n"
        
        "BOARD MEETING EXPERTISE:\n"
        "- Recognize common agenda items: budget approvals, policy changes, personnel decisions\n"
        "- Understand meeting types: regular, special, budget workshops, committee meetings\n"
        "- Identify decision outcomes: approved, denied, tabled, referred to committee\n"
        "- Distinguish between discussion and final action\n"
        "- Note when items are first reading vs. final approval\n"
        "- Track voting patterns and dissenting opinions when mentioned"
    ),

    "context_prompt": (
        "You have excerpts from board meeting transcripts, each labeled [S#]. These come from "
        "DIFFERENT meetings and/or time periods - treat each as independent.\n\n"
        
        "Meeting Sources:\n{context_chunks}\n\n"
        
        "Question: {user_question}\n\n"
        
        "ANALYSIS FRAMEWORK:\n"
        "1. IDENTIFY what each source says about the question\n"
        "2. NOTE any contradictions or different perspectives between sources\n"
        "3. DISTINGUISH between discussion and actual decisions/votes\n"
        "4. SPECIFY meeting types and contexts when relevant\n"
        "5. CLARIFY temporal relationships (before/after) when discussing multiple meetings\n\n"
        
        "RESPONSE STRUCTURE:\n"
        "- Lead with direct answer if clear from sources\n"
        "- Use bullet points for multi-source information\n"
        "- Cite every claim with [S#]\n"
        "- End with summary of what's missing if question not fully answered\n"
        "- For financial/numerical data, always include the specific meeting date [S#]\n"
        "- For policy changes, note if it's discussion vs. final approval [S#]\n\n"
        
        "Answer:"
    ),
    
    "financial_query_prompt": (
        "FINANCIAL ANALYSIS FOCUS:\n"
        "You're analyzing budget and financial information from board meetings.\n\n"
        
        "Meeting Sources:\n{context_chunks}\n\n"
        "Question: {user_question}\n\n"
        
        "For financial queries, specifically:\n"
        "- Quote exact dollar amounts with [S#] citations\n"
        "- Note if amounts are proposed, approved, or actual expenditures\n"
        "- Identify budget categories (operations, capital, personnel, etc.)\n"
        "- Distinguish between annual budgets vs. specific allocations\n"
        "- Flag any budget amendments or revisions mentioned\n"
        "- Compare figures across different meetings if multiple sources\n\n"
        
        "Answer:"
    ),
    
    "policy_query_prompt": (
        "POLICY ANALYSIS FOCUS:\n"
        "You're analyzing policy discussions and decisions from board meetings.\n\n"
        
        "Meeting Sources:\n{context_chunks}\n\n"
        "Question: {user_question}\n\n"
        
        "For policy queries, specifically:\n"
        "- Identify if policy is new, revised, or under review [S#]\n"
        "- Note reading status (first reading, second reading, final approval) [S#]\n"
        "- Mention any public comment or opposition discussed [S#]\n"
        "- Track policy through multiple meetings if sources span time [S#]\n"
        "- Distinguish between board discussion and administrative recommendation [S#]\n"
        "- Note effective dates and implementation timelines mentioned [S#]\n\n"
        
        "Answer:"
    ),
    
    "personnel_query_prompt": (
        "PERSONNEL ANALYSIS FOCUS:\n"
        "You're analyzing hiring, staffing, and personnel decisions from board meetings.\n\n"
        
        "Meeting Sources:\n{context_chunks}\n\n"
        "Question: {user_question}\n\n"
        
        "For personnel queries, specifically:\n"
        "- Note position titles and departments exactly as stated [S#]\n"
        "- Distinguish between recommendations and actual approvals [S#]\n"
        "- Include salary/compensation information if mentioned [S#]\n"
        "- Identify if positions are new, replacement, or restructured [S#]\n"
        "- Note any discussion of qualifications or selection process [S#]\n"
        "- Respect privacy - don't speculate about individuals not explicitly discussed [S#]\n\n"
        
        "Answer:"
    ),
    
    "temporal_query_prompt": (
        "TEMPORAL ANALYSIS FOCUS:\n"
        "You're tracking events, decisions, or changes over time across multiple meetings.\n\n"
        
        "Meeting Sources:\n{context_chunks}\n\n"
        "Question: {user_question}\n\n"
        
        "For timeline queries, specifically:\n"
        "- Organize information chronologically by meeting date\n"
        "- Note progression of issues across multiple meetings [S#]\n"
        "- Identify when items were first introduced vs. final action [S#]\n"
        "- Track status changes (discussed → tabled → approved) [S#]\n"
        "- Note any deadlines or target dates mentioned [S#]\n"
        "- Clarify gaps in timeline based on available sources [S#]\n\n"
        
        "Answer:"
    ),
    
    "comparative_query_prompt": (
        "COMPARATIVE ANALYSIS FOCUS:\n"
        "You're comparing information, decisions, or approaches across different contexts.\n\n"
        
        "Meeting Sources:\n{context_chunks}\n\n"
        "Question: {user_question}\n\n"
        
        "For comparison queries, specifically:\n"
        "- Present each comparison point with clear source attribution [S#]\n"
        "- Organize as side-by-side comparison when possible\n"
        "- Note contextual differences (different meetings, time periods, circumstances) [S#]\n"
        "- Identify similarities and differences explicitly [S#]\n"
        "- Avoid assuming causation between different sources [S#]\n"
        "- State limitations if comparison is incomplete due to missing data [S#]\n\n"
        
        "Answer:"
    ),
}

# Tooling (optional PATH injection on Windows)
CUDA_BIN_DIRS = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin",
    r"C:\Program Files\NVIDIA\CUDNN\v9.13\bin\12.9",
]
FFMPEG_DIR = r"C:\Users\ermeg\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin"

# Keywords used to identify board meeting videos.  These are used in
# `video_collection.py` to score videos based on their titles and
# descriptions.
BOARD_MEETING_KEYWORDS = [
    "board",
    "trustee",
    "trustees",
    "meeting",
    "meetings",
    "board meeting",
    "board of trustees",
    "regular meeting",
    "special meeting",
    "committee meeting",
    "budget workshop",
    "board workshop",
]

# Phrases that imply a high likelihood of a board meeting.  These receive
# additional points during scoring.
PRIORITY_PHRASES = [
    "board of trustees",
    "board meeting",
    "trustees meeting",
    "board of trustees meeting",
    "regular board meeting",
]

# Terms that should exclude a video from consideration.  Any video whose
# title or description contains one of these phrases is ignored.
EXCLUDE_TERMS = [
    "veterans' voice",
    "#shorts",
    "conference room",
    "community partnership meeting",
    "graduation",
    "commencement",
]

def ensure_directories() -> None:
    """Ensure that the data and captions directories exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CAPTIONS_DIR, exist_ok=True)