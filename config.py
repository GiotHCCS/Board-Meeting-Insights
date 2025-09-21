"""
Configuration for the Community College Board Meeting pipeline.

Usage:
  # Run full pipeline
  cd C:\Projects\BM_Full_Pipeline
  uv run python main.py

  # Launch RAG chat only (uses existing ChromaDB)
  uv run python -c "from rag_interface import interactive_chat; interactive_chat()"

  # run school video Table of Contents overview in LLM
  /overview school="El Paso Community College" vid={videoID}
"""

import os

# ===== Paths =====
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
CAPTIONS_DIR: str = os.path.join(PROJECT_ROOT, "captions")

# ===== YouTube OAuth (client_secret.json must be in project root) =====
YOUTUBE_API_KEY_FILE: str = "client_secret.json"
TOKEN_FILE: str = "token.json"
YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]

# ===== Colleges to track =====
# Website is used as a signal when ranking candidate channels.
COLLEGES = [
    {"name": "Houston City College", "website": "https://www.hccs.edu"},
    {"name": "Lone Star College", "website": "https://www.lonestar.edu/"},
    {"name": "Central Texas College", "website": "https://www.ctcd.edu/"},
    {"name": "El Paso Community College", "website": "https://www.epcc.edu/"},
    {"name": "MT. San Antonio College", "website": "https://www.mtsac.edu/"},
]

# ===== Discovery / collection parameters =====
MAX_VIDEOS_PER_CHANNEL: int = 3      # Target kept videos per channel (after scoring/filters)
SEARCH_CUTOFF_MONTHS: int = 6        # Only consider uploads within the last N months
MIN_RELEVANCE_SCORE: int = 2         # Minimum score for a video to be kept
MIN_MEETING_SECONDS: int = 45 * 60   # Hard minimum duration for meetings (e.g., 45 minutes)

# Output files
SCHOOL_IDS_FILE: str   = os.path.join(DATA_DIR, "school_IDs.csv")
VIDEO_LIST_FILE: str   = os.path.join(DATA_DIR, "video_List.csv")
CAPTION_FAILED_FILE: str = os.path.join(DATA_DIR, "caption_Failed.csv")

# ===== Whisper (ASR) =====
WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "cuda"          # "cuda" or "cpu"
WHISPER_COMPUTE_TYPE = "float16" # "float16" is ideal on RTX 5090
WHISPER_BEAM_SIZE = 1

# ===== Source toggles for video collection =====
# True = ENABLE that source. False = skip the calls (saves quota/time).
ENABLE_SOURCE_UPLOADS = False          # Pull recent items from the channel's "uploads" playlist T
ENABLE_SOURCE_CHANNEL_SEARCH = False  # Channel-scoped search("board", "trustees", etc.) F
ENABLE_SOURCE_PLAYLISTS = False        # Scan channel playlists with board/trustee/meeting in the title T

# Live/privacy filters (applied BEFORE scoring)
EXCLUDE_LIVE_AND_UPCOMING = False   # Drop live or scheduled/upcoming items T
EXCLUDE_NON_PUBLIC = False          # Drop private/unlisted if the API exposes privacy status T

# ===== Step toggles =====
ENABLE_SETUP = False                # Ensure folders exist T
ENABLE_CHANNEL_DISCOVERY = False    # Resolve channel IDs from college names/websites
ENABLE_VIDEO_COLLECTION = False      # Fetch + score videos → data/video_List.csv T
ENABLE_CAPTION_DOWNLOAD = False      # Download captions with yt-dlp for kept videos T

# ===== ASR (Step 2) =====
ENABLE_ASR_PROCESSING = True
ASR_ONLY_MISSING = True            # Skip if raw VTT already present
ENABLE_ASR_DOWNLOAD = True         # Download audio for Whisper
ENABLE_ASR_TRANSCRIBE = True
ENABLE_PROGRESS_BARS = True
LOG_ASR_RESULTS = True

ASR_BATCH_SIZE = 3
MAX_VIDEO_LENGTH_MINUTES = 240
TEMP_AUDIO_DIR = os.path.join(PROJECT_ROOT, "temp_audio")
CLEANUP_TEMP_FILES = True

# ===== Cleaning (Step 3) =====
ENABLE_TRANSCRIPT_CLEANING = True

# Output formatting
CLEAN_TIME_STYLE = "auto"        # "auto" → HH if ≥1h, else MM
CLEAN_TIME_INCLUDE_MS = False
CLEAN_MAX_SPAN_SECONDS = None    # e.g., 15 to clamp each line’s span

ASR_FAILED_FILE = os.path.join(DATA_DIR, "asr_failed.csv")

# Processing options
REPROCESS_EXISTING_FILES = True     # Skip if clean & pure already exist
WORDS_PER_CLEAN_SEGMENT = 5
PURE_TEXT_PARAGRAPH_LENGTH = 60

# Text cleanup
EXPAND_CONTRACTIONS = True
NORMALIZE_NUMBERS = True             # Spell 0–20; keep larger as digits
REMOVE_FILLER_WORDS = True
COLLAPSE_REPETITIONS = True

# Validation
ENABLE_OUTPUT_VALIDATION = True
MIN_PURE_TEXT_LENGTH = 100
MAX_PROCESSING_TIME_PER_FILE = 30    # Soft budget per file (seconds)

CLEANING_FAILED_FILE = os.path.join(DATA_DIR, "cleaning_failed.csv")

# ===== Vector DB + RAG (Steps 4–5) =====
ENABLE_VECTOR_DB_CREATION = True
ENABLE_RAG_SYSTEM = True

# ChromaDB
CHROMADB_PERSIST_DIR = os.path.join(PROJECT_ROOT, "chromadb_storage")
COLLECTION_NAME = "board_meeting_transcripts"
REBUILD_VECTOR_DB = True  # True = drop & rebuild collection on next run

# Ollama (embeddings + LLM)
OLLAMA_HOST = "http://127.0.0.1:11434"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIMENSIONS = 768
LLM_MODEL = "llama3.1:8b"

# Chunking (Step 4)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 300
MAX_CHUNK_SIZE = 2000
CHUNK_BY_SENTENCES = True
PRESERVE_PARAGRAPH_BREAKS = True

EMBEDDING_BATCH_SIZE = 50

# Timestamp correlation & metadata
ENABLE_METADATA_FILTERING = True
CHUNK_TIMESTAMP_TOPK = 5
CHUNK_TIMESTAMP_MIN_RATIO = 70  # fuzzy ratio 0..100

# ===== RAG Retrieval =====
RAG_RETRIEVAL_SETTINGS = {
    "max_chunks": 8,
    "similarity_threshold": 0.0,      # We'll filter post-hoc; keep recall high
    "enable_reranking": True,
    "enable_temporal_boosting": True, # Prefer more recent meetings when scores tie
    "enable_school_filtering": False, # Allow cross-school by default unless caller filters

    # Diversity heuristics (1st pass)
    "prefer_source_diversity": True,  # Prefer covering multiple videos before piling on one
    "max_per_video": 1,               # Cap per-video chunks during initial selection
    "candidate_multiplier": 4,        # Pull this many × max_chunks before diversification
}

# ===== RAG Generation =====
RAG_GENERATION_SETTINGS = {
    "temperature": 0.3,
    "max_response_tokens": 800,
    "enable_citations": True,
    "citation_style": "detailed",      # "simple" | "detailed"
    "enable_timestamp_references": True,
    "confidence_threshold": 0.6,
}

# ===== Single-meeting overview mode =====
VIDEO_OVERVIEW_CONFIG = {
    "max_chunks_for_overview": 22,     # Total chunks fed to LLM
    "priority_chunk_fraction": 0.35,   # Fraction of earliest chunks to prioritize
    "sample_rest": True,               # Sample the remainder for coverage
    "overview_temperature": 0.2,
    "overview_max_tokens": 1400,
}

# ===== Prompts =====
RAG_PROMPT_TEMPLATES = {
    # Multi-meeting analysis
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

    # Single-meeting overview
    "video_overview_system_prompt": (
        "You are analyzing ONE board meeting. All [S#] excerpts are segments from the SAME video.\n"
        "Goal: reconstruct the meeting flow and produce a structured, chronological overview.\n\n"
        "HARD RULES:\n"
        "- Treat all sources as one meeting; do NOT bring in external info.\n"
        "- If timestamps are missing, keep order by [S#].\n"
        "- Put an inline [S#] after EVERY factual claim, time, number, decision, quote.\n"
        "- Distinguish DISCUSSION vs. DECISION (approved/denied/tabled) when present.\n"
        "- If something is unclear or missing, say so explicitly.\n\n"
        "BOARD MEETING PATTERNS TO DETECT:\n"
        "- Opening protocol: call to order, roll call, minutes approval.\n"
        "- Agenda progression cues: 'next item', 'moving to', 'agenda item'.\n"
        "- Budget/finance, personnel actions, policy updates, facilities, public comment.\n"
        "- Voting outcomes and dissent.\n"
    ),
    
    "video_overview_prompt": (
        "You are creating a comprehensive overview of a SINGLE board meeting transcript.\n"
        "All sources [S#] are time-ordered segments from the SAME meeting.\n\n"
        "PRIORITIES:\n"
        "- Focus extra attention on EARLY segments for agenda structure and announcements.\n"
        "- Keep a strict chronological narrative (by time or by [S#] order if time absent).\n"
        "- Cite [S#] after every fact/time/number/decision.\n\n"
        "Meeting Segments:\n{context_chunks}\n\n"
        "User Request: {user_question}\n\n"
        "OUTPUT FORMAT:\n"
        "1) Meeting Overview\n"
        "   - Meeting type, date, notable attendees (if stated) [S#]\n"
        "   - One-paragraph summary of main topics and outcomes [S#]\n\n"
        "2) Table of Contents (Chronological Agenda)\n"
        "   1. <Agenda item> — <one-line summary> [S#]\n"
        "   2. <Agenda item> — <one-line summary> [S#]\n"
        "   (Continue in order)\n\n"
        "3) Timeline by Topic (minute:minute → topic)\n"
        "   [MM:SS - MM:SS] <Topic/Agenda item>\n"
        "   - Key points [S#]\n"
        "   - Decision/outcome (if any) [S#]\n"
        "   (Repeat for each major segment)\n\n"
        "4) Key Decisions & Actions\n"
        "   - <Decision>: <details> [S#]\n"
        "   - <Action Items>: <assignee / due date if stated> [S#]\n\n"
        "5) Gaps or Unknowns\n"
        "   - <What is missing/unclear from the provided segments>\n\n"
        "Produce the overview now."
    ),
    "focused_video_overview_prompt": (
        "You are analyzing a SINGLE board meeting for a focused overview of specific topics.\n"
        "All sources [S#] are segments from the SAME meeting.\n\n"
        "Meeting Segments:\n{context_chunks}\n\n"
        "Focus Area: {user_question}\n\n"
        "OUTPUT:\n"
        "1) Meeting Context: type, date, relevant attendees [S#]\n"
        "2) Focus Topic Narrative: how the topic surfaced, evolved, and concluded [S#]\n"
        "3) Key Discussion Points: main arguments, concerns, data [S#]\n"
        "4) Decisions & Outcomes: status (approved/denied/tabled/deferred) [S#]\n"
        "5) Timeline Pointers: when in the meeting the topic appears [S#]\n"
        "6) Open Questions/Next Steps: what remains unresolved [S#]\n\n"
        "Answer in a structured format."
    ),
}

# ===== Windows helper paths (optional) =====
CUDA_BIN_DIRS = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin",
    r"C:\Program Files\NVIDIA\CUDNN\v9.13\bin\12.9",
]
FFMPEG_DIR = r"C:\Users\ermeg\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin"

# ===== Keyword sets =====
BOARD_MEETING_KEYWORDS = [
    "board",
    "trustee", "trustees",
    "meeting", "meetings",
    "board meeting",
    "board of trustees",
    "regular meeting",
    "special meeting",
    "committee meeting",
    "budget workshop",
    "board workshop",
]

PRIORITY_PHRASES = [
    "board of trustees",
    "board meeting",
    "trustees meeting",
    "board of trustees meeting",
    "regular board meeting",
]

# Add/adjust to prune obvious non-meeting series (customize per college if needed)
EXCLUDE_TERMS = [
    "up to the minute", "up to the minute live",
    "unscripted", "the topic", "veterans’ voice", "veterans' voice",
    "#shorts",
    "conference room", "conference rooms",
    "community partnership meeting",
    "graduation", "commencement",
]

def ensure_directories() -> None:
    """Create required output directories."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CAPTIONS_DIR, exist_ok=True)
