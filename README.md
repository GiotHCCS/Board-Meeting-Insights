# Community College Board Meeting Pipeline

This repository contains a **multi-phase pipeline** for discovering, collecting, transcribing, and preparing 
community college board meeting recordings for analysis. The system is designed to improve **transparency and accessibility** 
by creating a searchable archive of board meetings published on official college YouTube channels.  

The pipeline automates the following:

1. **Channel Discovery** – Identify and validate official YouTube channels for each college.  
2. **Video Collection** – Fetch recent board meeting recordings, prioritizing uploads and playlists with keywords like *board*, *trustees*, or *meeting*.  
3. **Caption Downloading** – Collect available closed captions (YouTube or manually uploaded).  
4. **Automatic Transcription** – Run Whisper-based ASR on videos without captions.  
5. **Transcript Processing** – Clean, normalize, and segment transcripts into structured text.  
6. **Vector Database Construction** – Index transcripts into a vector database for question answering and retrieval-augmented generation (RAG).  
7. **Search & Analysis** – Provide search, retrieval, and analysis capabilities across institutions and time ranges.  

---

## Usage Overview

1. **Set up environment** with `uv` (or pip) and install all dependencies.  
2. Place your Google OAuth2 credentials in `client_secret.json` at the project root.  
3. Run the pipeline (`python main.py`) to:  
   - discover official college channels,  
   - fetch board meeting videos,  
   - download captions or generate transcripts,  
   - and process outputs into the vector database.  

On first run, a browser window opens for authentication. A `token.json` is stored for subsequent runs.

---

## Project Layout
```
BM_Full_Pipeline/
├─ pyproject.toml # uv project manifest (name, deps, Python)
├─ uv.lock # uv lockfile (exact dependency versions)
├─ .venv/ # local virtual environment created by uv
│
├─ main.py # pipeline entrypoint (steps 1→5 orchestration)
├─ config.py # all knobs/paths/toggles for every step
│
├─ youtube_discovery.py # Step 1a: find official college YouTube channels
├─ video_collection.py # Step 1b: collect candidate videos (uploads-first, playlists fallback)
├─ caption_downloader.py # Step 1c: download existing captions via yt-dlp
│
├─ asr_processor.py # Step 2: Whisper (faster-whisper) ASR for no-caption videos
│
├─ transcript_cleaner.py # Step 3: VTT → clean(5-word lines) + pure(text) outputs
│
├─ vector_database.py # Step 4: build ChromaDB, chunking + metadata
├─ rag_interface.py # Step 5: retrieval + generation (Ollama), prompts/citations
│
├─ verify_env.py # one-off environment sanity checks (CUDA, FFmpeg, models)
│
├─ client_secret.json # YouTube OAuth client (provided by user)
├─ token.json # YouTube OAuth token cache (auto-created/refreshes)
├─ .env # optional: local overrides (e.g., OLLAMA_HOST)
│
├─ data/ # pipeline CSV outputs and logs
│ ├─ school_IDs.csv # chosen channel per college (discovery result)
│ ├─ video_List.csv # kept video candidates with scores
│ ├─ caption_Failed.csv # videos lacking captions (input to Step 2)
│ ├─ asr_failed.csv # ASR failures (download or transcription issues)
│ ├─ cleaning_failed.csv # Step 3 parse/clean issues
│ └─ video_Rejected.csv # below-threshold or filtered-out candidates
│
├─ captions/ # transcripts by school (slugified)
│ ├─ <school-slug>/
│ │ ├─ raw/ # .vtt from YouTube or ASR (authoritative source)
│ │ ├─ clean/ # .txt (5-word lines with timestamps)
│ │ └─ pure/ # .txt (plain text for embedding)
│ └─ ... # one folder per school
│
├─ temp_audio/ # transient .wav audio for ASR (auto-cleaned if enabled)
│
└─ chromadb_storage/ # persistent ChromaDB database (embeddings + metadata)
```

---

## Dependency Installation

This pipeline uses the [uv](https://github.com/astral-sh/uv) package manager.  

```cmd
cd C:\Projects

:: Initialise project
uv init BM_Full_Pipeline
cd BM_Full_Pipeline

:: Add dependencies to pyproject.toml
uv add \
    google-api-python-client \
    google-auth-oauthlib \
    google-auth-httplib2 \
    yt-dlp \
    pandas \
    requests \
    python-dotenv \
    ctranslate2 \
    faster-whisper \
    chromadb \
    numpy \
    ollama

:: Sync the environment with the lockfile
uv sync
```
After setup, copy your client_secret.json into the root. On first run, authenticate via browser; token.json will be saved.

## Adding Colleges

To expand coverage, update the `COLLEGES` list in `config.py`.  
Each entry requires the **school name** and **website**.  

The system will then:  
- Identify the best-match YouTube channel  
- Search for board meeting uploads or playlists  
- Process videos into transcripts and the vector database  

---

## Current State

- ✅ End-to-end pipeline implemented (discovery → transcripts → vector DB)  
- ✅ Supports **hundreds of colleges** in Texas and beyond  
- ✅ Can process **thousands of videos** while respecting API quotas  
- ✅ Generates structured outputs (`filtered_video_list.csv`, caption logs, transcripts)  
- ✅ Builds a searchable **vector database** for downstream Q&A and analysis  

---

## Example Outputs

- `data/filtered_video_list.csv` – CSV of discovered board meeting videos (college, title, date, URL)  
- `captions/` – Downloaded caption files (.vtt)  
- `transcripts/` – ASR-generated transcripts for videos without captions  
- `vectors/` – Persistent vector store for semantic search  

---

## Future Work

- Web dashboard for transcript search and Q&A  
- Automated updates (daily/weekly fetch)  
- Expansion beyond Texas to all U.S. community colleges  
