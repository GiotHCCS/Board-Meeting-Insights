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

BM_Full_Pipeline/
├── main.py # Orchestrates the full pipeline
├── config.py # Central configuration (colleges, API, paths)
├── src/
│ ├── channel_discovery.py # Find the best YouTube channel for each school
│ ├── video_fetcher.py # Collect videos & playlists (uploads + board mtgs)
│ ├── captions.py # Download available captions with yt-dlp
│ ├── transcription.py # Run Whisper ASR on videos without captions
│ ├── processing.py # Clean & normalize transcript text
│ ├── vector_store.py # Build and manage the transcript vector database
│ └── command_center.py # Shared constants and configs
├── data/ # CSV metadata (filtered videos, logs, mappings)
├── captions/ # Downloaded VTT/SRT captions
├── transcripts/ # Whisper-generated transcripts
├── vectors/ # Persisted vector database
└── client_secret.json # OAuth2 credentials (not tracked in Git)


---

## Dependency Installation

This pipeline uses the [uv](https://github.com/astral-sh/uv) package manager.  

```cmd
cd C:\Projects

:: Initialise project
uv init BM_Full_Pipeline
cd BM_Full_Pipeline

:: Install dependencies
uv pip install \
    google-api-python-client \
    google-auth-oauthlib \
    google-auth-httplib2 \
    yt-dlp \
    pandas \
    requests \
    python-dotenv \
    ctranslate2 \
    openai-whisper \
    chromadb \
    numpy

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
