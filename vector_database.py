from __future__ import annotations
import os, glob, logging, math
from pathlib import Path
from typing import Dict, List, Tuple, Any
import re

import pandas as pd
from dateutil import parser as dtparser
from fuzzywuzzy import fuzz

import chromadb
from chromadb.config import Settings

try:
    from ollama import Client as OllamaClient
except Exception:
    OllamaClient = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --------------------- helpers ---------------------
def slugify(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", value)).strip("_").lower()

def _ensure_dirs():
    os.makedirs(config.CHROMADB_PERSIST_DIR, exist_ok=True)

def _load_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str).fillna("")

# Meeting type keywords
MEETING_TYPES = {
    "regular_meeting": ["regular", "monthly", "board meeting", "board of trustees meeting"],
    "special_meeting": ["special", "emergency", "called meeting"],
    "budget_workshop": ["budget", "workshop", "financial"],
    "committee_meeting": ["committee", "subcommittee"],
    "work_session": ["work session", "study session", "planning"]
}

def _classify_meeting_type(title: str) -> str:
    t = (title or "").lower()
    best = "regular_meeting"
    for k, kws in MEETING_TYPES.items():
        if any(kw in t for kw in kws):
            best = k
    return best

_DATE_PAT = re.compile(
    r"(?:(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|[A-Za-z]+\s+\d{1,2},\s*\d{4})",
    re.I
)

def _extract_date_from_title(title: str) -> str | None:
    m = _DATE_PAT.search(title or "")
    if not m:
        return None
    try:
        dt = dtparser.parse(m.group(0), fuzzy=True)
        return dt.date().isoformat()
    except Exception:
        return None

def _duration_estimate_from_clean(clean_path: Path) -> str | None:
    # Estimate by last timestamp on the last line
    try:
        lines = clean_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        if not lines:
            return None
        last = lines[-1]
        m = re.search(r"\[(.+?)\s*-\s*(.+?)\]", last)
        if not m:
            return None
        end = m.group(2)
        # parse MM:SS or HH:MM:SS
        parts = [int(x) for x in end.split(":")]
        seconds = 0
        if len(parts) == 2:
            seconds = parts[0]*60 + parts[1]
        elif len(parts) == 3:
            seconds = parts[0]*3600 + parts[1]*60 + parts[2]
        minutes = max(1, round(seconds/60))
        return f"{minutes}min"
    except Exception:
        return None

def initialize_chromadb():
    _ensure_dirs()
    client = chromadb.PersistentClient(
        path=config.CHROMADB_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    if getattr(config, "REBUILD_VECTOR_DB", False):
        try:
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass
    coll = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # cosine works well with most embeddings
    )
    return coll

def load_video_metadata() -> Dict[str, Dict]:
    df = _load_csv_safe(config.VIDEO_LIST_FILE)
    if df.empty:
        return {}
    meta: Dict[str, Dict] = {}

    # Preload caption_Failed to infer ASR vs YT source (best-effort)
    cf = _load_csv_safe(config.CAPTION_FAILED_FILE)
    failed_ids = set(cf["videoId"].tolist()) if not cf.empty else set()

    for _, r in df.iterrows():
        vid = r.get("videoId", "")
        if not vid:
            continue
        title = r.get("title", "")
        pub = r.get("publishedAt", "")  # ISO
        try:
            pub_date = dtparser.parse(pub).date().isoformat() if pub else ""
            year = int(pub_date[:4]) if pub_date else None
            month = int(pub_date[5:7]) if pub_date else None
        except Exception:
            pub_date, year, month = "", None, None

        meeting_date = _extract_date_from_title(title) or pub_date
        mt = _classify_meeting_type(title)
        college = r.get("college", "")
        school_slug = slugify(college)
        url = r.get("url", f"https://www.youtube.com/watch?v={vid}")

        transcript_source = "asr_generated" if vid in failed_ids else "youtube_captions"

        meta[vid] = {
            "video_id": vid,
            "school_name": college,
            "school_slug": school_slug,
            "video_title": title,
            "video_url": url,
            "published_date": pub_date,
            "published_year": year,
            "published_month": month,
            "meeting_type": mt,
            "meeting_date": meeting_date,
            "video_duration_estimate": None,  # filled later if we have clean file
            "transcript_source": transcript_source,
        }
    return meta

# --------------- chunking ----------------
_SENT_SPLIT = re.compile(r"(?<=[\.\?\!])\s+")
def _split_sentences(text: str) -> List[str]:
    # simple sentence split; your transcripts are mostly full stops
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]

def _smart_chunks(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    size = int(getattr(config, "CHUNK_SIZE", 1000))
    overlap = int(getattr(config, "CHUNK_OVERLAP", 200))
    min_sz = int(getattr(config, "MIN_CHUNK_SIZE", 300))
    max_sz = int(getattr(config, "MAX_CHUNK_SIZE", 2000))
    by_sents = bool(getattr(config, "CHUNK_BY_SENTENCES", True))

    if not by_sents:
        # pure char sliding window
        chunks = []
        i = 0
        while i < len(text):
            chunk = text[i:i+size]
            if len(chunk) >= min_sz:
                chunks.append(chunk)
            i += max(1, size - overlap)
        return chunks

    sents = _split_sentences(text)
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        if cur_len + len(s) + 1 <= size or cur_len == 0:
            cur.append(s); cur_len += len(s) + 1
        else:
            blob = " ".join(cur)
            if len(blob) >= min_sz:
                chunks.append(blob[:max_sz])
            # overlap by sentences approx via last N chars
            if overlap > 0 and cur:
                keep = " ".join(cur)[max(0, len(" ".join(cur)) - overlap):]
                cur = [keep] if keep else []
                cur_len = len(keep)
            else:
                cur, cur_len = [], 0
            cur.append(s); cur_len += len(s) + 1
    if cur:
        blob = " ".join(cur)
        if len(blob) >= min_sz:
            chunks.append(blob[:max_sz])
    return chunks

# --------------- clean correlation (timestamps) ---------------
_CLEAN_LINE = re.compile(r"^\[(?P<start>[^\]]+)\s*-\s*(?P<end>[^\]]+)\]\s+(?P<text>.+)$")

def _parse_clean_lines(clean_path: Path) -> List[Dict[str, Any]]:
    if not clean_path.exists():
        return []
    out = []
    for ln in clean_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _CLEAN_LINE.match(ln.strip())
        if not m:
            continue
        out.append({
            "start": m.group("start"),
            "end": m.group("end"),
            "text": m.group("text").lower().strip()
        })
    return out

def _estimate_chunk_timestamps(chunk_text: str, clean_segments: List[Dict[str, Any]]) -> Tuple[str|None, str|None]:
    if not chunk_text or not clean_segments:
        return None, None
    # score each clean segment by fuzzy partial ratio against chunk_text
    scores = []
    for seg in clean_segments:
        s = fuzz.partial_ratio(seg["text"], chunk_text.lower())
        scores.append((s, seg["start"], seg["end"]))
    scores.sort(key=lambda x: x[0], reverse=True)
    min_ratio = int(getattr(config, "CHUNK_TIMESTAMP_MIN_RATIO", 70))
    topk = int(getattr(config, "CHUNK_TIMESTAMP_TOPK", 5))
    top = [t for t in scores[:topk] if t[0] >= min_ratio]
    if not top:
        return None, None
    # use earliest start and latest end among top matches
    starts = [t[1] for t in top]
    ends   = [t[2] for t in top]
    return (min(starts), max(ends))

# --------------- embeddings ---------------
def _ollama_client():
    if OllamaClient is None:
        return None
    try:
        return OllamaClient(host=config.OLLAMA_HOST)
    except Exception:
        return None

def _embed_with_ollama(client, texts: List[str]) -> List[List[float]]:
    # call in small batches to be safe
    vecs: List[List[float]] = []
    for i in range(0, len(texts), max(1, config.EMBEDDING_BATCH_SIZE)):
        batch = texts[i:i+config.EMBEDDING_BATCH_SIZE]
        for t in batch:
            resp = client.embeddings(model=config.EMBEDDING_MODEL, prompt=t)
            vecs.append(resp["embedding"])
    return vecs

def _embed_with_sentence_transformers(texts: List[str]) -> List[List[float]]:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not available and Ollama unreachable.")
    # Using a robust default if user didn't set one; MiniLM dims=384
    model_name = "all-MiniLM-L6-v2"
    st = SentenceTransformer(model_name)
    embs = st.encode(texts, normalize_embeddings=True, convert_to_numpy=False, batch_size=32, show_progress_bar=False)
    return [list(map(float, v)) for v in embs]

# --------------- main build ---------------
def process_pure_texts_to_chunks(video_meta: Dict[str, Dict]) -> List[Dict]:
    chunks: List[Dict] = []
    cap_root = Path(config.CAPTIONS_DIR)
    for school_dir in cap_root.iterdir() if cap_root.exists() else []:
        if not school_dir.is_dir():
            continue
        pure_dir = school_dir / "pure"
        clean_dir = school_dir / "clean"
        if not pure_dir.exists():
            continue
        for txt_path in pure_dir.glob("*.txt"):
            vid = txt_path.stem
            txt = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not txt:
                continue
            clean_path = clean_dir / f"{vid}.txt"
            clean_segments = _parse_clean_lines(clean_path) if clean_path.exists() else []

            vm = video_meta.get(vid, {
                "video_id": vid,
                "school_name": school_dir.name.replace("-", " ").title(),
                "school_slug": school_dir.name,
                "video_title": "",
                "video_url": f"https://www.youtube.com/watch?v={vid}",
                "published_date": "",
                "published_year": None,
                "published_month": None,
                "meeting_type": "regular_meeting",
                "meeting_date": "",
                "video_duration_estimate": None,
                "transcript_source": "unknown",
            })

            # fill duration estimate from clean if available
            if not vm.get("video_duration_estimate"):
                vm["video_duration_estimate"] = _duration_estimate_from_clean(clean_path) if clean_path.exists() else None

            blob_chunks = _smart_chunks(txt)
            total = len(blob_chunks)
            for idx, ch_text in enumerate(blob_chunks):
                est_start, est_end = _estimate_chunk_timestamps(ch_text, clean_segments)
                meta = {
                    **vm,
                    "chunk_id": f"{vid}_chunk_{idx:04d}",
                    "chunk_index": idx,
                    "total_chunks": total,
                    "chunk_length": len(ch_text),
                    "estimated_start_time": est_start,
                    "estimated_end_time": est_end,
                }
                chunks.append({
                    "id": meta["chunk_id"],
                    "text": ch_text,
                    "metadata": meta,
                })
    return chunks

def embed_and_store_chunks(chunks: List[Dict], collection):
    if not chunks:
        logger.info("No chunks to embed/store.")
        return

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metas = [c["metadata"] for c in chunks]

    # compute embeddings
    client = _ollama_client()
    try:
        if client is not None:
            logger.info("Embedding with Ollama model: %s", config.EMBEDDING_MODEL)
            embs = _embed_with_ollama(client, texts)
        else:
            logger.info("Ollama not available; using sentence-transformers fallback.")
            embs = _embed_with_sentence_transformers(texts)
    except Exception as e:
        logger.warning("Primary embedding path failed (%s); falling back to sentence-transformers.", e)
        embs = _embed_with_sentence_transformers(texts)

    # upsert in batches (Chroma add() will fail on duplicate ids; upsert works if available)
    B = 200
    for i in range(0, len(ids), B):
        batch = slice(i, i+B)
        try:
            collection.upsert(
                ids=ids[batch],
                documents=texts[batch],
                metadatas=metas[batch],
                embeddings=embs[batch],
            )
        except Exception:
            # older versions: use add, else delete+add
            try:
                collection.add(
                    ids=ids[batch],
                    documents=texts[batch],
                    metadatas=metas[batch],
                    embeddings=embs[batch],
                )
            except Exception as e:
                logger.error("Failed to add chunk batch %d-%d: %s", i, i+B, e)

def create_vector_database():
    logger.info("Step 4: Initializing ChromaDB…")
    coll = initialize_chromadb()
    logger.info("Loading video metadata…")
    vmeta = load_video_metadata()
    logger.info("Building chunks from pure texts…")
    chunks = process_pure_texts_to_chunks(vmeta)
    logger.info("Prepared %d chunks. Embedding & storing…", len(chunks))
    embed_and_store_chunks(chunks, coll)
    logger.info("Vector DB build complete. Collection: %s", config.COLLECTION_NAME)
