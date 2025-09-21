# rag_interface.py
from __future__ import annotations

import os, re, json, logging, math, random, shlex
from typing import List, Dict, Any, Tuple, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

try:
    from ollama import Client as OllamaClient
except Exception:
    OllamaClient = None

import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_KV_RE = re.compile(r'(\w+)\s*=\s*(".*?"|\'.*?\'|[^ \t]+)')
_COLLECTION_HAS_EF = False

# =========================
# Low-level setup helpers
# =========================
def _chroma():
    """
    Open the persistent Chroma collection for reads/writes.

    IMPORTANT: Rebuilds/deletions are handled ONLY in vector_database.create_vector_database()
    (Step 4). The RAG layer must NEVER honor REBUILD_VECTOR_DB to avoid wiping the collection
    when launching the chat UI.
    """
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions

    global _COLLECTION_HAS_EF

    # Try to prepare an Ollama EF that matches stored vectors (e.g., "nomic-embed-text").
    ef = None
    try:
        ef = embedding_functions.OllamaEmbeddingFunction(
            url=getattr(config, "OLLAMA_HOST", "http://127.0.0.1:11434"),
            model_name=getattr(config, "EMBEDDING_MODEL", "nomic-embed-text"),
        )
    except Exception as e:
        logger.warning("Could not init OllamaEmbeddingFunction: %s", e)

    client = chromadb.PersistentClient(
        path=config.CHROMADB_PERSIST_DIR,
        settings=Settings(
            allow_reset=False,              # never allow resets from RAG
            anonymized_telemetry=False
        ),
    )

    # DO NOT delete/rebuild here, regardless of config.REBUILD_VECTOR_DB.
    # That is handled exclusively by Step 4 (vector_database).

    # First, attempt to open with EF (works if no EF conflict).
    if ef is not None:
        try:
            coll = client.get_collection(
                name=config.COLLECTION_NAME,
                embedding_function=ef,
            )
            _COLLECTION_HAS_EF = True
            return coll, client
        except Exception as e_get_with_ef:
            # If collection doesn't exist, create with EF.
            try:
                coll = client.create_collection(
                    name=config.COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=ef,
                )
                _COLLECTION_HAS_EF = True
                return coll, client
            except Exception as e_create_with_ef:
                # Likely EF mismatch with an existing collection; fall back to no-EF open.
                logger.info("Opening collection without EF (EF mismatch or other error): %s", e_create_with_ef)

    # Fallback: open without EF to avoid conflicts with existing stored EF.
    try:
        coll = client.get_collection(name=config.COLLECTION_NAME)
    except Exception:
        coll = client.create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    _COLLECTION_HAS_EF = False
    return coll, client


def _ollama():
    if OllamaClient is None:
        return None
    try:
        return OllamaClient(host=config.OLLAMA_HOST)
    except Exception as e:
        logger.warning("Failed to init Ollama: %s", e)
        return None


def initialize_rag_system():
    coll, _ = _chroma()
    oll = _ollama()
    if oll is None:
        logger.warning("Ollama client not available; responses will be extractive (no LLM).")
    return coll, oll
# =========================
# IDK
# =========================
def _cg(coll, where: dict, limit: int = 100_000):
    """
    Chroma get wrapper that always returns a list of metadatas.
    Never returns a tuple; never returns None.
    """
    try:
        got = coll.get(where=where, include=["metadatas"], limit=limit)
    except Exception:
        return []
    if isinstance(got, dict):
        return (got.get("metadatas") or [])
    # In case some client/adapter returns unexpected shapes
    return []


# =========================
# Embeddings
# =========================
def _embed_query(query: str, oll) -> Optional[List[float]]:
    if oll is None:
        return None
    try:
        resp = oll.embeddings(model=config.EMBEDDING_MODEL, prompt=query)
        return resp.get("embedding")
    except Exception as e:
        logger.warning("Embedding failed, falling back to text search: %s", e)
        return None


# =========================
# Formatting helpers
# =========================
def _parse_kv_args(s: str) -> dict:
    """
    Parse key=value pairs, allowing quoted values. Strips surrounding quotes.
    Example: school="El Paso Community College" vid=9YcoA_UgxVY
    """
    out = {}
    for k, v in _KV_RE.findall(s or ""):
        v = v.strip()
        if len(v) >= 2 and ((v[0] == v[-1]) and v[0] in ('"', "'")):
            v = v[1:-1]
        out[k] = v
    return out

def _flatten_chunk(ch: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten retrieved chunk into a uniform dict."""
    m = ch.get("metadata", {}) or {}
    return {
        "id": ch.get("id"),
        "text": ch.get("text", ""),
        "video_id": m.get("video_id"),
        "school_name": m.get("school_name"),
        "meeting_type": m.get("meeting_type"),
        "published_date": m.get("meeting_date") or m.get("published_date"),
        "estimated_start_time": m.get("estimated_start_time"),
        "estimated_end_time": m.get("estimated_end_time"),
        "video_title": m.get("video_title"),
        "chunk_index": m.get("chunk_index"),
        "total_chunks": m.get("total_chunks"),
        "topic_keywords": m.get("topic_keywords"),
    }


def _format_context_with_labels(chunks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Return a context string with [S#] labels and a parallel list of flattened chunks
    so the LLM can cite [S#] inline and we can print an enhanced Sources section.
    """
    lines = []
    flattened = []
    for i, ch in enumerate(chunks, 1):
        f = _flatten_chunk(ch)
        flattened.append(f)
        sch = f.get("school_name") or ""
        title = f.get("video_title") or "Board Meeting"
        date = f.get("published_date") or ""
        start = f.get("estimated_start_time") or "?"
        end = f.get("estimated_end_time") or "?"
        preview = (f.get("text") or "")[:900]
        lines.append(f"[S{i}] {sch} | {title} | {date} | ~{start}-{end}\n{preview}")
    return "\n\n".join(lines), flattened


def format_enhanced_citations(flat_chunks: List[Dict[str, Any]]) -> str:
    """
    Emit a detailed Sources list that maps [S#] to meeting context.
    Expects flattened chunks (from _format_context_with_labels).
    """
    out = ["Sources:"]
    for i, f in enumerate(flat_chunks, 1):
        sch = f.get("school_name") or ""
        mt = (f.get("meeting_type") or "").replace("_", " ").title()
        date = f.get("published_date") or ""
        start = f.get("estimated_start_time")
        end = f.get("estimated_end_time")
        vid = f.get("video_id")
        idx = f.get("chunk_index")
        tot = f.get("total_chunks")
        topics = f.get("topic_keywords") or []
        line = f"[S{i}] {sch} | {mt} | {date}"
        if start and end:
            line += f" | ~{start}-{end}"
        if vid is not None:
            line += f" | video_id={vid}"
        if idx is not None and tot is not None:
            line += f" | chunk={idx}/{tot}"
        if topics:
            line += " | topics: " + ", ".join(topics[:3])
        out.append(line)
    return "\n".join(out)


def _sanitize_citations(text: str, n_sources: int) -> str:
    """
    Keep only [S1]..[SN]. Replace any other [S#] with [S?].
    """
    def _fix(m):
        try:
            k = int(m.group(1))
            return f"[S{k}]" if 1 <= k <= n_sources else "[S?]"
        except Exception:
            return "[S?]"
    return re.sub(r"\[S(\d+)\]", _fix, text)


# =========================
# Ranking / retrieval
# =========================
def _rerank_by_simple_criteria(results: Dict[str, Any]) -> List[int]:
    """Temporal boost (newer first) with stable ordering."""
    rows = results.get("ids", [[]])[0]
    if not rows:
        return []
    ids = list(range(len(rows)))
    if not getattr(config, "RAG_RETRIEVAL_SETTINGS", {}).get("enable_temporal_boosting", True):
        return ids
    years = []
    for md in results.get("metadatas", [[]])[0]:
        y = md.get("published_year") or 0
        if isinstance(y, int):
            years.append(y)
        elif isinstance(y, str) and y.isdigit():
            years.append(int(y))
        else:
            years.append(0)
    # newer first; tie-breaker = original rank (stable)
    ids.sort(key=lambda i: (years[i], -i), reverse=True)
    return ids


def retrieve_relevant_chunks(query: str, collection) -> List[Dict[str, Any]]:
    """
    Retrieve top-N chunks with optional diversity heuristics to cover more than one video.

    Notes:
    - Prefers query_embeddings (safe regardless of collection EF binding).
    - Uses query_texts ONLY if the collection has a bound EF (_COLLECTION_HAS_EF=True).
    - If neither embeddings nor EF are available, returns [] gracefully.
    """
    settings = config.RAG_RETRIEVAL_SETTINGS
    n = int(settings.get("max_chunks", 8))
    cand_mult = int(settings.get("candidate_multiplier", 4))
    prefer_diverse = bool(settings.get("prefer_source_diversity", True))
    max_per_video = int(settings.get("max_per_video", 1))

    # Pull a larger candidate pool so we *can* diversify
    candidate_k = max(n * cand_mult, n + 5, 50)

    # --- Fetch candidates (embed first; else text if EF bound; else bail) ---
    oll = _ollama()
    q_emb = _embed_query(query, oll)

    try:
        if q_emb is not None:
            results = collection.query(query_embeddings=[q_emb], n_results=candidate_k)
        else:
            if _COLLECTION_HAS_EF:
                results = collection.query(query_texts=[query], n_results=candidate_k)
            else:
                logger.warning("No query embedding available and collection has no bound EF; skipping similarity search.")
                return []
    except Exception as e:
        logger.warning("Chroma query failed: %s", e)
        return []

    # If nothing came back, exit early
    if not results or not results.get("ids") or not results["ids"] or not results["ids"][0]:
        return []

    # Re-rank (e.g., temporal boost) or keep original
    order = (
        _rerank_by_simple_criteria(results)
        if settings.get("enable_reranking", True)
        else list(range(len(results["ids"][0])))
    )

    def _make_chunk(idx: int) -> Dict[str, Any]:
        md = results["metadatas"][0][idx]
        doc = results["documents"][0][idx]
        id_ = results["ids"][0][idx]
        return {"id": id_, "metadata": md, "text": doc}

    # Diversity pass: cap per video_id, then fill remaining slots
    if prefer_diverse:
        from collections import defaultdict
        counts = defaultdict(int)
        diversified: List[Dict[str, Any]] = []
        used_idxs = set()

        for i in order:
            ch = _make_chunk(i)
            vid = (ch.get("metadata") or {}).get("video_id")
            if counts[vid] < max_per_video:
                diversified.append(ch)
                counts[vid] += 1
                used_idxs.add(i)
                if len(diversified) >= n:
                    break

        if len(diversified) < n:
            for i in order:
                if i in used_idxs:
                    continue
                diversified.append(_make_chunk(i))
                if len(diversified) >= n:
                    break

        return diversified[:n]

    # Fallback: just top-n
    picked: List[Dict[str, Any]] = []
    for i in order:
        picked.append(_make_chunk(i))
        if len(picked) >= n:
            break
    return picked



# =========================
# Generation
# =========================
def _generate_with_ollama(oll, system_prompt: str, user_prompt: str, *, temperature: float = None, max_tokens: int = None) -> str:
    if oll is None:
        return ""
    try:
        opts = {"temperature": temperature if temperature is not None else config.RAG_GENERATION_SETTINGS.get("temperature", 0.3)}
        if max_tokens is not None:
            # Ollama uses num_predict to cap tokens generated
            opts["num_predict"] = int(max_tokens)
        resp = oll.chat(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",  "content": user_prompt},
            ],
            options=opts,
        )
        return resp.get("message", {}).get("content", "").strip()
    except Exception as e:
        logger.warning("LLM generation failed: %s", e)
        return ""


def _extractive_answer_fallback(query: str, chunks: List[Dict[str, Any]]) -> str:
    out = ["(LLM unavailable — returning top relevant excerpts)\n"]
    for ch in chunks[:3]:
        m = ch.get("metadata", {}) or {}
        tag = f"[{m.get('school_name','')} | {m.get('meeting_date') or m.get('published_date','')}]"
        out.append(f"{tag} {ch.get('text','')[:800]}")
    return "\n\n".join(out)


def _format_citation_line(m: Dict[str, Any]) -> str:
    style = config.RAG_GENERATION_SETTINGS.get("citation_style", "detailed")
    sch = m.get("school_name", "")
    title = m.get("video_title", "") or "Board Meeting"
    date = m.get("meeting_date") or m.get("published_date") or ""
    start = m.get("estimated_start_time")
    end = m.get("estimated_end_time")
    if style == "simple":
        return f"[{sch} - {date}]"
    if start and end and config.RAG_GENERATION_SETTINGS.get("enable_timestamp_references", True):
        return f"[{sch} - {title} - {date} - ~{start}-{end}]"
    return f"[{sch} - {title} - {date}]"


def format_citations(chunks: List[Dict[str, Any]]) -> str:
    seen = set()
    cites = []
    for ch in chunks:
        c = _format_citation_line(ch.get("metadata", {}))
        if c not in seen:
            seen.add(c)
            cites.append(c)
    if not cites:
        return ""
    return "Sources: " + "; ".join(cites)


# =========================
# Prompt selection (multi-meeting)
# =========================
def detect_query_type(query: str) -> str:
    q = (query or "").lower()
    if any(w in q for w in ["budget", "cost", "funding", "money", "dollar", "expense", "revenue", "allocation", "financial"]):
        return "financial_query_prompt"
    if any(w in q for w in ["policy", "rule", "regulation", "procedure", "guideline", "approve", "adopt", "revise"]):
        return "policy_query_prompt"
    if any(w in q for w in ["hire", "hiring", "staff", "employee", "position", "job", "personnel", "appointment", "resignation"]):
        return "personnel_query_prompt"
    if any(w in q for w in ["when", "timeline", "history", "progression", "over time", "changed", "evolution"]):
        return "temporal_query_prompt"
    if any(w in q for w in ["compare", "difference", "versus", "vs", "between", "contrast", "similar", "unlike"]):
        return "comparative_query_prompt"
    return "context_prompt"


def generate_response(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    oll = _ollama()
    context_str, flat = _format_context_with_labels(context_chunks)

    prompt_key = detect_query_type(query)
    sys_prompt = config.RAG_PROMPT_TEMPLATES["system_prompt"]
    user_template = config.RAG_PROMPT_TEMPLATES.get(prompt_key, config.RAG_PROMPT_TEMPLATES["context_prompt"])
    user_prompt = user_template.format(context_chunks=context_str, user_question=query)

    if oll is not None:
        answer = _generate_with_ollama(
            oll, sys_prompt, user_prompt,
            temperature=config.RAG_GENERATION_SETTINGS.get("temperature", 0.3),
            max_tokens=config.RAG_GENERATION_SETTINGS.get("max_response_tokens", 800),
        )
        if not answer.strip():
            answer = _extractive_answer_fallback(query, context_chunks)
    else:
        answer = _extractive_answer_fallback(query, context_chunks)

    return f"{answer}\n\n{format_enhanced_citations(flat)}"


# =========================
# Public API
# =========================
def ask(question: str) -> str:
    coll, _ = _chroma()
    ctx = retrieve_relevant_chunks(question, coll)
    if not ctx:
        return "No relevant transcript segments found. Try rephrasing your question or broadening the scope."
    return generate_response(question, ctx)


# =========================
# Single-meeting overview helpers (NEW)
# =========================
def _slugify_school(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "-", s)
    s = re.sub(r"^-+|-+$", "", s)
    return s or "unknown"


def _get_all_chunks_for_video(collection, video_id: str) -> List[Dict[str, Any]]:
    """
    Fetch every chunk for a given video_id via direct metadata filter (no similarity bias).
    """
    # 'ids' is always returned; do NOT put it in include=[]
    res = collection.get(
        where={"video_id": video_id},
        include=["metadatas", "documents"],
    )

    docs  = res.get("documents")  or []
    metas = res.get("metadatas")  or []
    ids   = res.get("ids")        or []

    rows: List[Dict[str, Any]] = []
    for id_, md, doc in zip(ids, metas, docs):
        rows.append({"id": id_, "text": doc, "metadata": md})

    # chronological by chunk_index when available
    rows.sort(key=lambda x: (x["metadata"].get("chunk_index") or 0))
    return rows


def _select_video_by_filters(coll, school=None, vid=None, date=None, title_keywords=None):
    """
    Selection rules:
    - If vid is provided, return that video directly: (video_id, rows_list).
    - Else query by school (name, dashed slug, underscored slug) and refine by date/title.
    Returns: (video_id: str, rows: List[dict])
    Raises RuntimeError with a clear message when nothing is found.
    """
    # 1) Direct by video_id
    if vid:
        mds = _cg(coll, {"video_id": vid})
        if mds:
            return vid, mds
        raise RuntimeError(f"No chunks found for video_id='{vid}'.")

    # 2) By school (try name, dashed slug, underscored slug)
    def _first_nonempty(*wheres):
        for w in wheres:
            mds = _cg(coll, w)
            if mds:
                return mds
        return []

    mds = []
    if school:
        slug_dash = re.sub(r"\s+", "-", school.strip().lower())
        slug_us   = re.sub(r"\s+", "_", school.strip().lower())
        mds = _first_nonempty(
            {"school_name": school},
            {"school_slug": slug_dash},
            {"school_slug": slug_us},
        )
    else:
        # last resort: load everything and pick later (can be large)
        try:
            got = coll.get(include=["metadatas"], limit=100_000)
            mds = (got.get("metadatas") or []) if isinstance(got, dict) else []
        except Exception:
            mds = []

    if not mds:
        raise RuntimeError(f"No chunks found for school='{school}'.")

    # Optional refinements
    def _title_match(t, kws):
        return bool(kws) and all(k.lower() in (t or "").lower() for k in kws.split())

    if date:
        filt = [m for m in mds if (m.get("published_date") == date or m.get("meeting_date") == date)]
        if filt:
            mds = filt
    if title_keywords:
        filt = [m for m in mds if _title_match(m.get("video_title", ""), title_keywords)]
        if filt:
            mds = filt

    # Group by video_id and pick the one with most chunks
    from collections import defaultdict
    groups = defaultdict(list)
    for m in mds:
        groups[m.get("video_id")] .append(m)
    if not groups:
        raise RuntimeError("No candidate videos after refinement.")

    best_vid = max(groups.items(), key=lambda kv: len(kv[1]))[0]
    return best_vid, groups[best_vid]




def _format_overview_context(chunks: List[Dict[str, Any]]) -> str:
    """Format chunks with [S#] labels, preserving chronological order."""
    parts = []
    for i, ch in enumerate(chunks, 1):
        md = ch["metadata"]
        sch = md.get("school_name", "")
        title = md.get("video_title", "") or "Board Meeting"
        date = md.get("meeting_date") or md.get("published_date") or ""
        start = md.get("estimated_start_time") or "?"
        end = md.get("estimated_end_time") or "?"
        idx = md.get("chunk_index")
        tot = md.get("total_chunks")
        head = f"[S{i}] {sch} | {title} | {date} | ~{start}-{end} | chunk {idx}/{tot}"
        parts.append(head + "\n" + (ch.get("text") or "")[:1500])
    return "\n\n".join(parts)


def generate_video_overview(
    school: str,
    upload_date: Optional[str] = None,
    video_id: Optional[str] = None,
    title_keywords: Optional[str] = None,
    focus_area: Optional[str] = None,
) -> str:
    """
    End-to-end: pick a single video for the school, pull its chunks, prioritize the opening,
    sample the rest for coverage, and prompt the LLM for a structured overview.
    """
    # Initialize RAG (your existing function that returns (collection, ollama_client_or_none))
    collection, oll = initialize_rag_system()

    # ---- small, safe helpers local to this function ----
    def _cg(where: dict, limit: int = 100_000):
        """Chroma get wrapper that always returns a list of metadatas (never None/tuple)."""
        try:
            got = collection.get(where=where, include=["metadatas"], limit=limit)
        except Exception:
            return []
        if isinstance(got, dict):
            return (got.get("metadatas") or [])
        return []

    def _fetch_all_chunks_for_video(vid: str):
        rows = _cg({"video_id": vid}, limit=100_000)
        # ensure stable order by chunk_index
        rows.sort(key=lambda m: (m.get("chunk_index") or 0))
        # wrap rows into the shape your context/formatters expect
        return [{"metadata": m} for m in rows]

    # 1) Choose video (uses the robust selector defined elsewhere; pass kwargs by name!)
    try:
        chosen_video_id, seed_rows = _select_video_by_filters(
            collection,
            school=school,
            vid=video_id,
            date=upload_date,
            title_keywords=title_keywords,
        )
    except RuntimeError as e:
        return str(e)

    if not chosen_video_id:
        return f"Unable to resolve a video for {school}."

    # derive a representative metadata dict for header fields
    md = (seed_rows[0] if seed_rows else {}) or {}

    # 2) Pull ALL chunks for that video
    all_chunks = _fetch_all_chunks_for_video(chosen_video_id)
    if not all_chunks:
        return f"Transcript chunks not found for video {chosen_video_id}."

    # 3) Prioritize opening + sample remainder
    cfg = getattr(config, "VIDEO_OVERVIEW_CONFIG", {})
    max_chunks = int(cfg.get("max_chunks_for_overview", 22))
    first_frac = float(cfg.get("priority_chunk_fraction", 0.35))
    sample_rest = bool(cfg.get("sample_rest", True))

    total = len(all_chunks)
    k_first = max(1, min(int(total * first_frac), max_chunks // 2))
    first_part = all_chunks[:k_first]

    rest_needed = max_chunks - len(first_part)
    if rest_needed > 0:
        remainder = all_chunks[k_first:]
        if sample_rest and remainder:
            want = min(rest_needed, len(remainder))
            step = max(1, len(remainder) // want)
            sampled = [remainder[i] for i in range(0, len(remainder), step)][:want]
        else:
            sampled = remainder[:rest_needed]
        selected = first_part + sampled
    else:
        selected = first_part

    # keep chronological order
    selected.sort(key=lambda x: (x["metadata"].get("chunk_index") or 0))

    # 4) Format context and prompts
    context_str = _format_overview_context(selected)

    n_sources = len(selected)
    allowed_labels = " ".join(f"[S{i}]" for i in range(1, n_sources + 1))

    base_sys = config.RAG_PROMPT_TEMPLATES["video_overview_system_prompt"]
    sys_prompt = base_sys + (
        f"\nValid citations are ONLY: {allowed_labels}. Never invent [S#] labels beyond this list."
        if n_sources > 0 else ""
    )

    tmpl_key = "focused_video_overview_prompt" if focus_area else "video_overview_prompt"
    user_prompt = config.RAG_PROMPT_TEMPLATES[tmpl_key].format(
        context_chunks=context_str,
        user_question=(focus_area or f"Create a comprehensive overview of this {md.get('school_name','')} board meeting.")
    )
    if n_sources < 3:
        user_prompt += (
            "\n\nNOTE: Only a small portion of the meeting is available. "
            "Limit the Table of Contents and Timeline to what is explicitly supported "
            "by the provided segments; do not infer additional agenda items."
        )

    # 5) Generate
    if oll is None:
        return "(LLM unavailable) — cannot generate the overview right now."
    try:
        answer = _generate_with_ollama(
            oll, sys_prompt, user_prompt,
            temperature=config.VIDEO_OVERVIEW_CONFIG.get("overview_temperature", 0.2),
            max_tokens=config.VIDEO_OVERVIEW_CONFIG.get("overview_max_tokens", 1400),
        )
        answer = _sanitize_citations(answer or "", n_sources)
    except Exception as e:
        answer = f"(Generation error) {e}"

    # 6) Header + enhanced sources
    header = (
        "=== BOARD MEETING OVERVIEW ===\n"
        f"School: {md.get('school_name','')}\n"
        f"Meeting: {md.get('video_title','')}\n"
        f"Date: {md.get('meeting_date') or md.get('published_date','')}\n"
        f"Video ID: {chosen_video_id}\n"
        f"Chunks Analyzed: {len(selected)} of {total}\n"
        f"Coverage: {100.0 * len(selected) / max(total,1):.1f}%\n\n"
    )

    flat = []
    for ch in selected:
        m = ch["metadata"]
        flat.append({
            "video_id": chosen_video_id,
            "school_name": md.get("school_name"),
            "meeting_type": md.get("meeting_type"),
            "published_date": md.get("meeting_date") or md.get("published_date"),
            "estimated_start_time": m.get("estimated_start_time"),
            "estimated_end_time": m.get("estimated_end_time"),
            "video_title": md.get("video_title"),
            "chunk_index": m.get("chunk_index"),
            "total_chunks": m.get("total_chunks"),
            "topic_keywords": m.get("topic_keywords"),
        })
    sources_block = format_enhanced_citations(flat)

    return header + (answer or "") + "\n\n" + sources_block


# =========================
# CLI (interactive)
# =========================
def interactive_chat():
    import re, json
    coll, _ = _chroma()
    print("RAG ready. Type your question, or /help for commands. Ctrl+C to exit.")

    # --- robust key=value parser (handles quoted values and strips the quotes) ---
    _KV_RE = re.compile(r'(\w+)\s*=\s*(".*?"|\'.*?\'|[^ \t]+)')

    def _parse_kv_args(s: str) -> dict:
        args = {}
        for k, v in _KV_RE.findall(s or ""):
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
                v = v[1:-1]
            args[k.lower()] = v
        return args

    while True:
        try:
            q = input("\n> ").strip()
            if not q:
                continue

            # ---- help / config / stats / quit ----
            if q == "/help":
                print('/overview school="Name" [date=YYYY-MM-DD] [vid=VIDEOID] [title="keywords"] [focus="topic"]')
                print("/config  — print retrieval/generation settings")
                print("/stats   — quick collection size hint")
                print("/quit    — exit")
                continue

            if q == "/config":
                print(json.dumps({
                    "retrieval":  config.RAG_RETRIEVAL_SETTINGS,
                    "generation": config.RAG_GENERATION_SETTINGS,
                    "overview":   config.VIDEO_OVERVIEW_CONFIG
                }, indent=2))
                continue

            if q == "/stats":
                try:
                    cnt = coll.count()
                    print(f"Collection: {config.COLLECTION_NAME} (count={cnt})")
                except Exception:
                    # fallback probe for older client versions
                    res = coll.query(query_texts=["dummy"], n_results=1)
                    got = len(res.get("ids", [[]])[0])
                    print(f"Collection: {config.COLLECTION_NAME} (probe returned {got})")
                continue

            if q in {"/quit", "/exit"}:
                break

            # ---- optional tiny debugs ----
            if q.startswith("/echoargs"):
                print(_parse_kv_args(q))
                continue

            if q.startswith("/checkvid"):
                args = _parse_kv_args(q)
                vid = args.get("vid") or args.get("video") or args.get("video_id")
                if not vid:
                    print("Usage: /checkvid vid=VIDEO_ID")
                    continue
                rows = coll.get(where={"video_id": vid}, include=["metadatas"], limit=5)
                mds = (rows or {}).get("metadatas") or []
                print(f"Rows for video_id={vid}: {len(mds)}")
                if not mds:
                    print("Try checking by school, e.g. /overview school=\"Exact School Name\"")
                continue

            # ---- overview command ----
            if q.startswith("/overview"):
                args = _parse_kv_args(q)
                school = args.get("school") or args.get("s")
                vid    = args.get("vid") or args.get("video") or args.get("video_id")
                date   = args.get("date") or args.get("upload_date")
                titlek = args.get("title") or args.get("title_keywords")
                focus  = args.get("focus") or args.get("topic")

                # If a video id is provided, don't require (or AND with) school/date/title.
                if not vid and not school:
                    print('Usage: /overview school="Name" [date=YYYY-MM-DD] [vid=VIDEOID] [title="keywords"] [focus="topic"]')
                    continue

                print("\nGenerating overview …")
                try:
                    print(generate_video_overview(
                        school=school,
                        upload_date=date,
                        video_id=vid,
                        title_keywords=titlek,
                        focus_area=focus
                    ))
                except Exception as e:
                    print(str(e))
                continue

            # ---- default: multi-meeting Q&A ----
            ctx = retrieve_relevant_chunks(q, coll)
            print("\n" + generate_response(q, ctx))

        except KeyboardInterrupt:
            break
        except EOFError:
            break
