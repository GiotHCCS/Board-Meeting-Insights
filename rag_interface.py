from __future__ import annotations
import os, logging, math
from typing import List, Dict, Any, Tuple
import json
import chromadb
from chromadb.config import Settings

try:
    from ollama import Client as OllamaClient
except Exception:
    OllamaClient = None

import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _chroma():
    client = chromadb.PersistentClient(
        path=config.CHROMADB_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    coll = client.get_or_create_collection(name=config.COLLECTION_NAME, metadata={"hnsw:space":"cosine"})
    return coll, client

def _ollama():
    if OllamaClient is None:
        return None
    try:
        return OllamaClient(host=config.OLLAMA_HOST)
    except Exception:
        return None

def initialize_rag_system():
    coll, _ = _chroma()
    oll = _ollama()
    if oll is None:
        logger.warning("Ollama client not available; responses will be extractive (no LLM).")
    return coll, oll

def _embed_query(query: str, oll):
    if oll is None:
        return None
    try:
        resp = oll.embeddings(model=config.EMBEDDING_MODEL, prompt=query)
        return resp["embedding"]
    except Exception:
        return None

def _format_context_for_prompt(chunks: List[Dict[str, Any]]) -> str:
    lines = []
    for ch in chunks:
        m = ch.get("metadata", {})
        sch = m.get("school_name","")
        title = m.get("video_title","")
        date = m.get("meeting_date") or m.get("published_date") or ""
        start = m.get("estimated_start_time") or "?"
        end = m.get("estimated_end_time") or "?"
        lines.append(
            f"- [{sch} | {title} | {date} | ~{start}-{end}] {ch.get('text','')[:900]}"
        )
    return "\n".join(lines)

def _rerank_by_simple_criteria(results: Dict[str, Any]) -> List[int]:
    rows = results.get("ids", [[]])[0]
    if not rows:
        return []
    ids = list(range(len(rows)))
    if not getattr(config, "RAG_RETRIEVAL_SETTINGS", {}).get("enable_temporal_boosting", True):
        return ids
    years = []
    for md in results.get("metadatas", [[]])[0]:
        y = md.get("published_year") or 0
        years.append(int(y) if (isinstance(y, int) or (isinstance(y, str) and y.isdigit())) else 0)
    ids.sort(key=lambda i: (years[i], i), reverse=True)  # newer first, stable
    return ids

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

def _flatten_chunk(ch: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten our retrieved chunk into top-level keys expected by formatting."""
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

def retrieve_relevant_chunks(query: str, collection) -> List[Dict[str, Any]]:
    settings = config.RAG_RETRIEVAL_SETTINGS
    n = int(settings.get("max_chunks", 8))
    cand_mult = int(settings.get("candidate_multiplier", 4))
    prefer_diverse = bool(settings.get("prefer_source_diversity", True))
    max_per_video = int(settings.get("max_per_video", 1))

    # Pull a larger candidate pool so we *can* diversify
    candidate_k = max(n * cand_mult, n + 5, 50)

    oll = _ollama()
    q_emb = _embed_query(query, oll)
    results = (collection.query(query_embeddings=[q_emb], n_results=candidate_k)
               if q_emb is not None else
               collection.query(query_texts=[query], n_results=candidate_k))

    # If nothing came back, exit early
    if not results or not results.get("ids") or not results["ids"] or not results["ids"][0]:
        return []

    # Re-rank (e.g., temporal boost) or keep original
    order = (_rerank_by_simple_criteria(results)
             if settings.get("enable_reranking", True)
             else list(range(len(results["ids"][0]))))

    def _make_chunk(idx: int) -> Dict[str, Any]:
        md = results["metadatas"][0][idx]
        doc = results["documents"][0][idx]
        id_ = results["ids"][0][idx]
        return {"id": id_, "metadata": md, "text": doc}

    # If you want diversity: first pass caps per video_id, second pass fills remaining slots
    if prefer_diverse:
        from collections import defaultdict
        counts = defaultdict(int)
        diversified: List[Dict[str, Any]] = []
        used_idxs = set()

        # Pass 1 — enforce max_per_video
        for i in order:
            ch = _make_chunk(i)
            vid = (ch.get("metadata") or {}).get("video_id")
            if counts[vid] < max_per_video:
                diversified.append(ch)
                counts[vid] += 1
                used_idxs.add(i)
                if len(diversified) >= n:
                    break

        # Pass 2 — top up from leftovers if we still need more
        if len(diversified) < n:
            for i in order:
                if i in used_idxs:
                    continue
                diversified.append(_make_chunk(i))
                if len(diversified) >= n:
                    break

        return diversified[:n]

    # Otherwise: just take top-n in rank order
    picked: List[Dict[str, Any]] = []
    for i in order:
        picked.append(_make_chunk(i))
        if len(picked) >= n:
            break
    return picked


def _generate_with_ollama(oll, system_prompt: str, prompt: str) -> str:
    try:
        resp = oll.chat(
            model=config.LLM_MODEL,
            messages=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":prompt},
            ],
            options={"temperature": config.RAG_GENERATION_SETTINGS.get("temperature", 0.3)}
        )
        return resp.get("message", {}).get("content", "").strip()
    except Exception as e:
        logger.warning("LLM generation failed: %s", e)
        return ""

def _extractive_answer_fallback(query: str, chunks: List[Dict[str, Any]]) -> str:
    # simple top-chunk extractive summary
    out = ["(LLM unavailable — returning top relevant excerpts)\n"]
    for ch in chunks[:3]:
        m = ch["metadata"]
        tag = f"[{m.get('school_name','')} | {m.get('meeting_date') or m.get('published_date','')}]"
        out.append(f"{tag} {ch['text'][:800]}")
    return "\n\n".join(out)

def _format_citation_line(m: Dict[str, Any]) -> str:
    style = config.RAG_GENERATION_SETTINGS.get("citation_style", "detailed")
    sch = m.get("school_name","")
    title = m.get("video_title","") or "Board Meeting"
    date = m.get("meeting_date") or m.get("published_date") or ""
    start = m.get("estimated_start_time")
    end = m.get("estimated_end_time")
    if style == "simple":
        return f"[{sch} - {date}]"
    # detailed / default
    if start and end and config.RAG_GENERATION_SETTINGS.get("enable_timestamp_references", True):
        return f"[{sch} - {title} - {date} - ~{start}-{end}]"
    return f"[{sch} - {title} - {date}]"

def format_citations(chunks: List[Dict[str, Any]]) -> str:
    # de-duplicate citations by chunk metadata range
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

def generate_response(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    oll = _ollama()
    # Build [S#]-labeled context and a flattened list for citations
    context_str, flat = _format_context_with_labels(context_chunks)

    # Choose specialized prompt
    prompt_key = detect_query_type(query)
    sys_prompt = config.RAG_PROMPT_TEMPLATES["system_prompt"]
    user_template = config.RAG_PROMPT_TEMPLATES.get(prompt_key, config.RAG_PROMPT_TEMPLATES["context_prompt"])
    user_prompt = user_template.format(context_chunks=context_str, user_question=query)

    # Generate
    if oll is not None:
        answer = _generate_with_ollama(oll, sys_prompt, user_prompt)
        if not answer.strip():
            answer = _extractive_answer_fallback(query, context_chunks)
    else:
        answer = _extractive_answer_fallback(query, context_chunks)

    # Append enhanced citations listing S# → meeting
    return f"{answer}\n\n{format_enhanced_citations(flat)}"

def ask(question: str) -> str:
    coll, _ = _chroma()
    ctx = retrieve_relevant_chunks(question, coll)
    if not ctx:
        return "No relevant transcript segments found. Try rephrasing your question or broadening the scope."
    return generate_response(question, ctx)

# Optional CLI for quick testing
def interactive_chat():
    coll, _ = _chroma()
    print("RAG ready. Type your question, or /help for commands. Ctrl+C to exit.")
    while True:
        try:
            q = input("\n> ").strip()
            if not q:
                continue
            if q == "/help":
                print("/schools (not implemented), /config, /stats, /quit")
                continue
            if q == "/config":
                print(json.dumps({"retrieval": config.RAG_RETRIEVAL_SETTINGS,
                                  "generation": config.RAG_GENERATION_SETTINGS}, indent=2))
                continue
            if q == "/stats":
                # Chroma doesn't expose count directly in all versions; a trick:
                res = coll.query(query_texts=["dummy"], n_results=1)
                total_hint = len(res.get("ids", [[]])[0])
                print(f"Collection: {config.COLLECTION_NAME} (hint n_results=1 returns {total_hint} row)")
                continue
            if q in {"/quit","/exit"}:
                break
            ctx = retrieve_relevant_chunks(q, coll)
            print("\n" + generate_response(q, ctx))
        except KeyboardInterrupt:
            break
        except EOFError:
            break
