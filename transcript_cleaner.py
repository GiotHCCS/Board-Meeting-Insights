from __future__ import annotations
import os, csv, logging, glob
from pathlib import Path
from typing import List, Tuple, Dict, Iterable
from datetime import timedelta
import regex as re
from dateutil import parser as dtparser
from tqdm import tqdm

import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_ANNOTS_BRACKET_RE = re.compile(
    r"""
    (?:                              # [ ... ] or ( ... )
       \[
         ([A-Z0-9 \-_/]{2,})         # ALL CAPS words/numbers/ - _ /
       \]
      |
       \(
         ([A-Z0-9 \-_/]{2,})
       \)
    )
    """,
    re.VERBOSE,
)

_HTML_TAG_RE   = re.compile(r"<[^>]+>")
_CUE_SPLIT_RE  = re.compile(r"\r?\n\r?\n")
_VTT_NOTE_RE   = re.compile(r"^\s*NOTE\b", re.IGNORECASE)

_ALLOWED_PUNCT = set(list(".,?!:;-—()'\""))
_SENT_END_RE = re.compile(r"(?<=[\.!?])\s+")

_STOPWORDS = {
    "the","a","an","and","or","but","of","to","in","on","for","with","at","by",
    "is","are","was","were","be","been","being","this","that","it","as","from",
    "we","you","they","he","she","i","our","your","their"
}

# ---------- helpers ----------
def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_").lower()

def ensure_school_folder_structure(school_slug: str) -> Dict[str, Path]:
    base = Path(config.CAPTIONS_DIR) / school_slug
    raw  = base / "raw"
    clean = base / "clean"
    pure  = base / "pure"
    raw.mkdir(parents=True, exist_ok=True)
    clean.mkdir(parents=True, exist_ok=True)
    pure.mkdir(parents=True, exist_ok=True)
    return {"base": base, "raw": raw, "clean": clean, "pure": pure}

def discover_school_folders() -> List[str]:
    cap = Path(config.CAPTIONS_DIR)
    if not cap.exists():
        return []
    return sorted([p.name for p in cap.iterdir() if p.is_dir()])

def extract_video_id_from_filename(vtt_path: str) -> str:
    # Normalize any list/tuple/set to a single string path
    if isinstance(vtt_path, (list, tuple, set)):
        if not vtt_path:
            return ""
        vtt_path = next(iter(vtt_path))
    vtt_path = str(vtt_path)

    name = Path(vtt_path).name  # e.g., 'VIDEOID.en.vtt' or 'VIDEOID.vtt'
    # Trim .vtt
    if name.lower().endswith(".vtt"):
        stem = name[:-4]
    else:
        stem = Path(name).stem

    # Strip trailing language tags like '.en' or '.en-US'
    if "." in stem:
        base, maybe_lang = stem.split(".", 1)
        if re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z]{2,3})?", maybe_lang):
            return base
    return stem

def should_reprocess(school_slug: str, video_id: str) -> bool:
    """Skip if both clean and pure already exist and reprocess is False."""
    base = Path(config.CAPTIONS_DIR) / school_slug
    clean_path = base / "clean" / f"{video_id}.txt"
    pure_path  = base / "pure" / f"{video_id}.txt"
    if not getattr(config, "REPROCESS_EXISTING_FILES", False):
        return not (clean_path.exists() and pure_path.exists())
    return True

def _as_str_path(p) -> str:
    """Return a usable path string even if a list/tuple/set sneaks in."""
    if isinstance(p, (list, tuple, set)):
        if not p:
            raise ValueError("Empty path collection")
        p = next(iter(p))
    return str(p)


# ---------- time formatting ----------
def _parse_vtt_ts(ts: str) -> float:
    # Accept H:MM:SS.mmm, MM:SS.mmm, etc.
    ts = ts.strip()
    # Fast path: HH:MM:SS.mmm
    m = re.match(r"(?:(\d{1,2}):)?(\d{2}):(\d{2})\.(\d{3})$", ts)
    if m:
        h = int(m.group(1) or 0)
        mnt = int(m.group(2)); s = int(m.group(3)); ms = int(m.group(4))
        return h*3600 + mnt*60 + s + ms/1000.0
    # Fallback: let dateutil try
    try:
        dt = dtparser.parse(ts)
        # We only care about the time part; this is best-effort
        return dt.hour*3600 + dt.minute*60 + dt.second + (dt.microsecond/1e6)
    except Exception:
        return 0.0

def _fmt_ts(seconds: float, force_hh=False, include_ms=False) -> str:
    total_ms = int(round(seconds * 1000))
    h = total_ms // 3_600_000
    rem = total_ms % 3_600_000
    m = rem // 60_000
    rem = rem % 60_000
    s = rem // 1000
    ms = rem % 1000
    if include_ms:
        return (f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}" if (force_hh or h > 0)
                else f"{m:02d}:{s:02d}.{ms:03d}")
    else:
        return (f"{h:02d}:{m:02d}:{s:02d}" if (force_hh or h > 0)
                else f"{m:02d}:{s:02d}")

def _should_force_hh(words_with_ts) -> bool:
    mode = getattr(config, "CLEAN_TIME_STYLE", "auto").lower()
    if mode == "hh":
        return True
    if mode == "mm":
        return False
    # auto
    return any(ts >= 3600 for ts, _ in words_with_ts)

# ---------- parsing ----------
_WORD_TAG_RE = re.compile(
    r"""
    <                # opening <
    (?P<ts>\d{2}:\d{2}:\d{2}\.\d{3})  # 00:00:00.000
    >
    (?:<c[^>]*>)?    # optional <c ...>
    (?P<w>[^<]+?)    # word content (until next tag)
    (?:</c>)?        # optional </c>
    """,
    re.VERBOSE
)

_CUE_TIME_RE = re.compile(
    r"""^(?P<start>\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})\s*-->\s*
        (?P<end>\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})""",
    re.VERBOSE
)

def _normalize_unicode_punct(text: str) -> str:
    # Normalize quotes/dashes/ellipsis to simple ASCII forms
    # curly quotes → straight
    text = text.replace("“", '"').replace("”", '"').replace("„", '"').replace("‟", '"')
    text = text.replace("‘", "'").replace("’", "'").replace("‚", "'").replace("‛", "'")
    # em/en dashes → hyphen
    text = text.replace("—", "-").replace("–", "-").replace("‒", "-").replace("―", "-")
    # ellipsis → "..."
    text = text.replace("…", "...")
    # non-breaking / zero-width spaces → regular space
    text = text.replace("\u00A0", " ").replace("\u200B", "")
    return text

def _strip_unwanted_symbols(text: str) -> str:
    # keep letters, digits, whitespace and allowed punctuation
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch.isspace() or ch in _ALLOWED_PUNCT:
            cleaned.append(ch)
        # else drop
    text = "".join(cleaned)
    # collapse multiple punctuation marks to one
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    # tidy spaces around punctuation
    text = re.sub(r"\s+([.,?!:;])", r"\1", text)   # no space before
    text = re.sub(r"([.,?!:;])([^\s])", r"\1 \2", text)  # single space after
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def _split_sentences_loose(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENT_END_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]

def _norm_no_stop(s: str) -> str:
    toks = re.findall(r"[A-Za-z0-9']+", s.lower())
    toks = [t for t in toks if t not in _STOPWORDS]
    return " ".join(toks)

def _dedupe_adjacent_sentences(sents: List[str]) -> List[str]:
    out = []
    prev_norm = ""
    prev_ns   = ""  # no-stopword norm
    for s in sents:
        cur_norm = _normalize_for_compare(s)
        cur_ns   = _norm_no_stop(s)

        # Exact duplicate or near-empty echo
        if cur_norm == prev_norm:
            continue

        # One/two-word echo (e.g., "afternoon." "afternoon.")
        if len(cur_norm.split()) <= 2 and cur_norm == prev_norm:
            continue

        # Very similar after stopword removal (handles tiny phrasing changes)
        if cur_ns and prev_ns and cur_ns == prev_ns:
            continue

        out.append(s)
        prev_norm = cur_norm
        prev_ns   = cur_ns
    return out

def _suppress_small_ngr_repeats(text: str, window: int = 24, n: int = 2) -> str:
    """
    Guard against immediate short n-gram repeats like 'call call', 'of of', or
    duplicated pairs from roll-up. Uses a sliding check on the output buffer.
    """
    toks = text.split()
    out: List[str] = []
    for tok in toks:
        out.append(tok)
        # Only check if we have twice the n-gram length
        if len(out) >= 2 * n:
            # immediate repeat of last n tokens?
            if out[-n:] == out[-2*n:-n]:
                del out[-n:]  # drop the second occurrence
                continue
            # optional: check short window for triple repeats (rare)
            if len(out) >= 3 * n and out[-n:] == out[-3*n:-2*n]:
                del out[-n:]
                continue
    return " ".join(out)


def _tidy_spaces_keep_punct(text: str) -> str:
    # Collapse whitespace; tighten before punctuation; do NOT force a space after '.'
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,?!:;])", r"\1", text)  # ok to tighten before , ? ! : ;
    text = re.sub(r"\.{2,}", "...", text)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    return text.strip()

def _normalize_for_compare(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokenize_simple(s: str) -> List[str]:
    return s.split()

def _rollup_delta(prev_payload: str, cur_payload: str) -> str:
    prev_norm = _normalize_for_compare(prev_payload)
    cur_norm  = _normalize_for_compare(cur_payload)
    if not prev_norm:
        return cur_payload

    if prev_norm == cur_norm:
        return ""  # duplicate cue, skip

    prev_tok = _tokenize_simple(prev_norm)
    cur_tok  = _tokenize_simple(cur_norm)

    # Find largest overlap where tail(prev) == head(cur)
    max_k = min(len(prev_tok), len(cur_tok), 60)
    overlap_k = 0
    for k in range(max_k, 0, -1):
        if prev_tok[-k:] == cur_tok[:k]:
            overlap_k = k
            break

    if overlap_k > 0:
        delta_tokens = cur_tok[overlap_k:]
        if len(delta_tokens) <= 2:   # almost all overlap → skip
            return ""
        return " ".join(delta_tokens)

    if cur_norm.startswith(prev_norm) and len(cur_norm) > len(prev_norm) + 2:
        return cur_payload[len(prev_payload):].lstrip()

    return cur_payload

def _extract_cue_texts_with_punct(vtt_path: str) -> List[str]:
    """
    Returns payload texts for each cue, punctuation-preserving, with:
    - roll-up delta handling
    - removal of ALL-CAPS bracketed annotations ([MUSIC], (LAUGHTER))
    - removal of YouTube speaker arrows '>>'
    """
    # Normalize path input
    if isinstance(vtt_path, (list, tuple, set)):
        if not vtt_path:
            return []
        vtt_path = next(iter(vtt_path))
    vtt_path = str(vtt_path)

    with open(vtt_path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    if not raw.lstrip().upper().startswith("WEBVTT"):
        raise ValueError("Not a WEBVTT file")

    cues = _CUE_SPLIT_RE.split(raw.strip())
    out: List[str] = []
    prev_txt_norm: str = ""

    for cue in cues:
        if _VTT_NOTE_RE.match(cue):
            continue
        lines = [ln for ln in cue.splitlines() if ln.strip()]
        if not lines:
            continue

        # Timing line: search first 3 lines to be tolerant
        time_line_idx = None
        for i, ln in enumerate(lines[:3]):
            if _CUE_TIME_RE.search(ln.strip()):
                time_line_idx = i
                break
        if time_line_idx is None:
            continue

        payload = "\n".join(lines[time_line_idx + 1:]).strip()
        if not payload:
            continue

        # Clean payload
        txt = _HTML_TAG_RE.sub("", payload)
        txt = _html_unescape(txt)            # your helper
        txt = txt.replace(">>", " ")         # <-- strip YouTube speaker arrows
        txt = _normalize_unicode_punct(txt)  # your helper
        txt = _ANNOTS_BRACKET_RE.sub(" ", txt)
        txt = _tidy_spaces_keep_punct(txt)
        if not txt:
            continue

        # Roll-up delta: keep only the new part
        delta = _rollup_delta(prev_txt_norm, txt)
        if not delta:
            continue

        out.append(delta)
        prev_txt_norm = _normalize_for_compare(txt)

    return out


_TAG_RE = re.compile(r"<[^>]+>")  # strip any remaining tags
_BRACKETED_RE = re.compile(r"(\[.*?\]|\(.*?\))")  # remove [MUSIC], (INAUDIBLE), etc.
_HTML_ENT_RE = re.compile(r"&(?:amp|lt|gt|quot|apos);")

def _html_unescape(text: str) -> str:
    return (text.replace("&amp;", "&")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&quot;", '"')
                .replace("&apos;", "'"))

def parse_vtt_file(vtt_path: str) -> List[Tuple[float, str]]:
    """
    Returns a list of (timestamp_seconds, word) in chronological order.
    Handles:
      - YouTube word-timed format: <00:00:01.500><c>word</c> ...
      - Simple segment lines: one text line per cue (Whisper output)
    """
    # Normalize path input (handles accidental list/tuple/set)
    if isinstance(vtt_path, (list, tuple, set)):
        if not vtt_path:
            raise ValueError("Empty path collection for VTT file")
        vtt_path = next(iter(vtt_path))
    vtt_path = str(vtt_path)

    with open(vtt_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    if not content.lstrip().upper().startswith("WEBVTT"):
        raise ValueError("Not a WEBVTT file")

    cues = re.split(r"\r?\n\r?\n", content.strip())
    words: List[Tuple[float, str]] = []

    for cue in cues:
        lines = [ln for ln in cue.splitlines() if ln.strip() != ""]
        if not lines:
            continue

        time_line_idx = None
        start_s = end_s = None
        for i, ln in enumerate(lines[:2]):  # usually first or second line
            m = _CUE_TIME_RE.search(ln.strip())
            if m:
                start_s = _parse_vtt_ts(m.group("start"))
                end_s   = _parse_vtt_ts(m.group("end"))
                time_line_idx = i
                break
        if time_line_idx is None or start_s is None or end_s is None or end_s < start_s:
            continue

        payload = "\n".join(lines[time_line_idx + 1:]).strip()
        if not payload:
            continue

        per_word = list(_WORD_TAG_RE.finditer(payload))
        if per_word:
            for m in per_word:
                ts = _parse_vtt_ts(m.group("ts"))
                w_raw = m.group("w")
                for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", w_raw):
                    words.append((ts, token.lower()))
            continue

        text = _BRACKETED_RE.sub(" ", _TAG_RE.sub("", payload))
        text = _html_unescape(text)
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", text)
        if not tokens:
            continue

        n = len(tokens)
        dur = max(0.001, end_s - start_s)
        if n == 1:
            words.append((start_s, tokens[0].lower()))
        else:
            step = dur / max(1, n - 1)
            for i, t in enumerate(tokens):
                ts = start_s + i * step
                words.append((ts, t.lower()))

    words.sort(key=lambda x: x[0])
    return words


# ---------- cleaning ----------
_CONTRACTIONS = {
    "don't":"do not","doesn't":"does not","didn't":"did not","won't":"will not","wouldn't":"would not",
    "can't":"cannot","couldn't":"could not","shouldn't":"should not","isn't":"is not","aren't":"are not",
    "wasn't":"was not","weren't":"were not","i'm":"i am","we're":"we are","you're":"you are","they're":"they are",
    "i've":"i have","we've":"we have","you've":"you have","they've":"they have","i'll":"i will","we'll":"we will",
    "you'll":"you will","they'll":"they will","it's":"it is","that's":"that is","there's":"there is","what's":"what is",
    "let's":"let us","could've":"could have","would've":"would have","should've":"should have","who's":"who is",
    "i’d":"i would","we’d":"we would","you’d":"you would","they’d":"they would","he’s":"he is","she’s":"she is",
    "he's":"he is","she's":"she is","’re":" are","’ll":" will","’ve":" have","’d":" would",
}
_NUM_0_20 = ["zero","one","two","three","four","five","six","seven","eight","nine","ten",
             "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen","twenty"]
_FILLERS = {"um","uh","er","erm","youknow","you-know"}

def _expand_contractions(text: str) -> str:
    if not getattr(config, "EXPAND_CONTRACTIONS", True):
        return text
    # replace smarter: word-boundary, case-insensitive
    def repl(m):
        raw = m.group(0).lower()
        return _CONTRACTIONS.get(raw, raw)
    pattern = re.compile(r"|".join(re.escape(k) for k in sorted(_CONTRACTIONS, key=len, reverse=True)), re.I)
    return pattern.sub(lambda m: _CONTRACTIONS.get(m.group(0).lower(), m.group(0)), text)

def _normalize_numbers_token(tok: str) -> str:
    if not getattr(config, "NORMALIZE_NUMBERS", True):
        return tok
    if tok.isdigit():
        n = int(tok)
        if 0 <= n <= 20:
            return _NUM_0_20[n]
    return tok

def _collapse_repetitions(words: List[str]) -> List[str]:
    if not getattr(config, "COLLAPSE_REPETITIONS", True):
        return words
    out = []
    prev = None
    for w in words:
        if w == prev:
            continue
        out.append(w)
        prev = w
    return out

def _remove_repetitive_fillers(words: List[str]) -> List[str]:
    if not getattr(config, "REMOVE_FILLER_WORDS", True):
        return words
    out: List[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        wnorm = w.replace(" ", "").replace("-", "")
        if wnorm in _FILLERS:
            # skip consecutive fillers entirely
            j = i+1
            while j < len(words) and words[j].replace(" ","").replace("-","") in _FILLERS:
                j += 1
            i = j
            continue
        out.append(w)
        i += 1
    return out

def create_clean_format(words_with_ts: List[Tuple[float,str]], output_path: str):
    k = max(1, int(getattr(config, "WORDS_PER_CLEAN_SEGMENT", 5)))
    if not words_with_ts:
        Path(output_path).write_text("", encoding="utf-8")
        return

    force_hh = _should_force_hh(words_with_ts)
    include_ms = bool(getattr(config, "CLEAN_TIME_INCLUDE_MS", False))
    max_span = getattr(config, "CLEAN_MAX_SPAN_SECONDS", None)

    lines = []
    for i in range(0, len(words_with_ts), k):
        chunk = words_with_ts[i:i+k]
        start = chunk[0][0]
        end   = chunk[-1][0]
        if max_span:
            end = min(end, start + float(max_span))
        text  = " ".join(w for _, w in chunk)
        lines.append(
            f"[{_fmt_ts(start, force_hh, include_ms)} - {_fmt_ts(end, force_hh, include_ms)}] {text}"
        )
    Path(output_path).write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

def create_pure_format_from_vtt(vtt_path: str, output_path: str):
    """
    Build 'pure' text from cue payloads (punctuation-preserving) with:
    - roll-up delta handling (in extractor)
    - adjacent sentence dedupe (stopword-aware)
    - small n-gram repetition suppression (default bigrams)
    - optional contractions expansion
    - paragraph wrapping by ~N words
    """
    if isinstance(vtt_path, (list, tuple, set)):
        if not vtt_path:
            Path(output_path).write_text("", encoding="utf-8")
            return
        vtt_path = next(iter(vtt_path))
    vtt_path = str(vtt_path)

    cue_texts = _extract_cue_texts_with_punct(vtt_path)
    if not cue_texts:
        Path(output_path).write_text("", encoding="utf-8")
        return

    stitched = " ".join(cue_texts).lower()
    if getattr(config, "EXPAND_CONTRACTIONS", True):
        stitched = _expand_contractions(stitched)

    sents = _split_sentences_loose(stitched)
    sents = _dedupe_adjacent_sentences(sents)

    text = " ".join(sents)
    text = _suppress_small_ngr_repeats(text, window=24, n=2)

    text = re.sub(r"\s{2,}", " ", text).strip()

    para_len = max(20, int(getattr(config, "PURE_TEXT_PARAGRAPH_LENGTH", 60)))
    if para_len > 0:
        tokens = text.split()
        paras = [" ".join(tokens[i:i+para_len]) for i in range(0, len(tokens), para_len)]
        text = "\n\n".join(paras)

    Path(output_path).write_text(text + "\n", encoding="utf-8")




# ---------- per-file driver ----------
def process_single_transcript(school_slug: str, video_id: str, vtt_file: str) -> bool:
    try:
        # Normalize path input (defensive)
        if isinstance(vtt_file, (list, tuple, set)):
            if not vtt_file:
                raise ValueError("Empty VTT path collection")
            vtt_file = next(iter(vtt_file))
        vtt_file = str(vtt_file)

        words = parse_vtt_file(vtt_file)
        if not words:
            raise ValueError("No words parsed from VTT")

        # prepare outputs
        dirs = ensure_school_folder_structure(school_slug)
        clean_out = dirs["clean"] / f"{video_id}.txt"
        pure_out  = dirs["pure"] / f"{video_id}.txt"

        create_clean_format(words, str(clean_out))
        create_pure_format_from_vtt(vtt_file, str(pure_out))

        # validation (basic)
        if getattr(config, "ENABLE_OUTPUT_VALIDATION", True):
            pure_txt = pure_out.read_text(encoding="utf-8", errors="ignore")
            if len(pure_txt.split()) < int(getattr(config, "MIN_PURE_TEXT_LENGTH", 100)):
                raise ValueError("Pure text too short")

        return True

    except Exception as e:
        # log failure row
        os.makedirs(config.DATA_DIR, exist_ok=True)
        write_header = not os.path.exists(config.CLEANING_FAILED_FILE)
        with open(config.CLEANING_FAILED_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["school","videoId","input_file","failure_reason","error_details"])
            w.writerow([school_slug, video_id, vtt_file, "processing_failed", str(e)])
        # include stack for debugging
        logger.exception("Transcript cleaning failed for %s/%s", school_slug, video_id)
        return False

# ---------- orchestrator ----------
def process_all_transcripts():
    config.ensure_directories()
    captions_root = Path(config.CAPTIONS_DIR)
    if not captions_root.exists():
        logger.info("No captions directory at %s", captions_root)
        return

    schools = discover_school_folders()
    if not schools:
        logger.info("No school folders to process under %s", captions_root)
        return

    total_vtts = sum(len(list((captions_root / s / "raw").glob("*.vtt"))) for s in schools)
    logger.info("Step 3: discovered %d VTT files across %d schools.", total_vtts, len(schools))

    school_iter = tqdm(schools, desc="Cleaning (schools)", unit="school") \
                  if getattr(config, "ENABLE_PROGRESS_BARS", True) else schools

    processed = skipped = failed = 0

    # Fresh failed file each run (optional but helpful)
    try:
        if os.path.exists(config.CLEANING_FAILED_FILE):
            os.remove(config.CLEANING_FAILED_FILE)
    except Exception:
        pass

    for school_slug in school_iter:
        dirs = ensure_school_folder_structure(school_slug)
        raw_dir = dirs["raw"]

        # Use Path.glob and keep paths as strings for downstream calls
        vtts = [str(p) for p in sorted(raw_dir.glob("*.vtt"))]

        vtt_iter = tqdm(vtts, desc=school_slug, unit="file", leave=False) \
                   if getattr(config, "ENABLE_PROGRESS_BARS", True) else vtts

        for vtt_file in vtt_iter:
            # Normalize path (belt & suspenders)
            if isinstance(vtt_file, (list, tuple, set)):
                if not vtt_file:
                    failed += 1
                    continue
                vtt_file = str(next(iter(vtt_file)))
            else:
                vtt_file = str(vtt_file)

            vid = extract_video_id_from_filename(vtt_file)  # handles legacy VIDEOID.en.vtt
            if not should_reprocess(school_slug, vid):
                skipped += 1
                continue

            ok = process_single_transcript(school_slug, vid, vtt_file)
            if ok:
                processed += 1
            else:
                failed += 1

    logger.info(
        "Transcript cleaning finished. Cleaned: %d | Skipped: %d | Failed: %d",
        processed, skipped, failed
    )
