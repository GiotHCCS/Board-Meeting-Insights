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
    return Path(vtt_path).stem

def should_reprocess(school_slug: str, video_id: str) -> bool:
    """Skip if both clean and pure already exist and reprocess is False."""
    base = Path(config.CAPTIONS_DIR) / school_slug
    clean_path = base / "clean" / f"{video_id}.txt"
    pure_path  = base / "pure" / f"{video_id}.txt"
    if not getattr(config, "REPROCESS_EXISTING_FILES", False):
        return not (clean_path.exists() and pure_path.exists())
    return True

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
    with open(vtt_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Basic sanity
    if not content.lstrip().upper().startswith("WEBVTT"):
        raise ValueError("Not a WEBVTT file")

    # Split cues by blank lines (tolerate \r\n)
    cues = re.split(r"\r?\n\r?\n", content.strip())

    words: List[Tuple[float, str]] = []

    for cue in cues:
        lines = [ln for ln in cue.splitlines() if ln.strip() != ""]
        if not lines:
            continue

        # Find the time line
        time_line_idx = None
        start_s = end_s = None
        for i, ln in enumerate(lines[:2]):  # typically the first or second line
            m = _CUE_TIME_RE.search(ln.strip())
            if m:
                start_s = _parse_vtt_ts(m.group("start"))
                end_s   = _parse_vtt_ts(m.group("end"))
                time_line_idx = i
                break
        if time_line_idx is None or start_s is None or end_s is None or end_s < start_s:
            # no usable time data, skip
            continue

        # The rest is text payload
        payload = "\n".join(lines[time_line_idx+1:]).strip()
        if not payload:
            continue

        # Priority 1: word-level tags
        per_word = list(_WORD_TAG_RE.finditer(payload))
        if per_word:
            for m in per_word:
                ts = _parse_vtt_ts(m.group("ts"))
                w_raw = m.group("w")
                # split on whitespace; keep hyphenated words together
                for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", w_raw):
                    words.append((ts, token.lower()))
            continue

        # Priority 2: plain text in cue -> interpolate
        # strip tags & bracketed annotations first
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

    # Sort just in case and return
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

def create_pure_format(words_with_ts: List[Tuple[float,str]], output_path: str):
    if not words_with_ts:
        Path(output_path).write_text("", encoding="utf-8")
        return
    words = [w for _, w in words_with_ts]
    # number normalization
    words = [_normalize_numbers_token(w) for w in words]
    # collapse repeats & remove repetitive fillers
    words = _collapse_repetitions(words)
    words = _remove_repetitive_fillers(words)
    text = " ".join(words).lower()

    # punctuation tidy (lightweight)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)

    # expand contractions
    text = _expand_contractions(text)

    # whitespace normalize
    text = re.sub(r"\s+", " ", text).strip()

    # paragraphing
    para_len = max(20, int(getattr(config, "PURE_TEXT_PARAGRAPH_LENGTH", 60)))
    out_lines = []
    if para_len > 0:
        tokens = text.split()
        for i in range(0, len(tokens), para_len):
            out_lines.append(" ".join(tokens[i:i+para_len]))
        text = "\n\n".join(out_lines)

    Path(output_path).write_text(text.strip() + "\n", encoding="utf-8")

# ---------- per-file driver ----------
def process_single_transcript(school_slug: str, video_id: str, vtt_file: str) -> bool:
    try:
        words = parse_vtt_file(vtt_file)
        if not words:
            raise ValueError("No words parsed from VTT")

        # prepare outputs
        dirs = ensure_school_folder_structure(school_slug)
        clean_out = dirs["clean"] / f"{video_id}.txt"
        pure_out  = dirs["pure"] / f"{video_id}.txt"

        create_clean_format(words, str(clean_out))
        create_pure_format(words, str(pure_out))

        # validation (basic)
        if getattr(config, "ENABLE_OUTPUT_VALIDATION", True):
            pure_txt = pure_out.read_text(encoding="utf-8", errors="ignore")
            if len(pure_txt.split()) < int(getattr(config, "MIN_PURE_TEXT_LENGTH", 100)):
                raise ValueError("Pure text too short")

        return True
    except Exception as e:
        # log failure
        os.makedirs(config.DATA_DIR, exist_ok=True)
        write_header = not os.path.exists(config.CLEANING_FAILED_FILE)
        with open(config.CLEANING_FAILED_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["school","videoId","input_file","failure_reason","error_details"])
            w.writerow([school_slug, video_id, vtt_file, "processing_failed", str(e)])
        logger.error("Transcript cleaning failed for %s/%s: %s", school_slug, video_id, e)
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

    total_vtts = 0
    for s in schools:
        total_vtts += len(list((captions_root / s / "raw").glob("*.vtt")))

    logger.info("Step 3: discovered %d VTT files across %d schools.", total_vtts, len(schools))

    school_iter = tqdm(schools, desc="Cleaning (schools)", unit="school") if getattr(config, "ENABLE_PROGRESS_BARS", True) else schools

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
        vtts = sorted(glob.glob(str(raw_dir / "*.vtt")))
        vtt_iter = tqdm(vtts, desc=school_slug, unit="file", leave=False) if getattr(config, "ENABLE_PROGRESS_BARS", True) else vtts
        for vtt_file in vtt_iter:
            vid = extract_video_id_from_filename(vtt_file)
            if not should_reprocess(school_slug, vid):
                skipped += 1
                continue
            ok = process_single_transcript(school_slug, vid, vtt_file)
            if ok:
                processed += 1
            else:
                failed += 1

    logger.info("Transcript cleaning finished. Cleaned: %d | Skipped: %d | Failed: %d",
                processed, skipped, failed)
