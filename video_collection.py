"""
Efficient collection of likely board meeting videos per college channel.

Order per channel (quota-friendly):
  1) uploads playlist (cheapest)
  2) board-related playlists (targeted)
  3) channel search (optional, priciest)

For each source we oversample a bit, then batch-query videos.list with:
  part=snippet,contentDetails,liveStreamingDetails,status
…to enforce:
  - EXCLUDE_LIVE_AND_UPCOMING
  - EXCLUDE_NON_PUBLIC (optional)
  - MIN_MEETING_SECONDS
  - SEARCH_CUTOFF_MONTHS
  - MIN_RELEVANCE_SCORE via score_video()

Outputs: data/video_List.csv with
  college,channelId,videoId,url,title,publishedAt,relevance_score
"""

from __future__ import annotations

import os
import re
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ISO8601_FMT = "%Y-%m-%dT%H:%M:%SZ"

# ------------------------------
# General helpers
# ------------------------------

def _iso_now_minus_months(months: int) -> str:
    """Return an ISO8601 'publishedAfter' like 2024-12-01T00:00:00Z."""
    dt = datetime.now(timezone.utc) - timedelta(days=30 * max(months, 0))
    return dt.strftime(ISO8601_FMT)

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def _paginate(method, cap: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
    """
    Generic paginator for list endpoints. Stops at `cap` items if provided.
    """
    out: List[Dict[str, Any]] = []
    page_token = None
    while True:
        try:
            resp = method(pageToken=page_token, **kwargs).execute()
        except HttpError as e:
            logger.warning("YouTube API error during pagination: %s", e)
            break
        items = resp.get("items", []) or []
        out.extend(items)
        if cap and len(out) >= cap:
            return out[:cap]
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return out

def _extract_video_id(item: Dict[str, Any]) -> Optional[str]:
    sn = item.get("snippet", {}) or {}
    rid = sn.get("resourceId") if isinstance(sn.get("resourceId"), dict) else {}
    vid = rid.get("videoId") or (item.get("id", {}) or {}).get("videoId") or item.get("id")
    if isinstance(vid, dict):
        vid = vid.get("videoId")
    return vid

def _published_at_from_item(it: Dict[str, Any]) -> str:
    sn = it.get("snippet", {}) or {}
    cd = it.get("contentDetails", {}) or {}
    return sn.get("publishedAt") or cd.get("videoPublishedAt") or ""

def _published_after_filter(item: Dict[str, Any], cutoff_iso: str) -> bool:
    pa = _published_at_from_item(item)
    if not pa:
        return True
    try:
        return pa >= cutoff_iso  # ISO sorts lexicographically
    except Exception:
        return True

def get_authenticated_service():
    """Return an authenticated YouTube API client (shared with discovery module)."""
    creds = None
    if os.path.exists(config.TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(config.TOKEN_FILE, config.YOUTUBE_SCOPES)
        except Exception as ex:
            logger.warning("Failed to load existing token: %s", ex)
            creds = None
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(config.YOUTUBE_API_KEY_FILE, config.YOUTUBE_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(config.TOKEN_FILE, "w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())
    return build("youtube", "v3", credentials=creds)

# ------------------------------
# Scoring helpers (reuse your config lists)
# ------------------------------

def contains_excluded_terms(text: str) -> bool:
    tl = text.lower()
    return any(term.lower() in tl for term in config.EXCLUDE_TERMS)

def count_keywords(text: str, keywords: List[str]) -> int:
    tl = text.lower()
    return sum(1 for kw in keywords if kw.lower() in tl)

def contains_priority_phrase(text: str) -> bool:
    tl = text.lower()
    return any(p.lower() in tl for p in config.PRIORITY_PHRASES)

def score_video(title: str, description: str) -> int:
    combined = f"{title}\n{description}"
    if contains_excluded_terms(combined):
        return -5
    score = 0
    if contains_priority_phrase(combined):
        score += 3
    kc = count_keywords(combined, config.BOARD_MEETING_KEYWORDS)
    if kc >= 2:
        score += 2
    elif kc == 1:
        score += 1
    return score

# ------------------------------
# Source fetchers
# ------------------------------

def _uploads_playlist_id(youtube, channel_id: str) -> Optional[str]:
    try:
        resp = youtube.channels().list(part="contentDetails", id=channel_id, maxResults=1).execute()
    except HttpError as e:
        logger.error("channels.list failed for %s: %s", channel_id, e)
        return None
    items = resp.get("items", [])
    if not items:
        return None
    return (items[0].get("contentDetails", {}) or {}).get("relatedPlaylists", {}).get("uploads")

def fetch_recent_uploads(youtube, channel_id: str, cutoff_iso: str, oversample: int) -> List[Dict[str, Any]]:
    """
    Get recent uploads from the authoritative 'uploads' playlist.
    Oversample to give downstream filtering room.
    """
    pid = _uploads_playlist_id(youtube, channel_id)
    if not pid:
        return []
    # Max 50 per page; oversample amount bounded
    raw = _paginate(
        youtube.playlistItems().list,
        cap=min(max(oversample, 1) * 2 * 50, 500),
        part="snippet,contentDetails",
        playlistId=pid,
        maxResults=50,
    )
    return [it for it in raw if _published_after_filter(it, cutoff_iso)]

def is_board_playlist_title(title: str) -> bool:
    t = _normalize(title)
    terms = set(kw.lower() for kw in config.BOARD_MEETING_KEYWORDS)
    base = {"board", "trustee", "trustees", "meeting", "meetings"}
    return any(x in t for x in (terms | base))

def fetch_board_playlist_items(youtube, channel_id: str, cutoff_iso: str, oversample: int) -> List[Dict[str, Any]]:
    """Find playlists with board-ish titles and pull their items."""
    pls = _paginate(
        youtube.playlists().list,
        cap=200,
        part="snippet,contentDetails",
        channelId=channel_id,
        maxResults=50,
    )
    boardish_ids = [pl["id"] for pl in pls if is_board_playlist_title((pl.get("snippet", {}) or {}).get("title", ""))]
    if not boardish_ids:
        return []

    collected: List[Dict[str, Any]] = []
    cap_items = min(max(oversample, 1) * 2 * 50, 600)
    for pid in boardish_ids:
        items = _paginate(
            youtube.playlistItems().list,
            cap=cap_items - len(collected),
            part="snippet,contentDetails",
            playlistId=pid,
            maxResults=50,
        )
        items = [it for it in items if _published_after_filter(it, cutoff_iso)]
        collected.extend(items)
        if len(collected) >= cap_items:
            break
    return collected

def fetch_channel_keyword_search(youtube, channel_id: str, cutoff_iso: str, oversample: int) -> List[Dict[str, Any]]:
    """
    Targeted queries to catch common titling patterns. This is the most
    expensive source; call only if you still need more after uploads+playlists.
    """
    queries = [
        "board of trustees",
        "board meeting",
        "trustees meeting",
        "regular meeting",
        "special meeting",
        "budget workshop",
        "work session",
    ]
    cap_total = min(max(oversample, 1) * 2 * 50, 400)
    out: List[Dict[str, Any]] = []
    for q in queries:
        left = cap_total - len(out)
        if left <= 0:
            break
        items = _paginate(
            youtube.search().list,
            cap=left,
            part="snippet",
            channelId=channel_id,
            q=q,
            type="video",
            order="date",
            maxResults=50,
            publishedAfter=cutoff_iso,
            safeSearch="none",
        )
        # Normalize into playlistItems-like shape
        for it in items:
            sn = it.get("snippet", {}) or {}
            vid = (it.get("id", {}) or {}).get("videoId")
            if not vid:
                continue
            out.append({
                "snippet": {
                    "title": sn.get("title", ""),
                    "description": sn.get("description", ""),
                    "resourceId": {"videoId": vid},
                    "publishedAt": sn.get("publishedAt", ""),
                },
                "contentDetails": {"videoPublishedAt": sn.get("publishedAt", "")},
            })
    return out

# ------------------------------
# Metadata gating (one batch call)
# ------------------------------

_METADATA_CACHE: Dict[str, Dict[str, Any]] = {}

def _fetch_metadata_batch(youtube, video_ids: List[str]) -> None:
    """
    Populate _METADATA_CACHE[vid] with:
      {'duration_sec': int,
       'is_live_or_upcoming': bool,
       'is_public': bool,
       'title': str,
       'description': str,
       'publishedAt': iso8601}
    """
    if not video_ids:
        return
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        try:
            resp = youtube.videos().list(
                part="snippet,contentDetails,liveStreamingDetails,status",
                id=",".join(batch),
                maxResults=50,
            ).execute()
        except HttpError as e:
            logger.warning("videos.list metadata failed: %s", e)
            continue

        for v in resp.get("items", []):
            vid = v.get("id")
            if not vid:
                continue
            sn = v.get("snippet", {}) or {}
            cd = v.get("contentDetails", {}) or {}
            lsd = v.get("liveStreamingDetails", {}) or {}
            st = v.get("status", {}) or {}

            # duration PT#H#M#S -> seconds
            dur = 0
            m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", cd.get("duration") or "")
            if m:
                h, m_, s = (int(m.group(1) or 0), int(m.group(2) or 0), int(m.group(3) or 0))
                dur = h * 3600 + m_ * 60 + s

            lbc = (sn.get("liveBroadcastContent") or "none").lower()  # 'none'|'live'|'upcoming'
            is_up = bool(lsd.get("scheduledStartTime")) and not lsd.get("actualStartTime")
            live_flag = (lbc in {"live", "upcoming"}) or is_up

            pub_ok = (st.get("privacyStatus") == "public")

            published = sn.get("publishedAt") or ""
            try:
                # normalize to UTC iso
                published = datetime.fromisoformat(published.replace("Z","+00:00")).astimezone(timezone.utc).isoformat()
            except Exception:
                published = published or ""

            _METADATA_CACHE[vid] = {
                "duration_sec": dur,
                "is_live_or_upcoming": live_flag,
                "is_public": pub_ok,
                "title": sn.get("title", "") or "",
                "description": sn.get("description", "") or "",
                "publishedAt": published,
            }

def _gate_and_score(
    youtube,
    items: List[Dict[str, Any]],
    college: str,
    channel_id: str,
    cutoff_iso: str,
    min_seconds: int,
    seen_ids: Set[str],
    log_reject,
) -> List[Dict[str, Any]]:
    """
    Apply metadata-based gates + scoring and return accepted normalized rows.
    """
    if not items:
        return []

    # Collect unique IDs and fetch metadata for uncached
    ids: List[str] = []
    for it in items:
        vid = _extract_video_id(it)
        if vid and vid not in ids:
            ids.append(vid)
    need = [vid for vid in ids if vid not in _METADATA_CACHE]
    if need:
        _fetch_metadata_batch(youtube, need)

    accepted: List[Dict[str, Any]] = []
    for it in items:
        vid = _extract_video_id(it)
        if not vid or vid in seen_ids:
            continue

        md = _METADATA_CACHE.get(vid, {})
        title = md.get("title") or (it.get("snippet", {}) or {}).get("title", "")
        desc = md.get("description") or (it.get("snippet", {}) or {}).get("description", "")
        publishedAt = md.get("publishedAt") or _published_at_from_item(it)

        # Gates
        if getattr(config, "EXCLUDE_LIVE_AND_UPCOMING", True) and md.get("is_live_or_upcoming", False):
            log_reject({"college": college, "channelId": channel_id, "videoId": vid,
                        "title": title, "publishedAt": publishedAt}, 0, "live_or_upcoming", False, 0, False)
            continue

        if getattr(config, "EXCLUDE_NON_PUBLIC", False) and not md.get("is_public", True):
            log_reject({"college": college, "channelId": channel_id, "videoId": vid,
                        "title": title, "publishedAt": publishedAt}, 0, "non_public", False, 0, False)
            continue

        if min_seconds > 0 and (md.get("duration_sec", 0) < min_seconds):
            log_reject({"college": college, "channelId": channel_id, "videoId": vid,
                        "title": title, "publishedAt": publishedAt}, 0, "too_short", False, 0, False)
            continue

        if publishedAt:
            try:
                # Compare as ISO strings; both normalized to UTC iso when from metadata.
                if publishedAt.replace("+00:00", "Z") < cutoff_iso:
                    log_reject({"college": college, "channelId": channel_id, "videoId": vid,
                                "title": title, "publishedAt": publishedAt}, 0, "outside_cutoff", False, 0, False)
                    continue
            except Exception:
                pass

        # Score
        sc = score_video(title, desc)
        combined = f"{title}\n{desc}".lower()
        matched_priority = any(p.lower() in combined for p in config.PRIORITY_PHRASES)
        kw_hits = sum(1 for kw in (k.lower() for k in config.BOARD_MEETING_KEYWORDS) if kw in combined)
        matched_exclude = any(x.lower() in combined for x in config.EXCLUDE_TERMS)

        if sc < config.MIN_RELEVANCE_SCORE:
            log_reject({"college": college, "channelId": channel_id, "videoId": vid,
                        "title": title, "publishedAt": publishedAt}, sc, "below_min_score",
                       matched_priority, kw_hits, matched_exclude)
            continue

        # Accept
        accepted.append({
            "college": college,
            "channelId": channel_id,
            "videoId": vid,
            "url": f"https://www.youtube.com/watch?v={vid}",
            "title": title,
            "publishedAt": publishedAt,
            "relevance_score": sc,
        })
        seen_ids.add(vid)

    return accepted

# ------------------------------
# Main entry
# ------------------------------

def collect_board_videos() -> None:
    """
    For each channel:
      - Try uploads; if needed, try board playlists; if needed, try channel search.
      - Enforce gates & scoring; stop when MAX_VIDEOS_PER_CHANNEL reached.
      - Write results to VIDEO_LIST_FILE (sorted by relevance, then date).
    """
    logger.info("Starting board meeting video collection…")
    config.ensure_directories()

    if not os.path.exists(config.SCHOOL_IDS_FILE):
        logger.error("Channel IDs file not found at %s. Run channel discovery first.", config.SCHOOL_IDS_FILE)
        return

    channels_df = pd.read_csv(config.SCHOOL_IDS_FILE)
    youtube = get_authenticated_service()

    cutoff_iso = _iso_now_minus_months(config.SEARCH_CUTOFF_MONTHS)
    min_secs = int(getattr(config, "MIN_MEETING_SECONDS", 0))
    per_channel_cap = int(getattr(config, "MAX_VIDEOS_PER_CHANNEL", 10))
    oversample_factor = max(2, per_channel_cap)  # small oversample per source

    all_rows: List[Dict[str, Any]] = []

    # rejected log
    rejected_path = os.path.join(config.DATA_DIR, "video_Rejected.csv")
    rejected_header_written = os.path.exists(rejected_path)

    def log_reject(row: Dict[str, Any], score: int, reason: str,
                   matched_priority: bool = False, kw_hits: int = 0, matched_exclude: bool = False):
        nonlocal rejected_header_written
        with open(rejected_path, "a", newline="", encoding="utf-8") as f:
            import csv
            w = csv.writer(f)
            if not rejected_header_written:
                w.writerow(["college", "channelId", "videoId", "title", "publishedAt",
                            "score", "matched_priority", "kw_hits", "matched_exclude", "reason"])
                rejected_header_written = True
            w.writerow([
                row.get("college",""), row.get("channelId",""), row.get("videoId",""), row.get("title",""), row.get("publishedAt",""),
                score, matched_priority, kw_hits, matched_exclude, reason
            ])

    for _, ch_row in channels_df.iterrows():
        college = ch_row["school_name"]
        channel_id = ch_row["youtube_channel_id"]
        logger.info("Collecting from %s (%s)…", college, channel_id)

        kept: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()

        # ---- 1) uploads ----
        if getattr(config, "ENABLE_SOURCE_UPLOADS", True) and len(kept) < per_channel_cap:
            src = fetch_recent_uploads(youtube, channel_id, cutoff_iso, oversample=oversample_factor)
            accepted = _gate_and_score(youtube, src, college, channel_id, cutoff_iso, min_secs, seen_ids, log_reject)
            if accepted:
                # newest-first ensure
                accepted.sort(key=lambda x: x["publishedAt"], reverse=True)
                space = per_channel_cap - len(kept)
                kept.extend(accepted[:space])

        # ---- 2) board playlists ----
        if getattr(config, "ENABLE_SOURCE_PLAYLISTS", True) and len(kept) < per_channel_cap:
            src = fetch_board_playlist_items(youtube, channel_id, cutoff_iso, oversample=oversample_factor)
            accepted = _gate_and_score(youtube, src, college, channel_id, cutoff_iso, min_secs, seen_ids, log_reject)
            if accepted:
                accepted.sort(key=lambda x: x["publishedAt"], reverse=True)
                space = per_channel_cap - len(kept)
                kept.extend(accepted[:space])

        # ---- 3) channel search (optional / pricier) ----
        if getattr(config, "ENABLE_SOURCE_CHANNEL_SEARCH", False) and len(kept) < per_channel_cap:
            src = fetch_channel_keyword_search(youtube, channel_id, cutoff_iso, oversample=oversample_factor)
            accepted = _gate_and_score(youtube, src, college, channel_id, cutoff_iso, min_secs, seen_ids, log_reject)
            if accepted:
                accepted.sort(key=lambda x: x["publishedAt"], reverse=True)
                space = per_channel_cap - len(kept)
                kept.extend(accepted[:space])

        logger.info("Kept %d videos for %s", len(kept), college)
        all_rows.extend(kept)

    # Final write
    all_rows.sort(key=lambda x: (x["relevance_score"], x["publishedAt"]), reverse=True)
    pd.DataFrame(all_rows).to_csv(config.VIDEO_LIST_FILE, index=False, encoding="utf-8")
    logger.info("Board meeting video collection complete. Wrote %s (%d rows)", config.VIDEO_LIST_FILE, len(all_rows))
