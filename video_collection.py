"""
Module for discovering board meeting videos on YouTube.

Given a set of confirmed college channels (produced by
`youtube_discovery.py`), this module fetches recent videos from each
channel, scores them based on their titles and descriptions against a
configured keyword list, and writes a list of likely board meeting
recordings to `data/video_List.csv`.

Videos that score below `config.MIN_RELEVANCE_SCORE` or contain
excluded terms are not included in the output.  The resulting CSV
contains the college name, channel ID, video ID, YouTube URL, title,
publication date, and the computed relevance score.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple
import os
import re

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

def _iso_now_minus_months(months: int) -> str:
    # Approximate months as 30 days — good enough for a rolling cutoff
    dt = datetime.now(timezone.utc) - timedelta(days=30 * months)
    return dt.strftime(ISO8601_FMT)

def _paginate_youtube(method, **kwargs):
    """Generic paginator for YouTube API list endpoints (search, playlistItems, playlists)."""
    items = []
    while True:
        resp = method(**kwargs).execute()
        items.extend(resp.get("items", []))
        token = resp.get("nextPageToken")
        if not token:
            break
        kwargs["pageToken"] = token
    return items

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def published_after_filter(item, cutoff_iso: str) -> bool:
    # Try snippet.publishedAt, fall back to contentDetails.videoPublishedAt if present
    sn = item.get("snippet", {}) or {}
    cd = item.get("contentDetails", {}) or {}
    pa = sn.get("publishedAt") or cd.get("videoPublishedAt")
    if not pa:
        return True  # keep if unknown
    try:
        return pa >= cutoff_iso
    except Exception:
        return True

def is_board_playlist_title(title: str, keywords: list[str]) -> bool:
    t = _normalize(title)
    return any(kw in t for kw in keywords + ["board", "trustee", "trustees", "meeting", "meetings"])

def fetch_channel_keyword_search(youtube, channel_id: str, keywords: list[str], published_after_iso: str, max_per_query: int = 50):
    # A few queries to catch typical board naming variants
    queries = [
        "board of trustees",
        "board meeting",
        "trustees meeting",
        "regular meeting",
        "special meeting",
        "workshop board",
        "budget workshop",
    ]
    results = []
    for q in queries:
        items = _paginate_youtube(
            youtube.search().list,
            part="snippet",
            channelId=channel_id,
            q=q,
            maxResults=min(max_per_query, 50),
            order="date",
            publishedAfter=published_after_iso,
            type="video",
            safeSearch="none",
        )
        results.extend(items)
    return results

def fetch_board_playlists_items(youtube, channel_id: str, keywords: list[str], published_after_iso: str, max_playlists: int = 20, max_items_per_pl: int = 200):
    # 1) Get channel playlists
    playlists = _paginate_youtube(
        youtube.playlists().list,
        part="snippet,contentDetails",
        channelId=channel_id,
        maxResults=min(max_playlists, 50),
    )
    # 2) Keep only board-like playlists
    board_pls = [pl for pl in playlists if is_board_playlist_title(pl.get("snippet", {}).get("title", ""), keywords)]
    # 3) For each board playlist, fetch items
    pl_items = []
    for pl in board_pls:
        pl_id = pl.get("id")
        if not pl_id:
            continue
        items = _paginate_youtube(
            youtube.playlistItems().list,
            part="snippet,contentDetails",
            playlistId=pl_id,
            maxResults=min(max_items_per_pl, 50),
        )
        # Apply date cutoff (based on contentDetails.videoPublishedAt when available)
        items = [it for it in items if published_after_filter(it, published_after_iso)]
        pl_items.extend(items)
    return pl_items

def merge_and_dedupe_items(*groups):
    by_vid = {}
    for group in groups:
        for it in group:
            # Normalize to have a videoId at the top-level for de-dupe
            sn = it.get("snippet", {}) or {}
            rid = (sn.get("resourceId") or {}) if isinstance(sn.get("resourceId"), dict) else {}
            vid = rid.get("videoId") or it.get("id", {}).get("videoId") or it.get("id")
            if isinstance(vid, dict):  # e.g., search().list items return id:{kind, videoId}
                vid = vid.get("videoId")
            if not vid:
                continue
            if vid not in by_vid:
                by_vid[vid] = it
    return list(by_vid.values())


def _extract_video_id(item: Dict[str, Any]) -> str | None:
    sn = item.get("snippet", {}) or {}
    rid = sn.get("resourceId") if isinstance(sn.get("resourceId"), dict) else {}
    vid = rid.get("videoId") or item.get("id", {}).get("videoId") or item.get("id")
    if isinstance(vid, dict):  # search().list returns id:{kind, videoId}
        vid = vid.get("videoId")
    return vid

def filter_completed_videos(youtube, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove videos that are live or upcoming (planned livestreams/premieres).
    Also optionally drop non-public videos.

    Strategy: batch call videos.list(part=snippet,liveStreamingDetails,status)
    and filter on:
      - snippet.liveBroadcastContent == "none"
      - NOT (liveStreamingDetails.scheduledStartTime present without actualStartTime)
      - (optional) status.privacyStatus == "public"
    """
    if not items:
        return items

    # Collect unique IDs
    vids = []
    seen = set()
    for it in items:
        v = _extract_video_id(it)
        if v and v not in seen:
            seen.add(v)
            vids.append(v)

    # Batch in chunks of 50
    keep_ids = set()
    for i in range(0, len(vids), 50):
        batch = vids[i:i+50]
        try:
            resp = youtube.videos().list(
                part="snippet,liveStreamingDetails,status",
                id=",".join(batch),
                maxResults=50,
            ).execute()
        except HttpError as e:
            # If the lookup fails, conservatively keep the batch to avoid false negatives
            logger.warning("videos.list failed, keeping batch unfiltered: %s", e)
            keep_ids.update(batch)
            continue

        for v in resp.get("items", []):
            vid = v.get("id")
            sn = v.get("snippet", {}) or {}
            lbc = (sn.get("liveBroadcastContent") or "none").lower()   # "none" | "live" | "upcoming"
            lsd = v.get("liveStreamingDetails", {}) or {}
            status = v.get("status", {}) or {}

            # Exclude live or upcoming
            is_upcoming_by_flag = lbc in {"live", "upcoming"}
            is_upcoming_by_time = bool(lsd.get("scheduledStartTime")) and not lsd.get("actualStartTime")

            # Exclude non-public if requested
            non_public = (status.get("privacyStatus") != "public") if getattr(config, "EXCLUDE_NON_PUBLIC", False) else False

            if not getattr(config, "EXCLUDE_LIVE_AND_UPCOMING", True):
                is_upcoming_by_flag = False
                is_upcoming_by_time = False

            if not (is_upcoming_by_flag or is_upcoming_by_time or non_public):
                keep_ids.add(vid)

    # Return only items whose IDs survived the filter
    filtered = []
    keep_set = keep_ids
    for it in items:
        v = _extract_video_id(it)
        if v and v in keep_set:
            filtered.append(it)
    return filtered


def get_authenticated_service() -> object:
    """Return an authenticated YouTube API client.

    This duplicates the helper used in `youtube_discovery.py` to avoid
    circular imports.  Both modules store the token in the same file.
    """
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


def contains_excluded_terms(text: str) -> bool:
    """Return True if the text contains any excluded term."""
    text_lower = text.lower()
    return any(term in text_lower for term in config.EXCLUDE_TERMS)


def count_keywords(text: str, keywords: List[str]) -> int:
    """Count how many keywords appear in the given text (case insensitive)."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def contains_priority_phrase(text: str) -> bool:
    """Return True if any high‑priority phrase appears in the text."""
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in config.PRIORITY_PHRASES)


def score_video(title: str, description: str) -> int:
    """Compute a relevance score for a video based on its title/description."""
    # Exclusion: immediate skip if excluded term present
    combined = f"{title}\n{description}"
    if contains_excluded_terms(combined):
        return -5
    score = 0
    if contains_priority_phrase(combined):
        score += 3
    keyword_count = count_keywords(combined, config.BOARD_MEETING_KEYWORDS)
    if keyword_count >= 2:
        score += 2
    elif keyword_count == 1:
        score += 1
    return score


def collect_board_videos() -> None:
    """
    Fetch recent videos for each college channel from three sources:
      1) uploads playlist (most recent uploads)        -> ENABLE_SOURCE_UPLOADS
      2) channel-scoped keyword search                 -> ENABLE_SOURCE_CHANNEL_SEARCH
      3) board-related playlists -> playlist items     -> ENABLE_SOURCE_PLAYLISTS

    Then: merge & dedupe, optionally drop live/upcoming, score, filter, write CSV.
    """
    logger.info("Starting board meeting video collection…")
    config.ensure_directories()

    if not os.path.exists(config.SCHOOL_IDS_FILE):
        logger.error("Channel IDs file not found at %s. Run channel discovery first.", config.SCHOOL_IDS_FILE)
        return

    channels_df = pd.read_csv(config.SCHOOL_IDS_FILE)
    youtube = get_authenticated_service()

    cutoff_iso = _iso_now_minus_months(config.SEARCH_CUTOFF_MONTHS)
    all_rows: List[Dict[str, Any]] = []

    # Optional rejected logging
    rejected_path = os.path.join(config.DATA_DIR, "video_Rejected.csv")
    rejected_header_written = os.path.exists(rejected_path)

    def log_reject(row: Dict[str, Any], score: int, reason: str, matched_priority: bool, kw_hits: int, matched_exclude: bool):
        nonlocal rejected_header_written
        with open(rejected_path, "a", newline="", encoding="utf-8") as f:
            import csv
            w = csv.writer(f)
            if not rejected_header_written:
                w.writerow(["college", "channelId", "videoId", "title", "publishedAt",
                            "score", "matched_priority", "kw_hits", "matched_exclude", "reason"])
                rejected_header_written = True
            w.writerow([
                row["college"], row["channelId"], row["videoId"], row["title"], row["publishedAt"],
                score, matched_priority, kw_hits, matched_exclude, reason
            ])

    def _extract_row(item: Dict[str, Any], college_name: str, channel_id: str) -> Dict[str, Any]:
        sn = item.get("snippet", {}) or {}
        rid = (sn.get("resourceId") or {}) if isinstance(sn.get("resourceId"), dict) else {}
        vid = rid.get("videoId") or item.get("id", {}).get("videoId") or item.get("id")
        if isinstance(vid, dict):  # search().list returns id:{kind,videoId}
            vid = vid.get("videoId")

        title = sn.get("title", "") or ""
        desc = sn.get("description", "") or ""
        published_at = sn.get("publishedAt") or item.get("contentDetails", {}).get("videoPublishedAt") or ""

        try:
            published_at_iso = datetime.fromisoformat(published_at.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
        except Exception:
            published_at_iso = published_at or ""

        return {
            "college": college_name,
            "channelId": channel_id,
            "videoId": vid,
            "url": f"https://www.youtube.com/watch?v={vid}" if vid else "",
            "title": title,
            "description": desc,
            "publishedAt": published_at_iso,
        }

    def _fetch_recent_uploads(youtube, channel_id: str, cutoff_iso: str, max_items: int = 200) -> List[Dict[str, Any]]:
        try:
            ch = youtube.channels().list(part="contentDetails", id=channel_id, maxResults=1).execute()
            items = ch.get("items", [])
            if not items:
                return []
            uploads_pl = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
            pl_items = _paginate_youtube(
                youtube.playlistItems().list,
                part="snippet,contentDetails",
                playlistId=uploads_pl,
                maxResults=min(max_items, 50),
            )
            return [it for it in pl_items if published_after_filter(it, cutoff_iso)]
        except HttpError as e:
            logger.error("Error fetching uploads for channel %s: %s", channel_id, e)
            return []

    for _, ch_row in channels_df.iterrows():
        college_name = ch_row["school_name"]
        channel_id = ch_row["youtube_channel_id"]
        logger.info("Collecting from %s (%s)…", college_name, channel_id)

        # Source A: uploads (toggle)
        uploads: List[Dict[str, Any]] = []
        if getattr(config, "ENABLE_SOURCE_UPLOADS", True):
            uploads = _fetch_recent_uploads(youtube, channel_id, cutoff_iso, max_items=config.MAX_VIDEOS_PER_CHANNEL * 4)

        # Source B: channel keyword search (toggle)
        chan_search: List[Dict[str, Any]] = []
        if getattr(config, "ENABLE_SOURCE_CHANNEL_SEARCH", True):
            chan_search = fetch_channel_keyword_search(
                youtube, channel_id, config.BOARD_MEETING_KEYWORDS, cutoff_iso, max_per_query=config.MAX_VIDEOS_PER_CHANNEL
            )

        # Source C: board playlists (toggle)
        pl_items: List[Dict[str, Any]] = []
        if getattr(config, "ENABLE_SOURCE_PLAYLISTS", True):
            pl_items = fetch_board_playlists_items(
                youtube, channel_id, config.BOARD_MEETING_KEYWORDS, cutoff_iso, max_playlists=20, max_items_per_pl=200
            )

        # Merge & de-dupe
        candidates = merge_and_dedupe_items(uploads, chan_search, pl_items)

        # ---- NEW: drop live/upcoming BEFORE scoring (toggle) ----
        if getattr(config, "EXCLUDE_LIVE_AND_UPCOMING", True) and candidates:
            before = len(candidates)
            candidates = filter_completed_videos(youtube, candidates)
            after = len(candidates)
            if before != after:
                logger.info("Filtered live/upcoming: %d -> %d for %s", before, after, college_name)

        kept_for_channel = 0
        for it in candidates:
            row = _extract_row(it, college_name=college_name, channel_id=channel_id)
            if not row.get("videoId"):
                continue

            # Date cutoff (belt & suspenders)
            try:
                if row["publishedAt"]:
                    dtp = datetime.fromisoformat(row["publishedAt"].replace("Z", "+00:00"))
                    if dtp.tzinfo is None:
                        dtp = dtp.replace(tzinfo=timezone.utc)
                    if dtp.isoformat() < cutoff_iso:
                        log_reject(row, score=0, reason="outside_cutoff", matched_priority=False, kw_hits=0, matched_exclude=False)
                        continue
            except Exception:
                pass

            score = score_video(row["title"], row.get("description", ""))

            combined = f"{row['title']}\n{row.get('description','')}".lower()
            matched_priority = any(p in combined for p in (s.lower() for s in config.PRIORITY_PHRASES))
            kw_hits = sum(1 for kw in (k.lower() for k in config.BOARD_MEETING_KEYWORDS) if kw in combined)
            matched_exclude = any(e in combined for e in (x.lower() for x in config.EXCLUDE_TERMS))

            if score < config.MIN_RELEVANCE_SCORE:
                log_reject(row, score=score, reason="below_min_score", matched_priority=matched_priority, kw_hits=kw_hits, matched_exclude=matched_exclude)
                continue

            all_rows.append({
                "college": row["college"],
                "channelId": row["channelId"],
                "videoId": row["videoId"],
                "url": row["url"],
                "title": row["title"],
                "publishedAt": row["publishedAt"],
                "relevance_score": score,
            })
            kept_for_channel += 1

        logger.info("Kept %d videos for %s", kept_for_channel, college_name)

    # Sort & write
    all_rows.sort(key=lambda x: (x["relevance_score"], x["publishedAt"]), reverse=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(config.VIDEO_LIST_FILE, index=False, encoding="utf-8")
    logger.info("Board meeting video collection complete. Wrote %s (%d rows)", config.VIDEO_LIST_FILE, len(df))

