"""
Functions for discovering official YouTube channels for a list of colleges.

This module uses the YouTube Data API (v3) to search for channels matching
each college name provided in `config.COLLEGES`.  Each candidate channel
receives a weighted score based on a number of heuristics (see
`score_channel`).  The top‑scoring channel for each college is written to
`data/school_IDs.csv` with its name, ID and confidence score.

Authentication is handled via OAuth2.  A browser window will open on the
first run to complete the consent flow.  The resulting token is stored in
`token.json` and reused on subsequent runs.
"""

from __future__ import annotations

import logging
import os
import re
import csv
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher
from typing import List, Dict, Tuple

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import config


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_authenticated_service() -> object:
    """Return an authenticated YouTube API client.

    Uses OAuth2 credentials stored in `config.TOKEN_FILE`.  If the token
    doesn't exist or is invalid, triggers the OAuth flow using the
    credentials specified in `config.YOUTUBE_API_KEY_FILE`.
    """
    creds = None
    # Load existing token if available
    if os.path.exists(config.TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(config.TOKEN_FILE, config.YOUTUBE_SCOPES)
        except Exception as ex:
            logger.warning("Failed to load existing token: %s", ex)
            creds = None
    # Refresh or obtain new token if necessary
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired token…")
            creds.refresh(Request())
        else:
            logger.info("Initiating OAuth flow.  A browser window will open for authentication.")
            flow = InstalledAppFlow.from_client_secrets_file(config.YOUTUBE_API_KEY_FILE, config.YOUTUBE_SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(config.TOKEN_FILE, "w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())
    # Build the API client
    return build("youtube", "v3", credentials=creds)


def fetch_latest_upload_date(youtube: object, channel_id: str) -> datetime | None:
    """Retrieve the published date of the most recent upload for a channel.

    Parameters
    ----------
    youtube : object
        An authenticated YouTube API client.
    channel_id : str
        The ID of the YouTube channel.

    Returns
    -------
    datetime or None
        The publication date of the most recent upload in UTC, or None if
        no videos are found.
    """
    try:
        response = youtube.search().list(
            channelId=channel_id,
            part="snippet",
            order="date",
            type="video",
            maxResults=1,
        ).execute()
    except HttpError as e:
        logger.error("HTTP error while fetching latest upload for channel %s: %s", channel_id, e)
        return None
    items = response.get("items", [])
    if not items:
        return None
    published_at = items[0]["snippet"].get("publishedAt")
    try:
        return datetime.fromisoformat(published_at.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def string_similarity(a: str, b: str) -> float:
    """Compute a similarity ratio between two strings using difflib.

    The ratio is between 0 and 1, where 1 indicates identical strings.
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def score_channel(college: Dict[str, str], channel_info: Dict[str, object], youtube: object) -> int:
    """Calculate a weighted relevance score for a YouTube channel.

    The scoring is based on heuristics specified in the project
    requirements.  High‑weight factors contribute 3 points, medium‑weight
    factors contribute 2 points, and low‑weight factors contribute 1 point.

    Parameters
    ----------
    college : dict
        A dictionary containing `name` and `website` of the college.
    channel_info : dict
        The channel resource returned from the YouTube Data API.
    youtube : object
        An authenticated YouTube API client used to fetch the latest upload
        date when required.

    Returns
    -------
    int
        The computed relevance score.
    """
    score = 0
    snippet = channel_info.get("snippet", {})
    statistics = channel_info.get("statistics", {})
    branding = channel_info.get("brandingSettings", {}).get("channel", {})

    channel_title = snippet.get("title", "")
    description = snippet.get("description", "") or ""
    custom_url = snippet.get("customUrl", "") or ""
    website = college.get("website", "").rstrip("/")

    # High weight: channel URL/handle contains .edu
    if ".edu" in custom_url.lower():
        score += 3

    # High weight: description contains exact website link
    if website.lower() in description.lower():
        score += 3

    # High weight: verification badge (approximate by checking for the word "verified" in description)
    # YouTube's API does not expose verification directly, so this heuristic
    # checks the description or keywords for the term.
    if re.search(r"\bverified\b", description, re.IGNORECASE):
        score += 3

    # Medium weight: channel name exactly matches school name
    if channel_title.strip().lower() == college["name"].strip().lower():
        score += 2

    # Medium weight: channel name contains both school name and "college"/"community college"
    if college["name"].split()[0].lower() in channel_title.lower() and "college" in channel_title.lower():
        score += 2

    # Medium weight: substantial subscriber count (>1000)
    try:
        subs = int(statistics.get("subscriberCount", "0"))
        if subs > 1000:
            score += 2
    except (TypeError, ValueError):
        pass

    # Low weight: description contains education keywords
    edu_keywords = ["education", "college", "university", "learning", "student"]
    if any(kw in description.lower() for kw in edu_keywords):
        score += 1

    # Low weight: partial name similarity (>70%)
    if string_similarity(channel_title, college["name"]) > 0.7:
        score += 1

    # Low weight: recent uploads within last 6 months
    latest_upload = fetch_latest_upload_date(youtube, channel_info.get("id"))
    if latest_upload is not None:
        if latest_upload > datetime.now(timezone.utc) - timedelta(days=180):
            score += 1

    return score


def discover_channels() -> None:
    """Discover the best YouTube channel for each college.

    Performs a channel search for every entry in `config.COLLEGES`, scores
    the results, and writes the top candidate per school to
    `config.SCHOOL_IDS_FILE`.  Existing CSVs are overwritten.
    """
    logger.info("Starting channel discovery…")
    youtube = get_authenticated_service()
    rows = []
    # Loop over each college and perform a channel search
    for college in config.COLLEGES:
        name = college["name"]
        website = college["website"]
        logger.info("Searching channels for %s", name)
        try:
            search_response = youtube.search().list(
                q=name,
                type="channel",
                part="id,snippet",
                maxResults=10,
            ).execute()
        except HttpError as e:
            logger.error("HTTP error during channel search for %s: %s", name, e)
            continue
        items = search_response.get("items", [])
        if not items:
            logger.warning("No channels found for %s", name)
            continue
        scored_channels: List[Tuple[Dict[str, object], int]] = []
        # Retrieve full details for each candidate channel and compute a score
        for item in items:
            channel_id = item["id"]["channelId"]
            try:
                channel_response = youtube.channels().list(
                    id=channel_id,
                    part="snippet,statistics,brandingSettings",
                ).execute()
            except HttpError as e:
                logger.error("HTTP error while fetching channel %s details: %s", channel_id, e)
                continue
            channel_items = channel_response.get("items", [])
            if not channel_items:
                continue
            channel_info = channel_items[0]
            # The channel ID is available in the top level, but ensure it's present
            channel_info["id"] = channel_id
            score = score_channel(college, channel_info, youtube)
            scored_channels.append((channel_info, score))
        if not scored_channels:
            logger.warning("No scored channels available for %s", name)
            continue
        # Select the channel with the highest score
        scored_channels.sort(key=lambda tup: tup[1], reverse=True)
        best_channel_info, best_score = scored_channels[0]
        rows.append({
            "school_name": name,
            "website": website,
            "youtube_channel_id": best_channel_info.get("id"),
            "youtube_channel_name": best_channel_info.get("snippet", {}).get("title", ""),
            "confidence_score": best_score,
        })
        logger.info("Selected channel %s with score %d for %s", best_channel_info.get("snippet", {}).get("title", ""), best_score, name)
    # Write results to CSV
    df = pd.DataFrame(rows)
    config.ensure_directories()
    df.to_csv(config.SCHOOL_IDS_FILE, index=False)
    logger.info("Channel discovery complete. Results written to %s", config.SCHOOL_IDS_FILE)