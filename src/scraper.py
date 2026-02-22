"""
Fetches all video IDs and transcripts from the 49W YouTube channel.

Usage:
    python src/scraper.py --channel-id UC... --output data/raw/
    python src/scraper.py --channel-id UC... --output data/raw/ --limit 50
"""

import os
import json
import time
import argparse
from pathlib import Path

from dotenv import load_dotenv
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from tqdm import tqdm

load_dotenv()


def get_all_video_ids(api_key: str, channel_id: str) -> list[dict]:
    """Fetch all video IDs and titles from a channel using YouTube Data API v3."""
    youtube = build("youtube", "v3", developerKey=api_key)
    videos = []
    next_page_token = None

    print(f"Fetching video list from channel: {channel_id}")

    while True:
        request = youtube.search().list(
            part="id,snippet",
            channelId=channel_id,
            maxResults=50,
            pageToken=next_page_token,
            type="video",
            order="date",
        )
        response = request.execute()

        for item in response["items"]:
            videos.append({
                "video_id": item["id"]["videoId"],
                "title": item["snippet"]["title"],
                "published_at": item["snippet"]["publishedAt"],
                "description": item["snippet"]["description"],
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(0.5)  # respect rate limits

    print(f"Found {len(videos)} videos")
    return videos


def fetch_transcript(video_id: str, languages: list[str] = ["tr", "en"]) -> list[dict] | None:
    """Fetch transcript for a single video. Returns None if unavailable."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        return transcript
    except (NoTranscriptFound, TranscriptsDisabled):
        return None
    except Exception as e:
        print(f"  Error fetching transcript for {video_id}: {e}")
        return None


def transcript_to_text(transcript: list[dict]) -> str:
    """Convert transcript segments to a single cleaned text string."""
    return " ".join(segment["text"].strip() for segment in transcript)


def scrape_channel(channel_id: str, api_key: str, output_dir: str, limit: int | None = None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    videos_file = output_path / "videos.json"

    # Load existing video list or fetch fresh
    if videos_file.exists():
        print(f"Loading existing video list from {videos_file}")
        with open(videos_file) as f:
            videos = json.load(f)
    else:
        videos = get_all_video_ids(api_key, channel_id)
        with open(videos_file, "w", encoding="utf-8") as f:
            json.dump(videos, f, ensure_ascii=False, indent=2)

    if limit:
        videos = videos[:limit]

    transcripts_dir = output_path / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)

    skipped, fetched, failed = 0, 0, 0

    for video in tqdm(videos, desc="Fetching transcripts"):
        vid_id = video["video_id"]
        out_file = transcripts_dir / f"{vid_id}.json"

        if out_file.exists():
            skipped += 1
            continue

        transcript = fetch_transcript(vid_id)

        if transcript is None:
            failed += 1
            continue

        record = {
            "video_id": vid_id,
            "title": video["title"],
            "published_at": video["published_at"],
            "description": video["description"],
            "transcript_raw": transcript,
            "transcript_text": transcript_to_text(transcript),
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        fetched += 1
        time.sleep(0.3)

    print(f"\nDone. Fetched: {fetched}, Skipped (cached): {skipped}, Failed (no transcript): {failed}")


def main():
    parser = argparse.ArgumentParser(description="Scrape 49W YouTube channel transcripts")
    parser.add_argument("--channel-id", default=os.getenv("CHANNEL_ID"), help="YouTube channel ID")
    parser.add_argument("--api-key", default=os.getenv("YOUTUBE_API_KEY"), help="YouTube Data API v3 key")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max number of videos to process")
    args = parser.parse_args()

    if not args.channel_id:
        raise ValueError("Channel ID required. Set CHANNEL_ID in .env or pass --channel-id.")
    if not args.api_key:
        raise ValueError("API key required. Set YOUTUBE_API_KEY in .env or pass --api-key.")

    scrape_channel(args.channel_id, args.api_key, args.output, args.limit)


if __name__ == "__main__":
    main()
