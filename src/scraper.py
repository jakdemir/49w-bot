"""
Fetches all video IDs and transcripts from the 49W YouTube channel.

Usage:
    python src/scraper.py --output data/raw/
    python src/scraper.py --output data/raw/ --limit 50
    python src/scraper.py --channel-url https://www.youtube.com/@49W --output data/raw/
"""

import json
import time
import argparse
import subprocess
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from tqdm import tqdm

DEFAULT_CHANNEL_URL = "https://www.youtube.com/@49W"


def get_all_videos(channel_url: str) -> list[dict]:
    """Fetch all video IDs and titles using yt-dlp (no API key needed)."""
    print(f"Fetching video list from: {channel_url}")
    result = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--print", "%(id)s\t%(title)s", f"{channel_url}/videos"],
        capture_output=True, text=True
    )
    videos = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            videos.append({"video_id": parts[0], "title": parts[1]})
    print(f"Found {len(videos)} videos")
    return videos


def fetch_transcript(video_id: str, languages: list[str] = ["tr", "en"]) -> list[dict] | None:
    """Fetch transcript for a single video. Returns None if unavailable."""
    try:
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=languages)
        return [{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched]
    except (NoTranscriptFound, TranscriptsDisabled):
        return None
    except Exception as e:
        print(f"  Error fetching transcript for {video_id}: {e}")
        return None


def transcript_to_text(transcript: list[dict]) -> str:
    """Convert transcript segments to a single cleaned text string."""
    return " ".join(segment["text"].strip() for segment in transcript)


def scrape_channel(channel_url: str, output_dir: str, limit: int | None = None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    videos_file = output_path / "videos.json"

    # Load existing video list or fetch fresh
    if videos_file.exists():
        print(f"Loading existing video list from {videos_file}")
        with open(videos_file) as f:
            videos = json.load(f)
    else:
        videos = get_all_videos(channel_url)
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
    parser.add_argument("--channel-url", default=DEFAULT_CHANNEL_URL, help="YouTube channel URL")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max number of videos to process")
    args = parser.parse_args()

    scrape_channel(args.channel_url, args.output, args.limit)


if __name__ == "__main__":
    main()
