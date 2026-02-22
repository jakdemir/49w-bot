"""
Downloads audio for all 49W videos locally using yt-dlp.
Saves as 16kHz mono WAV files (required format for Whisper).

Usage:
    python src/download_audio.py --output data/audio/
    python src/download_audio.py --output data/audio/ --limit 10
    python src/download_audio.py --video-id dwYmXADsKCU --output data/audio/  # single video
"""

import json
import argparse
import subprocess
from pathlib import Path

from tqdm import tqdm


def download_audio(video_id: str, output_dir: Path) -> bool:
    """Download a single video's audio as 16kHz mono WAV. Returns True on success."""
    out_file = output_dir / f"{video_id}.wav"
    if out_file.exists():
        return True  # already downloaded

    result = subprocess.run(
        [
            "yt-dlp",
            "-x",                          # extract audio only
            "--audio-format", "wav",
            "--audio-quality", "0",        # best quality
            "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",  # 16kHz mono for Whisper
            "-o", str(output_dir / f"{video_id}.%(ext)s"),
            "--no-playlist",
            "--quiet",
            f"https://www.youtube.com/watch?v={video_id}",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  Failed {video_id}: {result.stderr.strip()[:120]}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Download 49W video audio files")
    parser.add_argument("--output", default="data/audio", help="Output directory for WAV files")
    parser.add_argument("--videos", default="data/raw/videos.json", help="Path to videos.json from scraper")
    parser.add_argument("--limit", type=int, default=None, help="Max number of videos to download")
    parser.add_argument("--video-id", default=None, help="Download a single video by ID")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.video_id:
        success = download_audio(args.video_id, output_dir)
        print("Done." if success else "Failed.")
        return

    with open(args.videos, encoding="utf-8") as f:
        videos = json.load(f)

    if args.limit:
        videos = videos[:args.limit]

    print(f"Downloading audio for {len(videos)} videos to {output_dir}/")

    succeeded, skipped, failed = 0, 0, 0
    for video in tqdm(videos, desc="Downloading audio"):
        vid_id = video["video_id"]
        if (output_dir / f"{vid_id}.wav").exists():
            skipped += 1
            continue
        if download_audio(vid_id, output_dir):
            succeeded += 1
        else:
            failed += 1

    print(f"\nDone. Downloaded: {succeeded} | Cached: {skipped} | Failed: {failed}")
    print(f"Total WAV files: {len(list(output_dir.glob('*.wav')))}")


if __name__ == "__main__":
    main()
