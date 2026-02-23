"""
Batch transcribes all audio files locally using faster-whisper (CPU).
Saves transcription JSON per video to data/transcribed/

Usage:
    python src/transcribe.py --audio-dir data/audio --output data/transcribed
    python src/transcribe.py --audio-dir data/audio --output data/transcribed --model small
"""

import json
import argparse
import os
from pathlib import Path

from faster_whisper import WhisperModel
from tqdm import tqdm


def transcribe_file(model: WhisperModel, audio_path: str) -> list[dict]:
    segments, _ = model.transcribe(audio_path, language="tr", beam_size=5)
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]


def main():
    parser = argparse.ArgumentParser(description="Batch transcribe 49W audio files")
    parser.add_argument("--audio-dir", default="data/audio")
    parser.add_argument("--output", default="data/transcribed")
    parser.add_argument("--videos", default="data/raw/videos.json")
    parser.add_argument("--model", default="small", help="Whisper model: tiny, base, small, medium")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(Path(args.audio_dir).glob("*.wav"))
    if args.limit:
        audio_files = audio_files[:args.limit]

    # Load video metadata for titles
    with open(args.videos, encoding="utf-8") as f:
        video_meta = {v["video_id"]: v for v in json.load(f)}

    print(f"Loading Whisper ({args.model}) on CPU...")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    print(f"Transcribing {len(audio_files)} files...\n")

    done, skipped, failed = 0, 0, 0

    for audio_path in tqdm(audio_files, desc="Transcribing"):
        vid_id = audio_path.stem
        out_file = output_path / f"{vid_id}.json"

        if out_file.exists():
            skipped += 1
            continue

        try:
            segments = transcribe_file(model, str(audio_path))
            meta = video_meta.get(vid_id, {})
            record = {
                "video_id": vid_id,
                "title": meta.get("title", ""),
                "segments": segments,
            }
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            done += 1
        except Exception as e:
            print(f"  Failed {vid_id}: {e}")
            failed += 1

    print(f"\nDone. Transcribed: {done} | Cached: {skipped} | Failed: {failed}")


if __name__ == "__main__":
    main()
