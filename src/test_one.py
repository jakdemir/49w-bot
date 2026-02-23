"""
End-to-end test on a single video: transcribe + diarize.
Uses faster-whisper (CPU-friendly) + pyannote diarization.

Usage:
    python src/test_one.py --video-id dwYmXADsKCU
    python src/test_one.py --video-id dwYmXADsKCU --hf-token hf_...
"""

import json
import argparse
from pathlib import Path

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torch


def transcribe(audio_path: str, model_size: str = "small") -> list[dict]:
    print(f"Loading Whisper ({model_size}) on CPU...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, language="tr", beam_size=5)
    print(f"Detected language: {info.language} ({info.language_probability:.0%})")
    result = []
    for seg in segments:
        result.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
        print(f"  [{seg.start:.1f}s -> {seg.end:.1f}s] {seg.text.strip()}")
    return result


def diarize(audio_path: str, hf_token: str, num_speakers: int = 3) -> list[dict]:
    print(f"\nLoading diarization pipeline...")
    from huggingface_hub import login
    login(token=hf_token, add_to_git_credential=False)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
    )
    print("Running diarization...")
    diarization = pipeline(audio_path, num_speakers=num_speakers)
    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    return turns


def assign_speakers(segments: list[dict], turns: list[dict]) -> list[dict]:
    """Assign speaker label to each transcript segment by overlap."""
    def find_speaker(start, end):
        best_speaker, best_overlap = "UNKNOWN", 0
        for t in turns:
            overlap = min(end, t["end"]) - max(start, t["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = t["speaker"]
        return best_speaker

    for seg in segments:
        seg["speaker"] = find_speaker(seg["start"], seg["end"])
    return segments


def merge_turns(segments: list[dict]) -> list[dict]:
    """Merge consecutive segments from the same speaker."""
    merged = []
    for seg in segments:
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            merged[-1]["text"] += " " + seg["text"]
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", default="dwYmXADsKCU")
    parser.add_argument("--audio-dir", default="data/audio")
    parser.add_argument("--output-dir", default="data/diarized")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token for pyannote")
    parser.add_argument("--model", default="small", help="Whisper model size (tiny/base/small)")
    parser.add_argument("--no-diarize", action="store_true", help="Skip diarization (transcribe only)")
    args = parser.parse_args()

    audio_path = str(Path(args.audio_dir) / f"{args.video_id}.wav")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n=== Testing pipeline on: {args.video_id} ===\n")

    # Step 1: Transcribe (skip if already done)
    cached = Path(args.output_dir) / f"{args.video_id}.json"
    if cached.exists():
        import json as _json
        print(f"Loading cached transcription from {cached}")
        with open(cached) as f:
            data = _json.load(f)
        segments = data.get("segments", [])
        print(f"Loaded {len(segments)} segments")
    else:
        segments = transcribe(audio_path, args.model)
        print(f"\nTranscribed {len(segments)} segments")

    if not args.no_diarize:
        if not args.hf_token:
            print("\nNo --hf-token provided, skipping diarization.")
            print("Get a free token at: https://huggingface.co/settings/tokens")
            print("Accept model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1")
        else:
            # Step 2: Diarize
            turns = diarize(audio_path, args.hf_token)
            print(f"Diarized {len(turns)} speaker turns")

            # Step 3: Assign + merge
            segments = assign_speakers(segments, turns)
            segments = merge_turns(segments)

            print(f"\n=== Speaker turns (merged) ===")
            for seg in segments[:20]:  # show first 20
                print(f"  [{seg['speaker']}] {seg['text'][:80]}")

    # Save output
    out_file = Path(args.output_dir) / f"{args.video_id}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "video_id": args.video_id,
            "segments": segments,
            "speakers": list(set(s.get("speaker", "?") for s in segments)),
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
