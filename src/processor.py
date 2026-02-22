"""
Processes raw transcripts into fine-tuning format.

Each training example = one video:
  - prompt: video title
  - completion: transcript text

Usage:
    python src/processor.py --input data/raw/transcripts/ --output data/processed/
"""

import json
import argparse
from pathlib import Path

from tqdm import tqdm


SYSTEM_PROMPT_CHANNEL = (
    "Sen 49W YouTube kanalındaki bir konuşmacısın. "
    "Verilen başlığa göre, kanaldaki video içeriğini o kişilerin üslubu ve bilgisiyle üret."
)

SYSTEM_PROMPT_PERSONA = (
    "Sen {name} adlı 49W YouTube kanalı sunucusunun yapay zeka versiyonusun. "
    "Verilen başlık için {name}'in tarzı ve bilgisiyle içerik üret."
)


def build_training_example(record: dict) -> list[dict]:
    """
    Convert a record into training examples.
    - If the record has speaker turns (identified), creates one example per host.
    - Otherwise falls back to a single channel-level example.
    """
    title = record["title"].strip()
    if not title:
        return []

    examples = []

    # Speaker-identified format (from notebook 03)
    if "turns" in record:
        # Group turns by speaker
        speaker_texts: dict[str, list[str]] = {}
        for turn in record["turns"]:
            speaker = turn["speaker"]
            if speaker == "Unknown":
                continue
            speaker_texts.setdefault(speaker, []).append(turn["text"].strip())

        for speaker, texts in speaker_texts.items():
            text = " ".join(texts)
            if len(text.split()) < 50:
                continue
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_PERSONA.format(name=speaker)},
                    {"role": "user", "content": f"Başlık: {title}"},
                    {"role": "assistant", "content": text},
                ],
                "metadata": {"video_id": record["video_id"], "title": title, "speaker": speaker,
                             "word_count": len(text.split())},
            })

    # Fallback: no speaker info, use full transcript
    if not examples and "transcript_text" in record:
        text = record["transcript_text"].strip()
        if text:
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_CHANNEL},
                    {"role": "user", "content": f"Başlık: {title}"},
                    {"role": "assistant", "content": text},
                ],
                "metadata": {"video_id": record["video_id"], "title": title, "speaker": "channel",
                             "word_count": len(text.split())},
            })

    return examples


def process(input_dir: str, output_dir: str, min_words: int = 100, max_words: int = 8000):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = list(input_path.glob("*.json"))
    print(f"Processing {len(files)} files from {input_dir}...")

    examples = []
    skipped_short = 0
    skipped_long = 0

    for file in tqdm(files):
        with open(file, encoding="utf-8") as f:
            record = json.load(f)

        for example in build_training_example(record):
            word_count = example["metadata"]["word_count"]

            if word_count < min_words:
                skipped_short += 1
                continue
            if word_count > max_words:
                words = example["messages"][2]["content"].split()[:max_words]
                example["messages"][2]["content"] = " ".join(words)
                example["metadata"]["word_count"] = max_words
                skipped_long += 1

            examples.append(example)

    # Split 90/10 train/val
    split = int(len(examples) * 0.9)
    train_set = examples[:split]
    val_set = examples[split:]

    def write_jsonl(data: list, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_jsonl(train_set, output_path / "train.jsonl")
    write_jsonl(val_set, output_path / "val.jsonl")

    # Write stats
    stats = {
        "total_processed": len(examples),
        "train": len(train_set),
        "val": len(val_set),
        "skipped_too_short": skipped_short,
        "truncated_too_long": skipped_long,
        "avg_word_count": int(sum(e["metadata"]["word_count"] for e in examples) / len(examples)) if examples else 0,
    }

    with open(output_path / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone.")
    print(f"  Train examples : {len(train_set)}")
    print(f"  Val examples   : {len(val_set)}")
    print(f"  Too short (<{min_words} words): {skipped_short}")
    print(f"  Truncated (>{max_words} words): {skipped_long}")
    print(f"  Avg word count : {stats['avg_word_count']}")


def main():
    parser = argparse.ArgumentParser(description="Process transcripts into fine-tuning format")
    parser.add_argument("--input", default="data/raw/transcripts", help="Input directory with transcript JSONs")
    parser.add_argument("--output", default="data/processed", help="Output directory for JSONL files")
    parser.add_argument("--min-words", type=int, default=100, help="Skip transcripts shorter than this")
    parser.add_argument("--max-words", type=int, default=8000, help="Truncate transcripts longer than this")
    args = parser.parse_args()

    process(args.input, args.output, args.min_words, args.max_words)


if __name__ == "__main__":
    main()
