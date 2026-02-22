# 49W Bot

Fine-tunes an LLM on 49W YouTube channel transcripts to generate video content from a given title.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in YOUTUBE_API_KEY and CHANNEL_ID
```

## Pipeline

### 1. Scrape transcripts
```bash
python src/scraper.py --channel-id UC... --output data/raw/
```
Fetches all video titles + transcripts from the channel. Results cached in `data/raw/transcripts/`.

### 2. Process into training format
```bash
python src/processor.py --input data/raw/transcripts/ --output data/processed/
```
Outputs `data/processed/train.jsonl` and `data/processed/val.jsonl` in chat format.

### 3. Fine-tune
```bash
python src/train.py --data data/processed/ --output models/49w-v1
```
Uses QLoRA (4-bit) on `mistralai/Mistral-7B-Instruct-v0.3` by default. Requires a GPU with ~12GB VRAM.

### 4. Generate
```bash
python src/generate.py --model models/49w-v1 --title "Yapay Zeka ile Para Kazanmak"
```

## Getting a YouTube API Key
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable **YouTube Data API v3**
3. Create an API key and add it to `.env`

## Finding Your Channel ID
The channel ID starts with `UC`. You can find it in the channel's URL or via [this tool](https://www.youtube.com/account_advanced).
