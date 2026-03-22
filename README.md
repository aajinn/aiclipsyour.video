# Aiclipsyour.video

Remove silences and AI-smart-cut your videos. AssemblyAI transcribes, Groq picks the best segments, ffmpeg does the cutting — original audio untouched, perfectly in sync.

---

## Features

- Silence detection and removal
- AI smart cutting — AssemblyAI transcribes, Groq selects semantically complete segments (preserves opinions, emotional lines, complete thoughts)
- Pure ffmpeg export — audio and video cut at identical timestamps, no desync
- Duplicate segment detection — overlapping or repeated ranges are merged before export
- Real-time progress in the web UI with per-stage ETA
- Transcript export (.txt + .json)
- Web UI (FastAPI + Tailwind CSS) or CLI
- API cost breakdown in terminal output

---

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) on PATH

```bash
# Windows
winget install ffmpeg
```

---

## Setup

```bash
cd av_clipper
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```bash
copy .env.example .env
```

```env
ASSEMBLYAI_API_KEY=your-assemblyai-key
GROQ_API_KEY=your-groq-key
```

- AssemblyAI → https://www.assemblyai.com (free tier: 100 hrs/month)
- Groq → https://console.groq.com (free tier available)

Both keys are optional — without them the tool falls back to silence-detection-only mode.

---

## Web UI

```bash
python server.py
```

Open http://localhost:8000, drop in a video (minimum 3 minutes), and hit **Process →**.

Progress updates in real time — each stage appears as it starts, with an estimated time remaining based on your video length.

---

## CLI

```bash
# Silence removal only
python clipper.py input.mp4 output.mp4

# With AI smart cutting
python clipper.py input.mp4 output.mp4 --grok

# Interactive wizard
python clipper.py
```

### Options

```
--silence-thresh   INT     Silence threshold in dBFS (default: -40)
--min-silence      INT     Minimum silence length to cut in ms (default: 500)
--padding          INT     Padding to keep around speech in ms (default: 200)

--grok                     Enable AI smart cutting
--aai-key          STR     AssemblyAI API key (or set in .env)
--grok-key         STR     Groq API key (or set in .env)
--grok-model       STR     Groq model (default: llama-3.3-70b-versatile)
--language         STR     Language code for transcription (default: en)
--speaker-labels           Detect speaker labels
--no-transcript            Skip saving transcript files

--video-codec      STR     libx264 / libx265 / libvpx-vp9 (default: libx264)
--audio-codec      STR     aac / mp3 / libopus (default: aac)
--crf              INT     Quality 0–51, lower=better (default: 23)
--audio-only               Export audio only as .wav
```

---

## How cutting works

1. ffprobe reads the video duration
2. Audio is extracted at 16kHz mono for analysis
3. AssemblyAI transcribes (if keys present), Groq selects segments to keep
4. Silence detection fills in or replaces AI cuts as needed
5. Ranges are deduplicated, sorted, and overlaps merged
6. Each segment is cut from the source with `-c copy` — both audio and video streams at the same timestamps
7. All segments are joined with the ffmpeg concat demuxer and encoded once

Original audio is never filtered or processed — what went in comes out.

---

## API cost estimate

| Service | Pricing (2025) |
|---|---|
| AssemblyAI | $0.37 / hour of audio |
| Groq llama-3.3-70b | $0.59 / M input · $0.79 / M output tokens |
| Groq llama-3.1-8b | $0.05 / M input · $0.08 / M output tokens |

A 10-minute video typically costs < $0.10 total.

---

## File structure

```
av_clipper/
  clipper.py        # core processing + CLI
  server.py         # FastAPI web server
  static/
    index.html      # frontend
  .env              # API keys (gitignored)
  .env.example      # template
  requirements.txt
```
