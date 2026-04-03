# aiclipsyour.video

Remove silences and AI-smart-cut your videos. AssemblyAI transcribes, Groq picks the best segments, ffmpeg does the cutting — original audio untouched, perfectly in sync.

---

## How it works

1. Upload a video (min 3 minutes) via the web UI
2. ffprobe reads the duration; audio is extracted at 16kHz mono
3. AssemblyAI transcribes the audio (if keys are set)
4. Groq selects the best segments to keep using a senior-editor system prompt
5. Silence detection fills in or replaces AI cuts as needed
6. Ranges are deduplicated, sorted, and overlaps merged
7. Each segment is cut from the source with `-c copy` — both streams at identical timestamps
8. All segments are joined with the ffmpeg concat demuxer and encoded once
9. The output file is streamed back as a single-use download

Original audio is never filtered or re-encoded during cutting.

---

## Stack

| Layer | Tech |
|---|---|
| Frontend | Next.js 15, React 19, Tailwind CSS 4 |
| Backend | Node.js, Express 4 |
| Processing | ffmpeg / ffprobe, AssemblyAI, Groq |
| Deployment | Vercel (frontend), any Node host (backend) |

---

## Requirements

- Node.js 18+
- [ffmpeg](https://ffmpeg.org/download.html) on PATH

```bash
# Windows
winget install ffmpeg
```

---

## Setup

```bash
# Backend
cd backend
npm install

# Frontend
cd frontend
npm install
```

Copy `.env.example` to `backend/.env` and fill in your keys:

```env
ASSEMBLYAI_API_KEY=your-assemblyai-key
GROQ_API_KEY=your-groq-key

# CORS — comma-separated list of allowed origins
ALLOWED_ORIGIN=http://localhost:3000
```

Both API keys are optional. Without them the tool falls back to silence-detection-only mode.

- AssemblyAI → https://www.assemblyai.com (free tier: 100 hrs/month)
- Groq → https://console.groq.com (free tier available)

---

## Running locally

```bash
# Backend (port 8000)
cd backend
npm run dev

# Frontend (port 3000)
cd frontend
npm run dev
```

Open http://localhost:3000, drop in a video, and hit Process.

---

## API

All endpoints are prefixed `/api`.

| Method | Path | Description |
|---|---|---|
| `POST` | `/process` | Upload video, queue job → `{ job_id, token, duration }` |
| `GET` | `/stream/:jobId?token=` | SSE progress stream |
| `GET` | `/download/:jobId?token=` | Single-use output download (deletes file after) |
| `GET` | `/status/:jobId?token=` | Poll job status (JSON) |
| `GET` | `/health` | Service health + queue capacity |

Jobs are token-protected. The token is returned on upload and must be passed as a query param on all subsequent requests.

### SSE event types

```json
{ "type": "stage",    "stage": "Transcribing speech" }
{ "type": "done" }
{ "type": "error",    "msg": "Processing failed — check server logs." }
{ "type": "duration", "duration": 312.4 }
```

---

## Configuration

All backend config is via environment variables.

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | HTTP port |
| `ALLOWED_ORIGIN` | `*` | CORS origin(s), comma-separated |
| `MAX_JOBS` | `10` | Max concurrent processing jobs |
| `MAX_UPLOAD_MB` | `500` | Upload size limit |
| `JOB_TTL` | `3600` | Seconds before a job is cleaned up |
| `RATE_LIMIT_UPLOADS` | `5` | Max uploads per IP per hour |
| `RATE_LIMIT_STREAM` | `30` | Max download requests per IP per hour |
| `REQUEST_TIMEOUT` | `300` | Request timeout in seconds |
| `UPLOAD_DIR` | `./uploads` | Where uploads are stored |
| `OUTPUT_DIR` | `./outputs` | Where processed files are stored |
| `ASSEMBLYAI_API_KEY` | — | AssemblyAI key (enables transcription) |
| `GROQ_API_KEY` | — | Groq key (enables AI smart cutting) |

Frontend config:

| Variable | Description |
|---|---|
| `NEXT_PUBLIC_BACKEND_URL` | Backend base URL, set at build time |

---

## Processing defaults

| Parameter | Default | Description |
|---|---|---|
| `silenceThresh` | `-45 dBFS` | Silence threshold |
| `minSilenceLen` | `1500 ms` | Minimum silence length to cut |
| `paddingMs` | `500 ms` | Buffer kept around speech edges |
| `videoCodec` | `libx264` | Output video codec |
| `audioCodec` | `aac` | Output audio codec |
| `crf` | `23` | Quality (0–51, lower = better) |
| `groqModel` | `llama-3.3-70b-versatile` | Groq model for smart cutting |

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
/
  backend/
    server.js           # Express app entry point
    clipper.js          # Core processing: silence detection, transcription, AI cut, ffmpeg concat
    routes/
      process.js        # POST /api/process — upload + queue
      stream.js         # GET  /api/stream/:jobId — SSE
      download.js       # GET  /api/download/:jobId — single-use download
      status.js         # GET  /api/status/:jobId + /api/health
    lib/
      config.js         # All env var parsing
      jobs.js           # In-memory job store + semaphore + TTL cleanup
      runner.js         # Job executor (wraps clipper.js)
      middleware.js     # CORS, rate limiting, validation, security headers
      logger.js         # Structured logger
  frontend/
    app/                # Next.js app router
    components/
      Clipper.tsx        # Main UI state machine
      Dropzone.tsx       # File drop/select
      ProgressPanel.tsx  # Stage progress with ETA
      DoneBar.tsx        # Download + reset
      ErrorBar.tsx       # Error display
    lib/
      config.ts          # API URL config
  uploads/              # Temporary upload storage
  outputs/              # Processed output storage
```
