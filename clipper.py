"""
Audio/Video Clipper — Aiclipsyour.video
- Removes silent/empty spaces from video
- Reduces background noise from audio
"""

import sys
import json
import os
import shutil
import subprocess
import threading
import itertools
import time
from pathlib import Path
from pydub import AudioSegment, silence

# Load .env from the script's directory
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # dotenv optional — env vars still work without it


# ── Terminal helpers ──────────────────────────────────────────────────────────

def clr(code): return f"\033[{code}m"
RESET, BOLD, GREEN, CYAN, YELLOW, RED, DIM = (
    clr(0), clr(1), clr(32), clr(36), clr(33), clr(31), clr(2)
)

def enable_ansi():
    """Enable ANSI codes on Windows."""
    if sys.platform == "win32":
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.SetConsoleMode(kernel.GetStdHandle(-11), 7)

def banner():
    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════╗
║       AV Clipper & Noise Reducer     ║
╚══════════════════════════════════════╝{RESET}
""")

def step(n, total, msg):
    print(f"\n{BOLD}{CYAN}[{n}/{total}]{RESET} {msg}")

def ok(msg):   print(f"  {GREEN}✔{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET}  {msg}")
def err(msg):  print(f"  {RED}✖{RESET}  {msg}")
def info(msg): print(f"  {DIM}{msg}{RESET}")


class Spinner:
    """Simple CLI spinner for blocking operations."""
    def __init__(self, msg):
        self.msg = msg
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        for ch in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            if self._stop.is_set():
                break
            sys.stdout.write(f"\r  {CYAN}{ch}{RESET}  {self.msg} ")
            sys.stdout.flush()
            time.sleep(0.08)
        sys.stdout.write("\r" + " " * (len(self.msg) + 10) + "\r")
        sys.stdout.flush()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()


def progress_bar(current, total, width=36, label=""):
    pct = current / total if total else 0
    filled = int(width * pct)
    bar = f"{GREEN}{'█' * filled}{DIM}{'░' * (width - filled)}{RESET}"
    sys.stdout.write(f"\r  [{bar}] {int(pct*100):3d}%  {label}  ")
    sys.stdout.flush()
    if current >= total:
        print()


# ── Core processing ───────────────────────────────────────────────────────────

def get_non_silent_ranges(audio_path: str, silence_thresh: int, min_silence_len: int, padding: int):
    with Spinner(f"Scanning for silence  (thresh={silence_thresh} dBFS, min={min_silence_len}ms)"):
        audio = AudioSegment.from_file(audio_path)
        silent_ranges = silence.detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
        duration = len(audio)

        # Build speech segments as gaps between silences, with padding applied.
        # We work in ms throughout. Padding expands each speech segment outward
        # (earlier start, later end) to avoid cutting off breath/lead-in.
        if not silent_ranges:
            # No silence found — entire audio is speech
            return [(0, duration)], duration

        speech = []
        # Gap before first silence
        first_sil_start = silent_ranges[0][0]
        if first_sil_start > 0:
            speech.append((0, min(duration, first_sil_start + padding)))

        # Gaps between consecutive silences
        for i in range(len(silent_ranges) - 1):
            seg_start = max(0, silent_ranges[i][1] - padding)
            seg_end   = min(duration, silent_ranges[i + 1][0] + padding)
            if seg_end > seg_start:
                speech.append((seg_start, seg_end))

        # Gap after last silence
        last_sil_end = silent_ranges[-1][1]
        if last_sil_end < duration:
            speech.append((max(0, last_sil_end - padding), duration))

        # Merge any overlapping segments that padding may have created
        speech.sort(key=lambda r: r[0])
        merged: list = []
        for s, e in speech:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

    total_removed = sum((e - s) for s, e in silent_ranges) / 1000
    ok(f"Found {len(merged)} speech segments  |  ~{total_removed:.1f}s of silence to remove")
    return merged, duration


# ── Transcription (AssemblyAI) ────────────────────────────────────────────────

def transcribe_audio(audio_path: str, aai_api_key: str,
                     language_code: str = "en", speaker_labels: bool = False) -> tuple:
    """
    Transcribe audio using AssemblyAI.
    Returns (segments, audio_duration_seconds).
    segments: [{start, end, text}, ...]
    """
    try:
        import assemblyai as aai
    except ImportError:
        err("assemblyai not installed. Run: pip install assemblyai")
        return [], 0.0

    aai.settings.api_key = aai_api_key

    config = aai.TranscriptionConfig(
        language_code=language_code,
        speaker_labels=speaker_labels,
        punctuate=True,
        format_text=True,
    )

    with Spinner("Uploading audio to AssemblyAI"):
        transcriber = aai.Transcriber(config=config)

    print(f"\r  {CYAN}⠋{RESET}  Transcribing via AssemblyAI (this may take a minute)... ", end="", flush=True)
    transcript = transcriber.transcribe(audio_path)
    print("\r" + " " * 60 + "\r", end="")

    if transcript.status == aai.TranscriptStatus.error:
        err(f"AssemblyAI error: {transcript.error}")
        return [], 0.0

    audio_duration_s = (transcript.audio_duration or 0)
    segments = []

    # Priority 1: utterances (available when speaker_labels=True, best granularity)
    if speaker_labels and transcript.utterances:
        for u in transcript.utterances:
            segments.append({
                "start": u.start / 1000.0,
                "end":   u.end   / 1000.0,
                "text":  u.text.strip(),
            })
        info(f"Using {len(segments)} speaker utterances")

    # Priority 2: sentences (best for non-speaker-label mode)
    if not segments:
        try:
            sentences = transcript.get_sentences()
            for s in sentences:
                segments.append({
                    "start": s.start / 1000.0,
                    "end":   s.end   / 1000.0,
                    "text":  s.text.strip(),
                })
            if segments:
                info(f"Using {len(segments)} sentences")
        except Exception:
            pass

    # Priority 3: word-level chunking (~3s chunks, not 5s — better granularity for Groq)
    if not segments and transcript.words:
        chunk, chunk_start = [], None
        for w in transcript.words:
            if chunk_start is None:
                chunk_start = w.start / 1000.0
            chunk.append(w.text)
            if (w.end / 1000.0) - chunk_start >= 3.0:
                segments.append({"start": chunk_start, "end": w.end / 1000.0, "text": " ".join(chunk)})
                chunk, chunk_start = [], None
        if chunk and chunk_start is not None:
            segments.append({"start": chunk_start, "end": transcript.words[-1].end / 1000.0, "text": " ".join(chunk)})
        info(f"Using {len(segments)} word chunks (fallback)")

    ok(f"Transcribed {len(segments)} segments  ({audio_duration_s:.1f}s audio)")
    return segments, audio_duration_s


def save_transcript(segments: list, out_stem: str):
    """Save transcript as .txt and .json alongside the output file."""
    txt_path  = Path(out_stem + "_transcript.txt")
    json_path = Path(out_stem + "_transcript.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"[{seg['start']:.2f}s – {seg['end']:.2f}s]  {seg['text'].strip()}\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    ok(f"Transcript → {txt_path.name}  +  {json_path.name}")


# ── Groq AI smart cutting ──────────────────────────────────────────────────────

GROK_SYSTEM_PROMPT = """\
You are a senior human video editor with 10 years of experience cutting talking-head \
and interview content. You understand pacing, breath, and the natural rhythm of speech.

YOUR GOAL: produce a video that feels like it was never cut — smooth, natural, human.

PACING RULES (most important):
- Short pauses (under 1s) between sentences are NATURAL — never cut them, they give the viewer time to absorb
- A speaker taking a breath before a key point is intentional — keep it
- Do not create jump cuts by removing pauses mid-thought
- Adjacent kept segments must flow into each other without feeling jarring
- If two segments are less than 2 seconds apart, keep the gap — do not cut it

ALWAYS KEEP:
- The opening hook (first thing said) and the closing line (last meaningful sentence)
- Strong opinions, bold claims, emotional moments, controversial takes
- Clear value: advice, insight, a lesson, a punchline, a revelation
- Setup AND its payoff — never keep one without the other
- Any sentence that references something said earlier

REMOVE ONLY:
- True dead air (5+ seconds of nothing)
- Obvious false starts where the speaker immediately restarts the same sentence
- Clear off-topic tangents that break the narrative and have no payoff
- Repeated restatements of a point already made clearly

NEVER:
- Cut mid-sentence or mid-thought under any circumstances
- Remove a pause that gives emotional weight to what follows
- Create a sequence where the same word or sound appears twice in a row

Respond ONLY with a valid JSON array. No explanation. No markdown fences.
Each object:
  "start":  float  (seconds)
  "end":    float  (seconds)
  "reason": string (≤8 words)\
"""


def _compress_transcript(segments: list) -> str:
    """
    Compact transcript — index-based so Groq returns indices, not raw timestamps.
    We map indices back to real timestamps ourselves, eliminating hallucinated times.
    Format: index|start-end|text
    """
    lines = []
    for i, s in enumerate(segments):
        lines.append(f"{i}|{s['start']:.1f}-{s['end']:.1f}|{s['text'].strip()}")
    return "\n".join(lines)


def _merge_close_segments(cuts: list, min_gap_s: float = 2.0) -> list:
    """Merge kept segments that are less than min_gap_s apart."""
    if not cuts:
        return cuts
    merged = [dict(cuts[0])]
    for c in cuts[1:]:
        prev = merged[-1]
        if c["start"] - prev["end"] < min_gap_s:
            prev["end"] = max(prev["end"], c["end"])
        else:
            merged.append(dict(c))
    return merged


def _snap_cuts_to_segments(cuts: list, segments: list, video_duration: float) -> list:
    """
    Snap Groq's returned timestamps to the nearest real transcript segment boundaries.
    Deduplicates: if two cuts snap to the same boundaries, only one is kept.
    """
    if not segments:
        return cuts

    snapped = []
    seen: set = set()

    for c in cuts:
        raw_start = float(c.get("start", 0))
        raw_end   = float(c.get("end",   0))

        best_start_seg = min(segments, key=lambda s: abs(s["start"] - raw_start))
        best_end_seg   = min(segments, key=lambda s: abs(s["end"]   - raw_end))

        snapped_start = best_start_seg["start"]
        snapped_end   = best_end_seg["end"]

        if snapped_end <= snapped_start:
            snapped_end = best_end_seg["start"] if best_end_seg["start"] > snapped_start else snapped_start + 1.0
        snapped_end = min(snapped_end, video_duration)

        if snapped_end - snapped_start < 0.5:
            continue

        key = (round(snapped_start, 2), round(snapped_end, 2))
        if key in seen:
            continue
        seen.add(key)

        snapped.append({**c, "start": snapped_start, "end": snapped_end})

    return snapped



def _fallback_to_all_segments(segments: list, video_duration: float) -> list:
    """Return all transcript segments as cuts — used when Groq output is unusable."""
    return [{"start": s["start"], "end": min(s["end"], video_duration), "reason": "fallback"}
            for s in segments if s["end"] - s["start"] >= 0.5]


def grok_smart_cut(segments: list, api_key: str, model: str = "llama-3.3-70b-versatile",
                   video_duration: float = 0) -> tuple:
    """
    Send transcript to Groq and get semantically correct cut points.
    Returns (cuts, usage_dict).
    Snaps returned timestamps to real segment boundaries to prevent near-zero output.
    Falls back to all segments if Groq output covers < 10% of the video.
    """
    try:
        from groq import Groq
    except ImportError:
        err("groq package not installed. Run: pip install groq")
        return [], {}

    client = Groq(api_key=api_key)
    transcript_text = _compress_transcript(segments)

    user_msg = (
        f"Transcript ({len(segments)} segments, video duration {video_duration:.1f}s):\n\n"
        f"{transcript_text}\n\n"
        "Return JSON array of segments to KEEP. "
        "Use the EXACT start/end times shown above — do not invent new timestamps."
    )

    raw_parts = []
    usage = {}
    with Spinner(f"Asking Groq ({model}) for smart cut points"):
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": GROK_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.15,
            max_completion_tokens=4096,
            top_p=1,
            stream=True,
            stream_options={"include_usage": True},
            stop=None,
        )
        for chunk in stream:
            raw_parts.append(chunk.choices[0].delta.content or "")
            if chunk.usage:
                usage = {
                    "prompt_tokens":     chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens":      chunk.usage.total_tokens,
                }

    raw = "".join(raw_parts).strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        raw = raw.rsplit("```", 1)[0].strip()

    try:
        cuts = json.loads(raw)
    except json.JSONDecodeError as e:
        err(f"Groq returned invalid JSON: {e}")
        info(f"Raw response preview: {raw[:400]}")
        warn("Falling back to full transcript segments")
        return _fallback_to_all_segments(segments, video_duration), usage

    if not cuts:
        warn("Groq returned empty cut list — falling back to full transcript")
        return _fallback_to_all_segments(segments, video_duration), usage

    ok(f"Groq selected {len(cuts)}/{len(segments)} segments")

    # Log raw Groq timestamps for debugging
    info(f"Groq raw range: {cuts[0]['start']:.1f}s – {cuts[-1]['end']:.1f}s  (video: {video_duration:.1f}s)")

    # Snap timestamps to real segment boundaries
    cuts = _snap_cuts_to_segments(cuts, segments, video_duration)
    info(f"After snapping: {len(cuts)} valid segments")

    # Merge segments closer than 2s
    cuts = _merge_close_segments(cuts, min_gap_s=2.0)
    info(f"After merge: {len(cuts)} segments")

    # Sanity check — if Groq kept < 20% of the video, something went wrong
    total_kept = sum(c["end"] - c["start"] for c in cuts)
    coverage   = total_kept / video_duration if video_duration else 0
    info(f"Coverage: {coverage*100:.1f}% of video kept")
    if coverage < 0.20:
        warn(f"Groq only kept {coverage*100:.1f}% — falling back to full transcript (silence detection will trim)")
        cuts = _fallback_to_all_segments(segments, video_duration)
        cuts = _merge_close_segments(cuts, min_gap_s=1.0)

    for c in cuts:
        info(f"  {c['start']:.1f}s – {c['end']:.1f}s  →  {c.get('reason', '')}")

    return cuts, usage


def _ffmpeg_concat(input_path: str, output_path: str, ranges: list,
                   video_codec: str, audio_codec: str, crf: int,
                   audio_only: bool, tmp_dir: Path):
    """
    Cut and join segments with perfect audio/video sync and no duplicate frames.

    Strategy:
      1. Cut each segment with full re-encode (frame-accurate, no keyframe snapping).
         This eliminates duplicate frames at cut boundaries that -c copy causes.
      2. Concat the pre-encoded segments with -c copy (fast, lossless join).

    Uses absolute paths throughout to avoid Windows cwd issues.
    """
    n = len(ranges)

    # Resolve everything to absolute paths so ffmpeg never depends on cwd
    input_abs  = str(Path(input_path).resolve())
    output_abs = str(Path(output_path).resolve())
    tmp_abs    = Path(tmp_dir).resolve()
    tmp_abs.mkdir(parents=True, exist_ok=True)

    segment_files = []

    with Spinner(f"Cutting {n} segments"):
        for i, (start_ms, end_ms) in enumerate(ranges):
            progress_bar(i + 1, n, label=f"seg {i+1}/{n}")
            ss  = f"{start_ms / 1000:.6f}"
            to  = f"{end_ms   / 1000:.6f}"
            ext = ".wav" if audio_only else ".mp4"
            seg = str(tmp_abs / f"seg_{i:04d}{ext}")

            if audio_only:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", input_abs,
                    "-ss", ss, "-to", to,
                    "-vn", "-c:a", "pcm_s16le",
                    seg,
                ]
            else:
                # Re-encode each segment for frame-accurate cuts.
                # -ss/-to AFTER -i = slow seek but exact frame boundary.
                # This prevents keyframe-snap duplication at join points.
                cmd = [
                    "ffmpeg", "-y",
                    "-i", input_abs,
                    "-ss", ss, "-to", to,
                    "-c:v", video_codec,
                    "-crf", str(crf),
                    "-preset", "fast",
                    "-c:a", "pcm_s16le",   # lossless audio in tmp segments
                    "-ar", "48000",
                    "-avoid_negative_ts", "make_zero",
                    seg,
                ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Segment {i} cut failed:\n{result.stderr[-300:]}"
                )
            segment_files.append(seg)

    # Write concat list with absolute paths, forward slashes (ffmpeg cross-platform)
    list_file = str(tmp_abs / "concat_list.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for p in segment_files:
            # ffmpeg requires forward slashes even on Windows
            f.write(f"file '{Path(p).as_posix()}'\n")

    with Spinner(f"Encoding  {Path(output_path).name}"):
        if audio_only:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", list_file,
                "-c:a", audio_codec,
                "-ar", "48000",
                output_abs,
            ]
        else:
            # Segments are already video-encoded; just re-encode audio to final codec.
            # Video stream is copied (no quality loss on second pass).
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", list_file,
                "-c:v", "copy",
                "-c:a", audio_codec,
                "-b:a", "192k",
                "-ar", "48000",
                "-movflags", "+faststart",
                output_abs,
            ]

        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Concat encode failed:\n{result.stderr[-500:]}")

    ok(f"Encoded  ({n} segments, audio/video in sync)")


def clip_video(input_path, output_path, silence_thresh, min_silence_len,
               padding_ms,
               video_codec="libx264", audio_codec="aac", crf=23,
               audio_only=False,
               use_grok=False, grok_api_key=None, grok_model="llama-3.3-70b-versatile",
               aai_api_key=None, language_code="en", speaker_labels=False,
               save_transcript_file=True):

    total_steps = 4
    if use_grok:
        total_steps += 2  # +transcribe +grok
    input_path  = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    if audio_only:
        output_path = output_path.with_suffix(".wav")

    tmp_dir   = (output_path.parent / f"_tmp_{output_path.stem}").resolve()
    # Always start clean — wipe any leftover files from a previous failed run
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_audio = str(tmp_dir / "extracted_audio.wav")

    # Step 1 – probe duration
    step(1, total_steps, f"Loading  {input_path.name}")
    with Spinner("Probing video"):
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format",
             str(input_path)],
            capture_output=True, text=True, timeout=30
        )
        video_duration = float(json.loads(probe.stdout)["format"]["duration"])
    ok(f"Duration: {video_duration:.2f}s")

    # Step 2 – extract audio for analysis
    step(2, total_steps, "Extracting audio track")
    with Spinner("Writing WAV"):
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(input_path),
             "-vn", "-ar", "16000", "-ac", "1", tmp_audio],
            capture_output=True, check=True
        )
    ok("Audio extracted")

    current_step = 3

    # audio used for silence detection and transcription — original, unmodified
    audio_for_analysis = tmp_audio

    # Step (optional) – transcribe + Grok smart cut
    grok_ranges  = None
    cost_summary = {}   # collects usage for final display
    if use_grok:
        if not aai_api_key:
            err("AssemblyAI API key required for transcription. Set ASSEMBLYAI_API_KEY in .env or use --aai-key.")
        else:
            step(current_step, total_steps, "Transcribing audio with AssemblyAI")
            # AssemblyAI: $0.37 / hour (nano tier, as of 2025)
            AAI_PRICE_PER_HOUR = 0.37
            segments, audio_dur_s = transcribe_audio(audio_for_analysis, aai_api_key, language_code, speaker_labels)
            aai_cost = (audio_dur_s / 3600) * AAI_PRICE_PER_HOUR
            cost_summary["aai_audio_seconds"] = audio_dur_s
            cost_summary["aai_cost_usd"]      = aai_cost
            if save_transcript_file:
                out_stem = str(output_path.parent / output_path.stem)
                save_transcript(segments, out_stem)
            current_step += 1

            if segments and grok_api_key:
                step(current_step, total_steps, "Smart cutting with Groq AI")
                # Groq llama-3.3-70b: $0.59/M input, $0.79/M output tokens (as of 2025)
                GROQ_PRICES = {
                    "llama-3.3-70b-versatile": (0.59, 0.79),
                    "llama-3.1-8b-instant":    (0.05, 0.08),
                    "mixtral-8x7b-32768":      (0.24, 0.24),
                    "gemma2-9b-it":            (0.20, 0.20),
                }
                price_in, price_out = GROQ_PRICES.get(grok_model, (0.59, 0.79))
                grok_cuts, groq_usage = grok_smart_cut(segments, grok_api_key, grok_model,
                                                          video_duration=video_duration)
                if groq_usage:
                    groq_cost = (
                        groq_usage.get("prompt_tokens", 0)     / 1_000_000 * price_in +
                        groq_usage.get("completion_tokens", 0) / 1_000_000 * price_out
                    )
                    cost_summary["groq_tokens"]   = groq_usage
                    cost_summary["groq_cost_usd"] = groq_cost
                if grok_cuts:
                    grok_ranges = [(int(c["start"] * 1000), int(c["end"] * 1000)) for c in grok_cuts]
            elif not grok_api_key:
                warn("No Groq API key — skipping AI smart cut, using silence detection only")
            current_step += 1

    # Step – detect silence (used when Grok is off, or as fallback)
    step(current_step, total_steps, "Detecting silent segments")
    ranges, audio_duration_ms = get_non_silent_ranges(
        audio_for_analysis, silence_thresh, min_silence_len, padding_ms
    )
    current_step += 1

    # If silence detection found almost nothing, retry with more aggressive settings
    total_speech_ms = sum(e - s for s, e in ranges)
    if total_speech_ms < (audio_duration_ms * 0.15):
        warn(f"Silence detection too conservative ({total_speech_ms/1000:.1f}s found) — retrying with looser threshold")
        ranges, audio_duration_ms = get_non_silent_ranges(
            audio_for_analysis,
            silence_thresh=max(silence_thresh - 15, -60),  # go 15dB more aggressive
            min_silence_len=max(min_silence_len // 2, 200),
            padding=padding_ms,
        )
        info(f"Retry found {len(ranges)} segments ({sum(e-s for s,e in ranges)/1000:.1f}s)")

    # Prefer Grok ranges if available
    if grok_ranges:
        info(f"Using Grok AI cut points ({len(grok_ranges)} segments)")
        ranges = grok_ranges
    else:
        info(f"Using silence-detection cut points ({len(ranges)} segments)")

    if not ranges:
        err("No segments found. Try a lower --silence-thresh value.")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    # Deduplicate, sort, and merge overlapping ranges — runs regardless of source
    ranges = sorted(set(ranges), key=lambda r: r[0])
    merged: list = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    ranges = merged
    info(f"Final: {len(ranges)} unique non-overlapping segments")

    # Log every range so we can verify no duplicates before cutting
    for idx, (s, e) in enumerate(ranges):
        info(f"  range[{idx}]: {s/1000:.3f}s – {e/1000:.3f}s  ({(e-s)/1000:.3f}s)")

    # Step – cut & export via ffmpeg (no moviepy re-encode — preserves original audio)
    step(current_step, total_steps, "Cutting & exporting video")
    ok(f"Exporting {len(ranges)} segments")

    _ffmpeg_concat(
        input_path=str(input_path),
        output_path=str(output_path),
        ranges=ranges,
        video_codec=video_codec,
        audio_codec=audio_codec,
        crf=crf,
        audio_only=audio_only,
        tmp_dir=tmp_dir,
    )

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Measure durations via ffprobe
    def _probe_duration(p: str) -> float:
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", p],
                capture_output=True, text=True, timeout=15
            )
            return float(json.loads(r.stdout)["format"]["duration"])
        except Exception:
            return 0.0

    orig  = _probe_duration(str(input_path))
    final = _probe_duration(str(output_path))
    removed = orig - final

    # Build cost lines
    cost_lines = ""
    total_cost = 0.0
    if cost_summary:
        if "aai_cost_usd" in cost_summary:
            dur_s = cost_summary["aai_audio_seconds"]
            c     = cost_summary["aai_cost_usd"]
            total_cost += c
            cost_lines += f"\n  │ AssemblyAI : {dur_s:>6.1f}s audio  ${c:.4f}        │"
        if "groq_cost_usd" in cost_summary:
            tok = cost_summary["groq_tokens"]
            c   = cost_summary["groq_cost_usd"]
            total_cost += c
            cost_lines += f"\n  │ Groq       : {tok.get('total_tokens',0):>6} tokens  ${c:.4f}        │"
        cost_lines += f"\n  │ Total cost : {'':>20} ${total_cost:.4f}        │"
        cost_lines += f"\n  ├─────────────────────────────────┤"

    removed = orig - final
    print(f"""
{GREEN}{BOLD}  ✔ Done!{RESET}
  ┌─────────────────────────────────┐
  │ Original  : {orig:>8.2f}s             │
  │ Final     : {final:>8.2f}s             │
  │ Removed   : {removed:>8.2f}s  ({removed/orig*100:.1f}%)      │
  │ Output    : {str(output_path):<22} │
  │ Mode      : {'audio only' if audio_only else f'{video_codec} / {audio_codec} crf={crf}':<22} │{cost_lines}
  └─────────────────────────────────┘""")


# ── Interactive wizard ────────────────────────────────────────────────────────

def ask(prompt, default=None, cast=str, choices=None):
    hint = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"  {CYAN}?{RESET}  {prompt}{hint}: ").strip()
        value = raw if raw else (str(default) if default is not None else "")
        if not value:
            warn("Required — please enter a value.")
            continue
        try:
            value = cast(value)
        except (ValueError, TypeError):
            warn(f"Expected {cast.__name__}, got: {raw!r}")
            continue
        if choices and value not in choices:
            warn(f"Choose one of: {', '.join(map(str, choices))}")
            continue
        return value

def ask_bool(prompt, default=True):
    hint = "Y/n" if default else "y/N"
    raw = input(f"  {CYAN}?{RESET}  {prompt} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")

def wizard():
    banner()
    print(f"{BOLD}Interactive mode{RESET} — press Enter to accept defaults\n")

    input_path = ask("Input video file path", cast=str)
    while not Path(input_path).exists():
        err(f"File not found: {input_path}")
        input_path = ask("Input video file path")

    default_out = Path(input_path).stem + "_clipped" + Path(input_path).suffix
    output_path = ask("Output file path", default=default_out)

    print(f"\n  {DIM}── Silence removal ──────────────────{RESET}")
    silence_thresh  = ask("Silence threshold dBFS (lower = more aggressive)", default=-40, cast=int)
    min_silence_len = ask("Minimum silence length to remove (ms)", default=500, cast=int)
    padding_ms      = ask("Padding to keep around speech (ms)", default=200, cast=int)

    print(f"\n  {DIM}── Noise reduction ──────────────────{RESET}")
    noise_reduce   = ask_bool("Enable background noise reduction?", default=True)
    noise_strength = 0.75
    stationary     = False
    n_fft          = 2048
    if noise_reduce:
        noise_strength = ask("Noise reduction strength (0.0 – 1.0)", default=0.75, cast=float)
        while not 0.0 <= noise_strength <= 1.0:
            warn("Must be between 0.0 and 1.0")
            noise_strength = ask("Noise reduction strength", default=0.75, cast=float)
        stationary = ask_bool(
            "Stationary noise? (constant hum/hiss=yes, variable bg noise=no)", default=False
        )
        n_fft = ask("FFT window size — higher=more freq detail (512/1024/2048/4096)", default=2048, cast=int)
        while n_fft not in (512, 1024, 2048, 4096):
            warn("Choose one of: 512, 1024, 2048, 4096")
            n_fft = ask("FFT window size", default=2048, cast=int)

    print(f"\n  {DIM}── AI smart cutting (Groq + AssemblyAI) ──{RESET}")
    use_grok     = ask_bool("Enable AI smart cutting? (requires AssemblyAI + Groq API keys)", default=False)
    grok_api_key = None
    aai_api_key  = None
    grok_model   = "llama-3.3-70b-versatile"
    language_code   = "en"
    speaker_labels  = False
    save_tr         = True
    if use_grok:
        aai_api_key = os.environ.get("ASSEMBLYAI_API_KEY", "")
        if not aai_api_key:
            aai_api_key = ask("AssemblyAI API key (or set ASSEMBLYAI_API_KEY in .env)")
        else:
            ok("AssemblyAI API key loaded from .env")
        grok_api_key = os.environ.get("GROQ_API_KEY", "")
        if not grok_api_key:
            grok_api_key = ask("Groq API key (or set GROQ_API_KEY in .env)")
        else:
            ok("Groq API key loaded from .env")
        grok_model    = ask("Groq model", default="llama-3.3-70b-versatile",
                            choices=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"])
        language_code = ask("Language code for transcription", default="en")
        speaker_labels = ask_bool("Detect speaker labels?", default=False)
        save_tr = ask_bool("Save transcript files (.txt + .json)?", default=True)

    print(f"\n  {DIM}── Output settings ──────────────────{RESET}")
    audio_only  = ask_bool("Export audio only? (saves as .wav)", default=False)
    video_codec = "libx264"
    audio_codec = "aac"
    crf         = 23
    if not audio_only:
        video_codec = ask("Video codec", default="libx264",
                          choices=["libx264", "libx265", "libvpx-vp9"])
        audio_codec = ask("Audio codec", default="aac",
                          choices=["aac", "mp3", "libopus"])
        crf = ask("CRF quality (0=lossless, 51=worst; 18=high, 23=default, 28=smaller)", default=23, cast=int)
        while not 0 <= crf <= 51:
            warn("CRF must be 0–51")
            crf = ask("CRF quality", default=23, cast=int)

    print(f"\n  {DIM}────────────────────────────────────{RESET}")
    print(f"  {BOLD}Summary:{RESET}")
    print(f"    Input          : {input_path}")
    print(f"    Output         : {output_path}")
    print(f"    Silence thresh : {silence_thresh} dBFS")
    print(f"    Min silence    : {min_silence_len} ms")
    print(f"    Padding        : {padding_ms} ms")
    if noise_reduce:
        print(f"    Noise reduce   : strength={noise_strength}  stationary={stationary}  n_fft={n_fft}")
    else:
        print(f"    Noise reduce   : disabled")
    if use_grok:
        print(f"    AI cut         : grok={grok_model}  lang={language_code}  speakers={speaker_labels}  save_transcript={save_tr}")
    else:
        print(f"    Grok AI cut    : disabled")
    if audio_only:
        print(f"    Output mode    : audio only (.wav)")
    else:
        print(f"    Video codec    : {video_codec}  crf={crf}")
        print(f"    Audio codec    : {audio_codec}")
    print()

    if not ask_bool("Start processing?", default=True):
        print("Aborted.")
        return

    clip_video(input_path, output_path, silence_thresh, min_silence_len,
               padding_ms,
               video_codec=video_codec, audio_codec=audio_codec, crf=crf,
               audio_only=audio_only,
               use_grok=use_grok, grok_api_key=grok_api_key,
               grok_model=grok_model, aai_api_key=aai_api_key,
               language_code=language_code, speaker_labels=speaker_labels,
               save_transcript_file=save_tr)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Remove silences and reduce background noise from video.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input",  nargs="?", help="Input video file (omit for interactive mode)")
    parser.add_argument("output", nargs="?", help="Output video file")
    parser.add_argument("--silence-thresh", type=int,   default=-40,
                        help="Silence threshold in dBFS (default: -40)")
    parser.add_argument("--min-silence",    type=int,   default=500,
                        help="Minimum silence length in ms (default: 500)")
    parser.add_argument("--padding",        type=int,   default=200,
                        help="Padding in ms around speech (default: 200)")
    parser.add_argument("--video-codec",    default="libx264",
                        choices=["libx264", "libx265", "libvpx-vp9"],
                        help="Video codec (default: libx264)")
    parser.add_argument("--audio-codec",    default="aac",
                        choices=["aac", "mp3", "libopus"],
                        help="Audio codec (default: aac)")
    parser.add_argument("--crf",            type=int, default=23,
                        help="CRF quality 0-51 (lower=better; default: 23)")
    parser.add_argument("--audio-only",     action="store_true",
                        help="Export processed audio only as .wav")
    # AssemblyAI + Groq
    parser.add_argument("--grok",           action="store_true",
                        help="Enable AI smart cutting (AssemblyAI transcription + Groq analysis)")
    parser.add_argument("--aai-key",        default=None,
                        help="AssemblyAI API key (or set ASSEMBLYAI_API_KEY env var)")
    parser.add_argument("--grok-key",       default=None,
                        help="Groq API key (or set GROQ_API_KEY env var)")
    parser.add_argument("--grok-model",     default="llama-3.3-70b-versatile",
                        choices=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
                        help="Groq model (default: llama-3.3-70b-versatile)")
    parser.add_argument("--language",       default="en",
                        help="Language code for AssemblyAI transcription (default: en)")
    parser.add_argument("--speaker-labels", action="store_true",
                        help="Detect speaker labels in transcription")
    parser.add_argument("--no-transcript",  action="store_true",
                        help="Don't save transcript .txt/.json files")

    args = parser.parse_args()
    enable_ansi()

    if not args.input:
        wizard()
        return

    if not args.output:
        p = Path(args.input)
        args.output = str(p.parent / (p.stem + "_clipped" + p.suffix))
        info(f"No output path given — using: {args.output}")

    banner()
    aai_key  = args.aai_key  or os.environ.get("ASSEMBLYAI_API_KEY")
    grok_key = args.grok_key or os.environ.get("GROQ_API_KEY")
    if args.grok and not aai_key:
        warn("--grok enabled but no AssemblyAI key found. Set ASSEMBLYAI_API_KEY or use --aai-key.")
    if args.grok and not grok_key:
        warn("--grok enabled but no Groq key found. Set GROQ_API_KEY or use --grok-key.")

    clip_video(
        input_path=args.input,
        output_path=args.output,
        silence_thresh=args.silence_thresh,
        min_silence_len=args.min_silence,
        padding_ms=args.padding,
        video_codec=args.video_codec,
        audio_codec=args.audio_codec,
        crf=args.crf,
        audio_only=args.audio_only,
        use_grok=args.grok,
        aai_api_key=aai_key,
        grok_api_key=grok_key,
        grok_model=args.grok_model,
        language_code=args.language,
        speaker_labels=args.speaker_labels,
        save_transcript_file=not args.no_transcript,
    )


if __name__ == "__main__":
    main()
