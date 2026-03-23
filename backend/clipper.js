/**
 * clipper.js — Core audio/video processing logic
 * Removes silent segments, optionally transcribes + AI-smart-cuts via AssemblyAI + Groq
 */

import { execFileSync, execFile, spawnSync } from "child_process";
import { promisify } from "util";
import { createWriteStream, mkdirSync, rmSync, existsSync, writeFileSync } from "fs";
import { readFile, writeFile, unlink } from "fs/promises";
import path from "path";
import os from "os";

const execFileAsync = promisify(execFile);

// ── ffprobe / ffmpeg helpers ──────────────────────────────────────────────────

export function probeDuration(filePath) {
  const result = spawnSync("ffprobe", [
    "-v", "quiet",
    "-print_format", "json",
    "-show_format",
    filePath,
  ], { encoding: "utf8", timeout: 30000 });
  if (result.status !== 0) throw new Error(`ffprobe failed: ${result.stderr}`);
  return parseFloat(JSON.parse(result.stdout).format.duration);
}

async function extractAudio(inputPath, outputPath) {
  await execFileAsync("ffmpeg", [
    "-y", "-i", inputPath,
    "-vn", "-ar", "16000", "-ac", "1",
    outputPath,
  ]);
}

/**
 * Detect non-silent ranges using ffmpeg's silencedetect filter.
 * Returns array of [startMs, endMs] pairs.
 */
export async function getNonSilentRanges(audioPath, silenceThresh, minSilenceLen, paddingMs, onProgress) {
  onProgress?.("Scanning for silence...");

  // Run ffmpeg silencedetect
  const { stderr } = await execFileAsync("ffmpeg", [
    "-i", audioPath,
    "-af", `silencedetect=noise=${silenceThresh}dB:d=${minSilenceLen / 1000}`,
    "-f", "null", "-",
  ]).catch(e => ({ stderr: e.stderr ?? "" }));

  // Parse silence_start / silence_end from stderr
  const silentRanges = [];
  const startRe = /silence_start:\s*([\d.]+)/g;
  const endRe   = /silence_end:\s*([\d.]+)/g;
  let sm, em;
  const starts = [], ends = [];
  while ((sm = startRe.exec(stderr))) starts.push(parseFloat(sm[1]) * 1000);
  while ((em = endRe.exec(stderr)))   ends.push(parseFloat(em[1]) * 1000);

  for (let i = 0; i < Math.min(starts.length, ends.length); i++) {
    silentRanges.push([starts[i], ends[i]]);
  }
  // Handle trailing silence (no end marker)
  if (starts.length > ends.length) {
    const durationMs = probeDuration(audioPath) * 1000;
    silentRanges.push([starts[starts.length - 1], durationMs]);
  }

  const durationMs = probeDuration(audioPath) * 1000;

  if (silentRanges.length === 0) {
    return { ranges: [[0, durationMs]], durationMs };
  }

  // Build speech segments as gaps between silences, with padding
  const speech = [];
  const firstSilStart = silentRanges[0][0];
  if (firstSilStart > 0) {
    speech.push([0, Math.min(durationMs, firstSilStart + paddingMs)]);
  }
  for (let i = 0; i < silentRanges.length - 1; i++) {
    const segStart = Math.max(0, silentRanges[i][1] - paddingMs);
    const segEnd   = Math.min(durationMs, silentRanges[i + 1][0] + paddingMs);
    if (segEnd > segStart) speech.push([segStart, segEnd]);
  }
  const lastSilEnd = silentRanges[silentRanges.length - 1][1];
  if (lastSilEnd < durationMs) {
    speech.push([Math.max(0, lastSilEnd - paddingMs), durationMs]);
  }

  // Merge overlapping
  speech.sort((a, b) => a[0] - b[0]);
  const merged = [];
  for (const [s, e] of speech) {
    if (merged.length && s <= merged[merged.length - 1][1]) {
      merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], e);
    } else {
      merged.push([s, e]);
    }
  }

  const totalRemoved = silentRanges.reduce((acc, [s, e]) => acc + (e - s), 0) / 1000;
  onProgress?.(`Found ${merged.length} speech segments | ~${totalRemoved.toFixed(1)}s silence to remove`);
  return { ranges: merged, durationMs };
}

// ── AssemblyAI transcription ──────────────────────────────────────────────────

export async function transcribeAudio(audioPath, apiKey, languageCode = "en", speakerLabels = false, onProgress) {
  onProgress?.("Uploading audio to AssemblyAI...");
  const { AssemblyAI } = await import("assemblyai");
  const client = new AssemblyAI({ apiKey });

  const transcript = await client.transcripts.transcribe({
    audio: audioPath,
    language_code: languageCode,
    speaker_labels: speakerLabels,
    punctuate: true,
    format_text: true,
  });

  if (transcript.status === "error") {
    throw new Error(`AssemblyAI error: ${transcript.error}`);
  }

  const audioDurationS = (transcript.audio_duration ?? 0);
  let segments = [];

  // Priority 1: utterances (speaker labels)
  if (speakerLabels && transcript.utterances?.length) {
    segments = transcript.utterances.map(u => ({
      start: u.start / 1000,
      end:   u.end   / 1000,
      text:  u.text.trim(),
    }));
  }

  // Priority 2: sentences
  if (!segments.length) {
    try {
      const sentences = await client.transcripts.sentences(transcript.id);
      segments = sentences.sentences.map(s => ({
        start: s.start / 1000,
        end:   s.end   / 1000,
        text:  s.text.trim(),
      }));
    } catch { /* fall through */ }
  }

  // Priority 3: word chunks (~3s)
  if (!segments.length && transcript.words?.length) {
    let chunk = [], chunkStart = null;
    for (const w of transcript.words) {
      if (chunkStart === null) chunkStart = w.start / 1000;
      chunk.push(w.text);
      if ((w.end / 1000) - chunkStart >= 3.0) {
        segments.push({ start: chunkStart, end: w.end / 1000, text: chunk.join(" ") });
        chunk = []; chunkStart = null;
      }
    }
    if (chunk.length && chunkStart !== null) {
      segments.push({ start: chunkStart, end: transcript.words.at(-1).end / 1000, text: chunk.join(" ") });
    }
  }

  onProgress?.(`Transcribed ${segments.length} segments (${audioDurationS.toFixed(1)}s audio)`);
  return { segments, audioDurationS };
}

export async function saveTranscript(segments, outStem) {
  const txt  = segments.map(s => `[${s.start.toFixed(2)}s – ${s.end.toFixed(2)}s]  ${s.text.trim()}`).join("\n");
  const json = JSON.stringify(segments, null, 2);
  await writeFile(outStem + "_transcript.txt",  txt,  "utf8");
  await writeFile(outStem + "_transcript.json", json, "utf8");
}

// ── Groq AI smart cutting ─────────────────────────────────────────────────────

const GROQ_SYSTEM_PROMPT = `You are a senior human video editor with 10 years of experience cutting talking-head \
and interview content. You understand pacing, breath, and the natural rhythm of speech.

YOUR GOAL: produce a video that feels like it was never cut — smooth, natural, human.

PACING RULES (most important):
- Short pauses (under 1s) between sentences are NATURAL — never cut them
- A speaker taking a breath before a key point is intentional — keep it
- Do not create jump cuts by removing pauses mid-thought
- If two segments are less than 2 seconds apart, keep the gap — do not cut it

ALWAYS KEEP:
- The opening hook and the closing line
- Strong opinions, bold claims, emotional moments
- Clear value: advice, insight, a lesson, a punchline, a revelation
- Setup AND its payoff — never keep one without the other

REMOVE ONLY:
- True dead air (5+ seconds of nothing)
- Obvious false starts where the speaker immediately restarts the same sentence
- Clear off-topic tangents with no payoff
- Repeated restatements of a point already made clearly

NEVER:
- Cut mid-sentence or mid-thought
- Remove a pause that gives emotional weight to what follows

Respond ONLY with a valid JSON array. No explanation. No markdown fences.
Each object: "start": float, "end": float, "reason": string (≤8 words)`;

function compressTranscript(segments) {
  return segments.map((s, i) => `${i}|${s.start.toFixed(1)}-${s.end.toFixed(1)}|${s.text.trim()}`).join("\n");
}

function mergeCloseSegments(cuts, minGapS = 2.0) {
  if (!cuts.length) return cuts;
  const merged = [{ ...cuts[0] }];
  for (const c of cuts.slice(1)) {
    const prev = merged[merged.length - 1];
    if (c.start - prev.end < minGapS) {
      prev.end = Math.max(prev.end, c.end);
    } else {
      merged.push({ ...c });
    }
  }
  return merged;
}

function snapCutsToSegments(cuts, segments, videoDuration) {
  if (!segments.length) return cuts;
  const snapped = [];
  const seen = new Set();
  for (const c of cuts) {
    const rawStart = parseFloat(c.start ?? 0);
    const rawEnd   = parseFloat(c.end   ?? 0);
    const bestStart = segments.reduce((a, b) => Math.abs(a.start - rawStart) < Math.abs(b.start - rawStart) ? a : b);
    const bestEnd   = segments.reduce((a, b) => Math.abs(a.end   - rawEnd)   < Math.abs(b.end   - rawEnd)   ? a : b);
    let sStart = bestStart.start;
    let sEnd   = Math.min(bestEnd.end, videoDuration);
    if (sEnd <= sStart) sEnd = sStart + 1.0;
    if (sEnd - sStart < 0.5) continue;
    const key = `${sStart.toFixed(2)}-${sEnd.toFixed(2)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    snapped.push({ ...c, start: sStart, end: sEnd });
  }
  return snapped;
}

function fallbackToAllSegments(segments, videoDuration) {
  return segments
    .filter(s => s.end - s.start >= 0.5)
    .map(s => ({ start: s.start, end: Math.min(s.end, videoDuration), reason: "fallback" }));
}

export async function groqSmartCut(segments, apiKey, model = "llama-3.3-70b-versatile", videoDuration = 0, onProgress) {
  onProgress?.(`Asking Groq (${model}) for smart cut points...`);
  const { Groq } = await import("groq-sdk");
  const client = new Groq({ apiKey });

  const transcriptText = compressTranscript(segments);
  const userMsg = `Transcript (${segments.length} segments, video duration ${videoDuration.toFixed(1)}s):\n\n${transcriptText}\n\nReturn JSON array of segments to KEEP. Use the EXACT start/end times shown above.`;

  const stream = await client.chat.completions.create({
    model,
    messages: [
      { role: "system", content: GROQ_SYSTEM_PROMPT },
      { role: "user",   content: userMsg },
    ],
    temperature: 0.15,
    max_completion_tokens: 4096,
    stream: true,
  });

  let raw = "";
  let usage = {};
  for await (const chunk of stream) {
    raw += chunk.choices[0]?.delta?.content ?? "";
    if (chunk.usage) usage = chunk.usage;
  }

  raw = raw.trim();
  if (raw.startsWith("```")) {
    raw = raw.split("\n").slice(1).join("\n").replace(/```$/, "").trim();
  }

  let cuts;
  try {
    cuts = JSON.parse(raw);
  } catch {
    onProgress?.("Groq returned invalid JSON — falling back to full transcript");
    return { cuts: fallbackToAllSegments(segments, videoDuration), usage };
  }

  if (!cuts?.length) {
    onProgress?.("Groq returned empty list — falling back");
    return { cuts: fallbackToAllSegments(segments, videoDuration), usage };
  }

  onProgress?.(`Groq selected ${cuts.length}/${segments.length} segments`);

  cuts = snapCutsToSegments(cuts, segments, videoDuration);
  cuts = mergeCloseSegments(cuts, 2.0);

  const totalKept = cuts.reduce((a, c) => a + (c.end - c.start), 0);
  const coverage  = videoDuration ? totalKept / videoDuration : 0;
  if (coverage < 0.20) {
    onProgress?.(`Groq only kept ${(coverage * 100).toFixed(1)}% — falling back`);
    cuts = mergeCloseSegments(fallbackToAllSegments(segments, videoDuration), 1.0);
  }

  return { cuts, usage };
}

// ── ffmpeg concat ─────────────────────────────────────────────────────────────

export async function ffmpegConcat(inputPath, outputPath, ranges, videoCodec, audioCodec, crf, audioOnly, tmpDir, onProgress) {
  mkdirSync(tmpDir, { recursive: true });
  const n = ranges.length;
  const segFiles = [];

  for (let i = 0; i < n; i++) {
    onProgress?.(`Cutting segment ${i + 1}/${n}`);
    const [startMs, endMs] = ranges[i];
    const ss  = (startMs / 1000).toFixed(6);
    const to  = (endMs   / 1000).toFixed(6);
    const ext = audioOnly ? ".wav" : ".mp4";
    const seg = path.join(tmpDir, `seg_${String(i).padStart(4, "0")}${ext}`);

    const args = audioOnly
      ? ["-y", "-i", inputPath, "-ss", ss, "-to", to, "-vn", "-c:a", "pcm_s16le", seg]
      : ["-y", "-i", inputPath, "-ss", ss, "-to", to,
         "-c:v", videoCodec, "-crf", String(crf), "-preset", "fast",
         "-c:a", "pcm_s16le", "-ar", "48000",
         "-avoid_negative_ts", "make_zero", seg];

    await execFileAsync("ffmpeg", args);
    segFiles.push(seg);
  }

  // Write concat list
  const listFile = path.join(tmpDir, "concat_list.txt");
  writeFileSync(listFile, segFiles.map(p => `file '${p.replace(/\\/g, "/")}'`).join("\n"), "utf8");

  onProgress?.("Encoding final output...");
  const concatArgs = audioOnly
    ? ["-y", "-f", "concat", "-safe", "0", "-i", listFile, "-c:a", audioCodec, "-ar", "48000", outputPath]
    : ["-y", "-f", "concat", "-safe", "0", "-i", listFile,
       "-c:v", "copy", "-c:a", audioCodec, "-b:a", "192k", "-ar", "48000",
       "-movflags", "+faststart", outputPath];

  await execFileAsync("ffmpeg", concatArgs);
  onProgress?.(`Encoded ${n} segments`);
}

// ── Main clip_video ───────────────────────────────────────────────────────────

const BEST_PARAMS = {
  silenceThresh:  -35,
  minSilenceLen:  800,
  paddingMs:      300,
  videoCodec:     "libx264",
  audioCodec:     "aac",
  crf:            23,
  audioOnly:      false,
  groqModel:      "llama-3.3-70b-versatile",
  languageCode:   "en",
  speakerLabels:  true,
  saveTranscript: true,
};

export { BEST_PARAMS };

/**
 * Main processing function.
 * @param {object} opts
 * @param {string} opts.inputPath
 * @param {string} opts.outputPath
 * @param {number} opts.silenceThresh   dBFS, e.g. -35
 * @param {number} opts.minSilenceLen   ms
 * @param {number} opts.paddingMs
 * @param {string} opts.videoCodec
 * @param {string} opts.audioCodec
 * @param {number} opts.crf
 * @param {boolean} opts.audioOnly
 * @param {boolean} opts.useGroq
 * @param {string|null} opts.aaiApiKey
 * @param {string|null} opts.groqApiKey
 * @param {string} opts.groqModel
 * @param {string} opts.languageCode
 * @param {boolean} opts.speakerLabels
 * @param {boolean} opts.saveTranscriptFile
 * @param {function} opts.onStage        called with stage label string
 * @param {function} opts.onProgress     called with detail string
 */
export async function clipVideo(opts) {
  const {
    inputPath, outputPath,
    silenceThresh = -35, minSilenceLen = 800, paddingMs = 300,
    videoCodec = "libx264", audioCodec = "aac", crf = 23,
    audioOnly = false,
    useGroq = false, aaiApiKey = null, groqApiKey = null,
    groqModel = "llama-3.3-70b-versatile",
    languageCode = "en", speakerLabels = false,
    saveTranscriptFile = true,
    onStage = () => {}, onProgress = () => {},
  } = opts;

  const resolvedOutput = audioOnly
    ? outputPath.replace(/\.[^.]+$/, ".wav")
    : outputPath;

  const tmpDir = path.join(path.dirname(resolvedOutput), `_tmp_${path.basename(resolvedOutput, path.extname(resolvedOutput))}`);
  if (existsSync(tmpDir)) rmSync(tmpDir, { recursive: true, force: true });

  // Step 1 — probe duration
  onStage("Preparing your video");
  onProgress("Probing video duration...");
  const videoDuration = probeDuration(inputPath);

  // Step 2 — extract audio
  onProgress("Extracting audio track...");
  const tmpAudio = path.join(tmpDir, "extracted_audio.wav");
  mkdirSync(tmpDir, { recursive: true });
  await extractAudio(inputPath, tmpAudio);

  let groqRanges = null;
  const costSummary = {};

  // Step 3 (optional) — transcribe + Groq smart cut
  if (useGroq) {
    if (!aaiApiKey) {
      onProgress("No AssemblyAI key — skipping transcription");
    } else {
      onStage("Transcribing speech");
      const { segments, audioDurationS } = await transcribeAudio(tmpAudio, aaiApiKey, languageCode, speakerLabels, onProgress);
      const AAI_PRICE_PER_HOUR = 0.37;
      costSummary.aaiAudioSeconds = audioDurationS;
      costSummary.aaiCostUsd      = (audioDurationS / 3600) * AAI_PRICE_PER_HOUR;

      if (saveTranscriptFile) {
        const outStem = resolvedOutput.replace(/\.[^.]+$/, "");
        await saveTranscript(segments, outStem);
      }

      if (segments.length && groqApiKey) {
        onStage("AI smart cutting");
        const GROQ_PRICES = {
          "llama-3.3-70b-versatile": [0.59, 0.79],
          "llama-3.1-8b-instant":    [0.05, 0.08],
          "mixtral-8x7b-32768":      [0.24, 0.24],
          "gemma2-9b-it":            [0.20, 0.20],
        };
        const [priceIn, priceOut] = GROQ_PRICES[groqModel] ?? [0.59, 0.79];
        const { cuts, usage } = await groqSmartCut(segments, groqApiKey, groqModel, videoDuration, onProgress);
        if (usage?.prompt_tokens) {
          costSummary.groqTokens  = usage;
          costSummary.groqCostUsd = (usage.prompt_tokens / 1e6) * priceIn + (usage.completion_tokens / 1e6) * priceOut;
        }
        if (cuts?.length) {
          groqRanges = cuts.map(c => [Math.round(c.start * 1000), Math.round(c.end * 1000)]);
        }
      } else if (!groqApiKey) {
        onProgress("No Groq key — using silence detection only");
      }
    }
  }

  // Step 4 — silence detection
  onStage("Detecting silences");
  let { ranges, durationMs } = await getNonSilentRanges(tmpAudio, silenceThresh, minSilenceLen, paddingMs, onProgress);

  // Retry if too conservative
  const totalSpeechMs = ranges.reduce((a, [s, e]) => a + (e - s), 0);
  if (totalSpeechMs < durationMs * 0.15) {
    onProgress("Silence detection too conservative — retrying with looser threshold");
    ({ ranges, durationMs } = await getNonSilentRanges(
      tmpAudio,
      Math.max(silenceThresh - 15, -60),
      Math.max(Math.floor(minSilenceLen / 2), 200),
      paddingMs,
      onProgress,
    ));
  }

  // Prefer Groq ranges if available
  if (groqRanges?.length) {
    onProgress(`Using Groq AI cut points (${groqRanges.length} segments)`);
    ranges = groqRanges;
  }

  if (!ranges.length) throw new Error("No segments found. Try a lower silence threshold.");

  // Deduplicate, sort, merge
  ranges = [...new Map(ranges.map(r => [r[0], r])).values()].sort((a, b) => a[0] - b[0]);
  const merged = [];
  for (const [s, e] of ranges) {
    if (merged.length && s <= merged[merged.length - 1][1]) {
      merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], e);
    } else {
      merged.push([s, e]);
    }
  }
  ranges = merged;

  // Step 5 — cut & export
  onStage("Exporting final video");
  await ffmpegConcat(inputPath, resolvedOutput, ranges, videoCodec, audioCodec, crf, audioOnly, tmpDir, onProgress);

  // Cleanup
  rmSync(tmpDir, { recursive: true, force: true });

  const origDuration  = probeDuration(inputPath);
  const finalDuration = probeDuration(resolvedOutput);

  return {
    outputPath:    resolvedOutput,
    origDuration,
    finalDuration,
    removed:       origDuration - finalDuration,
    segments:      ranges.length,
    costSummary,
  };
}
