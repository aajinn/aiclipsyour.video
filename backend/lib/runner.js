/**
 * Job runner — pulls from the semaphore and executes clipVideo.
 */

import { unlinkSync } from "fs";
import { clipVideo, BEST_PARAMS } from "../clipper.js";
import { jobs, semaphore, pushSSE, closeSSEClients } from "./jobs.js";
import { logger } from "./logger.js";

const STAGE_MAP = {
  "preparing":  "Preparing your video",
  "transcrib":  "Transcribing speech",
  "ai smart":   "AI smart cutting",
  "detecting":  "Detecting silences",
  "exporting":  "Exporting final video",
  "encoding":   "Exporting final video",
  "cutting":    "Exporting final video",
};

function mapStage(msg) {
  const lower = msg.toLowerCase();
  for (const [key, label] of Object.entries(STAGE_MAP)) {
    if (lower.includes(key)) return label;
  }
  return null;
}

export function buildParams(overrides = {}) {
  const aaiKey  = process.env.ASSEMBLYAI_API_KEY;
  const groqKey = process.env.GROQ_API_KEY;
  return {
    silence_thresh:  overrides.silenceThresh  ?? BEST_PARAMS.silenceThresh,
    min_silence_len: overrides.minSilenceLen  ?? BEST_PARAMS.minSilenceLen,
    padding_ms:      overrides.paddingMs      ?? BEST_PARAMS.paddingMs,
    video_codec:     overrides.videoCodec     ?? BEST_PARAMS.videoCodec,
    audio_codec:     overrides.audioCodec     ?? BEST_PARAMS.audioCodec,
    crf:             overrides.crf            ?? BEST_PARAMS.crf,
    audio_only:      overrides.audioOnly      ?? BEST_PARAMS.audioOnly,
    use_grok:        !!(aaiKey && groqKey),
    aai_api_key:     aaiKey,
    grok_api_key:    groqKey,
    grok_model:      overrides.groqModel      ?? BEST_PARAMS.groqModel,
    language_code:   overrides.languageCode   ?? BEST_PARAMS.languageCode,
    speaker_labels:  overrides.speakerLabels  ?? BEST_PARAMS.speakerLabels,
    save_transcript: overrides.saveTranscript ?? BEST_PARAMS.saveTranscript,
  };
}

export async function runJob(jobId, inputPath, outputPath, params) {
  await semaphore.acquire();

  const job = jobs.get(jobId);
  if (!job) { semaphore.release(); return; }

  job.status = "running";
  logger.info("job_start", { jobId, file: job.filename });

  try {
    const result = await clipVideo({
      inputPath,
      outputPath,
      silenceThresh:      params.silence_thresh,
      minSilenceLen:      params.min_silence_len,
      paddingMs:          params.padding_ms,
      videoCodec:         params.video_codec,
      audioCodec:         params.audio_codec,
      crf:                params.crf,
      audioOnly:          params.audio_only,
      useGroq:            params.use_grok,
      aaiApiKey:          params.aai_api_key,
      groqApiKey:         params.grok_api_key,
      groqModel:          params.grok_model,
      languageCode:       params.language_code,
      speakerLabels:      params.speaker_labels,
      saveTranscriptFile: params.save_transcript,
      onStage: (stage) => {
        job.stage = stage;
        pushSSE(job, { type: "stage", stage });
      },
      onProgress: (msg) => {
        const stage = mapStage(msg);
        if (stage) {
          job.stage = stage;
          pushSSE(job, { type: "stage", stage });
        }
      },
    });

    job.status     = "done";
    job.outputPath = result.outputPath;
    pushSSE(job, { type: "done" });
    logger.info("job_done", {
      jobId,
      origDuration:  result.origDuration,
      finalDuration: result.finalDuration,
      removed:       result.removed,
    });
  } catch (err) {
    job.status = "error";
    job.error  = err.message;
    pushSSE(job, { type: "error", msg: "Processing failed — check server logs." });
    logger.error("job_failed", { jobId, msg: err.message });
  } finally {
    semaphore.release();
    closeSSEClients(job);
    try { unlinkSync(inputPath); } catch { /* ignore */ }
  }
}
