"use client";

import { useCallback, useRef, useState } from "react";
import Dropzone from "./Dropzone";
import ProgressPanel from "./ProgressPanel";
import DoneBar from "./DoneBar";
import ErrorBar from "./ErrorBar";
import { api } from "@/lib/config";

export type AppState = "idle" | "uploading" | "processing" | "done" | "error";

export const STAGES = [
  { label: "Prepare",    match: "preparing",   pct: 8,  weight: 0.08, hint: "Reading your video file…" },
  { label: "Transcribe", match: "transcribing", pct: 50, weight: 0.42, hint: "Converting speech to text via AI…" },
  { label: "AI cut",     match: "ai smart",     pct: 68, weight: 0.18, hint: "Finding the best moments to keep…" },
  { label: "Silence",    match: "detecting",    pct: 83, weight: 0.15, hint: "Scanning for silent gaps…" },
  { label: "Export",     match: "exporting",    pct: 95, weight: 0.17, hint: "Encoding final video…" },
];

export default function Clipper() {
  const [appState,      setAppState]      = useState<AppState>("idle");
  const [currentStage,  setCurrentStage]  = useState(-1);
  const [errorMsg,      setErrorMsg]      = useState("");
  const [jobId,         setJobId]         = useState("");
  const [jobToken,      setJobToken]      = useState("");
  const [videoDuration, setVideoDuration] = useState<number | null>(null);
  const [startTime,     setStartTime]     = useState<number | null>(null);
  const [stageStartTimes, setStageStartTimes] = useState<Record<number, number>>({});

  const evtSourceRef = useRef<EventSource | null>(null);

  const advanceToStage = useCallback((idx: number) => {
    setCurrentStage(prev => {
      if (idx <= prev) return prev;
      setStageStartTimes(t => ({ ...t, [idx]: Date.now() }));
      return idx;
    });
  }, []);

  const handleFile = useCallback(async (file: File) => {
    setAppState("uploading");
    setCurrentStage(-1);
    setStageStartTimes({});
    setStartTime(Date.now());
    setVideoDuration(null);

    const fd = new FormData();
    fd.append("file", file);

    let data: { job_id: string; token: string; duration: number; detail?: string; error?: string };
    try {
      const res = await fetch(api.process, { method: "POST", body: fd });
      const text = await res.text();
      let parsed: typeof data;
      try {
        parsed = JSON.parse(text);
      } catch {
        setErrorMsg(`Server error: ${text.slice(0, 120)}`);
        setAppState("error");
        return;
      }
      if (!res.ok) {
        setErrorMsg(parsed.detail ?? parsed.error ?? "Upload failed");
        setAppState("error");
        return;
      }
      data = parsed;
    } catch (e) {
      setErrorMsg(`Upload failed: ${e}`);
      setAppState("error");
      return;
    }

    if (data.duration) setVideoDuration(data.duration);
    setJobId(data.job_id);
    setJobToken(data.token);
    setAppState("processing");
    advanceToStage(0);

    // Open SSE stream
    const url = api.stream(data.job_id, data.token);
    const es = new EventSource(url);
    evtSourceRef.current = es;

    es.onmessage = (e) => {
      const event = JSON.parse(e.data);

      if (event.type === "duration" && event.duration) setVideoDuration(event.duration);

      if (event.type === "stage") {
        const lower = (event.stage as string).toLowerCase();
        const idx = STAGES.findIndex(s => lower.includes(s.match));
        if (idx !== -1) advanceToStage(idx);
      }

      if (event.type === "done") {
        es.close();
        setAppState("done");
      }

      if (event.type === "error") {
        es.close();
        setErrorMsg(event.msg ?? "Processing failed.");
        setAppState("error");
      }
    };

    es.onerror = () => { /* SSE auto-reconnects; handled by server keepalive */ };
  }, [advanceToStage]);

  const reset = useCallback(() => {
    evtSourceRef.current?.close();
    evtSourceRef.current = null;
    setAppState("idle");
    setCurrentStage(-1);
    setErrorMsg("");
    setJobId("");
    setJobToken("");
    setVideoDuration(null);
    setStartTime(null);
    setStageStartTimes({});
  }, []);

  return (
    <div className="w-full max-w-xl space-y-6">
      {/* Logo */}
      <div className="text-center space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">
          <span className="text-indigo-400">aiclipsyour</span>
          <span className="text-gray-400">.video</span>
        </h1>
        <p className="text-gray-600 text-xs">drop a video. get a clean cut.</p>
      </div>

      {/* Upload zone — hidden while processing/done */}
      {(appState === "idle" || appState === "error") && (
        <Dropzone onFile={handleFile} />
      )}

      {/* Progress */}
      {(appState === "uploading" || appState === "processing") && (
        <ProgressPanel
          currentStage={currentStage}
          uploading={appState === "uploading"}
          videoDuration={videoDuration}
          startTime={startTime}
          stageStartTimes={stageStartTimes}
        />
      )}

      {/* Done */}
      {appState === "done" && (
        <DoneBar
          jobId={jobId}
          token={jobToken}
          startTime={startTime}
          onReset={reset}
        />
      )}

      {/* Error */}
      {appState === "error" && (
        <ErrorBar message={errorMsg} onReset={reset} />
      )}
    </div>
  );
}
