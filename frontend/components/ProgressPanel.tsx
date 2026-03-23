"use client";

import { useEffect, useRef, useState } from "react";
import { STAGES } from "./Clipper";

interface Props {
  currentStage:    number;
  uploading:       boolean;
  videoDuration:   number | null;
  startTime:       number | null;
  stageStartTimes: Record<number, number>;
}

export default function ProgressPanel({ currentStage, uploading, videoDuration, startTime, stageStartTimes }: Props) {
  const [displayedPct, setDisplayedPct] = useState(2);
  const [elapsed,      setElapsed]      = useState("0:00");
  const [eta,          setEta]          = useState("estimating…");
  const rafRef = useRef<number>(0);

  // Target pct from current stage
  const targetPct = uploading ? 2 : (currentStage >= 0 ? STAGES[currentStage].pct : 2);

  // Smooth tween
  useEffect(() => {
    function tick() {
      setDisplayedPct(prev => {
        if (prev >= targetPct) return prev;
        const gap   = targetPct - prev;
        const speed = Math.max(0.03, Math.min(0.4, gap * 0.04));
        return Math.min(targetPct, prev + speed);
      });
      rafRef.current = requestAnimationFrame(tick);
    }
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [targetPct]);

  // Clock + ETA
  useEffect(() => {
    const id = setInterval(() => {
      if (!startTime) return;
      const s   = Math.floor((Date.now() - startTime) / 1000);
      const m   = Math.floor(s / 60);
      const sec = s % 60;
      setElapsed(`${m}:${String(sec).padStart(2, "0")}`);

      if (!videoDuration || currentStage < 0) return;
      const now          = Date.now();
      const elapsedS     = (now - startTime) / 1000;
      const estimatedTotal = videoDuration * 1.2;
      let weightDone = 0;
      for (let i = 0; i < currentStage; i++) weightDone += STAGES[i].weight;
      const stageStart   = stageStartTimes[currentStage] ?? now;
      const stageElapsed = (now - stageStart) / 1000;
      const stageEst     = estimatedTotal * STAGES[currentStage].weight;
      const stageFrac    = stageEst > 0 ? Math.min(0.9, stageElapsed / stageEst) : 0;
      weightDone += STAGES[currentStage].weight * stageFrac;
      const modelRemain  = estimatedTotal * (1 - weightDone);
      const actualRemain = weightDone > 0.05 ? (elapsedS / weightDone) * (1 - weightDone) : modelRemain;
      const remain       = modelRemain * 0.7 + actualRemain * 0.3;
      if (remain < 8) {
        setEta("almost done…");
      } else {
        const mins = Math.floor(remain / 60);
        const secs = Math.floor(remain % 60);
        setEta(mins > 0 ? `~${mins}m ${secs}s left` : `~${secs}s left`);
      }
    }, 1000);
    return () => clearInterval(id);
  }, [startTime, videoDuration, currentStage, stageStartTimes]);

  const stageLabel = uploading ? "Uploading…" : (currentStage >= 0 ? STAGES[currentStage].label + "…" : "Starting…");
  const stageHint  = uploading ? "Sending your video to the server…" : (currentStage >= 0 ? STAGES[currentStage].hint : "");

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6 space-y-4">
      {/* Top row */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Spinner />
          <span className="text-indigo-300 text-sm font-semibold">{stageLabel}</span>
        </div>
        <span className="text-gray-600 text-xs tabular-nums">{elapsed}</span>
      </div>

      {/* Bar */}
      <div className="w-full bg-gray-800 rounded-full h-2 overflow-hidden">
        <div
          className="h-2 rounded-full bar-active"
          style={{ width: `${displayedPct}%`, transition: "width 0.8s cubic-bezier(0.4,0,0.2,1)" }}
        />
      </div>

      {/* Stage pipeline */}
      <div className="flex items-start justify-between pt-1">
        {STAGES.map((s, i) => (
          <div key={s.label} className="flex flex-col items-center gap-1.5" style={{ width: `${100 / STAGES.length}%` }}>
            <div className={`w-2 h-2 rounded-full transition-all duration-300 ${
              i < currentStage  ? "bg-indigo-400" :
              i === currentStage ? "bg-indigo-500 dot-pulse" :
              "bg-gray-700"
            }`} />
            <span className={`text-center leading-tight transition-colors duration-300 ${
              i < currentStage  ? "text-indigo-400" :
              i === currentStage ? "text-indigo-300 font-semibold" :
              "text-gray-700"
            }`} style={{ fontSize: "9px" }}>
              {s.label}
            </span>
          </div>
        ))}
      </div>

      {/* ETA row */}
      <div className="flex items-center justify-between pt-1">
        <span className="text-gray-600 text-xs italic">{stageHint}</span>
        <span className="text-gray-500 text-xs tabular-nums">{eta}</span>
      </div>
    </div>
  );
}

function Spinner() {
  return (
    <svg className="w-4 h-4 text-indigo-400 flex-shrink-0 animate-spin" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
    </svg>
  );
}
