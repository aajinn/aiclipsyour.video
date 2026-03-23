"use client";

import { api } from "@/lib/config";

interface Props {
  jobId:     string;
  token:     string;
  startTime: number | null;
  onReset:   () => void;
}

export default function DoneBar({ jobId, token, startTime, onReset }: Props) {
  const elapsed = startTime ? Math.round((Date.now() - startTime) / 1000) : 0;
  const downloadUrl = api.download(jobId, token);

  return (
    <div className="flex items-center justify-between bg-gray-900 border border-green-900 rounded-xl px-5 py-4">
      <div className="space-y-0.5">
        <p className="text-green-400 text-sm font-semibold">Done ✓</p>
        <p className="text-gray-500 text-xs">processed in {elapsed}s</p>
      </div>
      <div className="flex items-center gap-3">
        <a
          href={downloadUrl}
          className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-xs font-semibold transition"
        >
          ⬇ Download
        </a>
        <button onClick={onReset} className="text-gray-600 hover:text-gray-400 text-xs transition">
          New
        </button>
      </div>
    </div>
  );
}
