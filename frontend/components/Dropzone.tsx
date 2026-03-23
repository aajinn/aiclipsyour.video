"use client";

import { useRef, useState } from "react";

interface Props {
  onFile: (file: File) => void;
}

export default function Dropzone({ onFile }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [selected, setSelected] = useState<File | null>(null);
  const [hovering, setHovering] = useState(false);

  function pick(file: File) {
    setSelected(file);
  }

  function onDragOver(e: React.DragEvent) {
    e.preventDefault();
    setHovering(true);
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setHovering(false);
    const f = e.dataTransfer.files[0];
    if (f) pick(f);
  }

  return (
    <div
      className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-200 ${
        hovering ? "drop-hover" : "border-gray-800"
      }`}
      onClick={() => !selected && inputRef.current?.click()}
      onDragOver={onDragOver}
      onDragLeave={() => setHovering(false)}
      onDrop={onDrop}
    >
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        accept="video/*,audio/*"
        onChange={e => { const f = e.target.files?.[0]; if (f) pick(f); }}
      />

      {!selected ? (
        <div className="space-y-3">
          <div className="text-5xl select-none">🎬</div>
          <p className="text-gray-400 text-sm">Drop your video here</p>
          <p className="text-gray-700 text-xs">MP4 · MOV · MKV · AVI · WebM</p>
          <p className="text-gray-700 text-xs mt-1">minimum 3 minutes for AI cutting</p>
        </div>
      ) : (
        <div className="space-y-2">
          <div className="text-4xl select-none">✅</div>
          <p className="text-indigo-300 text-sm font-semibold truncate">{selected.name}</p>
          <p className="text-gray-600 text-xs">{(selected.size / 1024 / 1024).toFixed(1)} MB</p>
          <div className="flex items-center justify-center gap-3 mt-3">
            <button
              onClick={e => { e.stopPropagation(); onFile(selected); }}
              className="px-6 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-sm font-semibold transition"
            >
              Process →
            </button>
            <button
              onClick={e => { e.stopPropagation(); setSelected(null); if (inputRef.current) inputRef.current.value = ""; }}
              className="text-gray-600 hover:text-gray-400 text-xs transition"
            >
              Change
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
