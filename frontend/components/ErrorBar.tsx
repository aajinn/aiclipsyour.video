"use client";

interface Props {
  message: string;
  onReset: () => void;
}

export default function ErrorBar({ message, onReset }: Props) {
  return (
    <div className="flex items-center justify-between bg-gray-900 border border-red-900 rounded-xl px-5 py-4">
      <p className="text-red-400 text-xs">{message || "Something went wrong."}</p>
      <button onClick={onReset} className="text-gray-600 hover:text-gray-400 text-xs ml-4 transition">
        Try again
      </button>
    </div>
  );
}
