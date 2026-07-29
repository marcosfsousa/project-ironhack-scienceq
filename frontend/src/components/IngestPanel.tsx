import { useState } from "react";
import type { IngestState } from "@/hooks/useIngest";
import { INGEST_STEPS } from "@/hooks/useIngest";

interface IngestPanelProps {
  state: IngestState;
  onClose: () => void;
  onAsk: (title?: string) => void;
  onStart: (url: string) => void;
}

/** Slides over the sidebar while a video is being indexed. */
export function IngestPanel({ state, onClose, onAsk, onStart }: IngestPanelProps) {
  const [inputUrl, setInputUrl] = useState("");
  const failed = state.phase === "failed";
  const complete = state.phase === "complete";

  if (state.phase === "idle") {
    const submit = () => {
      const v = inputUrl.trim();
      if (v) { onStart(v); setInputUrl(""); }
    };
    return (
      <div className="absolute inset-0 z-20 flex animate-slideIn flex-col bg-ink-sidebar">
        <div className="border-b border-line px-[18px] pb-3.5 pt-5">
          <div className="flex items-center justify-between">
            <span className="text-[13px] font-semibold tracking-[0.02em] text-mut-100">
              Index a video
            </span>
            <button
              onClick={onClose}
              className="flex h-[26px] w-[26px] cursor-pointer items-center justify-center rounded-[7px] border border-line-strong bg-transparent text-[14px] text-mut-500 hover:bg-[#181d24]"
            >
              ✕
            </button>
          </div>
          <input
            autoFocus
            type="url"
            value={inputUrl}
            onChange={(e) => setInputUrl(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submit()}
            placeholder="Paste a YouTube URL…"
            className="mt-3 w-full rounded-lg border border-line bg-ink-panel px-3 py-2 font-mono text-[11.5px] text-mut-200 outline-hidden placeholder:text-mut-700 focus:border-accent"
          />
          <button
            onClick={submit}
            className="mt-2 w-full cursor-pointer rounded-[9px] border-0 bg-accent py-2 text-[13px] font-semibold text-ink hover:brightness-110"
          >
            Index video
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="absolute inset-0 z-20 flex animate-slideIn flex-col bg-ink-sidebar">
      <div className="border-b border-line px-[18px] pb-3.5 pt-5">
        <div className="flex items-center justify-between">
          <span className="text-[13px] font-semibold tracking-[0.02em] text-mut-100">
            Indexing video
          </span>
          <button
            onClick={onClose}
            className="flex h-[26px] w-[26px] cursor-pointer items-center justify-center rounded-[7px] border border-line-strong bg-transparent text-[14px] text-mut-500 hover:bg-[#181d24]"
          >
            ✕
          </button>
        </div>
        <div className="mt-3 flex items-center gap-2 rounded-lg border border-line bg-ink-panel px-2.5 py-2.5">
          <span className="h-[7px] w-[7px] shrink-0 rounded-xs bg-[#ff5252]" />
          <span className="truncate font-mono text-[11.5px] text-mut-400">
            {state.url || "youtube.com/watch?v=…"}
          </span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-[18px]">
        {INGEST_STEPS.map((label, i) => {
          const stepState =
            complete || i < state.stepIndex
              ? "done"
              : i === state.stepIndex && !failed
                ? "active"
                : failed && i === state.stepIndex
                  ? "failed"
                  : "pending";
          return (
            <div key={label} className="flex items-start gap-3 py-[7px]">
              <Marker state={stepState} />
              <span
                className={
                  "pt-px text-[13px] leading-normal " +
                  (stepState === "pending"
                    ? "text-mut-700"
                    : stepState === "active"
                      ? "text-mut-100"
                      : stepState === "failed"
                        ? "text-[#f0a0a0]"
                        : "text-mut-400")
                }
              >
                {label}
              </span>
            </div>
          );
        })}
      </div>

      {complete && state.result && (
        <div className="animate-fadeUp border-t border-line px-[18px] pb-5 pt-4">
          <div className="mb-3 flex items-center gap-[7px]">
            <span className="flex h-4 w-4 items-center justify-center rounded-full bg-ok text-[10px] font-bold text-[#06120c]">
              ✓
            </span>
            <span className="text-[12.5px] font-semibold text-ok-soft">
              {state.result.already_indexed ? "Already indexed" : "Indexed successfully"}
            </span>
          </div>
          <div className="rounded-[10px] border border-line bg-ink-panel p-[13px]">
            <div className="text-[13.5px] font-semibold leading-[1.35] text-mut-100">
              {state.result.title ?? "Untitled video"}
            </div>
            {state.result.channel && (
              <div className="mt-[5px] text-[12px] text-mut-500">{state.result.channel}</div>
            )}
            <div className="mt-[11px] flex gap-[7px]">
              {state.result.topic && (
                <span className="rounded-[5px] border border-accent-bd bg-accent-soft px-[7px] py-0.5 text-[11px] text-accent">
                  {state.result.topic}
                </span>
              )}
              {typeof state.result.chunk_count === "number" && (
                <span className="rounded-[5px] bg-ink-chip px-[7px] py-0.5 font-mono text-[11px] text-mut-400">
                  {state.result.chunk_count} chunks
                </span>
              )}
            </div>
          </div>
          <button
            onClick={() => onAsk(state.result?.title)}
            className="mt-3 w-full cursor-pointer rounded-[9px] border-0 bg-accent py-2.5 text-[13px] font-semibold text-ink"
          >
            Ask about this video
          </button>
        </div>
      )}

      {failed && (
        <div className="animate-fadeUp border-t border-line px-[18px] pb-5 pt-4">
          <div className="rounded-[10px] border border-[#5a2020] bg-[#1a1012] p-[13px] text-[12.5px] leading-[1.5] text-[#f0a0a0]">
            Couldn’t index this video. {state.result?.error ?? "Please check the URL and try again."}
          </div>
          <button
            onClick={onClose}
            className="mt-3 w-full cursor-pointer rounded-[9px] border border-line-strong bg-transparent py-2.5 text-[13px] font-medium text-mut-200 hover:bg-ink-panel"
          >
            Close
          </button>
        </div>
      )}
    </div>
  );
}

function Marker({ state }: { state: string }) {
  if (state === "done")
    return (
      <span className="flex h-[18px] w-[18px] items-center justify-center rounded-full bg-accent text-[10px] font-bold text-ink">
        ✓
      </span>
    );
  if (state === "active")
    return (
      <span className="h-[18px] w-[18px] animate-spin2 rounded-full border-2 border-white/10 border-t-accent" />
    );
  if (state === "failed")
    return (
      <span className="flex h-[18px] w-[18px] items-center justify-center rounded-full bg-[#5a2020] text-[10px] font-bold text-[#f0a0a0]">
        ✕
      </span>
    );
  return <span className="box-border h-[18px] w-[18px] rounded-full border-2 border-white/10" />;
}
