import { useRef, useState } from "react";
import { isYouTubeUrl } from "@/lib/format";

interface ComposerProps {
  onSend: (text: string) => void;
  onIngest: (url: string) => void;
}

export function Composer({ onSend, onIngest }: ComposerProps) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  const submit = () => {
    const v = value.trim();
    if (!v) return;
    if (isYouTubeUrl(v)) onIngest(v);
    else onSend(v);
    setValue("");
    if (ref.current) ref.current.style.height = "auto";
  };

  const grow = () => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 140) + "px";
  };

  return (
    <div className="shrink-0 bg-gradient-to-b from-transparent to-ink px-6 pb-5 pt-3">
      <div className="mx-auto max-w-[760px]">
        <div className="flex items-end gap-2.5 rounded-[15px] border border-line-strong bg-ink-panel py-2 pl-3.5 pr-2">
          <button
            onClick={() => onIngest("")}
            aria-label="Index a YouTube video"
            title="Index a YouTube video"
            className="flex h-[34px] w-[34px] shrink-0 cursor-pointer items-center justify-center rounded-[10px] border border-line bg-ink-chip text-[17px] leading-none text-mut-400 hover:bg-[#222a33] hover:text-accent"
          >
            +
          </button>
          <textarea
            ref={ref}
            rows={1}
            value={value}
            onChange={(e) => {
              setValue(e.target.value);
              grow();
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder="Ask about the videos, or paste a YouTube URL to index it…"
            className="max-h-[140px] min-h-[34px] flex-1 resize-none border-0 bg-transparent px-1 py-[7px] text-[14.5px] leading-[1.45] text-mut-100 outline-none"
          />
          <button
            onClick={submit}
            aria-label="Send message"
            className="flex h-[34px] w-[34px] shrink-0 cursor-pointer items-center justify-center rounded-[10px] bg-accent text-[16px] font-bold text-ink hover:brightness-110"
          >
            ↑
          </button>
        </div>
        <p className="mx-0.5 mt-2 text-center text-[11px] text-mut-800">
          Answers are grounded in video transcripts and may be incomplete. Check the cited source.
        </p>
      </div>
    </div>
  );
}
