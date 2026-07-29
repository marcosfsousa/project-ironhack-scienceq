import { useRef, useState } from "react";
import { isYouTubeUrl } from "@/lib/format";
import { useIsMobile } from "@/hooks/useIsMobile";

interface ComposerProps {
  onSend: (text: string) => void;
  onIngest: (url: string) => void;
  onOpenPrivacy: () => void;
  isMobile: boolean;
}

export function Composer({ onSend, onIngest, onOpenPrivacy, isMobile }: ComposerProps) {
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

  // Narrow catches the tablet range (768-1100px) where the sidebar eats viewport width.
  const isNarrow = useIsMobile(1100);

  const btnSize = isMobile ? "h-11 w-11" : "h-[34px] w-[34px]";
  const btnRadius = isMobile ? "rounded-[12px]" : "rounded-[10px]";
  const placeholder = isMobile
    ? "Ask about the videos…"
    : isNarrow
    ? "Ask about the videos or add a URL…"
    : "Ask about the videos, or paste a YouTube URL to index it…";

  return (
    <div
      className="shrink-0 bg-linear-to-b from-transparent to-ink px-3 pt-3 sm:px-6"
      style={{
        paddingBottom: isMobile
          ? "max(14px, env(safe-area-inset-bottom))"
          : "20px",
      }}
    >
      <div className="mx-auto max-w-[760px]">
        <div className="flex items-end gap-2.5 rounded-[15px] border border-line-strong bg-ink-panel py-2 pl-3.5 pr-2">
          <button
            onClick={() => onIngest("")}
            aria-label="Index a YouTube video"
            title="Index a YouTube video"
            className={`${btnSize} ${btnRadius} flex shrink-0 cursor-pointer items-center justify-center border border-line bg-ink-chip text-[17px] leading-none text-mut-400 hover:bg-[#222a33] hover:text-accent`}
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
            placeholder={placeholder}
            // placeholder:text-mut-400 is explicit on purpose. v3's preflight
            // pinned placeholders to a fixed grey; v4 derives them from
            // currentColor at 50%, which against text-mut-100 renders dimmer.
            // Naming the colour keeps this off framework defaults entirely.
            className={`max-h-[140px] min-w-0 flex-1 resize-none border-0 bg-transparent px-1 leading-[1.45] text-mut-100 outline-hidden placeholder:text-mut-400${isMobile ? " placeholder:text-[14px]" : ""}`}
            style={{
              minHeight: isMobile ? "44px" : "34px",
              fontSize: isMobile ? "16px" : "14.5px",
              paddingTop: isMobile ? "10px" : "7px",
              paddingBottom: isMobile ? "10px" : "7px",
            }}
          />
          <button
            onClick={submit}
            aria-label="Send message"
            className={`${btnSize} ${btnRadius} flex shrink-0 cursor-pointer items-center justify-center bg-accent text-[16px] font-bold text-ink hover:brightness-110`}
          >
            ↑
          </button>
        </div>
        <p className="mx-0.5 mt-2 text-center text-[11px] text-mut-800">
          Answers are grounded in video transcripts and may be incomplete. Check
          the cited source.{" "}
          <button
            onClick={onOpenPrivacy}
            className="cursor-pointer border-0 bg-transparent p-0 text-[11px] text-mut-600 underline underline-offset-2 hover:text-accent"
          >
            Privacy &amp; your data
          </button>
        </p>
      </div>
    </div>
  );
}
