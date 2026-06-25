import { useState } from "react";
import type { ChatMessage as Msg } from "@/types";
import { renderCitations } from "@/lib/citations";
import { useIsMobile } from "@/hooks/useIsMobile";
import { BrandMark } from "./BrandMark";
import { SourceRow } from "./SourceRow";
import { VideoEmbed } from "./VideoEmbed";

interface ChatMessageProps {
  message: Msg;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isMobile = useIsMobile();
  const [open, setOpen] = useState<Set<number>>(new Set());
  const toggle = (i: number) =>
    setOpen((prev) => {
      const next = new Set(prev);
      next.has(i) ? next.delete(i) : next.add(i);
      return next;
    });

  if (message.role === "user") {
    return (
      <div className="mb-1 mt-[22px] flex animate-fadeUp justify-end">
        <div
          className="rounded-[14px_14px_4px_14px] border border-accent-bd bg-accent-soft px-[15px] py-[11px] text-[14.5px] leading-normal text-mut-100"
          style={{ maxWidth: isMobile ? "90%" : "80%" }}
        >
          {message.text}
        </div>
      </div>
    );
  }

  const streaming = message.status === "streaming";
  const done = message.status === "done";
  const top = message.sources[0];

  const avatarSize = isMobile ? "h-6 w-6" : "h-[30px] w-[30px]";
  const avatarRadius = isMobile ? "rounded-[7px]" : "rounded-[9px]";
  const msgGap = isMobile ? "gap-[9px]" : "gap-[13px]";

  return (
    <div className={`mb-1.5 mt-5 flex animate-fadeUp ${msgGap}`}>
      <div className={`flex ${avatarSize} shrink-0 items-center justify-center ${avatarRadius} border border-accent-bd bg-accent-soft text-accent`}>
        <BrandMark size={isMobile ? 13 : 17} className="flex text-accent" />
      </div>

      <div className="min-w-0 flex-1 pt-[3px]">
        {streaming && message.text === "" ? (
          <div className="flex items-center gap-2.5 py-0.5 text-[14px] text-mut-600">
            <span className="h-[7px] w-[7px] animate-pulse2 rounded-full bg-accent" />
            Searching transcripts…
          </div>
        ) : (
          <div className="whitespace-pre-wrap text-[15.5px] leading-[1.78] text-mut-250">
            {done
              ? renderCitations(message.text, message.sources, (i) =>
                  setOpen((p) => new Set(p).add(i))
                )
              : message.text}
            {streaming && (
              <span className="ml-px inline-block h-[15px] w-2 translate-y-0.5 animate-blink bg-accent align-[-2px]" />
            )}
          </div>
        )}

        {message.status === "error" && (
          <div className="mt-3 rounded-[10px] border border-[#5a2020] bg-[#1a1012] px-3 py-2.5 text-[13px] text-[#f0a0a0]">
            Something went wrong streaming this answer. {message.error}
          </div>
        )}

        {done && top && (
          <div className="animate-fadeUp">
            <VideoEmbed
              title={top.title}
              link={top.link}
              timestamp={top.timestamp}
            />

            <div className="mt-[18px]">
              <div className="mb-2.5 text-[11px] font-semibold tracking-[0.12em] text-mut-700">
                SOURCES
              </div>
              <div className="grid gap-[7px]">
                {message.sources.map((s, i) => (
                  <SourceRow
                    key={`${s.link}-${i}`}
                    source={s}
                    index={i}
                    isOpen={open.has(i)}
                    onToggle={() => toggle(i)}
                  />
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
