import { useEffect, useRef, useCallback } from "react";
import type { Accent, ChatMessage as Msg } from "@/types";
import { ChatMessage } from "./ChatMessage";
import { Hero } from "./Hero";
import { Composer } from "./Composer";

const ACCENTS: Accent[] = ["Indigo", "Blue", "Amber", "Cyan"];
const SWATCH: Record<Accent, string> = {
  Indigo: "#8b93f8",
  Blue: "#4d96ff",
  Amber: "#f6b73c",
  Cyan: "#34d6ee",
};

interface ChatViewProps {
  messages: Msg[];
  suggestions: string[];
  accent: Accent;
  onAccentChange: (a: Accent) => void;
  onSend: (text: string) => void;
  onIngest: (url: string) => void;
}

export function ChatView({
  messages,
  suggestions,
  accent,
  onAccentChange,
  onSend,
  onIngest,
}: ChatViewProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const buttonRefs = useRef<Map<Accent, HTMLButtonElement>>(new Map());

  // Auto-stick to the bottom while a conversation is active.
  useEffect(() => {
    if (messages.length === 0) return;
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  // Roving tabIndex for the accent radio group: arrow keys move focus+selection.
  // Focus must follow selection so repeated keypresses continue to cycle.
  const handleAccentKeyDown = useCallback(
    (e: React.KeyboardEvent, current: Accent) => {
      const idx = ACCENTS.indexOf(current);
      let next = -1;
      if (e.key === "ArrowRight" || e.key === "ArrowDown") next = (idx + 1) % ACCENTS.length;
      if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = (idx - 1 + ACCENTS.length) % ACCENTS.length;
      if (next === -1) return;
      e.preventDefault();
      const nextAccent = ACCENTS[next];
      onAccentChange(nextAccent);
      buttonRefs.current.get(nextAccent)?.focus();
    },
    [onAccentChange]
  );

  return (
    <main className="flex h-full min-w-0 flex-1 flex-col bg-ink">
      <div className="flex h-[52px] shrink-0 items-center justify-end gap-2.5 border-b border-line px-[22px]">
        <div className="flex items-center gap-1" role="radiogroup" aria-label="Accent color">
          {ACCENTS.map((a) => (
            <button
              key={a}
              ref={(el) => { if (el) buttonRefs.current.set(a, el); else buttonRefs.current.delete(a); }}
              aria-label={a}
              aria-checked={accent === a}
              role="radio"
              tabIndex={accent === a ? 0 : -1}
              onClick={() => onAccentChange(a)}
              onKeyDown={(e) => handleAccentKeyDown(e, a)}
              className={
                "h-4 w-4 rounded-full border transition " +
                (accent === a ? "border-white/60" : "border-transparent")
              }
              style={{ background: SWATCH[a] }}
            />
          ))}
        </div>
        <span className="flex items-center gap-[7px] rounded-[20px] border border-line bg-ink-panel px-[11px] py-[5px] text-[11.5px] text-mut-600">
          <span className="h-[6px] w-[6px] rounded-full bg-ok" />
          GPT-OSS-120B · Groq
        </span>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-[760px] px-6 pb-10 pt-[26px]">
          {messages.length === 0 ? (
            <Hero suggestions={suggestions} onPick={onSend} />
          ) : (
            messages.map((m) => <ChatMessage key={m.id} message={m} />)
          )}
        </div>
      </div>

      <Composer onSend={onSend} onIngest={onIngest} />
    </main>
  );
}
