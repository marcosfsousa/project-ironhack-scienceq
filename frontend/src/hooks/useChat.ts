import { useCallback, useRef, useState } from "react";
import type { ChatMessage } from "@/types";
import { streamChat } from "@/lib/sse";
import { uid } from "@/lib/format";

/**
 * Chat state machine over the SSE endpoint.
 * - send(text): appends a user message + an empty assistant message, then
 *   streams tokens into it. Sources arrive in one frame; [DONE] flips status.
 * - stop(): aborts the in-flight stream.
 * - reset(): clears the transcript ("New conversation").
 */
export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const ctrl = useRef<AbortController | null>(null);

  const patch = useCallback((id: string, fn: (m: ChatMessage) => ChatMessage) => {
    setMessages((ms) => ms.map((m) => (m.id === id ? fn(m) : m)));
  }, []);

  const send = useCallback(
    async (text: string) => {
      const q = text.trim();
      if (!q) return;

      const aId = uid("a");
      const history = messages.map((m) => ({ role: m.role, content: m.text }));

      setMessages((ms) => [
        ...ms,
        { id: uid("u"), role: "user", text: q, sources: [], status: "done" },
        { id: aId, role: "assistant", text: "", sources: [], status: "streaming" },
      ]);

      ctrl.current?.abort();
      ctrl.current = new AbortController();

      try {
        await streamChat(
          q,
          ctrl.current.signal,
          {
            onToken: (t) => patch(aId, (m) => ({ ...m, text: m.text + t })),
            onSources: (s) => patch(aId, (m) => ({ ...m, sources: s })),
            onMeta: (g) => patch(aId, (m) => ({ ...m, generation: g })),
            onDone: () => patch(aId, (m) => ({ ...m, status: "done" })),
          },
          history
        );
      } catch (e) {
        if ((e as Error).name === "AbortError") {
          patch(aId, (m) => ({ ...m, status: "done" }));
          return;
        }
        patch(aId, (m) => ({ ...m, status: "error", error: String(e) }));
      }
    },
    [messages, patch]
  );

  const stop = useCallback(() => ctrl.current?.abort(), []);
  const reset = useCallback(() => {
    ctrl.current?.abort();
    setMessages([]);
  }, []);

  const isStreaming = messages.some((m) => m.status === "streaming");

  return { messages, send, stop, reset, isStreaming };
}
