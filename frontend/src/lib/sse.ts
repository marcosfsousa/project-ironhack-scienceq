import type { GenerationMeta, Source } from "@/types";

export interface ChatStreamHandlers {
  onToken: (token: string) => void;
  onSources: (sources: Source[]) => void;
  onDone: () => void;
  onMeta?: (meta: GenerationMeta) => void;
}

/**
 * Stream an answer from POST /api/chat/stream.
 *
 * Backend SSE protocol:
 *   data: [META]{ "ai_generated", "model", "mode" }   — leads the stream
 *   data: <token>
 *   data: [SOURCES][{ "title","timestamp","link","score","rerank_score","text" }, ...]
 *   data: [DONE]
 *
 * Multi-line tokens (e.g. the metadata list) arrive as multiple consecutive
 * data: lines within a single event. Per the SSE spec, those lines are joined
 * with \n and dispatched as one onToken call — so whitespace-pre-wrap in the
 * chat bubble renders them correctly.
 */
export async function streamChat(
  question: string,
  signal: AbortSignal,
  h: ChatStreamHandlers,
  history: { role: string; content: string }[] = []
): Promise<void> {
  const res = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ question, history }),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(`chat stream failed: ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let eventData = ""; // accumulates data: lines within one SSE event

  const dispatchEvent = (data: string) => {
    if (!data) return;
    if (data === "[DONE]") { h.onDone(); return; }
    if (data.startsWith("[META]")) {
      try {
        h.onMeta?.(JSON.parse(data.slice("[META]".length)) as GenerationMeta);
      } catch (e) {
        console.warn("Failed to parse [META] frame", e);
      }
      return;
    }
    if (data.startsWith("[SOURCES]")) {
      try {
        h.onSources(JSON.parse(data.slice("[SOURCES]".length)) as Source[]);
      } catch (e) {
        console.warn("Failed to parse [SOURCES] frame", e);
      }
      return;
    }
    h.onToken(data);
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    let nl: number;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).replace(/\r$/, "");
      buf = buf.slice(nl + 1);

      if (line === "") {
        // Blank line = SSE event boundary.
        dispatchEvent(eventData);
        eventData = "";
      } else if (line.startsWith("data:")) {
        const chunk = line.slice(5).replace(/^ /, "");
        eventData = eventData === "" ? chunk : eventData + "\n" + chunk;
      }
      // id:, event:, retry: are intentionally ignored
    }
  }
  // Flush if the stream ended without a trailing blank line.
  dispatchEvent(eventData);
}
