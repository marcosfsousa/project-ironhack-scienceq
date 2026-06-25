import type { Source } from "@/types";

export interface ChatStreamHandlers {
  onToken: (token: string) => void;
  onSources: (sources: Source[]) => void;
  onDone: () => void;
}

/**
 * Stream an answer from POST /api/chat/stream.
 *
 * Backend SSE protocol (from the brief):
 *   data: <token>
 *   data: [SOURCES][{ "title","timestamp","link","score","rerank_score","text" }, ...]
 *   data: [DONE]
 *
 * Tokens are appended; the [SOURCES] frame carries a JSON array; [DONE] closes.
 * We read the raw body stream and split on newlines (SSE frames end in \n\n,
 * but tokens can contain spaces and the stream chunks arbitrarily, so we buffer).
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

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    // Process complete lines; keep the trailing partial in the buffer.
    let nl: number;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      handleLine(line, h);
    }
  }
  if (buf) handleLine(buf, h);
}

function handleLine(line: string, h: ChatStreamHandlers): void {
  const trimmed = line.replace(/\r$/, "");
  if (!trimmed.startsWith("data:")) return; // ignore comments / blank separators
  const data = trimmed.slice(5).replace(/^ /, ""); // strip "data:" + one space

  if (data === "[DONE]") {
    h.onDone();
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
  // Anything else is an answer token (may be a single word or partial).
  h.onToken(data);
}
