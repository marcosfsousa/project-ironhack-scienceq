import { useCallback, useRef, useState } from "react";
import type { IngestStatus } from "@/types";
import { startIngestJob, pollIngest } from "@/lib/ingest";

/**
 * The labelled steps shown in the slide-over panel. The backend only reports
 * pending → complete/failed, so this is a *presentational* stepper: while the
 * job is pending we advance through the labels on a timer to communicate
 * progress, then snap to the final state. Swap to real per-step events if the
 * backend ever emits them.
 */
export const INGEST_STEPS = [
  "URL detected",
  "Fetching video metadata",
  "Downloading transcript",
  "Splitting into chunks",
  "Generating embeddings",
  "Indexing to vector store",
] as const;

export interface IngestState {
  active: boolean;
  url: string;
  stepIndex: number; // 0..STEPS.length
  phase: "running" | "complete" | "failed";
  result: IngestStatus | null;
}

const IDLE: IngestState = { active: false, url: "", stepIndex: 0, phase: "running", result: null };

export function useIngest(onComplete?: (s: IngestStatus) => void) {
  const [state, setState] = useState<IngestState>(IDLE);
  const ticker = useRef<number | null>(null);
  const aborter = useRef<AbortController | null>(null);

  const clearTicker = () => {
    if (ticker.current != null) window.clearInterval(ticker.current);
    ticker.current = null;
  };

  const start = useCallback(
    async (url: string) => {
      clearTicker();
      aborter.current?.abort();
      aborter.current = new AbortController();
      setState({ active: true, url, stepIndex: 0, phase: "running", result: null });

      // Presentational stepper: advance until the second-to-last label, then
      // hold there until the real job resolves (so we never "complete" early).
      ticker.current = window.setInterval(() => {
        setState((s) =>
          s.phase === "running" && s.stepIndex < INGEST_STEPS.length - 1
            ? { ...s, stepIndex: s.stepIndex + 1 }
            : s
        );
      }, 850);

      try {
        const job = await startIngestJob(url);
        const result = await pollIngest(job.job_id, { signal: aborter.current.signal });
        clearTicker();
        if (result.status === "failed") {
          setState((s) => ({ ...s, phase: "failed", result }));
        } else {
          setState((s) => ({ ...s, stepIndex: INGEST_STEPS.length, phase: "complete", result }));
          onComplete?.(result);
        }
      } catch (e) {
        if ((e as Error).name === "AbortError") return;
        clearTicker();
        setState((s) => ({
          ...s,
          phase: "failed",
          result: { status: "failed", error: String(e) },
        }));
      }
    },
    [onComplete]
  );

  const close = useCallback(() => {
    clearTicker();
    aborter.current?.abort();
    setState(IDLE);
  }, []);

  return { ingest: state, start, close };
}
