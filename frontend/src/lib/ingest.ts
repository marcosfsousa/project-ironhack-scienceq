import type { IngestJob, IngestStatus } from "@/types";

/** Kick off indexing. POST /api/ingest -> 202 { job_id, status:"pending" }. */
export async function startIngestJob(url: string): Promise<IngestJob> {
  const res = await fetch("/api/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  if (!res.ok) throw new Error(`ingest start failed: ${res.status}`);
  return (await res.json()) as IngestJob;
}

/** One status read. GET /api/ingest/:job_id. */
export async function getIngestStatus(jobId: string): Promise<IngestStatus> {
  const res = await fetch(`/api/ingest/${encodeURIComponent(jobId)}`);
  if (!res.ok) throw new Error(`ingest status failed: ${res.status}`);
  return (await res.json()) as IngestStatus;
}

/**
 * Poll until the job resolves. The backend only exposes pending/complete/failed
 * (no per-step granularity), so the UI's labelled stepper is an indeterminate
 * affordance driven by elapsed time — see useIngest. This helper just resolves
 * with the terminal status.
 */
export async function pollIngest(
  jobId: string,
  opts: { intervalMs?: number; signal?: AbortSignal } = {}
): Promise<IngestStatus> {
  const interval = opts.intervalMs ?? 1200;
  while (true) {
    if (opts.signal?.aborted) throw new DOMException("aborted", "AbortError");
    const status = await getIngestStatus(jobId);
    if (status.status !== "pending") return status;
    await new Promise((r) => setTimeout(r, interval));
  }
}
