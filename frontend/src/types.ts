// Domain types — mirror the backend contract documented in the brief.

/** A retrieved transcript chunk returned in the [SOURCES] SSE frame. */
export interface Source {
  title: string;
  timestamp: string; // "mm:ss" or "h:mm:ss"
  link: string; // YouTube URL, usually already carrying &t=<seconds>
  score: number; // retrieval score 0..1
  rerank_score: number | null;
  text: string; // the chunk text (used for hover preview / debugging)
}

export type Role = "user" | "assistant";

export type MessageStatus = "streaming" | "done" | "error";

/**
 * Machine-readable generation provenance for an answer, from the [META] SSE
 * frame (and the blocking response's `generation` field). `ai_generated` is
 * true only for LLM-generated prose; static fallbacks report false with a
 * null model. Visible presentation is owned by the disclosure work (#11) —
 * this type only carries the data onto the message.
 *
 * `mode` is kept as a widened `string`: the server may add discriminators
 * (it already emits "static" as a defensive fallback), and an unknown value
 * must not break parsing of a frame we otherwise understand.
 */
export interface GenerationMeta {
  ai_generated: boolean;
  model: string | null;
  /** "generated" | "no_context" | "metadata" | "ingest" | "static" (fallback). */
  mode: string;
}

export interface ChatMessage {
  id: string;
  role: Role;
  text: string;
  sources: Source[];
  status: MessageStatus;
  error?: string;
  generation?: GenerationMeta;
}

/** GET /api/catalog item. */
export interface CatalogVideo {
  video_id: string;
  title: string;
  channel: string;
  topic: string;
  duration: string;
  url: string;
  source: "corpus" | "live";
}

/** POST /api/ingest response. */
export interface IngestJob {
  job_id: string;
  status: "pending";
}

/** GET /api/ingest/:job_id response. */
export interface IngestStatus {
  status: "pending" | "complete" | "failed";
  title?: string;
  channel?: string;
  topic?: string;
  chunk_count?: number;
  already_indexed?: boolean;
  error?: string;
}

export type Accent = "Indigo" | "Blue" | "Amber" | "Cyan";
export type SidebarDensity = "Detailed" | "Compact";

/** Topic group used by the corpus browser (derived client-side from CatalogVideo[]). */
export interface TopicGroup {
  topic: string;
  videos: CatalogVideo[];
}
