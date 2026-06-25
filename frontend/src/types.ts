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

export interface ChatMessage {
  id: string;
  role: Role;
  text: string;
  sources: Source[];
  status: MessageStatus;
  error?: string;
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
