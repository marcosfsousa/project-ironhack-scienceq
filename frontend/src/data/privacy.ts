// Single source of truth for the privacy notice (issue #17).
//
// The notice describes what happens to a question when it is asked. Concrete
// vendor names live here ONCE — the notice copy and every surface reference
// these constants rather than repeating vendor names, so a provider change is
// a one-line edit here rather than a hunt across components.

export interface Processor {
  /** What this service does in the course of answering a question. */
  role: string;
  /** The concrete provider — named once, here. */
  vendor: string;
  /** What is sent to it. */
  receives: string;
}

export const PROCESSORS: Processor[] = [
  {
    role: "Answer generation",
    vendor: "Groq",
    receives: "your question and the transcript excerpts retrieved for it",
  },
  {
    role: "Embeddings & reranking",
    vendor: "Cohere",
    receives: "your question text",
  },
  {
    role: "Vector search",
    vendor: "Pinecone",
    receives: "the numeric embedding of your question — not the text itself",
  },
];

export const PRIVACY_NOTICE = {
  /** Accessible name of the dialog and the text of its heading. */
  title: "How your questions are handled",
  intro:
    "When you ask a question, ScienceQ sends it to a few third-party services to find and generate an answer. Here is what goes where.",
  contextNote:
    "For follow-up questions, the recent conversation is sent along too, so answers stay in context.",
  notCollected: [
    "No account or sign-in — the product never asks who you are.",
    "No server-side history — your conversation lives only in this browser tab and is gone when you close it.",
  ],
  region:
    "ScienceQ is hosted in the EU (Google Cloud, europe-west1). Some providers above may process your question outside the EU.",
} as const;
