// Strip LLM-generated citation markers so prose renders clean.
// The SOURCES rail below the video already carries all attribution.

// Matches [Title, 03:11] or [Title, 03:11-04:13] or [Title, 03:11–04:13]
const INLINE_CITE_RE = /\s*\[[^\][]+?,\s*[\d:–‑-]+\]/g;

// Matches a trailing "Sources" / "References" block the LLM sometimes appends
const SOURCES_BLOCK_RE = /\n{1,2}(Sources|References|Citations)[^\n]*(\n[\s\S]*)?$/i;

export function cleanAnswer(text: string): string {
  return text
    .replace(SOURCES_BLOCK_RE, "")
    .replace(INLINE_CITE_RE, "")
    .trimEnd();
}
