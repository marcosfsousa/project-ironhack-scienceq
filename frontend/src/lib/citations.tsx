// Strip LLM-generated citation markers so prose renders clean.
// The SOURCES rail below the video already carries all attribution.

// Matches [Title, 03:11] or [Title, 03:11-04:13] or [Title, 03:11–04:13]
const INLINE_CITE_RE = /\s*\[[^\][]+?,\s*[\d:–‑-]+\]/g;

// Matches a trailing section heading ("Sources", "References", "Citations") on its
// own line, followed by the rest of the text. Requires a newline after the heading
// word so it won't match inline uses like "Sources of this data include...".
const SOURCES_BLOCK_RE = /\n{1,2}(Sources|References|Citations)\s*[.:]?\s*\n[\s\S]*$/i;

export function cleanAnswer(text: string): string {
  return text
    .replace(SOURCES_BLOCK_RE, "")
    .replace(INLINE_CITE_RE, "")
    .trimEnd();
}
