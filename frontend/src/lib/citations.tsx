import type { ReactNode } from "react";
import type { Source } from "@/types";

/**
 * The backend writes inline citations into the answer text as
 * "[Video Title, mm:ss]". We turn each one into a numbered superscript chip
 * that maps to the matching entry in the [SOURCES] array, so the prose reads
 * cleanly and the number lines up with the Sources rail below.
 *
 * Returns an array of ReactNodes (strings + <sup> chips) for {children}.
 */
const CITE_RE = /\[([^\][]+?),\s*(\d{1,2}:\d{2}(?::\d{2})?)\]/g;

export function renderCitations(
  text: string,
  sources: Source[],
  onCite?: (index: number) => void
): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  let key = 0;
  let m: RegExpExecArray | null;

  CITE_RE.lastIndex = 0;
  while ((m = CITE_RE.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const title = m[1].trim().toLowerCase();
    const idx = sources.findIndex(
      (s) => s.title.toLowerCase() === title || s.title.toLowerCase().startsWith(title)
    );
    if (idx === -1) {
      out.push(m[0]); // no matching source — leave the bracket as written
    } else {
      const n = idx + 1;
      out.push(
        <sup
          key={`c${key++}`}
          title={`${sources[idx].title} · ${sources[idx].timestamp}`}
          onClick={() => onCite?.(idx)}
          className="mx-[2px] cursor-pointer rounded-[5px] border border-accent-bd bg-accent-soft px-[5px] py-px align-super text-[10.5px] font-bold leading-tight text-accent"
        >
          {n}
        </sup>
      );
    }
    last = CITE_RE.lastIndex;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}
