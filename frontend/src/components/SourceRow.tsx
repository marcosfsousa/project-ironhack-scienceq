import type { Source } from "@/types";
import { embedUrl, tsToSec, youTubeStart } from "@/lib/format";

interface SourceRowProps {
  source: Source;
  index: number; // 0-based; shown as index+1
  isOpen: boolean;
  onToggle: () => void;
}

/** A numbered source with a rerank-score bar that expands to an inline player. */
export function SourceRow({ source, index, isOpen, onToggle }: SourceRowProps) {
  const pct = Math.round((source.rerank_score ?? source.score) * 100);
  const start = youTubeStart(source.link) || tsToSec(source.timestamp);

  return (
    <div className="overflow-hidden rounded-[10px] border border-line bg-ink-card">
      <button
        onClick={onToggle}
        aria-expanded={isOpen}
        aria-controls={`source-${index}-player`}
        aria-label={isOpen ? `Collapse source ${index + 1}` : `Expand source ${index + 1}`}
        className="flex w-full cursor-pointer items-center gap-3 border-0 bg-transparent px-3 py-2.5 text-left font-sans hover:bg-ink-hover"
      >
        <span className="flex h-[22px] w-[22px] shrink-0 items-center justify-center rounded-md border border-accent-bd bg-accent-soft font-mono text-[11px] font-bold text-accent">
          {index + 1}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-[13px] font-medium text-mut-200">
            {source.title}
          </span>
          <span className="mt-0.5 block text-[11.5px] text-mut-600">
            {/* channel/topic if the backend supplies them in text or elsewhere */}
            {source.text ? source.text.slice(0, 70) + "…" : ""}
          </span>
        </span>
        <span className="flex shrink-0 items-center gap-2.5">
          <span className="font-mono text-[11px] text-mut-400">{source.timestamp}</span>
          <span className="h-1 w-[38px] overflow-hidden rounded-[3px] bg-[#20262e]">
            <span
              className="block h-full origin-left animate-barGrow bg-accent"
              style={{ width: `${pct}%` }}
            />
          </span>
          <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-[7px] border border-accent-bd bg-accent-soft text-[10px] text-accent">
            {isOpen ? "✕" : "▶"}
          </span>
        </span>
      </button>

      {isOpen && (
        <div id={`source-${index}-player`} className="aspect-video border-t border-line bg-black">
          <iframe
            src={embedUrl(source.link, { autoplay: true, startSec: start })}
            title={source.title}
            loading="lazy"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
            allowFullScreen
            className="block h-full w-full border-0"
          />
        </div>
      )}
    </div>
  );
}
