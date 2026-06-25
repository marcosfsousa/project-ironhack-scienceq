import type { SidebarDensity, TopicGroup } from "@/types";
import type { IngestState } from "@/hooks/useIngest";
import { BrandMark } from "./BrandMark";
import { CorpusBrowser } from "./CorpusBrowser";
import { IngestPanel } from "./IngestPanel";

interface SidebarProps {
  groups: TopicGroup[];
  total: number;
  density: SidebarDensity;
  onPickVideo: (title: string) => void;
  onNewConversation: () => void;
  ingest: IngestState;
  onCloseIngest: () => void;
  onAskIngested: (title?: string) => void;
  onStartIngest: (url: string) => void;
  isMobile: boolean;
  isOpen: boolean;
  onClose: () => void;
}

export function Sidebar({
  groups,
  total,
  density,
  onPickVideo,
  onNewConversation,
  ingest,
  onCloseIngest,
  onAskIngested,
  onStartIngest,
  isMobile,
  isOpen,
  onClose,
}: SidebarProps) {
  const positionClasses = isMobile
    ? `fixed left-0 top-0 z-50 h-full w-[284px] transition-transform duration-300 ease-[cubic-bezier(.22,.61,.36,1)] ${isOpen ? "translate-x-0" : "-translate-x-full"}`
    : "relative h-full w-[312px] shrink-0";

  return (
    <aside className={`flex flex-col border-r border-line bg-ink-sidebar ${positionClasses}`}>
      <div className="px-[18px] pb-3.5 pt-5">
        <div className="flex items-center gap-2.5">
          <BrandMark size={21} className="flex text-accent" />
          <span className="text-[18px] font-bold tracking-[-0.01em]">ScienceQ</span>
          {isMobile && (
            <button
              onClick={onClose}
              aria-label="Close sidebar"
              className="ml-auto flex h-[30px] w-[30px] shrink-0 cursor-pointer items-center justify-center rounded-[8px] border border-line-strong bg-transparent text-[15px] text-mut-500 hover:bg-ink-panel"
            >
              ✕
            </button>
          )}
        </div>
        <p className="mt-2.5 text-[12.5px] leading-[1.5] text-mut-600">
          Ask questions across a curated library of science videos — grounded in the actual
          transcripts.
        </p>
        <button
          onClick={() => { onNewConversation(); if (isMobile) onClose(); }}
          className="mt-3.5 flex w-full cursor-pointer items-center justify-center gap-[7px] rounded-[9px] border border-line-strong bg-ink-panel py-2.5 text-[13px] font-medium text-mut-200 hover:bg-[#1a1f27]"
        >
          <span className="relative inline-block h-[13px] w-[13px] rounded border-[1.5px] border-mut-500" />
          New conversation
        </button>
      </div>

      <div className="flex items-center justify-between px-[18px] pb-1.5 pt-1">
        <span className="text-[10.5px] font-semibold tracking-[0.14em] text-mut-700">CORPUS</span>
        <span className="font-mono text-[10.5px] font-medium tracking-[0.04em] text-mut-700">
          {total} VIDEOS
        </span>
      </div>

      <CorpusBrowser
        groups={groups}
        density={density}
        onPickVideo={(t) => { onPickVideo(t); if (isMobile) onClose(); }}
        defaultOpen="Biology"
      />

      <div className="border-t border-line px-[18px] py-3 font-mono text-[11px] text-mut-800">
        LangChain · Groq · Pinecone
      </div>

      {ingest.active && (
        <IngestPanel state={ingest} onClose={onCloseIngest} onAsk={onAskIngested} onStart={onStartIngest} />
      )}
    </aside>
  );
}
