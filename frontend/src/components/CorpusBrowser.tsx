import { useState } from "react";
import type { SidebarDensity, TopicGroup } from "@/types";

interface CorpusBrowserProps {
  groups: TopicGroup[];
  density: SidebarDensity;
  onPickVideo: (title: string) => void;
  defaultOpen?: string; // topic expanded on first render
}

const toTopicId = (topic: string) =>
  "topic-" + topic.toLowerCase().replace(/\s+/g, "-").replace(/[^a-z0-9-]/g, "");

export function CorpusBrowser({ groups, density, onPickVideo, defaultOpen }: CorpusBrowserProps) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>(
    defaultOpen ? { [defaultOpen]: true } : {}
  );
  const detailed = density !== "Compact";

  return (
    <div className="flex-1 overflow-y-auto px-3 pb-4 pt-0.5">
      {groups.map((g) => {
        const isOpen = !!expanded[g.topic];
        const panelId = toTopicId(g.topic);
        return (
          <div key={g.topic} className="mb-0.5">
            <button
              onClick={() => setExpanded((e) => ({ ...e, [g.topic]: !e[g.topic] }))}
              aria-expanded={isOpen}
              aria-controls={panelId}
              className="flex w-full cursor-pointer items-center gap-2 rounded-lg border-0 bg-transparent px-2 py-2.5 text-left font-sans hover:bg-white/[0.035]"
            >
              <span className="w-3 text-[10px] text-accent">{isOpen ? "▾" : "▸"}</span>
              <span className="flex-1 text-[13.5px] font-medium text-mut-200">{g.topic}</span>
              <span className="font-mono text-[11px] text-mut-700">{g.videos.length}</span>
            </button>

            {isOpen && (
              <div id={panelId} className="ml-[5px] border-l border-line py-0.5 pb-2 pl-5">
                {g.videos.map((v) => (
                  <button
                    key={v.video_id}
                    onClick={() => onPickVideo(v.title)}
                    className="my-px block w-full cursor-pointer rounded-[7px] border-0 bg-transparent px-2 py-1.5 text-left font-sans hover:bg-white/[0.04]"
                  >
                    <span className="block truncate text-[12.5px] leading-[1.35] text-mut-300">
                      {v.title}
                    </span>
                    {detailed && (
                      <span className="mt-[3px] flex items-center gap-1.5">
                        <span className="text-[11px] text-mut-600">{v.channel}</span>
                        <span className="text-[10px] text-[#4d535b]">·</span>
                        <span className="font-mono text-[10.5px] text-mut-600">{v.duration}</span>
                        {v.source === "live" && (
                          <span className="rounded border border-accent-bd bg-accent-soft px-1 py-px text-[9px] font-semibold tracking-[0.06em] text-accent">
                            LIVE
                          </span>
                        )}
                      </span>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
