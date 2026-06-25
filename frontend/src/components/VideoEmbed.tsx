import { embedUrl } from "@/lib/format";

interface VideoEmbedProps {
  title: string;
  channel?: string;
  link: string; // YouTube URL (carries the timestamp)
  timestamp?: string;
}

/** Inline live player that starts at the cited moment. */
export function VideoEmbed({ title, channel, link, timestamp }: VideoEmbedProps) {
  return (
    <div className="mt-[18px] max-w-[520px] overflow-hidden rounded-[13px] border border-line-strong bg-ink-card">
      <div className="relative aspect-video bg-black">
        <iframe
          src={embedUrl(link)}
          title={title}
          loading="lazy"
          allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
          allowFullScreen
          className="absolute inset-0 h-full w-full border-0"
        />
      </div>
      <div className="flex items-center justify-between px-[13px] py-[11px]">
        <div className="min-w-0">
          <div className="truncate text-[13px] font-semibold text-mut-100">{title}</div>
          {channel && <div className="mt-0.5 text-[11.5px] text-mut-500">{channel}</div>}
        </div>
        <a
          href={link}
          target="_blank"
          rel="noopener noreferrer"
          className="ml-3 shrink-0 whitespace-nowrap text-[12px] font-medium text-accent no-underline"
        >
          Watch on YouTube ↗
          {timestamp ? ` · ${timestamp}` : ""}
        </a>
      </div>
    </div>
  );
}
