import { useEffect, useRef } from "react";
import { PROCESSORS, PRIVACY_NOTICE } from "@/data/privacy";

interface PrivacyNoticeProps {
  open: boolean;
  onClose: () => void;
}

/**
 * Privacy notice dialog (issue #17). A native <dialog> gives us the accessible
 * dialog role, focus trapping and Escape-to-close for free; we drive it from
 * React `open` state via showModal()/close(). The ::backdrop is styled in
 * index.css. Copy and vendor names come from src/data/privacy.ts.
 */
export function PrivacyNotice({ open, onClose }: PrivacyNoticeProps) {
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (open && !el.open) el.showModal();
    else if (!open && el.open) el.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      aria-labelledby="privacy-title"
      onClose={onClose}
      onClick={(e) => {
        // The modal dialog fills the viewport behind its content, so a click
        // landing on the element itself (not its inner panel) is a backdrop hit.
        if (e.target === ref.current) onClose();
      }}
      className="m-auto w-[calc(100vw-32px)] max-w-[520px] rounded-[14px] border border-line bg-ink-panel p-0 text-mut-200"
    >
      <div className="p-6">
        <div className="mb-4 flex items-start justify-between gap-4">
          <h2 id="privacy-title" className="m-0 text-[18px] font-bold text-mut-100">
            {PRIVACY_NOTICE.title}
          </h2>
          <button
            onClick={onClose}
            aria-label="Close"
            className="flex h-7 w-7 shrink-0 cursor-pointer items-center justify-center rounded-[8px] border border-line-strong bg-transparent text-[14px] text-mut-500 hover:bg-ink-hover hover:text-mut-200"
          >
            ✕
          </button>
        </div>

        <p className="mb-4 text-[14px] leading-[1.6]">{PRIVACY_NOTICE.intro}</p>

        <ul className="mb-4 flex list-none flex-col gap-2.5 p-0">
          {PROCESSORS.map((p) => (
            <li
              key={p.vendor}
              className="rounded-[10px] border border-line bg-ink-raised px-3.5 py-3 text-[13.5px] leading-[1.5]"
            >
              <div className="font-semibold text-mut-100">
                {p.role} · {p.vendor}
              </div>
              <div className="text-mut-400">Receives {p.receives}.</div>
            </li>
          ))}
        </ul>

        <p className="mb-5 text-[13.5px] leading-[1.6] text-mut-400">{PRIVACY_NOTICE.contextNote}</p>

        <div className="mb-[11px] text-[11px] font-semibold tracking-[0.12em] text-mut-700">
          WHAT WE DON'T COLLECT
        </div>
        <ul className="mb-5 flex list-none flex-col gap-1.5 p-0 text-[13.5px] leading-[1.5]">
          {PRIVACY_NOTICE.notCollected.map((line) => (
            <li key={line} className="flex gap-2">
              <span aria-hidden="true" className="text-accent">
                ✓
              </span>
              <span>{line}</span>
            </li>
          ))}
        </ul>

        <p className="text-[12.5px] leading-[1.55] text-mut-500">{PRIVACY_NOTICE.region}</p>
      </div>
    </dialog>
  );
}
