interface BrandMarkProps {
  size?: number;
  className?: string;
}

/** ScienceQ atom mark. Inherits color from `currentColor` (use text-accent). */
export function BrandMark({ size = 21, className }: BrandMarkProps) {
  return (
    <span className={className ?? "flex text-accent"}>
      <svg viewBox="0 0 28 28" width={size} height={size} fill="none" className="block">
        <circle cx="14" cy="14" r="2.7" fill="currentColor" />
        <ellipse
          cx="14"
          cy="14"
          rx="11.5"
          ry="4.6"
          stroke="currentColor"
          strokeWidth="1.8"
          transform="rotate(30 14 14)"
        />
        <ellipse
          cx="14"
          cy="14"
          rx="11.5"
          ry="4.6"
          stroke="currentColor"
          strokeWidth="1.8"
          transform="rotate(-30 14 14)"
        />
      </svg>
    </span>
  );
}
