/** Convert "mm:ss" or "h:mm:ss" to seconds. Returns 0 for non-timestamps. */
export function tsToSec(ts?: string): number {
  if (!ts || !/\d/.test(ts)) return 0;
  const p = ts.split(":").map((x) => parseInt(x, 10));
  if (p.some((n) => Number.isNaN(n))) return 0;
  return p.length === 3 ? p[0] * 3600 + p[1] * 60 + p[2] : p[0] * 60 + p[1];
}

/** Pull the 11-char video id out of any YouTube URL form. */
export function youTubeId(url: string): string | null {
  const m =
    url.match(/[?&]v=([A-Za-z0-9_-]{11})/) ||
    url.match(/youtu\.be\/([A-Za-z0-9_-]{11})/) ||
    url.match(/embed\/([A-Za-z0-9_-]{11})/);
  return m ? m[1] : null;
}

/** Seconds encoded in a YouTube link (&t=90s / ?t=90 / #t=1m30s). */
export function youTubeStart(url: string): number {
  const m = url.match(/[?&#]t=([0-9hms]+)/i);
  if (!m) return 0;
  const v = m[1];
  if (/^\d+s?$/.test(v)) return parseInt(v, 10);
  let s = 0;
  const h = v.match(/(\d+)h/),
    mn = v.match(/(\d+)m/),
    sc = v.match(/(\d+)s/);
  if (h) s += +h[1] * 3600;
  if (mn) s += +mn[1] * 60;
  if (sc) s += +sc[1];
  return s;
}

/** Build a privacy-enhanced embed URL that starts at the cited moment. */
export function embedUrl(
  link: string,
  opts: { autoplay?: boolean; startSec?: number } = {}
): string {
  const id = youTubeId(link) ?? "";
  const start = opts.startSec ?? youTubeStart(link);
  const params = new URLSearchParams({
    start: String(start),
    rel: "0",
    modestbranding: "1",
  });
  if (opts.autoplay) params.set("autoplay", "1");
  return `https://www.youtube-nocookie.com/embed/${id}?${params.toString()}`;
}

/** True if the user typed a YouTube URL (route to ingest, not chat). */
export function isYouTubeUrl(s: string): boolean {
  return /(youtube\.com\/watch|youtu\.be\/|youtube\.com\/shorts)/i.test(s.trim());
}

export const uid = (p = "m") =>
  p + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
