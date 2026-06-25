import { useEffect, useMemo, useState } from "react";
import type { CatalogVideo, TopicGroup } from "@/types";
import { CATALOG_FIXTURE, TOPIC_ORDER } from "@/data/fixtures";

/**
 * Loads GET /api/catalog and groups it by topic for the sidebar.
 * Falls back to the bundled fixture so the UI renders without a backend.
 * `extra` lets the app inject videos that were just ingested this session.
 */
export function useCatalog(extra: CatalogVideo[] = []) {
  const [videos, setVideos] = useState<CatalogVideo[]>(CATALOG_FIXTURE);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const res = await fetch("/api/catalog");
        if (!res.ok) throw new Error(String(res.status));
        const data = (await res.json()) as CatalogVideo[];
        if (alive && Array.isArray(data) && data.length) setVideos(data);
      } catch {
        // keep fixture
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const groups = useMemo<TopicGroup[]>(() => {
    // extra takes precedence; deduplicate by video_id so the same video
    // doesn't appear in both extra (just-ingested) and videos (catalog fetch).
    const seen = new Set<string>();
    const all: CatalogVideo[] = [];
    for (const v of [...extra, ...videos]) {
      if (!seen.has(v.video_id)) {
        seen.add(v.video_id);
        all.push(v);
      }
    }
    const byTopic = new Map<string, CatalogVideo[]>();
    for (const v of all) {
      if (!byTopic.has(v.topic)) byTopic.set(v.topic, []);
      byTopic.get(v.topic)!.push(v);
    }
    const ordered = [...byTopic.keys()].sort((a, b) => {
      const ia = TOPIC_ORDER.indexOf(a);
      const ib = TOPIC_ORDER.indexOf(b);
      return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
    });
    return ordered.map((topic) => ({ topic, videos: byTopic.get(topic)! }));
  }, [videos, extra]);

  const total = useMemo(() => {
    const ids = new Set([...extra.map((v) => v.video_id), ...videos.map((v) => v.video_id)]);
    return ids.size;
  }, [videos, extra]);
  return { groups, total, loading };
}
