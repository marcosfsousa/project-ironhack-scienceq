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
    const all = [...extra, ...videos];
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

  const total = videos.length + extra.length;
  return { groups, total, loading };
}
