import { useCallback, useEffect, useState } from "react";
import type { Accent, CatalogVideo, IngestStatus, SidebarDensity } from "@/types";
import { useChat } from "@/hooks/useChat";
import { useIngest } from "@/hooks/useIngest";
import { useCatalog } from "@/hooks/useCatalog";
import { Sidebar } from "@/components/Sidebar";
import { ChatView } from "@/components/ChatView";
import { SUGGESTIONS } from "@/data/fixtures";
import { uid } from "@/lib/format";

export default function App() {
  const [accent, setAccent] = useState<Accent>("Indigo");
  const [density] = useState<SidebarDensity>("Detailed");
  const [extra, setExtra] = useState<CatalogVideo[]>([]);

  // Apply the accent at the document root so the CSS variables cascade.
  useEffect(() => {
    document.documentElement.dataset.accent = accent;
  }, [accent]);

  const chat = useChat();

  const onIngestComplete = useCallback((result: IngestStatus) => {
    if (!result.title || result.already_indexed) return;
    setExtra((e) => [
      {
        video_id: uid("live"),
        title: result.title!,
        channel: result.channel ?? "",
        topic: result.topic ?? "Technology",
        duration: "",
        url: "",
        source: "live",
      },
      ...e,
    ]);
  }, []);

  const ingest = useIngest(onIngestComplete);
  const catalog = useCatalog(extra);

  const handleIngest = (url: string) => { if (url) ingest.start(url); else ingest.open(); };

  const handleAskIngested = (title?: string) => {
    ingest.close();
    chat.send(
      title ? `What is the key takeaway from “${title}”?` : "Summarize the video I just added."
    );
  };

  return (
    <div className="flex h-screen overflow-hidden bg-ink text-mut-100">
      <Sidebar
        groups={catalog.groups}
        total={catalog.total}
        density={density}
        onPickVideo={(t) => chat.send(`Tell me about “${t}”`)}
        onNewConversation={chat.reset}
        ingest={ingest.ingest}
        onCloseIngest={ingest.close}
        onAskIngested={handleAskIngested}
        onStartIngest={ingest.start}
      />
      <ChatView
        messages={chat.messages}
        suggestions={SUGGESTIONS}
        accent={accent}
        onAccentChange={setAccent}
        onSend={chat.send}
        onIngest={handleIngest}
      />
    </div>
  );
}
