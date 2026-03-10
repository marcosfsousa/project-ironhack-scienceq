"""
streamlit_app.py
----------------
Streamlit web UI for the YouTube QA Bot.

Layout:
  Sidebar  — mode info, corpus video browser, session controls
  Main     — chat interface with streaming output and source citations

Session state keys:
  agent             — YouTubeQAAgent singleton (persists across reruns)
  messages          — list of {"role": "user"|"assistant", "content": str, "sources": list}
  last_embed_source — top-scoring RAG source chunk dict for video embed (None if not RAG)

Run locally:
  streamlit run app/streamlit_app.py

Deploy:
  Push to GitHub → Streamlit Community Cloud → set env vars in dashboard
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Path setup ─────────────────────────────────────────────────────────────────
_ROOT         = Path(__file__).resolve().parent.parent
_AGENT_DIR    = _ROOT / "agent"
_PIPELINE_DIR = _ROOT / "pipeline"

for p in [str(_AGENT_DIR), str(_PIPELINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from agent import YouTubeQAAgent, _classify_intent_fast  # noqa: E402
from tools import _KNOWN_TOPICS  # noqa: E402

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ScienceQ",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Constrain main content width on large screens */
  .block-container {
    max-width: 1368px;
    padding-left: 2rem;
    padding-right: 2rem;
  }
  /* Constrain chat input to match content width */
  [data-testid="stBottomBlockContainer"] {
    max-width: 1368px;
    margin: 0 auto;
    padding-left: 2rem;
    padding-right: 2rem;
  }
  /* Tighten chat bubbles */
  .stChatMessage { padding: 0.5rem 0; }
  /* Source pill styling */
  .source-pill {
    display: inline-block;
    background: #1e3a5f;
    color: #a8d8f0;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.72rem;
    margin: 2px 3px;
    text-decoration: none;
    max-width: 200px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    vertical-align: middle;
  }
  .source-pill:hover { background: #2a5080; }
  /* Sidebar section headers */
  .sidebar-section {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888;
    margin: 1rem 0 0.3rem;
  }
  /* Sidebar video links — classes used so media query can override */
  .sidebar-video-title { color: #a8d8f0; text-decoration: none; display: block; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .sidebar-video-channel { color: #888; }
  /* Starter question label */
  .starter-label {
    font-size: 0.85rem;
    color: #888;
    text-align: center;
    margin-bottom: 0.5rem;
  }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ───────────────────────────────────────────────

def _init_session() -> None:
    if "agent" not in st.session_state:
        st.session_state.agent             = YouTubeQAAgent()
        st.session_state.messages          = []   # list of dicts: role / content / sources
        st.session_state.last_embed_source = None  # top RAG chunk for video embed


# ── Metadata loader (corpus video catalog) ────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_corpus_metadata() -> list[dict]:
    """Load data/metadata.json and return a flat list of video dicts."""
    meta_path = _ROOT / "data" / "metadata.json"
    if not meta_path.exists():
        return []
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        videos = raw.get("videos", raw) if isinstance(raw, dict) else raw
        if isinstance(videos, dict):
            return list(videos.values())
        return videos
    except Exception:
        return []
    
# -- Rewrite the Youtube URLs (safeguard) ---------------------------------------

def _safe_yt_url(video_id: str, start: int = 0) -> str:
    """Construct YouTube URL from video_id rather than trusting stored URLs."""
    vid_id_clean = re.sub(r"[^A-Za-z0-9_-]", "", video_id)[:11]
    return f"https://www.youtube.com/watch?v={vid_id_clean}&t={start}"

# ── Source rendering ───────────────────────────────────────────────────────────

def _render_sources(sources: list[dict]) -> None:
    """Render source citations as clickable pills under an assistant message."""
    if not sources:
        return
    pills_html = ""
    seen = set()
    for s in sources:
        vid_id = s.get("video_id", "")
        start  = int(s.get("start", 0))
        title  = html.escape(s.get("title", vid_id)[:28])
        key    = f"{vid_id}_{start}"
        if key in seen:
            continue
        seen.add(key)
        ts_label = f"{start // 60}:{start % 60:02d}"
        url = _safe_yt_url(vid_id, start)
        pills_html += (
            f'<a class="source-pill" href="{url}" target="_blank">'
            f"▶ {title} @ {ts_label}"
            f"</a>"
        )
    if pills_html:
        st.markdown(
            f'<div style="margin-top:4px">{pills_html}</div>',
            unsafe_allow_html=True,
        )


# ── Video embed ───────────────────────────────────────────────────────────────

def _render_video_embed(sources: list[dict]) -> None:
    """
    Render a st.video() embed for the top-scoring source chunk.

    Displayed outside the chat bubble as a separate block, inside a collapsed
    expander so it doesn't dominate the layout by default.

    Called from main() using st.session_state["last_embed_source"], so it
    survives st.rerun() and persists until the next RAG answer or reset.
    Never called from _render_history().

    Source dict shape expected (streaming path from agent.last_sources):
        title, video_id, start, end, chunk_text, channel
    The blocking path (source_chunks_for_display) lacks raw video_id + start
    integers and is never used for RAG in the Streamlit UI, so it is not handled.

    Top chunk selection:
        Pinecone returns results sorted by score descending, so sources[0] is
        already the highest scorer. A defensive fallback to max(score) is used
        in case order is ever disrupted (e.g. multi-namespace merge reordering).
    """
    if not sources:
        return

    # Defensive top-chunk selection: trust index 0 (Pinecone score-sorted) but
    # fall back to explicit max if score field is present.
    if all("score" in s for s in sources):
        top = max(sources, key=lambda s: s.get("score", 0))
    else:
        top = sources[0]

    video_id = re.sub(r"[^A-Za-z0-9_-]", "", str(top.get("video_id", "")))[:11]
    start    = int(top.get("start", 0))
    title    = top.get("title", video_id)[:50]

    if not video_id:
        return  # no valid video_id — skip silently

    ts_label = f"{start // 60}:{start % 60:02d}"

    with st.expander(f"▶ Watch: {title} @ {ts_label}", expanded=False):
        st.video(
            f"https://www.youtube.com/watch?v={video_id}",
            start_time=start,
        )

# ── Sidebar ────────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.title("🎬 ScienceQ")
        st.caption("Ask questions about science videos from Veritasium, Kurzgesagt, and Big Think.")

        # ── Session controls (top) ─────────────────────────────────────────────
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.agent.reset()
            st.session_state.messages          = []
            st.session_state.last_embed_source = None
            st.rerun()

        # ── Corpus browser ─────────────────────────────────────────────────────
        videos = _load_corpus_metadata()
        video_count = len(videos) if videos else 0
        st.markdown(f'<div class="sidebar-section">Corpus — {video_count} videos</div>', unsafe_allow_html=True)
        if videos:
            # Group by topic, sorted alphabetically
            topics_map: dict = {}
            for v in videos:
                topic = v.get("topic", "Other")
                topics_map.setdefault(topic, []).append(v)

            for topic in sorted(topics_map.keys()):
                topic_videos = sorted(topics_map[topic], key=lambda x: x.get("title", ""))
                with st.expander(f"{topic} ({len(topic_videos)})", expanded=False):
                    for v in topic_videos:
                        title    = html.escape(v.get("title", v.get("video_id", "Unknown"))[:60])
                        channel  = html.escape(v.get("channel", ""))
                        duration = v.get("duration", "")
                        vid_id   = v.get("video_id", "")
                        yt_url   = _safe_yt_url(vid_id) if vid_id else "#"
                        # Duration badge: append to channel line if available
                        meta = f"{channel} · {duration}" if duration else channel
                        st.markdown(
                            f'<a href="{yt_url}" target="_blank" class="sidebar-video-title" style="font-size:0.8rem;">▶ {title}</a>'
                            f'<div class="sidebar-video-channel" style="font-size:0.7rem; margin-bottom:8px">{meta}</div>',
                            unsafe_allow_html=True,
                        )
        else:
            st.caption("metadata.json not found — run bootstrap_metadata.py first.")

        # ── Footer ────────────────────────────────────────────────────────────
        st.markdown("---")
        st.caption("Built with LangChain · Groq · Pinecone · Streamlit")


# ── Chat history rendering ─────────────────────────────────────────────────────

def _render_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            content = msg["content"]
            if msg["role"] == "assistant" and content.startswith("METADATA_LIST:"):
                try:
                    videos = json.loads(content[len("METADATA_LIST:"):])
                    _render_metadata_list(videos, msg.get("query", ""))
                except Exception:
                    st.markdown(content)
            else:
                st.markdown(content)
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])


# ── Suggested starter questions ────────────────────────────────────────────────

def _render_starters() -> None:
    if st.session_state.messages:
        return  # only show on empty chat

    count = len(_load_corpus_metadata())
    st.markdown(
        f'<div class="intro-text" style="text-align: justify; font-size: 0.95rem; margin-bottom: 1rem;">'
        f"Ask questions about science and ideas from <strong>{count} curated YouTube videos</strong> — "
        "Veritasium, Kurzgesagt, 3Blue1Brown, PBS Space Time, and more. \n\n"
        "\n\nAnswers are grounded in the actual transcripts, with source timestamps "
        "so you can jump straight to the relevant moment.\n"
        "\nYou can also paste any YouTube URL into the chat to ask questions about "
        "a video not in the library."
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="starter-label">What would you like to explore?</div>', unsafe_allow_html=True)
    _, col, _ = st.columns([1, 3, 1])
    starters = [
        "Why has no one measured the speed of light?",
        "How does natural selection actually work?",
        "What happens at the edge of the observable universe?",
        "What videos do you have on mathematics?",
    ]
    for i, q in enumerate(starters):
        if col.button(q, use_container_width=True, key=f"starter_{i}"):
            _handle_user_input(q)
            st.rerun()


# ── Metadata list renderer ─────────────────────────────────────────────────────

def _render_metadata_list(videos: list[dict], query: str) -> None:
    """
    Render a clean video list from structured metadata, bypassing LLM formatting.
    Called when agent returns a METADATA_LIST:<json> signal.
    """
    count = len(videos)
    # Infer display label from query
    q = query.lower()
    topic_words = [w for w in _KNOWN_TOPICS if w in q]
    label = topic_words[0].capitalize() if topic_words else "your query"

    st.markdown(f"Here are the **{count} {label} video{'s' if count != 1 else ''}** in the library:")

    for v in videos:
        title    = v.get("title", "Unknown")
        channel  = v.get("channel", "")
        topic    = v.get("topic", "")
        duration = v.get("duration", "")
        url      = v.get("url", "")

        meta_parts = [p for p in [channel, topic, duration] if p]
        meta_str   = " · ".join(meta_parts)

        if url:
            st.markdown(f"- [**{title}**]({url})  \n  <span class='metadata-channel' style='font-size:0.8rem'>{meta_str}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"- **{title}**  \n  <span class='metadata-channel' style='font-size:0.8rem'>{meta_str}</span>", unsafe_allow_html=True)


# ── Input handler ──────────────────────────────────────────────────────────────

def _handle_user_input(user_input: str) -> None:
    """
    Append user message, stream assistant response, append assistant message.
    Called both from chat_input and starter buttons.
    """
    # Append user message to history
    st.session_state.messages.append({
        "role":    "user",
        "content": user_input,
        "sources": [],
    })

    # Display it immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream the assistant response
    with st.chat_message("assistant"):
        agent: YouTubeQAAgent = st.session_state.agent

        # Use blocking chat() for ingest + metadata (no streaming needed)
        # Use stream_chat() for RAG (token-by-token)
        intent = _classify_intent_fast(user_input)

        if intent == "rag":
            # Streaming path
            placeholder  = st.empty()
            full_answer  = ""
            with st.spinner("Searching..."):
                try:
                    for token in agent.stream_chat(user_input):
                        full_answer += token
                        for char in token:
                            placeholder.markdown(full_answer + "▌")
                # Handling Error 429 gracefully and avoiding streaming error messages in chat
                except Exception as e:
                    if "rate_limit_exceeded" in str(e) or "429" in str(e):
                        full_answer = (
                            "The service is currently at capacity. "
                            "Please try again in a few minutes."
                        )
                    else:
                        resp        = agent.chat(user_input)
                        full_answer = resp.answer
                    placeholder.markdown(full_answer)
            sources = agent.last_sources
        else:
            # Blocking path (ingest / metadata)
            with st.spinner(
                "Ingesting video..." if intent == "ingest" else "Looking up..."
            ):
                resp = agent.chat(user_input)
            full_answer = resp.answer

            # Metadata tool returns "METADATA_LIST:<json>" — render directly
            # without passing through the LLM to avoid inconsistent reformatting.
            if full_answer.startswith("METADATA_LIST:"):
                try:
                    videos = json.loads(full_answer[len("METADATA_LIST:"):])
                    _render_metadata_list(videos, user_input)
                    # Keep the signal in full_answer — _render_history() will
                    # detect it on rerun and re-render the list correctly.
                except Exception:
                    st.markdown(full_answer)
            else:
                st.markdown(full_answer)
            sources = resp.sources

        _render_sources(sources)

    # Store top source chunk for video embed — rendered in main() to survive rerun
    if intent == "rag" and sources:
        st.session_state["last_embed_source"] = sources[0]
    elif intent != "rag":
        st.session_state.pop("last_embed_source", None)

    # Persist assistant message
    st.session_state.messages.append({
        "role":    "assistant",
        "content": full_answer,
        "sources": sources,
        "query":   user_input,   # preserved for METADATA_LIST re-render in _render_history
    })


# ── Main layout ────────────────────────────────────────────────────────────────

def main() -> None:
    _init_session()
    _render_sidebar()

    st.header("🎬 ScienceQ", divider="blue")

    # chat_input must be declared at top level — column nesting is unreliable
    # across Streamlit versions. The value is captured here and processed inside
    # chat_col below, which is safe since it's just a string.
    user_input = st.chat_input(
        placeholder=(
            "Ask a question about the existing videos, or simply paste a YouTube URL and ask a question about it..."
        )
    )

    # 2-column layout: chat left (60%), embed right (40%).
    # Right column only rendered when a RAG source exists — avoids blank column
    # on first load and for metadata/ingest responses.
    has_embed = bool(st.session_state.get("last_embed_source"))

    if has_embed:
        chat_col, embed_col = st.columns([3, 2])
    else:
        chat_col = st.container()

    with chat_col:
        _render_history()
        _render_starters()
        if user_input and user_input.strip():
            _handle_user_input(user_input.strip())
            st.rerun()

    if has_embed:
        with embed_col:
            _render_video_embed([st.session_state["last_embed_source"]])


if __name__ == "__main__":
    main()
