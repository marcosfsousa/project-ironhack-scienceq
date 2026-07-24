// Shared SSE chat-stream fixture for Playwright tests, matching the
// /api/chat/stream wire format (data: tokens, [SOURCES] payload, [DONE]).
export const SSE_FIXTURE = [
  "data: CRISPR is a revolutionary gene-editing tool that allows scientists to make precise changes to DNA.\n\n",
  "data: The technology works by using a protein called Cas9 as molecular scissors, guided by a short RNA sequence to the exact location in the genome that needs to be cut.\n\n",
  "data: [SOURCES]" +
    JSON.stringify([
      {
        title: "Genetic Engineering Will Change Everything Forever – CRISPR",
        timestamp: "3:11",
        link: "https://www.youtube.com/watch?v=jAhjPd4uNFY&t=191",
        score: 0.95,
        rerank_score: 0.95,
        text: "CRISPR stands for Clustered Regularly Interspaced Short Palindromic Repeats.",
      },
    ]) +
    "\n\n",
  "data: [DONE]\n\n",
].join("");
