import { test } from "@playwright/test";
import path from "path";

const CATALOG_FIXTURE = [
  {
    video_id: "jAhjPd4uNFY",
    title: "Genetic Engineering Will Change Everything Forever – CRISPR",
    channel: "Kurzgesagt",
    topic: "Biology",
    duration: "16:04",
    url: "https://www.youtube.com/watch?v=jAhjPd4uNFY",
    source: "corpus",
  },
  {
    video_id: "MRjxi-qkPkQ",
    title: "Why Quantum Mechanics is Weird",
    channel: "Veritasium",
    topic: "Physics",
    duration: "20:12",
    url: "https://www.youtube.com/watch?v=MRjxi-qkPkQ",
    source: "corpus",
  },
  {
    video_id: "La_Teoria",
    title: "La Teoría Clásica y la Revolución de la Física Moderna",
    channel: "QuantumFracture",
    topic: "Other",
    duration: "18:30",
    url: "https://www.youtube.com/watch?v=La_Teoria",
    source: "live",
  },
];

const SSE_FIXTURE = [
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

test.beforeEach(async ({ page }) => {
  await page.route("/api/catalog", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(CATALOG_FIXTURE) })
  );
  await page.route("/api/chat/stream", (route) =>
    route.fulfill({ status: 200, contentType: "text/event-stream", body: SSE_FIXTURE })
  );
});

function shot(testInfo: { project: { name: string } }, name: string) {
  return path.join("test-results", `${name}-${testInfo.project.name}.png`);
}

test("hero — empty state", async ({ page }, testInfo) => {
  await page.goto("/");
  await page.waitForLoadState("networkidle");
  await page.screenshot({ path: shot(testInfo, "hero"), fullPage: true });
});

test("sidebar", async ({ page, isMobile }, testInfo) => {
  await page.goto("/");
  await page.waitForLoadState("networkidle");
  if (isMobile) {
    await page.click('[aria-label="Open sidebar"]');
    await page.waitForTimeout(350); // slide-in transition
  }
  await page.screenshot({ path: shot(testInfo, "sidebar"), fullPage: true });
});

test("full answer — bubble + video + sources", async ({ page, isMobile }, testInfo) => {
  await page.goto("/");
  await page.waitForLoadState("networkidle");
  const textarea = page.getByPlaceholder(/ask about the videos/i);
  await textarea.fill("How does CRISPR work?");
  await textarea.press("Enter");
  await page.waitForSelector("text=SOURCES", { timeout: 10000 });
  await page.waitForTimeout(300); // let fadeUp animation settle
  await page.screenshot({ path: shot(testInfo, "answer"), fullPage: true });
});

test("ingest panel", async ({ page, isMobile }, testInfo) => {
  await page.goto("/");
  await page.waitForLoadState("networkidle");
  await page.click('[aria-label="Index a YouTube video"]');
  if (isMobile) {
    await page.waitForTimeout(350); // sidebar slide-in
  }
  await page.waitForSelector("text=Index a video", { timeout: 5000 });
  await page.screenshot({ path: shot(testInfo, "ingest"), fullPage: true });
});
