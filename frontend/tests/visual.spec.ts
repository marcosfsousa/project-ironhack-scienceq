import { test, expect } from "@playwright/test";
import path from "path";
import { CATALOG_FIXTURE } from "../src/data/fixtures";
import { ACCENTS } from "../src/data/accents";
import { SSE_FIXTURE } from "./sse-fixture";

// Issue #89: these four tests used to end in a `page.screenshot()` and assert
// nothing — they passed if the selectors resolved and the page did not throw,
// while the PNGs landed in test-results/ (which Playwright wipes between runs)
// and nothing ever compared them. The captures are kept as failure artefacts,
// but every test now pins something first.
//
// Assertions query the accessibility tree by role/name, matching the house
// style in privacy.spec.ts: substance is pinned, exact copy is not.

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

  await expect(page.getByRole("heading", { name: "ScienceQ", level: 1 })).toBeVisible();
  await expect(page.getByText(/ScienceQ is an AI assistant/i)).toBeVisible();
  await expect(page.getByPlaceholder(/ask about the videos/i)).toBeVisible();

  await page.screenshot({ path: shot(testInfo, "hero"), fullPage: true });
});

test("sidebar", async ({ page, isMobile }, testInfo) => {
  await page.goto("/");
  await page.waitForLoadState("networkidle");
  if (isMobile) {
    await page.click('[aria-label="Open sidebar"]');
    // Replaces a bare waitForTimeout: the close control is only reachable once
    // the panel has slid in and `inert` has lifted, so this waits on the state
    // that matters rather than on the transition duration.
    await expect(page.getByRole("button", { name: /close sidebar/i })).toBeVisible();
  }

  // Sidebar content is in the accessibility tree, not merely painted.
  await expect(page.getByRole("button", { name: /new conversation/i })).toBeVisible();

  await page.screenshot({ path: shot(testInfo, "sidebar"), fullPage: true });
});

test(
  "full answer — bubble + video + sources",
  { tag: "@smoke" },
  async ({ page }, testInfo) => {
    await page.goto("/");
    await page.waitForLoadState("networkidle");
    const textarea = page.getByPlaceholder(/ask about the videos/i);
    await textarea.fill("How does CRISPR work?");
    await textarea.press("Enter");

    // The answer carries its AI-generated badge — the #11/#18 disclosure
    // surface. This is the small "AI" chip beside the avatar, not the bubble.
    await expect(page.getByRole("note", { name: /ai-generated answer/i })).toBeVisible({
      timeout: 10000,
    });

    // The streamed tokens actually render. Asserted against page text rather
    // than the bubble element, which carries no role or label to address it by.
    await expect(page.getByText(/CRISPR is a revolutionary gene-editing tool/i)).toBeVisible();
    await expect(page.getByText(/molecular scissors/i)).toBeVisible();

    // The [SOURCES] frame is parsed into a real source row, not dropped.
    await expect(page.getByText("SOURCES", { exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: /expand source 1/i })).toBeVisible();

    // [META] is consumed by the parser, never rendered as answer text.
    await expect(page.getByText("[META]")).toHaveCount(0);

    await page.screenshot({ path: shot(testInfo, "answer"), fullPage: true });
  }
);

test("ingest panel", async ({ page }, testInfo) => {
  await page.goto("/");
  await page.waitForLoadState("networkidle");
  await page.click('[aria-label="Index a YouTube video"]');

  // No isMobile slide-in wait needed: the panel overlays the sidebar, so these
  // assertions auto-retry until it has arrived.
  //
  // Both matchers are narrowed to dodge a collision, and in both cases the
  // collision only exists on one viewport — which is why an unanchored version
  // passes mobile and fails desktop. `exact` because Hero's copy ("…to index a
  // video that isn't in the library yet") substring-matches the heading; `^`
  // because the desktop composer placeholder ("Ask about the videos, or paste a
  // YouTube URL to index it…") substring-matches the input.
  await expect(page.getByText("Index a video", { exact: true })).toBeVisible();
  await expect(page.getByPlaceholder(/^paste a youtube url/i)).toBeVisible();

  await page.screenshot({ path: shot(testInfo, "ingest"), fullPage: true });
});

// The accent radiogroup was the one React 19 semantic change in the tree and
// nothing touched it (#89). ChatView's ref callback deletes from `buttonRefs`
// on unmount; React 19 skips that null-call when a ref callback *returns* a
// cleanup function. The callback there has a block body returning undefined, so
// the delete branch still fires — but if it ever stopped,
// `buttonRefs.current.get(next)?.focus()` would silently no-op and every other
// check would stay green. These tests fail loudly in that case: they assert
// focus moved, not just that selection did.

const radioName = (a: string) => new RegExp(`^${a}$`, "i");

test("accent radiogroup: arrow keys move focus and selection together", async ({ page }) => {
  await page.goto("/");
  await page.waitForLoadState("networkidle");

  const group = page.getByRole("radiogroup", { name: /accent color/i });
  const radio = (a: string) => group.getByRole("radio", { name: radioName(a) });

  // Establish a known starting point rather than leaning on App.tsx's default.
  await radio(ACCENTS[0]).click();
  await expect(radio(ACCENTS[0])).toHaveAttribute("aria-checked", "true");
  await radio(ACCENTS[0]).focus();

  await page.keyboard.press("ArrowRight");
  await expect(radio(ACCENTS[1])).toBeFocused();
  await expect(radio(ACCENTS[1])).toHaveAttribute("aria-checked", "true");
  await expect(radio(ACCENTS[0])).toHaveAttribute("aria-checked", "false");

  // Roving tabIndex: only the selected radio is tabbable.
  await expect(radio(ACCENTS[1])).toHaveAttribute("tabindex", "0");
  await expect(radio(ACCENTS[0])).toHaveAttribute("tabindex", "-1");

  await page.keyboard.press("ArrowLeft");
  await expect(radio(ACCENTS[0])).toBeFocused();
  await expect(radio(ACCENTS[0])).toHaveAttribute("aria-checked", "true");
});

test("accent radiogroup: ArrowDown/ArrowUp mirror ArrowRight/ArrowLeft", async ({ page }) => {
  await page.goto("/");
  await page.waitForLoadState("networkidle");

  const group = page.getByRole("radiogroup", { name: /accent color/i });
  const radio = (a: string) => group.getByRole("radio", { name: radioName(a) });

  await radio(ACCENTS[0]).click();
  await radio(ACCENTS[0]).focus();

  await page.keyboard.press("ArrowDown");
  await expect(radio(ACCENTS[1])).toBeFocused();

  await page.keyboard.press("ArrowUp");
  await expect(radio(ACCENTS[0])).toBeFocused();
});

test("accent radiogroup: focus wraps at both ends", async ({ page }) => {
  await page.goto("/");
  await page.waitForLoadState("networkidle");

  const group = page.getByRole("radiogroup", { name: /accent color/i });
  const radio = (a: string) => group.getByRole("radio", { name: radioName(a) });
  const last = ACCENTS[ACCENTS.length - 1];

  // Backward off the first wraps to the last.
  await radio(ACCENTS[0]).click();
  await radio(ACCENTS[0]).focus();
  await page.keyboard.press("ArrowLeft");
  await expect(radio(last)).toBeFocused();
  await expect(radio(last)).toHaveAttribute("aria-checked", "true");

  // Forward off the last wraps back to the first.
  await page.keyboard.press("ArrowRight");
  await expect(radio(ACCENTS[0])).toBeFocused();
  await expect(radio(ACCENTS[0])).toHaveAttribute("aria-checked", "true");
});

test("accent radiogroup: a full cycle returns to the start", async ({ page }) => {
  await page.goto("/");
  await page.waitForLoadState("networkidle");

  const group = page.getByRole("radiogroup", { name: /accent color/i });
  const radio = (a: string) => group.getByRole("radio", { name: radioName(a) });

  await radio(ACCENTS[0]).click();
  await radio(ACCENTS[0]).focus();

  // Repeated presses must keep cycling — that only holds if focus follows
  // selection each step, which is exactly what the stale-ref failure breaks.
  for (let i = 1; i < ACCENTS.length; i++) {
    await page.keyboard.press("ArrowRight");
    await expect(radio(ACCENTS[i])).toBeFocused();
  }
  await page.keyboard.press("ArrowRight");
  await expect(radio(ACCENTS[0])).toBeFocused();
});
