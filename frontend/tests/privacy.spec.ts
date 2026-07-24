import { test, expect } from "@playwright/test";
import { CATALOG_FIXTURE } from "../src/data/fixtures";
import { SSE_FIXTURE } from "./sse-fixture";

// Functional tests for the privacy notice (issue #17). Backend is mocked at
// the network boundary, mirroring disclosure.spec.ts / visual.spec.ts.
// Assertions query visible text and the accessibility tree by role/name —
// never class names or DOM structure. Substance (the notice is present,
// reachable, accessible, and names the actual processors) is pinned; exact
// phrasing is not, so copy can be tuned without breaking these tests.

test.beforeEach(async ({ page }) => {
  await page.route("/api/catalog", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(CATALOG_FIXTURE) })
  );
});

test("landing state offers a privacy-notice affordance before first interaction", async ({ page }) => {
  await page.goto("/");
  await page.waitForLoadState("networkidle");

  // A control that opens the notice sits on the landing state, alongside the
  // AI-nature disclosure, before any interaction.
  await expect(page.getByRole("button", { name: /how your questions are handled/i })).toBeVisible();
});

test("privacy notice is reachable, names the real processors, and is in the accessibility tree", async ({ page }) => {
  await page.goto("/");
  await page.waitForLoadState("networkidle");

  await page.getByRole("button", { name: /how your questions are handled/i }).click();

  // The notice surfaces as a dialog exposed to assistive tech by role + name.
  const dialog = page.getByRole("dialog", { name: /how your questions are handled/i });
  await expect(dialog).toBeVisible();

  // It names the actual third-party processors (not boilerplate), the
  // categories of data sent to them, and what is not collected.
  await expect(dialog.getByText(/Groq/)).toBeVisible();
  await expect(dialog.getByText(/Cohere/)).toBeVisible();
  await expect(dialog.getByText(/Receives your question text/i)).toBeVisible();
  await expect(dialog.getByText(/no server-side history/i)).toBeVisible();

  // Reachable at any point without cost: it closes and leaves the session intact.
  await page.getByRole("button", { name: /close/i }).click();
  await expect(dialog).not.toBeVisible();
});

test("privacy notice stays reachable during a conversation", async ({ page }) => {
  await page.route("/api/chat/stream", (route) =>
    route.fulfill({ status: 200, contentType: "text/event-stream", body: SSE_FIXTURE })
  );
  await page.goto("/");
  await page.waitForLoadState("networkidle");

  const textarea = page.getByPlaceholder(/ask about the videos/i);
  await textarea.fill("How does CRISPR work?");
  await textarea.press("Enter");
  await page.waitForSelector("text=SOURCES", { timeout: 10000 });

  // The persistent (footer-level) affordance survives leaving the landing
  // state, so consulting the notice mid-session costs nothing.
  const persistent = page.getByRole("button", { name: /privacy/i });
  await expect(persistent).toBeVisible();
  await persistent.click();
  await expect(page.getByRole("dialog", { name: /how your questions are handled/i })).toBeVisible();
});
