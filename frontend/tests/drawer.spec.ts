import { test, expect } from "@playwright/test";
import { CATALOG_FIXTURE } from "../src/data/fixtures";

// Pins the mobile drawer's open/closed geometry (issue #76).
//
// Why it exists: the closed state is a CSS offset, not `display: none`, so
// "closed" means "translated off-screen" while still `fixed left-0 z-50`. If the
// offset declaration is ever dropped, the panel stays at left:0 covering the app
// while `inert` and `aria-hidden` make it unclickable. That shipped once during
// the v4 port -- v4 emits translate-x-* as the individual `translate` property,
// which is above the pinned `build.target` -- so Sidebar.tsx now writes the
// offset as a `transform:` value instead.
//
// Scope, so this test is not mistaken for the guard it is not: this runs on
// modern Chromium, where both spellings work. It therefore cannot catch the
// browser-floor bug itself -- scripts/check-css-floor.mjs does that, against the
// built CSS. What this catches is the refactor regressing the drawer outright,
// or a future change reintroducing a `display`-based toggle that breaks the
// transition. The two are complementary: one asserts the CSS the floor allows,
// this one asserts the geometry the user gets.

test.beforeEach(async ({ page }) => {
  await page.route("/api/catalog", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(CATALOG_FIXTURE) })
  );
});

test("mobile drawer sits off-screen when closed and flush left when open", async ({ page, isMobile }) => {
  test.skip(!isMobile, "the drawer only exists on the mobile layout");

  await page.goto("/");
  await page.waitForLoadState("networkidle");

  const drawer = page.locator("aside");
  const width = (await drawer.boundingBox())!.width;
  expect(width).toBeGreaterThan(0);

  // Closed: fully off the left edge. Asserted as a real offset rather than via
  // visibility, because the failure mode being guarded is a *visible* panel.
  const closed = (await drawer.boundingBox())!;
  expect(closed.x + closed.width).toBeLessThanOrEqual(1);
  await expect(drawer).toHaveAttribute("aria-hidden", "true");

  await page.click('[aria-label="Open sidebar"]');
  await expect(drawer).not.toHaveAttribute("aria-hidden", "true");
  await page.waitForTimeout(400); // the 300ms slide, plus a margin

  const open = (await drawer.boundingBox())!;
  expect(Math.abs(open.x)).toBeLessThanOrEqual(1);

  await page.click('[aria-label="Close sidebar"]');
  await page.waitForTimeout(400);

  const reclosed = (await drawer.boundingBox())!;
  expect(reclosed.x + reclosed.width).toBeLessThanOrEqual(1);
});
