import { test, expect, type Page } from "@playwright/test";
import { CATALOG_FIXTURE } from "../src/data/fixtures";

// Issue #93: the composer's send button renders past the row's right edge on
// mobile and gets sliced. App.tsx sets `overflow-hidden` on the root, so the
// overflow never becomes a page-level scroll — `documentElement.scrollWidth >
// clientWidth` stays false throughout, which is why the obvious assertion
// passes while the button is visibly clipped. The only check that can see it
// compares the button's border box against the row's *content* box.

const WIDTHS = [320, 360, 365, 375, 390, 412];

interface Metrics {
  btnRight: number;
  btnWidth: number;
  btnHeight: number;
  contentRight: number;
  past: number;
  textareaMinWidth: string;
}

test.beforeEach(async ({ page }) => {
  await page.route("/api/catalog", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(CATALOG_FIXTURE),
    })
  );
});

function measure(page: Page): Promise<Metrics> {
  return page.evaluate(() => {
    const btn = document.querySelector<HTMLElement>('[aria-label="Send message"]');
    if (!btn || !btn.parentElement) throw new Error("send button not found");
    const row = btn.parentElement;
    const textarea = row.querySelector("textarea");
    if (!textarea) throw new Error("composer textarea not found");
    const b = btn.getBoundingClientRect();
    const r = row.getBoundingClientRect();
    const cs = getComputedStyle(row);
    const contentRight =
      r.right - parseFloat(cs.borderRightWidth) - parseFloat(cs.paddingRight);
    return {
      btnRight: b.right,
      btnWidth: b.width,
      btnHeight: b.height,
      contentRight,
      past: b.right - contentRight,
      textareaMinWidth: getComputedStyle(textarea).minWidth,
    };
  });
}

for (const width of WIDTHS) {
  test(`send button stays inside the composer row at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 800 });
    await page.goto("/");
    await page.waitForLoadState("networkidle");
    await page.waitForSelector('[aria-label="Send message"]');

    const m = await measure(page);
    // Printed on pass as well as fail — a clean run is the evidence that Gecko
    // does not reproduce, so the numbers need to be in the log either way.
    console.log(
      `${width}px  btn ${m.btnWidth.toFixed(2)}x${m.btnHeight.toFixed(2)}  ` +
        `right ${m.btnRight.toFixed(2)}  contentRight ${m.contentRight.toFixed(2)}  ` +
        `past ${m.past.toFixed(2)}`
    );

    // The behavioural assertion. Sub-pixel slack: layout rounding can
    // legitimately land a hair over. Note this only *fails* on Gecko — Blink
    // resolves the textarea's automatic minimum small enough that it never
    // binds, so under the chromium projects this passes either way. Which is
    // why the next assertion exists.
    expect(m.past, "button overflows the row's content box").toBeLessThanOrEqual(0.5);

    // The engine-independent invariant, and the one with teeth in CI: CI runs
    // chromium only, so reverting `min-w-0` would not trip the check above.
    // This defends the fix itself rather than the symptom.
    expect(
      m.textareaMinWidth,
      "textarea lost min-w-0, so the row has no slack and will clip on Gecko"
    ).toBe("0px");

    // Guards a different regression from the one above: if the button ever
    // loses `shrink-0` it compresses instead of overflowing. This does NOT
    // detect clipping — getBoundingClientRect returns the layout border box,
    // which an ancestor's overflow-hidden does not shrink, so the box stays
    // square while pixels are visibly cut off (as in #93's screenshot).
    expect(
      Math.abs(m.btnWidth - m.btnHeight),
      "button is not square, so it is being compressed"
    ).toBeLessThanOrEqual(0.5);
  });
}
