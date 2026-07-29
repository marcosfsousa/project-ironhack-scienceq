import { test, expect } from "@playwright/test";
import { CATALOG_FIXTURE } from "../src/data/fixtures";

// Regression guard for the accent system (issue #76). Unlike the rest of the
// suite this test does assert on class names, deliberately: the thing under
// test *is* the CSS contract, not a user-visible behaviour reachable by role.
//
// Why it exists: accent colours are indirections. index.css defines
// `--color-accent: var(--acc)` inside `@theme inline`, and App.tsx reassigns
// --acc on <html> per [data-accent]. The `inline` is what makes the utility
// emit `color: var(--acc)` rather than `color: var(--color-accent)` — without
// it Tailwind resolves the var() where the theme variable is *defined*, which
// freezes every accent at its :root value. That failure compiles, ships, and
// looks fine until someone switches accent, so a green build does not cover
// it and neither does a screenshot of the default theme.

const ACCENTS: Record<string, { text: string; soft: string; bd: string }> = {
  Indigo: { text: "rgb(139, 147, 248)", soft: "rgba(129, 140, 248, 0.15)", bd: "rgba(129, 140, 248, 0.45)" },
  Blue: { text: "rgb(77, 150, 255)", soft: "rgba(77, 150, 255, 0.15)", bd: "rgba(77, 150, 255, 0.45)" },
  Amber: { text: "rgb(246, 183, 60)", soft: "rgba(246, 183, 60, 0.16)", bd: "rgba(246, 183, 60, 0.5)" },
  Cyan: { text: "rgb(52, 214, 238)", soft: "rgba(52, 214, 238, 0.15)", bd: "rgba(52, 214, 238, 0.45)" },
};

test.beforeEach(async ({ page }) => {
  await page.route("/api/catalog", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(CATALOG_FIXTURE) })
  );
});

test("accent utilities track a runtime [data-accent] change", async ({ page }) => {
  await page.goto("/");
  await page.waitForLoadState("networkidle");

  // A probe carrying the three accent utilities the app actually uses.
  await page.evaluate(() => {
    const el = document.createElement("div");
    el.id = "accent-probe";
    el.className = "text-accent bg-accent-soft border-accent-bd border";
    document.body.appendChild(el);
  });

  const seen: string[] = [];
  for (const [accent, expected] of Object.entries(ACCENTS)) {
    await page.evaluate((a) => {
      document.documentElement.dataset.accent = a;
    }, accent);

    const got = await page.evaluate(() => {
      const s = getComputedStyle(document.getElementById("accent-probe")!);
      return { text: s.color, soft: s.backgroundColor, bd: s.borderTopColor };
    });

    expect(got.text, `text-accent under ${accent}`).toBe(expected.text);
    expect(got.soft, `bg-accent-soft under ${accent}`).toBe(expected.soft);
    expect(got.bd, `border-accent-bd under ${accent}`).toBe(expected.bd);
    seen.push(got.text);
  }

  // Belt and braces. The per-accent assertions above already catch a frozen
  // accent, since three of the four expected values differ from :root — but
  // they only do so while they stay exact. This keeps the "they must actually
  // differ" property pinned if someone later loosens them.
  expect(new Set(seen).size, "four accents must resolve to four distinct colors").toBe(4);
});

test("accent resolves at the point of use, not at :root", async ({ page }) => {
  // This is the test that `@theme inline` is actually load-bearing for.
  //
  // The test above passes with or without `inline`, because App.tsx sets
  // data-accent on <html> — the same element @theme emits into — so the
  // indirection resolves against the right --acc either way. The two forms only
  // diverge when --acc is reassigned *below* :root: a plain @theme has already
  // substituted --color-accent at :root, and the nested override never reaches
  // the utility. So this pins the property rather than today's DOM shape, and
  // it is what would break if someone scoped an accent to a subtree.
  await page.goto("/");
  await page.waitForLoadState("networkidle");

  const nested = await page.evaluate(() => {
    document.documentElement.dataset.accent = "Indigo";
    const wrapper = document.createElement("div");
    wrapper.dataset.accent = "Amber";
    const el = document.createElement("div");
    el.className = "text-accent";
    wrapper.appendChild(el);
    document.body.appendChild(wrapper);
    return getComputedStyle(el).color;
  });

  expect(nested, "a nested [data-accent] must win over the :root value").toBe("rgb(246, 183, 60)");
});
