import path from "path";
import { fileURLToPath } from "url";
import type { Page } from "@playwright/test";

const FONT_DIR = fileURLToPath(new URL("./fixtures/fonts/", import.meta.url));

/**
 * Cuts every off-origin request the app makes, so a run depends on nothing but
 * this repo. `ci.yml` already describes the UI suite as mocked at the network
 * boundary; before this, that was true of `/api/*` and nothing else.
 *
 * Two things reached the internet:
 *
 * - `VideoEmbed` renders a live youtube.com/embed iframe. A screenshot baseline
 *   containing it is a picture of whatever thumbnail YouTube served that minute
 *   — it drifts when the CDN does and fails outright without egress. Stubbed to
 *   an empty document, which renders as a stable black box.
 *
 * - `index.html` loads IBM Plex Sans/Mono from Google Fonts. Blocking those is
 *   not an option: the UI falls back to system-ui and the baselines then depict
 *   a page no user sees. But leaving them live puts a third-party CDN inside a
 *   required status check compared at zero tolerance, where a 429 or a DNS blip
 *   silently yields fallback rendering and a red `UI tests (Playwright)` — on a
 *   repo where `main` deploys, indistinguishable from a real regression.
 *   `document.fonts.ready` does not help: it resolves whether or not the fetch
 *   succeeded. So the exact woff2 files Google serves are committed under
 *   tests/fixtures/fonts and served from disk, keeping real IBM Plex rendering
 *   with no network involved.
 *
 * Refresh the fixtures by re-fetching the URL in index.html with a Chrome UA and
 * taking the latin (U+0000-00FF) blocks; renaming is cosmetic, the route below
 * maps on basename.
 */
export async function stubOffOrigin(page: Page) {
  // Registered first so anything a caller adds later wins — Playwright resolves
  // routes in reverse registration order.
  await page.route("**/*", (route) => {
    const { hostname } = new URL(route.request().url());
    if (hostname === "localhost" || hostname === "127.0.0.1") return route.continue();
    return route.fulfill({ status: 200, contentType: "text/html", body: "" });
  });

  await page.route(/fonts\.googleapis\.com\/css2/, (route) =>
    route.fulfill({
      status: 200,
      contentType: "text/css",
      path: path.join(FONT_DIR, "fonts.css"),
    })
  );

  await page.route(/fonts\.gstatic\.com\//, (route) => {
    const file = path.basename(new URL(route.request().url()).pathname);
    return route.fulfill({
      status: 200,
      contentType: "font/woff2",
      path: path.join(FONT_DIR, file),
    });
  });
}
