import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  outputDir: "./test-results",
  // `list` for the live log, `html` for the report CI uploads on failure.
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: "http://localhost:5173",
    browserName: "chromium",
    // Traces are how a CI-only failure gets diagnosed; kept only for failures
    // so passing runs stay cheap.
    trace: "retain-on-failure",
  },
  projects: [
    { name: "desktop", use: { viewport: { width: 1280, height: 800 } } },
    { name: "mobile",  use: { viewport: { width: 390, height: 844 }, isMobile: true } },
    // Gecko is opt-in via GECKO=1. Issue #93 reproduces on Gecko and not on
    // Blink, and until this project existed nothing here had ever run on Gecko
    // at all — but CI installs the chromium binary only (ci.yml), so an
    // unconditional project would fail the run on a missing browser. Firefox
    // also rejects `isMobile`; the app keys its mobile branch off
    // window.innerWidth (useIsMobile), so the viewport alone gets us there.
    ...(process.env.GECKO
      ? [{
          name: "firefox-mobile",
          use: { browserName: "firefox" as const, viewport: { width: 365, height: 800 } },
        }]
      : []),
  ],
  webServer: {
    command: "npm run dev",
    url: "http://localhost:5173",
    // Locally, reuse a dev server you already have running. On CI the runner is
    // fresh, so anything already on :5173 would not be this commit's build.
    reuseExistingServer: !process.env.CI,
  },
});
