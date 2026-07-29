import { defineConfig } from "@playwright/test";

// Issue #89, gap 2: the suite only ever exercised the Vite dev server, so the
// `dist/` bundle that Dockerfile.web ships and cloudbuild-web.yaml deploys was
// built by CI and executed by nothing — production was the first thing to run
// it. `preview-smoke` closes that: it serves the real build via `vite preview`
// and runs the @smoke-tagged path against it.
//
// Gated rather than unconditional because it costs a `vite build` on every run,
// and Playwright starts every configured webServer regardless of which projects
// are selected — an ungated one would tax every local `--project=desktop` run.
// Always on in CI; locally opt in with PREVIEW=1. Same idiom as GECKO below.
const previewEnabled = !!process.env.CI || !!process.env.PREVIEW;

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
    // Needs the browser first: `npx playwright install firefox`, then
    // `GECKO=1 npm run test:e2e -- --project=firefox-mobile`.
    ...(process.env.GECKO
      ? [{
          name: "firefox-mobile",
          use: { browserName: "firefox" as const, viewport: { width: 365, height: 800 } },
        }]
      : []),
    // Runs only the @smoke-tagged path, and against :4173 (the built bundle)
    // rather than the shared baseURL. Deliberately does not exclude that path
    // from `desktop`/`mobile` — this is an extra run against the production
    // artifact, not a relocation of the dev-server coverage.
    ...(previewEnabled
      ? [{
          name: "preview-smoke",
          grep: /@smoke/,
          use: { viewport: { width: 1280, height: 800 }, baseURL: "http://localhost:4173" },
        }]
      : []),
  ],
  webServer: [
    {
      command: "npm run dev",
      url: "http://localhost:5173",
      // Locally, reuse a dev server you already have running. On CI the runner is
      // fresh, so anything already on :5173 would not be this commit's build.
      reuseExistingServer: !process.env.CI,
    },
    ...(previewEnabled
      ? [{
          // Builds here rather than in a separate ci.yml step so the production
          // artifact cannot go stale relative to the run, and so PREVIEW=1 works
          // locally with no extra step. `vite preview` serves dist/ and would
          // otherwise happily serve whatever build was lying around.
          // --strictPort so a busy 4173 fails loudly instead of drifting to
          // another port that baseURL would not match.
          command: "npm run build && npm run preview -- --port 4173 --strictPort",
          url: "http://localhost:4173",
          // The default 60s covers serving, not a cold `vite build`.
          timeout: 180_000,
          reuseExistingServer: !process.env.CI,
        }]
      : []),
  ],
});
