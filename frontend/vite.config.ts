import path from "path";
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  return {
    plugins: [react()],
    // Pinned to what vite 6 called `'modules'`, which was the default this
    // project built against until the vite 8 bump. Vite 8 changed the default
    // to 'baseline-widely-available' — chrome111/edge111/firefox114/safari16.4
    // — so leaving it unset would have moved the browser floor as a side
    // effect of a dependency bump rather than as a decision.
    //
    // The visible consequence was in CSS, not JS, and only because
    // `build.cssTarget` defaults to this value. Lightning CSS emitted the one
    // media query in the bundle as `@media (width>=640px)`; range syntax is
    // Safari 16.4+, and a browser that cannot parse the prelude drops the
    // whole block, so `sm:px-6` silently stopped applying below 16.4 while
    // everything else kept working. The JS side was already safe by accident
    // — nothing in the emitted chunk exceeds Safari 14 either way, a 342-byte
    // difference — but that is a property of today's source, not a guarantee.
    //
    // Spelled out rather than written as `target: "modules"`: rolldown removed
    // that keyword, and `--target modules` now fails the build outright with
    // "Invalid target". Raising this floor is a real decision with real
    // reasons behind it; make it deliberately, in its own change.
    build: {
      target: ["es2020", "edge88", "firefox78", "chrome87", "safari14"],
    },
    resolve: {
      alias: { "@": path.resolve(__dirname, "./src") },
    },
    server: {
      port: 5173,
      proxy: {
        // The RAG backend (FastAPI / nginx Cloud Run service) is a separate origin.
        // In dev we proxy /api to it so the SSE stream and fetches are same-origin.
        // Point this at your backend; in prod nginx already routes /api.
        "/api": {
          target: env.VITE_API_TARGET ?? "http://localhost:8000",
          changeOrigin: true,
        },
      },
    },
  };
});
