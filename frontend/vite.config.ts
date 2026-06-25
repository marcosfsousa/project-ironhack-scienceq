import path from "path";
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  return {
    plugins: [react()],
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
