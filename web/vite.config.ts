/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

// /api/* is forwarded to the FastAPI service (the /api prefix is stripped).
// Demo build (npm run build:demo): static files for GitHub Pages under /knowledge-search/.
export default defineConfig(({ mode }) => ({
  base: loadEnv(mode, ".").VITE_DEMO === "1" ? "/knowledge-search/" : "/",
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: "http://localhost:8000", rewrite: (p) => p.replace(/^\/api/, "") },
    },
  },
  test: { environment: "jsdom", setupFiles: "./src/setupTests.ts", globals: true },
}));
