/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// /api/* is forwarded to the FastAPI service (the /api prefix is stripped).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: "http://localhost:8000", rewrite: (p) => p.replace(/^\/api/, "") },
    },
  },
  test: { environment: "jsdom", setupFiles: "./src/setupTests.ts", globals: true },
});
