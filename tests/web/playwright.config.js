import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: ".",
  testMatch: "*.spec.js",
  use: { baseURL: "http://127.0.0.1:8766", viewport: { width: 1440, height: 900 } },
  webServer: { command: "node server.js", url: "http://127.0.0.1:8766", reuseExistingServer: false },
});
