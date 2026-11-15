import { defineConfig } from "@playwright/test";

// PLAYWRIGHT_CHROMIUM_EXECUTABLE runs the suite on a Chromium already on the machine (for example an older
// build in the Playwright cache) when `npx playwright install chromium` cannot download the pinned one.
const executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE || undefined;

export default defineConfig({
  testDir: ".",
  testMatch: "*.spec.js",
  use: {
    baseURL: "http://127.0.0.1:8766",
    viewport: { width: 1440, height: 900 },
    launchOptions: executablePath ? { executablePath } : {},
  },
  webServer: { command: "node server.js", url: "http://127.0.0.1:8766", reuseExistingServer: false },
});
