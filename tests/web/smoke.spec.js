import { test, expect } from "@playwright/test";

const model = { name: "fake", label: "Fake model", available: true, open_weights: true, native: ["text"] };
const request = { model: "fake", dataset: "esc50", modality: "audio", scope: "closed" };
const card = { bridge: "text:captions-with-a-long-name", kind: "text", modality: "audio", dataset: "esc50", accuracy: 0.8, chance: 0.2, n: 40, verdict: "installed", controls: { swapped: 0.2 }, reason: "Passed." };
const event = (step, status = "ok") => ({ step, status, message: `${step} complete`, t: 1, data: {} });

async function setup(page, options = {}) {
  const errors = [];
  page.on("pageerror", (err) => errors.push(err.message));
  await page.route(/fonts\.(googleapis|gstatic)\.com/, (route) => route.abort());
  const calls = [];
  let cancelled = false;
  await page.route("**/api/**", async (route) => {
    const req = route.request(), path = new URL(req.url()).pathname;
    const json = (body, status = 200) => route.fulfill({ status, json: body });
    calls.push({ path, method: req.method(), body: req.postData() });
    if (path === "/api/models") return json([model]);
    if (path === "/api/datasets") return json([{ name: "esc50", title: "ESC-50", modality: "audio", available: true, labels: ["dog", "rain"] }]);
    if (path === "/api/senses") return json([{ ...card, card }]);
    if (path === "/api/samples") return json([]);
    if (path === "/api/runs") return json([{ id: "recorded", request, status: "done", installed: card.bridge, seconds: 20 }]);
    if (path === "/api/chat/stream") {
      if (options.stream) return options.stream(route);
      return json({ detail: "Not Found" }, 404);
    }
    if (path === "/api/chat") {
      if (options.chat) return options.chat(route, calls.filter((c) => c.path === path).length);
      return json({ reply: "I hear a dog.", used: [] });
    }
    if (["/api/auto", "/api/fit", "/api/replay"].includes(path)) {
      cancelled = false;
      return json({ job_id: "job-1" });
    }
    if (path === "/api/jobs/job-1" && req.method() === "DELETE") {
      if (options.cancelError) return json({ detail: "Cannot be cancelled yet" }, 409);
      cancelled = true;
      return json({ cancelled: true });
    }
    if (path === "/api/jobs/job-1") return json({
      request: options.auto ? { model: "fake", goal: "Recognise this sound" } : request,
      status: cancelled ? "failed" : "running", error: cancelled ? "Cancelled." : null,
      events: (options.auto ? ["detect", "research", "data", "inspect", "plan", "fit", "verify", "claims", "decide", "install"] : ["inspect", "plan"]).map((s) => event(s)),
      cards: [card],
    });
    throw new Error(`Unexpected API call ${req.method()} ${path}`);
  });
  await page.goto("/");
  await expect(page.locator("#model-label")).toHaveText("Fake model");
  return { calls, errors };
}

async function send(page, text) {
  await page.locator("#message").fill(text);
  await page.locator("#send-button").click();
  await expect(page.locator("#send-button")).toBeEnabled();
}

test("graph fits 1440 and 1280; section IDs and run names remain readable", async ({ page }) => {
  const { errors } = await setup(page);
  const indices = await page.locator(".pane-title > .idx").allTextContents();
  expect(new Set(indices).size).toBe(indices.length);
  await page.locator("#runs-button").click();
  const outcome = page.locator(".run-item-outcome").first();
  await expect(outcome).toContainText(card.bridge);
  expect(await outcome.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(true);
  await page.getByRole("button", { name: "Replay", exact: true }).click();
  for (const width of [1440, 1280, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    await page.getByRole("tab", { name: "Graph", exact: true }).click();
    await expect.poll(() => page.locator("#graph-canvas").evaluate((el) => el.scrollWidth - el.clientWidth)).toBeLessThanOrEqual(1);
    await page.getByRole("tab", { name: "Report", exact: true }).click();
  }
  await page.locator("#mode-button").click();
  await page.getByRole("tab", { name: "Graph", exact: true }).click();
  await expect.poll(() => page.locator("#graph-canvas").evaluate((el) => el.scrollWidth - el.clientWidth)).toBeLessThanOrEqual(1);
  expect(errors).toEqual([]);
});

test("chats survive reload, history stays isolated, editing removes stale follow-ups", async ({ page }) => {
  const { calls, errors } = await setup(page);
  await send(page, "First question");
  await send(page, "Follow-up");
  expect(calls.filter((c) => c.path === "/api/chat").at(-1).body).toContain('"content":"First question"');
  await page.reload();
  await expect(page.locator(".msg--user .bubble")).toHaveText(["First question", "Follow-up"]);
  await page.getByRole("button", { name: "Edit", exact: true }).first().click();
  await send(page, "Corrected question");
  await expect(page.locator(".msg--user .bubble")).toHaveText(["Corrected question"]);
  expect(calls.filter((c) => c.path === "/api/chat").at(-1).body).not.toContain("First question");
  await page.locator("#new-chat-button").click();
  await send(page, "Separate conversation");
  expect(calls.filter((c) => c.path === "/api/chat").at(-1).body).not.toContain("Corrected question");
  await page.reload();
  await expect(page.locator("#chats-list button")).toHaveCount(2);
  await page.locator("#chats-list button").filter({ hasText: "Corrected question" }).click();
  await expect(page.locator(".msg--user .bubble")).toHaveText(["Corrected question"]);
  expect(errors).toEqual([]);
});

test("failure keeps its message; retry succeeds once without duplicate history", async ({ page }) => {
  const { calls } = await setup(page, { chat: (route, n) => route.fulfill({
    status: n === 1 ? 500 : 200, json: n === 1 ? { detail: "Model temporarily unavailable" } : { reply: "Recovered", used: [] },
  }) });
  await send(page, "Try this");
  await expect(page.locator(".turn .error")).toContainText("temporarily unavailable");
  await page.getByRole("button", { name: "Retry", exact: true }).click();
  await expect(page.locator(".msg-text")).toHaveText("Recovered");
  await expect(page.locator(".msg--user .bubble")).toHaveCount(1);
  expect(calls.filter((c) => c.path === "/api/chat")).toHaveLength(2);
});

test("restored uploads require reattachment before retry, and saved text is not HTML", async ({ page }) => {
  await setup(page);
  await page.locator("#file-input").setInputFiles({ name: "clip.wav", mimeType: "audio/wav", buffer: Buffer.from("audio") });
  await send(page, '<img src=x onerror="alert(1)">');
  await page.reload();
  await expect(page.locator(".msg--user img")).toHaveCount(0);
  await page.getByRole("button", { name: "Retry", exact: true }).click();
  await expect(page.locator("#chat-error")).toContainText("Reattach");
  await page.locator("#send-button").click();
  await expect(page.locator("#chat-error")).toContainText("Reattach");
  await expect(page.locator(".msg--user .bubble")).toHaveCount(1);
});

test("SSE metadata and tokens become a saved reply; truncated stream is retryable", async ({ page }) => {
  let complete = true;
  await setup(page, { stream: (route) => route.fulfill({
    contentType: "text/event-stream",
    body: 'event: used\r\ndata: [{"kind":"tool","bridge":"tool:esc50","source":"clip.wav"}]\r\n\r\nevent: token\r\ndata: {"text":"Dog 🐕"}\r\n\r\n'
      + (complete ? 'event: done\r\ndata: {"reply":"Dog 🐕"}\r\n\r\n' : ""),
  }) });
  await send(page, "Stream this");
  await expect(page.locator(".msg-text")).toHaveText("Dog 🐕");
  await expect(page.locator(".used-item")).toContainText("tool:esc50");
  await page.reload();
  await expect(page.locator(".msg-text")).toHaveText("Dog 🐕");
  complete = false;
  await send(page, "Incomplete");
  await expect(page.locator(".turn .error")).toContainText("ended early");
});

test("goal plus a file starts auto-discovery with ordered steps and cancellation", async ({ page }) => {
  const { calls, errors } = await setup(page, { auto: true });
  await page.locator("#new-sense-button").click();
  await expect(page.locator("#manual-fields")).toBeHidden();
  await expect(page.locator("#run-button")).toBeDisabled();
  await page.locator("#fit-goal").fill("Recognise this sound");
  await page.locator("#fit-file").setInputFiles({ name: "clip.wav", mimeType: "audio/wav", buffer: Buffer.from("audio") });
  await page.locator("#run-button").click();
  await expect(page.locator(".step-title")).toHaveText([
    "Detect the input", "Research the task", "Find labelled examples", "Check the base model", "Choose bridges",
    "Build bridges", "Verify on held-out", "Report supported claims", "Install or refuse", "Install",
  ]);
  expect(calls.find((c) => c.path === "/api/auto").body).toContain('name="file"; filename="clip.wav"');
  await page.locator("#job-stop").click();
  await expect(page.locator("#trace-status")).toHaveText("Failed");
  await expect(page.locator("#trace-outcome")).toContainText("Cancelled.");
  expect(errors).toEqual([]);
});

test("labelled-set fit and replay remain available; replay can stop", async ({ page }) => {
  const { calls } = await setup(page);
  await page.locator("#new-sense-button").click();
  await page.locator('label[for="fit-manual"]').click();
  await page.locator("#run-button").click();
  expect(calls.find((c) => c.path === "/api/fit").body).toContain('"dataset":"esc50"');
  await expect(page.locator(".step-title")).toHaveCount(6);
  await page.locator("#job-stop").click();
  await expect(page.locator("#trace-status")).toHaveText("Failed");
  await page.locator("#runs-button").click();
  await page.getByRole("button", { name: "Replay", exact: true }).click();
  await expect(page.getByRole("button", { name: "Stop replay", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Stop replay", exact: true }).click();
  expect(calls.filter((c) => c.method === "DELETE")).toHaveLength(2);
});

test("replay stops through the job endpoint; a rejected cancellation remains visible", async ({ page }) => {
  const { calls } = await setup(page, { cancelError: true });
  await page.locator("#runs-button").click();
  await page.getByRole("button", { name: "Replay", exact: true }).click();
  await page.getByRole("button", { name: "Stop replay", exact: true }).click();
  await expect(page.locator("#trace-error")).toContainText("Cannot be cancelled yet");
  // A later poll must not erase an actionable cancellation failure.
  await expect.poll(() => calls.filter((c) => c.path === "/api/jobs/job-1" && c.method === "GET").length).toBeGreaterThan(1);
  await expect(page.locator("#trace-error")).toContainText("Cannot be cancelled yet");
});

test("stream fragments render before completion and Stop receiving releases the composer", async ({ page }) => {
  await setup(page, { stream: (route) => route.continue({ url: "http://127.0.0.1:8766/test-stream" }) });
  await page.locator("#message").fill("Stream this slowly");
  await page.locator("#send-button").click();
  await expect(page.locator(".msg-text")).toHaveText("Dog 🐕");
  await expect(page.locator("#send-button")).toBeDisabled();
  await page.locator("#chat-stop").click();
  await expect(page.locator(".turn .error")).toContainText("Stopped receiving");
  await expect(page.locator("#send-button")).toBeEnabled();
  await page.getByRole("button", { name: "Retry", exact: true }).click();
  await expect(page.locator(".msg-text")).toHaveText("Dog 🐕");
  await expect(page.locator("#send-button")).toBeEnabled();
});

test("a reply arriving after chat switching stays with its original chat", async ({ page }) => {
  let finish;
  await setup(page, { chat: (route) => new Promise((resolve) => {
    finish = async () => { await route.fulfill({ json: { reply: "Late answer", used: [] } }); resolve(); };
  }) });
  await page.locator("#message").fill("Slow question");
  await page.locator("#send-button").click();
  await expect.poll(() => Boolean(finish)).toBe(true);
  await expect(page.getByRole("region", { name: "Working", exact: true })).toContainText("Slow question");
  await page.locator("#new-chat-button").click();
  await expect(page.locator("#chat-activity")).toContainText("Slow question");
  await expect(page.locator("#chat-stop")).toBeHidden();
  await page.getByRole("button", { name: "View working chat" }).click();
  await expect(page.locator(".msg--user .bubble")).toHaveText("Slow question");
  await expect(page.locator("#chat-stop")).toBeVisible();
  await page.locator("#chats-list button").filter({ hasText: "New chat" }).click();
  await finish();
  await expect(page.locator("#send-button")).toBeEnabled();
  await expect(page.getByRole("region", { name: "Working", exact: true })).toHaveCount(0);
  await expect(page.locator(".msg-text")).toHaveCount(0);
  await page.reload();
  await page.locator("#chats-list button").filter({ hasText: "Slow question" }).click();
  await expect(page.locator(".msg-text")).toHaveText("Late answer");
});

test("corrupt storage and interrupted saved replies do not break startup", async ({ page }) => {
  const { errors } = await setup(page);
  await page.evaluate(() => localStorage.setItem("sense.chats.v1", "{broken"));
  await page.reload();
  await expect(page.locator("#send-button")).toBeEnabled();
  await page.evaluate(() => localStorage.setItem("sense.chats.v1", JSON.stringify({
    chats: [{ id: 1, messages: [{ model: { name: "fake" }, message: "Interrupted", status: "pending" }] }], chatId: 1,
  })));
  await page.reload();
  await expect(page.locator(".turn .error")).toContainText("interrupted by reload");
  await page.getByRole("button", { name: "Retry", exact: true }).click();
  await expect(page.locator(".msg-text")).toHaveText("I hear a dog.");
  expect(errors).toEqual([]);
});

test("unavailable browser storage still allows chat and reports that it was not saved", async ({ page }) => {
  await page.addInitScript(() => {
    Storage.prototype.setItem = () => { throw new DOMException("Storage full", "QuotaExceededError"); };
  });
  const { errors } = await setup(page);
  await send(page, "Still works");
  await expect(page.locator(".msg-text")).toHaveText("I hear a dog.");
  await expect(page.locator("#chat-storage")).toContainText("chats are not saved");
  expect(errors).toEqual([]);
});

test("chat search finds previous messages and switching preserves separate drafts and files", async ({ page }) => {
  await setup(page);
  await send(page, "Rain in the forest");
  await page.locator("#message").fill("First draft");
  await page.locator("#file-input").setInputFiles({ name: "forest.wav", mimeType: "audio/wav", buffer: Buffer.from("audio") });
  await page.locator("#new-chat-button").click();
  await expect(page.locator("#message")).toHaveValue("");
  await expect(page.locator("#attachments")).toBeHidden();
  await send(page, "Music in the city");
  await page.locator("#message").fill("Second draft");
  await page.getByRole("searchbox", { name: "Search chats" }).fill("forest");
  await expect(page.locator("#chats-list button")).toHaveCount(1);
  await page.locator("#chats-list button").click();
  await expect(page.locator("#message")).toHaveValue("First draft");
  await expect(page.locator("#attachments")).toContainText("forest.wav");
  await page.getByRole("searchbox", { name: "Search chats" }).fill("");
  await page.getByRole("region", { name: "Previous chats", exact: true }).getByRole("button").click();
  await expect(page.locator("#message")).toHaveValue("Second draft");
  await expect(page.locator("#attachments")).toBeHidden();
});

test("runs and new-sense dialogs contain keyboard focus and return it on Escape", async ({ page }) => {
  await setup(page);
  await expect(page.locator("#runs-list")).toBeHidden();
  await page.locator("#runs-button").focus();
  await page.keyboard.press("Enter");
  const dialog = page.getByRole("dialog", { name: "Runs", exact: true });
  await expect(dialog).toBeVisible();
  for (let i = 0; i < 12; i++) {
    await page.keyboard.press("Tab");
    expect(await dialog.evaluate((el) => el.contains(document.activeElement))).toBe(true);
  }
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await expect(page.locator("#runs-button")).toBeFocused();
  await page.locator("#new-sense-button").click();
  await expect(page.locator("#fit-goal")).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(page.locator("#new-sense-button")).toBeFocused();
});

test("inspector tabs work with arrows; brain senses have keyboard-accessible reports", async ({ page }) => {
  await setup(page);
  await page.locator("#report-tab").focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.locator("#run-tab")).toBeFocused();
  await expect(page.locator("#run-view")).toBeVisible();
  await page.keyboard.press("End");
  await expect(page.locator("#graph-tab")).toBeFocused(); // disabled Log is skipped
  await page.keyboard.press("ArrowRight");
  await expect(page.locator("#report-tab")).toBeFocused();
  await page.getByRole("tab", { name: "Graph", exact: true }).click();
  await page.locator("#brain-access button").focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#report-view")).toBeVisible();
  await expect(page.locator("#report")).toContainText("80.0%");
  await expect(page.locator("#report")).toContainText(card.reason);
});

test("chat navigation, composer and runs reflow at phone and zoomed widths", async ({ page }) => {
  const { errors } = await setup(page);
  for (const width of [768, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    await expect(page.locator("#new-chat-button")).toBeVisible();
    await expect(page.locator("#message")).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.locator("#runs-button").click();
    await expect(page.locator("#runs-dialog")).toBeVisible();
    expect(await page.locator("#runs-dialog").evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(true);
    await page.keyboard.press("Escape");
  }
  await page.locator("#mode-button").click();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});
