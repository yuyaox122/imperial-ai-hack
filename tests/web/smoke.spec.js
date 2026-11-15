import { test, expect } from "@playwright/test";

const model = { name: "fake", label: "Fake model", available: true, open_weights: true, native: ["text"] };
const request = { model: "fake", dataset: "esc50", modality: "audio", scope: "closed" };
// What GET /api/jobs/<id> really returns for an auto run before its data step (agent.jobs.start_auto): no goal.
const autoRequest = { model: "fake", dataset: "", modality: "", scope: "closed" };
const goal = "Recognise this sound";
const card = { bridge: "text:captions-with-a-long-name", kind: "text", modality: "audio", dataset: "esc50", accuracy: 0.8, chance: 0.2, n: 40, verdict: "installed", controls: { swapped: 0.2 }, reason: "Passed." };
const esc50 = { name: "esc50", title: "ESC-50", modality: "audio", available: true, labels: ["dog", "rain"] };
const event = (step, status = "ok", message = `${step} complete`, data = {}) => ({ step, status, message, t: 1, data });
const detected = [event("detect", "start"), event("detect", "ok", "An audio clip.", { file: "clip.wav", goal, modality: "audio" })];

async function setup(page, options = {}) {
  const errors = [];
  page.on("pageerror", (err) => errors.push(err.message));
  await page.route(/fonts\.(googleapis|gstatic)\.com/, (route) => route.abort());
  const calls = [];
  const count = (path) => calls.filter((c) => c.path === path).length;
  let cancelled = false;
  await page.route("**/api/**", async (route) => {
    const req = route.request(), path = new URL(req.url()).pathname;
    const json = (body, status = 200) => route.fulfill({ status, json: body });
    calls.push({ path, method: req.method(), body: req.postData() });
    if (path === "/api/models") return options.models ? options.models(route, count(path)) : json([model]);
    if (path === "/api/datasets") return json(options.datasets ? options.datasets(count(path)) : [esc50]);
    if (path === "/api/senses") return json(options.senses || [{ ...card, card }]);
    if (path === "/api/samples") return json([]);
    if (path === "/api/runs") return json(options.runs || [{ id: "recorded", request, status: "done", installed: card.bridge, seconds: 20 }]);
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
    if (path === "/api/jobs/job-1") {
      if (options.job) return json({ id: "job-1", ...options.job(cancelled) });
      return json({
        id: "job-1",
        request: options.auto ? autoRequest : request,
        status: cancelled ? "failed" : "running", error: cancelled ? "Cancelled." : null,
        events: options.auto ? [...detected, event("research", "start")] : ["inspect", "plan"].map((s) => event(s)),
        cards: options.auto ? [] : [card],
      });
    }
    throw new Error(`Unexpected API call ${req.method()} ${path}`);
  });
  await page.goto("/");
  if (options.models) return { calls, errors };
  await expect(page.locator("#model-label")).toHaveText("Fake model");
  return { calls, errors };
}

/** A finished run's GET /api/jobs body: steps from inspect to done, then the final cards. */
function finishedJob(req, cards, installed, { kinds = cards.map((c) => c.kind), decide = "", before = [] } = {}) {
  return {
    request: req, status: "done", error: null, installed, cards,
    events: [
      ...before,
      event("inspect", "start"), event("inspect"),
      event("plan", "start"), event("plan", "ok", "Bridges to try.", { kinds, planner: "llm" }),
      ...cards.flatMap((c) => [event("fit", "start", "Fitting.", { kind: c.kind }), event("fit", "ok", "Built.", { kind: c.kind, bridge: c.bridge })]),
      ...cards.flatMap((c) => [event("verify", "start", "Scoring.", { bridge: c.bridge }), event("verify", "ok", "Scored.", { card: { ...c, verdict: "candidate" } })]),
      event("decide", "start"), event("decide", "ok", decide, { install: installed }),
      ...(installed ? [event("install", "start"), event("install", "ok", "Installed.", { bridge: installed })] : [event("install", "info", "Installed nothing.")]),
      event("done", "ok", installed ? `Installed ${installed}.` : "Nothing installed.", { installed }),
    ],
  };
}

async function replayFirstRun(page) {
  await page.locator("#runs-button").click();
  await page.getByRole("button", { name: "Replay", exact: true }).first().click();
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

test("goal plus a file starts auto-discovery with ordered steps; the goal stays in view; Stop is not a failure", async ({ page }) => {
  const { calls, errors } = await setup(page, { auto: true, senses: [] });
  await page.locator("#new-sense-button").click();
  await expect(page.locator("#manual-fields")).toBeHidden();
  await expect(page.locator("#run-button")).toBeDisabled();
  await page.locator("#fit-goal").fill(goal);
  await page.locator("#fit-file").setInputFiles({ name: "clip.wav", mimeType: "audio/wav", buffer: Buffer.from("audio") });
  await page.locator("#run-button").click();
  await expect(page.locator(".step-title")).toHaveText([
    "Detect the input", "Research the task", "Find labelled examples", "Check the base model", "Choose bridges",
    "Build bridges", "Verify on held-out", "Report supported claims", "Install or refuse", "Install",
  ]);
  const body = calls.find((c) => c.path === "/api/auto").body;
  expect(body).toContain('name="file"; filename="clip.wav"');
  expect(body).not.toContain('name="examples"'); // the optional folder was left empty
  // The server's request has no goal and dataset "" until the data step: the goal still heads the run.
  await expect(page.locator("#trace-request")).toHaveText(`Fake model · ${goal} · step 2 of 10`);
  await expect(page.locator("#report")).toContainText(`Audio → Fake model · ${goal}`);
  await expect(page.locator("#report")).toContainText("clears chance and swapped audio by a fixed margin");
  await page.locator("#runs-button").click();
  await expect(page.locator(".run-item--live")).toContainText(`Fake model · ${goal}`);
  await page.keyboard.press("Escape");
  await page.locator("#job-stop").click();
  await expect(page.locator("#trace-status")).toHaveText("Stopped");
  await expect(page.locator("#trace-outcome")).toContainText("You stopped the run before it finished.");
  await expect(page.locator("#trace-outcome [role=alert]")).toHaveCount(0);
  await expect(page.locator(".step").nth(1)).toHaveClass(/step--stopped/);
  await expect(page.locator(".msg--agent")).toContainText("You stopped the run");
  await expect(page.locator(".msg--agent")).not.toContainText("failed");
  await expect(page.locator("#status-job")).toContainText("stopped");
  expect(errors).toEqual([]);
});

test("labelled-set fit and replay remain available; replay can stop; no stray null text", async ({ page }) => {
  const { calls, errors } = await setup(page);
  await page.locator("#new-sense-button").click();
  await page.locator('label[for="fit-manual"]').click();
  await page.locator("#run-button").click();
  expect(calls.find((c) => c.path === "/api/fit").body).toContain('"dataset":"esc50"');
  await expect(page.locator(".step-title")).toHaveCount(6);
  await page.locator("#job-stop").click();
  await expect(page.locator("#trace-status")).toHaveText("Stopped");
  await page.locator("#runs-button").click();
  await page.getByRole("button", { name: "Replay", exact: true }).click();
  const bar = page.locator("#replay-bar");
  await expect(page.getByRole("button", { name: "Stop replay", exact: true })).toBeVisible();
  await expect(bar).toContainText("chat stays live");
  expect(await bar.textContent()).not.toContain("null");
  await page.getByRole("button", { name: "Stop replay", exact: true }).click();
  expect(calls.filter((c) => c.method === "DELETE")).toHaveLength(2);
  await expect(bar).toContainText("Replay stopped");
  await expect(page.getByRole("button", { name: "Close the replay bar" })).toBeVisible();
  expect(await bar.textContent()).not.toContain("null");
  // Hover across the brain: no tooltip ends in "null" (regions without a bridge have no hint line).
  const box = await page.locator("#brain").boundingBox();
  const tips = [];
  for (let i = 1; i < 20; i += 1) {
    for (const y of [0.3, 0.45, 0.62]) {
      await page.mouse.move(box.x + (box.width * i) / 20, box.y + box.height * y);
      if (await page.locator("#brain-tip").isVisible()) tips.push(await page.locator("#brain-tip").textContent());
    }
  }
  expect(tips.length).toBeGreaterThan(0);
  expect(tips.filter((t) => t.includes("null"))).toEqual([]);
  expect(errors).toEqual([]);
});

test("the stream's done.used is final: a tool the model did not call is not reported as heard", async ({ page }) => {
  const early = { kind: "tool", bridge: "tool:esc50", source: "clip.wav", detail: "the model can call the listen tool to hear this clip" };
  let detail = "the model could call listen but did not";
  await setup(page, { stream: (route) => route.fulfill({
    contentType: "text/event-stream",
    body: `event: used\ndata: ${JSON.stringify([early])}\n\nevent: token\ndata: {"text":"Rock."}\n\n`
      + `event: done\ndata: ${JSON.stringify({ reply: "Rock.", used: [{ ...early, detail }] })}\n\n`,
  }) });
  await send(page, "Which genre?");
  await expect(page.locator(".used-item")).toContainText("could call listen but did not");
  await expect(page.locator(".msg--model .name-line-tags")).toHaveText("Tool not called");
  detail = "the model called listen 1 time: Nearest labels: rock 0.90.";
  await send(page, "And this one?");
  await expect(page.locator(".used-item").last()).toContainText("called listen 1 time");
  await expect(page.locator(".msg--model .name-line-tags").last()).toHaveText("Heard via tool bridge");
  await page.reload();
  await expect(page.locator(".used-item")).toHaveCount(2);
  await expect(page.locator(".used-item").first()).toContainText("could call listen but did not");
  await expect(page.locator("#thread")).not.toContainText("can call the listen tool");
});

test("a file the model takes as it is (kind native) reads as is, not as a bridge", async ({ page }) => {
  const native = { kind: "native", bridge: "", source: "note.txt", detail: "its text was given to the model with the message" };
  await setup(page, { chat: (route) => route.fulfill({ json: { reply: "It says hello.", used: [native] } }) });
  expect(await page.locator("#file-input").getAttribute("accept")).toContain(".txt");
  await page.locator("#file-input").setInputFiles({ name: "note.txt", mimeType: "text/plain", buffer: Buffer.from("hello") });
  await send(page, "What does it say?");
  await expect(page.locator(".used-item")).toHaveText("note.txt: Read as is · its text was given to the model with the message");
  await expect(page.locator(".msg--model .name-line-tags")).toHaveText("Read as is");
  await expect(page.locator(".msg--model")).not.toContainText("bridge");
});

test("a bridge that failed its checks is refused whatever the plan order; one that passed and lost is not", async ({ page }) => {
  // socket_pilot/artifacts/run_qwen3.5-0.8b_esc50_open.json: the planner put text first; vector failed on unseen classes.
  const open = { ...request, scope: "open" };
  const text = { bridge: "text:captions", kind: "text", modality: "audio", dataset: "esc50", scope: "open", accuracy: 0.805, chance: 0.2, n: 200, verdict: "installed",
    controls: { swapped: 0.18, no_input: 0.21, lookup: 0.995, specialist_alone: 0.99 },
    reason: "Installed text:captions: it scored 80.5% against 20.0% chance. vector:esc50 failed: it scored 51.2% on classes it was not fitted on." };
  const vector = { bridge: "vector:esc50", kind: "vector", modality: "audio", dataset: "esc50", scope: "open", accuracy: 0.965, chance: 0.2, n: 200, verdict: "rejected",
    controls: { swapped: 0.18, no_input: 0.21, lookup: 0.995, specialist_alone: 0.99, text_bridge: 0.805, unseen_classes: 0.512 },
    reason: "vector:esc50 scored 96.5% on 200 held-out questions. text:captions was installed instead: Installed text:captions. vector:esc50 failed: it scored 51.2% on classes it was not fitted on." };
  // run_openai-chat_esc50_closed.json: text passed and lost to the tool bridge.
  const passed = { ...card, bridge: "text:esc50", accuracy: 0.92, verdict: "rejected", controls: { swapped: 0.16 },
    reason: "text:esc50 passed: it scored 92.0% against 20.0% chance. It was not installed because tool:esc50 was chosen instead." };
  const tool = { ...card, bridge: "tool:esc50", kind: "tool", accuracy: 0.96, controls: { swapped: 0.18 }, reason: "Installed tool:esc50." };
  // An older record whose reasons say neither "passed" nor "failed": the rule planner's bars decide.
  const quiet = { ...card, bridge: "text:gtzan-folder", accuracy: 0.48, verdict: "rejected", controls: { swapped: 0.19 }, reason: "The text bridge is weaker at 48%." };
  const vectorWon = { ...card, bridge: "vector:gtzan-folder", kind: "vector", accuracy: 0.77, controls: { swapped: 0.16 }, reason: "Install vector:gtzan-folder." };
  const scenarios = [
    finishedJob(open, [text, vector], "text:captions", { kinds: ["text", "vector"], decide: text.reason }),
    finishedJob(request, [passed, tool], "tool:esc50", { kinds: ["text", "tool"] }),
    finishedJob(request, [quiet, vectorWon], "vector:gtzan-folder", { kinds: ["text", "vector"] }),
  ];
  let scenario = 0;
  await setup(page, { job: () => scenarios[scenario] });

  await replayFirstRun(page);
  await expect(page.locator("#trace-status")).toHaveText("Refused · fallback installed");
  await expect(page.locator(".msg--agent").last()).toContainText("Vector bridge refused");
  await expect(page.locator(".msg--agent").last()).toContainText("96.5%");
  await expect(page.locator(".step").filter({ hasText: "Install or refuse" })).toHaveClass(/step--refused/);
  await page.getByRole("button", { name: "See why →" }).last().click();
  await expect(page.locator("#report-verdict")).toHaveText("Refused");
  await expect(page.locator("#report")).toContainText("Why it was refused");
  await expect(page.locator("#report")).toContainText("Scope: any sound, also tested on categories it was not fitted on.");
  await page.getByRole("tab", { name: "Graph", exact: true }).click();
  await expect(page.locator(".node--refused")).toContainText("Vector bridge");
  // The installed text bridge was not tested on unseen categories, and its card must not say it was.
  await page.locator(".node--installed").click();
  await expect(page.locator("#report-verdict")).toHaveText("Installed");
  await expect(page.locator("#report")).not.toContainText("also tested on categories it was not fitted on");
  await expect(page.locator("#report")).toContainText("This bridge was not tested on categories it was not fitted on");

  for (const [i, loser, winner] of [[1, "text:esc50", "tool:esc50"], [2, "text:gtzan-folder", "vector:gtzan-folder"]]) {
    scenario = i;
    await replayFirstRun(page);
    await expect(page.locator("#trace-status")).toHaveText("Sense installed");
    await expect(page.locator(".msg--agent").last()).toContainText("installed");
    await expect(page.locator(".msg--agent").last()).not.toContainText("refused");
    await page.getByRole("tab", { name: "Graph", exact: true }).click();
    await expect(page.locator(".node--refused")).toHaveCount(0);
    await page.locator(".node").filter({ hasText: "Text bridge" }).click();
    await expect(page.locator("#report-verdict")).toHaveText("not installed");
    await expect(page.locator("#report .report-sub")).not.toContainText(winner);
    expect(await page.locator("#report .report-sub").getAttribute("title")).toBe(loser);
  }
});

test("a step that recovered from a failed sub-step is done; a run needing examples asks for a folder", async ({ page }) => {
  const recovered = finishedJob(request, [card], card.bridge, { before: [
    ...detected,
    event("research", "start"), event("research"),
    event("data", "start", "Deciding.", { action: "need" }), event("data", "ok", "Need labels.", { action: "need" }),
    event("data", "start", "Searching.", { action: "search" }), event("data", "ok", "Found 1 dataset candidates.", { action: "search" }),
    event("data", "start", "Fetching.", { action: "fetch" }), event("data", "fail", "Could not use example/fake-sounds: fake mode has no network", { action: "candidate" }),
    event("data", "start", "Looking for a prewired set.", { action: "fallback" }), event("data", "ok", "Using esc50 (prewired) to verify the goal.", { dataset: "esc50" }),
  ] });
  const needsUser = "I could not find labelled audio examples for this goal. Put at least 10 clips per category in a folder with one sub-folder per label and run again with that folder.";
  const stuck = {
    request: autoRequest, status: "done", error: null, installed: null, cards: [],
    events: [
      ...detected, event("research", "start"), event("research"),
      event("data", "start", "Deciding.", { action: "need" }), event("data", "ok", "Need labels.", { action: "need" }),
      event("data", "start", "Searching.", { action: "search" }), event("data", "fail", "Dataset search was unavailable.", { action: "search" }),
      event("data", "start", "Looking for a prewired set.", { action: "fallback" }), event("data", "info", "No matching prewired labelled set is available.", { action: "fallback" }),
      event("done", "ok", needsUser, { installed: null, needs_user: true }),
    ],
  };
  let job = recovered;
  const { calls, errors } = await setup(page, { job: () => job });
  await page.locator("#new-sense-button").click();
  await page.locator("#fit-goal").fill(goal);
  await page.locator("#fit-file").setInputFiles({ name: "clip.wav", mimeType: "audio/wav", buffer: Buffer.from("audio") });
  await page.locator("#run-button").click();
  await expect(page.locator("#trace-status")).toHaveText("Sense installed");
  await expect(page.locator(".step").filter({ hasText: "Find labelled examples" })).toHaveClass(/step--done/);

  job = stuck;
  await page.locator("#new-sense-button").click();
  await page.locator("#run-button").click();
  await expect(page.locator("#trace-status")).toHaveText("Needs examples");
  await expect(page.locator(".step").filter({ hasText: "Find labelled examples" })).toHaveClass(/step--refused/);
  const message = page.locator(".msg--agent").last();
  await expect(message).toContainText("The agent needs labelled examples");
  expect((await message.textContent()).split("Put at least 10 clips").length - 1).toBe(1); // said once, not twice
  await page.getByRole("button", { name: "Run again with a folder of examples →" }).click();
  await expect(page.locator("#fit-examples")).toBeFocused();
  await expect(page.locator("#fit-goal")).toHaveValue(goal);
  await page.locator("#fit-examples").fill("/data/bird-calls");
  await page.locator("#run-button").click();
  const body = calls.filter((c) => c.path === "/api/auto").at(-1).body;
  expect(body).toContain('name="examples"');
  expect(body).toContain("/data/bird-calls");
  expect(errors).toEqual([]);
});

test("a labelled set a run built joins the set list, the graph and the titles", async ({ page }) => {
  const birds = { name: "birds", title: "Bird calls", modality: "audio", available: true, labels: ["robin", "wren"], n: 40 };
  const built = { ...card, bridge: "vector:birds", kind: "vector", dataset: "birds", controls: { swapped: 0.2 } };
  const { calls } = await setup(page, {
    datasets: (n) => (n > 1 ? [esc50, birds] : [esc50]),
    job: () => finishedJob({ ...request, dataset: "birds" }, [built], built.bridge, { before: detected }),
  });
  await page.locator("#new-sense-button").click();
  await page.locator("#fit-goal").fill(goal);
  await page.locator("#fit-file").setInputFiles({ name: "clip.wav", mimeType: "audio/wav", buffer: Buffer.from("audio") });
  await page.locator("#run-button").click();
  await expect(page.locator("#trace-status")).toHaveText("Sense installed");
  await expect.poll(() => calls.filter((c) => c.path === "/api/datasets").length).toBe(2);
  await expect(page.locator("#dataset-select option")).toContainText(["ESC-50 (2 labels)", "Bird calls (2 labels)"]);
  await expect(page.locator("#dataset-select")).toHaveValue("esc50"); // the user's choice is kept
  await expect(page.locator("#trace-request")).toContainText("Bird calls");
  await page.getByRole("tab", { name: "Graph", exact: true }).click();
  await expect(page.locator("#graph-canvas")).toContainText("Bird calls");
});

test("a replay shows on the model it was recorded for; a run for a model not loaded here offers no card link", async ({ page }) => {
  const other = { ...model, name: "other", label: "Other model" };
  let runModel = "other";
  const runs = () => [{ id: "recorded", request: { ...request, model: runModel }, status: "done", installed: card.bridge, seconds: 20 }];
  await setup(page, {
    models: (route) => route.fulfill({ json: [model, other] }),
    get runs() { return runs(); },
    job: () => finishedJob({ ...request, model: runModel }, [card], card.bridge),
  });
  await expect(page.locator("#model-label")).toHaveText("Fake model");
  await replayFirstRun(page);
  await expect(page.locator("#model-select")).toHaveValue("other");
  await expect(page.locator("#trace-status")).toHaveText("Sense installed");
  await page.getByRole("button", { name: "Open report card →" }).last().click();
  await expect(page.locator("#report-verdict")).toHaveText("Installed");
  await expect(page.locator("#report")).toContainText("80.0%");

  runModel = "ghost";
  await page.locator("#model-select").selectOption("fake");
  await replayFirstRun(page);
  await expect(page.locator("#trace-status")).toHaveText("Sense installed");
  await expect(page.locator("#model-select")).toHaveValue("fake");
  const message = page.locator(".msg--agent").last();
  await expect(message).toContainText("which this server cannot load");
  await expect(message.getByRole("button")).toHaveCount(0);
});

test("a failed model list shows its reason and Try again where it can be seen", async ({ page }) => {
  const { errors } = await setup(page, {
    models: (route, n) => (n === 1
      ? route.fulfill({ status: 500, json: { detail: "Could not list models: out of memory." } })
      : route.fulfill({ json: [model] })),
  });
  await expect(page.locator("#model-error")).toBeVisible();
  await expect(page.locator("#model-error")).toContainText("Could not list models: out of memory.");
  await page.locator("#model-error").getByRole("button", { name: "Try again" }).click();
  await expect(page.locator("#model-label")).toHaveText("Fake model");
  await expect(page.locator("#model-error")).toBeHidden();
  expect(errors).toEqual([]);
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
