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
    if (path === "/api/jobs/latest") return json({ job_id: options.latest ? options.latest(count(path)) : null });
    if (path === "/api/jobs/job-1/approval") {
      if (options.approval) return options.approval(route, req);
      return json({ detail: "Not Found" }, 404);
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
  expect(await page.locator("#file-input").getAttribute("accept")).toBeNull(); // any file: the server decides what it is
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

// ---------------------------------------------------------------- agent-written adapters
// The shapes agent.synth.run_synth and agent.jobs really send (agent/auto.py's adapter flow, tests/test_adapter_flow.py).
const meshGoal = "Tell which kind of chair this 3D model is";
const adapterDetect = (file = "chair.obj", format = "obj") => [
  event("detect", "start", `Detecting the contents of ${file}.`),
  event("detect", "ok", `${file} is a ${format} mesh. Since Fake model does not read 3d files and no specialist is wired for them, the agent will write an adapter: code that turns the file into something Fake model reads.`,
    { file, goal: meshGoal, modality: "3d", format, how: "content", flow: "adapter" }),
];
const meshDesign = (changes = {}) => ({
  modality: "3d", format: "obj", route: "text", description: "Counts the mesh's vertices and faces and measures its bounding box.",
  libraries: ["trimesh"], question: "What kind of chair is this 3D model?", label_hint: "chair styles", labels: ["armchair", "stool"],
  probe_plan: "dataset", probe_reason: "Public mesh datasets label furniture.", install: [], designed_modality: "3d-model", ...changes,
});
const adapterCode = 'import trimesh\n\ndef adapt(path):\n    mesh = trimesh.load(path, force="mesh")\n    return {"text": f"The mesh has {len(mesh.vertices)} vertices."}\n';
const adapterCard = {
  model: "fake", modality: "3d", dataset: "3d-dataset-2a8c4c7ee5", scope: "closed", bridge: "adapter:3d-obj", kind: "text", n: 40,
  chance: 0.5, accuracy: 0.9, controls: { swapped: 0.45, no_input: 0.5, native: 0.5, unseen_categories: 0.6 },
  control_chance: { unseen_categories: 0.3333 }, verdict: "installed", interval: [0.77, 0.96],
  reason: "adapter:3d-obj passed: it scored 90.0% against 50.0% chance, meets the 65.0% bar for a text bridge, 40.0 points above the 50.0% the model scored on the raw file.",
};
const probes = (purpose, source, labels, licence) => {
  const facts = { purpose, provenance: "dataset", hub_id: source, source, kind: "hub" };
  return [
    event("data", "start", "Searching the Hugging Face Hub and the web.", { ...facts, action: "search" }),
    event("data", "ok", "Found 1 candidate dataset.", { ...facts, action: "search", found: 1 }),
    event("data", "start", `Trying ${source}.`, { ...facts, action: "candidate", licence, url: `https://huggingface.co/datasets/${source}` }),
    event("data", "start", `Fetching labelled files from ${source}.`, { ...facts, action: "fetch" }),
    event("data", "ok", `Fetched files from ${source}.`, { ...facts, action: "fetch", counts: Object.fromEntries(labels.map((l) => [l, 5])) }),
    event("data", "ok", `Using ${source}: ${labels.length} labels.`, { ...facts, action: "candidate", labels, licence, url: `https://huggingface.co/datasets/${source}` }),
  ];
};
/** A finished adapter run: detect, design, build (one failed attempt, then the code), probe files from a public
    dataset, unseen categories from another one (or the note that they were not measured), verify ... done. */
function adapterJob({ unseen = true, card = adapterCard } = {}) {
  const summary = {
    name: card.dataset, provenance: "dataset", source: "https://huggingface.co/datasets/shapes/chairs", licence: "cc-by-4.0",
    labels: ["armchair", "stool"], counts: { armchair: 20, stool: 20 }, n: 40, split: { fit: 24, dev: 8, test: 8 }, meta: {}, label_leaks: [],
    fetched: { hub_id: "shapes/chairs", url: "https://huggingface.co/datasets/shapes/chairs", title: "shapes/chairs", licence: "cc-by-4.0", licence_source: "the Hub dataset card" },
  };
  const note = "Unseen categories not measured: the Hub search found no other open dataset with other labels.";
  return {
    request: { model: "fake", dataset: card.dataset, modality: "3d", scope: "closed" }, status: "done", error: null, installed: card.bridge, cards: [card],
    events: [
      ...adapterDetect(),
      event("design", "start", "Designing an adapter that lets Fake model sense chair.obj."),
      event("design", "ok", "A text adapter for 3d (obj) files: Counts the mesh's vertices and faces and measures its bounding box.", meshDesign()),
      event("build", "start", "Writing adapt(path) with trimesh and running it on chair.obj in the sandbox.", { libraries: ["trimesh"], sandboxed: true, max_attempts: 3 }),
      event("build", "info", "Attempt 1 failed: ModuleNotFoundError: No module named 'pygltf' Asking the LLM for a repair.",
        { attempt: 1, error: { reason: "ModuleNotFoundError: No module named 'pygltf'", cause: "run" } }),
      event("build", "ok", "The adapter ran on chair.obj after 2 attempts (0.4 s).", {
        attempts: 2, libraries: ["trimesh"], code_sha256: "7fef4b7682f585a4d03c29d10698ec38", code: adapterCode,
        preview: "The mesh has 412 vertices.", output_kind: "text", notes: [], seconds: 0.42 }),
      ...probes("probes", "shapes/chairs", ["armchair", "stool"], "cc-by-4.0"),
      event("data", "ok", "Using 40 dataset probe files in 2 labels (armchair, stool); all are held out.", summary),
      ...(unseen === "rejected"
        ? [ // the second search tries a candidate and rejects it: the data step's own outcome stands
          event("data", "start", "Searching for labelled files of other categories.", { purpose: "unseen", provenance: "dataset", action: "search" }),
          event("data", "ok", "Found 1 candidate dataset.", { purpose: "unseen", provenance: "dataset", action: "search", found: 1 }),
          event("data", "start", "Trying shapes/captions.", { purpose: "unseen", provenance: "dataset", action: "candidate", hub_id: "shapes/captions", source: "shapes/captions" }),
          event("data", "fail", "shapes/captions cannot be used: it has no mesh column.", { purpose: "unseen", provenance: "dataset", action: "candidate", hub_id: "shapes/captions", source: "shapes/captions", reason: "no mesh column" }),
          event("data", "info", note, { purpose: "unseen", reason: "no candidate dataset gave usable probes" })]
        : unseen
        ? [...probes("unseen", "other/benches", ["bench", "sofa", "table"], "mit"),
          event("data", "info", "Also scoring 15 files of 3 other categories (bench, sofa, table), which the adapter was not built around.", {
            unseen: { name: "3d-unseen-1", source: "https://huggingface.co/datasets/other/benches", labels: ["bench", "sofa", "table"], per_label: { bench: 5, sofa: 5, table: 5 }, n: 15 } })]
        : [event("data", "info", note, { purpose: "unseen", reason: "the Hub search found no other open dataset with other labels" })]),
      event("verify", "start", "Scoring adapter:3d-obj on dataset probe files.", { bridge: card.bridge, n_test: 40, native: true, unseen: unseen ? "3d-unseen-1" : null }),
      event("verify", "ok", "adapter:3d-obj scored 90.0% on 40 held-out probe files.", { card: { ...card, verdict: "candidate" }, controls: card.controls, control_chance: card.control_chance, seconds: 1.0 }),
      event("decide", "start", "Checking adapter:3d-obj against the bars for a text bridge."),
      event("decide", "ok", card.reason, { install: card.bridge, accepted: true, card }),
      event("install", "start", `Installing ${card.bridge}.`, { bridge: card.bridge }),
      event("install", "ok", `Installed ${card.bridge}; Fake model can now take 3d files in chat.`, { bridge: card.bridge }),
      event("done", "ok", `Fake model can now take 3d files through ${card.bridge}: it scored 90.0% on 40 probe files, against 50.0% chance.`, { installed: card.bridge, needs_user: false }),
    ],
  };
}
const adapterSense = { bridge: "adapter:3d-glb", kind: "text", modality: "3d", dataset: "3d-dataset-2a8c4c7ee5", card: { ...adapterCard, bridge: "adapter:3d-glb" } };

/** An install request: agent.installer's plan (one package with a caution, a dependency, one already installed). */
const installPlan = {
  requested: [{ name: "pygltflib", version: null, why: "reads glTF scenes" }],
  packages: [
    { name: "pygltflib", version: "1.16.2", verdict: "caution", reasons: ["fewer than 3 releases", "first release under 90 days ago"], facts: { already_installed: false, requested: true, via: [] } },
    { name: "dataclasses-json", version: "0.6.7", verdict: "ok", reasons: [], facts: { already_installed: false, requested: false, via: ["pygltflib"] } },
    { name: "numpy", version: "2.1.3", verdict: "ok", reasons: ["already installed in the environment at 2.1.3; not vetted or reinstalled"], facts: { already_installed: true, requested: false, via: ["pygltflib"] } },
  ],
  verdict: "caution", requirements: "", created: "2026-11-15T03:00:00+00:00", problems: [], target: {},
  summary: "Caution, pygltflib needs a look before approval (2 new packages for the toolbox, 1 already installed): pygltflib: fewer than 3 releases.",
};
const pendingInstall = {
  id: "req-1", kind: "install", timeout: 600, asked: 0.4,
  payload: { plan: installPlan, summary: installPlan.summary, review: "A small glTF reader; its release history is short, so check its home page." },
};
/** A live adapter run waiting on the install request until it is answered (answer: null, true or false), then
    going on to the build step. */
function waitingJob(answer) {
  const requested = installPlan.requested;
  const record = { requested, verdict: "caution", summary: installPlan.summary };
  const answered = answer === null ? [] : answer
    ? [event("approval", "ok", "Approved.", { id: "req-1", kind: "install", answer: true, reason: "approved" }),
      event("install_request", "ok", "Installed pygltflib 1.16.2 into the toolbox; the adapter may import it.",
        { ...record, approved: true, installed: [{ name: "pygltflib", version: "1.16.2", sha256: "0".repeat(64) }], note: "" }),
      event("build", "start", "Writing adapt(path) with trimesh, pygltflib and running it on chair.glb in the sandbox.")]
    : [event("approval", "fail", "Declined.", { id: "req-1", kind: "install", answer: false, reason: "declined" }),
      event("install_request", "info", "Not installing pygltflib: the request was declined or not answered. Writing the adapter with the toolbox as it is.",
        { ...record, approved: false, installed: [], note: "declined or not answered" }),
      event("build", "start", "Writing adapt(path) with trimesh and running it on chair.glb in the sandbox.")];
  return {
    request: autoRequest, status: "running", error: null, installed: null, cards: [],
    events: [
      ...adapterDetect("chair.glb", "glb"),
      event("design", "start", "Designing an adapter that lets Fake model sense chair.glb."),
      event("design", "ok", "A text adapter for 3d (glb) files.", meshDesign({ format: "glb", install: [{ name: "pygltflib", why: "reads glTF scenes" }] })),
      event("install_request", "start", "The design asks for pygltflib, which the toolbox lacks: checking it before asking you.", { requested }),
      event("approval", "start", `Waiting for your approval (install): ${installPlan.summary}`, { id: "req-1", kind: "install", payload: pendingInstall.payload, timeout: 600 }),
      ...answered,
    ],
    pending_approval: answer === null ? pendingInstall : null,
  };
}

/** Start an agent run from New sense with a file the browser has no special type for. */
async function runOnMesh(page, name = "chair.glb") {
  await page.locator("#new-sense-button").click();
  await page.locator("#fit-goal").fill(meshGoal);
  await page.locator("#fit-file").setInputFiles({ name, mimeType: "model/gltf-binary", buffer: Buffer.from("glTF") });
  await page.locator("#run-button").click();
}

test("an install request shows the checks apart from the LLM's opinion; Approve by keyboard answers it", async ({ page }) => {
  let answer = null;
  const { calls, errors } = await setup(page, {
    senses: [],
    job: () => waitingJob(answer),
    approval: (route, req) => {
      answer = JSON.parse(req.postData()).approve;
      return route.fulfill({ json: { answered: true, approve: answer } });
    },
  });
  // Any file starts a run: the New sense window has no type filter and says so.
  expect(await page.locator("#fit-file").getAttribute("accept")).toBeNull();
  await page.locator("#new-sense-button").click();
  await expect(page.locator("#fit-drop")).toContainText("3D models, molecules, coordinates");
  await page.keyboard.press("Escape");
  await runOnMesh(page);
  expect(calls.find((c) => c.path === "/api/auto").body).toContain('filename="chair.glb"');

  const card = page.getByRole("region", { name: "Install pygltflib?" });
  await expect(card).toBeVisible();
  await expect(card).toContainText(installPlan.summary);
  await expect(card).toContainText("checks: caution");
  await expect(card.locator("tbody tr")).toHaveCount(2); // the package to install and its dependency
  await expect(card.locator("tbody tr").first()).toContainText(["pygltflib", "requested", "1.16.2", "caution", "fewer than 3 releases"].join(""));
  await expect(card.locator("tbody tr").nth(1)).toContainText("needed by pygltflib");
  await expect(card.locator(".approval-have summary")).toHaveText("1 already installed: not vetted or installed again");
  await expect(card.locator(".approval-have li")).toBeHidden(); // collapsed
  await expect(card.locator(".approval-review")).toContainText("an LLM's review, not one of the checks");
  await expect(card.locator(".approval-review")).toContainText("release history is short");
  await expect(card).toContainText("No answer within 10 minutes counts as declined.");
  await expect(page.locator(".step").filter({ hasText: "Your approval" }).locator(".step-state")).toHaveText("Waiting");
  // No event comes while the run waits, but its clock goes on (every event of this fixture is at 1 s).
  await expect.poll(() => page.locator("#trace-elapsed").textContent(), { timeout: 5000 }).toMatch(/^(1\.[5-9]|[2-9]\.\d|\d{2,}\.\d) s$/);
  await expect(page.locator(".step-title")).toHaveText([
    "Detect the input", "Design an adapter", "Check requested packages", "Your approval", "Write and run the adapter",
    "Find labelled probe files", "Verify on probe files", "Install or refuse", "Install",
  ]);

  // The chat says the agent is waiting, and its link takes the keyboard to the card.
  const note = page.locator(".msg--agent").first(); // the only agent message while the run waits
  await expect(note).toContainText("asks to install pygltflib");
  await page.getByRole("tab", { name: "Report", exact: true }).click();
  await note.getByRole("button", { name: "Review the request →" }).focus();
  await page.keyboard.press("Enter");
  await expect(card.locator(".approval-title")).toBeFocused();
  await page.keyboard.press("Tab"); // the folded list of installed packages
  await page.keyboard.press("Tab");
  await expect(card.getByRole("button", { name: "Approve" })).toBeFocused();
  await page.keyboard.press("Enter");

  await expect.poll(() => calls.filter((c) => c.path === "/api/jobs/job-1/approval").length).toBe(1);
  expect(JSON.parse(calls.find((c) => c.path === "/api/jobs/job-1/approval").body)).toEqual({ id: "req-1", approve: true });
  await expect(card.locator(".approval-outcome")).toHaveText(/^Approved\./);
  await expect(card.locator(".approval-outcome")).toBeFocused();
  await expect(note).toContainText("You approved installing pygltflib.");
  // Once the run has moved on to the build step, the card goes; the steps keep the outcome, and the keyboard
  // stays in the run (at its heading) rather than falling back to the start of the page.
  await expect(page.locator("#approval .approval")).toHaveCount(0);
  await expect(page.locator("#trace-heading")).toBeFocused();
  await expect(page.locator(".step").filter({ hasText: "Your approval" })).toHaveClass(/step--done/);
  await expect(page.locator(".step").filter({ hasText: "Check requested packages" })).toContainText("pygltflib 1.16.2");
  expect(calls.filter((c) => c.path === "/api/jobs/job-1/approval")).toHaveLength(1);
  expect(errors).toEqual([]);
});

test("Decline answers the request; the step shows it was declined, not failed", async ({ page }) => {
  let answer = null;
  const { calls, errors } = await setup(page, {
    senses: [],
    job: () => waitingJob(answer),
    approval: (route, req) => {
      answer = JSON.parse(req.postData()).approve;
      return route.fulfill({ json: { answered: true, approve: answer } });
    },
  });
  await runOnMesh(page);
  const card = page.getByRole("region", { name: "Install pygltflib?" });
  await card.getByRole("button", { name: "Decline" }).click();
  await expect(card.locator(".approval-outcome")).toHaveText(/^Declined\. Nothing is installed/);
  expect(JSON.parse(calls.find((c) => c.path === "/api/jobs/job-1/approval").body)).toEqual({ id: "req-1", approve: false });
  await expect(card.getByRole("button", { name: /^(Approve|Decline)$/ })).toHaveCount(0);
  await expect(page.locator("#approval .approval")).toHaveCount(0);
  const approval = page.locator(".step").filter({ hasText: "Your approval" });
  await expect(approval).toHaveClass(/step--declined/);
  await expect(approval.locator(".step-state")).toHaveText("Declined");
  await expect(page.locator(".step").filter({ hasText: "Check requested packages" })).toHaveClass(/step--declined/);
  await expect(page.locator(".msg--agent").filter({ hasText: "You declined installing pygltflib" })).toBeVisible();
  expect(errors).toEqual([]);
});

test("a tab that did not start the run finds it waiting for approval and can answer it", async ({ page }) => {
  let answer = null;
  const { calls, errors } = await setup(page, {
    senses: [],
    latest: () => "job-1", // GET /api/jobs/latest: the run another tab started
    job: () => waitingJob(answer),
    approval: (route, req) => {
      answer = JSON.parse(req.postData()).approve;
      return route.fulfill({ json: { answered: true, approve: answer } });
    },
    runs: [{ id: "early", request: autoRequest, status: "done", installed: null, seconds: 1 }],
  });
  expect(calls.filter((c) => c.path === "/api/auto")).toHaveLength(0); // nothing was started from this tab
  const card = page.getByRole("region", { name: "Install pygltflib?" });
  await expect(card).toBeVisible();
  await expect(page.locator("#trace-status")).toHaveText("Running");
  await card.getByRole("button", { name: "Decline" }).click();
  await expect.poll(() => answer).toBe(false);
  // A recorded run that ended before its data step says so, rather than naming no modality.
  await page.locator("#runs-button").click();
  await expect(page.locator(".run-item-title").filter({ hasText: "ended before" })).toHaveText("Agent run · ended before it chose labelled files");
  expect(errors).toEqual([]);
});

test("an answer to a request that is no longer open says so and changes nothing", async ({ page }) => {
  await setup(page, {
    senses: [],
    job: () => waitingJob(null),
    approval: (route) => route.fulfill({ status: 409, json: { detail: "Request req-1 of job job-1 is not open: it timed out." } }),
  });
  await runOnMesh(page);
  const card = page.getByRole("region", { name: "Install pygltflib?" });
  await card.getByRole("button", { name: "Approve" }).click();
  await expect(card.getByRole("alert")).toContainText("is not open: it timed out.");
  await expect(card.getByRole("button", { name: "Approve" })).toBeDisabled();
  await expect(card.locator(".approval-outcome")).toHaveCount(0);
});

test("an adapter run shows its design, attempts and code, where the probe files came from, and both honesty controls", async ({ page }) => {
  let scenario = adapterJob();
  const { errors } = await setup(page, {
    senses: [{ bridge: adapterCard.bridge, kind: "text", modality: "3d", dataset: adapterCard.dataset, card: adapterCard }],
    job: () => scenario,
  });
  await runOnMesh(page, "chair.obj");
  await expect(page.locator("#trace-status")).toHaveText("Sense installed");
  await page.getByRole("tab", { name: "Run", exact: true }).click();
  await expect(page.locator(".step-title")).toHaveText([
    "Detect the input", "Design an adapter", "Write and run the adapter", "Find labelled probe files",
    "Verify on probe files", "Install or refuse", "Install",
  ]);
  const design = page.locator(".step").filter({ hasText: "Design an adapter" });
  await expect(design.locator(".facts")).toContainText("Reads3D model (obj)");
  await expect(design.locator(".facts")).toContainText("Routetext, which the model reads");
  await expect(design.locator(".facts")).toContainText("ExtractsCounts the mesh's vertices and faces");
  await expect(design.locator(".facts")).toContainText("Librariestrimesh");

  const build = page.locator(".step").filter({ hasText: "Write and run the adapter" });
  await expect(build.locator(".facts")).toContainText("Attempts2");
  await expect(build.locator(".attempts li")).toHaveText(["1: ModuleNotFoundError: No module named 'pygltf'"]);
  await expect(build.locator(".facts")).toContainText("Ran in0.4 s");
  const code = build.locator("details.code-block");
  await expect(code.locator("summary")).toHaveText("The adapter's code · 5 lines · sha256 7fef4b7682f5");
  await expect(code.locator("pre")).toBeHidden();
  await code.locator("summary").click();
  await expect(code.locator("pre")).toContainText("def adapt(path):");

  const data = page.locator(".step").filter({ hasText: "Find labelled probe files" });
  await expect(data.locator(".facts")).toContainText("Provenancea public dataset");
  await expect(data.getByRole("link", { name: "shapes/chairs" })).toHaveAttribute("href", "https://huggingface.co/datasets/shapes/chairs");
  await expect(data.locator(".facts")).toContainText("Licencecc-by-4.0 (from the Hub dataset card)");
  await expect(data.locator(".facts")).toContainText("Labelsarmchair 20, stool 20 · 40 files");
  await expect(data.getByRole("link", { name: "other/benches" })).toHaveAttribute("href", "https://huggingface.co/datasets/other/benches");
  await expect(data.locator(".facts")).toContainText("bench 5, sofa 5, table 5 · mit");

  // The report card: the native and unseen-categories controls, each against its own chance, and the rule.
  await page.getByRole("tab", { name: "Report", exact: true }).click();
  await expect(page.locator("#report-verdict")).toHaveText("Installed");
  await expect(page.locator("#report .report-sub")).toHaveText("3D model → Fake model · agent-written adapter (text) · probe files: shapes/chairs");
  const native = page.locator(".metric").filter({ hasText: "Model reads the raw file" });
  await expect(native.locator(".metric-value")).toHaveText("50.0%");
  await expect(native.locator(".tick")).toHaveAttribute("style", "left: 50%");
  const unseen = page.locator(".metric").filter({ hasText: "Unseen categories" });
  await expect(unseen.locator(".metric-value")).toHaveText("60.0%");
  await expect(unseen.locator(".metric-chance")).toHaveText("own chance 33.3%");
  await expect(unseen.locator(".tick")).toHaveAttribute("style", "left: 33.33%");
  await expect(page.locator("#report")).toContainText("Install rule for adapters");
  await expect(page.locator("#report")).toContainText("at least 10 points above the model reading the raw file");
  await expect(page.locator(".msg--agent").last()).toContainText("3D model installed");
  await expect(page.locator(".msg--agent").last()).toContainText("probe files: shapes/chairs");
  await expect(page.locator("#senses-list .sense-title")).toHaveText(["Text", "3D model"]);
  await expect(page.locator("#senses-list")).toContainText("Agent-written adapter (text)");
  await page.getByRole("tab", { name: "Graph", exact: true }).click();
  await expect(page.locator(".node").filter({ hasText: "Adapter (text)" })).toContainText("Counts the mesh's vertices");
  await expect(page.locator(".node").filter({ hasText: "shapes/chairs" })).toContainText("2 labels · 40 dataset probe files");

  // A run whose unseen categories could not be found says so in the data step.
  scenario = adapterJob({ unseen: false, card: { ...adapterCard, controls: { swapped: 0.45, no_input: 0.5, native: 0.5 }, control_chance: {} } });
  await runOnMesh(page, "chair.obj");
  await expect(page.locator("#trace-status")).toHaveText("Sense installed");
  await page.getByRole("tab", { name: "Run", exact: true }).click();
  await expect(page.locator(".step").filter({ hasText: "Find labelled probe files" }).locator(".fact-warn"))
    .toHaveText("Unseen categories not measured: the Hub search found no other open dataset with other labels.");
  await page.getByRole("tab", { name: "Report", exact: true }).click();
  await expect(page.locator(".metric").filter({ hasText: "Unseen categories" })).toHaveCount(0);
  await expect(page.locator("#report .metric-chance")).toHaveCount(0);

  // A candidate the unseen-categories search rejects is not a failure of the data step, which found its probe files.
  scenario = adapterJob({ unseen: "rejected", card: { ...adapterCard, controls: { swapped: 0.45, no_input: 0.5, native: 0.5 }, control_chance: {} } });
  await runOnMesh(page, "chair.obj");
  await expect(page.locator("#trace-status")).toHaveText("Sense installed");
  await page.getByRole("tab", { name: "Run", exact: true }).click();
  const probeStep = page.locator(".step").filter({ hasText: "Find labelled probe files" });
  await expect(probeStep).toHaveClass(/step--done/);
  await expect(probeStep.locator(".step-state")).toHaveText("Done");
  await expect(probeStep.locator(".fact-warn")).toContainText("Unseen categories not measured");
  await expect(page.locator(".step--refused")).toHaveCount(0);
  // The Senses list names where an adapter's probe files came from, never the hashed name of its probe set.
  await expect(page.locator("#senses-list")).toContainText("Agent-written adapter (text) · probe files from a public dataset");
  await expect(page.locator("#senses-list")).not.toContainText("3d-dataset-");
  expect(errors).toEqual([]);
});

test("chat takes any file: a .glb goes through its adapter sense, and a file the adapter cannot read is reported", async ({ page }) => {
  const viaAdapter = { kind: "text", bridge: "adapter:3d-glb", source: "chair.glb", detail: 'text from the agent-written adapter: "The mesh has 412 vertices."' };
  const failed = { kind: "none", bridge: "adapter:3d-glb", source: "broken.glb", detail: "the adapter adapter:3d-glb failed on this file: ValueError: not a glTF file" };
  const { calls, errors } = await setup(page, {
    senses: [adapterSense],
    chat: (route) => route.fulfill({ json: { reply: "An armchair.", used: [viaAdapter, failed] } }),
  });
  expect(await page.locator("#file-input").getAttribute("accept")).toBeNull();
  await expect(page.locator(".drop-sub")).toContainText("Any file: audio, video, 3D models, molecules, coordinates or anything else.");
  await expect(page.locator("#sense-chips .chip-label")).toHaveText(["Text", "3D model"]);
  await expect(page.locator("#senses-list .sense-title")).toHaveText(["Text", "3D model"]);
  await page.locator("#file-input").setInputFiles([
    { name: "chair.glb", mimeType: "model/gltf-binary", buffer: Buffer.from("glTF") },
    { name: "broken.glb", mimeType: "application/octet-stream", buffer: Buffer.from("nope") },
  ]);
  await expect(page.locator("#attachments .attachment-meta")).toHaveText(["3D model · upload", "3D model · upload"]);
  await send(page, "What kind of chair is this?");
  const body = calls.filter((c) => c.path === "/api/chat").at(-1).body;
  expect(body).toContain('filename="chair.glb"');
  await expect(page.locator(".used-item").first()).toHaveText(`chair.glb: Read via adapter:3d-glb · ${viaAdapter.detail}`);
  await expect(page.locator(".used-item--warn")).toContainText("broken.glb not received: the adapter adapter:3d-glb failed on this file");
  await expect(page.locator(".msg--model .name-line-tags .tag")).toHaveText(["Read via adapter", "File not received"]);
  await expect(page.locator(".msg--model")).not.toContainText("Heard via text bridge");
  await expect(page.locator(".msg--user .attachment-meta")).toHaveText(["3D model · upload", "3D model · upload"]);
  // Its report card, opened from the Senses list with no run at hand, names where the probe files came from.
  await page.locator("#senses-list .sense-text").filter({ hasText: "3D model" }).click();
  await expect(page.locator("#report .report-sub")).toHaveText("3D model → Fake model · agent-written adapter (text) · probe files from a public dataset");
  expect(errors).toEqual([]);
});
