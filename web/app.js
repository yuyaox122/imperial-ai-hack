"use strict";

/*
 * Sense harness interface: senses panel, chat and agent trace.
 * Talks only to the HTTP API documented in server/app.py. No framework, no build step.
 */

const POLL_MS = 700;
const MAX_POLL_RETRIES = 5;

const CONTROL_LABELS = {
  swapped: "Swapped audio",
  no_input: "No audio",
  lookup: "Nearest example",
  text_bridge: "Text bridge",
  unseen_classes: "Unseen classes",
};
const KIND_LABELS = { native: "Native", text: "Text", tool: "Tool", image: "Image", vector: "Vector" };
const VERDICT_LABELS = { installed: "Installed", rejected: "Rejected", candidate: "Candidate" };
const SCOPE_LABELS = { closed: "These categories only", open: "Any sound" };
const SCOPE_HINTS = {
  closed: "The agent checks that the model tells these labels apart on held-out clips.",
  open: "The agent also tests labels it never fitted on, and refuses a bridge that only knows the labels it was shown.",
};
const STATUS_WORDS = { start: "started", ok: "succeeded", fail: "failed", info: "note" };
const JOB_WORDS = { running: "Running", done: "Finished", failed: "Failed", lost: "Disconnected" };

const state = {
  models: [],
  model: "",
  datasets: [],
  dataset: "",
  samples: [],
  sampleRefs: new Map(), // sample id -> {item, play, attach}
  attachedSamples: new Map(), // sample id -> sample
  files: [],
  playing: null, // sample id
  job: null,
  starting: false,
  chatting: false,
};

const player = new Audio();

// ---------------------------------------------------------------- small helpers

const $ = (id) => document.getElementById(id);

/** Create an element. attrs: "class", "text", "on<event>" handlers, other keys become attributes. */
function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : String(child));
  }
  return el;
}

const ICONS = {
  play: { d: "M5.5 3.5v9l7-4.5z", fill: true },
  stop: { d: "M4.5 4.5h7v7h-7z", fill: true },
  check: { d: "M3.5 8.5l3 3 6-7" },
  cross: { d: "M4.5 4.5l7 7M11.5 4.5l-7 7" },
  warn: { d: "M8 2.5l6.2 11H1.8zM8 6.5v3.2M8 11.8v.1" },
};

function icon(name, size = 12) {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 16 16");
  svg.setAttribute("width", String(size));
  svg.setAttribute("height", String(size));
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("class", "icon");
  const path = document.createElementNS(ns, "path");
  path.setAttribute("d", ICONS[name].d);
  if (ICONS[name].fill) {
    path.setAttribute("fill", "currentColor");
  } else {
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", "currentColor");
    path.setAttribute("stroke-width", "2");
    path.setAttribute("stroke-linecap", "round");
    path.setAttribute("stroke-linejoin", "round");
  }
  svg.append(path);
  return svg;
}

const store = {
  get(area, key) {
    try { return window[area].getItem(key); } catch { return null; }
  },
  set(area, key, value) {
    try {
      if (value === null) window[area].removeItem(key);
      else window[area].setItem(key, value);
    } catch {
      // Storage is unavailable (private mode): nothing to remember.
    }
  },
};

const enc = encodeURIComponent;
const toNumber = (x) => (typeof x === "number" && Number.isFinite(x) ? x : null);
const clamp01 = (x) => Math.min(1, Math.max(0, x));
const pct = (x) => `${(x * 100).toFixed(1)}%`;
const seconds = (t) => (toNumber(t) === null ? "" : `${t.toFixed(1)} s`);
const capitalise = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : "");
const humanise = (key) => capitalise(String(key).replace(/_/g, " "));
const kindLabel = (kind) => KIND_LABELS[kind] || capitalise(kind || "unknown");

const currentModel = () => state.models.find((m) => m.name === state.model) || null;
const currentDataset = () => state.datasets.find((d) => d.name === state.dataset) || null;
const modelLabel = (name) => (state.models.find((m) => m.name === name) || {}).label || name || "the model";
const datasetTitle = (name) => (state.datasets.find((d) => d.name === name) || {}).title || name || "";
const jobRunning = () => Boolean(state.job && state.job.status === "running");

// ---------------------------------------------------------------- API

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

/** The server's "detail" as one readable line (FastAPI validation errors arrive as a list). */
function detailText(body, status) {
  const detail = body && body.detail;
  if (typeof detail === "string" && detail.trim()) return detail.trim();
  if (Array.isArray(detail) && detail.length) {
    return detail
      .map((d) => {
        if (!d || typeof d !== "object") return String(d);
        const where = Array.isArray(d.loc) ? d.loc.filter((part) => part !== "body").join(".") : "";
        return where ? `${where}: ${d.msg}` : String(d.msg);
      })
      .join("; ");
  }
  return `The server answered with status ${status} and no explanation. Check the server log, then try again.`;
}

/** Fetch JSON from the API. Throws ApiError with the server's detail on any non-2xx answer. */
async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new ApiError("Could not reach the server. Check that it is running, then try again.", 0);
  }
  const isJson = (response.headers.get("content-type") || "").includes("application/json");
  let body = null;
  if (isJson) {
    body = await response.json().catch(() => null);
  } else if (!response.ok) {
    const text = await response.text().catch(() => "");
    body = text.trim() ? { detail: text.trim().slice(0, 300) } : null;
  }
  if (!response.ok) throw new ApiError(detailText(body, response.status), response.status);
  return body;
}

/** Show an error (or clear it with null) in an error box, with an optional "Try again" action. */
function showError(el, err, retry) {
  el.replaceChildren();
  if (!err) {
    el.hidden = true;
    return;
  }
  el.append(h("span", { text: err.message || String(err) }));
  if (retry) el.append(h("button", { type: "button", class: "link-button", text: "Try again", onclick: retry }));
  el.hidden = false;
}

/** Enable or disable the action buttons from the current state. */
function syncControls() {
  const model = currentModel();
  const usable = Boolean(model && model.available);
  const running = jobRunning();

  const run = $("run-button");
  run.disabled = !usable || !state.dataset || state.starting || running;
  run.textContent = running ? "Agent running…" : state.starting ? "Starting…" : "Run agent";
  if (state.starting || running) run.setAttribute("aria-busy", "true");
  else run.removeAttribute("aria-busy");

  const send = $("send-button");
  send.disabled = !usable || state.chatting;
  send.textContent = state.chatting ? "Waiting…" : "Send";
  if (state.chatting) send.setAttribute("aria-busy", "true");
  else send.removeAttribute("aria-busy");

  $("chat-model").textContent = usable ? model.label : "";
}

// ---------------------------------------------------------------- models

async function loadModels() {
  const select = $("model-select");
  showError($("model-error"), null);
  select.disabled = true;
  select.replaceChildren(h("option", { value: "", text: "Loading models…" }));
  try {
    const models = await api("/api/models");
    state.models = Array.isArray(models) ? models : [];
  } catch (err) {
    state.models = [];
    select.replaceChildren(h("option", { value: "", text: "Models could not be loaded" }));
    showError($("model-error"), err, loadModels);
    setModel("");
    return;
  }
  renderModelOptions();
  if (state.job) renderTraceHead();
}

function renderModelOptions() {
  const select = $("model-select");
  const usable = state.models.filter((m) => m.available);
  const options = state.models.map((m) =>
    h("option", {
      value: m.name,
      disabled: !m.available,
      text: m.available ? m.label : `${m.label} (unavailable${m.note ? `: ${m.note}` : ""})`,
    }),
  );
  if (!usable.length) options.unshift(h("option", { value: "", text: "No model is available" }));
  select.replaceChildren(...options);
  select.disabled = usable.length === 0;
  const saved = store.get("localStorage", "sense.model");
  const pick = usable.find((m) => m.name === saved) || usable[0];
  select.value = pick ? pick.name : "";
  setModel(pick ? pick.name : "");
}

function setModel(name) {
  const changed = name !== state.model;
  state.model = name;
  if (name) store.set("localStorage", "sense.model", name);
  renderModelInfo();
  syncControls();
  if (changed && name) noteInThread(`Now talking to ${modelLabel(name)}.`);
  loadSenses();
}

function renderModelInfo() {
  const info = $("model-info");
  const model = currentModel();
  if (model) {
    const native = (model.native || []).join(", ") || "no input";
    const weights = model.open_weights
      ? "Open weights, so the vector bridge can be tried."
      : "Closed weights, so the vector bridge is not possible.";
    info.textContent = `Accepts ${native} natively. ${weights}${model.note ? ` ${model.note}` : ""}`;
  } else if (state.models.length) {
    const notes = state.models.map((m) => `${m.label}: ${m.note || "unavailable"}.`).join(" ");
    info.textContent = `No model can be loaded right now. ${notes}`;
  } else {
    info.textContent = "";
  }
}

// ---------------------------------------------------------------- senses

async function loadSenses() {
  const list = $("senses-list");
  const model = state.model;
  showError($("senses-error"), null);
  if (!model) {
    list.replaceChildren(h("li", { class: "empty", text: "Choose a base model to see its senses." }));
    return;
  }
  list.setAttribute("aria-busy", "true");
  try {
    const senses = await api(`/api/senses?model=${enc(model)}`);
    if (model !== state.model) return; // the user picked another model meanwhile
    renderSenses(Array.isArray(senses) ? senses : []);
  } catch (err) {
    if (model !== state.model) return;
    list.replaceChildren();
    showError($("senses-error"), err, loadSenses);
  } finally {
    list.removeAttribute("aria-busy");
  }
}

function renderSenses(senses) {
  const list = $("senses-list");
  if (!senses.length) {
    list.replaceChildren(
      h("li", { class: "empty", text: `${modelLabel(state.model)} has no installed senses yet. Run the agent below to give it one.` }),
    );
    return;
  }
  list.replaceChildren(...senses.map(senseItem));
}

function senseItem(sense) {
  const card = {
    bridge: sense.bridge,
    kind: sense.kind,
    modality: sense.modality,
    dataset: sense.dataset,
    verdict: "installed",
    controls: {},
    ...(sense.card || {}),
  };
  const errorEl = h("div", { class: "error", role: "alert", hidden: true });
  const remove = h("button", {
    type: "button",
    class: "button button--small button--ghost",
    text: "Remove",
    "aria-label": `Remove the ${sense.modality || ""} sense ${sense.bridge || ""}`.replace(/\s+/g, " "),
  });
  remove.addEventListener("click", () => removeSense(sense, remove, errorEl));
  return h("li", { class: "sense" }, renderCard(card, remove), errorEl);
}

async function removeSense(sense, button, errorEl) {
  const model = state.model;
  button.disabled = true;
  button.setAttribute("aria-busy", "true");
  button.textContent = "Removing…";
  showError(errorEl, null);
  try {
    await api(`/api/senses?model=${enc(model)}&modality=${enc(sense.modality)}`, { method: "DELETE" });
    await loadSenses();
  } catch (err) {
    showError(errorEl, err);
    button.disabled = false;
    button.removeAttribute("aria-busy");
    button.textContent = "Remove";
  }
}

// ---------------------------------------------------------------- report card

/** One report card: accuracy as a bar, each control as a thinner bar beneath it, chance as a dashed line. */
function renderCard(card, action) {
  const verdict = VERDICT_LABELS[card.verdict] ? card.verdict : "candidate";
  const chance = toNumber(card.chance);
  const n = toNumber(card.n);
  const meta = [
    capitalise(card.modality),
    datasetTitle(card.dataset),
    SCOPE_LABELS[card.scope] ? SCOPE_LABELS[card.scope].toLowerCase() : null,
    n !== null ? `${n} held-out items` : null,
    chance !== null ? `chance ${pct(chance)} (dashed line)` : null,
  ]
    .filter(Boolean)
    .join(" · ");
  const controls = Object.entries(card.controls || {});
  return h(
    "article",
    { class: `card card--${verdict}` },
    h(
      "header",
      { class: "card-head" },
      h("span", { class: "kind", text: kindLabel(card.kind) }),
      h("span", { class: "card-title", text: card.bridge || "bridge", title: card.bridge || "" }),
      h("span", { class: `badge badge--${verdict}`, text: VERDICT_LABELS[verdict] }),
      action || null,
    ),
    meta ? h("p", { class: "card-meta", text: meta }) : null,
    h(
      "div",
      { class: "bars" },
      barRow("Accuracy", card.accuracy, "main", chance),
      controls.map(([key, value]) => barRow(CONTROL_LABELS[key] || humanise(key), value, "control", chance)),
    ),
    card.reason ? h("p", { class: "card-reason", text: card.reason }) : null,
  );
}

function barRow(label, value, variant, chance) {
  const v = toNumber(value);
  const track = h(
    "div",
    { class: "track", "aria-hidden": "true" },
    h("div", { class: "fill", style: `width: ${v === null ? 0 : clamp01(v) * 100}%` }),
    chance !== null ? h("div", { class: "chance", style: `left: ${clamp01(chance) * 100}%` }) : null,
  );
  return h(
    "div",
    { class: `bar bar--${variant}` },
    h("span", { class: "bar-label", text: label, title: label }),
    track,
    h("span", { class: "bar-value", text: v === null ? "n/a" : pct(v) }),
  );
}

// ---------------------------------------------------------------- datasets and samples

async function loadDatasets() {
  const select = $("dataset-select");
  showError($("dataset-error"), null);
  select.disabled = true;
  select.replaceChildren(h("option", { value: "", text: "Loading labelled sets…" }));
  try {
    const datasets = await api("/api/datasets");
    state.datasets = Array.isArray(datasets) ? datasets : [];
  } catch (err) {
    state.datasets = [];
    select.replaceChildren(h("option", { value: "", text: "Labelled sets could not be loaded" }));
    showError($("dataset-error"), err, loadDatasets);
    setDataset("");
    return;
  }
  if (!state.datasets.length) {
    select.replaceChildren(h("option", { value: "", text: "No labelled sets on the server" }));
    setDataset("");
    return;
  }
  select.replaceChildren(
    ...state.datasets.map((d) => h("option", { value: d.name, text: `${d.title} (${(d.labels || []).length} labels)` })),
  );
  select.disabled = false;
  const saved = store.get("localStorage", "sense.dataset");
  const pick = state.datasets.find((d) => d.name === saved) || state.datasets[0];
  select.value = pick.name;
  setDataset(pick.name);
  if (state.job) renderTraceHead();
  if (state.model) loadSenses(); // cards drawn before the sets arrived show the set's name, not its title
}

function setDataset(name) {
  state.dataset = name;
  if (name) store.set("localStorage", "sense.dataset", name);
  const dataset = currentDataset();
  const info = $("dataset-info");
  if (dataset) {
    const labels = dataset.labels || [];
    const shown = labels.slice(0, 6).join(", ");
    const more = labels.length > 6 ? ` and ${labels.length - 6} more` : "";
    info.textContent = `${dataset.n} ${dataset.modality || "audio"} clips. Labels: ${shown}${more}.`;
  } else {
    info.textContent = state.datasets.length ? "" : "Without labelled examples the agent has nothing to fit or verify on.";
  }
  $("samples-source").textContent = dataset ? dataset.title : "";
  syncControls();
  loadSamples();
}

async function loadSamples() {
  const name = state.dataset;
  stopPlayback();
  showError($("samples-error"), null);
  state.samples = [];
  if (!name) {
    renderSamples("Choose a labelled set to see its sample clips.");
    return;
  }
  renderSamples("Loading sample clips…");
  try {
    const samples = await api(`/api/samples?dataset=${enc(name)}`);
    if (name !== state.dataset) return;
    state.samples = Array.isArray(samples) ? samples : [];
    renderSamples("No held-out clips of this set are on disk. Attach your own audio file instead.");
  } catch (err) {
    if (name !== state.dataset) return;
    renderSamples("");
    showError($("samples-error"), err, loadSamples);
  }
}

function renderSamples(emptyText) {
  const list = $("samples-list");
  state.sampleRefs.clear();
  if (!state.samples.length) {
    list.replaceChildren(...(emptyText ? [h("li", { class: "samples-empty", text: emptyText })] : []));
    return;
  }
  list.replaceChildren(...state.samples.map(sampleItem));
}

function sampleItem(sample) {
  const name = sample.name || sample.id;
  const play = h("button", { type: "button", class: "icon-button" });
  play.addEventListener("click", () => togglePlay(sample));
  const attach = h("button", { type: "button", class: "toggle" });
  attach.addEventListener("click", () => toggleSample(sample));
  const item = h(
    "li",
    { class: "sample", title: name },
    play,
    h(
      "span",
      { class: "sample-text" },
      h("span", { class: "sample-label", text: sample.label || name }),
      h("span", { class: "sample-name", text: name }),
    ),
    attach,
  );
  state.sampleRefs.set(sample.id, { item, play, attach, name });
  syncSampleItem(sample.id);
  return item;
}

/** Update one sample's play and attach controls in place, so keyboard focus stays put. */
function syncSampleItem(id) {
  const ref = state.sampleRefs.get(id);
  if (!ref) return;
  const attached = state.attachedSamples.has(id);
  const playing = state.playing === id;
  ref.item.classList.toggle("is-attached", attached);
  ref.item.classList.toggle("is-playing", playing);
  ref.attach.setAttribute("aria-pressed", String(attached));
  ref.attach.textContent = attached ? "Attached" : "Attach";
  ref.attach.setAttribute("aria-label", `${attached ? "Detach" : "Attach"} ${ref.name}`);
  ref.play.replaceChildren(icon(playing ? "stop" : "play"));
  ref.play.setAttribute("aria-label", `${playing ? "Stop" : "Play"} ${ref.name}`);
}

function toggleSample(sample) {
  if (state.attachedSamples.has(sample.id)) state.attachedSamples.delete(sample.id);
  else state.attachedSamples.set(sample.id, sample);
  syncSampleItem(sample.id);
  renderAttachments();
}

function togglePlay(sample) {
  showError($("samples-error"), null);
  if (state.playing === sample.id) {
    stopPlayback();
    return;
  }
  const previous = state.playing;
  state.playing = sample.id;
  syncSampleItem(previous);
  syncSampleItem(sample.id);
  player.src = `/api/samples/${enc(sample.id)}`;
  player.play().catch((err) => {
    if (err && err.name === "AbortError") return; // replaced by another clip
    if (state.playing !== sample.id) return;
    showError($("samples-error"), new Error(`Could not play ${sample.name || sample.id}: ${err.message}`));
    stopPlayback();
  });
}

function stopPlayback() {
  const previous = state.playing;
  state.playing = null;
  player.pause();
  syncSampleItem(previous);
}

player.addEventListener("ended", stopPlayback);
player.addEventListener("error", () => {
  if (state.playing === null) return;
  const ref = state.sampleRefs.get(state.playing);
  showError(
    $("samples-error"),
    new Error(`Could not play ${ref ? ref.name : "the clip"}: the server did not return a playable audio file.`),
  );
  stopPlayback();
});

// ---------------------------------------------------------------- chat

function renderAttachments() {
  const list = $("attachments");
  const chips = [
    ...state.files.map((file) =>
      attachmentChip(file.name, "upload", () => {
        state.files = state.files.filter((f) => f !== file);
        renderAttachments();
      }),
    ),
    ...[...state.attachedSamples.values()].map((sample) =>
      attachmentChip(sample.name || sample.id, "sample", () => toggleSample(sample)),
    ),
  ];
  list.replaceChildren(...chips);
  list.hidden = chips.length === 0;
}

function attachmentChip(name, kind, onRemove) {
  return h(
    "li",
    { class: "attachment" },
    h("span", { class: "attachment-name", text: name, title: name }),
    h("span", { class: "attachment-kind", text: kind }),
    h("button", { type: "button", "aria-label": `Remove ${name}`, onclick: onRemove }, icon("cross", 10)),
  );
}

function addFiles(fileList) {
  const files = [...(fileList || [])];
  if (!files.length) return;
  state.files.push(...files);
  renderAttachments();
}

async function sendMessage(event) {
  event.preventDefault();
  const box = $("message");
  const errorEl = $("chat-error");
  const message = box.value.trim();
  showError(errorEl, null);
  if (state.chatting) return;
  const model = currentModel();
  if (!model || !model.available) {
    showError(errorEl, new Error("Choose an available base model in the Senses panel first."));
    return;
  }
  if (!message) {
    showError(errorEl, new Error("Type a question for the model first."));
    box.focus();
    return;
  }

  const files = [...state.files];
  const samples = [...state.attachedSamples.values()];
  const form = new FormData();
  form.append("model", model.name);
  form.append("message", message);
  for (const file of files) form.append("files", file, file.name);
  for (const sample of samples) form.append("sample_ids", sample.id);

  const turn = appendTurn(model, message, [...files.map((f) => f.name), ...samples.map((s) => s.name || s.id)]);
  state.chatting = true;
  syncControls();
  try {
    const result = await api("/api/chat", { method: "POST", body: form });
    fillReply(turn, result || {});
    if (box.value.trim() === message) box.value = "";
    state.files = state.files.filter((f) => !files.includes(f));
    for (const sample of samples) {
      state.attachedSamples.delete(sample.id);
      syncSampleItem(sample.id);
    }
    renderAttachments();
  } catch (err) {
    turn.root.remove();
    updateThreadEmpty();
    showError(errorEl, err);
  } finally {
    state.chatting = false;
    syncControls();
  }
}

function appendTurn(model, message, fileNames) {
  const reply = h("div", { class: "msg-body" }, h("span", { class: "pending", text: "Waiting for the model…" }));
  const root = h(
    "article",
    { class: "turn" },
    h(
      "div",
      { class: "msg msg--user" },
      h("div", { class: "msg-who", text: "You" }),
      h(
        "div",
        { class: "msg-body" },
        h("p", { class: "reply-text", text: message }),
        fileNames.length ? h("ul", { class: "msg-files" }, fileNames.map((n) => h("li", { text: n }))) : null,
      ),
    ),
    h("div", { class: "msg msg--model" }, h("div", { class: "msg-who", text: model.label }), reply),
  );
  $("thread").append(root);
  updateThreadEmpty();
  scrollToEnd($("thread"));
  return { root, reply };
}

function fillReply(turn, result) {
  const used = Array.isArray(result.used) ? result.used : [];
  const parts = [h("p", { class: "reply-text", text: result.reply || "The model returned an empty reply." })];
  if (used.length) parts.push(h("ul", { class: "chips", "aria-label": "How each file reached the model" }, used.map(usedChip)));
  turn.reply.replaceChildren(...parts);
  scrollToEnd($("thread"));
}

/** One chip per file: how it reached the model, or a warning when it did not. */
function usedChip(used) {
  const missing = used.kind === "none";
  const detail = missing
    ? used.detail || "No sense for this kind of file is installed, so the model did not receive it."
    : [used.bridge, used.detail].filter(Boolean).join(" · ");
  return h(
    "li",
    { class: missing ? "chip chip--warn" : "chip" },
    h("span", { class: "chip-source", text: used.source || "file" }),
    missing
      ? h("span", { class: "chip-kind" }, icon("warn"), "not received")
      : h("span", { class: "chip-kind", text: `${kindLabel(used.kind)} bridge` }),
    detail ? h("span", { class: "chip-detail", text: detail }) : null,
  );
}

function updateThreadEmpty() {
  $("thread-empty").hidden = $("thread").querySelector(".turn") !== null;
}

function noteInThread(text) {
  const thread = $("thread");
  if (!thread.querySelector(".turn")) return;
  thread.append(h("p", { class: "thread-note", text }));
  scrollToEnd(thread);
}

function scrollToEnd(el) {
  el.scrollTop = el.scrollHeight;
}

// ---------------------------------------------------------------- agent run and trace

async function runAgent(event) {
  event.preventDefault();
  const errorEl = $("fit-error");
  showError(errorEl, null);
  if (state.starting || jobRunning()) return;
  const model = currentModel();
  const dataset = currentDataset();
  if (!model || !model.available) {
    showError(errorEl, new Error("Choose an available base model first."));
    return;
  }
  if (!dataset) {
    showError(errorEl, new Error("Choose a set of labelled examples first."));
    return;
  }
  const scope = document.querySelector('input[name="scope"]:checked');
  const request = {
    model: model.name,
    dataset: dataset.name,
    modality: dataset.modality || "audio",
    scope: scope ? scope.value : "closed",
  };
  state.starting = true;
  syncControls();
  try {
    const result = await api("/api/fit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (!result || !result.job_id) throw new Error("The server accepted the run but returned no job_id, so it cannot be followed.");
    state.starting = false;
    startJob(result.job_id, request, false);
  } catch (err) {
    showError(errorEl, err);
  } finally {
    state.starting = false;
    syncControls();
  }
}

function startJob(id, request, resumed) {
  if (state.job && state.job.timer) clearTimeout(state.job.timer);
  state.job = {
    id,
    request,
    resumed,
    status: "running",
    seen: 0,
    elapsed: null,
    failures: 0,
    timer: null,
    open: new Map(), // step -> the "start" event element still waiting for ok or fail
    slots: new Map(), // bridge name -> timeline element holding its report card
    cardJson: new Map(), // bridge name -> last rendered card, to skip identical redraws
  };
  store.set("sessionStorage", "sense.job", id);
  $("trace-intro").hidden = true;
  $("trace-run").hidden = false;
  $("timeline").replaceChildren();
  $("trace-outcome").replaceChildren();
  showError($("trace-error"), null);
  renderTraceHead();
  syncControls();
  poll(state.job);
}

async function poll(job) {
  let data;
  try {
    data = await api(`/api/jobs/${enc(job.id)}`);
  } catch (err) {
    if (state.job !== job) return;
    const transient = err.status === 0 || err.status >= 500;
    if (transient && job.failures < MAX_POLL_RETRIES) {
      job.failures += 1;
      showError($("trace-error"), new Error(`${err.message} Retrying (${job.failures} of ${MAX_POLL_RETRIES}).`));
      job.timer = setTimeout(() => poll(job), POLL_MS * 2);
      return;
    }
    if (job.resumed && job.seen === 0) {
      resetTrace(); // the run remembered from before a reload is gone; show the introduction again
      return;
    }
    job.status = "lost";
    store.set("sessionStorage", "sense.job", null);
    closeOpenSteps(job);
    showError($("trace-error"), new Error(`The trace stopped updating. ${err.message}`));
    renderTraceHead();
    syncControls();
    return;
  }
  if (state.job !== job) return;
  job.failures = 0;
  showError($("trace-error"), null);
  const scroller = $("trace-scroll");
  const follow = nearEnd(scroller); // decided before anything grows, so redrawn cards cannot break it
  applyJob(job, data || {});
  if (job.status === "running") job.timer = setTimeout(() => poll(job), POLL_MS);
  else finishJob(job, data || {});
  if (follow) scrollToEnd(scroller);
}

/** True when the reader is at (or close to) the bottom, so new trace lines should keep it there. */
function nearEnd(el) {
  return el.scrollHeight - el.scrollTop - el.clientHeight < 120;
}

function applyJob(job, data) {
  if (data.request) job.request = data.request;
  job.status = typeof data.status === "string" ? data.status : "running";
  const events = Array.isArray(data.events) ? data.events : [];
  if (events.length < job.seen) {
    // The server's list shrank (for example after a restart): redraw from scratch.
    $("timeline").replaceChildren();
    job.seen = 0;
    job.open.clear();
    job.slots.clear();
    job.cardJson.clear();
  }
  for (const event of events.slice(job.seen)) appendEvent(job, event);
  job.seen = events.length;
  if (events.length) job.elapsed = toNumber(events[events.length - 1].t);
  for (const card of Array.isArray(data.cards) ? data.cards : []) upsertCard(job, card, null);
  renderTraceHead();
}

function appendEvent(job, event) {
  const status = STATUS_WORDS[event.status] ? event.status : "info";
  const step = event.step || "step";
  const mark = h("span", { class: "event-mark", "aria-hidden": "true" });
  if (status === "ok") mark.append(icon("check", 11));
  if (status === "fail") mark.append(icon("cross", 11));
  const item = h(
    "li",
    { class: `event event--${status}` },
    mark,
    h(
      "div",
      { class: "event-body" },
      h(
        "div",
        { class: "event-head" },
        h("span", { class: "event-step", text: step }),
        h("span", { class: "sr-only", text: STATUS_WORDS[status] }),
        h("span", { class: "event-time", text: seconds(event.t) }),
      ),
      event.message ? h("p", { class: "event-message", text: event.message }) : null,
    ),
  );
  $("timeline").append(item);

  const open = job.open.get(step);
  if (open && status !== "info") {
    open.classList.remove("is-active");
    job.open.delete(step);
  }
  if (status === "start" && job.status === "running") {
    item.classList.add("is-active");
    job.open.set(step, item);
  }

  const card = cardFrom(event.data);
  if (card) upsertCard(job, card, item);
}

const isCard = (x) => Boolean(x && typeof x === "object" && typeof x.bridge === "string" && "accuracy" in x);

/** A verify event carries a ReportCard dict, either as its data or under data.card. */
function cardFrom(data) {
  if (!data || typeof data !== "object") return null;
  if (isCard(data.card)) return data.card;
  if (isCard(data)) return data;
  return null;
}

/** Draw a card, or redraw it in place when the same bridge's card changes (for example its final verdict). */
function upsertCard(job, card, afterNode) {
  if (!isCard(card)) return;
  const json = JSON.stringify(card);
  let slot = job.slots.get(card.bridge);
  if (slot && job.cardJson.get(card.bridge) === json) return;
  if (!slot) {
    slot = h("li", { class: "event-card" });
    job.slots.set(card.bridge, slot);
    if (afterNode) afterNode.after(slot);
    else $("timeline").append(slot);
  }
  job.cardJson.set(card.bridge, json);
  slot.replaceChildren(renderCard(card));
}

function closeOpenSteps(job) {
  for (const item of job.open.values()) item.classList.remove("is-active");
  job.open.clear();
}

function finishJob(job, data) {
  closeOpenSteps(job);
  store.set("sessionStorage", "sense.job", null);
  const outcome = $("trace-outcome");
  const model = modelLabel(job.request ? job.request.model : "");
  const done = (Array.isArray(data.events) ? data.events : []).filter((e) => e && e.step === "done").pop();
  const earlier = (done && done.data) || {}; // "replaced" or "kept": the sense installed before this run
  if (job.status === "failed") {
    outcome.replaceChildren(
      h(
        "div",
        { class: "outcome outcome--fail", role: "alert" },
        h("strong", { text: "The run failed. " }),
        data.error || "The server gave no error text. Check the server log, then run the agent again.",
      ),
    );
  } else if (data.installed) {
    outcome.replaceChildren(
      h(
        "div",
        { class: "outcome outcome--pass" },
        h("strong", { text: "Installed. " }),
        `${data.installed} is now a sense of ${model}, listed under Senses and used in chat.` +
          (earlier.replaced ? ` It replaced the earlier ${earlier.replaced} sense.` : ""),
      ),
    );
  } else {
    outcome.replaceChildren(
      h(
        "div",
        { class: "outcome outcome--warn" },
        h("strong", { text: "Nothing installed. " }),
        "No candidate passed its checks; each card above gives the reason." +
          (earlier.kept ? ` Chat keeps using the earlier ${earlier.kept} sense.` : ""),
      ),
    );
  }
  renderTraceHead();
  syncControls();
  loadSenses();
}

function renderTraceHead() {
  const job = state.job;
  const pill = $("trace-status");
  if (!job) {
    pill.hidden = true;
    return;
  }
  pill.hidden = false;
  pill.className = `pill pill--${JOB_WORDS[job.status] ? job.status : "running"}`;
  pill.textContent = JOB_WORDS[job.status] || capitalise(job.status);
  const req = job.request;
  $("trace-request").textContent = req
    ? `${modelLabel(req.model)} · ${datasetTitle(req.dataset)} · ${(SCOPE_LABELS[req.scope] || req.scope || "").toLowerCase()}`
    : "Resuming the last run…";
  $("trace-elapsed").textContent = job.elapsed !== null ? seconds(job.elapsed) : "";
}

function resetTrace() {
  if (state.job && state.job.timer) clearTimeout(state.job.timer);
  state.job = null;
  store.set("sessionStorage", "sense.job", null);
  $("trace-run").hidden = true;
  $("trace-intro").hidden = false;
  renderTraceHead();
  syncControls();
}

// ---------------------------------------------------------------- start

function updateScopeHint() {
  const scope = document.querySelector('input[name="scope"]:checked');
  $("scope-hint").textContent = SCOPE_HINTS[scope ? scope.value : "closed"];
}

function bindEvents() {
  $("model-select").addEventListener("change", (e) => setModel(e.target.value));
  $("dataset-select").addEventListener("change", (e) => setDataset(e.target.value));
  for (const radio of document.querySelectorAll('input[name="scope"]')) radio.addEventListener("change", updateScopeHint);
  $("fit-form").addEventListener("submit", runAgent);
  $("chat-form").addEventListener("submit", sendMessage);

  $("message").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      $("chat-form").requestSubmit();
    }
  });

  const fileInput = $("file-input");
  fileInput.addEventListener("change", () => {
    addFiles(fileInput.files);
    fileInput.value = ""; // allow picking the same file again
  });

  const chat = $("chat-panel");
  const hasFiles = (e) => Boolean(e.dataTransfer && [...e.dataTransfer.types].includes("Files"));
  chat.addEventListener("dragover", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    chat.classList.add("is-dropping");
  });
  chat.addEventListener("dragleave", (e) => {
    if (!chat.contains(e.relatedTarget)) chat.classList.remove("is-dropping");
  });
  chat.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    chat.classList.remove("is-dropping");
    addFiles(e.dataTransfer.files);
  });
}

function init() {
  bindEvents();
  updateScopeHint();
  renderAttachments();
  renderSamples("Choose a labelled set to see its sample clips.");
  syncControls();
  loadModels();
  loadDatasets();
  const savedJob = store.get("sessionStorage", "sense.job");
  if (savedJob) startJob(savedJob, null, true);
}

init();
