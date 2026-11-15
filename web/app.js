"use strict";

/*
 * Sense harness interface ("Sensorium"): band, chat, sense graph, agent run, report card and status line.
 * Drawn in Michael's Henad design system as a minimal FUI; design/figma-notes.md maps the parts to Figma.
 * Talks only to the HTTP API documented in server/app.py. No framework, no build step.
 * Only data the API returns is shown.
 */

const POLL_MS = 700;
const MAX_POLL_RETRIES = 5;
const OUTCOME_MS = 2000; // an answered request's card shows its outcome at least this long

const CONTROL_LABELS = {
  swapped: "Swapped audio",
  no_input: "No input",
  lookup: "Nearest-clip lookup",
  text_bridge: "Text bridge",
  unseen_classes: "Unseen categories",
  specialist_alone: "Specialist alone, no LLM",
  // An agent-written adapter's card (sense.adapters.verify_files) may add these two.
  native: "Model reads the raw file",
  unseen_categories: "Unseen categories",
};
/* How each control is drawn in the report card, as in the Figma screens: the lookup and the specialist
   alone are references, the text bridge a fallback, the rest controls. Unknown controls are controls. */
const CONTROL_ROLES = {
  lookup: { kind: "reference", tag: "reference" },
  specialist_alone: { kind: "reference", tag: "reference" },
  text_bridge: { kind: "candidate", tag: "fallback" },
};
const CONTROL_ORDER = ["lookup", "specialist_alone", "text_bridge", "unseen_classes", "unseen_categories", "native", "no_input", "swapped"];
const KIND_LABELS = { native: "Native", text: "Text", tool: "Tool", image: "Image", vector: "Vector" };
/* What each bridge kind is, from the bridge table in README.md; shown as a node subtitle. */
const KIND_NOTES = {
  vector: "Closed-form map into early layers",
  text: "Description into the prompt",
  tool: "Specialist as a tool",
  image: "Input rendered as an image",
  native: "The model reads it as is",
};
const KIND_PORTS = {
  vector: ["embedding", "states"],
  text: ["label", "prompt"],
  tool: ["call", "result"],
  image: ["render", "pixels"],
  native: ["input", "input"],
};
const KIND_WIRE = { vector: "embedding", text: "text", tool: "text", image: "image", native: "input" };
const SCOPE_LABELS = { closed: "These categories only", open: "Any sound" };
const SCOPE_HINTS = {
  closed: "The agent checks that the model tells these labels apart on held-out clips.",
  open: "The agent also tests the vector bridge on labels it never fitted on, and refuses it if it only knows the labels it was shown.",
};
/* The rule agent/planner.py applies; the other controls are reported on the card, not used to decide. */
const RULE_TEXT =
  "The agent scores each candidate bridge on held-out clips with controls: swapped audio, no audio, a nearest-example lookup and the specialist alone. It installs a bridge only if it clears chance and swapped audio by a fixed margin (for any sound, the vector bridge must also hold up on categories it was not fitted on), preferring the most accurate one; otherwise it installs nothing and says why.";
/* The bars agent/planner.py (RulePlanner.assess) and agent/synth.py apply to an agent-written adapter's card. */
const ADAPTER_RULE_TEXT =
  "The adapter is scored on labelled probe files it never saw: it was written from your file alone, before any probe existed. It is installed only if it scores at least 15 points above chance and 10 points above swapped input; at least 10 points above the model reading the raw file, when that can be measured; at least 15 points above its own chance on categories it was not built around, when those were found; on at least 10 probe files, with the 95% interval clear of chance. Otherwise nothing is installed and the card says why.";
const STATUS_WORDS = { start: "started", ok: "succeeded", fail: "failed", info: "note" };
const JOB_WORDS = { running: "Running", done: "Finished", failed: "Failed", lost: "Disconnected" };

/* Auto-discovery steps surround the original fit loop. Legacy runs show only their own steps. An adapter run
   (agent/auto.py's adapter flow, agent.synth.run_synth) has steps of its own: ADAPTER_STEPS. */
const STEPS = [
  { key: "detect", title: "Detect the input", idle: "Identify what kind of file the model needs to understand.", auto: true },
  { key: "research", title: "Research the task", idle: "Find a suitable specialist and approach for the goal.", auto: true },
  { key: "data", title: "Find labelled examples", idle: "Choose data for fitting and independent verification.", auto: true },
  { key: "inspect", title: "Check the base model", idle: "Read what the model accepts and whether its weights are open." },
  { key: "plan", title: "Choose bridges", idle: "Choose which bridges to try." },
  { key: "fit", title: "Build bridges", idle: "Build each candidate from the labelled examples." },
  { key: "verify", title: "Verify on held-out", idle: "Score each one on held-out clips against the controls." },
  { key: "claims", title: "Report supported claims", idle: "State what the measurements support and their limits.", auto: true },
  { key: "decide", title: "Install or refuse", idle: "Pick the bridge that beats the controls, or refuse." },
  { key: "install", title: "Install", idle: "Add it to the model's senses, ready for chat." },
];
/* The adapter flow, in agent.synth.run_synth's order. The two optional steps appear when the design asks to
   install packages (or the run reached them). */
const ADAPTER_STEPS = [
  { key: "detect", title: "Detect the input", idle: "Identify what kind of file the model needs to understand." },
  { key: "design", title: "Design an adapter", idle: "Plan code that turns the file into text or images the model reads." },
  { key: "install_request", title: "Check requested packages", idle: "Vet the packages the design asks for before asking you.", optional: true },
  { key: "approval", title: "Your approval", idle: "Wait for you to approve or decline installing them.", optional: true },
  { key: "build", title: "Write and run the adapter", idle: "Write adapt(path) and run it on your file in the sandbox, repairing it on errors." },
  { key: "data", title: "Find labelled probe files", idle: "Your examples folder, else a public dataset, else generated files." },
  { key: "verify", title: "Verify on probe files", idle: "Score the adapter against swapped input, no input and the raw file." },
  { key: "decide", title: "Install or refuse", idle: "Install the adapter only if it clears the bars, or refuse." },
  { key: "install", title: "Install", idle: "Add it to the model's senses, ready for chat." },
];
const STEP_INDEX = Object.fromEntries([...ADAPTER_STEPS, ...STEPS].map((s) => [s.key, s]));
const STEP_WORDS = { done: "Done", running: "Running", pending: "Pending", refused: "Refused", failed: "Failed", stopped: "Stopped", declined: "Declined" };
/* An info event that ends these steps' open start: install_request ends with info when you decline. */
const INFO_CLOSES = new Set(["install_request"]);
/* Readable names for modalities agent.detect names, and for names designers give; others are capitalised. A
   designer's name for a file agent.detect calls "other" ("3d-model" for an unknown binary) keeps its own name, so
   that its sense is never shown as the same "3D model" as agent.detect's 3d sense. */
const MODALITY_NAMES = {
  "3d": "3D model", "point-cloud": "Point cloud", molecule: "Molecule",
  geo: "Coordinates", video: "Video", audio: "Audio", text: "Text", image: "Image", table: "Table",
  timeseries: "Time series", "time-series": "Time series", "medical-image": "Medical image", other: "Other file",
};

const state = {
  models: [],
  model: "",
  datasets: [],
  datasetsLoaded: false, // GET /api/datasets has answered once, so a set missing from the list is new
  dataset: "",
  samples: [],
  sampleRefs: new Map(), // sample id -> {item, play, attach}
  attachedSamples: new Map(), // sample id -> sample
  files: [],
  playing: null, // sample id
  job: null,
  starting: false,
  chatting: false,
  chatAbort: null,
  fitFile: null,
  senses: [], // the last /api/senses answer for state.model
  usedBridges: new Set(), // bridges a chat reply has used; their top-bar chip stops being "new"
  reportBridge: null, // which card the report panel shows
  reportCard: null, // the card it shows, for Export report
  tab: "report", // which inspector tab is open: report, run, graph or log
  runs: [], // the last /api/runs answer, newest first
  replaySpeed: 1,
  replayDismissed: null, // job id whose finished replay bar the user closed
  chats: [{ id: 1, title: "New chat", turns: 0, frag: null, messages: [] }],
  editing: null,
  chatId: 1,
  nextChatId: 2,
  server: "loading", // loading, ok or error: whether /api/models answered, for the status line
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

/* Interface icons: Material Design Icons 7.4.47 by Pictogrammers (Apache-2.0), the icon set Henad's app uses.
   24 px grid, drawn in the colour of the text beside them. */
const ICONS = {
  "plus": "M19,13H13V19H11V13H5V11H11V5H13V11H19V13Z", // mdi:plus
  "x": "M19,6.41L17.59,5L12,10.59L6.41,5L5,6.41L10.59,12L5,17.59L6.41,19L12,13.41L17.59,19L19,17.59L13.41,12L19,6.41Z", // mdi:close
  "refresh": "M17.65,6.35C16.2,4.9 14.21,4 12,4A8,8 0 0,0 4,12A8,8 0 0,0 12,20C15.73,20 18.84,17.45 19.73,14H17.65C16.83,16.33 14.61,18 12,18A6,6 0 0,1 6,12A6,6 0 0,1 12,6C13.66,6 15.14,6.69 16.22,7.78L13,11H20V4L17.65,6.35Z", // mdi:refresh
  "copy": "M19,21H8V7H19M19,5H8A2,2 0 0,0 6,7V21A2,2 0 0,0 8,23H19A2,2 0 0,0 21,21V7A2,2 0 0,0 19,5M16,1H4A2,2 0 0,0 2,3V17H4V3H16V1Z", // mdi:content-copy
  "chevron-down": "M7.41,8.58L12,13.17L16.59,8.58L18,10L12,16L6,10L7.41,8.58Z", // mdi:chevron-down
  "check": "M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z", // mdi:check
  "check-circle": "M12 2C6.5 2 2 6.5 2 12S6.5 22 12 22 22 17.5 22 12 17.5 2 12 2M12 20C7.59 20 4 16.41 4 12S7.59 4 12 4 20 7.59 20 12 16.41 20 12 20M16.59 7.58L10 14.17L7.41 11.59L6 13L10 17L18 9L16.59 7.58Z", // mdi:check-circle-outline
  "ban": "M12 2C17.5 2 22 6.5 22 12S17.5 22 12 22 2 17.5 2 12 6.5 2 12 2M12 4C10.1 4 8.4 4.6 7.1 5.7L18.3 16.9C19.3 15.5 20 13.8 20 12C20 7.6 16.4 4 12 4M16.9 18.3L5.7 7.1C4.6 8.4 4 10.1 4 12C4 16.4 7.6 20 12 20C13.9 20 15.6 19.4 16.9 18.3Z", // mdi:cancel
  "loader": "M12,4V2A10,10 0 0,0 2,12H4A8,8 0 0,1 12,4Z", // mdi:loading
  "circle": "M12,20A8,8 0 0,1 4,12A8,8 0 0,1 12,4A8,8 0 0,1 20,12A8,8 0 0,1 12,20M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2Z", // mdi:circle-outline
  "play": "M8,5.14V19.14L19,12.14L8,5.14Z", // mdi:play
  "stop": "M18,18H6V6H18V18Z", // mdi:stop
  "paperclip": "M16.5,6V17.5A4,4 0 0,1 12.5,21.5A4,4 0 0,1 8.5,17.5V5A2.5,2.5 0 0,1 11,2.5A2.5,2.5 0 0,1 13.5,5V15.5A1,1 0 0,1 12.5,16.5A1,1 0 0,1 11.5,15.5V6H10V15.5A2.5,2.5 0 0,0 12.5,18A2.5,2.5 0 0,0 15,15.5V5A4,4 0 0,0 11,1A4,4 0 0,0 7,5V17.5A5.5,5.5 0 0,0 12.5,23A5.5,5.5 0 0,0 18,17.5V6H16.5Z", // mdi:paperclip
  "arrow-up": "M13,20H11V8L5.5,13.5L4.08,12.08L12,4.16L19.92,12.08L18.5,13.5L13,8V20Z", // mdi:arrow-up
  "download": "M2 12H4V17H20V12H22V17C22 18.11 21.11 19 20 19H4C2.9 19 2 18.11 2 17V12M12 15L17.55 9.54L16.13 8.13L13 11.25V2H11V11.25L7.88 8.13L6.46 9.55L12 15Z", // mdi:tray-arrow-down
  "file": "M6,2A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2H6M6,4H13V9H18V20H6V4M8,12V14H16V12H8M8,16V18H13V16H8Z", // mdi:file-document-outline
  "database": "M12 3C7.58 3 4 4.79 4 7V17C4 19.21 7.59 21 12 21S20 19.21 20 17V7C20 4.79 16.42 3 12 3M18 17C18 17.5 15.87 19 12 19S6 17.5 6 17V14.77C7.61 15.55 9.72 16 12 16S16.39 15.55 18 14.77V17M18 12.45C16.7 13.4 14.42 14 12 14C9.58 14 7.3 13.4 6 12.45V9.64C7.47 10.47 9.61 11 12 11C14.39 11 16.53 10.47 18 9.64V12.45M12 9C8.13 9 6 7.5 6 7S8.13 5 12 5C15.87 5 18 6.5 18 7S15.87 9 12 9Z", // mdi:database-outline
  "plug": "M16 7V3H14V7H10V3H8V7C7 7 6 8 6 9V14.5L9.5 18V21H14.5V18L18 14.5V9C18 8 17 7 16 7M16 13.67L13.09 16.59L12.67 17H11.33L10.92 16.59L8 13.67V9.09C8 9.06 8.06 9 8.09 9H15.92C15.95 9 16 9.06 16 9.09V13.67Z", // mdi:power-plug-outline
  "cpu": "M6,4H18V5H21V7H18V9H21V11H18V13H21V15H18V17H21V19H18V20H6V19H3V17H6V15H3V13H6V11H3V9H6V7H3V5H6V4M11,15V18H12V15H11M13,15V18H14V15H13M15,15V18H16V15H15Z", // mdi:chip
  "info": "M11,9H13V7H11M12,20C7.59,20 4,16.41 4,12C4,7.59 7.59,4 12,4C16.41,4 20,7.59 20,12C20,16.41 16.41,20 12,20M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M11,17H13V11H11V17Z", // mdi:information-outline
  "shield-check": "M21,11C21,16.55 17.16,21.74 12,23C6.84,21.74 3,16.55 3,11V5L12,1L21,5V11M12,21C15.75,20 19,15.54 19,11.22V6.3L12,3.18L5,6.3V11.22C5,15.54 8.25,20 12,21M10,17L6,13L7.41,11.59L10,14.17L16.59,7.58L18,9", // mdi:shield-check-outline
  "sparkles": "M19,1L17.74,3.75L15,5L17.74,6.26L19,9L20.25,6.26L23,5L20.25,3.75M9,4L6.5,9.5L1,12L6.5,14.5L9,20L11.5,14.5L17,12L11.5,9.5M19,15L17.74,17.74L15,19L17.74,20.25L19,23L20.25,20.25L23,19L20.25,17.74", // mdi:creation
  "agent": "M17.5 15.5C17.5 16.61 16.61 17.5 15.5 17.5S13.5 16.61 13.5 15.5 14.4 13.5 15.5 13.5 17.5 14.4 17.5 15.5M8.5 13.5C7.4 13.5 6.5 14.4 6.5 15.5S7.4 17.5 8.5 17.5 10.5 16.61 10.5 15.5 9.61 13.5 8.5 13.5M23 15V18C23 18.55 22.55 19 22 19H21V20C21 21.11 20.11 22 19 22H5C3.9 22 3 21.11 3 20V19H2C1.45 19 1 18.55 1 18V15C1 14.45 1.45 14 2 14H3C3 10.13 6.13 7 10 7H11V5.73C10.4 5.39 10 4.74 10 4C10 2.9 10.9 2 12 2S14 2.9 14 4C14 4.74 13.6 5.39 13 5.73V7H14C17.87 7 21 10.13 21 14H22C22.55 14 23 14.45 23 15M21 16H19V14C19 11.24 16.76 9 14 9H10C7.24 9 5 11.24 5 14V16H3V17H5V20H19V17H21V16Z", // mdi:robot-outline
  "user": "M12,4A4,4 0 0,1 16,8A4,4 0 0,1 12,12A4,4 0 0,1 8,8A4,4 0 0,1 12,4M12,6A2,2 0 0,0 10,8A2,2 0 0,0 12,10A2,2 0 0,0 14,8A2,2 0 0,0 12,6M12,13C14.67,13 20,14.33 20,17V20H4V17C4,14.33 9.33,13 12,13M12,14.9C9.03,14.9 5.9,16.36 5.9,17V18.1H18.1V17C18.1,16.36 14.97,14.9 12,14.9Z", // mdi:account-outline
  "log": "M13,19V16H21V19H13M8.5,13L2.47,7H6.71L11.67,11.95C12.25,12.54 12.25,13.5 11.67,14.07L6.74,19H2.5L8.5,13Z", // mdi:console-line
  "graph": "M19.5 17C19.36 17 19.24 17 19.11 17.04L17.5 13.8C17.95 13.35 18.25 12.71 18.25 12C18.25 10.62 17.13 9.5 15.75 9.5C15.61 9.5 15.5 9.5 15.35 9.54L13.74 6.3C14.21 5.84 14.5 5.21 14.5 4.5C14.5 3.12 13.38 2 12 2S9.5 3.12 9.5 4.5C9.5 5.2 9.79 5.84 10.26 6.29L8.65 9.54C8.5 9.5 8.39 9.5 8.25 9.5C6.87 9.5 5.75 10.62 5.75 12C5.75 12.71 6.04 13.34 6.5 13.79L4.89 17.04C4.76 17 4.64 17 4.5 17C3.12 17 2 18.12 2 19.5C2 20.88 3.12 22 4.5 22S7 20.88 7 19.5C7 18.8 6.71 18.16 6.24 17.71L7.86 14.46C8 14.5 8.12 14.5 8.25 14.5C8.38 14.5 8.5 14.5 8.63 14.46L10.26 17.71C9.79 18.16 9.5 18.8 9.5 19.5C9.5 20.88 10.62 22 12 22S14.5 20.88 14.5 19.5C14.5 18.12 13.38 17 12 17C11.87 17 11.74 17 11.61 17.04L10 13.8C10.45 13.35 10.75 12.71 10.75 12C10.75 11.3 10.46 10.67 10 10.21L11.61 6.96C11.74 7 11.87 7 12 7C12.13 7 12.26 7 12.39 6.96L14 10.21C13.54 10.66 13.25 11.3 13.25 12C13.25 13.38 14.37 14.5 15.75 14.5C15.88 14.5 16 14.5 16.13 14.46L17.76 17.71C17.29 18.16 17 18.8 17 19.5C17 20.88 18.12 22 19.5 22S22 20.88 22 19.5C22 18.12 20.88 17 19.5 17M4.5 20.5C3.95 20.5 3.5 20.05 3.5 19.5S3.95 18.5 4.5 18.5 5.5 18.95 5.5 19.5 5.05 20.5 4.5 20.5M13 19.5C13 20.05 12.55 20.5 12 20.5S11 20.05 11 19.5 11.45 18.5 12 18.5 13 18.95 13 19.5M7.25 12C7.25 11.45 7.7 11 8.25 11S9.25 11.45 9.25 12 8.8 13 8.25 13 7.25 12.55 7.25 12M11 4.5C11 3.95 11.45 3.5 12 3.5S13 3.95 13 4.5 12.55 5.5 12 5.5 11 5.05 11 4.5M14.75 12C14.75 11.45 15.2 11 15.75 11S16.75 11.45 16.75 12 16.3 13 15.75 13 14.75 12.55 14.75 12M19.5 20.5C18.95 20.5 18.5 20.05 18.5 19.5S18.95 18.5 19.5 18.5 20.5 18.95 20.5 19.5 20.05 20.5 19.5 20.5Z", // mdi:graph-outline
  "alert": "M12,2L1,21H23M12,6L19.53,19H4.47M11,10V14H13V10M11,16V18H13V16", // mdi:alert-outline
};

const SVG_NS = "http://www.w3.org/2000/svg";

function icon(name, size = 16) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", String(size));
  svg.setAttribute("height", String(size));
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("class", "icon");
  svg.setAttribute("data-name", name);
  svg.setAttribute("fill", "currentColor");
  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute("d", ICONS[name] || ICONS.circle);
  svg.append(path);
  return svg;
}

/* Modality icons: Material Design Icons 7.4.47 by Pictogrammers (Apache-2.0), the set in the Figma frame
   "Modality icons (MDI)" (18:2). Purely visual: a modality key or alias picks an icon, anything else
   gets Other, so the server can name any modality. */
const MODALITY_ICONS = {
  "text": "M20.06,18C20,17.83 19.91,17.54 19.86,17.11C19.19,17.81 18.38,18.16 17.45,18.16C16.62,18.16 15.93,17.92 15.4,17.45C14.87,17 14.6,16.39 14.6,15.66C14.6,14.78 14.93,14.1 15.6,13.61C16.27,13.12 17.21,12.88 18.43,12.88H19.83V12.24C19.83,11.75 19.68,11.36 19.38,11.07C19.08,10.78 18.63,10.64 18.05,10.64C17.53,10.64 17.1,10.76 16.75,11C16.4,11.25 16.23,11.54 16.23,11.89H14.77C14.77,11.46 14.92,11.05 15.22,10.65C15.5,10.25 15.93,9.94 16.44,9.71C16.95,9.5 17.5,9.36 18.13,9.36C19.11,9.36 19.87,9.6 20.42,10.09C20.97,10.58 21.26,11.25 21.28,12.11V16C21.28,16.8 21.38,17.42 21.58,17.88V18H20.06M17.66,16.88C18.11,16.88 18.54,16.77 18.95,16.56C19.35,16.35 19.65,16.07 19.83,15.73V14.16H18.7C16.93,14.16 16.04,14.63 16.04,15.57C16.04,16 16.19,16.3 16.5,16.53C16.8,16.76 17.18,16.88 17.66,16.88M5.46,13.71H9.53L7.5,8.29L5.46,13.71M6.64,6H8.36L13.07,18H11.14L10.17,15.43H4.82L3.86,18H1.93L6.64,6Z",
  "image": "M19,19H5V5H19M19,3H5A2,2 0 0,0 3,5V19A2,2 0 0,0 5,21H19A2,2 0 0,0 21,19V5A2,2 0 0,0 19,3M13.96,12.29L11.21,15.83L9.25,13.47L6.5,17H17.5L13.96,12.29Z",
  "audio": "M22 12L20 13L19 14L18 13L17 16L16 13L15 21L14 13L13 15L12 13L11 17L10 13L9 22L8 13L7 19L6 13L5 14L4 13L2 12L4 11L5 10L6 11L7 5L8 11L9 2L10 11L11 7L12 11L13 9L14 11L15 3L16 11L17 8L18 11L19 10L20 11L22 12Z",
  "speech": "M9,5A4,4 0 0,1 13,9A4,4 0 0,1 9,13A4,4 0 0,1 5,9A4,4 0 0,1 9,5M9,15C11.67,15 17,16.34 17,19V21H1V19C1,16.34 6.33,15 9,15M16.76,5.36C18.78,7.56 18.78,10.61 16.76,12.63L15.08,10.94C15.92,9.76 15.92,8.23 15.08,7.05L16.76,5.36M20.07,2C24,6.05 23.97,12.11 20.07,16L18.44,14.37C21.21,11.19 21.21,6.65 18.44,3.63L20.07,2Z",
  "music": "M12 3V13.55C11.41 13.21 10.73 13 10 13C7.79 13 6 14.79 6 17S7.79 21 10 21 14 19.21 14 17V7H18V3H12Z",
  "video": "M15,8V16H5V8H15M16,6H4A1,1 0 0,0 3,7V17A1,1 0 0,0 4,18H16A1,1 0 0,0 17,17V13.5L21,17.5V6.5L17,10.5V7A1,1 0 0,0 16,6Z",
  "document": "M6,2A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2H6M6,4H13V9H18V20H6V4M8,12V14H16V12H8M8,16V18H13V16H8Z",
  "table": "M4,3H20A2,2 0 0,1 22,5V20A2,2 0 0,1 20,22H4A2,2 0 0,1 2,20V5A2,2 0 0,1 4,3M4,7V10H8V7H4M10,7V10H14V7H10M20,10V7H16V10H20M4,12V15H8V12H4M4,20H8V17H4V20M10,12V15H14V12H10M10,20H14V17H10V20M20,20V17H16V20H20M20,12H16V15H20V12Z",
  "timeseries": "M3,14L3.5,14.07L8.07,9.5C7.89,8.85 8.06,8.11 8.59,7.59C9.37,6.8 10.63,6.8 11.41,7.59C11.94,8.11 12.11,8.85 11.93,9.5L14.5,12.07L15,12C15.18,12 15.35,12 15.5,12.07L19.07,8.5C19,8.35 19,8.18 19,8A2,2 0 0,1 21,6A2,2 0 0,1 23,8A2,2 0 0,1 21,10C20.82,10 20.65,10 20.5,9.93L16.93,13.5C17,13.65 17,13.82 17,14A2,2 0 0,1 15,16A2,2 0 0,1 13,14L13.07,13.5L10.5,10.93C10.18,11 9.82,11 9.5,10.93L4.93,15.5L5,16A2,2 0 0,1 3,18A2,2 0 0,1 1,16A2,2 0 0,1 3,14Z",
  "code": "M8,3A2,2 0 0,0 6,5V9A2,2 0 0,1 4,11H3V13H4A2,2 0 0,1 6,15V19A2,2 0 0,0 8,21H10V19H8V14A2,2 0 0,0 6,12A2,2 0 0,0 8,10V5H10V3M16,3A2,2 0 0,1 18,5V9A2,2 0 0,0 20,11H21V13H20A2,2 0 0,0 18,15V19A2,2 0 0,1 16,21H14V19H16V14A2,2 0 0,1 18,12A2,2 0 0,1 16,10V5H14V3H16Z",
  "3d": "M21,16.5C21,16.88 20.79,17.21 20.47,17.38L12.57,21.82C12.41,21.94 12.21,22 12,22C11.79,22 11.59,21.94 11.43,21.82L3.53,17.38C3.21,17.21 3,16.88 3,16.5V7.5C3,7.12 3.21,6.79 3.53,6.62L11.43,2.18C11.59,2.06 11.79,2 12,2C12.21,2 12.41,2.06 12.57,2.18L20.47,6.62C20.79,6.79 21,7.12 21,7.5V16.5M12,4.15L6.04,7.5L12,10.85L17.96,7.5L12,4.15M5,15.91L11,19.29V12.58L5,9.21V15.91M19,15.91V9.21L13,12.58V19.29L19,15.91Z",
  "geo": "M12,6.5A2.5,2.5 0 0,1 14.5,9A2.5,2.5 0 0,1 12,11.5A2.5,2.5 0 0,1 9.5,9A2.5,2.5 0 0,1 12,6.5M12,2A7,7 0 0,1 19,9C19,14.25 12,22 12,22C12,22 5,14.25 5,9A7,7 0 0,1 12,2M12,4A5,5 0 0,0 7,9C7,10 7,12 12,18.71C17,12 17,10 17,9A5,5 0 0,0 12,4Z",
  "graph": "M19.5 17C19.36 17 19.24 17 19.11 17.04L17.5 13.8C17.95 13.35 18.25 12.71 18.25 12C18.25 10.62 17.13 9.5 15.75 9.5C15.61 9.5 15.5 9.5 15.35 9.54L13.74 6.3C14.21 5.84 14.5 5.21 14.5 4.5C14.5 3.12 13.38 2 12 2S9.5 3.12 9.5 4.5C9.5 5.2 9.79 5.84 10.26 6.29L8.65 9.54C8.5 9.5 8.39 9.5 8.25 9.5C6.87 9.5 5.75 10.62 5.75 12C5.75 12.71 6.04 13.34 6.5 13.79L4.89 17.04C4.76 17 4.64 17 4.5 17C3.12 17 2 18.12 2 19.5C2 20.88 3.12 22 4.5 22S7 20.88 7 19.5C7 18.8 6.71 18.16 6.24 17.71L7.86 14.46C8 14.5 8.12 14.5 8.25 14.5C8.38 14.5 8.5 14.5 8.63 14.46L10.26 17.71C9.79 18.16 9.5 18.8 9.5 19.5C9.5 20.88 10.62 22 12 22S14.5 20.88 14.5 19.5C14.5 18.12 13.38 17 12 17C11.87 17 11.74 17 11.61 17.04L10 13.8C10.45 13.35 10.75 12.71 10.75 12C10.75 11.3 10.46 10.67 10 10.21L11.61 6.96C11.74 7 11.87 7 12 7C12.13 7 12.26 7 12.39 6.96L14 10.21C13.54 10.66 13.25 11.3 13.25 12C13.25 13.38 14.37 14.5 15.75 14.5C15.88 14.5 16 14.5 16.13 14.46L17.76 17.71C17.29 18.16 17 18.8 17 19.5C17 20.88 18.12 22 19.5 22S22 20.88 22 19.5C22 18.12 20.88 17 19.5 17M4.5 20.5C3.95 20.5 3.5 20.05 3.5 19.5S3.95 18.5 4.5 18.5 5.5 18.95 5.5 19.5 5.05 20.5 4.5 20.5M13 19.5C13 20.05 12.55 20.5 12 20.5S11 20.05 11 19.5 11.45 18.5 12 18.5 13 18.95 13 19.5M7.25 12C7.25 11.45 7.7 11 8.25 11S9.25 11.45 9.25 12 8.8 13 8.25 13 7.25 12.55 7.25 12M11 4.5C11 3.95 11.45 3.5 12 3.5S13 3.95 13 4.5 12.55 5.5 12 5.5 11 5.05 11 4.5M14.75 12C14.75 11.45 15.2 11 15.75 11S16.75 11.45 16.75 12 16.3 13 15.75 13 14.75 12.55 14.75 12M19.5 20.5C18.95 20.5 18.5 20.05 18.5 19.5S18.95 18.5 19.5 18.5 20.5 18.95 20.5 19.5 20.05 20.5 19.5 20.5Z",
  "molecule": "M7.27,10L9,7H14.42L15.58,5L15.5,4.5A1.5,1.5 0 0,1 17,3A1.5,1.5 0 0,1 18.5,4.5C18.5,5.21 18,5.81 17.33,5.96L16.37,7.63L17.73,10L18.59,8.5L18.5,8A1.5,1.5 0 0,1 20,6.5A1.5,1.5 0 0,1 21.5,8C21.5,8.71 21,9.3 20.35,9.46L18.89,12L20.62,15C21.39,15.07 22,15.71 22,16.5A1.5,1.5 0 0,1 20.5,18A1.5,1.5 0 0,1 19,16.5V16.24L17.73,14L16.37,16.37L17.33,18.04C18,18.19 18.5,18.79 18.5,19.5A1.5,1.5 0 0,1 17,21A1.5,1.5 0 0,1 15.5,19.5L15.58,19L14.42,17H10.58L9.42,19L9.5,19.5A1.5,1.5 0 0,1 8,21A1.5,1.5 0 0,1 6.5,19.5C6.5,18.79 7,18.19 7.67,18.04L8.63,16.37L4.38,9C3.61,8.93 3,8.29 3,7.5A1.5,1.5 0 0,1 4.5,6A1.5,1.5 0 0,1 6,7.5C6,7.59 6,7.68 6,7.76L7.27,10M10.15,9L8.42,12L10.15,15H14.85L16.58,12L14.85,9H10.15Z",
  "genome": "M4,2H6V4C6,5.44 6.68,6.61 7.88,7.78C8.74,8.61 9.89,9.41 11.09,10.2L9.26,11.39C8.27,10.72 7.31,10 6.5,9.21C5.07,7.82 4,6.1 4,4V2M18,2H20V4C20,6.1 18.93,7.82 17.5,9.21C16.09,10.59 14.29,11.73 12.54,12.84C10.79,13.96 9.09,15.05 7.88,16.22C6.68,17.39 6,18.56 6,20V22H4V20C4,17.9 5.07,16.18 6.5,14.79C7.91,13.41 9.71,12.27 11.46,11.16C13.21,10.04 14.91,8.95 16.12,7.78C17.32,6.61 18,5.44 18,4V2M14.74,12.61C15.73,13.28 16.69,14 17.5,14.79C18.93,16.18 20,17.9 20,20V22H18V20C18,18.56 17.32,17.39 16.12,16.22C15.26,15.39 14.11,14.59 12.91,13.8L14.74,12.61M7,3H17V4L16.94,4.5H7.06L7,4V3M7.68,6H16.32C16.08,6.34 15.8,6.69 15.42,7.06L14.91,7.5H9.07L8.58,7.06C8.2,6.69 7.92,6.34 7.68,6M9.09,16.5H14.93L15.42,16.94C15.8,17.31 16.08,17.66 16.32,18H7.68C7.92,17.66 8.2,17.31 8.58,16.94L9.09,16.5M7.06,19.5H16.94L17,20V21H7V20L7.06,19.5Z",
  "biosignal": "M7.5,4A5.5,5.5 0 0,0 2,9.5C2,10 2.09,10.5 2.22,11H6.3L7.57,7.63C7.87,6.83 9.05,6.75 9.43,7.63L11.5,13L12.09,11.58C12.22,11.25 12.57,11 13,11H21.78C21.91,10.5 22,10 22,9.5A5.5,5.5 0 0,0 16.5,4C14.64,4 13,4.93 12,6.34C11,4.93 9.36,4 7.5,4V4M3,12.5A1,1 0 0,0 2,13.5A1,1 0 0,0 3,14.5H5.44L11,20C12,20.9 12,20.9 13,20L18.56,14.5H21A1,1 0 0,0 22,13.5A1,1 0 0,0 21,12.5H13.4L12.47,14.8C12.07,15.81 10.92,15.67 10.55,14.83L8.5,9.5L7.54,11.83C7.39,12.21 7.05,12.5 6.6,12.5H3Z",
  "brain": "M21.33,12.91C21.42,14.46 20.71,15.95 19.44,16.86L20.21,18.35C20.44,18.8 20.47,19.33 20.27,19.8C20.08,20.27 19.69,20.64 19.21,20.8L18.42,21.05C18.25,21.11 18.06,21.14 17.88,21.14C17.37,21.14 16.89,20.91 16.56,20.5L14.44,18C13.55,17.85 12.71,17.47 12,16.9C11.5,17.05 11,17.13 10.5,17.13C9.62,17.13 8.74,16.86 8,16.34C7.47,16.5 6.93,16.57 6.38,16.56C5.59,16.57 4.81,16.41 4.08,16.11C2.65,15.47 1.7,14.07 1.65,12.5C1.57,11.78 1.69,11.05 2,10.39C1.71,9.64 1.68,8.82 1.93,8.06C2.3,7.11 3,6.32 3.87,5.82C4.45,4.13 6.08,3 7.87,3.12C9.47,1.62 11.92,1.46 13.7,2.75C14.12,2.64 14.56,2.58 15,2.58C16.36,2.55 17.65,3.15 18.5,4.22C20.54,4.75 22,6.57 22.08,8.69C22.13,9.8 21.83,10.89 21.22,11.82C21.29,12.18 21.33,12.54 21.33,12.91M16.33,11.5C16.9,11.57 17.35,12 17.35,12.57A1,1 0 0,1 16.35,13.57H15.72C15.4,14.47 14.84,15.26 14.1,15.86C14.35,15.95 14.61,16 14.87,16.07C20,16 19.4,12.87 19.4,12.82C19.34,11.39 18.14,10.27 16.71,10.33A1,1 0 0,1 15.71,9.33A1,1 0 0,1 16.71,8.33C17.94,8.36 19.12,8.82 20.04,9.63C20.09,9.34 20.12,9.04 20.12,8.74C20.06,7.5 19.5,6.42 17.25,6.21C16,3.25 12.85,4.89 12.85,5.81V5.81C12.82,6.04 13.06,6.53 13.1,6.56A1,1 0 0,1 14.1,7.56C14.1,8.11 13.65,8.56 13.1,8.56V8.56C12.57,8.54 12.07,8.34 11.67,8C11.19,8.31 10.64,8.5 10.07,8.56V8.56C9.5,8.61 9.03,8.21 9,7.66C8.92,7.1 9.33,6.61 9.88,6.56C10.04,6.54 10.82,6.42 10.82,5.79V5.79C10.82,5.13 11.07,4.5 11.5,4C10.58,3.75 9.59,4.08 8.59,5.29C6.75,5 6,5.25 5.45,7.2C4.5,7.67 4,8 3.78,9C4.86,8.78 5.97,8.87 7,9.25C7.5,9.44 7.78,10 7.59,10.54C7.4,11.06 6.82,11.32 6.3,11.13C5.57,10.81 4.75,10.79 4,11.07C3.68,11.34 3.68,11.9 3.68,12.34C3.68,13.08 4.05,13.77 4.68,14.17C5.21,14.44 5.8,14.58 6.39,14.57C6.24,14.31 6.11,14.04 6,13.76C5.81,13.22 6.1,12.63 6.64,12.44C7.18,12.25 7.77,12.54 7.96,13.08C8.36,14.22 9.38,15 10.58,15.13C11.95,15.06 13.17,14.25 13.77,13C14,11.62 15.11,11.5 16.33,11.5M18.33,18.97L17.71,17.67L17,17.83L18,19.08L18.33,18.97M13.68,10.36C13.7,9.83 13.3,9.38 12.77,9.33C12.06,9.29 11.37,9.53 10.84,10C10.27,10.58 9.97,11.38 10,12.19A1,1 0 0,0 11,13.19C11.57,13.19 12,12.74 12,12.19C12,11.92 12.07,11.65 12.23,11.43C12.35,11.33 12.5,11.28 12.66,11.28C13.21,11.31 13.68,10.9 13.68,10.36Z",
  "medical-image": "M2 0C.9 0 0 .9 0 2V6H2V2H6V0H2M18 0V2H22V6H24V2C24 .9 23.1 0 22 0H18M9.5 13C8.7 13 8 12.3 8 11.5S8.7 10 9.5 10 11 10.7 11 11.5 10.3 13 9.5 13M11 15L12 13L13 15H11M14.5 13C13.7 13 13 12.3 13 11.5S13.7 10 14.5 10 16 10.7 16 11.5 15.3 13 14.5 13M0 18V22C0 23.1 .9 24 2 24H6V22H2V18H0M22 18V22H18V24H22C23.1 24 24 23.1 24 22V18H22M12 3C7.6 3 4 6.6 4 11C4 13 4.8 14.9 6 16.3V21H18V16.3C19.2 14.9 20 13.1 20 11C20 6.6 16.4 3 12 3M16 15.4V19H14V17H13V19H11V17H10V19H8V15.4C6.8 14.3 6 12.7 6 11C6 7.7 8.7 5 12 5S18 7.7 18 11C18 12.8 17.2 14.3 16 15.4Z",
  "motion": "M12,2L16,6H13V13.85L19.53,17.61L21,15.03L22.5,20.5L17,21.96L18.53,19.35L12,15.58L5.47,19.35L7,21.96L1.5,20.5L3,15.03L4.47,17.61L11,13.85V6H8L12,2Z",
  "range": "M19.07,4.93L17.66,6.34C19.1,7.79 20,9.79 20,12A8,8 0 0,1 12,20A8,8 0 0,1 4,12C4,7.92 7.05,4.56 11,4.07V6.09C8.16,6.57 6,9.03 6,12A6,6 0 0,0 12,18A6,6 0 0,0 18,12C18,10.34 17.33,8.84 16.24,7.76L14.83,9.17C15.55,9.9 16,10.9 16,12A4,4 0 0,1 12,16A4,4 0 0,1 8,12C8,10.14 9.28,8.59 11,8.14V10.28C10.4,10.63 10,11.26 10,12A2,2 0 0,0 12,14A2,2 0 0,0 14,12C14,11.26 13.6,10.62 13,10.28V2H12A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12C22,9.24 20.88,6.74 19.07,4.93Z",
  "other": "M11,13.5V21.5H3V13.5H11M9,15.5H5V19.5H9V15.5M12,2L17.5,11H6.5L12,2M12,5.86L10.08,9H13.92L12,5.86M17.5,13C20,13 22,15 22,17.5C22,20 20,22 17.5,22C15,22 13,20 13,17.5C13,15 15,13 17.5,13M17.5,15A2.5,2.5 0 0,0 15,17.5A2.5,2.5 0 0,0 17.5,20A2.5,2.5 0 0,0 20,17.5A2.5,2.5 0 0,0 17.5,15Z",
};
const MODALITY_ALIASES = {
  txt: "text", prompt: "text", photo: "image", picture: "image", sound: "audio", voice: "speech",
  movie: "video", pdf: "document", tabular: "table", csv: "table", "time-series": "timeseries",
  time_series: "timeseries", signal: "timeseries", sensor: "timeseries", source: "code",
  "point-cloud": "3d", mesh: "3d", location: "geo", gps: "geo", map: "geo", network: "graph",
  chemistry: "molecule", smiles: "molecule", dna: "genome", protein: "genome", ecg: "biosignal",
  ppg: "biosignal", eeg: "brain", fmri: "brain", xray: "medical-image", "x-ray": "medical-image",
  ct: "medical-image", mri: "medical-image", imu: "motion", lidar: "range", radar: "range", depth: "range",
};
/* Data colours (--wire-* in style.css) by modality; modalities without their own colour share one. */
const MODALITY_TOKENS = {
  audio: "audio", speech: "audio", music: "audio",
  text: "text", document: "text", code: "text",
  image: "image", video: "image", "medical-image": "image",
  table: "data", timeseries: "data", graph: "data", molecule: "data", genome: "data", biosignal: "data",
  brain: "data", motion: "data", range: "data", "3d": "data", geo: "data",
};

function modalityKey(modality) {
  const key = String(modality || "").toLowerCase().trim();
  if (MODALITY_ICONS[key]) return key;
  return MODALITY_ALIASES[key] || "other";
}
const modalityColour = (modality) => `var(--wire-${MODALITY_TOKENS[modalityKey(modality)] || "input"})`;
const wireColour = (token) => `var(--wire-${token || "input"})`;

function modalityIcon(modality, size = 16) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", String(size));
  svg.setAttribute("height", String(size));
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("class", "icon");
  svg.setAttribute("fill", "currentColor");
  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute("d", MODALITY_ICONS[modalityKey(modality)] || "");
  svg.append(path);
  return svg;
}

/** Put the named icons into the static markup: <span data-icon="plus"></span>. */
function hydrateIcons() {
  for (const el of document.querySelectorAll("[data-icon]")) el.replaceChildren(icon(el.dataset.icon));
}

/** Status badge: the hue's 900 fill with its 200 label. tone: neutral, info, running, success, warning,
    danger, new (the accent, for a sense not used yet). */
function tag(label, tone = "neutral") {
  return h("span", { class: `tag tag--${tone}`, text: label });
}
function setTag(el, label, tone) {
  el.className = `tag tag--${tone}`;
  el.textContent = label;
  el.hidden = false;
}

/** Callout: a rule down the left under a title strip. tone: neutral, info, success, warning, danger. */
function callout(tone, title, body, { value = null, iconName = null, action = null } = {}) {
  return h("div", { class: `callout callout--${tone}` },
    h("div", { class: "callout-head" },
      h("span", { class: "callout-title" }, iconName ? icon(iconName, 14) : null, title),
      value ? h("span", { class: "callout-value", text: value }) : null),
    body ? h("p", { class: "callout-body", text: body }) : null,
    action);
}

/** A status light: a small square, coloured by state (ok, run, error, or idle). */
const led = (tone = "idle") => h("span", { class: `led led--${tone}`, "aria-hidden": "true" });

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
const seconds = (t) => (toNumber(t) === null ? "" : `${t < 0.1 ? t.toFixed(2) : t.toFixed(1)} s`);
const capitalise = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : "");
const humanise = (key) => capitalise(String(key).replace(/_/g, " "));
const kindLabel = (kind) => KIND_LABELS[kind] || capitalise(kind || "unknown");
/** A modality as people say it: "3D model", "Coordinates", "Molecule"; an unknown name is capitalised. */
function modalityName(modality) {
  const key = String(modality || "").toLowerCase().trim();
  return MODALITY_NAMES[key] || capitalise(key.replace(/[-_]+/g, " ")) || "Sense";
}
/** An adapter run's probe set: agent.synth names it <modality>-dataset-<hash>, <modality>-examples-<hash> or generated-<modality>-<hash>. */
const isProbeSet = (name) => typeof name === "string" && /^generated-.+-[0-9a-f]{6,}$|-(dataset|examples)-[0-9a-f]{6,}$/.test(name);
/** A labelled set's title for subtitles; an adapter's probe set, named by a hash, by where its files came from
    (the name says that much; the run's data step, when it is at hand, says more: probeSourceText). */
function setTitle(name) {
  if (!isProbeSet(name)) return datasetTitle(name);
  if (name.startsWith("generated-")) return "generated probe files";
  return /-examples-[0-9a-f]+$/.test(name) ? "probe files from your examples" : "probe files from a public dataset";
}
/** An agent-written adapter's bridge is named "adapter:<slug>" (sense.adapters.AdapterBridge). */
const isAdapter = (bridge) => typeof bridge === "string" && bridge.startsWith("adapter:");
/** What a card's bridge is, for subtitles: "Agent-written adapter (text)" or "Vector bridge". */
const bridgeLabel = (card) => (isAdapter(card.bridge) ? `Agent-written adapter (${card.kind === "image" ? "images" : "text"})` : `${kindLabel(card.kind)} bridge`);
/** How a card is drawn. The API's verdict, except that a rejected bridge that passed its checks and lost to
    the installed one was not needed rather than refused: screen 02 draws it as a plain candidate. */
function shownVerdict(card) {
  if (card.verdict === "installed") return "installed";
  if (card.verdict !== "rejected") return "candidate";
  const job = modelJob();
  if (job && job.installed && job.cards.get(card.bridge) === card && !cardFailed(card, job)) return "not-needed";
  return "refused";
}

/** Whether a rejected card failed the planner's checks, as the planner judged it. agent.loop writes
    "<bridge> failed: ..." or "<bridge> passed: ..." into the card's reason; older records may say it only in
    the decide step's reason, or not at all, and then the rule planner's bars decide (passesRules). */
function cardFailed(card, job) {
  const said = (text) => {
    if (typeof text !== "string") return null;
    const failed = text.indexOf(`${card.bridge} failed`);
    const passed = text.indexOf(`${card.bridge} passed`);
    if (failed < 0 && passed < 0) return null;
    return passed < 0 || (failed >= 0 && failed < passed);
  };
  const decide = job ? job.steps.get("decide") : null;
  const decided = decide ? decide.events.findLast((e) => e.status === "ok") : null;
  for (const text of [card.reason, decided && decided.message]) {
    const failed = said(text);
    if (failed !== null) return failed;
  }
  const passes = passesRules(card);
  return passes === null ? true : !passes;
}

/** The bars of agent/planner.py (RulePlanner._check), only for records whose reasons do not say whether a
    card passed. null when they do not apply. */
function passesRules(card) {
  const EPS = 1e-9;
  const acc = toNumber(card.accuracy);
  const chance = toNumber(card.chance);
  const swapped = toNumber((card.controls || {}).swapped);
  const unseen = toNumber((card.controls || {}).unseen_classes);
  if (acc === null || chance === null) return null;
  if (card.kind === "vector") {
    const bar = chance + 0.25;
    if (acc + EPS < bar || swapped === null || acc + EPS < swapped + 0.25) return false;
    if (card.scope === "open" && (unseen === null || unseen + EPS < bar || unseen + EPS < acc - 0.2)) return false;
    return true;
  }
  if (card.kind === "text" || card.kind === "tool") {
    if (swapped !== null && acc + EPS < swapped + 0.1) return false;
    return acc + EPS >= chance + 0.15;
  }
  return null;
}
const SHOWN = {
  installed: { kind: "installed", tag: "installed", node: "installed" },
  refused: { kind: "refused", tag: "refused", node: "refused" },
  candidate: { kind: "candidate", tag: "candidate", node: "default" },
  "not-needed": { kind: "candidate", tag: "not installed", node: "default" },
};
const bestCard = (cards) => cards.reduce((a, c) => (a && toNumber(a.accuracy) >= toNumber(c.accuracy) ? a : c), null);
const NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
const numberWord = (n) => NUMBER_WORDS[n] || String(n);

const currentModel = () => state.models.find((m) => m.name === state.model) || null;
const currentDataset = () => state.datasets.find((d) => d.name === state.dataset) || null;
const modelLabel = (name) => (state.models.find((m) => m.name === name) || {}).label || name || "the model";
const datasetTitle = (name) => (state.datasets.find((d) => d.name === name) || {}).title || name || "";
const jobRunning = () => Boolean(state.job && state.job.status === "running");
/** The job to draw for the current model: the latest run, if it was for this model. */
const modelJob = () => (state.job && (!state.job.request || state.job.request.model === state.model) ? state.job : null);
/** The modality a job is about: its request's or, for an auto run before its data step, the detected one. */
const jobModality = (job) => (job ? (job.request && job.request.modality) || job.detected || "" : "");
/** Whether the band can switch to a model: it is listed and available. */
const selectableModel = (name) => state.models.some((m) => m.name === name && m.available);

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
  const auto = $("fit-auto").checked;
  run.disabled = !usable || (auto ? !state.fitFile || !$("fit-goal").value.trim() : !state.dataset) || state.starting || running;
  $("run-label").textContent = running ? "Agent running…" : state.starting ? "Starting…" : "Run agent";
  if (state.starting || running) run.setAttribute("aria-busy", "true");
  else run.removeAttribute("aria-busy");

  const send = $("send-button");
  send.disabled = !usable || state.chatting;
  send.setAttribute("aria-label", state.chatting ? "Waiting for the model" : "Send");
  send.replaceChildren(icon(state.chatting ? "loader" : "arrow-up"));
  if (state.chatting) send.setAttribute("aria-busy", "true");
  else send.removeAttribute("aria-busy");
  const working = state.chats.find((chat) => chat.messages.some((m) => m.status === "pending"));
  const elsewhere = working && working.id !== state.chatId;
  const activity = $("chat-activity");
  activity.hidden = !elsewhere;
  // Do not repeatedly announce or replace a focused navigation button on unrelated updates.
  if (activity.dataset.chat !== String(elsewhere ? working.id : "")) {
    activity.dataset.chat = elsewhere ? String(working.id) : "";
    activity.replaceChildren(...(elsewhere ? [
      h("span", { text: `Replying in “${working.title}”. You can browse chats while it finishes.` }),
      h("button", { type: "button", class: "link-button", text: "View working chat", onclick: () => switchChat(working.id) }),
    ] : []));
  }
  $("chat-stop").hidden = !state.chatting || Boolean(elsewhere);
  $("thread").setAttribute("aria-busy", String(Boolean(working && !elsewhere)));
  for (const button of document.querySelectorAll(".msg-actions button")) button.disabled = state.chatting;

  const label = usable ? model.label : "";
  $("chat-model").textContent = label;
  $("fit-model").textContent = label || "the model";
  $("senses-model").textContent = label || "the model";
  renderStatus();
}

/** The status line on the band's footer: server, model, labelled set, job and senses, all from the API. */
function renderStatus() {
  const server = { loading: ["idle", "connecting"], ok: ["ok", "ok"], error: ["error", "error"] }[state.server];
  $("status-server").replaceChildren(led(server[0]), server[1]);
  const model = currentModel();
  $("status-model").textContent = model ? model.name : "none";
  const dataset = currentDataset();
  $("status-set").textContent = dataset ? dataset.name : "none";
  const job = state.job;
  if (!job) {
    $("status-job").replaceChildren("none");
  } else {
    const tone = job.status === "running" ? "run" : job.status === "done" ? "ok" : job.cancelled ? "idle" : "error";
    const parts = [
      String(job.id).slice(0, 8),
      job.cancelled ? "stopped" : (JOB_WORDS[job.status] || job.status).toLowerCase(),
      `${job.seen} event${job.seen === 1 ? "" : "s"}`,
      job.elapsed !== null ? seconds(job.elapsed) : null,
    ];
    $("status-job").replaceChildren(led(tone), parts.filter(Boolean).join(" · "));
  }
  const refused = refusedRun();
  const active = (model ? (model.native || []).length : 0) + state.senses.length;
  $("status-senses").textContent = model ? `${active} active${refused ? " · 1 refused" : ""}` : "none";
}

// ---------------------------------------------------------------- the New sense popover

function openFit(anchor) {
  const pop = $("fit-popover");
  pop.hidden = false;
  pop.dataset.anchor = anchor === $("add-sense-button") ? "report" : "top";
  if (!pop.open) pop.showModal();
  $("new-sense-button").setAttribute("aria-expanded", "true");
  $("add-sense-button").setAttribute("aria-expanded", "true");
  const target = $("fit-auto").checked ? $("fit-goal") : $("dataset-select");
  target.focus();
}

/** Show the fields of the chosen way to find examples: the agent finds them, or a labelled set. */
function syncFitMode() {
  $("auto-fields").hidden = !$("fit-auto").checked;
  $("manual-fields").hidden = $("fit-auto").checked;
  syncControls();
}

/** After a run that found no labelled examples: reopen New sense on the examples folder field. The goal and
    file of that run are still filled in. */
function openFitForExamples() {
  $("fit-auto").checked = true;
  syncFitMode();
  openFit($("new-sense-button"));
  $("fit-examples").focus();
}

function closeFit(returnFocus) {
  const pop = $("fit-popover");
  if (pop.hidden) return;
  pop.close();
  pop.hidden = true;
  $("new-sense-button").setAttribute("aria-expanded", "false");
  $("add-sense-button").setAttribute("aria-expanded", "false");
  if (returnFocus) $(pop.dataset.anchor === "report" ? "add-sense-button" : "new-sense-button").focus();
}

// ---------------------------------------------------------------- models and the top bar

async function loadModels() {
  const select = $("model-select");
  showError($("model-error"), null);
  select.disabled = true;
  select.replaceChildren(h("option", { value: "", text: "Loading models…" }));
  $("model-label").textContent = "Loading models…";
  state.server = "loading";
  try {
    const models = await api("/api/models");
    state.models = Array.isArray(models) ? models : [];
    state.server = "ok";
  } catch (err) {
    state.models = [];
    state.server = "error";
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
  if (changed) {
    state.senses = [];
    state.reportBridge = null;
  }
  renderModelInfo();
  syncControls();
  if (changed && name) noteInThread(`Now talking to ${modelLabel(name)}.`);
  renderGraph();
  loadSenses();
}

function renderModelInfo() {
  const info = $("model-info");
  const model = currentModel();
  const picker = $("model-picker");
  if (model) {
    const native = (model.native || []).join(", ") || "no input";
    const weights = model.open_weights
      ? "Open weights, so the vector bridge can be tried."
      : "Closed weights, so the vector bridge is not possible.";
    info.textContent = `Accepts ${native} natively. ${weights}${model.note ? ` ${model.note}` : ""}`;
    $("model-label").textContent = model.label;
    $("model-weights").textContent = model.open_weights ? "Open weights" : "Closed weights";
    picker.classList.toggle("is-closed", !model.open_weights);
    picker.title = info.textContent;
  } else {
    if (state.models.length) {
      const notes = state.models.map((m) => `${m.label}: ${m.note || "unavailable"}.`).join(" ");
      info.textContent = `No model can be loaded right now. ${notes}`;
      $("model-label").textContent = "No model is available";
    } else {
      info.textContent = "";
      $("model-label").textContent = $("model-error").hidden ? "Loading models…" : "Models could not be loaded";
    }
    $("model-weights").textContent = "";
    picker.classList.remove("is-closed");
    picker.removeAttribute("title");
  }
  renderChips();
  renderDropModalities();
}

/** Sense chips on the band: native, installed (marked new until used in chat), checking, refused. */
/* The brain card: one region per data colour, lit from the same API data as the senses list. */
let brainView = null;
let brainRegions = {};
const BRAIN_RANK = { refused: 1, native: 2, running: 3, installed: 4 };

function renderBrain() {
  const canvas = $("brain");
  if (!canvas || !window.SenseBrain) return;
  if (!brainView) {
    const installedBridge = (g) => ((brainRegions[g] || {}).senses || []).find((s) => s.bridge);
    brainView = SenseBrain.mount(canvas, {
      onHover: showBrainTip,
      clickable: (g) => Boolean(installedBridge(g)),
      onClick: (g) => { const s = installedBridge(g); if (s) selectReport(s.bridge); },
    });
  }
  const regions = {};
  const add = (modality, entry) => {
    const g = SenseBrain.regionFor(MODALITY_TOKENS[modalityKey(modality)] || "input");
    const r = (regions[g] = regions[g] || { level: null, senses: [] });
    r.senses.push(entry);
    if (!r.level || BRAIN_RANK[entry.level] > BRAIN_RANK[r.level]) r.level = entry.level;
  };
  const model = currentModel();
  for (const m of model ? model.native || [] : []) add(m, { level: "native", title: modalityName(m), sub: "Native" });
  for (const sense of state.senses) {
    const card = senseCard(sense);
    const acc = toNumber(card.accuracy);
    add(sense.modality, { level: "installed", title: modalityName(sense.modality), sub: [bridgeLabel(card), acc !== null ? pct(acc) : ""].filter(Boolean).join(" · "), bridge: sense.bridge });
  }
  const refused = refusedRun();
  if (refused) add(refused.modality, { level: "refused", title: modalityName(refused.modality), sub: `Refused for ${setTitle(refused.dataset)}` });
  const job = modelJob();
  const target = jobModality(job);
  if (job && job.status === "running" && target && !(model ? model.native || [] : []).includes(target)) {
    add(target, { level: "running", title: modalityName(target), sub: "Agent run in progress" });
  }
  brainRegions = regions;
  const levels = {};
  for (const g in regions) levels[g] = regions[g].level;
  brainView.update(levels);
  const lit = Object.keys(regions).filter((g) => regions[g].level !== "refused").length;
  $("brain-meta").textContent = model ? `${lit} of ${Object.keys(SenseBrain.REGIONS).length} regions` : "";
  const entries = Object.entries(regions).flatMap(([g, r]) => r.senses.map((s) => ({ ...s, region: SenseBrain.REGIONS[g].role })));
  const summary = entries.map((s) => `${s.region}: ${s.title}, ${s.sub}`).join(". ");
  canvas.setAttribute("aria-label", summary || "Choose a model to see its senses");
  // Text and keyboard equivalents of the canvas hover/click targets.
  const access = $("brain-access");
  const signature = JSON.stringify(entries);
  if (access.dataset.signature !== signature) {
    access.dataset.signature = signature;
    access.replaceChildren(...entries.map((s) => h("li", {},
      s.bridge ? h("button", {
        type: "button", class: "link-button", text: `${s.title} · ${s.sub}`,
        "aria-label": `${s.region}: ${s.title}, ${s.sub}. Open report`,
        onclick: () => selectReport(s.bridge),
      }) : h("span", { text: `${s.title} · ${s.sub}` }))));
  }
}

function showBrainTip(g, e) {
  const tip = $("brain-tip");
  if (!g || !e) { tip.hidden = true; return; }
  const region = SenseBrain.REGIONS[g];
  const senses = (brainRegions[g] || {}).senses || [];
  // Native replaceChildren would print a null child as the text "null", so leave it out.
  tip.replaceChildren(...[
    h("div", { class: "brain-tip-head", text: region.role }),
    ...(senses.length
      ? senses.map((s) => h("div", { class: `brain-tip-row brain-tip-row--${s.level}` }, h("span", { text: s.title }), h("span", { text: s.sub })))
      : [h("div", { class: "brain-tip-row", text: "No sense here yet" })]),
    senses.some((s) => s.bridge) ? h("div", { class: "brain-tip-hint", text: "Click to open its report card" }) : null,
  ].filter(Boolean));
  tip.hidden = false;
  const stage = tip.parentElement.getBoundingClientRect();
  tip.style.left = `${Math.min(e.clientX - stage.left + 12, stage.width - tip.offsetWidth - 4)}px`;
  tip.style.top = `${Math.min(e.clientY - stage.top + 12, stage.height - tip.offsetHeight - 4)}px`;
}

function renderChips() {
  renderBrain();
  const list = $("sense-chips");
  const model = currentModel();
  if (!model) {
    list.replaceChildren();
    return;
  }
  const chip = (cls, modality, meta, extra, title) =>
    h("li", { class: `chip ${cls}`.trim(), title, style: `--swatch: ${modalityColour(modality)}` },
      modalityIcon(modality, 13),
      h("span", { class: "chip-label", text: modalityName(modality) }),
      h("span", { class: "chip-meta", text: meta }),
      extra);
  const chips = (model.native || []).map((m) => chip("", m, "native", null, `${modalityName(m)}: native`));
  for (const sense of state.senses) {
    const card = senseCard(sense);
    const fresh = !state.usedBridges.has(sense.bridge);
    const acc = toNumber(card.accuracy);
    const meta = [isAdapter(sense.bridge) ? "adapter" : kindLabel(card.kind).toLowerCase(), acc !== null ? pct(acc) : null].filter(Boolean).join(" · ");
    chips.push(chip("", sense.modality, meta, fresh ? tag("New", "new") : null, `${modalityName(sense.modality)}: ${sense.bridge}`));
  }
  const job = modelJob();
  const target = jobModality(job) || null;
  if (job && job.status === "running" && target && !(model.native || []).includes(target) && !state.senses.some((s) => s.modality === target)) {
    chips.push(chip("chip--checking", target, "checking", null, `The agent is giving ${modelLabel(state.model)} a ${target} sense`));
  }
  const refused = refusedRun();
  if (refused && !state.senses.some((s) => s.modality === refused.modality)) {
    chips.push(chip("chip--refused", refused.modality, "refused", null, `${modalityName(refused.modality)}: the agent installed no bridge`));
  }
  list.replaceChildren(...chips);
}

/** Modality tiles in the empty chat's drop zone: what the model can take in right now. */
function renderDropModalities() {
  const list = $("drop-modalities");
  const model = currentModel();
  const modalities = [...new Set([...(model ? model.native || [] : []), ...state.senses.map((s) => s.modality)])];
  list.replaceChildren(
    ...modalities.map((m) =>
      h("li", { class: "mod", style: `--mod: ${modalityColour(m)}`, title: modalityName(m) }, modalityIcon(m, 14), h("span", { class: "sr-only", text: modalityName(m) })),
    ),
  );
}

// ---------------------------------------------------------------- senses

async function loadSenses() {
  const list = $("senses-list");
  const model = state.model;
  showError($("senses-error"), null);
  if (!model) {
    state.senses = [];
    list.replaceChildren(h("li", { class: "sense-empty", text: "Choose a base model to see its senses." }));
    $("senses-count").textContent = "";
    renderChips();
    renderDropModalities();
    renderReport();
    renderGraph();
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

/** A sense from /api/senses as a full report card (the card fields win over the summary fields). */
function senseCard(sense) {
  return {
    bridge: sense.bridge,
    kind: sense.kind,
    modality: sense.modality,
    dataset: sense.dataset,
    verdict: "installed",
    controls: {},
    ...(sense.card || {}),
  };
}

/** The latest finished run for this model that installed nothing, as {modality, dataset, best card}. */
function refusedRun() {
  const job = modelJob();
  if (!job || job.status !== "done" || job.installed || !job.request || !job.cards.size) return null;
  const best = bestCard([...job.cards.values()]);
  return { modality: job.request.modality || (best && best.modality) || "audio", dataset: job.request.dataset, best };
}

/** A bridge the run refused (it failed its checks) while it installed another one instead: the most
    accurate such bridge, whatever order the planner tried them in. */
function fallbackRun(job) {
  if (!job || job.status !== "done" || !job.installed) return null;
  const refused = [...job.cards.values()].filter((c) => c.bridge !== job.installed && c.verdict === "rejected" && cardFailed(c, job));
  if (!refused.length) return null;
  return { refused: bestCard(refused), installed: job.cards.get(job.installed) || null };
}

/** A modality icon in its data colour, for rows and tiles. */
const modalityMark = (modality, size = 15) => {
  const svg = modalityIcon(modality, size);
  svg.style.color = modalityColour(modality);
  return svg;
};

/** Senses on the model: native, installed, and a run that installed nothing. */
function renderSenses(senses) {
  state.senses = senses;
  const list = $("senses-list");
  const model = currentModel();
  const rows = [];
  for (const m of model ? model.native || [] : []) {
    rows.push(
      h("li", { class: "sense-row sense-row--native" },
        modalityMark(m),
        h("span", { class: "sense-text" }, h("span", { class: "sense-title", text: modalityName(m) }), h("span", { class: "sense-sub", text: "Read by the model as is" })),
        h("span"),
        tag("Native")),
    );
  }
  for (const sense of senses) rows.push(senseItem(sense));
  if (!senses.length) {
    rows.push(h("li", { class: "sense-empty", text: `${modelLabel(state.model)} has no installed senses yet. Press Add a sense to run the agent.` }));
  }
  const refused = refusedRun();
  if (refused) {
    const acc = refused.best ? toNumber(refused.best.accuracy) : null;
    rows.push(
      h("li", { class: "sense-row sense-row--refused" },
        modalityMark(refused.modality),
        h("button", { type: "button", class: "sense-text", "data-bridge": refused.best ? refused.best.bridge : "", onclick: () => selectReport(refused.best && refused.best.bridge) },
          h("span", { class: "sense-title", text: modalityName(refused.modality) }),
          h("span", { class: "sense-sub", text: `Nothing installed for ${setTitle(refused.dataset)}` })),
        acc !== null ? h("span", { class: "sense-value", text: pct(acc) }) : h("span"),
        tag("Refused", "danger")),
    );
  }
  list.replaceChildren(...rows);
  const active = (model ? (model.native || []).length : 0) + senses.length;
  $("senses-count").textContent = `${active} active${refused ? " · 1 refused" : ""}`;
  renderStatus();
  renderChips();
  renderDropModalities();
  renderReport();
  renderGraph();
}

function senseItem(sense) {
  const card = senseCard(sense);
  const acc = toNumber(card.accuracy);
  const errorEl = h("div", { class: "error", role: "alert", hidden: true });
  const remove = h("button", {
    type: "button",
    class: "btn btn--small btn--remove",
    text: "Remove",
    "aria-label": `Remove the ${sense.modality || ""} sense ${sense.bridge || ""}`.replace(/\s+/g, " "),
  });
  remove.addEventListener("click", () => removeSense(sense, remove, errorEl));
  const sub = [bridgeLabel(card), setTitle(card.dataset)].filter(Boolean).join(" · ");
  return h(
    "li",
    { class: "sense-row sense-row--on" },
    modalityMark(sense.modality),
    h("button", { type: "button", class: "sense-text", "data-bridge": sense.bridge, title: `Show the report card of ${sense.bridge}`, onclick: () => selectReport(sense.bridge) },
      h("span", { class: "sense-title", text: modalityName(sense.modality) }),
      h("span", { class: "sense-sub", text: sub })),
    acc !== null ? h("span", { class: "sense-value", text: pct(acc) }) : h("span"),
    remove,
    errorEl,
  );
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

/** Cards the report panel can show: this model's latest run, then its installed senses. */
function reportCards() {
  const job = modelJob();
  const fromRun = job ? [...job.cards.values()] : [];
  const seen = new Set(fromRun.map((c) => c.bridge));
  return [...fromRun, ...state.senses.map(senseCard).filter((c) => !seen.has(c.bridge))];
}

function selectReport(bridge) {
  if (!bridge) return;
  state.reportBridge = bridge;
  setTab("report");
  renderReport();
  renderGraph();
  const heading = $("report-heading");
  heading.focus({ preventScroll: true });
  heading.scrollIntoView({ block: "nearest" });
}

/** The report card: verdict, accuracy, every control against chance, and the agent's reason. */
function renderReport() {
  const el = $("report");
  const verdictTag = $("report-verdict");
  const all = reportCards();
  const job = modelJob();
  const exportButton = $("export-button");
  state.reportCard = null;
  exportButton.disabled = true;

  if (!all.length) {
    if (job && job.status === "running") {
      // The agent is still verifying and no card has arrived.
      setTag(verdictTag, "Verifying", "running");
      const kinds = planKinds(job);
      el.replaceChildren(
        h("p", {
          class: "report-sub",
          text: [
            jobModality(job) ? `${modalityName(jobModality(job))} → ${modelLabel(state.model)}` : modelLabel(state.model),
            job.goal || null,
            kinds.length && isAdapterFlow(job) ? `candidate: an agent-written adapter (${kinds[0] === "image" ? "images" : "text"})` : null,
            kinds.length && !isAdapterFlow(job) ? `candidates: ${kinds.map((k) => `${k} bridge`).join(", ")}` : null,
          ]
            .filter(Boolean)
            .join(" · "),
        }),
        callout("info", "Install rule", isAdapterFlow(job) ? ADAPTER_RULE_TEXT : RULE_TEXT, { iconName: "info" }),
      );
      return;
    }
    verdictTag.hidden = true;
    el.replaceChildren(
      h("p", {
        class: "report-empty",
        text: state.model
          ? `No report card yet. When the agent runs for ${modelLabel(state.model)}, each bridge it verifies gets a card here: held-out accuracy against swapped input, no input, a lookup and the text bridge.`
          : "Choose a base model to see its report cards.",
      }),
    );
    return;
  }
  const card = all.find((c) => c.bridge === state.reportBridge) || all.find((c) => c.verdict === "installed") || all[all.length - 1];
  state.reportBridge = card.bridge;
  state.reportCard = card;
  exportButton.disabled = false;
  const shown = SHOWN[shownVerdict(card)];
  const kind = shown.kind;
  if (kind === "installed") setTag(verdictTag, "Installed", "success");
  else if (kind === "refused") setTag(verdictTag, "Refused", "danger");
  else setTag(verdictTag, shown.tag, "neutral");

  const chance = toNumber(card.chance);
  const n = toNumber(card.n);
  const acc = toNumber(card.accuracy);
  const adapter = isAdapter(card.bridge);
  const dataset = state.datasets.find((d) => d.name === card.dataset);
  const nLabels = dataset && Array.isArray(dataset.labels) ? dataset.labels.length : null;
  const layer = job ? fitData(job, card.bridge).layer : null;
  // An adapter's probe set is named by a hash; the run's data step says where its files came from.
  const probes = adapter && job && job.cards.get(card.bridge) === card ? probeFacts(job) : null;
  const context = [
    `${modalityName(card.modality)} → ${modelLabel(card.model || state.model)}`,
    adapter ? bridgeLabel(card).toLowerCase() : `${kindLabel(card.kind).toLowerCase()} bridge`,
    toNumber(layer) !== null ? `block ${layer}` : null,
    probes ? `probe files: ${probeSourceText(probes)}` : setTitle(card.dataset),
  ].filter(Boolean);
  const interval = Array.isArray(card.interval) && card.interval.length === 2 ? card.interval.map(toNumber) : null;
  const headlineNote = [
    chance !== null && chance > 0 ? `${capitalise(numberWord(Math.round(1 / chance)))}-option questions` : null,
    chance !== null ? `chance ${pct(chance)}` : null,
    interval && interval[0] !== null && interval[1] !== null
      ? `95% CI ${(interval[0] * 100).toFixed(1)}–${(interval[1] * 100).toFixed(1)}`
      : null,
  ].filter(Boolean);
  const unseenMeasured = toNumber((card.controls || {}).unseen_categories) !== null;
  const scope = adapter
    ? unseenMeasured
      ? "Scored on probe files the adapter never saw, and on categories it was not built around."
      : "Scored on probe files the adapter never saw while it was written."
    : card.scope === "closed"
      ? `Scope: ${nLabels ? `the ${nLabels} categories it was fitted on` : "the categories it was fitted on"}.`
      : card.scope === "open"
        ? toNumber((card.controls || {}).unseen_classes) !== null
          ? "Scope: any sound, also tested on categories it was not fitted on."
          : "Scope: any sound. This bridge was not tested on categories it was not fitted on; the agent runs that test for the vector bridge."
        : "";

  const controls = Object.entries(card.controls || {}).sort(
    ([a], [b]) => (CONTROL_ORDER.indexOf(a) + 1 || 99) - (CONTROL_ORDER.indexOf(b) + 1 || 99),
  );
  // A control scored on another set of labels (unseen categories) has its own chance: card.control_chance.
  const ownChance = card.control_chance && typeof card.control_chance === "object" ? card.control_chance : {};
  el.replaceChildren(...[
    h("p", { class: "report-sub", text: context.join(" · "), title: card.bridge }),
    h("div", { class: "readout brackets" },
      h("div", { class: "display" },
        h("span", { class: "display-value", text: acc === null ? "n/a" : pct(acc) }),
        h("span", { class: "display-caption", text: n !== null ? `on ${n} held-out ${adapter ? "probe files" : "questions"}` : "held-out accuracy" })),
      headlineNote.length ? h("p", { class: "report-chance", text: headlineNote.join(" · ") }) : null),
    h("div", { class: "metrics" },
      metricRow(adapter ? bridgeLabel(card) : `${kindLabel(card.kind)} bridge`, acc, kind, chance, shown.tag),
      controls.map(([key, value]) => {
        const role = CONTROL_ROLES[key] || { kind: "control", tag: "control" };
        const own = toNumber(ownChance[key]);
        const label = adapter && key === "swapped" ? "Swapped input" : CONTROL_LABELS[key] || humanise(key);
        return metricRow(label, value, role.kind, own !== null ? own : chance, role.tag,
          own !== null && own !== chance ? `own chance ${pct(own)}` : null);
      })),
    chance !== null ? h("p", { class: "chance-key", text: `chance ${pct(chance)}` }) : null,
    callout(
      kind === "refused" ? "danger" : kind === "installed" ? "success" : "neutral",
      kind === "refused" ? "Why it was refused" : kind === "installed" ? "Why it was installed" : "Agent's reason",
      [scope, card.reason].filter(Boolean).join(" ") || "The agent gave no reason for this card.",
    ),
    adapter ? callout("neutral", "Install rule for adapters", ADAPTER_RULE_TEXT, { iconName: "info" }) : null,
  ].filter(Boolean)); // native replaceChildren would print a null child as the text "null"
}

/** Metric row: label and badge, value, a bar, and the chance tick (at the control's own chance when it has one,
    named in `note`). */
function metricRow(label, value, kind, chance, tagLabel, note = null) {
  const v = toNumber(value);
  const tone = kind === "installed" ? "success" : kind === "refused" ? "danger" : "neutral";
  return h(
    "div",
    { class: `metric metric--${kind}` },
    h("div", { class: "metric-head" },
      h("span", { class: "metric-label-group" },
        h("span", { class: "metric-label", text: label, title: label }),
        tagLabel ? tag(tagLabel, tone) : null,
        note ? h("span", { class: "metric-chance", text: note }) : null),
      h("span", { class: "metric-value", text: v === null ? "n/a" : pct(v) })),
    h("div", { class: "track", "aria-hidden": "true" },
      h("div", { class: "fill", style: `width: ${v === null ? 0 : clamp01(v) * 100}%` }),
      chance !== null ? h("div", { class: "tick", style: `left: ${clamp01(chance) * 100}%` }) : null),
  );
}

/** Export report: the card on screen, as the API returned it, saved as JSON. */
function exportReport() {
  const card = state.reportCard;
  if (!card) return;
  const blob = new Blob([JSON.stringify(card, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = h("a", { href: url, download: `report-${String(card.bridge || "card").replace(/[^\w.-]+/g, "_")}.json` });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ---------------------------------------------------------------- sense graph

/** Bridge kinds the planner chose, then any kind that shows up in fit events or cards. */
function planKinds(job) {
  const kinds = [];
  const design = job.steps.get("design");
  const designed = design ? design.events.findLast((e) => e.status === "ok" && e.data && e.data.route) : null;
  if (designed) kinds.push(designed.data.route === "images" ? "image" : "text");
  const plan = job.steps.get("plan");
  if (plan) for (const e of plan.events) if (Array.isArray((e.data || {}).kinds)) kinds.splice(0, kinds.length, ...e.data.kinds);
  const fit = job.steps.get("fit");
  if (fit) for (const e of fit.events) if (e.data && e.data.kind && !kinds.includes(e.data.kind)) kinds.push(e.data.kind);
  for (const c of job.cards.values()) if (c.kind && !kinds.includes(c.kind)) kinds.push(c.kind);
  return kinds;
}

/** What the fit events say about a bridge (by name) or a kind: {bridge, seconds, layer, started, built}. */
function fitData(job, bridgeOrKind) {
  const out = { bridge: null, seconds: null, layer: null, started: false, built: false };
  const fit = job.steps.get("fit");
  if (!fit) return out;
  for (const e of fit.events) {
    const d = e.data || {};
    if (d.bridge !== bridgeOrKind && d.kind !== bridgeOrKind) continue;
    if (e.status === "start") out.started = true;
    if (e.status === "ok") {
      out.built = true;
      out.bridge = d.bridge || out.bridge;
      out.seconds = toNumber(d.seconds);
      out.layer = toNumber(d.layer);
    }
  }
  return out;
}

/** Whether the verify step has started on a bridge and not finished it yet. */
function verifying(job, bridge) {
  const verify = job.steps.get("verify");
  if (!verify || job.status !== "running" || !bridge) return false;
  let open = false;
  for (const e of verify.events) {
    if ((e.data || {}).bridge === bridge && e.status === "start") open = true;
    const c = cardFrom(e.data);
    if (c && c.bridge === bridge) open = false;
  }
  return open;
}

/** One port row: labels inside, square ports on the node edge, tagged for wiring. */
function portRow(inLabel, outLabel, ids, colours, extraClass = "") {
  return h("div", { class: `port-row ${extraClass}`.trim() },
    h("span", { class: "port-label", text: inLabel || "" }),
    outLabel ? h("span", { class: "port-label", text: outLabel }) : null,
    ids.in ? h("span", { class: "port port--in", "data-port": ids.in, style: `--port: ${colours.in || wireColour()}` }) : null,
    ids.out ? h("span", { class: "port port--out", "data-port": ids.out, style: `--port: ${colours.out || wireColour()}` }) : null);
}

/** Graph node: index, icon, title and badge in a head strip tinted by state, port rows, footer. */
function graphNode({ nodeState = "default", idx, iconEl, title, subtitle, ports, footer, status, statusTone, badge, onclick, pressed }) {
  const tagName = onclick ? "button" : "div";
  const attrs = { class: `node node--${nodeState}${pressed ? " is-selected" : ""}` };
  if (onclick) Object.assign(attrs, { type: "button", onclick, "aria-pressed": String(Boolean(pressed)) });
  return h(tagName, attrs,
    h("span", { class: "node-head" },
      idx ? h("span", { class: "node-idx", text: idx, "aria-hidden": "true" }) : null,
      h("span", { class: "node-tile" }, iconEl),
      h("span", { class: "node-title", text: title }),
      badge || null),
    subtitle ? h("span", { class: "node-sub", text: subtitle }) : null,
    ports && ports.length ? h("span", { class: "node-ports" }, ports) : null,
    footer || status
      ? h("span", { class: "node-foot" },
          footer ? h("span", { class: "node-chip", text: footer }) : h("span"),
          status ? h("span", { class: "node-status", style: `--dot: ${statusTone || "var(--mcs-green-500)"}` }, led(), status) : null)
      : null);
}

/** The sense graph from real data: labelled set → candidate bridges with the held-out check → base model. */
function renderGraph() {
  const canvas = $("graph-canvas");
  const model = currentModel();
  const job = modelJob();
  const datasetName = job && job.request ? job.request.dataset : state.dataset;
  const dataset = state.datasets.find((d) => d.name === datasetName) || null;
  const modality = jobModality(job) || (dataset && dataset.modality) || "audio";
  $("graph-sub").textContent = model ? `${modalityName(modality)} → ${model.label}` : "";
  if (!model) {
    canvas.replaceChildren(h("p", { class: "graph-empty", text: "Choose a base model to see how a sense reaches it." }));
    return;
  }
  let counter = 0;
  const nextIdx = () => `N${String((counter += 1)).padStart(2, "0")}`;

  // Column 1: the labelled examples the agent fits and verifies on.
  const cards = job ? [...job.cards.values()] : [];
  const n = cards.length ? toNumber(cards[0].n) : null;
  const scopeInput = document.querySelector('input[name="scope"]:checked');
  const scope = job && job.request ? job.request.scope : scopeInput ? scopeInput.value : "closed";
  const adapterRun = isAdapterFlow(job);
  // An adapter run's probe files stand where a labelled set would.
  const dataNode = adapterRun ? probeNode(job, n, nextIdx) : dataset
    ? graphNode({
        idx: nextIdx(),
        iconEl: icon("database"),
        title: dataset.title,
        subtitle: `${(dataset.labels || []).length} labels · ${dataset.n} ${dataset.modality || "audio"} clips`,
        ports: [
          portRow("", "examples to fit", { out: "data-fit" }, { out: wireColour("data") }),
          portRow("", n !== null ? `${n} held out` : "held out", { out: "data-held" }, { out: wireColour("data") }),
        ],
        footer: dataset.name,
        status: scope === "open" ? "any sound" : "these labels only",
        statusTone: "var(--mcs-gray-600)",
      })
    : h("p", {
        class: "graph-empty",
        text: job && job.request && !datasetName
          ? job.status === "running" ? "The agent has not chosen a labelled set yet." : "This run chose no labelled set."
          : "No labelled set is chosen.",
      });

  // Column 2: the candidate bridges and the held-out check, inside the bridge group.
  const group = h("div", { class: "bridge-group" }, h("div", { class: "group-head" }, icon("plug", 13), h("span", { text: "Candidate bridges" })));
  if (!job) {
    group.append(h("p", { class: "group-empty", text: "The agent chooses which bridges to try when it runs. Press New sense to start." }));
  } else {
    const kinds = planKinds(job);
    if (kinds.length) group.querySelector(".group-head").append(h("span", { class: "group-count", text: `· ${kinds.length} tried` }));
    else group.append(h("p", { class: "group-empty", text: job.status === "running" ? "Planning…" : "No bridge was tried." }));
    for (const kind of kinds) {
      const fd = fitData(job, kind);
      const card = cards.find((c) => c.kind === kind) || null;
      const bridge = (card && card.bridge) || fd.bridge;
      // An adapter run has one candidate, its design; it is written at the build step and scored at verify.
      const designed = adapterRun ? ((job.steps.get("design")?.events.findLast((e) => e.status === "ok") || {}).data || {}) : null;
      const build = designed ? job.steps.get("build") : null;
      const buildState = build ? stepStatus("build", build, job) : "pending";
      let nodeState = "default";
      let status = "queued";
      let tone = "var(--mcs-gray-600)";
      if (card) {
        nodeState = SHOWN[shownVerdict(card)].node;
        status = pct(card.accuracy);
        tone = nodeState === "refused" ? "var(--mcs-red-500)" : nodeState === "installed" ? "var(--mcs-green-500)" : "var(--mcs-gray-400)";
      } else if (designed && job.status === "running" && (buildState === "running" || (buildState === "done" && job.steps.has("verify")))) {
        nodeState = "running";
        status = buildState === "running" ? "writing" : "testing";
        tone = "var(--mcs-blue-500)";
      } else if (designed && buildState === "done") {
        status = job.status === "running" ? "built" : "not verified";
      } else if (designed && job.status !== "running") {
        nodeState = "skipped";
        status = buildState === "refused" ? "did not run" : "not built";
      } else if (job.status === "running" && ((fd.started && !fd.built) || verifying(job, bridge))) {
        nodeState = "running";
        status = fd.built ? "testing" : "fitting";
        tone = "var(--mcs-blue-500)";
      } else if (fd.built) {
        status = job.status === "running" ? "built" : "not verified";
      } else if (job.status !== "running") {
        nodeState = "skipped";
        status = "not built";
      }
      const ports = KIND_PORTS[kind] || ["input", "output"];
      const badge =
        nodeState === "installed" ? tag("Installed", "success")
        : nodeState === "refused" ? tag("Refused", "danger")
        : nodeState === "running" ? tag("Running", "running")
        : nodeState === "skipped" ? tag("Skipped")
        : card ? tag(SHOWN[shownVerdict(card)].tag)
        : null;
      group.append(
        graphNode({
          nodeState,
          idx: nextIdx(),
          iconEl: icon("plug"),
          title: designed ? `Adapter (${kind === "image" ? "images" : "text"})` : `${kindLabel(kind)} bridge`,
          subtitle: designed ? designed.description || "Agent-written code in the sandbox" : [KIND_NOTES[kind], fd.layer !== null ? `block ${fd.layer}` : null].filter(Boolean).join(" · "),
          ports: [portRow(ports[0], ports[1], { in: `bridge-in-${kind}`, out: `bridge-out-${kind}` }, { in: wireColour("data"), out: wireColour(KIND_WIRE[kind]) })],
          footer: fd.seconds !== null ? `fit ${seconds(fd.seconds)}` : bridge || kind,
          status,
          statusTone: tone,
          badge,
          onclick: card ? () => selectReport(card.bridge) : null,
          pressed: Boolean(card && card.bridge === state.reportBridge),
        }),
      );
    }
    const nControls = cards.reduce((m, c) => Math.max(m, Object.keys(c.controls || {}).length), 0);
    const checking = job.status === "running" && job.steps.has("verify") && !job.steps.has("decide");
    group.append(
      h("div", { class: `held-check${checking ? " is-running" : ""}` },
        h("span", { class: "port port--in", "data-port": "held-in", style: `--port: ${wireColour("data")}` }),
        icon(checking ? "loader" : "shield-check", 15),
        h("span", { class: "held-text" },
          h("span", { class: "held-title", text: "Held-out check" }),
          h("span", {
            class: "held-sub",
            text: cards.length
              ? `${n !== null ? `${n} questions` : "held-out questions"} · ${nControls} controls · ${cards.length} of ${kinds.length} scored`
              : checking ? "scoring" : "not started",
          }))),
    );
  }

  // Column 3: the base model, with what it reads natively and the sense being added.
  const sense = state.senses.find((s) => s.modality === modality) || null;
  const refused = refusedRun();
  const natives = model.native || [];
  let targetState = "no sense";
  if (natives.includes(modality)) targetState = "native"; // an auto run on a file the model reads as is
  else if (sense) targetState = state.usedBridges.has(sense.bridge) ? "installed" : "new";
  else if (job && job.status === "running") targetState = "pending";
  else if (refused && refused.modality === modality) targetState = "refused";
  const modelPorts = [
    portRow(`${modality} · ${targetState}`, "reply", { in: "model-in-target", out: "model-out" },
      { in: modalityColour(modality), out: wireColour("text") },
      `port-row--target port-row--${targetState.replace(/\s+/g, "-")}`),
    ...natives.filter((m) => m !== modality).map((m) => portRow(`${m} · native`, "", { in: `model-in-${m}` }, { in: modalityColour(m) })),
  ];
  const modelNode = graphNode({
    nodeState: "model",
    idx: nextIdx(),
    iconEl: icon("cpu"),
    title: model.label,
    subtitle: `Base model · ${model.open_weights ? "open" : "closed"} weights`,
    ports: modelPorts,
    footer: model.name,
    status: model.available ? "ready" : "unavailable",
    statusTone: model.available ? "var(--mcs-green-500)" : "var(--mcs-red-500)",
  });
  modelNode.dataset.installedKind = sense ? sense.kind : "";

  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("class", "wires");
  svg.setAttribute("aria-hidden", "true");
  canvas.replaceChildren(
    svg,
    h("div", { class: "graph-col" }, dataNode),
    h("div", { class: "graph-col graph-col--bridges" }, group),
    h("div", { class: "graph-col" }, modelNode),
  );
  drawWires();
}

/** The graph's first column for an adapter run: its probe files, all held out, or why there are none yet. */
function probeNode(job, n, nextIdx) {
  const probes = probeFacts(job);
  if (!probes) {
    return h("p", { class: "graph-empty", text: job.status === "running" ? "The agent has not found probe files yet." : "This run found no probe files." });
  }
  return graphNode({
    idx: nextIdx(),
    iconEl: icon("database"),
    title: probeSourceText(probes),
    subtitle: `${(probes.labels || []).length} labels · ${probes.n} ${probes.provenance} probe files`,
    ports: [
      portRow("", "never seen by the adapter", { out: "data-fit" }, { out: wireColour("data") }),
      portRow("", n !== null ? `${n} scored` : "all scored", { out: "data-held" }, { out: wireColour("data") }),
    ],
    footer: probes.licence || probes.name,
    status: "all held out",
    statusTone: "var(--mcs-gray-600)",
  });
}

/** Draw the wires between ports at right angles, coloured by the data that flows. */
function drawWires() {
  const canvas = $("graph-canvas");
  const svg = canvas.querySelector("svg.wires");
  if (!svg || canvas.hidden) return;
  svg.replaceChildren();
  const box = canvas.getBoundingClientRect();
  const at = (id) => {
    const el = canvas.querySelector(`[data-port="${id}"]`);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return {
      x: Math.round(r.left + r.width / 2 - box.left + canvas.scrollLeft),
      y: Math.round(r.top + r.height / 2 - box.top + canvas.scrollTop),
    };
  };
  const wire = (from, to, colour, dashed, faint) => {
    const a = at(from);
    const b = at(to);
    if (!a || !b) return;
    const mx = Math.round((a.x + b.x) / 2);
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", a.y === b.y ? `M${a.x},${a.y} H${b.x}` : `M${a.x},${a.y} H${mx} V${b.y} H${b.x}`);
    path.setAttribute("class", `wire${dashed ? " wire--dashed" : ""}${faint ? " wire--faint" : ""}`);
    path.style.stroke = colour;
    svg.append(path);
  };
  // Measure the content without the old SVG size, which otherwise prevents the canvas shrinking.
  svg.setAttribute("width", "0");
  svg.setAttribute("height", "0");
  svg.setAttribute("width", String(canvas.scrollWidth));
  svg.setAttribute("height", String(canvas.scrollHeight));
  const job = modelJob();
  if (!job) return;
  const modelNode = canvas.querySelector(".node[data-installed-kind]");
  const installedKind = modelNode ? modelNode.dataset.installedKind : "";
  for (const kind of planKinds(job)) {
    const card = [...job.cards.values()].find((c) => c.kind === kind);
    const refused = Boolean(card && shownVerdict(card) === "refused");
    const into = (card && card.verdict === "installed") || (!card && installedKind === kind);
    wire("data-fit", `bridge-in-${kind}`, wireColour("data"), false, false);
    wire(`bridge-out-${kind}`, "model-in-target", refused ? "var(--mcs-red-500)" : wireColour(KIND_WIRE[kind]), !into, refused);
  }
  wire("data-held", "held-in", wireColour("data"), true, false);
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
    state.datasetsLoaded = true;
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
  // Reloading (a run built a new set) keeps the user's choice and its samples.
  const saved = store.get("localStorage", "sense.dataset");
  const pick = state.datasets.find((d) => d.name === state.dataset) || state.datasets.find((d) => d.name === saved) || state.datasets[0];
  select.value = pick.name;
  setDataset(pick.name, pick.name !== state.dataset);
  if (state.job) renderTraceHead();
  renderReport(); // dataset titles and label counts may now be known
  renderRuns();
  if (state.model && state.senses.length) renderSenses(state.senses);
}

function setDataset(name, reloadSamples = true) {
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
  renderGraph();
  if (reloadSamples) loadSamples();
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

/** A sample clip as an Attachment row (Figma 22:234), with play and attach controls. */
function sampleItem(sample) {
  const name = sample.name || sample.id;
  const modality = (currentDataset() || {}).modality || "audio";
  const play = h("button", { type: "button", class: "icon-button" });
  play.addEventListener("click", () => togglePlay(sample));
  const attach = h("button", { type: "button", class: "btn btn--small" });
  attach.addEventListener("click", () => toggleSample(sample));
  const item = h(
    "li",
    { class: "attachment", title: name },
    h("span", { class: "tile", style: `color: ${modalityColour(modality)}` }, modalityIcon(modality, 16)),
    h("span", { class: "attachment-text" },
      h("span", { class: "attachment-name", text: sample.label || name }),
      h("span", { class: "attachment-meta", text: name })),
    play,
    attach,
  );
  state.sampleRefs.set(sample.id, { item, play, attach, name });
  syncSampleItem(sample.id);
  return item;
}

/** Update one sample's play and attach controls in place, so keyboard focus stays put. */
function syncSampleItem(id) {
  if (id === null || id === undefined) return;
  const ref = state.sampleRefs.get(id);
  const attached = state.attachedSamples.has(id);
  const playing = state.playing === id;
  if (ref) {
    ref.item.classList.toggle("is-attached", attached);
    ref.item.classList.toggle("is-playing", playing);
    ref.attach.setAttribute("aria-pressed", String(attached));
    ref.attach.textContent = attached ? "Attached" : "Attach";
    ref.attach.setAttribute("aria-label", `${attached ? "Detach" : "Attach"} ${ref.name}`);
    ref.play.replaceChildren(icon(playing ? "stop" : "play", 14));
    ref.play.setAttribute("aria-label", `${playing ? "Stop" : "Play"} ${ref.name}`);
  }
  // Play buttons on attachments already sent in the chat follow the same clip.
  for (const b of document.querySelectorAll("[data-play-sample]")) {
    if (b.dataset.playSample !== id) continue;
    b.replaceChildren(icon(playing ? "stop" : "play", 14));
    b.setAttribute("aria-label", `${playing ? "Stop" : "Play"} ${b.dataset.name || "clip"}`);
  }
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
      attachmentChip(file.name, fileModality(file), `${fileKindName(file)} · upload`, () => {
        state.files = state.files.filter((f) => f !== file);
        renderAttachments();
      }),
    ),
    ...[...state.attachedSamples.values()].map((sample) =>
      attachmentChip(sample.name || sample.id, (currentDataset() || {}).modality || "audio", `sample · ${sample.label || ""}`.replace(/ · $/, ""), () => toggleSample(sample)),
    ),
  ];
  list.replaceChildren(...chips);
  list.hidden = chips.length === 0;
}

/* File extensions for the icons of modalities browsers give no MIME type for. Icons only. */
const EXTENSION_MODALITIES = {
  "3d": ["glb", "gltf", "obj", "stl", "ply", "fbx", "off", "3mf", "dae", "usdz"],
  molecule: ["sdf", "mol", "mol2", "pdb", "xyz", "smi", "smiles", "cif", "mmcif"],
  geo: ["geojson", "gpx", "kml", "kmz", "shp"],
  table: ["csv", "tsv", "parquet"],
  genome: ["fasta", "fa", "fastq", "fna"],
};

/** The modality an upload's MIME type or extension suggests, for its icon and label only; the server decides
    from the file's bytes (agent.detect). */
function fileModality(file) {
  const type = (file && file.type) || "";
  const ext = String((file && file.name) || "").toLowerCase().split(".").slice(1).pop() || "";
  const byExt = Object.keys(EXTENSION_MODALITIES).find((m) => EXTENSION_MODALITIES[m].includes(ext));
  if (byExt) return byExt;
  if (type.startsWith("model/")) return "3d";
  return ["audio", "image", "video", "text"].find((m) => type.startsWith(`${m}/`)) || "other";
}
/** How a sent file is labelled under its name: "3D model", "Audio", or "File" when nothing suggests more. */
const fileKindName = (file) => (fileModality(file) === "other" ? "File" : modalityName(fileModality(file)));

/** A file attached to the next message, as a compact Attachment (Figma 22:234). */
function attachmentChip(name, modality, meta, onRemove) {
  return h(
    "li",
    { class: "attachment" },
    h("span", { class: "tile", style: `color: ${modalityColour(modality)}` }, modalityIcon(modality, 14)),
    h("span", { class: "attachment-text" },
      h("span", { class: "attachment-name", text: name, title: name }),
      h("span", { class: "attachment-meta", text: meta })),
    h("button", { type: "button", class: "icon-button", "aria-label": `Remove ${name}`, onclick: onRemove }, icon("x", 14)),
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
  if (message.startsWith("/")) {
    runCommand(message);
    box.value = "";
    autosize(box);
    $("cmd-hints").hidden = true;
    return;
  }
  if (state.chatting) return;
  const model = currentModel();
  if (!model || !model.available) {
    showError(errorEl, new Error("Choose an available base model in the top bar first."));
    return;
  }
  if (!message) {
    showError(errorEl, new Error("Type a question for the model first."));
    box.focus();
    return;
  }

  const files = [...state.files];
  const samples = [...state.attachedSamples.values()];
  const chat = currentChat();
  if (state.editing && state.editing.record.files.some((f) => !files.some((picked) => picked.name === f.name))) {
    showError(errorEl, new Error("Reattach the original uploaded files before resending this message."));
    return;
  }
  if (state.editing && state.editing.chat === chat) truncateChat(chat, state.editing.record);
  state.editing = null;
  const record = { model: { name: model.name, label: model.label }, message, files, samples, status: "pending" };
  chat.messages.push(record);
  record.turn = drawMessage(record, chat);
  box.value = "";
  autosize(box);
  state.files = [];
  state.attachedSamples.clear();
  for (const sample of samples) syncSampleItem(sample.id);
  renderAttachments();
  await requestReply(chat, record);
}

/** Retry/edit replaces this turn and its follow-ups, so history never includes a stale answer. */
function truncateChat(chat, record) {
  const index = chat.messages.indexOf(record);
  if (index < 0) return;
  for (const removed of chat.messages.splice(index)) removed.turn?.root.remove();
}

async function requestReply(chat, record) {
  const { model, message, files, samples } = record;
  const form = new FormData();
  form.append("model", model.name);
  form.append("message", message);
  const index = chat.messages.indexOf(record);
  const history = chat.messages.slice(0, index).filter((m) => m.status === "done" && m.model.name === model.name)
    .slice(-6).flatMap((m) => [{ role: "user", content: m.message }, { role: "assistant", content: m.result.reply || "" }]);
  form.append("history", JSON.stringify(history));
  for (const file of files) form.append("files", file, file.name);
  for (const sample of samples) form.append("sample_ids", sample.id);

  record.status = "pending";
  chat.updatedAt = Date.now();
  record.error = null;
  record.turn.reply.replaceChildren(h("p", { class: "msg-text pending", text: "Waiting for the model" }));
  record.turn.tags.replaceChildren();
  state.chatting = true;
  state.chatAbort = new AbortController();
  saveChats();
  renderChats();
  syncControls();
  try {
    const result = await streamReply(form, record.turn, state.chatAbort.signal);
    record.result = result || {};
    record.status = "done";
    fillReply(record.turn, record.result);
  } catch (err) {
    record.status = "failed";
    record.error = err.name === "AbortError" ? "Stopped receiving the reply. The server may still be finishing generation." : err.message;
    record.turn.reply.replaceChildren(h("p", { class: "error", role: "alert", text: record.error }));
  } finally {
    state.chatting = false;
    state.chatAbort = null;
    chat.updatedAt = Date.now();
    saveChats();
    renderChats();
    syncControls();
  }
}

/** SSE is framed by blank lines, not network chunks. Older servers fall back before generation starts. */
async function streamReply(form, turn, signal) {
  const response = await fetch("/api/chat/stream", { method: "POST", body: form, signal });
  if (response.status === 404 || response.status === 405) {
    const fallback = await fetch("/api/chat", { method: "POST", body: form, signal });
    const body = await fallback.json();
    if (!fallback.ok) throw new ApiError(detailText(body, fallback.status), fallback.status);
    return body;
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new ApiError(detailText(body, response.status), response.status);
  }
  if (!(response.headers.get("content-type") || "").includes("text/event-stream") || !response.body) {
    throw new Error("The server did not return a reply stream.");
  }
  const text = h("p", { class: "msg-text" });
  turn.reply.replaceChildren(text);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let lineBuffer = "", eventType = "", dataLines = [], reply = "", used = [], result = null;
  const line = (value) => {
    if (value === "") {
      if (dataLines.length) {
        const data = JSON.parse(dataLines.join("\n"));
        if (eventType === "used") used = Array.isArray(data) ? data : [];
        else if (eventType === "token") {
          reply += data.text || "";
          text.textContent = reply;
          if (turn.root.isConnected && nearEnd($("thread"))) scrollToEnd($("thread"));
        } else if (eventType === "done") {
          // done.used is final: a tool input's detail now says whether the model called the tool.
          result = { reply: data.reply ?? reply, used: Array.isArray(data.used) ? data.used : used };
        } else if (eventType === "error") throw new Error(data.detail || "The model failed to answer.");
      }
      eventType = "";
      dataLines = [];
    } else if (value.startsWith("event:")) eventType = value.slice(6).trim();
    else if (value.startsWith("data:")) dataLines.push(value.slice(5).replace(/^ /, ""));
  };
  try {
    while (!result) {
      const chunk = await reader.read();
      const lines = (lineBuffer + decoder.decode(chunk.value, { stream: !chunk.done })).split("\n");
      lineBuffer = lines.pop();
      for (const value of lines) line(value.replace(/\r$/, ""));
      if (chunk.done) {
        if (lineBuffer) line(lineBuffer.replace(/\r$/, ""));
        line("");
        if (!result) throw new Error("The reply stream ended early. Retry this message.");
      }
    }
    return result;
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

function drawMessage(record, chat) {
  const turn = appendTurn(record.model, record.message, record.files, record.samples);
  if (record.status === "done") fillReply(turn, record.result || {});
  else if (record.status !== "pending") turn.reply.replaceChildren(h("p", { class: "error", text: record.error || "Reply interrupted. Retry to send again." }));
  turn.root.append(h("div", { class: "msg-actions" },
    h("button", { type: "button", class: "link-button", text: "Retry", onclick: () => retryMessage(chat, record) }),
    h("button", { type: "button", class: "link-button", text: "Edit", onclick: () => editMessage(chat, record) })));
  return turn;
}

function editMessage(chat, record) {
  if (state.chatting) return;
  state.editing = { chat, record };
  $("message").value = record.message;
  state.files = record.files.filter((f) => f instanceof File);
  state.attachedSamples = new Map(record.samples.map((s) => [s.id, s]));
  if (state.models.some((m) => m.name === record.model.name && m.available)) {
    $("model-select").value = record.model.name;
    setModel(record.model.name);
  }
  renderAttachments();
  autosize($("message"));
  $("message").focus();
  showError($("chat-error"), new Error(record.files.some((f) => !(f instanceof File))
    ? "Reattach the uploaded files before resending. Sending replaces this message and its follow-ups."
    : "Editing this message. Sending replaces it and its follow-ups. Escape cancels."));
}

function retryMessage(chat, record) {
  if (state.chatting) return;
  if (record.files.some((f) => !(f instanceof File))) { editMessage(chat, record); return; }
  if (!state.models.some((m) => m.name === record.model.name && m.available)) {
    showError($("chat-error"), new Error("This message's model is unavailable. Edit it to select an available model."));
    return;
  }
  if (state.editing) cancelEdit();
  showError($("chat-error"), null);
  truncateChat(chat, record);
  chat.messages.push(record);
  record.turn = drawMessage(record, chat);
  requestReply(chat, record);
}

function cancelEdit() {
  state.editing = null;
  $("message").value = "";
  state.files = [];
  state.attachedSamples.clear();
  renderAttachments();
  showError($("chat-error"), null);
  autosize($("message"));
}

/** A user turn (name line, attachments, message) and the model's answer under its own name line. */
function appendTurn(model, message, files, samples) {
  const modality = (currentDataset() || {}).modality || "audio";
  const sent = [
    ...samples.map((s) =>
      h("li", { class: "attachment" },
        h("span", { class: "tile", style: `color: ${modalityColour(modality)}` }, modalityIcon(modality, 16)),
        h("span", { class: "attachment-text" },
          h("span", { class: "attachment-name", text: s.name || s.id }),
          h("span", { class: "attachment-meta", text: `${modalityName(modality)} · sample · ${s.label || ""}`.replace(/ · $/, "") })),
        h("button", {
          type: "button",
          class: "icon-button",
          "data-play-sample": s.id,
          "data-name": s.name || s.id,
          "aria-label": `Play ${s.name || s.id}`,
          onclick: () => togglePlay(s),
        }, icon(state.playing === s.id ? "stop" : "play", 14))),
    ),
    ...files.map((f) =>
      h("li", { class: "attachment" },
        h("span", { class: "tile", style: `color: ${modalityColour(fileModality(f))}` }, modalityIcon(fileModality(f), 16)),
        h("span", { class: "attachment-text" },
          h("span", { class: "attachment-name", text: f.name }),
          h("span", { class: "attachment-meta", text: `${fileKindName(f)} · upload` }))),
    ),
  ];
  const reply = h("div", { class: "msg-body" }, h("p", { class: "msg-text pending", text: "Waiting for the model" }));
  const tags = h("span", { class: "name-line-tags" });
  const root = h(
    "article",
    { class: "turn" },
    h("div", { class: "msg msg--user" },
      h("div", { class: "name-line" }, icon("user", 13), h("span", { class: "name", text: "You" })),
      sent.length ? h("ul", { class: "msg-files", "aria-label": "Attached files" }, sent) : null,
      h("p", { class: "bubble", text: message })),
    h("div", { class: "msg msg--model" },
      h("div", { class: "name-line" }, icon("cpu", 13), h("span", { class: "name", text: model.label }), tags),
      reply),
  );
  $("thread").append(root);
  updateThreadEmpty();
  scrollToEnd($("thread"));
  return { root, reply, tags };
}

/** A tool input the model could have called but did not (server/app.py: "the model could call <tool> but did not"). */
const toolNotCalled = (u) => u.kind === "tool" && /\bcould call \S+ but did not\b/.test(u.detail || "");

function fillReply(turn, result) {
  const used = Array.isArray(result.used) ? result.used : [];
  const heard = used.filter((u) => u.kind !== "none" && !toolNotCalled(u));
  const ways = [...new Set(heard.map(usedTag))];
  const missing = used.some((u) => u.kind === "none");
  const uncalled = used.some(toolNotCalled);
  turn.tags.replaceChildren(
    ...ways.map((label) => tag(label, "info")),
    ...(uncalled ? [tag("Tool not called", "warning")] : []),
    ...(missing ? [tag("File not received", "danger")] : []),
  );
  const replyText = result.reply || "The model returned an empty reply.";
  const copy = h("button", { type: "button", class: "icon-button msg-copy", title: "Copy the reply", "aria-label": "Copy the reply" }, icon("copy", 13));
  copy.addEventListener("click", () => {
    if (navigator.clipboard) navigator.clipboard.writeText(replyText).then(() => copy.replaceChildren(icon("check", 13)), () => {});
  });
  const parts = [h("p", { class: "msg-text", text: replyText }), copy];
  if (used.length) parts.push(h("ul", { class: "used-list", "aria-label": "How each file reached the model" }, used.map(usedItem)));
  turn.reply.replaceChildren(...parts);
  let changed = false;
  for (const u of heard) {
    if (u.bridge && !state.usedBridges.has(u.bridge)) {
      state.usedBridges.add(u.bridge);
      changed = true;
    }
  }
  if (changed) {
    renderChips();
    renderGraph();
  }
  scrollToEnd($("thread"));
}

/** The name-line tag for a file that reached the model: read as is, through an agent-written adapter, or a bridge. */
function usedTag(used) {
  if (used.kind === "native") return "Read as is";
  if (isAdapter(used.bridge)) return "Read via adapter";
  return `Heard via ${kindLabel(used.kind).toLowerCase()} bridge`;
}

/** One line per file: how it reached the model, or a warning when it did not. */
function usedItem(used) {
  const missing = used.kind === "none";
  // Kind "native" (bridge ""): the model took the file as it is, a text file's text or an image. An agent-written
  // adapter (bridge "adapter:<slug>", kind "text" or "image") ran its code on the file in the sandbox.
  const how = used.kind === "native" ? "Read as is"
    : isAdapter(used.bridge) ? `Read via ${used.bridge}`
    : `${kindLabel(used.kind)} bridge ${used.bridge || ""}`.trim();
  const detail = missing
    ? used.detail || "No sense for this kind of file is installed, so the model did not receive it."
    : [how, used.detail].filter(Boolean).join(" · ");
  return h(
    "li",
    { class: missing ? "used-item used-item--warn" : "used-item" },
    h("strong", { text: `${used.source || "file"}${missing ? " not received" : ""}` }),
    detail ? `: ${detail}` : "",
  );
}

/** The agent's verdict in the chat when a run ends, as a callout. */
function postAgentResult(job, data) {
  const cards = [...job.cards.values()];
  const modality = modalityName(jobModality(job) || (cards[0] && cards[0].modality) || "sense");
  const installedCard = cards.find((c) => c.bridge === data.installed) || null;
  const fallback = fallbackRun(job);
  const done = job.doneData || {};
  let result;
  if (job.cancelled) {
    result = callout("warning", `You stopped the ${job.replay ? "replay" : "run"}`, null, { iconName: "stop" });
  } else if (job.status === "failed") {
    result = resultCard(job, "refused", "The run failed", "", data.error || "The server gave no error text. Check the server log, then run the agent again.", null, null);
  } else if (!cards.length && !data.installed) {
    // The done message, shown above, already says why; the callout only names the outcome.
    if (done.needs_user) {
      // A folder of examples helps once the agent got as far as looking for data (agent/auto.py).
      result = callout("warning", "The agent needs labelled examples", null, {
        iconName: "alert",
        action: job.steps.has("data") && !job.replay
          ? h("button", { type: "button", class: "link-button", text: "Run again with a folder of examples →", onclick: openFitForExamples })
          : null,
      });
    } else if (done.native) {
      result = callout("info", "No sense needed", null, { iconName: "check-circle" });
    } else {
      result = callout("info", "Agent finished", job.doneMessage ? null : "No bridge was installed.");
    }
  } else if (fallback) {
    // Screen 03: a bridge failed its checks and the agent installed another one instead.
    const r = fallback.refused;
    const i = fallback.installed;
    const body = [r.reason, i ? `Installed the ${kindLabel(i.kind).toLowerCase()} bridge instead: ${pct(i.accuracy)}.` : null].filter(Boolean).join(" ");
    result = resultCard(job, "refused", `${kindLabel(r.kind)} bridge refused`, pct(r.accuracy), body, r.bridge, "See why →");
  } else if (data.installed) {
    const c = installedCard;
    const facts = c
      ? [bridgeLabel(c), isAdapter(c.bridge) && probeFacts(job) ? `probe files: ${probeSourceText(probeFacts(job))}` : setTitle(c.dataset), toNumber(c.n) !== null ? `${c.n} held-out ${isAdapter(c.bridge) ? "probe files" : "questions"}` : null, toNumber(c.chance) !== null ? `chance ${pct(c.chance)}` : null]
      : [data.installed];
    result = resultCard(job, "installed", `${modality} installed`, c && toNumber(c.accuracy) !== null ? pct(c.accuracy) : "", facts.filter(Boolean).join(" · "), data.installed, null);
  } else {
    const best = bestCard(cards);
    result = resultCard(
      job,
      "refused",
      `${modality} refused`,
      best && toNumber(best.accuracy) !== null ? pct(best.accuracy) : "",
      best && best.reason ? best.reason : "No candidate passed its checks.",
      best && best.bridge,
      "See why →",
    );
  }
  // The done event's message already names an earlier sense the run replaced or kept (done.data).
  const text = job.cancelled ? "" : job.doneMessage;
  const msg = h(
    "article",
    { class: "turn msg msg--agent" },
    h("div", { class: "name-line" },
      icon("agent", 13),
      h("span", { class: "name", text: "Agent" }),
      job.elapsed !== null ? h("span", { class: "name-meta", text: `· ${seconds(job.elapsed)}` }) : null),
    text ? h("p", { class: "msg-text", text }) : null,
    result,
  );
  $("thread").append(msg);
  updateThreadEmpty();
  scrollToEnd($("thread"));
}

/** The run's outcome as a callout. Its link opens the card, switching the band to the run's model first; a run
    recorded for a model this server cannot load gets no link, since its card cannot be shown. */
function resultCard(job, verdict, title, value, body, bridge, linkText) {
  const model = job.request && job.request.model;
  const unshowable = Boolean(bridge && model && model !== state.model && !selectableModel(model));
  if (unshowable) {
    body = [body, `This run was for ${modelLabel(model)}, which this server cannot load, so its report card cannot be shown here.`].filter(Boolean).join(" ");
  }
  return callout(verdict === "installed" ? "success" : "danger", title, body, {
    value,
    iconName: verdict === "installed" ? "check-circle" : "ban",
    action: bridge && !unshowable
      ? h("button", { type: "button", class: "link-button", text: linkText || "Open report card →", onclick: () => openRunReport(job, bridge) })
      : null,
  });
}

/** Open a run's report card. The report shows the selected model's runs, so select the run's model first. */
function openRunReport(job, bridge) {
  const model = job.request && job.request.model;
  if (model && model !== state.model && selectableModel(model)) chooseModel(model);
  selectReport(bridge);
}

/** Select a model in the band, as if the user had picked it. */
function chooseModel(name) {
  $("model-select").value = name;
  setModel(name);
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

/** Grow the composer with its text, up to the CSS max-height. */
function autosize(box) {
  box.style.height = "auto";
  box.style.height = `${box.scrollHeight}px`;
}

// ---------------------------------------------------------------- agent run and trace

async function runAgent(event) {
  event.preventDefault();
  const errorEl = $("fit-error");
  showError(errorEl, null);
  if (state.starting || jobRunning()) return;
  const model = currentModel();
  const dataset = currentDataset();
  const auto = $("fit-auto").checked;
  if (!model || !model.available) {
    showError(errorEl, new Error("Choose an available base model first."));
    return;
  }
  if (!auto && !dataset) {
    showError(errorEl, new Error("Choose a set of labelled examples first."));
    return;
  }
  if (auto && (!state.fitFile || !$("fit-goal").value.trim())) {
    showError(errorEl, new Error("Drop a file and describe what you want the model to understand."));
    return;
  }
  const scope = document.querySelector('input[name="scope"]:checked');
  const request = auto ? { model: model.name, goal: $("fit-goal").value.trim() } : {
    model: model.name,
    dataset: dataset.name,
    modality: dataset.modality || "audio",
    scope: scope ? scope.value : "closed",
  };
  state.starting = true;
  syncControls();
  try {
    const form = new FormData();
    if (auto) {
      form.append("model", request.model);
      form.append("goal", request.goal);
      form.append("file", state.fitFile, state.fitFile.name);
      const examples = $("fit-examples").value.trim();
      if (examples) form.append("examples", examples); // a folder on the server, one sub-folder per label
    }
    const result = await api(auto ? "/api/auto" : "/api/fit", auto ? { method: "POST", body: form } : {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (!result || !result.job_id) throw new Error("The server accepted the run but returned no job_id, so it cannot be followed.");
    state.starting = false;
    closeFit(false);
    startJob(result.job_id, request, false);
  } catch (err) {
    if (auto && (err.status === 404 || err.status === 405)) {
      err = new Error("Automatic discovery is not available on this server yet. Choose a labelled set, or update the server.");
    }
    showError(errorEl, err);
    if (err.status === 409 && !state.job) followRunElsewhere(); // a run from another tab holds the slot: show it
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
    lastT: null, // the newest event's t, and when this page first saw it (Date.now()), for the clock while waiting
    lastAt: 0,
    failures: 0,
    timer: null,
    steps: new Map(), // step -> {events, open}: what each Agent step card shows
    cards: new Map(), // bridge name -> its latest report card
    cardJson: new Map(), // bridge name -> last card as JSON, to skip identical redraws
    installed: null,
    doneMessage: "",
    doneData: {}, // the done event's data: installed, and native or needs_user when an auto run ended early
    // The server's request for an auto run has no goal, and dataset and modality stay "" until the data step,
    // so the goal and the detected modality are kept here (from the request, then the detect event).
    goal: (request && typeof request.goal === "string" && request.goal) || "",
    detected: "",
    flow: "", // the detect event's data.flow: "native", "specialist" or "adapter"
    approval: null, // the open request (pending_approval) the card shows, with what the user answered here
    approvalNote: null, // the agent's line in the chat thread about that request
    stepEls: new Map(), // step key -> {sig, el}: unchanged steps keep their element, focus and open details
    openDetails: new Set(), // step keys whose code block the user opened
    cancelled: false, // ended "failed" with error "Cancelled.": the user stopped it
  };
  state.reportBridge = null;
  store.set("sessionStorage", "sense.job", id);
  $("timeline").replaceChildren();
  $("log-count").textContent = "";
  $("log-button").disabled = false;
  $("trace-outcome").replaceChildren();
  $("approval").replaceChildren();
  showError($("trace-error"), null);
  if (!resumed) setTab("run");
  renderTraceHead();
  renderSteps();
  renderGraph();
  renderReport();
  renderChips();
  renderRuns();
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
      resetTrace(); // the run remembered from before a reload is gone; show the empty strip again
      return;
    }
    job.status = "lost";
    store.set("sessionStorage", "sense.job", null);
    showError($("trace-error"), new Error(`The trace stopped updating. ${err.message}`));
    renderTraceHead();
    renderSteps();
    renderGraph();
    syncControls();
    return;
  }
  if (state.job !== job) return;
  job.failures = 0;
  showError($("trace-error"), job.cancelError || null);
  applyJob(job, data || {});
  if (job.status === "running") job.timer = setTimeout(() => poll(job), POLL_MS);
  else finishJob(job, data || {});
}

function applyJob(job, data) {
  if (data.request) job.request = data.request;
  if (data.replay) job.replay = data.replay;
  job.status = typeof data.status === "string" ? data.status : "running";
  const events = Array.isArray(data.events) ? data.events : [];
  if (events.length < job.seen) {
    // The server's list shrank (for example after a restart): redraw from scratch.
    $("timeline").replaceChildren();
    job.seen = 0;
    job.steps.clear();
    job.cards.clear();
    job.cardJson.clear();
    job.stepEls.clear();
  }
  const logView = $("log-view");
  const follow = !logView.hidden && nearEnd(logView);
  let cardsChanged = false;
  for (const event of events.slice(job.seen)) {
    appendEvent(job, event);
    const card = cardFrom(event.data);
    if (card) cardsChanged = upsertCard(job, card) || cardsChanged;
  }
  job.seen = events.length;
  const lastT = events.length ? toNumber(events[events.length - 1].t) : null;
  if (lastT !== job.lastT) {
    job.lastT = lastT;
    job.lastAt = Date.now();
  }
  // No event comes while the run waits for your approval (up to 10 minutes), so its clock goes on from the last
  // event by this page's clock rather than standing still.
  const waiting = job.status === "running" && !job.replay && Boolean(data.pending_approval) && lastT !== null;
  if (events.length) job.elapsed = waiting ? lastT + (Date.now() - job.lastAt) / 1000 : lastT;
  // agent.jobs and server.replay end a stopped run "failed" with error "Cancelled."; a replay's last event says so too.
  const last = events[events.length - 1];
  job.cancelled = job.status === "failed"
    && (data.error === "Cancelled." || Boolean(last && last.step === "error" && last.data && last.data.cancelled === true));
  for (const card of Array.isArray(data.cards) ? data.cards : []) cardsChanged = upsertCard(job, card) || cardsChanged;
  // A set the run built (a fetched set or an examples folder) is new to the list loaded at start.
  const dataset = job.request && job.request.dataset;
  if (dataset && state.datasetsLoaded && job.datasetsReloaded !== dataset && !state.datasets.some((d) => d.name === dataset)) {
    job.datasetsReloaded = dataset;
    loadDatasets();
  }
  trackApproval(job, data.pending_approval);
  renderTraceHead();
  renderSteps();
  renderApproval(job);
  renderGraph();
  if (cardsChanged || job.status === "running") renderReport();
  if (follow) scrollToEnd(logView);
}

/** True when the reader is at (or close to) the bottom, so new trace lines should keep it there. */
function nearEnd(el) {
  return el.scrollHeight - el.scrollTop - el.clientHeight < 120;
}

/** Record one trace event: a line in the full log, and its step's card. */
function appendEvent(job, event) {
  const status = STATUS_WORDS[event.status] ? event.status : "info";
  const step = event.step || "step";
  const mark = h("span", { class: "event-mark", "aria-hidden": "true" }, icon(status === "ok" ? "check" : status === "fail" ? "x" : status === "start" ? "circle" : "info", 12));
  $("timeline").append(
    h("li", { class: `event event--${status}` },
      h("span", { class: "event-t", text: seconds(event.t) }),
      h("span", { class: "event-step", text: step }),
      mark,
      h("span", { class: "event-message" }, h("span", { class: "sr-only", text: `${STATUS_WORDS[status]}: ` }), event.message || "")),
  );
  $("log-count").textContent = String($("timeline").children.length);

  if (step === "done") {
    job.doneMessage = event.message || "";
    job.doneData = event.data && typeof event.data === "object" ? event.data : {};
    return;
  }
  if (step === "error") return;
  if (step === "detect" && event.status === "ok" && event.data) {
    if (typeof event.data.goal === "string" && event.data.goal) job.goal = event.data.goal;
    if (typeof event.data.modality === "string") job.detected = event.data.modality;
    if (typeof event.data.flow === "string") job.flow = event.data.flow;
  }
  let s = job.steps.get(step);
  if (!s) {
    s = { events: [], open: 0 };
    job.steps.set(step, s);
  }
  s.events.push(event);
  if (status === "start") s.open += 1;
  else if (status === "ok" || status === "fail" || (status === "info" && INFO_CLOSES.has(step))) s.open = Math.max(0, s.open - 1);
}

const isCard = (x) => Boolean(x && typeof x === "object" && typeof x.bridge === "string" && "accuracy" in x);

/** A verify event carries a ReportCard dict, either as its data or under data.card. */
function cardFrom(data) {
  if (!data || typeof data !== "object") return null;
  if (isCard(data.card)) return data.card;
  if (isCard(data)) return data;
  return null;
}

/** Keep a bridge's latest card; returns true when it changed. The report follows the newest card. */
function upsertCard(job, card) {
  if (!isCard(card)) return false;
  const json = JSON.stringify(card);
  if (job.cardJson.get(card.bridge) === json) return false;
  job.cardJson.set(card.bridge, json);
  job.cards.set(card.bridge, card);
  if (job.status === "running") state.reportBridge = card.bridge;
  return true;
}

function finishJob(job, data) {
  store.set("sessionStorage", "sense.job", null);
  job.installed = data.installed || null;
  const outcome = $("trace-outcome");
  if (job.cancelled) {
    // The user pressed Stop: not a failure, so no alert.
    outcome.replaceChildren(callout("warning", job.replay ? "Replay stopped" : "Run stopped", `You stopped the ${job.replay ? "replay" : "run"} before it finished.`, { iconName: "stop" }));
  } else if (job.status === "failed") {
    const failed = callout("danger", "Run failed", data.error || "The server gave no error text. Check the server log, then run the agent again.", { iconName: "alert" });
    failed.setAttribute("role", "alert");
    outcome.replaceChildren(failed);
  } else {
    outcome.replaceChildren();
  }
  const cards = [...job.cards.values()];
  const fallback = fallbackRun(job);
  const installedCard = cards.find((c) => c.bridge === job.installed);
  const best = bestCard(cards);
  state.reportBridge = fallback ? fallback.refused.bridge : installedCard ? installedCard.bridge : best ? best.bridge : null;
  trackApproval(job, null, true);
  renderApproval(job);
  renderTraceHead();
  renderSteps();
  renderReport();
  renderGraph();
  postAgentResult(job, data);
  if (state.tab === "run") {
    // The Run tab is hidden for the report: a reader whose focus was in it (the approval card's outcome, a step,
    // Stop) goes on at the report's heading, not at the start of the page.
    const hadFocus = $("run-view").contains(document.activeElement);
    setTab("report");
    if (hadFocus) $("report-heading").focus({ preventScroll: true });
  }
  syncControls();
  loadSenses();
  renderRuns();
  if (!job.replay) setTimeout(loadRuns, 800); // the server records a finished run
}

/** A step's state from its own events. Only the step's last outcome counts: a sub-step it recovered from (a
    dataset search that was unavailable, a candidate set that could not be used, then the prewired set) leaves
    the step done. A step left unfinished, or whose last outcome is a failure, is where the run stopped, failed
    or needed the user. An adapter run's search for unseen categories (data sub-steps with purpose "unseen") comes
    after the data step's own outcome and never decides it: a candidate it rejects is not a failure of the step,
    and a search that found nothing is said by its "not measured" line and the card. */
function stepStatus(key, s, job) {
  if (!s || !s.events.length) return "pending";
  const counts = (e) => !(key === "data" && e.data && e.data.purpose === "unseen");
  const outcome = s.events.findLast((e) => (e.status === "ok" || e.status === "fail") && counts(e));
  const failedLast = Boolean(outcome && outcome.status === "fail");
  if (declined(key, s) && !job.cancelled) return "declined"; // your choice, not a failure: the run goes on
  if (job.status === "running") {
    // The newest step may still recover from its failed sub-step.
    if (s.open > 0 || (failedLast && [...job.steps.keys()].at(-1) === key)) return "running";
    return failedLast ? "refused" : "done";
  }
  if (s.open > 0 || failedLast) return job.cancelled ? "stopped" : "refused";
  if (key === "decide" && job.status === "done" && (!job.installed || fallbackRun(job))) return "refused"; // the agent declined a bridge
  return "done";
}

/** A short, real summary of what a step did, from the data its events carry. */
function stepCode(key, s) {
  if (!s) return key;
  const datas = s.events.map((e) => e.data || {});
  const pick = (k) => {
    for (let i = datas.length - 1; i >= 0; i -= 1) if (datas[i][k] !== undefined && datas[i][k] !== null) return datas[i][k];
    return undefined;
  };
  if (key === "inspect") {
    const m = pick("model");
    return m && typeof m === "object" ? `${m.name} · ${m.open_weights ? "open" : "closed"} weights` : key;
  }
  if (key === "plan") {
    const kinds = pick("kinds");
    const planner = pick("planner");
    return Array.isArray(kinds) ? `${planner ? `${planner}: ` : ""}${kinds.join(", ")}` : key;
  }
  if (key === "fit") {
    const built = datas.map((d) => d.bridge).filter(Boolean);
    return built.length ? built.join(", ") : key;
  }
  if (key === "verify") {
    const cards = datas.map(cardFrom).filter(Boolean);
    return cards.length ? `${cards.length} card${cards.length > 1 ? "s" : ""} · n=${cards[0].n}` : key;
  }
  if (key === "decide") {
    if (!s.events.some((e) => e.status !== "start")) return key;
    return `install: ${pick("install") || "none"}`;
  }
  if (key === "install") return String(pick("bridge") || key);
  if (key === "design") {
    const route = pick("route");
    const libs = pick("libraries");
    return route ? `${route} · ${Array.isArray(libs) && libs.length ? libs.join(", ") : "standard library"}` : key;
  }
  if (key === "build") {
    const attempts = pick("attempts");
    const sha = pick("code_sha256");
    return toNumber(attempts) !== null ? [`${attempts} attempt${attempts === 1 ? "" : "s"}`, sha ? `sha256 ${String(sha).slice(0, 12)}` : null].filter(Boolean).join(" · ") : key;
  }
  if (key === "data") {
    const summary = s.events.findLast((e) => e.status === "ok" && e.data && !e.data.action && e.data.provenance);
    return summary ? `${summary.data.provenance} · n=${summary.data.n}` : key;
  }
  if (key === "install_request") {
    const verdict = pick("verdict");
    return verdict ? `checks: ${verdict}` : key;
  }
  if (key === "approval") {
    const reason = pick("reason");
    return reason ? String(reason) : "waiting";
  }
  return key;
}

function visibleSteps(job = state.job) {
  if (job && isAdapterFlow(job)) {
    const asks = job.steps.has("install_request") || job.steps.has("approval") || designAsksInstall(job);
    return ADAPTER_STEPS.filter((s) => !s.optional || asks);
  }
  const auto = !job || Boolean(job.goal) || STEPS.some((s) => s.auto && job.steps.has(s.key));
  return STEPS.filter((s) => auto || !s.auto);
}

/** Whether a run took the adapter flow: its detect event says so, or (a record from before that field) it has
    the adapter's own steps. */
function isAdapterFlow(job) {
  return Boolean(job) && (job.flow === "adapter" || job.steps.has("design") || job.steps.has("build"));
}

/** Whether the adapter's design asked to install packages (agent.synth: design.install). */
function designAsksInstall(job) {
  const design = job.steps.get("design");
  const ok = design ? design.events.findLast((e) => e.status === "ok") : null;
  return Boolean(ok && ok.data && Array.isArray(ok.data.install) && ok.data.install.length);
}

/** A request you declined (or left unanswered): the approval step's last event fails with that reason, and the
    install_request step ends with info. */
function declined(key, s) {
  const last = s.events[s.events.length - 1];
  if (!last) return false;
  if (key === "approval") return last.status === "fail" && Boolean(last.data) && ["declined", "timed out"].includes(last.data.reason);
  if (key === "install_request") return last.status === "info" && Boolean(last.data) && last.data.approved === false;
  return false;
}

/** Known steps in execution order, followed by any newly introduced server steps. A step whose events and state
    have not changed keeps its element, so a reader's focus and an open code block survive the next poll. */
function renderSteps() {
  const job = state.job;
  const defs = visibleSteps(job);
  const byKey = Object.fromEntries(defs.map((d) => [d.key, d]));
  const extra = job ? [...job.steps.keys()].filter((k) => !byKey[k]) : [];
  const keys = [...defs.map((d) => d.key), ...extra];
  const list = $("steps");
  list.style.setProperty("--steps", String(keys.length));
  const items = keys.map((key, i) => {
    const def = byKey[key] || STEP_INDEX[key] || { title: humanise(key), idle: "" };
    const s = job ? job.steps.get(key) : null;
    const status = job ? stepStatus(key, s, job) : "pending";
    const word = status === "refused" && job && (job.status === "failed" || job.status === "lost") ? STEP_WORDS.failed
      : status === "running" && key === "approval" ? "Waiting"
      : STEP_WORDS[status];
    const events = s ? s.events : [];
    const sig = [i, def.title, status, word, events.length, job ? job.status : "", job ? job.installed : ""].join("|");
    const kept = job ? job.stepEls.get(key) : null;
    if (kept && kept.sig === sig) return kept.el;
    const last = events[events.length - 1];
    let desc = last && last.message ? last.message : def.idle;
    if (!events.length && job && job.status !== "running") desc = key === "install" && job.status === "done" ? "Nothing to install." : "Not reached.";
    const time = events.length > 1 ? toNumber(last.t) - toNumber(events[0].t) : null;
    const code = stepCode(key, s);
    const el = h(
      "li",
      { class: `step step--${status}`, title: last && last.message ? last.message : undefined },
      h("div", { class: "step-head" },
        h("span", { class: "step-idx", text: String(i + 1).padStart(2, "0"), "aria-hidden": "true" }),
        h("span", { class: "step-state" }, h("span", { class: "led", "aria-hidden": "true" }), word),
        h("span", { class: "step-time", text: time !== null && Number.isFinite(time) ? seconds(time) : "" })),
      h("p", { class: "step-title", text: def.title }),
      h("p", { class: "step-desc", text: desc }),
      job && s ? stepDetail(key, s, job) : null,
      h("p", { class: "step-code", text: code, title: code }),
    );
    if (job) job.stepEls.set(key, { sig, el });
    return el;
  });
  // Move only what changed: replacing every child would drop the focus of a reader inside a step.
  items.forEach((el, i) => {
    const current = list.children[i];
    if (current !== el) list.insertBefore(el, current || null);
  });
  while (list.children.length > items.length) list.lastElementChild.remove();
}

// ---------------------------------------------------------------- adapter run details

/** The data step's own summary of an adapter run's probe files (its ok event without an action): name,
    provenance (user | dataset | generated), source, licence, labels, counts, n, fetched, label_leaks, caveat. */
function probeFacts(job) {
  const s = job ? job.steps.get("data") : null;
  const summary = s ? s.events.findLast((e) => e.status === "ok" && e.data && !e.data.action && e.data.provenance) : null;
  return summary ? summary.data : null;
}

/** The unseen categories of an adapter run, from the data step: {measured: true, name, source, labels, per_label,
    n, licence, url} when a second set was scored, {measured: false, note} when it was not, or null. */
function unseenFacts(job) {
  const s = job ? job.steps.get("data") : null;
  if (!s) return null;
  const info = s.events.findLast((e) => e.status === "info" && e.data
    && (e.data.unseen || (e.data.purpose === "unseen" && !e.data.action)));
  if (!info) return null;
  if (info.data.unseen && typeof info.data.unseen === "object") {
    const found = s.events.findLast((e) => e.status === "ok" && e.data && e.data.purpose === "unseen" && e.data.action === "candidate");
    return {
      measured: true, ...info.data.unseen, licence: found ? found.data.licence : null, url: found ? found.data.url : null,
      title: found ? found.data.title || found.data.hub_id || found.data.source : null,
    };
  }
  return { measured: false, note: info.message };
}

/** Where probe files came from, in a few words. */
function probeSourceText(p) {
  if (p.provenance === "generated") return "generated by the agent";
  if (p.provenance === "user") return "your examples folder";
  const fetched = p.fetched && typeof p.fetched === "object" ? p.fetched : {};
  return String(fetched.title || fetched.hub_id || p.source || "a public dataset");
}

/** A source as a link when it is a web address, else as text (a folder on the server). */
function sourceLink(text, url) {
  const href = typeof url === "string" && /^https?:\/\//i.test(url) ? url : null;
  return href ? h("a", { href, target: "_blank", rel: "noopener noreferrer", class: "step-link", text: text || href }) : h("span", { text: text || "" });
}

/** labels {label: count} as "ball 20, box 20". */
const countsText = (labels, counts) => (labels || Object.keys(counts || {}))
  .map((label) => (counts && toNumber(counts[label]) !== null ? `${label} ${counts[label]}` : String(label))).join(", ");

/** One fact row of a step: a label and a value (text or nodes). */
const fact = (label, ...value) => h("div", { class: "fact" }, h("dt", { text: label }), h("dd", {}, ...value));

/** What an adapter run's step did, beyond its last message: the design, the attempts and the code, where the
    probe files came from, the packages asked for. null for the other steps. */
function stepDetail(key, s, job) {
  const lastOk = s.events.findLast((e) => e.status === "ok");
  const d = (lastOk && lastOk.data) || {};
  if (key === "design" && lastOk) {
    const libs = Array.isArray(d.libraries) && d.libraries.length ? d.libraries.join(", ") : "the standard library only";
    const asked = Array.isArray(d.install) ? d.install : [];
    return h("dl", { class: "facts" },
      fact("Reads", `${modalityName(d.modality)}${d.format ? ` (${d.format})` : ""}`),
      fact("Route", d.route === "images" ? "images, shown to the model" : "text, which the model reads"),
      d.description ? fact("Extracts", d.description) : null,
      fact("Libraries", libs),
      d.question ? fact("Question", d.question) : null,
      asked.length ? fact("Asks to install", asked.map((r) => `${r.name}${r.why ? ` (${r.why})` : ""}`).join("; ")) : null);
  }
  if (key === "build") {
    const failures = s.events.filter((e) => e.status === "info" && e.data && toNumber(e.data.attempt) !== null);
    const warning = s.events.find((e) => e.data && e.data.warning === "unsandboxed");
    const ended = s.events.findLast((e) => e.status === "ok" || e.status === "fail");
    const rows = [
      ended ? fact("Attempts", `${ended.status === "ok" ? toNumber(d.attempts) ?? failures.length + 1 : failures.length || toNumber((ended.data || {}).attempts) || "?"}${ended.status === "fail" ? ", none worked" : ""}`) : null,
      failures.length ? fact("Errors", h("ol", { class: "attempts" }, failures.map((e) => h("li", { text: shortError(e) })))) : null,
      warning ? fact("Sandbox", h("span", { class: "fact-warn", text: "none here: the code ran with network and file access" })) : null,
      lastOk && toNumber(d.seconds) !== null ? fact("Ran in", seconds(d.seconds)) : null,
      lastOk && d.preview ? fact("Output", h("q", { text: d.preview })) : null,
    ];
    if (lastOk && typeof d.code === "string") {
      const lines = d.code.replace(/\n+$/, "").split("\n").length;
      rows.push(h("details", {
        class: "code-block",
        open: job.openDetails.has(key),
        ontoggle: (e) => (e.currentTarget.open ? job.openDetails.add(key) : job.openDetails.delete(key)),
      },
        h("summary", { text: `The adapter's code · ${lines} line${lines === 1 ? "" : "s"}${d.code_sha256 ? ` · sha256 ${String(d.code_sha256).slice(0, 12)}` : ""}` }),
        h("pre", { tabindex: "0", "aria-label": "The adapter's code" }, h("code", { text: d.code }))));
    }
    return rows.some(Boolean) ? h("dl", { class: "facts" }, rows) : null;
  }
  if (key === "data") {
    const p = probeFacts(job);
    const unseen = unseenFacts(job);
    const asked = s.events.findLast((e) => e.status === "info" && e.data && e.data.wanted);
    if (!p && !unseen && !asked) return null;
    const rows = [];
    if (p) {
      const provenance = { user: "your examples (user)", dataset: "a public dataset", generated: "generated by the agent" }[p.provenance] || p.provenance;
      const fetched = p.fetched && typeof p.fetched === "object" ? p.fetched : {};
      rows.push(
        fact("Provenance", provenance),
        p.provenance !== "generated" ? fact("Source", sourceLink(probeSourceText(p) === "your examples folder" ? p.source : probeSourceText(p), fetched.url || p.source)) : null,
        p.licence ? fact("Licence", `${p.licence}${fetched.licence_source ? ` (from ${fetched.licence_source})` : ""}`) : null,
        fact("Labels", `${countsText(p.labels, p.counts)}${toNumber(p.n) !== null ? ` · ${p.n} files` : ""}`),
        p.caveat ? fact("Caveat", h("span", { class: "fact-warn", text: p.caveat })) : null,
        Array.isArray(p.label_leaks) && p.label_leaks.length ? fact("Warning", h("span", { class: "fact-warn", text: p.label_leaks.join("; ") })) : null,
      );
    }
    if (unseen && unseen.measured) {
      rows.push(fact("Unseen categories", sourceLink(unseen.title || unseen.source, unseen.url || unseen.source),
        ` · ${countsText(unseen.labels, unseen.per_label)}${unseen.licence ? ` · ${unseen.licence}` : ""}`));
    } else if (unseen) {
      rows.push(fact("Unseen categories", h("span", { class: "fact-warn", text: unseen.note })));
    }
    if (!p && asked) {
      const wanted = asked.data.wanted || {};
      const labels = Array.isArray(wanted.labels) && wanted.labels.length ? wanted.labels.join(", ") : wanted.label_hint;
      if (labels) rows.push(fact("Wanted", `${labels}${wanted.format ? ` · ${wanted.format} files` : ""}${toNumber(wanted.min_per_label) !== null ? ` · at least ${wanted.min_per_label} per label` : ""}`));
    }
    return rows.some(Boolean) ? h("dl", { class: "facts" }, rows) : null;
  }
  if (key === "install_request") {
    const first = s.events[0];
    const requested = (first && first.data && Array.isArray(first.data.requested)) ? first.data.requested : [];
    const last = s.events[s.events.length - 1];
    const ld = (last && last.data) || {};
    const installed = Array.isArray(ld.installed) ? ld.installed : [];
    return h("dl", { class: "facts" },
      requested.length ? fact("Requested", requested.map((r) => `${r.name}${r.why ? ` (${r.why})` : ""}`).join("; ")) : null,
      ld.verdict ? fact("Checks", String(ld.verdict)) : null,
      installed.length ? fact("Installed", installed.map((p) => `${p.name} ${p.version || ""}`.trim()).join(", ")) : null,
      ld.note ? fact("Outcome", String(ld.note)) : null);
  }
  return null;
}

/** A failed attempt in a few words: its error's first line, cut. */
function shortError(event) {
  const err = event.data && event.data.error && typeof event.data.error === "object" ? event.data.error : {};
  const text = String(err.reason || event.message || "").replace(/^Attempt \d+ failed: /, "").replace(/ Asking the LLM for a repair\.$/, "");
  const line = text.split("\n")[0].trim();
  return `${event.data.attempt}: ${line.length > 140 ? `${line.slice(0, 139)}…` : line}`;
}

/** The run's progress as a one-pixel line along the band: the share of the six steps that have finished. */
function renderProgress() {
  const bar = $("run-progress-bar");
  const job = state.job;
  if (!job) {
    bar.style.width = "0";
    delete bar.dataset.state;
    return;
  }
  const steps = visibleSteps(job);
  const finished = steps.filter((s) => ["done", "refused", "stopped", "declined"].includes(stepStatus(s.key, job.steps.get(s.key), job))).length;
  bar.style.width = `${(job.status === "running" ? finished / steps.length : 1) * 100}%`;
  bar.dataset.state = job.status === "running" ? "running" : job.status === "done" ? "done" : job.cancelled ? "stopped" : "failed";
}

function renderTraceHead() {
  const job = state.job;
  const pill = $("trace-status");
  renderProgress();
  renderStatus();
  renderReplayBar();
  $("job-stop").hidden = !job || job.status !== "running";
  $("job-stop").disabled = Boolean(job?.cancelling);
  $("job-stop").textContent = job?.cancelling ? "Stopping…" : "Stop";
  const led = $("run-tab-led");
  led.hidden = !(job && job.status === "running");
  led.className = "led led--run";
  if (!job) {
    pill.hidden = true;
    $("trace-request").textContent = "Not run yet. Press New sense to start the agent.";
    $("trace-request").removeAttribute("title");
    $("trace-elapsed").textContent = "";
    return;
  }
  if (job.status === "running") setTag(pill, "Running", "running");
  else if (job.cancelled) setTag(pill, "Stopped", "warning");
  else if (job.status === "failed") setTag(pill, "Failed", "danger");
  else if (job.status === "lost") setTag(pill, JOB_WORDS.lost, "warning");
  else if (fallbackRun(job)) setTag(pill, "Refused · fallback installed", "warning");
  else if (job.installed) setTag(pill, "Sense installed", "success");
  else if (job.status === "done" && !job.cards.size && job.doneData.needs_user) setTag(pill, "Needs examples", "warning");
  else if (job.status === "done" && !job.cards.size && job.doneData.native) setTag(pill, "No sense needed", "neutral");
  else if (job.status === "done" && !job.cards.size) setTag(pill, "Finished", "neutral");
  else if (job.status === "done") setTag(pill, "Refused", "danger");
  else setTag(pill, JOB_WORDS[job.status] || job.status, "neutral");
  const req = job.request;
  const reached = [...job.steps.keys()].filter((k) => STEP_INDEX[k]).length;
  const probes = isAdapterFlow(job) ? probeFacts(job) : null;
  // An auto run's request has no labelled set (dataset "") until its data step, and so no scope yet either.
  const text = req
    ? [
        job.replay ? "Recorded run" : null,
        modelLabel(req.model),
        job.goal || null,
        probes ? `probe files: ${probeSourceText(probes)}` : req.dataset && !isProbeSet(req.dataset) ? datasetTitle(req.dataset) : null,
        req.dataset && !isProbeSet(req.dataset) ? (SCOPE_LABELS[req.scope] || req.scope || "").toLowerCase() : null,
        job.status === "running" ? `step ${Math.max(1, reached)} of ${visibleSteps(job).length}` : `${reached} step${reached === 1 ? "" : "s"}`,
      ].filter(Boolean).join(" · ")
    : "Resuming the last run…";
  $("trace-request").textContent = text;
  $("trace-request").title = text;
  $("trace-elapsed").textContent = job.elapsed !== null ? seconds(job.elapsed) : "";
}

async function cancelJob() {
  const job = state.job;
  if (!job || job.status !== "running" || job.cancelling) return;
  job.cancelling = true;
  job.cancelError = null;
  renderTraceHead();
  try {
    const result = await api(`/api/jobs/${enc(job.id)}`, { method: "DELETE" });
    if (!result?.cancelled) throw new Error("The run could not be cancelled; it may already have finished.");
  } catch (err) {
    job.cancelling = false;
    job.cancelError = err;
    if (state.job === job) showError($("trace-error"), err);
  }
  if (state.job === job) renderTraceHead();
}

// ---------------------------------------------------------------- approvals

/** Follow the run's open request (GET /api/jobs/<id> pending_approval; agent.jobs). A new request opens its card
    and the Run tab, and the agent says so in the chat. A request that closes with no answer from here (timed out,
    cancelled) closes the card; one answered here shows its outcome until the run has moved on past it (and at least
    OUTCOME_MS). `final`: the run has ended, so the card goes. */
function trackApproval(job, pending, final = false) {
  const open = pending && typeof pending === "object" && typeof pending.id === "string" ? pending : null;
  if (open && !final && (!job.approval || job.approval.id !== open.id)) {
    job.approval = { ...open, answer: null, sending: false, closed: false, answeredAt: 0 };
    job.approvalNote = null;
    noteApproval(job);
    if (state.tab !== "run") setTab("run");
    return;
  }
  const a = job.approval;
  if (!a || a.closed || (open && !final)) return;
  if (a.sending && !final) return; // the answer is on its way: its outcome decides
  if (final || a.answer === null) {
    a.closed = true;
    noteApproval(job);
    return;
  }
  const past = ["detect", "design", "install_request", "approval"];
  const movedOn = job.status !== "running" || [...job.steps.keys()].some((k) => !past.includes(k));
  if (movedOn && Date.now() - a.answeredAt >= OUTCOME_MS) a.closed = true;
}

/** The packages a request names, for sentences: "meshkit" or "trimesh, pygltflib". */
function approvalNames(a) {
  const plan = a && a.payload && typeof a.payload.plan === "object" && a.payload.plan ? a.payload.plan : {};
  const requested = Array.isArray(plan.requested) ? plan.requested.map((r) => r && r.name).filter(Boolean) : [];
  return requested.join(", ") || "the packages";
}

/** Draw the open request's card in the Run tab, or clear it. Redrawn only when the request or its answer
    changes, so the buttons keep their focus while the run is polled. */
function renderApproval(job) {
  const slot = $("approval");
  const a = job && state.job === job ? job.approval : null;
  if (!a || a.closed) {
    // The card goes with the focus in it (its outcome line, after Approve or Decline): the keyboard stays in the
    // run, at its heading, rather than falling back to the start of the page.
    const hadFocus = slot.contains(document.activeElement);
    if (slot.firstChild) slot.replaceChildren();
    delete slot.dataset.sig;
    if (hadFocus) $("trace-heading").focus({ preventScroll: true });
    return;
  }
  const sig = `${a.id}|${a.answer}`;
  if (slot.dataset.sig === sig && slot.firstChild) return;
  slot.dataset.sig = sig;
  slot.replaceChildren(approvalCard(job, a));
}

const VERDICT_TONES = { ok: "success", caution: "warning", refuse: "danger" };

/** The card: what the checks found (agent.installer's plan: one row per package to install, the ones already
    installed folded away), the LLM's opinion apart from them, and Approve / Decline; then the answer. */
function approvalCard(job, a) {
  const payload = a.payload && typeof a.payload === "object" ? a.payload : {};
  const plan = payload.plan && typeof payload.plan === "object" ? payload.plan : {};
  const packages = Array.isArray(plan.packages) ? plan.packages.filter((p) => p && typeof p.name === "string") : [];
  const requested = Array.isArray(plan.requested) ? plan.requested.filter((r) => r && r.name) : [];
  const fresh = packages.filter((p) => !(p.facts && p.facts.already_installed));
  const have = packages.filter((p) => p.facts && p.facts.already_installed);
  const names = approvalNames(a);
  const answered = a.answer !== null;
  const titleId = `approval-title-${a.id}`;
  const install = a.kind === "install";
  const summary = typeof payload.summary === "string" && payload.summary ? payload.summary : plan.summary;
  const minutes = toNumber(a.timeout) !== null ? (a.timeout >= 120 ? `${Math.round(a.timeout / 60)} minutes` : `${Math.round(a.timeout)} seconds`) : null;
  const error = h("div", { class: "error", role: "alert", hidden: true });
  const approve = h("button", { type: "button", class: "btn btn--primary", "aria-describedby": titleId }, icon("check", 15), "Approve");
  const decline = h("button", { type: "button", class: "btn", "aria-describedby": titleId }, icon("ban", 15), "Decline");
  const refs = { approve, decline, error };
  approve.addEventListener("click", () => answerApproval(job, true, refs));
  decline.addEventListener("click", () => answerApproval(job, false, refs));
  const table = fresh.length ? h("table", { class: "approval-table" },
    h("caption", { class: "sr-only", text: "Packages this plan would install, with what the checks found" }),
    h("thead", {}, h("tr", {}, ["Package", "Version", "Checks", "Reasons"].map((label) => h("th", { scope: "col", text: label })))),
    h("tbody", {}, fresh.map((p) => {
      const facts = p.facts || {};
      const via = Array.isArray(facts.via) && facts.via.length ? `needed by ${facts.via.join(", ")}` : facts.requested ? "requested" : "";
      const reasons = Array.isArray(p.reasons) ? p.reasons.filter(Boolean) : [];
      return h("tr", {},
        h("th", { scope: "row" }, h("span", { class: "pkg-name", text: p.name }), via ? h("span", { class: "pkg-via", text: via }) : null),
        h("td", { class: "num", text: p.version || "?" }),
        h("td", {}, tag(String(p.verdict || "?"), VERDICT_TONES[p.verdict] || "neutral")),
        h("td", {}, reasons.length ? h("ul", { class: "pkg-reasons" }, reasons.map((r) => h("li", { text: r }))) : h("span", { class: "pkg-none", text: "No concerns" })));
    }))) : null;
  const review = typeof payload.review === "string" && payload.review.trim() ? payload.review.trim() : null;
  const outcome = answered ? h("p", {
    class: "approval-outcome", role: "status", tabindex: "-1",
    text: a.answer
      ? `Approved. The agent installs ${names} into its toolbox, then writes the adapter; the run goes on.`
      : `Declined. Nothing is installed: the agent writes the adapter without ${names}.`,
  }) : null;
  return h("section", { class: `approval callout callout--${!answered ? "warning" : a.answer ? "success" : "neutral"}`, "aria-labelledby": titleId },
    h("div", { class: "callout-head" },
      h("div", { class: "callout-title" }, icon(answered ? (a.answer ? "check-circle" : "ban") : "shield-check", 14),
        h("h3", { id: titleId, class: "approval-title", tabindex: "-1", text: install ? `Install ${names}?` : `Approve this ${a.kind} request?` })),
      plan.verdict ? tag(`checks: ${plan.verdict}`, VERDICT_TONES[plan.verdict] || "neutral") : null),
    summary ? h("p", { class: "callout-body", text: summary }) : null,
    requested.length ? h("p", { class: "approval-asked" },
      h("span", { class: "approval-label", text: "The agent asked for " }),
      requested.map((r) => `${r.name}${r.version ? ` ${r.version}` : ""}${r.why ? `: ${r.why}` : ""}`).join("; ")) : null,
    table,
    have.length ? h("details", { class: "approval-have" },
      h("summary", { text: `${have.length} already installed: not vetted or installed again` }),
      h("ul", {}, have.map((p) => h("li", { text: `${p.name} ${p.version || ""}`.trim() })))) : null,
    install && !packages.length ? h("p", { class: "approval-note", text: "The checks listed no package details." }) : null,
    review ? h("div", { class: "approval-review" },
      h("p", { class: "approval-label", text: "The agent's opinion: an LLM's review, not one of the checks" }),
      h("p", { text: review })) : null,
    !answered && install ? h("p", { class: "approval-note", text: `Approved packages go into a toolbox folder apart from the main environment, which only the agent's adapters import.${minutes ? ` No answer within ${minutes} counts as declined.` : ""}` }) : null,
    !answered ? h("div", { class: "approval-actions" }, approve, decline) : null,
    outcome,
    error);
}

/** POST /api/jobs/<id>/approval with your answer. The buttons wait in place (keeping focus); the card then shows
    the outcome, which takes the focus. A request no longer open (409) says so and the next poll closes it. */
async function answerApproval(job, approve, refs) {
  const a = job.approval;
  if (!a || a.answer !== null || a.sending) return;
  a.sending = true;
  refs.approve.disabled = true;
  refs.decline.disabled = true;
  (approve ? refs.approve : refs.decline).setAttribute("aria-busy", "true");
  showError(refs.error, null);
  let failure = null;
  try {
    await api(`/api/jobs/${enc(job.id)}/approval`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: a.id, approve }),
    });
    a.answer = approve;
    a.answeredAt = Date.now();
  } catch (err) {
    failure = err;
  } finally {
    a.sending = false;
  }
  if (state.job !== job || job.approval !== a) return;
  if (failure) {
    refs.approve.removeAttribute("aria-busy");
    refs.decline.removeAttribute("aria-busy");
    const closed = failure.status === 409;
    refs.approve.disabled = closed;
    refs.decline.disabled = closed;
    showError(refs.error, closed ? new Error(`${failure.message} The run's trace says how the request ended.`) : failure);
    return;
  }
  noteApproval(job);
  renderApproval(job);
  const out = $("approval").querySelector(".approval-outcome");
  if (out) out.focus({ preventScroll: true });
}

/** The agent's line in the chat about the request: asked, then answered or closed. */
function noteApproval(job) {
  const a = job.approval;
  if (!a) return;
  const names = approvalNames(a);
  const text = a.answer === true ? `You approved installing ${names}.`
    : a.answer === false ? `You declined installing ${names}; the agent writes the adapter without it.`
    : a.closed ? `The request to install ${names} closed with no answer from here; the Run tab's trace says how it ended.`
    : `The agent asks to install ${names} before it writes the adapter. Read what the checks found, then approve or decline in the Run tab.`;
  if (!job.approvalNote) {
    const textEl = h("p", { class: "msg-text" });
    const action = h("button", { type: "button", class: "link-button", text: "Review the request →", onclick: focusApproval });
    const msg = h("article", { class: "turn msg msg--agent" },
      h("div", { class: "name-line" }, icon("agent", 13), h("span", { class: "name", text: "Agent" })),
      textEl, action);
    textEl.textContent = text;
    $("thread").append(msg);
    updateThreadEmpty();
    scrollToEnd($("thread"));
    job.approvalNote = { textEl, action };
  }
  job.approvalNote.textEl.textContent = text;
  job.approvalNote.action.hidden = a.answer !== null || a.closed;
}

/** Show the open request's card and move the focus to its heading. */
function focusApproval() {
  setTab("run");
  const heading = $("approval").querySelector(".approval-title");
  if (!heading) return;
  heading.focus({ preventScroll: true });
  heading.scrollIntoView({ block: "nearest" });
}

function resetTrace() {
  if (state.job && state.job.timer) clearTimeout(state.job.timer);
  state.job = null;
  store.set("sessionStorage", "sense.job", null);
  $("timeline").replaceChildren();
  $("log-count").textContent = "";
  $("log-button").disabled = true;
  if (state.tab === "log") setTab("run");
  $("trace-outcome").replaceChildren();
  $("approval").replaceChildren();
  renderTraceHead();
  renderSteps();
  renderGraph();
  renderReport();
  syncControls();
}

/** Event log tab: swaps the sense graph for every event of the run, as the server sent them. */
/** The inspector's tabs. The graph needs more room than the others, so its tab widens the inspector. */
function setTab(name) {
  const selected = document.querySelector(`.insp-tabs [data-tab="${name}"]`);
  if (!selected || selected.disabled) return;
  state.tab = name;
  const views = { report: "report-view", run: "run-view", graph: "graph-view", log: "log-view" };
  for (const [key, id] of Object.entries(views)) $(id).hidden = key !== name;
  for (const tab of document.querySelectorAll(".insp-tabs .tab")) {
    tab.setAttribute("aria-selected", String(tab.dataset.tab === name));
    tab.tabIndex = tab.dataset.tab === name ? 0 : -1;
  }
  $("workspace").classList.toggle("is-wide", name === "graph");
  if (name === "log") scrollToEnd($("log-view"));
  if (name === "graph") requestAnimationFrame(drawWires);
}

// ---------------------------------------------------------------- runs and replay

function openRuns() {
  closeFit(false);
  const dialog = $("runs-dialog");
  if (!dialog.open) dialog.showModal();
  loadRuns();
}

function closeRuns() {
  if ($("runs-dialog").open) $("runs-dialog").close();
}

async function loadRuns() {
  showError($("runs-error"), null);
  try {
    const runs = await api("/api/runs");
    state.runs = Array.isArray(runs) ? runs : [];
  } catch (err) {
    showError($("runs-error"), err, loadRuns);
  }
  renderRuns();
}

const clock = (iso) => {
  const d = iso ? new Date(iso) : null;
  return d && !Number.isNaN(d.getTime()) ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "";
};

/** One row per recorded run (newest first), each with its outcome and a Replay button. */
function renderRuns() {
  const list = $("runs-list");
  const job = state.job;
  const busy = jobRunning() || state.starting;
  $("runs-summary").textContent = jobRunning() ? (job.replay ? "Replaying" : "Running") : `${state.runs.length} recorded`;
  $("runs-current").hidden = !job;
  const replayOf = job && job.replay ? job.replay.of : null;
  const rows = [];
  if (jobRunning() && !job.replay) {
    rows.push(h("li", { class: "run-item run-item--live" },
      h("span", { class: "run-item-main" },
        h("span", { class: "run-item-title" }, h("span", { class: "led led--run" }), " Live run"),
        h("span", { class: "run-item-meta", text: job.request ? [modelLabel(job.request.model), job.goal, setTitle(job.request.dataset)].filter(Boolean).join(" · ") : "" }))));
  }
  for (const run of state.runs) {
    const r = run.request || {};
    const active = replayOf === run.id;
    const outcome = run.status === "failed" ? "failed" : run.installed ? `installed · ${run.installed}` : "nothing installed";
    // An auto run that ended before its data step (a native file, one that needed the user, or a stop) has no set.
    const meta = [modelLabel(r.model), r.dataset && !isProbeSet(r.dataset) ? (SCOPE_LABELS[r.scope] || r.scope || "").toLowerCase() : "", toNumber(run.seconds) !== null ? seconds(run.seconds) : "", clock(run.recorded), run.origin]
      .filter(Boolean).join(" · ");
    rows.push(h("li", { class: `run-item${active ? " is-active" : ""}` },
      h("span", { class: "run-item-main" },
        h("span", { class: "run-item-title", text: r.modality || r.dataset
          ? `${modalityName(r.modality || "sense")} · ${isProbeSet(r.dataset) ? "agent-written adapter" : r.dataset ? datasetTitle(r.dataset) : "no labelled set chosen"}`
          : "Agent run · ended before it chose labelled files" }),
        h("span", { class: "run-item-meta", text: meta }),
        h("span", { class: `run-item-outcome run-item-outcome--${run.installed ? "ok" : run.status === "failed" ? "fail" : "none"}`, text: outcome })),
      h("button", {
        type: "button",
        class: "btn btn--small",
        disabled: busy,
        title: busy ? "Wait for the current run to finish" : `Replay at ${state.replaySpeed}×`,
        onclick: () => startReplay(run),
      }, icon("play", 12), active && busy ? "Playing" : "Replay")));
  }
  if (!rows.length) rows.push(h("li", { class: "runs-empty", text: "No recorded runs yet. Every finished agent run lands here, ready to replay." }));
  list.replaceChildren(...rows);
}

async function startReplay(run) {
  showError($("runs-error"), null);
  if (jobRunning() || state.starting) return;
  state.starting = true;
  renderRuns();
  syncControls();
  try {
    const result = await api("/api/replay", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run: run.id, speed: state.replaySpeed }),
    });
    if (!result || !result.job_id) throw new Error("The server started no replay.");
    state.starting = false;
    state.replayDismissed = null;
    // The report, graph and chat follow the band's model: show the replay on the model it was recorded for.
    const model = run.request && run.request.model;
    if (model && model !== state.model && selectableModel(model)) chooseModel(model);
    startJob(result.job_id, run.request || null, false);
    state.job.replay = { of: run.id, recorded: run.recorded, speed: state.replaySpeed };
    renderTraceHead();
    renderRuns();
    closeRuns();
    $("run-tab").focus();
  } catch (err) {
    if (!$("runs-dialog").open) $("runs-dialog").showModal();
    showError($("runs-error"), err, () => startReplay(run));
  } finally {
    state.starting = false;
    renderRuns();
    syncControls();
  }
}

/** The bar above the chat while a recorded run plays. Chat stays live under it. */
function renderReplayBar() {
  const bar = $("replay-bar");
  const job = state.job;
  if (!job || !job.replay || state.replayDismissed === job.id) {
    bar.hidden = true;
    return;
  }
  const run = state.runs.find((r) => r.id === job.replay.of);
  const total = run ? toNumber(run.seconds) : null;
  const done = job.status !== "running";
  const frac = done ? 1 : total && job.elapsed !== null ? Math.min(1, job.elapsed / total) : 0;
  const last = [...job.steps.keys()].pop();
  bar.hidden = false;
  const said = job.cancelled ? "Replay stopped" : job.status === "failed" || job.status === "lost" ? "Replay failed" : done ? "Replay finished" : "Replaying a recorded run";
  // Native replaceChildren would print a null child as the text "null", so leave it out.
  bar.replaceChildren(...[
    h("span", { class: "replay-mark", text: done ? "■" : "▶" }),
    h("span", { class: "replay-text", text: `${said} · ${job.replay.speed || 1}×${job.replay.recorded ? ` · recorded ${clock(job.replay.recorded)}` : ""}` }),
    h("span", { class: "replay-track" }, h("span", { class: "replay-fill", style: `width:${(frac * 100).toFixed(1)}%` })),
    h("span", { class: "replay-step num", text: [last || "", job.elapsed !== null ? seconds(job.elapsed) : "", total !== null ? `/ ${seconds(total)}` : ""].filter(Boolean).join(" ") }),
    h("span", { class: "replay-note", text: "chat stays live" }),
    !done ? h("button", { type: "button", class: "btn btn--small", disabled: job.cancelling, text: job.cancelling ? "Stopping…" : "Stop replay", onclick: cancelJob }) : null,
    done ? h("button", { type: "button", class: "icon-button", "aria-label": "Close the replay bar", onclick: () => { state.replayDismissed = job.id; renderReplayBar(); } }, icon("x", 13)) : null,
  ].filter(Boolean));
}

// ---------------------------------------------------------------- chats

const currentChat = () => state.chats.find((c) => c.id === state.chatId) || state.chats[0];

/** Store data, never DOM/HTML or uploaded bytes. Rebuild handlers and text nodes on reload. */
function saveChats() {
  for (const chat of state.chats) {
    chat.turns = chat.messages.length;
    chat.title = chat.turns ? chat.messages[0].message.slice(0, 46) : "New chat";
  }
  const chats = state.chats.slice(0, 40).map((chat) => ({
    id: chat.id,
    updatedAt: chat.updatedAt || null,
    messages: chat.messages.slice(-100).map(({ model, message, files, samples, status, result, error }) => ({
      model, message, files: files.map((f) => ({ name: f.name, type: f.type })), samples, status, result, error,
    })),
  }));
  try {
    localStorage.setItem("sense.chats.v1", JSON.stringify({ chats, chatId: state.chatId }));
    $("chat-storage").textContent = "Chats saved in this browser · uploads are not saved";
  } catch {
    $("chat-storage").textContent = "Browser storage unavailable or full · chats are not saved";
  }
}

function restoreChats() {
  try {
    const saved = JSON.parse(store.get("localStorage", "sense.chats.v1") || "null");
    if (!saved || !Array.isArray(saved.chats)) return;
    const ids = new Set();
    const chats = saved.chats.filter((c) => {
      if (!c || !Number.isSafeInteger(c.id) || c.id < 1 || ids.has(c.id) || !Array.isArray(c.messages)) return false;
      ids.add(c.id);
      return true;
    }).slice(0, 40).map((c) => ({
      id: c.id, turns: 0, title: "New chat", frag: null,
      updatedAt: toNumber(c.updatedAt),
      messages: c.messages.filter((m) => m && typeof m.message === "string" && m.model && typeof m.model.name === "string").slice(-100).map((m) => ({
        model: { name: m.model.name, label: typeof m.model.label === "string" ? m.model.label : m.model.name },
        message: m.message,
        files: (Array.isArray(m.files) ? m.files : []).filter((f) => f && typeof f.name === "string").map((f) => ({ name: f.name, type: typeof f.type === "string" ? f.type : "" })),
        samples: (Array.isArray(m.samples) ? m.samples : []).filter((s) => s && typeof s.id === "string"),
        status: m.status === "done" && m.result && typeof m.result.reply === "string" ? "done" : "failed",
        result: m.result && typeof m.result.reply === "string" ? {
          reply: m.result.reply,
          used: (Array.isArray(m.result.used) ? m.result.used : []).filter((u) => u && typeof u.kind === "string")
            .map((u) => Object.fromEntries(["kind", "bridge", "source", "detail"].map((k) => [k, typeof u[k] === "string" ? u[k] : ""]))),
        } : null,
        error: typeof m.error === "string" ? m.error : "Reply interrupted by reload. Retry to send again.",
      })),
    }));
    if (!chats.length) return;
    state.chats = chats;
    state.nextChatId = Math.max(...chats.map((c) => c.id)) + 1;
    for (const chat of chats) {
      for (const record of chat.messages) record.turn = drawMessage(record, chat);
      stashThread(chat);
    }
    showChat(chats.find((c) => c.id === saved.chatId) || chats[0]);
  } catch {
    // A corrupt/older save must not prevent the app from starting.
  }
}

/** Park the open conversation's messages so another chat can use the thread; replies still land in them. */
function stashThread(chat) {
  if (state.editing?.chat === chat) cancelEdit();
  chat.draft = $("message").value;
  chat.draftFiles = [...state.files];
  chat.draftSamples = new Map(state.attachedSamples);
  const frag = document.createDocumentFragment();
  for (const el of [...$("thread").children]) if (el.id !== "thread-empty") frag.append(el);
  chat.frag = frag;
}

function showChat(chat) {
  if (state.editing) cancelEdit();
  showError($("chat-error"), null);
  state.chatId = chat.id;
  $("message").value = chat.draft || "";
  state.files = chat.draftFiles || [];
  state.attachedSamples = new Map(chat.draftSamples || []);
  autosize($("message"));
  renderAttachments();
  for (const id of state.sampleRefs.keys()) syncSampleItem(id);
  if (chat.frag) $("thread").append(chat.frag);
  chat.frag = null;
  updateThreadEmpty();
  scrollToEnd($("thread"));
  saveChats();
  renderChats();
  syncControls();
}

/** New chat: an empty thread, same model and senses. An empty current chat is reused. */
function newChat() {
  $("chat-search").value = "";
  const current = currentChat();
  if (current.turns === 0 && !$("thread").querySelector(".turn")) {
    $("message").focus();
    return;
  }
  stashThread(current);
  const chat = { id: state.nextChatId++, title: "New chat", turns: 0, frag: null, messages: [] };
  state.chats.unshift(chat);
  showChat(chat);
  $("message").focus();
}

function switchChat(id) {
  if (id === state.chatId) return;
  const target = state.chats.find((c) => c.id === id);
  if (!target) return;
  stashThread(currentChat());
  showChat(target);
  $("message").focus();
}

function renderChats() {
  const list = $("chats-list");
  const query = $("chat-search").value.trim().toLowerCase();
  const chats = state.chats.filter((chat) => !query || chat.title.toLowerCase().includes(query)
    || chat.messages.some((m) => m.message.toLowerCase().includes(query) || (m.result?.reply || "").toLowerCase().includes(query)));
  const working = (chat) => chat.messages.some((m) => m.status === "pending");
  const groups = [
    ["Working", chats.filter(working)],
    ["Current chat", chats.filter((c) => !working(c) && c.id === state.chatId)],
    ["Previous chats", chats.filter((c) => !working(c) && c.id !== state.chatId).sort((a, b) => (b.updatedAt || b.id) - (a.updatedAt || a.id))],
  ];
  const focusedId = list.contains(document.activeElement) ? document.activeElement.dataset.chatId : null;
  list.replaceChildren(...groups.filter(([, items]) => items.length).map(([label, items]) =>
    h("section", { class: "chat-group", "aria-label": label },
      h("h3", { class: "chat-group-title", text: label }),
      h("ul", {}, ...items.map((chat) => {
        const last = chat.messages.at(-1);
        const status = working(chat) ? "Replying…" : last?.status === "failed" ? "Needs attention" : chat.turns ? `${chat.turns} message${chat.turns === 1 ? "" : "s"}` : "Ready to start";
        const date = chat.updatedAt ? new Date(chat.updatedAt).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "";
        return h("li", {}, h("button", {
          type: "button",
          class: `chat-item${chat.id === state.chatId ? " is-active" : ""}`,
          "aria-current": chat.id === state.chatId ? "true" : null,
          "data-chat-id": chat.id,
          title: chat.title,
          onclick: () => switchChat(chat.id),
        },
          h("span", { class: "chat-item-title", text: chat.title }),
          h("span", { class: `chat-item-meta${working(chat) ? " is-working" : ""}`, text: [status, date].filter(Boolean).join(" · ") })));
      })))));
  if (!chats.length) list.append(h("p", { class: "chats-empty", text: "No matching chats. Try another word." }));
  if (focusedId) list.querySelector(`[data-chat-id="${focusedId}"]`)?.focus();
  $("chats-count").textContent = query ? `${chats.length} of ${state.chats.length}` : String(state.chats.length);
}

/** Terminal view: one look switch (CSS does the rest), remembered in this browser. */
function setMode(term) {
  document.body.classList.toggle("mode-term", term);
  $("mode-button").setAttribute("aria-pressed", String(term));
  $("mode-label").textContent = term ? "Normal view" : "Terminal";
  store.set("localStorage", "sense.mode", term ? "term" : null);
  requestAnimationFrame(drawWires);
}

function toggleSamples(show) {
  const pop = $("samples-pop");
  const open = show === undefined ? pop.hidden : show;
  pop.hidden = !open;
  $("samples-button").setAttribute("aria-expanded", String(open));
}

// ---------------------------------------------------------------- slash commands

const COMMANDS = [
  { cmd: "new", help: "Start a new chat" },
  { cmd: "term", help: "Switch between the terminal and the normal view" },
  { cmd: "fit", help: "Give the model a new sense (opens New sense)" },
  { cmd: "samples", help: "Pick a sample clip to attach" },
  { cmd: "replay", args: "[n]", help: "Replay the n-th most recent run (default 1)" },
  { cmd: "runs", help: "Open recorded runs and replay" },
  { cmd: "report", help: "Show the report card" },
  { cmd: "run", help: "Show the agent run" },
  { cmd: "graph", help: "Show the sense graph" },
  { cmd: "log", help: "Show the event log" },
  { cmd: "clear", help: "Clear the conversation on screen" },
  { cmd: "help", help: "List these commands" },
];

function renderCommandHints() {
  const box = $("message");
  const list = $("cmd-hints");
  const text = box.value;
  if (!text.startsWith("/") || text.includes(" ")) {
    list.hidden = true;
    return;
  }
  const typed = text.slice(1).toLowerCase();
  const matches = COMMANDS.filter((c) => c.cmd.startsWith(typed));
  list.replaceChildren(...matches.map((c) =>
    h("li", {
      class: "cmd-hint",
      role: "option",
      "data-cmd": c.cmd,
      onmousedown: (e) => {
        e.preventDefault();
        box.value = `/${c.cmd} `;
        box.focus();
        renderCommandHints();
      },
    },
      h("span", { class: "cmd-name", text: `/${c.cmd}${c.args ? ` ${c.args}` : ""}` }),
      h("span", { class: "cmd-help", text: c.help }))));
  list.hidden = matches.length === 0;
}

function runCommand(text) {
  const [name, ...args] = text.slice(1).trim().split(/\s+/);
  const cmd = (name || "").toLowerCase();
  const note = (t) => {
    $("thread").append(h("p", { class: "thread-note turn", text: t }));
    updateThreadEmpty();
    scrollToEnd($("thread"));
  };
  if (cmd === "new") newChat();
  else if (cmd === "term") setMode(!document.body.classList.contains("mode-term"));
  else if (cmd === "samples") toggleSamples(true);
  else if (cmd === "fit" || cmd === "sense") openFit($("new-sense-button"));
  else if (cmd === "runs") openRuns();
  else if (["report", "run", "graph", "log"].includes(cmd)) setTab(cmd);
  else if (cmd === "replay") {
    const n = Math.max(1, parseInt(args[0], 10) || 1);
    const run = state.runs[n - 1];
    if (!run) note(state.runs.length ? `There are only ${state.runs.length} recorded runs.` : "No recorded runs yet.");
    else if (jobRunning()) note("A run is in progress. Wait for it to finish, then replay.");
    else startReplay(run);
  } else if (cmd === "clear") {
    if (state.chatting) return;
    for (const el of [...$("thread").children]) if (el.id !== "thread-empty") el.remove();
    currentChat().messages = [];
    currentChat().turns = 0;
    state.editing = null;
    saveChats();
    renderChats();
    updateThreadEmpty();
  } else if (cmd === "help" || cmd === "") {
    note(COMMANDS.map((c) => `/${c.cmd}${c.args ? ` ${c.args}` : ""}  ${c.help}`).join("\n"));
  } else note(`Unknown command /${cmd}. Type /help for the list.`);
}

// ---------------------------------------------------------------- start

function updateScopeHint() {
  const scope = document.querySelector('input[name="scope"]:checked');
  $("scope-hint").textContent = SCOPE_HINTS[scope ? scope.value : "closed"];
}

function bindEvents() {
  // Native dialogs make the background inert; keep Tab in the dialog rather than browser chrome.
  document.addEventListener("keydown", (e) => {
    const dialog = document.querySelector("dialog[open]");
    if (!dialog || e.key !== "Tab") return;
    const controls = [...dialog.querySelectorAll("button, input, select, textarea, a[href], [tabindex]")]
      .filter((el) => !el.disabled && el.tabIndex >= 0 && el.getClientRects().length);
    const first = controls[0], last = controls.at(-1);
    if (!first) { e.preventDefault(); return; }
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  });
  $("chat-stop").addEventListener("click", () => state.chatAbort?.abort());
  $("job-stop").addEventListener("click", cancelJob);
  $("fit-goal").addEventListener("input", syncControls);
  const chooseFitFile = (file) => {
    state.fitFile = file || null;
    $("fit-file-name").textContent = file ? file.name : "Choose a file to start.";
    syncControls();
  };
  $("fit-file").addEventListener("change", (e) => chooseFitFile(e.target.files[0]));
  $("fit-drop").addEventListener("dragover", (e) => { e.preventDefault(); e.currentTarget.classList.add("is-dropping"); });
  $("fit-drop").addEventListener("dragleave", (e) => e.currentTarget.classList.remove("is-dropping"));
  $("fit-drop").addEventListener("drop", (e) => {
    e.preventDefault();
    e.currentTarget.classList.remove("is-dropping");
    chooseFitFile(e.dataTransfer.files[0]);
  });
  for (const radio of document.querySelectorAll('input[name="fit-mode"]')) radio.addEventListener("change", syncFitMode);
  $("model-select").addEventListener("change", (e) => setModel(e.target.value));
  $("dataset-select").addEventListener("change", (e) => setDataset(e.target.value));
  for (const radio of document.querySelectorAll('input[name="scope"]')) {
    radio.addEventListener("change", () => {
      updateScopeHint();
      if (!modelJob()) renderGraph();
    });
  }
  $("fit-form").addEventListener("submit", runAgent);
  $("chat-form").addEventListener("submit", sendMessage);

  // New sense (top bar) and Add a sense (report footer) open the form that starts the agent.
  $("new-sense-button").addEventListener("click", (e) => ($("fit-popover").hidden ? openFit(e.currentTarget) : closeFit(true)));
  $("add-sense-button").addEventListener("click", (e) => openFit(e.currentTarget));
  $("fit-close").addEventListener("click", () => closeFit(true));
  $("fit-popover").addEventListener("cancel", (e) => { e.preventDefault(); closeFit(true); });
  document.addEventListener("mousedown", (e) => {
    const pop = $("fit-popover");
    if (pop.hidden || pop.contains(e.target)) return;
    if (e.target.closest("#new-sense-button, #add-sense-button")) return;
    closeFit(false);
  });

  for (const tab of document.querySelectorAll(".insp-tabs .tab")) tab.addEventListener("click", () => setTab(tab.dataset.tab));
  document.querySelector(".insp-tabs").addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return;
    const tabs = [...document.querySelectorAll(".insp-tabs .tab:not(:disabled)")];
    const index = tabs.indexOf(e.target);
    if (index < 0) return;
    e.preventDefault();
    const next = e.key === "Home" ? 0 : e.key === "End" ? tabs.length - 1 : (index + (e.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    setTab(tabs[next].dataset.tab);
    tabs[next].focus();
  });
  $("chat-search").addEventListener("input", renderChats);
  $("runs-button").addEventListener("click", openRuns);
  $("runs-close").addEventListener("click", closeRuns);
  $("runs-dialog").addEventListener("close", () => $("runs-button").focus());
  $("runs-current").addEventListener("click", () => {
    closeRuns();
    setTab("run");
    $("run-tab").focus();
  });
  $("runs-refresh").addEventListener("click", loadRuns);
  $("new-chat-button").addEventListener("click", newChat);
  $("mode-button").addEventListener("click", () => setMode(!document.body.classList.contains("mode-term")));
  $("chat-new").addEventListener("click", newChat);
  $("samples-button").addEventListener("click", () => toggleSamples());
  document.addEventListener("mousedown", (e) => {
    if (!$("samples-pop").hidden && !e.target.closest("#samples-pop, #samples-button")) toggleSamples(false);
  });
  for (const b of document.querySelectorAll(".speed")) {
    b.addEventListener("click", () => {
      state.replaySpeed = Number(b.dataset.speed) || 1;
      for (const other of document.querySelectorAll(".speed")) other.setAttribute("aria-pressed", String(other === b));
      renderRuns();
    });
  }
  // As in other chat harnesses, Ctrl/Cmd+K focuses the composer from anywhere.
  document.addEventListener("keydown", (e) => {
    if (document.querySelector("dialog[open]")) return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      $("message").focus();
    }
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "o") {
      e.preventDefault();
      newChat();
    }
    if (e.key === "Escape" && !$("samples-pop").hidden) toggleSamples(false);
  });
  $("export-button").addEventListener("click", exportReport);

  const box = $("message");
  box.addEventListener("input", () => {
    autosize(box);
    renderCommandHints();
  });
  box.addEventListener("blur", () => setTimeout(() => ($("cmd-hints").hidden = true), 150));
  box.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && state.editing) {
      cancelEdit();
    }
    const hints = $("cmd-hints");
    if (e.key === "Tab" && !hints.hidden && hints.firstChild) {
      e.preventDefault();
      box.value = `/${hints.firstChild.dataset.cmd} `;
      renderCommandHints();
      return;
    }
    if (e.key === "Escape" && !hints.hidden) {
      hints.hidden = true;
      return;
    }
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

  if ("ResizeObserver" in window) new ResizeObserver(() => drawWires()).observe($("graph-canvas"));
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(drawWires);
}

function init() {
  hydrateIcons();
  bindEvents();
  restoreChats();
  updateScopeHint();
  renderAttachments();
  renderSamples("Choose a labelled set to see its sample clips.");
  renderTraceHead();
  renderSteps();
  renderGraph();
  renderReport();
  syncControls();
  loadModels();
  loadDatasets();
  loadRuns();
  renderChats();
  if (store.get("localStorage", "sense.mode") === "term") setMode(true);
  const savedJob = store.get("sessionStorage", "sense.job");
  if (savedJob) startJob(savedJob, null, true);
  else followRunElsewhere();
}

/** Follow an agent run started in another tab or window (GET /api/jobs/latest): its trace, its approval card,
    which only one run at a time may hold open for up to 10 minutes, and Stop. */
async function followRunElsewhere() {
  try {
    const latest = await api("/api/jobs/latest");
    if (latest && typeof latest.job_id === "string" && latest.job_id && !state.job && !state.starting) {
      startJob(latest.job_id, null, true);
    }
  } catch {
    // An older server without the route, or none yet: nothing to follow.
  }
}

init();
