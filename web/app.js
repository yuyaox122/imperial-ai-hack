"use strict";

/*
 * Sense harness interface ("Sensorium" in the team Figma file): top bar, chat, agent run and report card.
 * Talks only to the HTTP API documented in server/app.py. No framework, no build step.
 * Every part is drawn from a component on the Figma "Design system" page; design/figma-notes.md maps them.
 * Only data the API returns is shown: nothing here is a placeholder number.
 */

const POLL_MS = 700;
const MAX_POLL_RETRIES = 5;

const CONTROL_LABELS = {
  swapped: "Swapped audio",
  no_input: "No audio",
  lookup: "Nearest-example lookup",
  text_bridge: "Text bridge",
  unseen_classes: "Unseen categories",
};
const KIND_LABELS = { native: "Native", text: "Text", tool: "Tool", image: "Image", vector: "Vector" };
const VERDICT_LABELS = { installed: "installed", rejected: "refused", candidate: "candidate" };
const SCOPE_LABELS = { closed: "These categories only", open: "Any sound" };
const SCOPE_HINTS = {
  closed: "The agent checks that the model tells these labels apart on held-out clips.",
  open: "The agent also tests labels it never fitted on, and refuses a bridge that only knows the labels it was shown.",
};
const STATUS_WORDS = { start: "started", ok: "succeeded", fail: "failed", info: "note" };
const JOB_WORDS = { running: "Running", done: "Finished", failed: "Failed", lost: "Disconnected" };

/* The six steps of the agent loop, in the order of the Figma "Agent step" strip. Unknown steps from the
   server are shown after these. Before a run each card explains what the step will do. */
const STEPS = [
  { key: "inspect", title: "Check the base model", idle: "Read what the model accepts and whether its weights are open." },
  { key: "plan", title: "Choose bridges", idle: "Choose which bridges to try." },
  { key: "fit", title: "Build bridges", idle: "Build each candidate from the labelled examples." },
  { key: "verify", title: "Verify on held-out", idle: "Score each one on held-out clips against the controls." },
  { key: "decide", title: "Install or refuse", idle: "Pick the bridge that beats the controls, or refuse." },
  { key: "install", title: "Install", idle: "Add it to the model's senses, ready for chat." },
];
const STEP_INDEX = Object.fromEntries(STEPS.map((s) => [s.key, s]));
const STEP_ICONS = { done: "check-circle", running: "loader", pending: "circle", refused: "ban" };

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
  senses: [], // the last /api/senses answer for state.model
  usedBridges: new Set(), // bridges a chat reply has used; their top-bar chip stops being "new"
  reportBridge: null, // which card the report panel shows
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

/* Interface icons: Lucide glyphs (ISC licence) exported from the Figma "Icons" set (19:57), 16 px grid. */
const ICONS = {
  sparkles: [
    "M6.62467 10.3333C6.56515 10.1026 6.44489 9.89206 6.27641 9.72358C6.10793 9.5551 5.89738 9.43485 5.66667 9.37533L1.57667 8.32066C1.50689 8.30085 1.44547 8.25883 1.40174 8.20096C1.35801 8.14309 1.33435 8.07253 1.33435 7.99999C1.33435 7.92746 1.35801 7.8569 1.40174 7.79903C1.44547 7.74116 1.50689 7.69913 1.57667 7.67933L5.66667 6.62399C5.8973 6.56453 6.1078 6.44438 6.27627 6.27602C6.44474 6.10766 6.56504 5.89725 6.62467 5.66666L7.67933 1.57666C7.69894 1.50661 7.74092 1.44489 7.79888 1.40092C7.85684 1.35696 7.92759 1.33316 8.00033 1.33316C8.07308 1.33316 8.14383 1.35696 8.20179 1.40092C8.25974 1.44489 8.30173 1.50661 8.32133 1.57666L9.37533 5.66666C9.43485 5.89737 9.55511 6.10792 9.72359 6.27641C9.89207 6.44489 10.1026 6.56514 10.3333 6.62466L14.4233 7.67866C14.4937 7.69806 14.5557 7.74 14.5999 7.79804C14.6441 7.85609 14.668 7.92703 14.668 7.99999C14.668 8.07295 14.6441 8.1439 14.5999 8.20194C14.5557 8.25999 14.4937 8.30193 14.4233 8.32133L10.3333 9.37533C10.1026 9.43485 9.89207 9.5551 9.72359 9.72358C9.55511 9.89206 9.43485 10.1026 9.37533 10.3333L8.32067 14.4233C8.30106 14.4934 8.25908 14.5551 8.20112 14.5991C8.14316 14.643 8.07241 14.6668 7.99967 14.6668C7.92692 14.6668 7.85617 14.643 7.79821 14.5991C7.74026 14.5551 7.69827 14.4934 7.67867 14.4233L6.62467 10.3333Z",
    "M13.3334 2V4.66667",
    "M14.6667 3.33333H12",
  ],
  cpu: [
    "M12 2.66667H3.99996C3.26358 2.66667 2.66663 3.26363 2.66663 4.00001V12C2.66663 12.7364 3.26358 13.3333 3.99996 13.3333H12C12.7363 13.3333 13.3333 12.7364 13.3333 12V4.00001C13.3333 3.26363 12.7363 2.66667 12 2.66667Z",
    "M10 6H6V10H10V6Z",
    "M10 1.33333V2.66666M10 13.3333V14.6667M1.33337 10H2.66671M1.33337 6H2.66671M13.3334 10H14.6667M13.3334 6H14.6667M6.00004 1.33333V2.66666M6.00004 13.3333V14.6667",
  ],
  "chevron-down": ["M4 6L8 10L12 6"],
  plus: ["M3.33325 8H12.6666", "M8 3.33333V12.6667"],
  check: ["M13.3333 4L5.99996 11.3333L2.66663 8"],
  "check-circle": [
    "M8.00004 14.6667C11.6819 14.6667 14.6667 11.6819 14.6667 8C14.6667 4.3181 11.6819 1.33333 8.00004 1.33333C4.31814 1.33333 1.33337 4.3181 1.33337 8C1.33337 11.6819 4.31814 14.6667 8.00004 14.6667Z",
    "M6 8.00001L7.33333 9.33334L10 6.66667",
  ],
  ban: [
    "M7.99992 14.6667C11.6818 14.6667 14.6666 11.6819 14.6666 8C14.6666 4.3181 11.6818 1.33333 7.99992 1.33333C4.31802 1.33333 1.33325 4.3181 1.33325 8C1.33325 11.6819 4.31802 14.6667 7.99992 14.6667Z",
    "M3.2666 3.26666L12.7333 12.7333",
  ],
  loader: [
    "M14 8C13.9999 9.26706 13.5988 10.5016 12.854 11.5266C12.1092 12.5517 11.059 13.3146 9.85392 13.7061C8.64886 14.0976 7.3508 14.0976 6.14576 13.706C4.94073 13.3144 3.89059 12.5514 3.14584 11.5263C2.4011 10.5013 1.99999 9.26673 2 7.99967C2.00001 6.73261 2.40114 5.49808 3.14589 4.473C3.89065 3.44793 4.9408 2.68494 6.14584 2.29339C7.35088 1.90183 8.64895 1.90182 9.854 2.29334",
  ],
  circle: ["M8 14C11.3137 14 14 11.3137 14 8C14 4.68629 11.3137 2 8 2C4.68629 2 2 4.68629 2 8C2 11.3137 4.68629 14 8 14Z"],
  x: ["M12 4L4 12", "M4 4L12 12"],
  "arrow-up": ["M3.33337 8L8.00004 3.33333L12.6667 8", "M8 12.6667V3.33333"],
  paperclip: [
    "M14.2934 7.36666L8.1667 13.4933C7.41613 14.2439 6.39815 14.6656 5.3367 14.6656C4.27524 14.6656 3.25726 14.2439 2.5067 13.4933C1.75613 12.7428 1.33447 11.7248 1.33447 10.6633C1.33447 9.60187 1.75613 8.58389 2.5067 7.83333L8.22003 2.11999C8.72041 1.61873 9.39941 1.33678 10.1077 1.33615C10.8159 1.33553 11.4954 1.61628 11.9967 2.11666C12.498 2.61703 12.7799 3.29604 12.7805 4.0043C12.7812 4.71256 12.5004 5.39207 12 5.89333L6.27336 11.6067C6.02318 11.8568 5.68385 11.9974 5.33003 11.9974C4.97621 11.9974 4.63688 11.8568 4.3867 11.6067C4.13651 11.3565 3.99596 11.0171 3.99596 10.6633C3.99596 10.3095 4.13651 9.97018 4.3867 9.71999L10.0467 4.06666",
  ],
  play: ["M4 2L13.3333 8L4 14V2Z"],
  stop: ["M4 4H12V12H4Z"],
  download: [
    "M14 10V12.6667C14 13.0203 13.8595 13.3594 13.6095 13.6095C13.3594 13.8595 13.0203 14 12.6667 14H3.33333C2.97971 14 2.64057 13.8595 2.39052 13.6095C2.14048 13.3594 2 13.0203 2 12.6667V10",
    "M4.66675 6.66667L8.00008 10L11.3334 6.66667",
    "M8 10V2",
  ],
  info: [
    "M7.99992 14.6667C11.6818 14.6667 14.6666 11.6819 14.6666 8C14.6666 4.3181 11.6818 1.33333 7.99992 1.33333C4.31802 1.33333 1.33325 4.3181 1.33325 8C1.33325 11.6819 4.31802 14.6667 7.99992 14.6667Z",
    "M8 10.6667V8",
    "M8 5.33333H8.00667",
  ],
};

const SVG_NS = "http://www.w3.org/2000/svg";

function icon(name, size = 16) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 16 16");
  svg.setAttribute("width", String(size));
  svg.setAttribute("height", String(size));
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("class", "icon");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "1.33333");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  for (const d of ICONS[name] || ICONS.circle) {
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", d);
    svg.append(path);
  }
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
/* Modality colour tokens from the Figma Color collection; modalities without their own token share one. */
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
const modalityColour = (modality) => `var(--color-modality-${MODALITY_TOKENS[modalityKey(modality)] || "input"})`;

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

/** Tag (Figma 19:235). tone: neutral, accent, success, warning, danger, installed, refused. */
function tag(label, tone = "neutral", iconName = null) {
  return h("span", { class: `tag tag--${tone}` }, iconName ? icon(iconName, 11) : null, label);
}
function setTag(el, label, tone, iconName = null) {
  el.className = `tag tag--${tone}`;
  el.replaceChildren(...(iconName ? [icon(iconName, 11)] : []), label);
  el.hidden = false;
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
const seconds = (t) => (toNumber(t) === null ? "" : `${t < 0.1 ? t.toFixed(2) : t.toFixed(1)} s`);
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
  $("run-label").textContent = running ? "Agent running…" : state.starting ? "Starting…" : "Run agent";
  if (state.starting || running) run.setAttribute("aria-busy", "true");
  else run.removeAttribute("aria-busy");

  const send = $("send-button");
  send.disabled = !usable || state.chatting;
  send.setAttribute("aria-label", state.chatting ? "Waiting for the model" : "Send");
  send.replaceChildren(icon(state.chatting ? "loader" : "arrow-up"));
  if (state.chatting) send.setAttribute("aria-busy", "true");
  else send.removeAttribute("aria-busy");

  const label = usable ? model.label : "";
  $("chat-model").textContent = label;
  $("fit-model").textContent = label || "the model";
  $("senses-model").textContent = label || "the model";
}

// ---------------------------------------------------------------- models and the top bar

async function loadModels() {
  const select = $("model-select");
  showError($("model-error"), null);
  select.disabled = true;
  select.replaceChildren(h("option", { value: "", text: "Loading models…" }));
  $("model-label").textContent = "Loading models…";
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
  if (changed) {
    state.senses = [];
    state.reportBridge = null;
  }
  renderModelInfo();
  syncControls();
  if (changed && name) noteInThread(`Now talking to ${modelLabel(name)}.`);
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
    $("crumb-model").textContent = model.label;
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
    $("crumb-model").textContent = "Sense harness";
    picker.classList.remove("is-closed");
    picker.removeAttribute("title");
  }
  renderChips();
  renderDropModalities();
}

/** Sense chips in the top bar (Figma 19:279): native, new (amber until used in chat), refused. */
function renderChips() {
  const list = $("sense-chips");
  const model = currentModel();
  if (!model) {
    list.replaceChildren();
    return;
  }
  const chips = (model.native || []).map((m) =>
    h("li", { class: "chip", title: `${capitalise(m)}: native` },
      h("span", { class: "dot", style: `--dot: ${modalityColour(m)}` }),
      h("span", { class: "chip-label", text: capitalise(m) }),
      h("span", { class: "chip-meta", text: "native" })),
  );
  for (const sense of state.senses) {
    const card = senseCard(sense);
    const fresh = !state.usedBridges.has(sense.bridge);
    const acc = toNumber(card.accuracy);
    const meta = [kindLabel(card.kind).toLowerCase(), acc !== null ? pct(acc) : null].filter(Boolean).join(" · ");
    chips.push(
      h("li", { class: fresh ? "chip chip--new" : "chip", title: `${capitalise(sense.modality)}: ${sense.bridge}` },
        h("span", { class: "dot", style: `--dot: ${modalityColour(sense.modality)}` }),
        h("span", { class: "chip-label", text: capitalise(sense.modality) }),
        h("span", { class: "chip-meta", text: meta }),
        fresh ? h("span", { class: "badge-new", text: "NEW" }) : null),
    );
  }
  const refused = refusedRun();
  if (refused && !state.senses.some((s) => s.modality === refused.modality)) {
    chips.push(
      h("li", { class: "chip chip--refused", title: `${capitalise(refused.modality)}: the agent installed no bridge` },
        h("span", { class: "dot", style: "--dot: var(--color-status-danger)" }),
        h("span", { class: "chip-label", text: capitalise(refused.modality) }),
        h("span", { class: "chip-meta", text: "refused" })),
    );
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
      h("li", { class: "mod", style: `--mod: ${modalityColour(m)}`, title: capitalise(m) }, modalityIcon(m, 14), h("span", { class: "sr-only", text: m })),
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
  const job = state.job;
  if (!job || job.status !== "done" || job.installed || !job.request || job.request.model !== state.model) return null;
  const cards = [...job.cards.values()];
  const best = cards.reduce((a, c) => (a && toNumber(a.accuracy) >= toNumber(c.accuracy) ? a : c), null);
  return { modality: job.request.modality || (best && best.modality) || "audio", dataset: job.request.dataset, best };
}

/** Senses on the model (Figma Sense row, 22:327): native, installed, and a run that installed nothing. */
function renderSenses(senses) {
  state.senses = senses;
  const list = $("senses-list");
  const model = currentModel();
  const rows = [];
  for (const m of model ? model.native || [] : []) {
    rows.push(
      h("li", { class: "sense-row sense-row--native" },
        h("span", { class: "dot", style: `--dot: ${modalityColour(m)}` }),
        h("span", { class: "sense-text" }, h("span", { class: "sense-title", text: capitalise(m) }), h("span", { class: "sense-sub", text: "Native" })),
        tag("native")),
    );
  }
  for (const sense of senses) rows.push(senseItem(sense));
  if (!senses.length) {
    rows.push(h("li", { class: "sense-empty", text: `${modelLabel(state.model)} has no installed senses yet. Run the agent to give it one.` }));
  }
  const refused = refusedRun();
  if (refused) {
    const acc = refused.best ? toNumber(refused.best.accuracy) : null;
    rows.push(
      h("li", { class: "sense-row sense-row--refused" },
        h("span", { class: "dot", style: "--dot: var(--color-status-danger)" }),
        h("button", { type: "button", class: "sense-text", onclick: () => selectReport(refused.best && refused.best.bridge) },
          h("span", { class: "sense-title", text: capitalise(refused.modality) }),
          h("span", { class: "sense-sub", text: `Nothing installed for ${datasetTitle(refused.dataset)}` })),
        acc !== null ? h("span", { class: "sense-value", text: pct(acc) }) : null,
        tag("Refused", "refused", "ban")),
    );
  }
  list.replaceChildren(...rows);
  const active = (model ? (model.native || []).length : 0) + senses.length;
  $("senses-count").textContent = `${active} active${refused ? " · 1 refused" : ""}`;
  renderChips();
  renderDropModalities();
  renderReport();
}

function senseItem(sense) {
  const card = senseCard(sense);
  const acc = toNumber(card.accuracy);
  const errorEl = h("div", { class: "error", role: "alert", hidden: true });
  const remove = h("button", {
    type: "button",
    class: "btn btn--ghost btn--small",
    text: "Remove",
    "aria-label": `Remove the ${sense.modality || ""} sense ${sense.bridge || ""}`.replace(/\s+/g, " "),
  });
  remove.addEventListener("click", () => removeSense(sense, remove, errorEl));
  const sub = [`${kindLabel(card.kind)} bridge`, datasetTitle(card.dataset)].filter(Boolean).join(" · ");
  return h(
    "li",
    { class: `sense-row sense-row--on${state.reportBridge === sense.bridge ? " is-selected" : ""}` },
    h("span", { class: "dot", style: `--dot: ${modalityColour(sense.modality)}` }),
    h("button", { type: "button", class: "sense-text", title: `Show the report card of ${sense.bridge}`, onclick: () => selectReport(sense.bridge) },
      h("span", { class: "sense-title", text: capitalise(sense.modality) }),
      h("span", { class: "sense-sub", text: sub })),
    acc !== null ? h("span", { class: "sense-value", text: pct(acc) }) : null,
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
  const job = state.job;
  const fromRun = job && job.cards.size && (!job.request || job.request.model === state.model) ? [...job.cards.values()] : [];
  const seen = new Set(fromRun.map((c) => c.bridge));
  const fromSenses = state.senses.map(senseCard).filter((c) => !seen.has(c.bridge));
  return { fromRun, all: [...fromRun, ...fromSenses] };
}

function selectReport(bridge) {
  if (!bridge) return;
  state.reportBridge = bridge;
  renderReport();
  for (const row of document.querySelectorAll(".sense-row--on")) row.classList.remove("is-selected");
  const heading = $("report-heading");
  heading.setAttribute("tabindex", "-1");
  heading.focus({ preventScroll: true });
  heading.scrollIntoView({ block: "nearest" });
}

/** The report card: verdict, accuracy, every control against chance, and the agent's reason. */
function renderReport() {
  const el = $("report");
  const verdictTag = $("report-verdict");
  const { fromRun, all } = reportCards();
  if (!all.length) {
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
  const card =
    all.find((c) => c.bridge === state.reportBridge) || all.find((c) => c.verdict === "installed") || all[all.length - 1];
  state.reportBridge = card.bridge;
  const verdict = VERDICT_LABELS[card.verdict] ? card.verdict : "candidate";
  const kind = verdict === "installed" ? "installed" : verdict === "rejected" ? "refused" : "candidate";
  if (kind === "installed") setTag(verdictTag, "Installed", "installed", "check");
  else if (kind === "refused") setTag(verdictTag, "Refused", "refused", "ban");
  else setTag(verdictTag, "Candidate", "neutral");

  const chance = toNumber(card.chance);
  const n = toNumber(card.n);
  const acc = toNumber(card.accuracy);
  const dataset = state.datasets.find((d) => d.name === card.dataset);
  const nLabels = dataset && Array.isArray(dataset.labels) ? dataset.labels.length : null;
  const scope =
    card.scope === "closed"
      ? `Scope: ${nLabels ? `these ${nLabels} categories only` : "these categories only"}`
      : card.scope === "open"
        ? "Scope: any sound, also tested on categories it was not fitted on"
        : null;
  const facts = [
    chance !== null && chance > 0 ? `${Math.round(1 / chance)}-option questions · chance ${pct(chance)}, marked by the tick` : null,
    scope,
  ].filter(Boolean);

  const parts = [
    h("p", {
      class: "report-sub",
      text: [`${capitalise(card.modality)} → ${modelLabel(card.model || state.model)}`, `${kindLabel(card.kind)} bridge`, datasetTitle(card.dataset)]
        .filter(Boolean)
        .join(" · "),
      title: card.bridge,
    }),
    h("div", { class: "display" },
      h("span", { class: `display-value display-value--${kind}`, text: acc === null ? "n/a" : pct(acc) }),
      h("span", { class: "display-caption", text: n !== null ? `on ${n} held-out questions` : "held-out accuracy" })),
    facts.length ? h("p", { class: "report-chance", text: facts.join(". ") + "." }) : null,
    h("div", { class: "metrics" },
      metricRow(`${kindLabel(card.kind)} bridge`, acc, kind, chance, VERDICT_LABELS[verdict]),
      Object.entries(card.controls || {}).map(([key, value]) => metricRow(CONTROL_LABELS[key] || humanise(key), value, "control", chance, "control"))),
    card.reason
      ? h("p", { class: kind === "installed" ? "note note--installed" : "note" }, icon("info", 14), h("span", { text: card.reason }))
      : null,
  ];

  if (fromRun.length > 1) {
    parts.push(
      h("div", { class: "section-head" }, h("h3", { text: "Bridges in this run" }), h("span", { class: "t-caption", text: "Choose one to see its card" })),
      h("div", { class: "metrics" },
        fromRun.map((c) => {
          const v = VERDICT_LABELS[c.verdict] ? c.verdict : "candidate";
          const k = v === "installed" ? "installed" : v === "rejected" ? "refused" : "candidate";
          return h(
            "button",
            { type: "button", class: "bridge-pick", "aria-pressed": String(c.bridge === card.bridge), onclick: () => selectReport(c.bridge) },
            metricRow(c.bridge, c.accuracy, k, toNumber(c.chance), VERDICT_LABELS[v]),
          );
        })),
    );
  }
  el.replaceChildren(...parts);
  for (const row of document.querySelectorAll(".sense-row--on")) {
    const button = row.querySelector(".sense-text");
    row.classList.toggle("is-selected", Boolean(button && button.title.endsWith(` ${card.bridge}`)));
  }
}

/** Metric row (Figma 22:126): label and verdict tag, value, a bar, and the chance tick. */
function metricRow(label, value, kind, chance, tagLabel) {
  const v = toNumber(value);
  const tone = kind === "installed" ? "installed" : kind === "refused" ? "refused" : "neutral";
  const tagIcon = kind === "installed" ? "check" : kind === "refused" ? "ban" : null;
  return h(
    "div",
    { class: `metric metric--${kind}` },
    h("div", { class: "metric-head" },
      h("span", { class: "metric-label-group" },
        h("span", { class: "metric-label", text: label, title: label }),
        tagLabel ? tag(tagLabel, tone, tagIcon) : null),
      h("span", { class: "metric-value", text: v === null ? "n/a" : pct(v) })),
    h("div", { class: "track", "aria-hidden": "true" },
      h("div", { class: "fill", style: `width: ${v === null ? 0 : clamp01(v) * 100}%` }),
      chance !== null ? h("div", { class: "tick", style: `left: ${clamp01(chance) * 100}%` }) : null),
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
  renderReport(); // dataset titles and label counts may now be known
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

/** A sample clip as an Attachment row (Figma 22:234), with play and attach controls. */
function sampleItem(sample) {
  const name = sample.name || sample.id;
  const modality = (currentDataset() || {}).modality || "audio";
  const play = h("button", { type: "button", class: "icon-button" });
  play.addEventListener("click", () => togglePlay(sample));
  const attach = h("button", { type: "button", class: "btn btn--secondary btn--small" });
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
  const ref = state.sampleRefs.get(id);
  if (!ref) return;
  const attached = state.attachedSamples.has(id);
  const playing = state.playing === id;
  ref.item.classList.toggle("is-attached", attached);
  ref.item.classList.toggle("is-playing", playing);
  ref.attach.setAttribute("aria-pressed", String(attached));
  ref.attach.textContent = attached ? "Attached" : "Attach";
  ref.attach.setAttribute("aria-label", `${attached ? "Detach" : "Attach"} ${ref.name}`);
  ref.play.replaceChildren(icon(playing ? "stop" : "play", 14));
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
      attachmentChip(file.name, file.type && file.type.startsWith("audio") ? "audio" : "other", "upload", () => {
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
    if (box.value.trim() === message) {
      box.value = "";
      autosize(box);
    }
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

/** A user bubble and the model's answer under a name line (Figma Chat message, 22:204). */
function appendTurn(model, message, fileNames) {
  const reply = h("div", { class: "msg-body" }, h("p", { class: "msg-text pending", text: "Waiting for the model" }));
  const tags = h("span", { class: "name-line-tags" });
  const root = h(
    "article",
    { class: "turn" },
    h("div", { class: "msg msg--user" },
      h("span", { class: "sr-only", text: "You:" }),
      h("p", { class: "bubble", text: message }),
      fileNames.length ? h("ul", { class: "msg-files", "aria-label": "Attached files" }, fileNames.map((n) => h("li", {}, tag(n)))) : null),
    h("div", { class: "msg msg--model" },
      h("div", { class: "name-line" }, icon("cpu", 14), h("span", { class: "name", text: model.label }), tags),
      reply),
  );
  $("thread").append(root);
  updateThreadEmpty();
  scrollToEnd($("thread"));
  return { root, reply, tags };
}

function fillReply(turn, result) {
  const used = Array.isArray(result.used) ? result.used : [];
  const kinds = [...new Set(used.filter((u) => u.kind !== "none").map((u) => u.kind))];
  const missing = used.some((u) => u.kind === "none");
  turn.tags.replaceChildren(
    ...kinds.map((k) => tag(`heard via ${kindLabel(k).toLowerCase()} bridge`, "warning")),
    ...(missing ? [tag("file not received", "danger")] : []),
  );
  const parts = [h("p", { class: "msg-text", text: result.reply || "The model returned an empty reply." })];
  if (used.length) parts.push(h("ul", { class: "used-list", "aria-label": "How each file reached the model" }, used.map(usedItem)));
  turn.reply.replaceChildren(...parts);
  let changed = false;
  for (const u of used) {
    if (u.kind !== "none" && u.bridge && !state.usedBridges.has(u.bridge)) {
      state.usedBridges.add(u.bridge);
      changed = true;
    }
  }
  if (changed) renderChips();
  scrollToEnd($("thread"));
}

/** One line per file: how it reached the model, or a warning when it did not. */
function usedItem(used) {
  const missing = used.kind === "none";
  const detail = missing
    ? used.detail || "No sense for this kind of file is installed, so the model did not receive it."
    : [`${kindLabel(used.kind)} bridge ${used.bridge || ""}`.trim(), used.detail].filter(Boolean).join(" · ");
  return h(
    "li",
    { class: missing ? "used-item used-item--warn" : "used-item" },
    h("strong", { text: `${used.source || "file"}${missing ? " not received" : ""}` }),
    detail ? `: ${detail}` : "",
  );
}

/** The agent's verdict in the chat when a run ends (Figma Result card, 22:229). */
function postAgentResult(job, data) {
  const cards = [...job.cards.values()];
  const modality = capitalise((job.request && job.request.modality) || (cards[0] && cards[0].modality) || "sense");
  const installedCard = cards.find((c) => c.bridge === data.installed) || null;
  const best = cards.reduce((a, c) => (a && toNumber(a.accuracy) >= toNumber(c.accuracy) ? a : c), null);
  let result;
  if (job.status === "failed") {
    result = resultCard("refused", "The run failed", "", data.error || "The server gave no error text. Check the server log, then run the agent again.", null);
  } else if (data.installed) {
    const c = installedCard;
    const facts = c
      ? [`${kindLabel(c.kind)} bridge`, datasetTitle(c.dataset), toNumber(c.n) !== null ? `${c.n} held-out questions` : null, toNumber(c.chance) !== null ? `chance ${pct(c.chance)}` : null]
      : [data.installed];
    result = resultCard("installed", `${modality} installed`, c && toNumber(c.accuracy) !== null ? pct(c.accuracy) : "", facts.filter(Boolean).join(" · "), data.installed);
  } else {
    result = resultCard(
      "refused",
      `${modality} refused`,
      best && toNumber(best.accuracy) !== null ? pct(best.accuracy) : "",
      best ? `Best candidate: ${best.bridge}. No bridge passed its checks; its report card gives the reason.` : "No candidate passed its checks.",
      best && best.bridge,
    );
  }
  const text = job.doneMessage || (job.status === "failed" ? "" : "");
  const msg = h(
    "article",
    { class: "turn msg msg--agent" },
    h("div", { class: "name-line" },
      icon("sparkles", 14),
      h("span", { class: "name", text: "Agent" }),
      job.elapsed !== null ? h("span", { class: "name-meta", text: `· ${seconds(job.elapsed)}` }) : null),
    text ? h("p", { class: "msg-text", text }) : null,
    result,
  );
  $("thread").append(msg);
  updateThreadEmpty();
  scrollToEnd($("thread"));
}

function resultCard(verdict, title, value, body, bridge) {
  return h(
    "div",
    { class: `result result--${verdict}` },
    h("div", { class: "result-head" },
      h("span", { class: "result-title" }, icon(verdict === "installed" ? "check-circle" : "ban", 16), title),
      value ? h("span", { class: "result-value", text: value }) : null),
    body ? h("p", { class: "result-body", text: body }) : null,
    bridge ? h("button", { type: "button", class: "link-button", text: "Open report card →", onclick: () => selectReport(bridge) }) : null,
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
    steps: new Map(), // step -> {events, open}: what each Agent step card shows
    cards: new Map(), // bridge name -> its latest report card
    cardJson: new Map(), // bridge name -> last card as JSON, to skip identical redraws
    installed: null,
    doneMessage: "",
  };
  state.reportBridge = null;
  store.set("sessionStorage", "sense.job", id);
  $("timeline").replaceChildren();
  $("trace-log").hidden = false;
  $("trace-outcome").replaceChildren();
  showError($("trace-error"), null);
  renderTraceHead();
  renderSteps();
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
    syncControls();
    return;
  }
  if (state.job !== job) return;
  job.failures = 0;
  showError($("trace-error"), null);
  applyJob(job, data || {});
  if (job.status === "running") job.timer = setTimeout(() => poll(job), POLL_MS);
  else finishJob(job, data || {});
}

function applyJob(job, data) {
  if (data.request) job.request = data.request;
  job.status = typeof data.status === "string" ? data.status : "running";
  const events = Array.isArray(data.events) ? data.events : [];
  if (events.length < job.seen) {
    // The server's list shrank (for example after a restart): redraw from scratch.
    $("timeline").replaceChildren();
    job.seen = 0;
    job.steps.clear();
    job.cards.clear();
    job.cardJson.clear();
  }
  const log = $("timeline");
  const follow = log.closest("details").open && nearEnd($("trace-panel").closest(".center"));
  let cardsChanged = false;
  for (const event of events.slice(job.seen)) {
    appendEvent(job, event);
    const card = cardFrom(event.data);
    if (card) cardsChanged = upsertCard(job, card) || cardsChanged;
  }
  job.seen = events.length;
  if (events.length) job.elapsed = toNumber(events[events.length - 1].t);
  for (const card of Array.isArray(data.cards) ? data.cards : []) cardsChanged = upsertCard(job, card) || cardsChanged;
  renderTraceHead();
  renderSteps();
  if (cardsChanged) renderReport();
  if (follow) scrollToEnd($("trace-panel").closest(".center"));
}

/** True when the reader is at (or close to) the bottom, so new trace lines should keep it there. */
function nearEnd(el) {
  return el.scrollHeight - el.scrollTop - el.clientHeight < 120;
}

/** Record one trace event: a line in the full trace, and its step's card. */
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
  $("log-count").textContent = `(${$("timeline").children.length} events)`;

  if (step === "done") {
    job.doneMessage = event.message || "";
    return;
  }
  if (step === "error") {
    job.errorMessage = event.message || "";
    return;
  }
  let s = job.steps.get(step);
  if (!s) {
    s = { events: [], open: 0 };
    job.steps.set(step, s);
  }
  s.events.push(event);
  if (status === "start") s.open += 1;
  else if (status === "ok" || status === "fail") s.open = Math.max(0, s.open - 1);
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
  if (job.status === "failed") {
    outcome.replaceChildren(
      h("div", { class: "outcome", role: "alert" },
        h("strong", { text: "The run failed. " }),
        data.error || "The server gave no error text. Check the server log, then run the agent again."),
    );
  } else {
    outcome.replaceChildren();
  }
  const cards = [...job.cards.values()];
  const installedCard = cards.find((c) => c.bridge === job.installed);
  const best = cards.reduce((a, c) => (a && toNumber(a.accuracy) >= toNumber(c.accuracy) ? a : c), null);
  state.reportBridge = installedCard ? installedCard.bridge : best ? best.bridge : null;
  renderTraceHead();
  renderSteps();
  renderReport();
  postAgentResult(job, data);
  syncControls();
  loadSenses();
}

function stepStatus(key, s, job) {
  if (!s || !s.events.length) return "pending";
  if (s.events.some((e) => e.status === "fail")) return "refused";
  if (s.open > 0) {
    if (job.status === "running") return "running";
    if (job.status === "failed" || job.status === "lost") return "refused";
  }
  if (key === "decide" && job.status === "done" && !job.installed) return "refused"; // the agent declined to install
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
  return key;
}

/** The six Agent step cards (Figma 22:177), filled from the trace, plus any step the server adds. */
function renderSteps() {
  const job = state.job;
  const extra = job ? [...job.steps.keys()].filter((k) => !STEP_INDEX[k]) : [];
  const keys = [...STEPS.map((s) => s.key), ...extra];
  $("steps").replaceChildren(
    ...keys.map((key, i) => {
      const def = STEP_INDEX[key] || { title: humanise(key), idle: "" };
      const s = job ? job.steps.get(key) : null;
      const status = job ? stepStatus(key, s, job) : "pending";
      const events = s ? s.events : [];
      const last = events[events.length - 1];
      let desc = last && last.message ? last.message : def.idle;
      if (!events.length && job && job.status !== "running") desc = key === "install" && job.status === "done" ? "Nothing to install." : "Not reached.";
      const time = events.length > 1 ? toNumber(last.t) - toNumber(events[0].t) : null;
      return h(
        "li",
        { class: `step step--${status}`, title: last && last.message ? last.message : undefined },
        h("div", { class: "step-head" },
          h("span", { class: "step-num" }, icon(STEP_ICONS[status], 14), h("span", { class: "t-overline", text: `Step ${i + 1}` })),
          h("span", { class: "step-time", text: time !== null && Number.isFinite(time) ? seconds(time) : "" })),
        h("p", { class: "step-title", text: def.title }),
        h("p", { class: "step-desc", text: desc }),
        h("p", { class: "step-code", text: stepCode(key, s), title: stepCode(key, s) }),
        h("span", { class: "sr-only", text: `Status: ${status}` }),
      );
    }),
  );
}

function renderTraceHead() {
  const job = state.job;
  const pill = $("trace-status");
  if (!job) {
    pill.hidden = true;
    $("trace-request").textContent = "Not run yet. Each step below fills in from the agent's trace as it works.";
    $("trace-elapsed").textContent = "";
    return;
  }
  if (job.status === "running") setTag(pill, "Running", "accent");
  else if (job.status === "failed") setTag(pill, "Failed", "danger");
  else if (job.status === "lost") setTag(pill, JOB_WORDS.lost, "warning");
  else if (job.installed) setTag(pill, "Sense installed", "success", "check");
  else if (job.status === "done") setTag(pill, "Nothing installed", "danger", "ban");
  else setTag(pill, JOB_WORDS[job.status] || capitalise(job.status), "neutral");
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
  $("timeline").replaceChildren();
  $("trace-log").hidden = true;
  $("trace-outcome").replaceChildren();
  renderTraceHead();
  renderSteps();
  renderReport();
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

  // "New sense" in the top bar leads to the form that starts the agent.
  $("new-sense-button").addEventListener("click", () => {
    $("fit-form").scrollIntoView({ block: "nearest", behavior: "smooth" });
    const target = $("dataset-select").disabled ? $("run-button") : $("dataset-select");
    target.focus({ preventScroll: true });
  });

  const box = $("message");
  box.addEventListener("input", () => autosize(box));
  box.addEventListener("keydown", (e) => {
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
  hydrateIcons();
  bindEvents();
  updateScopeHint();
  renderAttachments();
  renderSamples("Choose a labelled set to see its sample clips.");
  renderSteps();
  renderReport();
  syncControls();
  loadModels();
  loadDatasets();
  const savedJob = store.get("sessionStorage", "sense.job");
  if (savedJob) startJob(savedJob, null, true);
}

init();
