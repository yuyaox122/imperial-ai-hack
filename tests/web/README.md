# UI smoke tests

From `tests/web`:

```sh
npm ci
npx playwright install chromium
npm test
```

`npx playwright install chromium` downloads the Chromium build the pinned Playwright
expects. Without a network, point the suite at a Chromium already on the machine, for
example an older build in the Playwright cache:

```sh
PLAYWRIGHT_CHROMIUM_EXECUTABLE=~/Library/Caches/ms-playwright/chromium_headless_shell-1243/chrome-headless-shell-mac-arm64/chrome-headless-shell npm test
```

Uses the real HTML, CSS and JavaScript with deterministic HTTP fixtures; no model,
Python server, network fonts or GPU required. Exercises layout at demo and phone sizes,
chat reload/history/edit/retry, upload restoration, streamed replies, auto-run
submission, cancellation and legacy fit/replay. Also checks working/previous chat
navigation, search, isolated drafts and attachments, dialog focus and Escape,
keyboard inspector tabs and brain report links. The job fixtures use the shapes the
server really sends (an auto run's request has no goal and dataset "" until its data
step), and cover: the stream's final `used` for a tool the model did or did not call,
refused versus not-needed bridges whatever the plan order, steps that recover from a
failed sub-step, a run that needs an examples folder, a set built by a run, replays on
the recorded model, a failed model list, and Stop shown as stopped rather than failed.
For agent-written adapters they cover the install approval card (its checks apart from the
LLM's opinion, Approve by keyboard, Decline, an answer to a request no longer open), an
adapter run's design, attempts, code and probe-file provenance, a card with the native and
unseen-categories controls each against its own chance, and chat taking any file, with a
`.glb` routed through an adapter sense. The API answers are intercepted per test in
`smoke.spec.js`; `server.js` serves only the interface's files and one slow stream.
These are UI contract checks; backend/model integration remains covered by the Python
tests.
