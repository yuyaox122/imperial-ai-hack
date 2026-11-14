# UI smoke tests

From `tests/web`:

```sh
npm ci
npx playwright install chromium
npm test
```

Uses the real HTML, CSS and JavaScript with deterministic HTTP fixtures; no model,
Python server, network fonts or GPU required. Exercises layout at demo sizes,
chat reload/history/edit/retry, upload restoration, streamed replies, auto-run
submission, cancellation and legacy fit/replay. These are UI contract checks;
backend/model integration remains covered by the Python tests.
