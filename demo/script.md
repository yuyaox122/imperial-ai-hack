# Demo script

Draft by the lead's Claude, 14 November 19:40; updated 15 November for the release. Timings assume a 3-minute
live demo inside the pitch; cut from the bottom (scenario 3, then the triage) if the slot is shorter. Every
number below comes from a recorded run on Qwen3.5-0.8B or `openai-chat` (`socket_pilot/artifacts/run_*.json`
and `triage_*.json`) and is in Databricks. Say nothing on stage that is not on a card.

Rule from the plan: **lead with the agent run and its report card, not the chat.**

## Before going on stage

Do the rows in order.

| Check | How |
|---|---|
| Demo laptop has weights, embeddings, CLAP, and Databricks auth (`~/.databrickscfg`, or `DATABRICKS_HOST` and `DATABRICKS_TOKEN` in `.env`) | `.venv/bin/python scripts/setup_machine.py check`; `.venv/bin/python -m integrations.databricks smoke` |
| Server writes runs to Databricks | `.venv/bin/python -m integrations.databricks serve --port 8000` |
| Backup runs that carry their sense | Rehearse scenarios 1 and 2 through the server (the interface, **Choose a labelled set**). Each finished run is recorded under `.sense/runs` with a copy of the sense it installed. Check that `curl -s http://127.0.0.1:8000/api/runs` shows `"has_sense": true` for both; the **Runs** list does not show it |
| No sense installed at the start (so the first run installs one) | `rm -rf .sense/qwen3.5-0.8b` (the recorded runs under `.sense/runs` keep their copies) |
| Scenario 3 ready | Live, it needs `OPENAI_API_KEY`. Its tool card came from the answer cache under `<SENSE_HOME>/api_cache` on Michael's laptop; on any other laptop a live run makes the tool card's API calls again (the direct verify made 200) and its number can differ. There, either run it once in rehearsal and quote that card, or replay the record: import it with `.venv/bin/python -c "from sense import runs; print(runs.import_record('socket_pilot/artifacts/run_openai-chat_esc50_closed.json'))"` and pick it in **Runs** |
| Held-out clips ready: one rock track, one dog bark | `.venv/bin/python -c "from sense.datasets import sample_files; print(sample_files('gtzan'))"` |
| Databricks tab open on `workspace.sensorium.report_cards` and `triage_batches` | SQL editor, saved queries |
| Screen recording of a full rehearsal on the laptop | backup if the live run fails; see below |

## The run (3:00)

**0:00 – 0:15 · The gap.** Pick Qwen3.5 0.8B. Drop the rock track and ask "What genre is this?"
> "This model reads text and images. It has no hearing. We're going to ask an agent to give it a new sense, and to prove that the sense works before installing it."

**0:15 – 1:10 · Scenario 1: seen categories.** Ask for music genres (GTZAN, closed scope). In **New sense**, pick **Choose a labelled set**: the interface defaults to **Let the agent find them** (file and goal), but for audio its web fetch has not yet produced an install (the recorded run, `run_auto_qwen3.5-0.8b_music-genres.json`, fell back to the pre-wired GTZAN after HTTP 502s), so the labelled set is the reliable path here. Let the trace run:
inspect, plan, fit, verify, decide, install. The fit takes about 25 s, so talk over it, or replay at 2x.
> "It reads the model: open weights, so it can try a vector bridge that writes CLAP's embedding straight into the model's early layers, and a text bridge. It fits both, then scores each one on 200 held-out questions against controls."

Point at the report card:
- `vector:gtzan` **79.5%** [73.4, 84.5], chance 20%, swapped audio **12.5%**, no audio **22.0%**.
- "Swapped audio collapses to chance, so the model is reading the audio and not guessing from the question."
- Installed. Ask about the rock track again; the reply now names the genre, and the chat shows which bridge carried it.

**1:10 – 1:55 · Scenario 2: the agent refuses its own best number.** Ask for "any sound" (ESC-50, open scope).
- `vector:esc50` scores **96.5%**, the highest number of the day, but on sounds left out of fitting it gets **41 of 80 (51.25%)**. Rejected.
- `text:captions` (nearest of 12,000 AudioCaps captions) scores **80.5%** [74.5, 85.4], swapped 17.5%. Installed.
> "A lower score that holds on sounds it has never seen beats a higher one that doesn't. The agent made that call from the card, not from vibes."

**1:55 – 2:35 · A real task: forest monitoring.** Run `integrations.databricks triage` (or show the saved table).
> "A team listening for illegal logging wants chainsaws, hand saws and engines flagged, so a ranger doesn't listen to every clip."
- Our first wording, framed around illegal logging, was run once on the test fold (ESC-50 fold 5) and failed its no-input control: the 0.8B said "chainsaw" to all 400 clips with no audio. So we chose how to ask on a dev fold (fold 4) and ran the chosen wording once on fold 5. On dev, three multiple-choice wordings flagged every one of the 400 clips **with no audio at all**. Only an open question beat no input (21 flagged, 13 of 24 caught, 61.9% precision).
- Held-out fold 5, open question: 400 clips, 24 of them targets. Flagged **26**, caught **13 of 24**, precision **50.0%** (no input flags 0). CLAP lookup alone: 24 flagged, **23 of 24** caught, **95.8%** precision.
> "So on this task the sense does the work and the base model adds nothing. We'd ship the lookup. Our harness is what told us so: the no-input row is a control, the same as on the report cards."
- Show the no-input row: what the model says without the audio.
- Switch to Databricks: the `triage_batches` and `report_cards` tables. "Every install, every control, every flag is a row here."

**2:35 – 2:55 · Scenario 3: an API model.** With `openai-chat`, the vector bridge is unavailable (no hidden states), so the agent tries the text bridge and a `listen` tool that calls CLAP (record: `run_openai-chat_esc50_closed.json`, ESC-50, closed scope, 50 items).
- The agent installed `tool:esc50` at **96.0%** [86.5, 98.9] (swapped 22.0%, no input 18.0%) over `text:esc50` at **92.0%** [81.2, 96.9]. CLAP alone also scores 96.0%: the model asks CLAP and reports the answer.
- If asked how it was measured: the tool card's answers came from the cache of the direct verify on the same questions and clips, so that run made no new verification calls for it.

**2:55 – 3:00 · Close.**
> "Pick a model and a file it can't read. The agent finds a way in, proves it, installs it, or tells you why not, and every claim is a row in Databricks."

## Questions judges are likely to ask

| Question | Answer |
|---|---|
| Why not just use CLAP? | On ESC-50, CLAP alone scores 99.0% on the five-option questions, so we don't claim to beat it. The sense puts the audio inside a general model you can keep chatting with, and the card says exactly what that costs against the specialist. On GTZAN, CLAP's text matching scores 53.5% and the fitted bridge 79.5%. |
| The nearest-example lookup scores 84.5%, above the bridge's 79.5%. | Yes, and the card shows it. Lookup is a classifier over the labelled clips; the bridge lets the model talk about the clip. It's a control, not a competitor. |
| How do you know it isn't memorising? | Held-out clips; a split where duplicate recordings stay together (GTZAN has 14 byte-identical tracks); swapped-audio and no-audio controls; and for open scope, a refit with a fifth of the sounds left out. |
| Is it reproducible? | The GTZAN run gave identical cards when repeated (Michael, 19:30), and again through Databricks (run `bfe0e441afdf`). Cards carry 95% Wilson intervals. |
| Free text? | Weaker. Asked open questions, the vector bridge names the right genre 37.0% of the time (swapped 6.0%) and the right sound at least 45.0% of the time on strict matching. We say so in the README. |
| Does the agent find its own data? | It can: from one file and a goal it searches the web with Linkup, checks each model and dataset on the Hugging Face Hub, fetches a capped sample, verifies on it, and marks each web claim confirmed, refuted or not testable. It falls back to a pre-wired set, then asks for a folder. For 3D, video, molecules and coordinates the agent found every labelled set itself (records `socket_pilot/artifacts/run_auto_*.json`); for audio the recorded run fell back to the pre-wired GTZAN when the dataset viewer returned HTTP 502. |
| What's Databricks doing? | It holds every run, report card, trace event and triage flag in Delta tables, and logs one MLflow run per card. A sense install is the pass record in `report_cards`. |

## Backup

1. **Replay:** **Runs → Replay** on a run the server recorded during rehearsal (see the checklist); it plays at 1x, 2x or 4x and reinstalls its sense, so chat still works afterwards. A record imported from `socket_pilot/artifacts` replays the trace and cards only: it brings no sense back, so do not chat about audio after it.
2. **Screen recording:** a full rehearsal recorded on the demo laptop by Sun 10:30, also used for Devpost.
3. **Static:** screenshots of the three report cards and the Databricks tables in the slides.

## Rehearsals

| # | When | Who | Time | What broke | Fixed |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
