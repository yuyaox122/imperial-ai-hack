# Demo script

Draft by the lead's Claude, 14 November 19:40. Timings assume a 3-minute live demo inside the pitch; cut
from the bottom (scenario 3, then the triage) if the slot is shorter. Every number below comes from a
recorded run on Qwen3.5-0.8B (`socket_pilot/artifacts/run_*.json`) and is in Databricks. Say nothing on stage
that is not on a card.

Rule from the plan: **lead with the agent run and its report card, not the chat.**

## Before going on stage

| Check | How |
|---|---|
| Demo laptop has weights, embeddings, CLAP, and `~/.databrickscfg` | `scripts/setup_machine.py`; `python -m integrations.databricks smoke` |
| Server writes runs to Databricks | `.venv/bin/python -m integrations.databricks serve --port 8000` |
| No sense installed at the start (so the first run installs one) | `rm -rf .sense/qwen3.5-0.8b` |
| Recorded runs importable for replay | `python -c "from sense import runs; print(runs.import_record('socket_pilot/artifacts/run_qwen3.5-0.8b_gtzan_closed.json'))"` (and the ESC-50 open record) |
| Held-out clips ready: one rock track, one dog bark | `python -c "from sense.datasets import sample_files; print(sample_files('gtzan'))"` |
| Databricks tab open on `workspace.sensorium.report_cards` and `triage_batches` | SQL editor, saved queries |
| Screen recording of a full rehearsal on the laptop | backup if the live run fails; see below |

## The run (3:00)

**0:00 – 0:15 · The gap.** Pick Qwen3.5 0.8B. Drop the rock track and ask "What genre is this?"
> "This model reads text and images. It has no hearing. We're going to ask an agent to give it a new sense, and to prove that the sense works before installing it."

**0:15 – 1:10 · Scenario 1: seen categories.** Ask for music genres (GTZAN, closed scope). Let the trace run:
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
- 400 clips, 24 of them targets. Flagged **[N]**, caught **[X] of 24**, so a ranger listens to **[Y]%** of the batch. *(Numbers come from Michael's real run of PR #54; fill them in before rehearsal 1.)*
- Show the no-input row: what the model says without the audio.
- Switch to Databricks: the `triage_batches` and `report_cards` tables. "Every install, every control, every flag is a row here."

**2:35 – 2:55 · Scenario 3: an API model.** With `openai-chat`, the vector bridge is unavailable (no hidden states), so the agent gives the model a `listen` tool that calls CLAP.
- `tool:esc50` **96.0%** [86.5, 98.9] on 50 items, swapped 22.0%. *(Verified directly; the full agent run waits on Jayden's loop change. Only say "the agent installed it" once that run exists.)*

**2:55 – 3:00 · Close.**
> "Pick a model and a file it can't read. The agent finds a way in, proves it, installs it, or tells you why not, and every claim is a row in Databricks."

## Questions judges are likely to ask

| Question | Answer |
|---|---|
| Why not just use CLAP? | On ESC-50, CLAP alone scores 99.0% on the five-option questions, so we don't claim to beat it. The sense puts the audio inside a general model you can keep chatting with, and the card says exactly what that costs against the specialist. On GTZAN, CLAP's text matching scores 53.5% and the fitted bridge 79.5%. |
| The nearest-example lookup scores 84.5%, above the bridge's 79.5%. | Yes, and the card shows it. Lookup is a classifier over the labelled clips; the bridge lets the model talk about the clip. It's a control, not a competitor. |
| How do you know it isn't memorising? | Held-out clips; a split where duplicate recordings stay together (GTZAN has 14 byte-identical tracks); swapped-audio and no-audio controls; and for open scope, a refit with a fifth of the sounds left out. |
| Is it reproducible? | The GTZAN run gave an identical card twice (Michael, 19:30). Cards carry 95% Wilson intervals. |
| Free text? | Weaker. Asked open questions, the vector bridge names the right genre 37.0% of the time (swapped 6.0%) and the right sound at least 45.0% of the time on strict matching. We say so in the README. |
| What's Databricks doing? | It holds every run, report card, trace event and triage flag in Delta tables, and logs one MLflow run per card. A sense install is the pass record in `report_cards`. |

## Backup

1. **Replay:** Alex's replay toggle (PR #48) plays a recorded run at 1x, 2x or 4x and reinstalls its sense, so chat still works afterwards.
2. **Screen recording:** a full rehearsal recorded on the demo laptop by Sun 10:30, also used for Devpost.
3. **Static:** screenshots of the three report cards and the Databricks tables in the slides.

## Rehearsals

| # | When | Who | Time | What broke | Fixed |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
