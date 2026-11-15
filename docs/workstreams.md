# Workloads

Status: approved by the team lead (Yuyao) on Saturday 14 November at 18:17, as proposed by Michael at
about 17:45. Four workloads, one per teammate, built on the prototype on `main` (PR #13).

Owners: Jayden has workload 1, Michael workload 2 and Alex workload 3. Yuyao took workload 4
at 19:00 (issue #39). From 15 November Michael owns the whole codebase (the owner table in `AGENTS.md`);
this split is kept as the history of who built what.

The lead also confirmed the main track, Autonomous Agents, and the sponsor prizes, Databricks and
Linkup.

## The split

| # | Workload | Owns | Serves |
|---|---|---|---|
| 1 | Agent autonomy and research | `agent/` | Main track, Linkup prize |
| 2 | Harness core and verification | `sense/`, `server/`, `scripts/`, `socket_pilot/` | Main track |
| 3 | Interface and design | `web/`, `design/` | Figma prize |
| 4 | Data platform, real task, demo and submission | `integrations/`, `eval/`, `demo/`, `docs/`, `README.md` | Databricks prize, the submission |

Tests live in `tests/`; each owner keeps the tests for their own folders.

Both reviews on the board put real feature time at about 8 hours per person before the freeze at 01:00
on Sunday. Each workload below is sized to that.

## 1. Agent autonomy and research

**Goal:** the agent, not a person, decides what it needs and finds it. Today a person picks one of two
pre-wired datasets and the audio specialist is fixed, so the only choice the agent makes is which bridge
to install.

This workload carries Michael's direction (the agent decides whether it needs a dataset and gets one
itself), Jayden's run-time discovery idea and his proposals A and C, and the Linkup role Alex recommended
(researching specialist candidates and how to integrate them).

| Task | Rough time |
|---|---|
| Data step. From the file and the user's goal, decide whether labelled examples are needed. If so, search for a matching labelled dataset, cite its source, licence, size and labels, fetch a capped sample, write and run the loading code, and hand workload 2 a folder of labelled files. Fall back to the pre-wired sets, then to asking the user. | 3 h |
| Specialist discovery with a claims ledger. A Linkup structured search returns candidate specialists, each with a claimed metric and a source link. Verification marks each claim confirmed or refuted on our data. Run-time discovery is the first move; the pre-wired registry is the fallback. | 3 h |
| Detect the modality from the file. Narrow "any format" to three or four formats that work every time. | 1 h |
| Make the LLM planner reliable. On two identical runs it proposed different plans, so the loop currently verifies every bridge regardless. | 1 h |

**Rule to keep:** the agent chooses data and specialists. Splitting, controls and the acceptance gate
stay in `sense/verify.py` and `RulePlanner` in `agent/planner.py`, so the agent cannot grade itself.

**Provides:** `agent.jobs.add_listener(fn)`, called with each trace event and the final report cards, for
workload 4. New trace step names for workload 3 to render.

**Done when:** a run that starts from one audio file and one sentence ends with a verified, installed
sense and a trace that cites its sources, twice in a row.

## 2. Harness core and verification

**Goal:** every number on a report card is right, and the harness runs on demand on more than one
machine.

Already done from Jayden's review: the specialist-alone control, a Wilson interval on the card, "these
categories only" wording, and one acceptance gate for both planners. This workload takes his remaining
points 5 and 6.

| Task | Rough time |
|---|---|
| `sense.datasets.from_folder(path, ...)`: build a labelled set from a folder with one sub-folder per label, with a capped embedding pass, a seeded split and a duplicate check across splits. Workload 1 depends on this. | 1 h |
| Score free-text answers through the vector bridge: how often the reply names the true label on held-out clips. Record the fit time from real runs, since the pilot's figure is in no saved result. | 1 h |
| A second machine: copy the Qwen3.5-0.8B weights and the cached embeddings, and write down the steps. | 1 h |
| Scenario 3 of the plan: run the API base model through a full agent run, and add a tool bridge so an API model can call the specialist. | 3 h |
| Replay mode: serve a recorded agent run so the demo survives a bad network or a slow model. | 1 h |

**Constraint:** real-model runs need the laptop that holds the weights (Michael's) until the second
machine exists. Everyone else works in fake mode: `SENSE_FAKE=1`.

**Provides:** `from_folder` for workload 1; the HTTP contract in the `server/app.py` docstring for
workload 3, changed only with a board post.

**Done when:** the three scenarios in `docs/plan.md` run on the demo laptop, the same run gives the same
card twice, and a second machine can run a fit.

## 3. Interface and design

**Goal:** the working interface matches the team's Figma design, and the evidence for the Figma prize
exists. That prize asks the team to show in the presentation how Figma was used.

Starting points: Yuyao's mockup in `design/sensorium/` (chat and file drop, model picker, sense graph,
six-step agent run, report card), the Figma file Michael posted, and issue #11, where a session of
Michael's Claude is already restyling `web/` on branch `11-interface-figma`. That session works under
whoever takes this workload, or hands over.

| Task | Rough time |
|---|---|
| Finish #11: `web/` built from the Figma tokens and components, laid out like the mockup. | 3 h |
| Show the new card fields: the interval and the specialist-alone control. Show "these categories only" on closed-scope senses. | 1 h |
| The sense graph: file, specialist, candidate bridges, held-out check, base model, drawn from the job's trace events, not hard-coded. | 2 h |
| `design/figma-notes.md`: which frames were implemented, with links, and screenshots for the submission. | 1 h |
| A replay toggle in the interface, once workload 2 has replay mode. | 0.5 h |

**Rule to keep:** show only what the API returns. The mockup's timings, cost and accuracy figures are
placeholders and mix runs.

**Done when:** all three scenarios can be driven from the interface on the demo laptop, and the Figma
notes are written.

## 4. Data platform, real task, demo and submission

**Goal:** Databricks does real work in the product, the agent is shown finishing a task for a named
user, and the team submits on time.

This carries Jayden's proposals B and D and the Databricks role Alex recommended: Databricks holds the
held-out data and the scored predictions, and provides the pass or fail record behind each install.

| Task | Rough time |
|---|---|
| Databricks smoke test before anything else: sign in, upload to a Volume, insert into a table, log one MLflow run from a laptop. MLflow tracking from a laptop on Free Edition is unconfirmed. If it is not working by 19:30, keep to tables through the SQL connector. | 0.5 h |
| Log every verification as a run and write every report card to a table, through the listener from workload 1. | 2.5 h |
| Finish a task: after an install, run the sense over an unlabelled batch and write a triage table. Jayden's suggested user is a forest-monitoring team flagging chainsaw, hand saw and engine sounds, with ESC-50 standing in for field recordings and said so. | 1.5 h |
| Demo script, three timed rehearsals, and a recorded backup run. | 1.5 h |
| Submission: make the repository public, keep `README.md` current, write the Devpost text with the list of models, APIs and datasets, and submit by 11:30 on Sunday. Submissions close at 12:00. | 1 h |
| The pitch for judging on Sunday afternoon. Lead with the agent run and its report card, not the chat. | 1 h |

**Needs:** the listener from workload 1; a real run on the demo laptop from workload 2; screenshots from
workload 3.

**Done when:** Devpost is submitted with a recorded run, and Databricks shows the runs from a live agent
run.

## Shared rules

- Stay in your folders. For a change in someone else's, ask on the board.
- `sense/contracts.py` and the HTTP contract change only with a board post first.
- `SENSE_FAKE=1 .venv/bin/python -m pytest -q` passes before every merge to `main`.
- Small PRs into `main`, merged by their author.
- Not building, per Jayden's review: Synthesia, MongoDB, Tavily as a second search, a real DNA bridge.

## Timeline (London time)

| When | What |
|---|---|
| Sat 19:30 | Databricks smoke test decided; `from_folder` merged |
| Sat 21:00 | Venue closes; each workload has one thing merged |
| Sun 01:00 | Feature freeze; fixes only after this |
| Sun 10:30 | Recorded backup run exists |
| Sun 11:30 | Devpost submitted |
| Sun 13:00 to 16:30 | Judging |

## Still open for the lead

Nothing at 19:20 on Saturday.

## Decided by the lead on 14 November

- `web/` belongs to Alex, who decides what merges there. Michael's Henad redesign (#17) and Alex's brain
  and neuron drafts both go ahead, and Alex decides how they combine.
- Michael's direction stands: the agent decides whether it needs data and finds it itself, while
  splitting, controls and the acceptance gate stay in `sense/verify.py` and `agent/planner.py`.
- 19:20: we do not commit the 39 MB of cached embeddings to the public repository. #23's fetch, checksums
  and rebuild are the route, plus a direct copy from Michael for the demo laptop.
- 19:20: the pilot's GTZAN figures are re-measured on the de-duplicated split; until then the README quotes
  only the new agent-run numbers and marks the pilot figures as approximate. The unseen-sounds result is
  written "41 of 80 (51.25%)" everywhere.
- 19:20: Databricks keeps MLflow, since the smoke test passed on every step.
