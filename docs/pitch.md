# Pitch (draft)

Draft by the lead's Claude, 14 November 20:15; updated 15 November for the release. Five minutes for Sunday judging, with the three-minute demo in
`demo/script.md` in the middle. If the slot is shorter, cut the demo from the bottom (scenario 3, then the
triage) and keep the opening and the evidence slide. Every number here is on a card in Databricks or in a
record in `socket_pilot/artifacts/`. Rule from the plan: **lead with the agent run and its report card, not
the chat.**

## Running order

**0:00 – 0:30 · The problem (slide 1).**
> "Most models read text and images, and they can't hear. The usual fix is to retrain, or to switch to a bigger model and lose what you had. And when someone does bolt a new capability on, nobody shows you that it works."

**0:30 – 1:00 · What Sensorium is (slide 2).**
> "Sensorium is an agent. You give it the model you already use, a file the model can't read, and what you want from it. It works out how to get that file in: it looks for a specialist and labelled examples on the web, fits a few candidate bridges, tests each one against controls on held-out data, and installs the one that passes, with a report card. If nothing passes, it installs nothing and tells you why."

Point at the three bridges on the slide: write the audio encoder's vector into the model's early layers (open
weights), paste the nearest caption as text (any model), or give an API model a `listen` tool.

**1:00 – 4:00 · Live demo.** Follow `demo/script.md`: the gap, scenario 1 (installed), scenario 2 (refused its
own best number), the forest triage, scenario 3 if time allows.

**4:00 – 4:40 · The evidence (slide 3).** One table, three cards on Qwen3.5-0.8B:

| Request | Bridge | Accuracy [95% CI] | Verdict |
|---|---|---|---|
| Music genres | `vector:gtzan` | 79.5% [73.4, 84.5] | installed |
| Any sound | `vector:esc50` | 96.5% on fitted sounds, 51.25% on sounds left out | **refused** |
| Any sound | `text:captions` | 80.5% [74.5, 85.4] | installed |

> "Chance is 20%. Every card has swapped-audio, no-audio, lookup and specialist-alone controls. The same run gives identical cards when repeated. And when the controls say we lose, we say so: on the forest task, CLAP alone beats the language model, and that's what we'd ship."

**4:40 – 5:00 · Close (slide 4).**
> "Pick a model and a file it can't read. The agent finds a way in, proves it, and installs it, or tells you why not. It checks what the web told it against its own numbers, and every run, card and trace event is a row in Databricks. Next: senses beyond audio, where the agent builds the adapter itself and finds its own probe data."

Name the team: Yuyao (lead, data platform, demo), Michael (harness and verification), Jayden (agent), Alex
(interface and design).

## Sponsor lines (one sentence each, only to the matching judge)

- **Linkup:** "The agent's research step asks Linkup's structured search which encoders and labelled datasets fit the goal, checks each answer on the Hugging Face Hub, and keeps every claim with its source in a ledger that our own measurements settle." Show a claims step in the trace. No real autonomous run is recorded, so quote no number from one.
- **Databricks:** "Every agent run, report card, trace event and triage flag is a Delta table in Unity Catalog, with an MLflow run per verified bridge. An install is backed by a pass row in `report_cards`." Show the `report_cards` table.
- **Figma:** "Figma and the build run one loop: Henad's 131 variables carry their CSS names as code syntax, `web/style.css` declares the same names, and the running app was captured back into Figma, where its parts became components. Every part that shows a verdict has a Refused variant." Lead with the **★ Playable proof** page (https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=78-1027): press Play, let the judge pick A (music genres, installed) or B (any sound, refused, then the text bridge). It is a Figma prototype of the verdict: 10 states, smart animate, after-delay timers, real numbers. Then the **★ Pitch** page (node 70-76), P02 for how we used Figma.

## Slides

1. Problem: a model, an audio file, a question mark.
2. The agent loop: detect, research, find data, inspect, plan, fit, verify, check claims, decide, install; the three bridges.
3. The evidence table above, plus one line: "Triage: CLAP lookup 23/24 at 95.8% precision; the 0.8B 13/24 at 50.0%."
4. Close and team.
Backup slides: screenshots of the three report cards and the Databricks tables (see `demo/script.md`, Backup).

## Don't say

- That the agent's `tool:esc50` card (96.0%, installed over `text:esc50` at 92.0%) was a fresh measurement: its answers came from the cache of the direct verify on the same questions. Saying the agent installed it is fine.
- An autonomous-run number as if it were stable: each is one run (`socket_pilot/artifacts/run_auto_*.json`), and the adapter is rewritten every time; the molecule adapter passed in one run and missed by 2 points in another.
- That the bridge beats CLAP. It doesn't on ESC-50 (CLAP alone 99.0%), and the cards show it.
- Any number that isn't on a card or in a record.
