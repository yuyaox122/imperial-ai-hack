# Pitch (draft)

Draft by the lead's Claude, 14 November 20:15. Five minutes for Sunday judging, with the three-minute demo in
`demo/script.md` in the middle. If the slot is shorter, cut the demo from the bottom (scenario 3, then the
triage) and keep the opening and the evidence slide. Every number here is on a card in Databricks or in a
record in `socket_pilot/artifacts/`. Rule from the plan: **lead with the agent run and its report card, not
the chat.**

## Running order

**0:00 – 0:30 · The problem (slide 1).**
> "Most models read text and images, and they can't hear. The usual fix is to retrain, or to switch to a bigger model and lose what you had. And when someone does bolt a new capability on, nobody shows you that it works."

**0:30 – 1:00 · What Sensorium is (slide 2).**
> "Sensorium is an agent. You give it the model you already use and a file the model can't read. It works out how to get that file in, fits a few candidate bridges, tests each one against controls on held-out data, and installs the one that passes, with a report card. If nothing passes, it installs nothing and tells you why."

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

> "Chance is 20%. Every card has swapped-audio, no-audio, lookup and specialist-alone controls. The same run gives the identical card twice. And when the controls say we lose, we say so: on the forest task, CLAP alone beats the language model, and that's what we'd ship."

**4:40 – 5:00 · Close (slide 4).**
> "Pick a model and a file it can't read. The agent finds a way in, proves it, and installs it, or tells you why not. Every claim is a row in Databricks. Next: let the agent find a specialist for a sense we've never seen, and verify the specialist's own claim on our data."

Name the team: Yuyao (lead, data platform, demo), Michael (harness and verification), Jayden (agent), Alex
(interface and design).

## Sponsor lines (one sentence each, only to the matching judge)

- **Databricks:** "Every agent run, report card, trace event and triage flag is a Delta table in Unity Catalog, with an MLflow run per verified bridge. An install is backed by a pass row in `report_cards`." Show the `report_cards` table.
- **Figma:** "We designed the interface in Figma first: variables with code syntax, 18 components and 10 variant sets, and the web app is built from them." Show the design-system page, then the same screen in the app.

## Slides

1. Problem: a model, an audio file, a question mark.
2. The agent loop: inspect, plan, fit, verify, decide, install; the three bridges.
3. The evidence table above, plus one line: "Triage: CLAP lookup 23/24 at 95.8% precision; the 0.8B 13/24 at 50.0%."
4. Close and team.
Backup slides: screenshots of the three report cards and the Databricks tables (see `demo/script.md`, Backup).

## Don't say

- That the agent installed `tool:esc50`, unless the full agent run with Jayden's loop change exists by Sunday.
- That the bridge beats CLAP. It doesn't on ESC-50 (CLAP alone 99.0%), and the cards show it.
- Any number that isn't on a card or in a record.
