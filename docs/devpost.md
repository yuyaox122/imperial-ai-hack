# Devpost submission (draft)

Draft by the lead's Claude, 14 November 19:55. Submit by **Sun 11:30** (closes 12:00). Before submitting:
make the repository public, attach the recorded run, and fill in every **[TODO]**. Every number here must
match a card in Databricks or a record in `socket_pilot/artifacts/`.

- **Name:** Sensorium
- **Tagline:** An agent that gives a model a sense it doesn't have, proves the sense works, and installs it. If the proof fails, it refuses.
- **Track:** Autonomous Agents. **Sponsor prizes:** Databricks, Figma[, Linkup: only if the discovery step lands].
- **Links:** GitHub `yuyaox122/imperial-ai-hack` (public by 10:30), demo video [TODO], Figma pitch page https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=70-76 [TODO: the owner (Michael) sets "anyone with the link can view" before submitting].

## Inspiration

Most models can read text and images, and they can't hear. The usual fix is to retrain or swap models. We wanted an agent that takes the model you already use and a file it can't read, works out how to get that file in, and shows proof before it changes anything. A sense should come with a report card, not a promise.

## What it does

You pick a base model (local Qwen3.5-0.8B, or an API model) and ask for a new sense, such as "recognise music genres" or "handle any sound". The agent then:

1. **Inspects** the model: what it reads natively, and whether its weights are open.
2. **Plans** candidate *bridges*: a vector bridge that writes a frozen audio encoder's (CLAP) embedding straight into the model's early layers (open weights only); a text bridge that pastes the nearest caption or label; and a tool bridge that lets an API model call CLAP as a `listen` tool.
3. **Fits** each candidate. The vector map has a closed-form fit: 0.5 to 1.3 s per layer, about 25 s with the layer search.
4. **Verifies** each one on 200 held-out questions against controls: swapped audio, no audio, nearest-example lookup, the specialist alone, and, for "any sound", a refit with a fifth of the sounds left out.
5. **Decides and installs** the best bridge that passes the rules, or installs nothing and says why. Then you keep chatting with the same model, and each reply shows which bridge carried the audio.

Measured on Qwen3.5-0.8B:

| Request | Candidate | Accuracy [95% CI] | Swapped audio | No audio | Verdict |
|---|---|---|---|---|---|
| Music genres (GTZAN) | `vector:gtzan` | 79.5% [73.4, 84.5] | 12.5% | 22.0% | installed |
| Any sound (ESC-50) | `vector:esc50` | 96.5% [93.0, 98.3] | 18.0% | 21.0% | **rejected**: 41 of 80 (51.25%) on sounds left out of fitting |
| Any sound (ESC-50) | `text:captions` | 80.5% [74.5, 85.4] | 17.5% | 21.0% | installed |

Chance is 20% (five options). The agent turned down its highest score because that score doesn't hold on sounds it has never heard.

For an API model (`gpt-5.4-mini`), which has no hidden states to write into, the tool bridge scored 96.0% [86.5, 98.9] on 50 ESC-50 questions (swapped audio 22.0%). It equals CLAP alone, as it should: the model decides when to listen and puts the result into words.

**A task for a named user.** A forest-monitoring team wants chainsaw, hand saw and engine sounds flagged, so a ranger listens to fewer clips. ESC-50's held-out fold stands in for field recordings. We picked the wording on a dev fold (fold 4) and report once on fold 5. Our first wording, framed around illegal logging, failed its own control: the 0.8B answered "chainsaw" with no audio at all. On dev, all three multiple-choice wordings flagged every clip with no input; only an open question beat no input. On fold 5, with the 0.8B asked the open question through the caption sense, it flagged 26 of 400 clips and caught 13 of the 24 targets (54.2% recall, 50.0% precision); with no input it flagged none. CLAP lookup alone flagged 24 and caught 23 (95.8% precision). So on this task the base model adds nothing, and we say so: a ranger should run the lookup. The no-input control is what caught the bad wording, the same control the report cards use.

## How we built it

- **Harness (`sense/`):** contracts for base models, specialists, bridges, report cards and installed senses. Qwen3.5 runs through Transformers, with forward hooks that write vectors into placeholder tokens. CLAP is the frozen audio and text encoder. Verification gives every card a Wilson interval.
- **Agent (`agent/`):** a fit, verify, decide, install loop with a visible trace. An LLM planner proposes which bridges to try and which to install, and a rule planner holds the acceptance bars, so the model can't install a bridge that fails them.
- **Interface (`web/`, `server/`):** a chat harness with a model picker, file drop, sense graph, report-card inspector and replay of recorded runs, designed in Figma: a first pass in our own design system (variables with code syntax, 18 components, 10 variant sets), then the shipped Henad redesign, whose 131 variables carry the CSS names in `web/style.css` and whose parts were captured back from the running build as components; the graph node, callout, agent step, metric row and status badge each have a Refused variant.
- **Databricks (`integrations/databricks/`):** every agent run, report card and trace event goes into Delta tables in Unity Catalog. Each run record is copied to a Volume, and each verified bridge is an MLflow run with its controls as metrics. A sense install is backed by a pass row in `report_cards`. Triage flags go to `triage_batches` and `triage_clips`. The writer runs as a listener on the agent's job stream, off the run's thread.
- **Fake mode:** `SENSE_FAKE=1` swaps in a deterministic model, encoder and data, so 400+ tests and the whole interface run in seconds without weights.

## Challenges we ran into

- **Our own data leaked.** GTZAN holds 14 byte-identical recordings. Once copies were kept within one split, the pilot's 84.9% became 83.7%, and we re-measured every number.
- **A high score that doesn't generalise.** The vector bridge scores 96.5% on fitted sounds and 51.25% on sounds left out of fitting, so "any sound" needed its own control.
- **An LLM planner that misstates arithmetic.** It once called a passing bridge "below threshold". The rules, not the LLM, decide installs, and we list the misstatement as a known issue.
- **Question wording.** One phrase ("illegal logging") made the 0.8B answer "chainsaw" to silence. We now choose wording on a dev fold and report on the test fold.

## Accomplishments that we're proud of

- The agent refuses its own best number when a control says it's wrong.
- Each card carries swapped-input, no-input, lookup and specialist-alone controls, so a claim is checked against simpler explanations.
- The same run gives the identical card twice.
- An install is a database row you can query, not a log line.

## What we learned

- A frozen specialist is often as good as the bridge on its own task. The value of the bridge is getting the sense into a model you can keep talking to, and the card should say what that costs.
- Small models carry strong priors from the question; the no-input control catches it.

## What's next

- Run-time specialist discovery: search for an encoder for a new modality and verify its claimed metric on our data.
- Image and native bridges, chat with history, and a second machine for fits.

## Built with

Python, PyTorch, Hugging Face Transformers, NumPy, SciPy, soundfile, FastAPI, Uvicorn, Databricks (Unity Catalog, Delta tables, SQL warehouse, Volumes, MLflow), the OpenAI API, Figma.

**Models:** Qwen/Qwen3.5-0.8B (frozen base model); Qwen/Qwen3.5-2B (listed, research runs only); laion/clap-htsat-unfused (frozen audio and text encoder); gpt-5.4-mini through the OpenAI API (LLM planner and API base model).
**Datasets:** GTZAN (music genres); ESC-50 (environmental sounds; also the stand-in for field recordings); AudioCaps (caption text only).
**APIs:** OpenAI Chat Completions; Databricks SDK and Statement Execution API; MLflow tracking.

All project code was written during the hackathon.

## Team

Yuyao (lead; data platform, demo, submission), Michael (harness and verification), Jayden (agent autonomy), Alex (interface and design).
