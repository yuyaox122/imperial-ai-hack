# Devpost submission (draft)

Draft by the lead's Claude, 14 November 19:55. Rewritten by Michael's Claude on 15 November for the omnimodal release. Submit by **Sun 11:30** (closes 12:00). Before submitting: make the repository public, attach the demo video, and fill in the two **[TODO]** items under Links. Every number here matches a record in `socket_pilot/artifacts/`; [reference.md](reference.md) gives each one with its source.

- **Name:** Sensorium
- **Tagline:** An agent that gives your AI model a sense it doesn't have, tests it, and refuses to install it if it doesn't help.
- **Track:** Autonomous Agents. **Sponsor prizes:** Databricks, Figma, Linkup.
- **Links:** GitHub `yuyaox122/imperial-ai-hack` (public by 10:30), demo video [TODO: upload `demo/sensorium_demo.mp4` (1:56, captions, no audio) to YouTube or Vimeo and paste the link], Figma pitch page https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=70-76 [TODO: the owner (Michael) sets "anyone with the link can view" before submitting].

## Elevator pitch (at most 200 characters)

> Your AI can't see 3D or smell molecules. Our agent builds it new senses, tests them, and refuses any that don't work.

(117 characters.)

## About the project

Paste everything below this line into Devpost's "About the project" field. It is Markdown, and the formulas use Devpost's `$$` LaTeX blocks.

---

Sensorium gives an AI model a sense it doesn't have. Drop in a file the model can't read, such as a 3D model, a video or a molecule, and say what you want to know. An agent works out how to get the file in, tests the result, and installs the new sense only if it helps. The model's own weights never change.

### Inspiration

- Language models read text and images. A lot of useful data comes as sound, 3D models, molecules, video or MIDI, and a chat model given one of these guesses or gives up.
- The usual fixes are expensive: switch to a specialist model and lose the assistant you already use, or fine-tune, which needs labelled data, GPUs and ML skills.
- Confident wrong answers are dangerous, because people act on them. In our own tests, a small model asked to flag chainsaws in forest recordings answered "chainsaw" for all 400 clips when it got no audio at all.
- So we built an agent that keeps your model, writes whatever it takes to get the file in, and proves the result before installing anything.

### What it does

Pick a base model (Qwen3.5-0.8B running locally, GPT through the OpenAI API, or Claude Sonnet 5.5), add a file and type a goal such as "What object is this?". The agent then:

1. Detects the file type from its bytes. If the model already reads that type, it stops.
2. Builds a bridge. For audio, it fits a map from a frozen audio encoder (CLAP) into the model, or passes CLAP's output as text or a tool. For anything else, it writes an adapter: Python code, run in a sandbox, that turns the file into text or images the model can read.
3. Finds labelled test data on the Hugging Face Hub and the web (Linkup). If nothing usable exists, it asks you for examples.
4. Verifies the bridge on held-out questions against controls: another file's input (swapped), no input, the model reading the raw file, and categories it never saw.
5. Installs the sense or refuses, by fixed rules. You keep chatting with the same model, and each reply says which sense handled the file.

Each row is one recorded run. Questions have five options, so guessing scores 20%. Brackets are 95% Wilson intervals.

| File and goal | Base model | Score | The control that mattered | Outcome |
|---|---|---|---|---|
| music clip, "recognise music genres" | Qwen3.5-0.8B | 79.5% [73.4, 84.5] | swapped input 12.5% | installed |
| any sound (ESC-50) | Qwen3.5-0.8B | 96.5% | 51.25% on sounds left out of fitting | refused; a caption bridge at 80.5% [74.5, 85.4] installed |
| 3D mesh, "what object is this" | Claude Sonnet 5.5 | 100.0% [92.9, 100] | raw file 42%, unseen categories 64% | installed |
| video, "what action" | Claude Sonnet 5.5 | 78.0% [64.8, 87.3] | no input 20%, unseen actions 92% | installed |
| molecule, "what does it smell like" | Claude Sonnet 5.5 | 58.0% [44.2, 70.6] | raw SMILES 46% | installed |
| coordinates, "which country" | Claude Sonnet 5.5 | 100.0% | raw coordinates 98% | refused: the model already reads coordinates |
| MIDI, "what genre" (blind test) | Claude Sonnet 5.5 | 32.0% [20.8, 45.8] | no input 20% | refused: below the bar |

- For the 3D, video, molecule and coordinate runs, the agent found every labelled set itself: ModelNet40 meshes, HMDB51 clips, an MIT-licensed odour dataset and a world-cities table.
- The ESC-50 bridge had the best score of its run, then fell to 51.25% on sounds it was never fitted on. The agent refused it and installed a caption bridge at 80.5% instead.
- On coordinates, Claude already scored 98% on the raw numbers, so the adapter had nothing to add and the agent left it out.

### Who it's for

- Anyone with data their AI model can't read and no time or skills to train a new model: a fragrance chemist with molecules, a researcher labelling actions in video, a forest-monitoring team cutting down the clips a ranger has to listen to.
- A run takes minutes. The fully autonomous music run went from one uploaded clip to an installed sense in 110 seconds; 3D takes about six minutes, video and molecules three to four. There's no training job, because the base model stays frozen.
- A sense that fails on new data, or adds nothing over what the model already does, never reaches the user, and its report card says why.
- When a simpler tool does better, the card says so. In the forest task, CLAP on its own caught 23 of 24 target sounds against 13 and 14 for the two chat models, so our write-up tells the ranger to use CLAP.
- Every run, card and trace event is stored in Databricks, so anyone can check later why a sense was installed.

### How we built it

#### The install rule

Each report card gives accuracy $\hat p$ on $n$ held-out questions with a 95% Wilson interval:

$$\frac{\hat p + \frac{z^2}{2n}}{1 + \frac{z^2}{n}} \pm \frac{z}{1 + \frac{z^2}{n}} \sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}, \qquad z = 1.96$$

With accuracy $a$, chance $c$, swapped input $s$, the model on the raw file $r$, and unseen-category accuracy $u$ (chance $c_u$), a text or image bridge, adapters included, is installed only if

$$a \ge c + 0.15, \qquad a \ge s + 0.10, \qquad a \ge r + 0.10, \qquad u \ge c_u + 0.15$$

In words, the bridge must:

- beat guessing by 15 points;
- beat swapped input by 10 points, which shows the model is using this file;
- beat the model reading the raw file by 10 points, which shows the bridge adds something;
- beat guessing by 15 points on categories it never saw, which shows it generalises.

The last two apply when those controls are measured. A vector bridge needs $a \ge \max(c, s) + 0.25$, and for open scope its accuracy on held-out classes must reach $c + 0.25$ and stay within 20 points of $a$. An LLM planner may propose which bridge to install, and rule code checks the choice against these bars.

#### The audio vector bridge

- It maps a CLAP embedding $x$ to the model's own representation $t$ of the label's text at an early block $\ell$, with a closed-form ridge fit $W = (X^\top X + \lambda I)^{-1} X^\top (T - \bar t)$.
- $\lambda$ is chosen by 5-fold cross-validated $R^2$, and $\ell \in \{0, \dots, 4\}$ is picked on dev questions.
- The mapped vector is written into placeholder tokens after block $\ell$. One block fits in 0.5 to 1.3 s.

#### Adapters for any other file

- `gpt-6.1-terra` writes each adapter from the user's file alone, before any labelled file exists, so every test file is held out.
- The code runs under macOS `sandbox-exec` with no network, a scrubbed environment, and limits on time, disk and memory. Errors go back to the model for repair, up to three attempts.
- A helper kit gives adapters mesh renders (trimesh, matplotlib), video frames (ffmpeg), RDKit descriptors and a Natural Earth gazetteer.
- Any other package goes through our installer: hash-pinned PyPI wheels only, checked on OSV for known vulnerabilities, release history and look-alike names, and installed into a separate toolbox after the user approves.

#### Finding fair test data

- Sources: the model's and our own Hub queries, the Hugging Face dataset viewer, folder-labelled repositories, and Linkup web search for data tables.
- A found set must have one file type, at least three labels and no file-format words as labels, and an LLM checks that the labels answer the goal.
- For an unknown binary format, test files must start with the same bytes as the user's file.

#### Around it

- A FastAPI server and a no-build web interface designed in Figma, with a live trace, report cards, an install-approval card and replay.
- Databricks Delta tables for every run, card and trace event, with one MLflow run per card.
- A fake mode that runs 1,640 tests in about 80 seconds with no weights or network, plus 29 Playwright browser tests.

### Challenges we ran into

- Our own data leaked. GTZAN holds 14 byte-identical recordings; once we kept the copies in one split, 84.9% became 83.7%, and we re-measured everything.
- The model already knew. Claude scored 98% on raw coordinates, so we added the raw-file control. Without it, a 100% adapter that added nothing would have been installed.
- Single runs move, because the agent rewrites the adapter every time. Reruns from an empty folder all installed again: 3D at 100.0% on a different mesh, video at 86.0% (78.0% the first time), molecule at 58.0%. An unrecorded molecule run missed the bar by 2 points, so molecules remain borderline.
- The MIDI blind test broke three assumptions. MP3 files reached the MIDI adapter as test data, so test files for an unknown format must now match its first bytes. The word "music" pulled MIDI into our audio profile, so unknown formats now get a profile of their own. "midi" and "genre" were searched separately, so the format word is now paired with the goal's words.
- Question wording changed answers: "illegal logging" made the 0.8B say "chainsaw" with no audio. The no-input control caught it, and we now choose wording on a dev fold and run it once on the test fold.

### Accomplishments that we're proud of

- A format our code never names. Given a MIDI file with no metadata, the agent asked to install `mido`, wrote an adapter that ran first time, and asked us for examples when the one Hub set was unreadable. On 60 piano MIDI files in five genres it scored 32.0%, below the 35% bar, and refused to install its own work.
- Every number on this page comes from a recorded run in the repository (`socket_pilot/artifacts/`), refusals included.

### What we learned

- The control that matters depends on the input. On 3D, swapped input (12%) only showed the model used the file; the raw-file control showed what the adapter added, 100% against 42%.
- Finding fair test data is harder than writing the adapter. The MIDI adapter worked first time; usable labelled MIDI never turned up.
- A frozen specialist is often enough. CLAP alone scores 99.0% on ESC-50. A bridge adds a sense inside a model you can keep talking to, and the card should show what that costs.
- A small model may not be able to use an adapter: Qwen3.5-0.8B scored 24.4% with the coordinate adapter's text and 23.8% on the raw file.

### What's next

- Read datasets packed as an archive plus a label index, which would have opened the MIDI set, while still flagging labels a classifier produced.
- Repeat each run over several seeds and report the spread.
- Learned bridges beyond audio. Today CLAP is the only specialist the harness runs.
- Larger local models (Qwen3.5-2B).

---

## Built with

**Tags (10):** python, javascript, pytorch, hugging-face, openai, anthropic, fastapi, databricks, linkup, figma

Python, PyTorch, Hugging Face Transformers, NumPy, SciPy, soundfile, Pillow, trimesh, matplotlib, RDKit, ffmpeg, mido (installed by the agent during the blind test), FastAPI, Uvicorn, Databricks (Unity Catalog, Delta tables, SQL warehouse, Volumes, MLflow), the OpenAI API, the Anthropic API, Linkup, the Hugging Face Hub, Playwright, Figma.

**Models:** Qwen/Qwen3.5-0.8B (frozen local base model); laion/clap-htsat-unfused (frozen audio and text encoder); `gpt-6.1-terra` through the OpenAI API (adapter design and code, LLM planner, need step, and the `openai-chat` base model); Claude Sonnet 5.5 through the Anthropic API (the `claude-sonnet` base model); `gpt-5.4-mini` behind `openai-chat` in the earlier ESC-50 runs.

**Datasets:** GTZAN (music genres); ESC-50 (environmental sounds, also the stand-in for field recordings in the triage task); AudioCaps (caption text only). Found by the agent at run time: ModelNet40 meshes in a folder-labelled Hub repository, codepawl/scanned-objects-glb (CC BY 4.0), HMDB51, Hari5115/molecular-odor-dataset (MIT) and bamwor/world-cities. Supplied for the blind test and kept local: 60 files from ADL Piano MIDI.

**APIs:** OpenAI Chat Completions; Anthropic Messages; Linkup Search API; Hugging Face Hub API and dataset viewer; PyPI JSON API and OSV (installer checks); Databricks SDK and Statement Execution API; MLflow tracking.

All project code was written during the hackathon.

## Team

Yuyao (lead; data platform, demo, submission), Michael (harness and verification), Jayden (agent autonomy), Alex (interface and design).
