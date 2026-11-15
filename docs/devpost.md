# Devpost submission (draft)

Draft by the lead's Claude, 14 November 19:55. Rewritten by Michael's Claude on 15 November for the omnimodal release. Submit by **Sun 11:30** (closes 12:00). Before submitting: make the repository public, attach the demo video, and fill in the two **[TODO]** items under Links. Every number here matches a record in `socket_pilot/artifacts/`; [reference.md](reference.md) gives each one with its source.

- **Name:** Sensorium
- **Tagline:** An agent that gives a model a sense it doesn't have, proves the sense works, and installs it. If the proof fails, it refuses.
- **Track:** Autonomous Agents. **Sponsor prizes:** Databricks, Figma, Linkup.
- **Links:** GitHub `yuyaox122/imperial-ai-hack` (public by 10:30), demo video [TODO: upload `demo/sensorium_demo.mp4` (1:56, captions, no audio) to YouTube or Vimeo and paste the link], Figma pitch page https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=70-76 [TODO: the owner (Michael) sets "anyone with the link can view" before submitting].

## Elevator pitch (at most 200 characters)

> Give a model a sense it lacks. Drop in a file it can't read: an agent writes the adapter, finds test data on the web, and installs the sense only if it beats the controls.

(171 characters.)

## About the project

Paste everything below this line into Devpost's "About the project" field. It is Markdown, and the formulas use Devpost's `$$` LaTeX blocks.

---

### Inspiration

A language model reads text and, if you are lucky, images. Hand it a 3D mesh, a video, a molecule or a MIDI file and it guesses or gives up. The usual fix is a different model or a fine-tune. We wanted an agent that keeps the model you already use, writes whatever it takes to get the file in, and proves the result before trusting it. A new sense should come with a report card, not a promise.

### What it does

You pick a base model (local Qwen3.5-0.8B, GPT through the OpenAI API, or Claude Sonnet 5.5), drop in a file it cannot read, and say what you want to know. The agent then:

1. **Detects** the file from its bytes: audio, image, video, 3D, molecule, coordinates, text, or unknown. If the model already reads it, the run stops.
2. **Builds a bridge.** For audio it fits a map from a frozen audio encoder (CLAP) into the model's early layers, or gives the model CLAP's output as text or as a tool. For anything else it **writes an adapter**: Python code, run in a sandbox, that turns the file into text or images the model reads. A package the adapter needs is vetted and installed only after you approve it.
3. **Finds labelled probe data** on the Hugging Face Hub and the web (Linkup), plus a second source whose categories the adapter was not built around. If nothing usable exists, it asks you for examples.
4. **Verifies** the bridge on held-out multiple-choice questions against controls: another file's input (swapped), no input, the model reading the raw file, and unseen categories.
5. **Installs or refuses** by fixed rules. Then you keep chatting with the same model, and each reply names the sense that carried the file.

Measured, one run per row (five options, so chance is 20%; brackets are 95% Wilson intervals):

| File and goal | Base model | Score | The control that mattered | Outcome |
|---|---|---|---|---|
| music clip, "recognise music genres" | Qwen3.5-0.8B | 79.5% [73.4, 84.5] | swapped input 12.5% | installed |
| any sound (ESC-50) | Qwen3.5-0.8B | 96.5% | 51.25% on sounds left out of fitting | refused; a caption bridge at 80.5% [74.5, 85.4] installed |
| 3D mesh, "what object is this" | Claude Sonnet 5.5 | 100.0% [92.9, 100] | raw file 42%, unseen categories 64% | installed |
| video, "what action" | Claude Sonnet 5.5 | 78.0% [64.8, 87.3] | no input 20%, unseen actions 92% | installed |
| molecule, "what does it smell like" | Claude Sonnet 5.5 | 58.0% [44.2, 70.6] | raw SMILES 46% | installed |
| coordinates, "which country" | Claude Sonnet 5.5 | 100.0% | raw coordinates 98% | refused: the model already reads coordinates |
| MIDI, "what genre" (blind test) | Claude Sonnet 5.5 | 32.0% [20.8, 45.8] | no input 20% | refused: below the bar |

For the 3D, video, molecule and coordinate runs, the agent found every labelled set itself: ModelNet40 meshes, HMDB51 clips, an MIT-licensed odour dataset and a world-cities table. Two refusals matter as much as the installs. The ESC-50 bridge had the best score of its run but fell to 51.25% on sounds it was never fitted on. On coordinates, Claude already scored 98% on the raw numbers, so an adapter had nothing to add.

The **blind test** used a format our code never mentions: a MIDI file with no metadata. The agent asked to install `mido` (the installer rated the pinned plan ok), and the adapter it wrote ran on the first attempt. Its Hub search returned one MIDI genre set, in a layout it could not read, so it asked for examples. Given 60 piano MIDI files in five genres, the adapter scored 32.0%, below the 35% bar, and the agent installed nothing.

### How we built it

**Verification is the core.** Every bridge answers held-out questions, and each card reports accuracy $\hat p$ on $n$ questions with a 95% Wilson interval:

$$\frac{\hat p + \frac{z^2}{2n}}{1 + \frac{z^2}{n}} \pm \frac{z}{1 + \frac{z^2}{n}} \sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}, \qquad z = 1.96$$

A rule planner, not a language model, holds the bars. With accuracy $a$, chance $c$, swapped input $s$, the model on the raw file $r$, and unseen-category accuracy $u$ (chance $c_u$), a text or image bridge, adapters included, is installed only if

$$a \ge c + 0.15, \qquad a \ge s + 0.10, \qquad a \ge r + 0.10, \qquad u \ge c_u + 0.15$$

where the last two apply when those controls are measured. A vector bridge needs $a \ge \max(c, s) + 0.25$, and for open scope its accuracy on held-out classes must reach $c + 0.25$ and stay within 20 points of $a$. An LLM planner may propose which bridge to install, but its choice is checked against these rules.

**The vector bridge** maps a CLAP embedding $x$ to the model's own representation $t$ of the label's text at an early block $\ell$, with a closed-form ridge fit

$$W = (X^\top X + \lambda I)^{-1} X^\top (T - \bar t)$$

where $\lambda$ is chosen by 5-fold cross-validated $R^2$. The mapped vector is written into placeholder tokens after block $\ell$, with $\ell \in \{0, \dots, 4\}$ picked on dev questions. One block fits in 0.5 to 1.3 s.

**Adapters** are designed and written by `gpt-6.1-terra` from the user's file alone, before any labelled file exists, so every probe file is held out. The code runs under macOS `sandbox-exec` with no network, a scrubbed environment, and time, disk and memory limits. Each error goes back to the model for a repair, up to three attempts in all. A helper kit gives the adapter mesh renders (trimesh and matplotlib), video frames (ffmpeg), RDKit descriptors and a Natural Earth gazetteer. A package outside the toolbox goes through our installer, which allows only hash-pinned PyPI wheels, checks OSV for known vulnerabilities, release history and look-alike names, and installs into a separate toolbox after the user approves.

**Probe search** combines the model's and our own Hub queries, the Hugging Face dataset viewer, folder-labelled repositories and Linkup web search for data tables. Found sets must be fair tests: one file type, at least three labels, no file-format words as labels, and an LLM check that the labels answer the goal. For an unknown binary format, probe files must start with the same bytes as the user's file.

**Around it:**
- A FastAPI server and a no-build web interface with a live trace, report cards, an install-approval card and replay, designed in Figma.
- Every run, card and trace event goes to Databricks Delta tables, with one MLflow run per card.
- A fake mode runs 1,640 tests in about 80 seconds with no weights or network, plus 29 Playwright browser tests.

### Challenges we ran into

- **Our own data leaked.** GTZAN holds 14 byte-identical recordings. Once copies were kept in one split, the research run's 84.9% became 83.7%, and we re-measured everything.
- **The model already knew.** On coordinates Claude scored 98% with no adapter, so we added the raw-file control. Without it the 100% adapter would have been installed for nothing.
- **Single runs move.** The adapter is rewritten each time, so we ran the three installed adapter goals again from an empty sense folder. All three installed again: 3D at 100.0% on a different mesh, video at 86.0% (78.0% the first time), and molecule at 58.0%, 12 points over the raw file in both recorded runs. An earlier, unrecorded molecule run missed that bar by 2 points, so molecules remain the borderline case.
- **The blind test broke three assumptions.** MP3 genre files reached the MIDI adapter as probes (it failed on all 50), so probes for an unknown format now have to match the file's first bytes. The word "music" mapped the format onto our audio profile, so unknown formats now get a temporary profile of their own. And "midi" and "genre" were searched apart, so the format word is now paired with the goal's words.
- **Question wording.** In a forest-monitoring triage task, one phrase ("illegal logging") made the 0.8B answer "chainsaw" to silence. We caught it with the no-input control, and from then on chose wording on a dev fold and ran the choice once on the test fold.

### What we learned

- **The control that matters depends on the input.** On 3D, swapped input (12%) showed only that the model used the file; the raw-file control showed what the adapter added, 100% against 42%. On coordinates the same control showed it added nothing.
- **Finding fair test data is harder than writing the adapter.** The MIDI adapter worked first time; usable labelled MIDI never turned up.
- **A frozen specialist is often enough.** On ESC-50, CLAP alone scores 99.0%. The value of a bridge is a sense inside a model you can keep talking to, and the card should say what that costs.
- **Small models carry priors.** Qwen3.5-0.8B could not use the coordinate adapter's text (24.4% against 23.8% on the raw file).

### What's next

- Read datasets packed as an archive plus a label index, which would have opened the MIDI set, while still flagging labels that a classifier produced.
- Repeat each run over several seeds and report the spread instead of single runs.
- Learned bridges beyond audio: today CLAP is the only specialist the harness runs.
- Larger local models (Qwen3.5-2B), and a second machine set up with `scripts/setup_machine.py`.

---

## Built with

Python, PyTorch, Hugging Face Transformers, NumPy, SciPy, soundfile, Pillow, trimesh, matplotlib, RDKit, ffmpeg, mido (installed by the agent during the blind test), FastAPI, Uvicorn, Databricks (Unity Catalog, Delta tables, SQL warehouse, Volumes, MLflow), the OpenAI API, the Anthropic API, Linkup, the Hugging Face Hub, Playwright, Figma.

**Models:** Qwen/Qwen3.5-0.8B (frozen local base model); laion/clap-htsat-unfused (frozen audio and text encoder); `gpt-6.1-terra` through the OpenAI API (adapter design and code, LLM planner, need step, and the `openai-chat` base model); Claude Sonnet 5.5 through the Anthropic API (the `claude-sonnet` base model); `gpt-5.4-mini` behind `openai-chat` in the earlier ESC-50 runs.

**Datasets:** GTZAN (music genres); ESC-50 (environmental sounds, also the stand-in for field recordings in the triage task); AudioCaps (caption text only). Found by the agent at run time: ModelNet40 meshes in a folder-labelled Hub repository, codepawl/scanned-objects-glb (CC BY 4.0), HMDB51, Hari5115/molecular-odor-dataset (MIT) and bamwor/world-cities. Supplied for the blind test and kept local: 60 files from ADL Piano MIDI.

**APIs:** OpenAI Chat Completions; Anthropic Messages; Linkup Search API; Hugging Face Hub API and dataset viewer; PyPI JSON API and OSV (installer checks); Databricks SDK and Statement Execution API; MLflow tracking.

All project code was written during the hackathon.

## Team

Yuyao (lead; data platform, demo, submission), Michael (harness and verification), Jayden (agent autonomy), Alex (interface and design).
