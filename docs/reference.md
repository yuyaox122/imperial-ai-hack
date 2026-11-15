# Sensorium reference

The details behind the [README](../README.md): setup and settings, each step of a run, the acceptance rules, every measured result with its record in `socket_pilot/artifacts/`, the known limits, and the layout of the repository.

## Run it

`uv sync` creates `.venv` with every dependency, including pytest and httpx for the tests.

### Fake mode (no weights, no data files)

`SENSE_FAKE=1` swaps in deterministic fake models, a fake encoder and small synthetic datasets, so the whole system runs in under a second without loading anything or touching the network.

```bash
SENSE_FAKE=1 .venv/bin/python -m pytest -q
SENSE_FAKE=1 .venv/bin/python scripts/demo_cli.py --model fake --dataset gtzan --scope closed
SENSE_FAKE=1 .venv/bin/python -m uvicorn server.app:app --port 8000   # then open http://127.0.0.1:8000

# an autonomous run from one file and a goal
CLIP=$(SENSE_FAKE=1 .venv/bin/python -c "from sense.datasets import sample_files; print(sample_files('gtzan')[0]['path'])")
SENSE_FAKE=1 .venv/bin/python -m agent --model fake --file "$CLIP" --goal "Make it recognise music genres"
```

In fake mode there are two models: `fake`, an open-weight model, and `fake-api`, which stands in for an API model (not open-weight, can call tools, so it takes the tool bridge). The sample clips are one-second tones written to the system temporary directory. The interface's smoke tests run in a browser against HTTP fixtures: see [tests/web/README.md](../tests/web/README.md).

### Real mode

Real mode needs four things on the machine. `scripts/setup_machine.py` fetches the weights, copies or rebuilds the embeddings and checks the result; its module docstring is a step-by-step guide (`.venv/bin/python scripts/setup_machine.py check` lists what is missing).

1. **Qwen3.5 weights**: `Qwen/Qwen3.5-0.8B` (and `Qwen/Qwen3.5-2B` if you want it) in a Hugging Face cache. The harness looks for a snapshot directory holding `config.json` and `*.safetensors`: first at `SENSE_QWEN_0P8B` / `SENSE_QWEN_2B` if they are paths, then under `SENSE_HF_CACHE`, then in the default cache. It loads from that directory and never downloads.
2. **CLAP** (`laion/clap-htsat-unfused`): downloaded once with a network connection. It embeds a new file in chat and the files of a labelled set built from a folder (`sense.datasets.from_folder`: `--folder`, `POST /api/datasets` and the autonomous data step); fitting and verification on the built-in sets use cached embeddings. Once cached it loads offline, including from the `refs/pr/3` snapshot where that repository keeps its safetensors weights.
3. **Cached embeddings** in `socket_pilot/artifacts/`: `gtzan_clap.npz`, `esc50_clap.npz` and `audiocaps_text_clap.npz`. They are not in git. Copy them from a machine that has them, or rebuild them with `scripts/setup_machine.py rebuild` (it runs `socket_pilot/clap_embed_gtzan.py`, `clap_embed.py` and `clap_text.py`, also listed in [socket_pilot/README.md](../socket_pilot/README.md#run-it)), or set `SENSE_DATA_DIR` to a directory holding copies.
4. **Raw audio** (optional, for the sample clips in the interface and for `--ask`): set `SENSE_SAMPLE_DIRS` to the GTZAN `genres` directory (files as `<dir>/<genre>/<file>.wav`) and the ESC-50 `audio` directory, separated by `:`. A set offers samples only when every one of its files is found.

```bash
# closed scope: fit, verify, install, then ask about one held-out clip
.venv/bin/python scripts/demo_cli.py --model qwen3.5-0.8b --dataset gtzan --scope closed \
    --ask <path to a GTZAN clip> "Which genre of music is this?"

# open scope ("any sound"): the vector bridge is rejected, the caption text bridge is installed
.venv/bin/python scripts/demo_cli.py --model qwen3.5-0.8b --dataset esc50 --scope open

# your own labelled folder (one sub-folder of audio files per label)
.venv/bin/python scripts/demo_cli.py --model qwen3.5-0.8b --folder <folder> --scope closed

# the interface
.venv/bin/python -m uvicorn server.app:app --port 8000   # then open http://127.0.0.1:8000
```

The held-out clips the interface offers can be listed with `.venv/bin/python -c "from sense.datasets import sample_files; print(sample_files('gtzan'))"`. They are never used for fitting.

Settings can also go in `.env` or `.env.local` (`KEY=VALUE` lines) in the repository root; variables already set in the environment win.

| Variable | Meaning | Default |
|---|---|---|
| `SENSE_FAKE` | `1` for fake models, encoder, data and research answers | off |
| `SENSE_HOME` | where installed senses, recorded runs (`runs/`), uploads, the API answer cache and the research cache are stored; also the packages a user approved for adapters (`toolbox/`) and the matplotlib font cache every adapter run gets a copy of (`cache/matplotlib`, built once) | `<repo>/.sense` |
| `SENSE_DATA_DIR` | directory of the cached `.npz` embeddings | `<repo>/socket_pilot/artifacts` |
| `SENSE_SETS_DIR` | where labelled sets built from folders are saved | `<SENSE_HOME>/datasets` |
| `SENSE_HF_CACHE` | extra Hugging Face cache holding the Qwen3.5 weights | none |
| `SENSE_QWEN_0P8B`, `SENSE_QWEN_2B` | Hugging Face id or local directory | `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B` |
| `SENSE_CLAP` | Hugging Face id or local directory | `laion/clap-htsat-unfused` |
| `SENSE_SAMPLE_DIRS` | `:`-separated directories of raw audio | none |
| `OPENAI_API_KEY` | enables the `openai-chat` API base model, and the LLM planner and the LLM need step of an autonomous run when `SENSE_LLM_PROVIDER` is `openai`; never used in fake mode | none |
| `ANTHROPIC_API_KEY` | enables the `claude-sonnet` API base model, and the LLM planner and the need step when `SENSE_LLM_PROVIDER` is `anthropic`; never used in fake mode | none |
| `SENSE_LLM_PROVIDER` | which API answers the LLM planner and the need step: `openai` or `anthropic`; any other value stops the run with an error | `openai` |
| `SENSE_LLM_MODEL` | model for the LLM planner and the need step, at that provider | `gpt-6.1-terra` (OpenAI), `claude-sonnet-5-5` (Anthropic) |
| `SENSE_API_MODEL` | OpenAI model behind `openai-chat` | `gpt-6.1-terra` |
| `SENSE_CLAUDE_MODEL` | Anthropic model behind `claude-sonnet` | `claude-sonnet-5-5` |
| `SENSE_API_VERIFY_ITEMS` | held-out questions scored for an API base model; each costs three paid requests (up to seven through the tool bridge) | 50 (at least 10) |
| `SENSE_API_CACHE` | `0` turns off the cache of an API model's multiple-choice answers under `<SENSE_HOME>/api_cache` | on |
| `LINKUP_API_KEY` | enables live web search (Linkup) in the research and data steps of an autonomous run; never used in fake mode | none |
| `SENSE_ADAPTER_UNSANDBOXED` | `1` lets agent-written adapters run where macOS `sandbox-exec` exists but cannot be applied (for example inside another sandbox); otherwise they refuse to run there. Environment scrubbing, the dead proxy and the time, disk and memory limits still apply | off |
| `SENSE_ADAPTER_RUN` | not a setting: the harness sets a random token in every adapter process's environment so it can find and stop everything a run started | set per run |
| `SENSE_GAZETTEER` | the Natural Earth populated-places GeoJSON that `sense.adapter_kit.nearest_places` reads (public domain; `scripts/fetch_gazetteer.py` downloads it). It must not be the source of a coordinate probe set's labels, or every probe would match itself | `~/.cache/sense-harness/gazetteer/ne_10m_populated_places_simple.geojson` |
| `SENSE_DBX_CATALOG`, `SENSE_DBX_SCHEMA`, `SENSE_DBX_WAREHOUSE`, `SENSE_DBX_EXPERIMENT` | where the [Databricks](#databricks) writer puts runs | `workspace`, `sensorium`, the first SQL warehouse, `/Users/<you>/sensorium` |
| `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `DATABRICKS_CONFIG_PROFILE` | Databricks auth, read by the Databricks SDK (or `~/.databrickscfg`) | none |

Only one model process should run at a time on a 24 GB laptop. Loading a second local model through the harness releases the first.

## Autonomous run: one file and a goal

The interface's default (**New sense**, then **Let the agent find them**), `POST /api/auto` and `python -m agent` start a run from one file and one sentence:

```bash
.venv/bin/python -m agent --model qwen3.5-0.8b --file <clip.wav> --goal "Make it recognise music genres"
.venv/bin/python -m agent --model qwen3.5-0.8b --file <clip.wav> --goal "..." --examples <folder>   # your own labelled folder
```

The trace has these steps (`agent/auto.py`):

1. **detect** reads the file's bytes, never its name (`agent/detect.py`): WAV, FLAC, Ogg and MP3 are audio; PNG, JPEG and a one-frame GIF are images; MP4, MOV, WebM, MKV, AVI and an animated GIF are video; GLB, glTF, PLY, OFF, STL and OBJ are 3D; MOL, SDF, InChI and SMILES are molecules; GeoJSON, GPX, KML and plain latitude and longitude lines are places; otherwise text, CSV, FASTA or unknown. A file the model takes as it is in chat (text; an image for an API model) ends the run at once: no sense is needed. Audio goes on to the steps below. Everything else (video, 3D, molecules, places, CSV, FASTA, unknown files, and an image for a local model) goes to the adapter flow: see [Any input](#any-input-agent-written-adapters).
2. **research** asks Linkup's web search which pretrained encoders fit the goal, checks each named model on the Hugging Face Hub, and records each as a claim with its source (`agent/discover.py`, `agent/research.py`). Only a specialist the harness can load is used. Today that is CLAP, for audio; other inputs take the adapter flow instead of this step.
3. **data** decides whether the goal needs labelled categories (closed scope) or open description (open scope): the LLM decides when the key for `SENSE_LLM_PROVIDER` is set (`OPENAI_API_KEY`, or `ANTHROPIC_API_KEY` for `anthropic`), otherwise a word rule; the need in the trace names the model that answered (`answered_by`, none for the rule). It then asks Linkup for labelled audio datasets on the Hugging Face Hub, checks each one on the Hub, fetches a capped sample from at most two of them through the Hugging Face dataset viewer (at most 30 files per label, 10 labels, 200 MB and 3 minutes), and builds a labelled set from the sample with `sense.datasets.from_folder` (`agent/data.py`). If that fails, it falls back to the pre-wired set (GTZAN or ESC-50) whose title and labels best match the goal's words. If none matches, the run ends and asks for a folder with at least 10 clips per category. With `--examples` (or the `examples` field of `POST /api/auto`, a folder on the server) the user's folder is used instead, in closed scope.
4. **inspect, plan, fit, verify** are the same as for a labelled set (below).
5. **claims** settles each research claim against our own measurement (`agent/claims.py`). A specialist claim is confirmed when the specialist-alone control scores at least 25 points above chance on our held-out clips, refuted when it does not, and "not testable" when that specialist was not the one run. A dataset claim is compared with what the fetched sample actually held.
6. **decide, install, done** are the same as for a labelled set: `RulePlanner` has the last word.

Without `LINKUP_API_KEY` (or with the network down), research and dataset search use earlier answers cached under `<SENSE_HOME>/research` if there are any; otherwise the trace says research was unavailable, and the run uses CLAP and the pre-wired sets. Without that key, the need step uses the word rule. Fake mode never touches the network: research gives canned answers and every fetch fails, so a fake run that needs data, with no examples folder, ends on a pre-wired set or on a request for a folder.

Status: the autonomous flow is covered by tests that never touch the network: fake-mode runs, Hub and dataset-viewer answers recorded on 14 November, and a Linkup answer built from Linkup's documentation (`tests/fixtures/research/`). One real autonomous run is recorded (`run_auto_qwen3.5-0.8b_music-genres.json`, 15 November): Qwen3.5-0.8B, a GTZAN rock clip uploaded as `mystery clip.wav`, and the goal "Make it recognise music genres", with live Linkup search and `gpt-6.1-terra` for the need step and planner. Research named two models, neither loadable here, so CLAP was used. Dataset search found 6 candidates, but the Hugging Face dataset viewer answered HTTP 502 for both that were tried, so the run fell back to the pre-wired GTZAN set and said so. It installed `vector:gtzan` at 79.5% [73.4, 84.5] (swapped input 12.5%, no input 22.0%) over `text:gtzan` at 47.5%, in 110 s.

## Any input: agent-written adapters

For a file no specialist is wired for, the agent writes the missing piece itself (`agent/synth.py`, `sense/adapters.py`).

**How it works.** An LLM designs an adapter from what `agent/detect.py` found and the goal: what to extract from the file, whether to give the model text or images, and which libraries of the approved toolbox to use. It then writes `adapt(path)` from your file alone, and the harness runs that code on the file in a sandbox, sending each error back for a repair, up to three attempts. The agent finds labelled probe files: your examples folder, else a public dataset it finds on the Hugging Face Hub and the web (`agent/hub_probes.py`), else files the LLM generates with the label built into their content; if none can be had, the run stops and says which examples to provide. The frozen base model answers multiple-choice questions (up to five options) about each probe file through the adapter, and `RulePlanner` installs the adapter only if it clears the bars below. Chat then sends every later file that `agent/detect.py` describes like the files the adapter was verified on (the same modality, and the format of your file or of a probe file) through it, and the reply says so: `Read via adapter:<name>`. A file in another format is not given to it (an OBJ adapter never sees a PLY); a mesh, molecule or coordinates in text form then goes to the model as text, as it does when the adapter fails on the file. An adapter for a file `agent/detect.py` does not recognise is never installed under one of its modality names (an "audio" design for an M4A memo becomes `audio-m4a`), so it cannot replace a verified sense.

**Safety.** The code runs in a separate process under a macOS `sandbox-exec` profile with no network, on a private copy of the file, with secrets removed from its environment and time, disk and memory limits; where the sandbox cannot be applied it refuses to run unless `SENSE_ADAPTER_UNSANDBOXED` is `1`. Before it runs, the code is checked to import only the standard library and the toolbox, with no network modules and no `ctypes`. A package outside the toolbox is installed only after `agent/installer.py` has vetted a pinned, hashed plan (PyPI wheels only, known vulnerabilities, release history, look-alike names) and you have approved it, in a card in the interface's Run tab (`POST /api/jobs/<id>/approval`) or on the terminal; it goes into `<SENSE_HOME>/toolbox`, never the main environment, and an LLM's review is shown beside the checks but never decides.

**Honesty controls.** The adapter is written before any probe file exists and never sees a label, a label folder or the candidate answers, so every probe file is held out. Besides chance, swapped input and no input, it must beat the model reading the raw file by 10 points (`native`, when the files are text) and score 15 points above its own chance on categories from another source that it was not built around (`unseen_categories`, against `control_chance`); when no such set is found, the trace and the card say "Unseen categories not measured" and why. Generated probe files are marked as a consistency check, not ground truth, and labels written into the files' own content are flagged.

**Try it.** The design and the code come from the LLM of `SENSE_LLM_PROVIDER`, so a real run needs `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`); `LINKUP_API_KEY` adds web search to the Hub search for probe files. In the interface, press **New sense**, drop any file (a `.glb` or `.obj` mesh, a video, a molecule, a GPX track) and say what the model should tell from it. From the terminal:

```bash
# an install request is asked on the terminal, y/N, and No is the default
.venv/bin/python -m agent --model qwen3.5-0.8b --file chair.obj --goal "Tell which kind of furniture this 3D model is" \
    --save adapter_run.json
.venv/bin/python -m agent ... --examples <folder>   # your own labelled files, one sub-folder per label
.venv/bin/python -m agent ... --approve-installs    # approve every install request without asking (it says so)

# the whole flow offline, with canned LLM answers and an in-memory Hub
SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_adapter_flow.py
```

`--save` writes the run (request, timed trace, cards) in the `sense.runs` format, and `sense.runs.import_record` adds it to **Runs** for replay. In fake mode the LLM gives no design, so a fake run on such a file ends at the design step; the offline tests above stand in for it.

**Measured** (15 November, real mode, through `POST /api/auto` with one file and one goal and no examples folder; `gpt-6.1-terra` designed and wrote each adapter and judged the found datasets; every probe set was found by the agent on the web; 50 held-out questions per card for API models, 160 for Qwen; the records are `socket_pilot/artifacts/run_auto_*.json`):

| Input and goal | Base model | Probe data the agent found | Adapter | Controls | Outcome |
|---|---|---|---|---|---|
| 3D model (`.glb`), "what object is this" | Claude Sonnet 5.5 | ModelNet40 meshes, 8 objects (folder-labelled repository) | renders, 100.0% [92.9, 100] | swapped 12%, no input 20%, raw file 42%, unseen 64% (8 scanned-object categories, chance 20%) | installed |
| same | `openai-chat` (`gpt-6.1-terra`) | same | renders, 100.0% | swapped 18%, no input 28%, raw file 58%, unseen 72% | installed |
| video (`.mp4`), "what action" | Claude Sonnet 5.5 | HMDB51, 8 actions | frames, 78.0% [64.8, 87.3] | swapped 8%, no input 20%, unseen 92% (8 other HMDB51 actions: another upload of the same corpus) | installed |
| molecule (SMILES), "what does it smell like" | Claude Sonnet 5.5 | 8 odour labels (Hari5115/molecular-odor-dataset, MIT) | RDKit descriptors as text, 58.0% [44.2, 70.6] | swapped 10%, no input 14%, raw SMILES 46%; unseen not measured (no second source found) | installed, 12 points over the raw file; a repeat gave the same 58.0% against 46.0%, and an earlier, unrecorded run fell 2 points short of the 10-point bar |
| coordinates, "which country" | Claude Sonnet 5.5 | 8 countries (bamwor/world-cities) | nearest places as text, 100.0% | raw coordinates 98%, unseen 100% (8 other countries) | refused: the model already reads coordinates |
| same | Qwen3.5-0.8B (local) | same | 24.4% on 160 | swapped 21.9%, raw file 23.8% (chance 20%) | refused: the small model cannot use the adapter's text |

One run per row, so a single number moves between runs (the adapter is rewritten each time and the found data can differ); the intervals are on the cards. The video and molecule goals were also each run a second time on a fresh `SENSE_HOME` (15 November, about 08:10) and installed again. Video (`run_auto_claude-sonnet_video-repeat.json`): 86.0% [73.8, 93.1] on 50 questions, swapped input 8%, no input 22%, unseen actions 96% (first run 78.0%). Molecule (`run_auto_claude-sonnet_molecule-repeat.json`): 58.0% [44.2, 70.6], swapped input 10%, no input 16%, raw SMILES 46.0%, so 12 points over the raw file again; unseen categories were again not measured (the one candidate second source failed in the Hugging Face dataset viewer with HTTP 500). Both reruns found the same datasets as the first runs, so they repeat the adapter and the model's answers, not the data. The 3D goal on Claude Sonnet 5.5 was repeated once, with another ModelNet40 test mesh (a chair instead of the guitar) on a fresh `SENSE_HOME` while recording the demo video (`run_auto_claude-sonnet_3d-repeat.json`): it found the same two datasets and installed again at 100.0% [92.9, 100], with swapped input 8%, no input 18%, raw file 34% and unseen categories 70%, and in chat afterwards Claude named the chair through the adapter.

**Blind test: a format the code never names.** We gave Claude Sonnet 5.5 a MIDI file (written by a script: a waltz-like melody with no title or text events) and the goal "What genre of music is this?". MIDI appears nowhere in the detection code, the profiles or the toolbox. `agent/detect.py` reported unknown bytes, and the agent designed a text adapter. It asked to install `mido`; the installer rated the pinned plan ok, and the operator approved it. The adapter it wrote ran on the first attempt and reported tempo, meter, instruments, ranges and rhythm. Three runs followed (records in `socket_pilot/artifacts`):

1. `run_auto_claude-sonnet_midi-first.json`: the search found only audio genre sets, and their MP3 files reached the adapter, which failed on all 50, so nothing was verified. Since then, probe files for an unknown binary format must start with the same bytes as your file (PR #103).
2. `run_auto_claude-sonnet_midi.json`: the search now pairs the format's name with the goal's words ("midi genre") and ranks MIDI datasets first. It found the one MIDI genre set the Hub search returns, `projectlosangeles/lakh-clean-midi-genres`, and could not use it: the files sit in a tar archive with the labels in a separate index (the labels are also a genre classifier's predictions, not human ones). The run stopped and asked for labelled examples.
3. `run_auto_claude-sonnet_midi-examples.json`: given an examples folder of 60 piano MIDI files in 5 genres from ADL Piano MIDI (classical, country, jazz, latin, rock; one file per artist, renamed so no title reaches the model; used locally, as the set states no licence), the adapter scored 32.0% [20.8, 45.8] on 50 held-out files (chance 20%, swapped 10%, no input 20%). That is below the 35% bar, so nothing was installed. ADL keeps only the piano tracks, so the instrumentation that most marks a genre is gone.

The flow worked end to end on a format it had never seen: it wrote and sandboxed the code, vetted the install, refused unusable data and asked for examples. With those examples it measured a weak adapter and refused to install it.

## Audio bridges: measured

Measured on 14 and 15 November 2026 on an M4 Pro with 24 GB, Qwen3.5-0.8B in float16 on MPS, through `scripts/demo_cli.py` and the HTTP server. Each card scores 200 held-out five-option questions (chance 20%) and carries a 95% Wilson interval for its accuracy. GTZAN numbers use the de-duplicated split (PR #43): GTZAN holds 14 byte-identical recordings, and copies now stay in one split. The run records are in `socket_pilot/artifacts/run_*.json`, and every run is also in Databricks (see [Databricks](#databricks)).

The acceptance rules are in one place, `RulePlanner` in `agent/planner.py`. A vector bridge passes when it beats chance and swapped input by 25 points each; for open scope it must also score at least 45% on classes left out of fitting and stay within 20 points of its accuracy on fitted classes. A text, tool or image bridge passes when it beats chance by 15 points and swapped input by 10. The vector bridge is preferred when the two are within 3 points. An agent-written adapter's card (text or image) may carry two more controls, and must then also beat the model given the raw file by 10 points (`native`) and score 15 points above chance on categories it was not built around (`unseen_categories`, against that set's own chance in `control_chance`).

With the key for `SENSE_LLM_PROVIDER` set (`OPENAI_API_KEY`, or `ANTHROPIC_API_KEY` for `anthropic`), the LLM planner (`agent/planner_llm.py`) proposes which bridges to try and which to install; the plan and decide events name the model that answered (`answered_by`, none when the rules decided). The loop verifies every bridge the model can take whatever the planner proposes (vector and text for an open-weight model; text, plus tool when the model can call tools, for an API model), and the planner may only install a bridge that passes the rules above; otherwise the rules decide.

| Run | Candidate | Accuracy | Swapped input | No input | Nearest-example lookup | Specialist alone | Text bridge | Unseen classes | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| GTZAN, closed | `vector:gtzan` | 79.5% | 12.5% | 22.0% | 84.5% | 53.5% | 47.5% | | installed |
| GTZAN, closed | `text:gtzan` | 47.5% | 20.0% | 22.0% | 84.5% | 53.5% | | | passed the rules, not chosen |
| ESC-50, open | `vector:esc50` | 96.5% | 18.0% | 21.0% | 99.5% | 99.0% | 80.5% | 51.25% (41 of 80) | rejected |
| ESC-50, open | `text:captions` | 80.5% | 17.5% | 21.0% | 99.5% | 99.0% | | | installed |

`text:gtzan` pastes the label CLAP matches best ('sounds like "rock"'); `text:captions` pastes the nearest of 12,000 AudioCaps captions. Nearest-example lookup answers the same five-option questions without the base model: it picks the offered label whose fitted clip is closest to the held-out clip in CLAP space. Specialist alone is CLAP with no base model and no fitted examples: it picks the offered label whose text is closest to the clip. Unseen classes refits the vector bridge with a fifth of the labels left out and scores it on held-out clips of those labels: 40 clips of 2 genres for GTZAN, 80 clips of 10 sounds for ESC-50.

A whole agent run took 56.0 s (GTZAN closed) and 64.0 s (ESC-50 open). The vector bridge's layer is chosen among blocks 0 to 4 on 200 dev questions. The closed-form fit of one block takes 0.5 to 0.7 s on GTZAN (597 clips) and 1.1 to 1.3 s on ESC-50 (1,200 clips); the whole fit step, five blocks each scored on the dev questions, takes 23 to 26 s. A second GTZAN closed run on the same laptop (14 November, 19:30) gave identical cards in 53.6 s, and a third, through `integrations.databricks serve` (Databricks run `bfe0e441afdf`), gave the same cards again; these repeats were reported on the team board and are not in `socket_pilot/artifacts`.

A labelled set built from the GTZAN genre folders with `sense.datasets.from_folder` (50 clips per genre, 100 held-out questions) gave `vector:gtzan-folder` 77.0% [67.9, 84.2] (swapped input 16.0%, no input 17.0%) in a 30.9 s agent run (`run_qwen3.5-0.8b_gtzan-folder_closed.json`).

The research run (`socket_pilot/run_music.py`), re-measured on the de-duplicated split (PR #51), gives 83.7% [78.6, 88.8] for the GTZAN vector bridge on 196 held-out tracks (84.9% before de-duplication), against 79.5% here; its lookup scores 78.1% and swapped input 14.3%. With rock and classical both held out of fitting it scores 51.8% [44.7, 58.8], where chance is 50% (53.5% before). The two runs differ in split, scored tracks and option order as well as in the duplicates, and every change is inside the 95% intervals. `socket_pilot/README.md` shows both columns. Over three question seeds on the earlier split, blocks 0, 1 and 2 averaged 80.8, 82.2 and 79.0%, so the layer choice and the question draw also move the number by a few points (an earlier measurement, not re-run on the de-duplicated split and not in `socket_pilot/artifacts`).

In free chat the vector bridge is much weaker than on multiple choice. Asked "Which genre (or sound) is in the audio clip? Answer in a few words." on 100 held-out clips, the reply named the true label and no other 37.0% of the time on GTZAN (95% interval 28.2 to 46.8%; swapped input 6.0%, no input 8.0%) and 45.0% on ESC-50 (35.6 to 54.8%; 0.0% and 0.0%). On GTZAN the model always names some genre but leans to hip hop and rock. On ESC-50 the matching is strict, so "Clock" for clock tick counts as naming no label, and 45% is a floor. Results: `socket_pilot/artifacts/free_text_*.json`, from `scripts/score_free_text.py`.

In the GTZAN run the LLM planner chose the install; in the ESC-50 run its choice failed the rules and the rule planner decided. Its proposed plan differed between two otherwise identical ESC-50 runs (vector only, then text only), which is why the loop verifies every available bridge.

The API base model (`openai-chat`, `gpt-5.4-mini`) has been through two full agent runs, each on 50 held-out questions:

- **ESC-50, open scope** (14 November, `run_openai-chat_esc50_open.json`). The loop then tried only the text bridge for an API model. `text:captions` scored 90.0% (95% interval 78.6 to 95.7%), against 18.0% with swapped input, 18.0% with no input, 100.0% for nearest-example lookup and 96.0% for the specialist alone, and was installed. On this task the API model with captions does not beat CLAP alone. The run took 25.8 s and 150 API calls, plus the planner's 2; a repeat run gave the identical card from the answer cache in 2.0 s with no verification calls (reported on the team board, not recorded). In chat on five held-out clips (stored in the record), four were described correctly; a chainsaw clip got the nearest caption "A motorcycle engine revving up" and the model repeated it. Replies only reword the caption the model is given.
- **ESC-50, closed scope** (15 November, `run_openai-chat_esc50_closed.json`; scenario 3 of the plan). The loop tried text and tool (the vector bridge needs hidden states). `tool:esc50`, which lets the model call CLAP as a `listen` tool, scored 96.0% [86.5, 98.9] (swapped input 22.0%, no input 18.0%, specialist alone 96.0%) and was installed over `text:esc50` at 92.0% [81.2, 96.9]; both passed the rules. The run took 20.2 s. The tool card's answers came from the answer cache filled by the direct verify in `tool_openai-chat_closed.json` (the same questions and clips; 200 API calls in 41.4 s), so this run made no new calls for it. On a machine without that cache (`<SENSE_HOME>/api_cache`) the run makes those calls again, and the tool card's number can differ.

Chat through the installed `vector:gtzan` bridge worked on held-out clips (14 November, before the de-duplication fix; these chats are not in a record). The bridge mapped `reggae.00009.wav` to the label "reggae" (cosine 0.97), and the model's reply named reggae; it then described rhythm and instruments that the vector does not carry. Through the server, `classical.00082.wav` was answered as classical, and one message with an upload named `my song.wav` (a metal clip) and `jazz.00061.wav` was answered as metal and jazz. A `.txt` file in the same message was then reported as not shown to the model; since PR #78 a text file's text goes to the model as it is. An audio file that cannot be read is reported as not shown, while the other files still reach the model.

A run names the sense it affects. When it installs a bridge for a model that already has an audio sense, the trace says which sense it replaced and with what score. When it installs nothing, the trace names the earlier sense that chat keeps using. Each chat took 2.6 to 4.1 s, including CLAP's first load. The endpoints that existed on 14 November were exercised against the real model: models, datasets, fit, jobs (including 409 for a second run and 400 for a bad scope), senses, samples (the WAV is served), chat and the interface. The endpoints added since (`POST /api/datasets`, `POST /api/auto`, `POST /api/chat/stream`, chat history, `DELETE /api/jobs/<id>`, `GET /api/runs` and `POST /api/replay`) are covered by fake-mode tests; no real-mode record of them is in this repository.

## A real task: forest-monitoring triage

A forest-monitoring team wants chainsaw, hand saw and engine sounds flagged, so a ranger listens to fewer clips. ESC-50 stands in for field recordings: 400 clips per fold, 24 of them targets. `eval/triage.py` asks the base model about each clip through its installed sense, and asks again with no audio as a control. As a second control, CLAP lookup flags the clips whose nearest fitted example is a target, with no base model.

Our first wording ("ranger" below) was run once on fold 5 and failed its no-input control: with no audio the 0.8B answered "chainsaw" for all 400 clips, and through the sense it flagged 369 (`triage_qwen3.5-0.8b_esc50.json`). We then chose among four wordings on fold 4 and ran the chosen one once on fold 5. On fold 4, Qwen3.5-0.8B with `text:captions`:

| question | flagged | no input flags | caught | precision |
|---|---|---|---|---|
| ranger: "A forest ranger is listening for illegal logging. Which of these is in the audio clip?" | 372 | 400 | 24/24 | 6.5% |
| neutral: "Which of these is in the audio clip?" | 295 | 400 | 24/24 | 8.1% |
| yesno: "Is there a <target> in the audio clip?" | 212 | 400 | 23/24 | 10.8% |
| chat: "What sound is in the audio clip? Answer in a few words." | 21 | 0 | 13/24 | 61.9% |

The three wordings that offer the targets flag every clip with no audio at all; the model answers "chainsaw" to nothing. Only the open question beats no input. On fold 5 it flagged 26 clips and caught 13 of the 24 targets (54.2% recall, 50.0% precision, 3.5% false alarms): 6 of 8 chainsaws, 7 of 8 engines and none of the 8 hand saws. With no input it flagged none. CLAP lookup flagged 24 and caught 23 (95.8% precision, 0.3% false alarms).

The API model through its own caption sense (`openai-chat`, `text:captions` at 90.0%), asked the same open question on fold 5, flagged 23 and caught 14 of 24 (58.3% recall, 60.9% precision, 2.4% false alarms): 7 of 8 chainsaws, 6 of 8 engines and 1 of 8 hand saws; with no input it flagged none. On this task neither base model adds anything over CLAP, and a ranger should run the lookup.

Records in `socket_pilot/artifacts/`: `triage_qwen3.5-0.8b_esc50.json` (the first wording on fold 5), `triage_qwen3.5-0.8b_esc50_dev_*.json` (the four wordings on fold 4), `triage_qwen3.5-0.8b_esc50_test_chat.json` (fold 5), and `triage_openai-chat_esc50_dev_chat.json` and `triage_openai-chat_esc50_test_chat.json` (the API model on folds 4 and 5).

```bash
.venv/bin/python -m eval.triage --question chat --fold test                        # run it with the installed sense
.venv/bin/python -m integrations.databricks triage --question chat --fold test     # the same, written to Databricks
.venv/bin/python -m integrations.databricks triage --from socket_pilot/artifacts/triage_*.json   # write saved records
```

## Databricks

Every agent run is recorded in a Databricks workspace (Free Edition), so each install has a pass or fail record behind it.

- Delta tables `workspace.sensorium.runs`, `report_cards` and `trace_events`. Each report card is a row, with the accuracy, its interval, every control as its own column, the verdict and whether it was installed.
- The run record (`run.json`) in the Volume `/Volumes/workspace/sensorium/runs/<run id>/`.
- One MLflow run per verified bridge in the experiment `/Users/<you>/sensorium`, with the accuracy and controls as metrics and the verdict as a tag.
- Triage: `triage_batches` (one row per method per batch, with the question) and `triage_clips` (one row per clip, with the reply), plus one MLflow run per batch.

```bash
uv sync --group databricks                                  # client libraries, optional
.venv/bin/python -m integrations.databricks smoke           # checks auth, the warehouse, a Volume upload, a table insert, MLflow
.venv/bin/python -m integrations.databricks serve --port 8000    # the usual server; every finished run is written to Databricks
.venv/bin/python -m integrations.databricks backfill socket_pilot/artifacts/run_*.json   # recorded runs
```

Auth comes from `~/.databrickscfg` or `DATABRICKS_HOST` and `DATABRICKS_TOKEN` (in the environment or in `.env`). Writing a run replaces that run's rows, so backfilling twice is safe. The writer runs on its own thread after the run ends and never holds up an agent run (`integrations/databricks/sink.py`).

## Known limits

- **The autonomous audio flow has one recorded real run, and it used the pre-wired set.** In that run the Hugging Face dataset viewer failed (HTTP 502) for the fetched candidates, so the fetch-from-the-web path has not yet produced a recorded real install for audio. (An earlier attempt the same night installed nothing because the need step's "music genres" label blocked GTZAN; PR #80 fixed that.)
- **Only audio gets a learned bridge.** The harness can load one specialist, CLAP (`agent/discover.py`, `RUNNABLE`). Research may name encoders for other inputs, and they are recorded as claims, but none is run; other inputs get an agent-written adapter instead.
- **Adapter results rest on one or two runs.** The adapter is rewritten every time, so a number moves between runs. The 3D, video and molecule goals have two recorded runs each, all installed (video 78.0% then 86.0%); the molecule adapter cleared the raw file by 12 points in both, but fell 2 points short of the 10-point bar in an earlier, unrecorded run. The coordinates and MIDI rows are single runs, and the two video and two molecule runs used the same datasets. The video run's unseen actions come from another upload of the same corpus (HMDB51), so they are less independent than the other unseen sets, and the molecule run found no second source, so its unseen control was not measured. An adapter is only as good as what its code extracts: a pass shows the base model answers better with it than without it, not that it understands the file.
- **An unknown format needs data the agent can read.** In the blind MIDI test the one MIDI genre set the Hub search returned keeps its files in a tar archive with the labels in a separate index, a layout the probe search does not read, so the run asked for examples; with 60 examples the adapter scored 32.0%, below the bar.
- **No labelled data, no sense.** Verification needs labelled held-out clips. When the agent finds no usable dataset on the Hub and no pre-wired set matches the goal's words, it stops and asks for a folder; it does not fall back to an open-scope set, even for "Describe anything you hear".
- **GTZAN open scope sits on the bar.** In an earlier run, before the de-duplication fix and not recorded, `text:captions` scored 35.5% against a 35.0% bar, and 32.0 to 37.0% over question seeds 0 to 4, so whether it installs depends on the draw. Use ESC-50 to demonstrate open scope.
- **The unseen-classes control is noisy.** For ESC-50 it was 41.2% in an earlier, unrecorded run and 51.25% (41 of 80) in the recorded one (80 clips of 10 left-out sounds; the chosen layer differed). Both are far below the 96.5% on fitted sounds, which is what the within-20-points rule checks.
- **Nearest-example lookup beats the vector bridge.** On the same questions, lookup in CLAP space scores 84.5% on GTZAN against the bridge's 79.5%, and 99.5% on ESC-50 against 96.5%. The planner does not use lookup as a bar, so the bridge is installed anyway.
- **On ESC-50 the specialist alone is as good as the bridge.** CLAP matching label text scores 99.0% with no base model. On GTZAN it scores 53.5%, so there the fitted bridge adds 26.0 points over the specialist's own text matching. The tool bridge on `openai-chat` scores what CLAP alone scores (96.0%), as it should: the model asks CLAP and reports the answer.
- **The vector bridge passes a label, not a description.** The model treats the written vector like the word for the label: for the metal and jazz clips it said it was answering "Based on the names provided". Anything it says beyond the label is invented.
- **The LLM planner's plan is unreliable.** See above; its install choice is checked against the rules. One planning call and one decision call are made per run.
- **The LLM planner's stated reasons can be wrong.** In the GTZAN run it wrote that `text:gtzan` (47.5%, swapped input 20.0%) was "below the required accuracy > swapped + 0.10 threshold", which is false: the text bar is 35% and the swapped bar 30%, and it cleared both. The install was still right, because the rules check the choice, but the card's reason text comes from the model and is not checked.
- **No image bridge for audio.** No bridge renders audio as an image such as a spectrogram (agent-written adapters do give the model images: mesh renders and video frames). API models are verified on 50 held-out questions, not 200, so their intervals are wider (about 12 points either way).
- **Qwen3.5-2B is listed but untested in this harness.** It loads through the same code as the 0.8B; only the research scripts have run it.
- **Chat history is text only.** Up to 8 earlier turns (4,000 characters each) go with each message, but their files do not, so a follow-up about a file needs the file attached again.
- **Images in chat reach API models only.** Chat shows a PNG or JPEG as it is to an API model (`openai-chat`, `claude-sonnet`), reported with kind "native"; `QwenModel.chat` takes text and vector inputs only, so a local model gets an image only through an installed adapter that describes it as text, and is otherwise told why the image was not shown.
- **A live run does not survive a restart.** One run at a time. Finished runs are recorded under `<SENSE_HOME>/runs` and can be replayed (**Runs** in the interface, `GET /api/runs`, `POST /api/replay`), and `DELETE /api/jobs/<id>` cancels a live run or a replay, but the state of a run still going is held in memory and lost on restart. A run waiting for your approval of an install holds the one run slot until you answer or for up to 10 minutes, when it counts as declined; another tab or window follows such a run (`GET /api/jobs/latest`), with its approval card and Stop, when it opens or when it tries to start a run of its own.
- **One sense per modality and model.** A new adapter for a modality replaces the old one, and chat gives an adapter only the formats it was verified on, so an OBJ adapter installed after a GLB one leaves GLB files without a sense until the agent runs on one again.
- **Replaying a record from `socket_pilot/artifacts` does not bring its sense back.** A record carries its sense only when it was recorded or imported while that very sense was installed on the same machine, so the committed records replay the trace and cards only.
- **No `.m4a`.** `soundfile` cannot read AAC, so Voice Memos and QuickTime recordings are reported as not shown to the model; convert them to WAV first. Chat reads audio in `.wav .mp3 .flac .ogg .opus .aif .aiff .au`; it accepts any file, and reports one it cannot pass to the model.
- **One local model during a run.** While an agent run is going, chat with a different local model is refused with 409, so that two models are not loaded at once. Chat with the run's model still works.
- **Slow kernels.** `transformers` falls back to reference implementations for Qwen3.5's linear attention because `causal_conv1d` and `flash-linear-attention` are not installed; the timings above include this.
- **GTZAN repeats artists across tracks**, so a held-out track is weaker evidence than a held-out artist would be.
- **No second machine yet.** `scripts/setup_machine.py` is a guide and checker for one, but nobody has run it on a second machine.

## Models and datasets

- Base models (frozen): [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
- Audio and text encoder (frozen): [laion/clap-htsat-unfused](https://huggingface.co/laion/clap-htsat-unfused)
- Optional, through the OpenAI API: `gpt-6.1-terra` as the LLM planner, the need step of an autonomous run, and the `openai-chat` base model
- Optional, through the Anthropic API: Claude Sonnet 5.5 (`claude-sonnet-5-5`) as the `claude-sonnet` base model, and as the LLM planner and the need step when `SENSE_LLM_PROVIDER` is `anthropic`
- [GTZAN](https://huggingface.co/datasets/marsyas/gtzan): 999 music clips, 10 genres; split 60/20/20 per genre for fitting, dev and test
- [ESC-50](https://github.com/karolpiczak/ESC-50): 2,000 environmental sound clips, 50 classes; folds 1 to 3 fit, fold 4 dev, fold 5 test
- [AudioCaps](https://github.com/cdjkim/audiocaps): caption text only (12,000 captions from `train.csv`), for the caption text bridge
- At run time, an autonomous run may fetch a capped sample of a labelled audio dataset that research finds on the Hugging Face Hub; the trace and the claims ledger name its id, source page and licence
- Libraries: PyTorch, torchvision, Transformers, NumPy, SciPy, soundfile, Pillow, FastAPI, Uvicorn, python-multipart, the OpenAI Python SDK, the Anthropic Python SDK; optional: Databricks SDK, MLflow; tests: pytest, httpx, Playwright
- Services and APIs:
  - [Databricks](https://www.databricks.com) Free Edition (Unity Catalog, SQL warehouse, Volumes, MLflow) for the run records
  - the [OpenAI API](https://platform.openai.com/docs/api-reference/chat) (Chat Completions), for the LLM planner, the need step and `openai-chat`
  - the [Anthropic API](https://platform.claude.com/docs) (Messages, with structured outputs and the server-side refusal fallback beta), for `claude-sonnet` and, when `SENSE_LLM_PROVIDER` is `anthropic`, the LLM planner and the need step
  - the [Linkup](https://www.linkup.so) search API (`POST https://api.linkup.so/v1/search`, structured output), for the research and data steps
  - the Hugging Face Hub API (`https://huggingface.co/api`: model and dataset records) and the Hugging Face dataset viewer (`https://datasets-server.huggingface.co`: splits and rows of a dataset), for the research and data steps; all of this network access goes through `agent/research.py`

## Layout

| Path | Contents |
|---|---|
| `sense/` | contracts, settings, base models (local Qwen3.5, two API models), the CLAP specialist, datasets (built-in and `from_folder`), bridges, verification, the installed-sense registry, recorded runs, and agent-written adapters: the sandbox and verification (`adapters.py`) and the helpers adapters may import (`adapter_kit.py`) |
| `agent/` | rule planner and acceptance rules, LLM planner, the fit-verify-install loop (`loop.py`), background jobs, and the autonomous flow: `auto.py` (`python -m agent`), `detect.py`, `research.py`, `discover.py`, `data.py`, `claims.py`, `llm.py`, `llm_anthropic.py`; agent-written adapters: `synth.py` (design, write, repair, verify), `hub_probes.py` (labelled probe files from the web), `installer.py` (vetted package installs) |
| `server/app.py` | HTTP API and the static interface; its module docstring is the HTTP contract |
| `server/replay.py` | plays a recorded run back as a job |
| `web/` | the interface (no build step) |
| `scripts/demo_cli.py` | the agent from the command line, on a built-in set or a folder |
| `scripts/score_free_text.py` | free-text answers through an installed bridge |
| `scripts/fetch_demo_inputs.py` | rebuilds the 3D, video and molecule demo inputs from their public sources, checked against the recorded runs' inputs |
| `scripts/setup_machine.py` | second-machine guide: check, fetch the weights, copy or rebuild the embeddings |
| `eval/triage.py` | the forest-monitoring triage task |
| `integrations/databricks/` | the Databricks writer and its command line |
| `tests/` | fake-mode tests, including end to end through the API |
| `tests/web/` | browser smoke tests of the interface (Playwright, HTTP fixtures) |
| `socket_pilot/` | research scripts and results; `artifacts/` holds every recorded run behind the numbers here |
| `docs/reference.md` | this file: the detail behind the README |
| `docs/plan.md` | the original plan (14 November) |
| `docs/workstreams.md` | the four workloads the team lead approved on 14 November |
| `demo/reproduce.md` | how to repeat the 3D, video and molecule runs live |
| `docs/devpost.md`, `docs/pitch.md`, `demo/script.md` | the submission text, the pitch and the demo script |
| `design/` | interface mockup and board diagrams |
| `tools/board` | the team's message board on a pinned GitHub issue |

The team coordinates on the pinned board issue with `tools/board`; see [AGENTS.md](../AGENTS.md).
