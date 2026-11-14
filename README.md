# Sense harness

An agent that gives a frozen base model (local Qwen3.5, which reads text and images) an input sense it lacks: hearing, learned from a labelled set of audio clips. The agent builds candidate bridges from a frozen audio encoder (CLAP) into the model, scores each one on held-out clips against controls, and installs the best bridge that passes or installs nothing and says why, with the numbers. You then chat with the model about an audio file, and each reply shows which bridge carried the file to the model.

The idea, the research behind it and the open decisions are in [docs/plan.md](docs/plan.md). The research code and its results are in [socket_pilot/](socket_pilot/README.md).

## Run it

`uv sync` creates `.venv` with every dependency, including pytest and httpx for the tests.

### Fake mode (no weights, no data files)

`SENSE_FAKE=1` swaps in a deterministic fake model, fake encoder and small synthetic datasets, so the whole system runs in under a second without loading anything.

```bash
SENSE_FAKE=1 .venv/bin/python -m pytest -q
SENSE_FAKE=1 .venv/bin/python scripts/demo_cli.py --model fake --dataset gtzan --scope closed
SENSE_FAKE=1 .venv/bin/python -m uvicorn server.app:app --port 8000   # then open http://127.0.0.1:8000
```

In fake mode the only model is `fake`, and the sample clips are one-second tones written to the system temporary directory.

### Real mode

Real mode needs four things on the machine:

1. **Qwen3.5 weights**: `Qwen/Qwen3.5-0.8B` (and `Qwen/Qwen3.5-2B` if you want it) in a Hugging Face cache. The harness looks for a snapshot directory holding `config.json` and `*.safetensors`: first at `SENSE_QWEN_0P8B` / `SENSE_QWEN_2B` if they are paths, then under `SENSE_HF_CACHE`, then in the default cache. It loads from that directory and never downloads.
2. **CLAP** (`laion/clap-htsat-unfused`): downloaded once with a network connection. It is used only in chat, to embed a new file; fitting and verification use cached embeddings. Once cached it loads offline, including from the `refs/pr/3` snapshot where that repository keeps its safetensors weights.
3. **Cached embeddings** in `socket_pilot/artifacts/`: `gtzan_clap.npz`, `esc50_clap.npz` and `audiocaps_text_clap.npz`. They are not in git. Write them with `socket_pilot/clap_embed_gtzan.py`, `clap_embed.py` and `clap_text.py` (see [socket_pilot/README.md](socket_pilot/README.md)), or set `SENSE_DATA_DIR` to a directory holding copies.
4. **Raw audio** (optional, for the sample clips in the interface and for `--ask`): set `SENSE_SAMPLE_DIRS` to the GTZAN `genres` directory (files as `<dir>/<genre>/<file>.wav`) and the ESC-50 `audio` directory, separated by `:`. A set offers samples only when every one of its files is found.

```bash
# closed scope: fit, verify, install, then ask about one held-out clip
.venv/bin/python scripts/demo_cli.py --model qwen3.5-0.8b --dataset gtzan --scope closed \
    --ask <path to a GTZAN clip> "Which genre of music is this?"

# open scope ("any sound"): the vector bridge is rejected, the caption text bridge is installed
.venv/bin/python scripts/demo_cli.py --model qwen3.5-0.8b --dataset esc50 --scope open

# the interface
.venv/bin/python -m uvicorn server.app:app --port 8000   # then open http://127.0.0.1:8000
```

The held-out clips the interface offers can be listed with `.venv/bin/python -c "from sense.datasets import sample_files; print(sample_files('gtzan'))"`. They are never used for fitting.

Settings can also go in `.env` or `.env.local` (`KEY=VALUE` lines) in the repository root; variables already set in the environment win.

| Variable | Meaning | Default |
|---|---|---|
| `SENSE_FAKE` | `1` for fake models, encoder and data | off |
| `SENSE_HOME` | where installed senses are stored | `<repo>/.sense` |
| `SENSE_DATA_DIR` | directory of the cached `.npz` embeddings | `<repo>/socket_pilot/artifacts` |
| `SENSE_HF_CACHE` | extra Hugging Face cache holding the Qwen3.5 weights | none |
| `SENSE_QWEN_0P8B`, `SENSE_QWEN_2B` | Hugging Face id or local directory | `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B` |
| `SENSE_CLAP` | Hugging Face id or local directory | `laion/clap-htsat-unfused` |
| `SENSE_SAMPLE_DIRS` | `:`-separated directories of raw audio | none |
| `OPENAI_API_KEY` | enables the LLM planner (`SENSE_LLM_MODEL`, default `gpt-5.4-mini`) and the `openai-chat` API base model; never used in fake mode | none |

Only one model process should run at a time on a 24 GB laptop. Loading a second local model through the harness releases the first.

## What works today

Measured on 14 November 2026 on an M4 Pro with 24 GB, Qwen3.5-0.8B in float16 on MPS, through `scripts/demo_cli.py` and the HTTP server. Each card scores 200 held-out five-option questions (chance 20%) and carries a 95% Wilson interval for its accuracy.

The acceptance rules are in one place, `RulePlanner` in `agent/planner.py`. A vector bridge passes when it beats chance and swapped input by 25 points each; for open scope it must also score at least 45% on classes left out of fitting and stay within 20 points of its accuracy on fitted classes. A text bridge passes when it beats chance by 15 points and swapped input by 10. The vector bridge is preferred when the two are within 3 points.

With `OPENAI_API_KEY` set, the LLM planner (`agent/planner_llm.py`) proposes which bridges to try and which to install. The loop verifies every bridge the model can take whatever the planner proposes, and the planner may only install a bridge that passes the rules above; otherwise the rules decide.

| Run | Candidate | Accuracy | Swapped input | No input | Nearest-example lookup | Specialist alone | Text bridge | Unseen classes | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| GTZAN, closed | `vector:gtzan` | 81.0% | 12.5% | 22.0% | 84.5% | 52.5% | 48.5% | | installed |
| GTZAN, closed | `text:gtzan` | 48.5% | 20.0% | 22.0% | 84.5% | 52.5% | | | passed, not chosen |
| ESC-50, open | `vector:esc50` | 96.5% | 18.0% | 21.0% | 99.5% | 99.0% | 80.5% | 51.2% | rejected |
| ESC-50, open | `text:captions` | 80.5% | 17.5% | 21.0% | 99.5% | 99.0% | | | installed |

`text:gtzan` pastes the label CLAP matches best ('sounds like "rock"'); `text:captions` pastes the nearest of 12,000 AudioCaps captions. Nearest-example lookup answers the same five-option questions without the base model: it picks the offered label whose fitted clip is closest to the held-out clip in CLAP space. Specialist alone is CLAP with no base model and no fitted examples: it picks the offered label whose text is closest to the clip. Unseen classes refits the vector bridge with a fifth of the labels left out and scores it on held-out clips of those labels: 40 clips of 2 genres for GTZAN, 80 clips of 10 sounds for ESC-50.

A whole agent run took 57.7 s (GTZAN closed) and 64.0 s (ESC-50 open). The vector bridge's layer is chosen among blocks 0 to 4 on 200 dev questions.

The research run (`socket_pilot/run_music.py`) reported 84.9% for the GTZAN vector bridge, against 81.0% here. The difference is the layer choice and the question draw, not a bug: on held-out clips, blocks 0, 1 and 2 average 80.8, 82.2 and 79.0% over three question seeds.

In the GTZAN run the LLM planner chose the install; in the ESC-50 run its choice failed the rules and the rule planner decided. Its proposed plan differed between two otherwise identical ESC-50 runs (vector only, then text only), which is why the loop verifies every available bridge.

Chat through the installed `vector:gtzan` bridge worked on held-out clips. The bridge mapped `reggae.00009.wav` to the label "reggae" (cosine 0.97), and the model's reply named reggae; it then described rhythm and instruments that the vector does not carry. Through the server, `classical.00082.wav` was answered as classical, and one message with an upload named `my song.wav` (a metal clip) and `jazz.00061.wav` was answered as metal and jazz. A `.txt` file in the same message was reported as not shown to the model, and so is an audio file that cannot be read, while the other files still reach the model.

A run names the sense it affects. When it installs a bridge for a model that already has an audio sense, the trace says which sense it replaced and with what score. When it installs nothing, the trace names the earlier sense that chat keeps using. Each chat took 2.6 to 4.1 s, including CLAP's first load. Every endpoint in `server/app.py` was exercised against the real model: models, datasets, fit, jobs (including 409 for a second run and 400 for a bad scope), senses, samples (the WAV is served), chat and the interface.

## What does not work yet

- **GTZAN open scope sits on the bar.** In an earlier run `text:captions` scored 35.5% against a 35.0% bar, and 32.0 to 37.0% over question seeds 0 to 4, so whether it installs depends on the draw. Use ESC-50 to demonstrate open scope.
- **The unseen-classes control is noisy.** For ESC-50 it was 41.2% in one run and 51.2% in another (80 clips of 10 left-out sounds; the chosen layer differed). Both are far below the 96.5% on fitted sounds, which is what the within-20-points rule checks.
- **Nearest-example lookup beats the vector bridge.** On the same questions, lookup in CLAP space scores 84.5% on GTZAN against the bridge's 81.0%, and 99.5% on ESC-50 against 96.5%. The planner does not use lookup as a bar, so the bridge is installed anyway.
- **On ESC-50 the specialist alone is as good as the bridge.** CLAP matching label text scores 99.0% with no base model. On GTZAN it scores 52.5%, so there the fitted bridge adds 28.5 points over the specialist's own text matching.
- **The vector bridge passes a label, not a description.** The model treats the written vector like the word for the label: for the metal and jazz clips it said it was answering "Based on the names provided". Anything it says beyond the label is invented.
- **The LLM planner's plan is unreliable.** See above; its install choice is checked against the rules. One planning call and one decision call are made per run.
- **The API base model has not been through a full agent run.** `openai-chat` (`sense/models_api.py`) is registered when a key is set and takes text bridges only; its tests are offline. Tool, image and native bridges are not built, and Scenario 3 of the plan is not demonstrated.
- **Qwen3.5-2B is listed but untested in this harness.** It loads through the same code as the 0.8B; only the research scripts have run it.
- **Chat has no history.** Each message stands alone, so a follow-up question needs the file attached again. Image inputs are not wired into `QwenModel.chat`.
- **Jobs live in memory.** One run at a time, lost on restart; a run started in another browser tab shows up only as the 409 message.
- **No `.m4a`.** `soundfile` cannot read AAC, so Voice Memos and QuickTime recordings are reported as not shown to the model; convert them to WAV first. Chat takes `.wav .mp3 .flac .ogg .opus .aif .aiff .au`.
- **One local model during a run.** While an agent run is going, chat with a different local model is refused with 409, so that two models are not loaded at once. Chat with the run's model still works.
- **Slow kernels.** `transformers` falls back to reference implementations for Qwen3.5's linear attention because `causal_conv1d` and `flash-linear-attention` are not installed; the timings above include this.
- **GTZAN repeats artists across tracks**, so a held-out track is weaker evidence than a held-out artist would be.

## Models and datasets

- Base models (frozen): [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
- Audio and text encoder (frozen): [laion/clap-htsat-unfused](https://huggingface.co/laion/clap-htsat-unfused)
- Optional, through the OpenAI API: `gpt-5.4-mini` as the LLM planner and as the `openai-chat` base model
- [GTZAN](https://huggingface.co/datasets/marsyas/gtzan): 999 music clips, 10 genres; split 60/20/20 per genre for fitting, dev and test
- [ESC-50](https://github.com/karolpiczak/ESC-50): 2,000 environmental sound clips, 50 classes; folds 1 to 3 fit, fold 4 dev, fold 5 test
- [AudioCaps](https://github.com/cdjkim/audiocaps): caption text only (12,000 captions from `train.csv`), for the caption text bridge
- Libraries: PyTorch, Transformers, NumPy, SciPy, soundfile, FastAPI, Uvicorn

## Layout

| Path | Contents |
|---|---|
| `sense/` | contracts, settings, base models (local Qwen3.5, an API model), the CLAP specialist, datasets, bridges, verification, the installed-sense registry |
| `agent/` | rule planner and acceptance rules, LLM planner, the fit-verify-install loop, background jobs |
| `server/app.py` | HTTP API and the static interface |
| `web/` | the interface (no build step) |
| `scripts/demo_cli.py` | the agent from the command line |
| `tests/` | fake-mode tests, including one end to end through the API |
| `socket_pilot/` | research scripts and results |

The team coordinates on the pinned board issue with `tools/board`; see [AGENTS.md](AGENTS.md).
