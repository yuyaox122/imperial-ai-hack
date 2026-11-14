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
| `OPENAI_API_KEY` | reserved for an LLM planner and API models; nothing uses it yet | none |

Only one model process should run at a time on a 24 GB laptop. Loading a second local model through the harness releases the first.

## What works today

Measured on 14 November 2026 on an M4 Pro with 24 GB, Qwen3.5-0.8B in float16 on MPS, through `scripts/demo_cli.py` and the HTTP server. Each card scores 200 held-out five-option questions (chance 20%). The rule planner installs a vector bridge when it beats chance and swapped input by 25 points each (and, for open scope, scores at least 45% on classes left out of fitting), and a text bridge when it beats chance by 15 points. It prefers the vector bridge when the two are within 3 points.

| Run | Candidate | Accuracy | Swapped input | No input | Nearest-example lookup | Text bridge | Unseen classes | Verdict |
|---|---|---|---|---|---|---|---|---|
| GTZAN, closed | `vector:gtzan` | 79.5% | 12.5% | 22.0% | 84.5% | 48.5% | | installed |
| GTZAN, closed | `text:gtzan` | 48.5% | 20.0% | 22.0% | 84.5% | | | passed, not chosen |
| GTZAN, open | `vector:gtzan` | 79.5% | 12.5% | 22.0% | 84.5% | 35.5% | 12.5% | rejected |
| GTZAN, open | `text:captions` | 35.5% | 19.5% | 22.0% | 84.5% | | | installed |
| ESC-50, open | `vector:esc50` | 97.0% | 17.5% | 21.0% | 99.5% | 80.5% | 41.2% | rejected |
| ESC-50, open | `text:captions` | 80.5% | 17.5% | 21.0% | 99.5% | | | installed |

`text:gtzan` pastes the label CLAP matches best ('sounds like "rock"'); `text:captions` pastes the nearest of 12,000 AudioCaps captions. Nearest-example lookup answers the same five-option questions without the base model: it picks the offered label whose fitted clip is closest to the held-out clip in CLAP space. Unseen classes refits the vector bridge with a fifth of the labels left out and scores it on held-out clips of those labels: 40 clips of 2 genres for GTZAN, 80 clips of 10 sounds for ESC-50.

A whole agent run took 45.4 s (GTZAN closed), 48.6 s (GTZAN open) and 56.9 s (ESC-50 open). Fitting the vector bridge, with its search over blocks 0 to 4, took 13.0 to 17.9 s of that.

The research run (`socket_pilot/run_music.py`) reported 84.9% for the GTZAN vector bridge, against 79.5% here. The difference is the layer choice and the question draw, not a bug. The agent chose block 2 on 100 dev questions (blocks 0 to 4 scored 70.0, 74.0, 76.0, 54.0 and 52.0%), where the research run used block 0. On held-out clips, blocks 0, 1 and 2 average 80.8, 82.2 and 79.0% over three question seeds, and block 0 alone ranges from 78.0% to 85.0% across seeds.

Chat through the installed `vector:gtzan` bridge worked on held-out clips. The bridge mapped `reggae.00009.wav` to the label "reggae" (cosine 0.97), and the model's reply named reggae; it then described rhythm and instruments that the vector does not carry. Through the server, `classical.00082.wav` was answered as classical, and one message with an upload named `my song.wav` (a metal clip) and `jazz.00061.wav` was answered as metal and jazz. A `.txt` file in the same message was reported as not shown to the model, and so is an audio file that cannot be read, while the other files still reach the model.

A run names the sense it affects. When it installs a bridge for a model that already has an audio sense, the trace says which sense it replaced and with what score. When it installs nothing, the trace names the earlier sense that chat keeps using. Each chat took 2.6 to 4.1 s, including CLAP's first load. Every endpoint in `server/app.py` was exercised against the real model: models, datasets, fit, jobs (including 409 for a second run and 400 for a bad scope), senses, samples (the WAV is served), chat and the interface.

## What does not work yet

- **GTZAN open scope sits on the bar.** `text:captions` scored 35.5% against a 35.0% bar. Over question seeds 0 to 4 it scores 32.0 to 37.0% (mean 34.4%), so whether it installs depends on the draw. Verification uses seed 0, so runs are reproducible. Use ESC-50 to demonstrate open scope.
- **The ESC-50 rejection margin is small.** The vector bridge scored 41.2% on unseen sounds against the 45.0% bar.
- **Nearest-example lookup beats the vector bridge.** On the same questions, lookup in CLAP space scores 84.5% on GTZAN against the bridge's 79.5%, and 99.5% on ESC-50 against 97.0%. The planner does not use lookup as a bar, so the bridge is installed anyway. The bridge lets the model use what CLAP heard, but it does not yet answer better than a lookup over CLAP's embeddings.
- **The vector bridge passes a label, not a description.** The model treats the written vector like the word for the label: for the metal and jazz clips it said it was answering "Based on the names provided". Anything it says beyond the label is invented.
- **No LLM planner.** Only the rule planner exists; `agent/planner_llm.py` is not written, and `OPENAI_API_KEY` is unused.
- **No API models and no tool, image or native bridges.** Scenario 3 of the plan (Claude with a tool bridge) is not built.
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
- [GTZAN](https://huggingface.co/datasets/marsyas/gtzan): 999 music clips, 10 genres; split 60/20/20 per genre for fitting, dev and test
- [ESC-50](https://github.com/karolpiczak/ESC-50): 2,000 environmental sound clips, 50 classes; folds 1 to 3 fit, fold 4 dev, fold 5 test
- [AudioCaps](https://github.com/cdjkim/audiocaps): caption text only (12,000 captions from `train.csv`), for the caption text bridge
- Libraries: PyTorch, Transformers, NumPy, SciPy, soundfile, FastAPI, Uvicorn

## Layout

| Path | Contents |
|---|---|
| `sense/` | contracts, settings, base models, the CLAP specialist, datasets, bridges, verification, the installed-sense registry |
| `agent/` | planner, the fit-verify-install loop, background jobs |
| `server/app.py` | HTTP API and the static interface |
| `web/` | the interface (no build step) |
| `scripts/demo_cli.py` | the agent from the command line |
| `tests/` | fake-mode tests, including one end to end through the API |
| `socket_pilot/` | research scripts and results |

The team coordinates on the pinned board issue with `tools/board`; see [AGENTS.md](AGENTS.md).
