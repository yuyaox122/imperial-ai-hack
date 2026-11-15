# Reproduce the three adapter runs (3D, video, molecule)

This runbook repeats the recorded runs in which the agent gave Claude Sonnet 5.5 a sense it lacks: a 3D model, a video clip and a molecule. Each run starts from one file and one sentence, with no examples. The agent writes an adapter, finds labelled test data on the Hugging Face Hub, scores the model against controls, and installs the sense or refuses it.

Every number below comes from a record in `socket_pilot/artifacts/`. A live run can give a different number: the agent rewrites the adapter each time. Read the number off the screen, not from this page.

## 1. One-time setup

1. Install the environment: `uv sync`.
2. Put the API keys in `.env` (or `.env.local`) at the repository root, one `KEY=value` per line:
   - `ANTHROPIC_API_KEY`: the base model, Claude Sonnet 5.5 (`claude-sonnet` in the interface).
   - `OPENAI_API_KEY`: the agent's own LLM (`gpt-6.1-terra`), which designs and writes the adapter and judges the datasets it finds. This is what the recorded runs used. To use Claude for this too, set `SENSE_LLM_PROVIDER=anthropic` and skip the OpenAI key.
   - `LINKUP_API_KEY` (optional): adds web search to the Hub search for test data. All three recorded runs found their data on the Hub.
3. Install ffmpeg (`brew install ffmpeg`). The video adapter uses it, and so does step 2 below.
4. No local model weights are needed: Claude runs through the API.

## 2. Get the three input files

```bash
.venv/bin/python scripts/fetch_demo_inputs.py
```

This downloads each input from its public source at the revision the runs used and writes it to `~/.cache/sense-harness/demo-inputs/` as `mystery.<ext>`, so the name gives nothing away:

| Folder | File | What it is | Used by |
|---|---|---|---|
| `3d-guitar/` | `mystery.glb` | ModelNet40 `guitar_0235` (test split) | `run_auto_claude-sonnet_3d.json` |
| `3d-chair/` | `mystery.glb` | ModelNet40 `chair_0946` (test split) | `run_auto_claude-sonnet_3d-repeat.json` |
| `video/` | `mystery.mp4` | UCF101 horse riding, `v_HorseRiding_g24_c06`, first 10 s | `run_auto_claude-sonnet_video.json` and `-repeat` |
| `molecule/` | `mystery.smi` | vanillin, `COc1cc(C=O)ccc1O` | `run_auto_claude-sonnet_molecule.json` and `-repeat` |

The script checks each file against the input the recorded run used, and says whether it is identical. On the machine that recorded the runs all four were. Another ffmpeg version can change the video's bytes but not the clip. Use `--out <folder>` to write somewhere else.

## 3. Start the server on an empty sense folder

```bash
SENSE_HOME="$(mktemp -d)" .venv/bin/python -m uvicorn server.app:app --port 8000
```

Then open http://127.0.0.1:8000. An empty `SENSE_HOME` means Claude starts with no 3D, video or molecule sense, and no earlier answers are reused. The recorded reruns were made this way.

**For a demo:** rehearse in one folder (for example `SENSE_HOME=~/sensorium-rehearsal`), then start the real demo on a fresh, empty one. Keep the rehearsal folder: its runs are recorded with their senses, so if the live run fails you can restart the server on it and replay (see section 6).

Run one goal at a time. The server takes one run at a time and answers 409 to a second.

## 4. Run each goal in the interface

The same steps for all three:

1. **Pick the model.** In **Model** at the top left, choose **Claude Sonnet 5.5**.
2. **Show the gap (optional).** In the chat, attach the file (paperclip button, or drop it on the chat) and ask the chat question from the table below. For the mesh and the video the reply is marked **File not received**: Claude does not read these files. The molecule file is text, so Claude gets the raw SMILES and answers from that alone, which is what the raw-file control measures (46%).
3. **Start the run.** Press **New sense** (top right). Leave **Let the agent find them** selected, type the goal, drop the file, and press **Run agent**.
4. **Watch the trace.** The **Run** tab shows each step: detect the file, design an adapter, write and run it in the sandbox, find labelled examples (and a second set of other categories), verify, then install or refuse. No package approval is expected for these three: the libraries they use (trimesh, RDKit, ffmpeg) are already in the toolbox. If an approval card appears, it lists the installer's checks, and you choose **Approve** or **Decline**.
5. **Read the card.** The **Report** tab shows the score with its 95% interval and every control as a bar against the chance tick.
6. **Ask again.** Attach the same file and ask the same question. The reply now says **Read via adapter**, and the note under it names the adapter (`adapter:3d-glb`, `adapter:video-mp4` or `adapter:molecule-smiles`).

| Run | File | Goal (type this in New sense) | Chat question | Time | Recorded results (installed in every run) |
|---|---|---|---|---|---|
| 3D model | `3d-guitar/mystery.glb` or `3d-chair/mystery.glb` | `Tell me what object this 3D model is` | `What object is this?` | about 6 min | guitar: 100.0% [92.9, 100], swapped 12%, no input 20%, raw file 42%, unseen 64%. Chair: 100.0%, swapped 8%, no input 18%, raw file 34%, unseen 70%. Afterwards Claude named the chair. |
| Video | `video/mystery.mp4` | `What action is happening in this video?` | `What action is happening in this video?` | 3 to 4 min | 78.0% [64.8, 87.3], then 86.0% [73.8, 93.1] on the rerun; swapped 8%, no input 20 to 22%, unseen actions 92 to 96% |
| Molecule | `molecule/mystery.smi` | `What does this molecule smell like?` | `What does this molecule smell like?` | 3 to 4 min | 58.0% [44.2, 70.6] in both runs, against 46.0% for Claude reading the raw SMILES: 12 points over, where the bar is 10. Unseen categories not measured. |

Chance is 20% everywhere: five options per question, 50 held-out questions per card.

## 5. What can go differently live

- **The numbers move.** The agent writes a new adapter each run. The 3D and video results held in both recorded runs.
- **The molecule can be refused.** It clears the raw-file bar by 12 points where 10 are needed, and an earlier, unrecorded run missed by 2. A refusal is also a good demo: the card shows why.
- **Hugging Face outages.** The dataset viewer sometimes answers HTTP 500 or 502. The agent then tries other candidates. For the second set of categories it may report "Unseen categories not measured" (the molecule runs did), and if no labelled data works at all, it stops and asks for an examples folder.
- **Cost.** Each run makes a few hundred paid API requests, mostly Claude answering 50 questions under each control.

## 6. Backups

- **Replay a rehearsal.** Start the server on the rehearsal folder, open **Runs**, choose 1x, 2x or 4x, and press **Replay**. A run recorded by that server carries its sense, so chat with the file works after the replay.
- **Replay a committed record.** Import one into the current `SENSE_HOME`, then replay it from **Runs**:
  `.venv/bin/python -c "from sense import runs; print(runs.import_record('socket_pilot/artifacts/run_auto_claude-sonnet_3d.json'))"`.
  An imported record replays the trace and the card only: it does not bring the sense back, so do not chat about the file after it.
- **The video.** `demo/sensorium_demo.mp4` (1:56) shows the 3D run with the before and after chats, plus the coordinates and MIDI refusals.

## 7. Without the interface

From the terminal, with the same keys. Install requests are asked on the terminal, and No is the default:

```bash
export SENSE_HOME="$(mktemp -d)"
IN=~/.cache/sense-harness/demo-inputs
.venv/bin/python -m agent --model claude-sonnet --file "$IN/3d-guitar/mystery.glb" \
    --goal "Tell me what object this 3D model is" --save 3d.json
.venv/bin/python -m agent --model claude-sonnet --file "$IN/video/mystery.mp4" \
    --goal "What action is happening in this video?" --save video.json
.venv/bin/python -m agent --model claude-sonnet --file "$IN/molecule/mystery.smi" \
    --goal "What does this molecule smell like?" --save molecule.json
```

Or over HTTP, against a running server: `POST /api/auto` with the form fields `model`, `goal` and `file`. It returns a `job_id`; poll `GET /api/jobs/<id>` until `status` is no longer `running`:

```bash
curl -s -F model=claude-sonnet -F goal="What action is happening in this video?" \
     -F file=@"$HOME/.cache/sense-harness/demo-inputs/video/mystery.mp4" http://127.0.0.1:8000/api/auto
```
