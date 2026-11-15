# Demo video (2 minutes or less)

A shot list for the Devpost video: one live run of the agent on a 3D model, then two refusals. Read every number off the screen. The live run's card can differ from the recorded one (`socket_pilot/artifacts/run_auto_claude-sonnet_3d.json`: 100.0%, swapped 12%, no input 20%, raw file 42%, unseen 64%), so the voice-over must say what the card on screen says.

## Before recording

| Check | How |
|---|---|
| A clean sense folder, so the model starts without a 3D sense | start the server with `SENSE_HOME` pointing at an empty folder |
| Keys for a real run | `ANTHROPIC_API_KEY` (the base model) and `OPENAI_API_KEY` (the adapter writer) in `.env`; `LINKUP_API_KEY` for web search |
| The server | `SENSE_HOME=<empty folder> .venv/bin/python -m uvicorn server.app:app --port 8000`, then open http://127.0.0.1:8000 at 1440 by 900 |
| The input | a ModelNet40 test mesh as a `.glb` file named `mystery.glb`, so the name gives nothing away (our take used a chair, `chair_0946`) |
| The refusals | import the two records so **Runs** can replay them: `.venv/bin/python -c "from sense import runs; [print(runs.import_record(p)) for p in ('socket_pilot/artifacts/run_auto_claude-sonnet_coordinates.json', 'socket_pilot/artifacts/run_auto_claude-sonnet_midi-examples.json')]"` (with the same `SENSE_HOME`) |
| Time | the live run takes about 6 minutes; record all of it and speed up the waiting in the edit, with the speed shown on screen |

## Shots

| Time | Shot | On screen | Voice-over |
|---|---|---|---|
| 0:00 to 0:08 | Title card | The banner and the tagline | "Language models read text, and some read images. Give one a 3D model and it is stuck." |
| 0:08 to 0:22 | Chat, before | Pick **Claude Sonnet 5.5**. Attach `mystery.glb` and ask "What object is this?". The reply says the file was not shown to the model. | "Claude cannot read this file. So we ask an agent to give it the sense." |
| 0:22 to 0:30 | Start the run | **New sense**, drop `mystery.glb`, goal "Tell me what object this 3D model is", **Run agent** | "One file, one sentence. No training data." |
| 0:30 to 1:05 | The run, sped up (show "8x") | The trace: detect (3D), design (an adapter that renders views), build (runs in the sandbox), find labelled examples (a ModelNet40 set on the Hugging Face Hub, then a second set with other categories), verify | "It works out the file is a mesh and writes code that renders it. The code runs in a sandbox with no network. Then it goes looking for labelled 3D models on the web, and for a second set with categories it never saw." |
| 1:05 to 1:25 | The report card, close | Accuracy with its interval; swapped input, no input, raw file, unseen categories; the verdict | "Here is the test. With another file's input, it drops to [swapped]. Reading the raw file, Claude gets [raw file]. On categories the adapter was never built around, [unseen]. It clears every bar, so the agent installs it." |
| 1:25 to 1:40 | Chat, after | Ask "What object is this?" with `mystery.glb` again. The reply names the object (a chair, in our take) and shows `Read via adapter:3d-glb`. | "Same model, same question. Now it sees a chair, and the reply says which sense carried the file." |
| 1:40 to 1:55 | Two refusals | **Runs**, replay the coordinates run at 4x and stop on its card ("the model already reads this file as it is"), then the MIDI card (32.0%, below the bar) | "It also says no. Claude already reads coordinates, so an adapter adds nothing. For MIDI, a format our code never mentions, the adapter it wrote scored 32 percent, and the agent installed nothing." |
| 1:55 to 2:00 | End card | Sensorium, the repository URL | "Sensorium. A new sense comes with a report card, or not at all." |

## If the live run fails

Replay a server-recorded run from rehearsal (**Runs**, then Replay, labelled as a recording). Or cut to the recorded numbers in the README table and say they are from a recorded run.
