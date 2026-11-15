# Fresh demo footage, 15 November, 10:48 to 10:59

Screen-only footage for the final video, recorded at Alex's request on Michael's laptop. It is a live interface session in headless Chromium against the real server, from an empty `SENSE_HOME`, with Claude Sonnet 5.5 as the base model and `gpt-6.1-terra` as the agent's LLM. There is no narration, no captions and no webcam; sped-up parts carry a speed label. Every number below is in the two run records in `records/`.

## What happened

| Run | Goal | Data the agent found | Card | Verdict | Time |
|---|---|---|---|---|---|
| 3D: ModelNet40 `chair_0946` as `mystery.glb` | "Tell me what object this 3D model is" | `naderalfares/ModelNet40_Auto_aligned` (8 objects, 160 held-out files); unseen set `codepawl/scanned-objects-glb` (8 categories, 122 files) | 100.0% [92.9, 100] on 50 questions; swapped input 8%, no input 22%, raw file 36%, unseen categories 72% (chance 20%) | **installed** `adapter:3d-glb` | 376 s |
| Coordinates: `51.3608, 7.4717` (Hagen, Germany) as `mystery.txt` | "Which country is this?" | `bamwor/world-cities` (8 countries, 160 files); unseen set `Lilambd/world-cities` (8 other countries, 151 files) | 100.0% [92.9, 100]; swapped input 6%, no input 18%, **raw file 100%**, unseen categories 100% | **refused**: "the model already reads this file as it is" (the adapter must add 10 points over the raw file) | 140 s |

- **Before chat (3D):** the reply is marked **File not received**: "Claude Sonnet 5.5 does not read 3D model files as they are."
- **After chat (3D):** "This is a **chair**, shown from four different viewpoints…", tagged **Read via adapter** (`adapter:3d-glb · 4 images from the agent-written adapter`).
- **Before chat (coordinates):** the coordinates are text, so Claude answers from the raw numbers. That is why the raw-file control is 100% and the adapter is refused.
- **Agent code:** the adapter was written and ran on the first attempt (3D in 1.0 s). Its code is shown expanded in `raw_04`.
- **Web search:** the dataset step searches the Hugging Face Hub and the web (Linkup) side by side. See the start of `raw_05`.

## Files

- `rough_cut.mp4`: 1:47, 1440 by 900, screen only. This is the cut list:

| Time in cut | Source | Speed |
|---|---|---|
| 0:00 to 0:08 | app, then the 3D before chat | 1x |
| 0:08 to 0:13 | New sense, goal, file, Run agent | 1x |
| 0:13 to 0:19 | detect, design, build | 3x, labelled |
| 0:19 to 0:24 | the adapter's code, expanded | 1x |
| 0:24 to 0:27 | the dataset search starts (Hub and web) | 1x |
| 0:27 to 0:42 | the dataset search and verification | 24x, labelled |
| 0:42 to 0:52 | report card: 100.0% and its controls, installed | 1x |
| 0:52 to 1:01 | after chat: "This is a chair", Read via adapter | 1x |
| 1:01 to 1:11 | coordinates: before chat, New sense | 1x |
| 1:11 to 1:17 | coordinates run | 24x, labelled |
| 1:17 to 1:25 | coordinates report card: refused, raw file 100% | 1x |
| 1:25 to 1:37 | Figma: Henad tokens (55:75), then component variants (55:749) | exported frames, labelled |
| 1:37 to 1:43 | the same states in the app: installed card, then refused card | 1x |
| 1:43 to 1:47 | end card: Sensorium and the repository URL | |

- `raw/`: every beat at 1x with no overlays (`raw_01` to `raw_12`, about 10 minutes in all), for re-cutting.
- `figma/`: the two frames exported from the team file through the Figma API (`variables_55-75.png`, `components_55-749.png`), and the 6-second pans used in the cut. A capture of the live Figma interface still needs someone with the file open.
- `records/`: the two run records (request, timed trace, cards), with local paths scrubbed.

## Notes for the narration

- The coordinates refusal came from a raw-file score of 100%, not 98% as in the earlier recorded run. Say "Claude already reads coordinates perfectly, so the adapter adds nothing."
- The 3D object is a chair, and the before chat shows "File not received".
- Databricks has no footage here. It needs someone logged in to the workspace to record the `runs` and `report_cards` tables.
