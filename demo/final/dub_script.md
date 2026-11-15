# Sensorium demo: script for the voice-over

The video `sensorium_final.mp4` lasts 2:00 at 1920 by 1080 and has a silent stereo track, ready for dubbing. Each line below fits its window at about 2.5 words a second, with a short pause at each end. Read the numbers exactly as written: they are the numbers on screen. `sensorium_final.srt` holds the same lines with the same timings, for the dubbing tool or for captions.

**Pronunciation:** Sensorium is "sen-SOR-ee-um". Linkup is "LINK-up". ModelNet is "MOD-el-net". Claude is "clawd".

**Delivery:** calm and confident, product-demo pace. Leave a beat after "file not received" and before "one hundred percent".

| # | Time | On screen | Voice-over | Words |
|---|---|---|---|---|
| 1 | 0:00 to 0:03 | Title card | "This is Sensorium." | 3 |
| 2 | 0:03 to 0:15 | The problem; who it's for | "Your model reads text and images, but real work runs on 3D scans, video, molecules and GPS tracks. Sensorium gives the model you already use a new sense, with proof first." | 31 |
| 3 | 0:15 to 0:23 | Claude gets a 3D file: "file not received" | "Here's Claude with a 3D file. It never reaches the model. File not received." | 14 |
| 4 | 0:23 to 0:28 | New sense: one file, one goal | "So we ask the agent. One file, one sentence." | 9 |
| 5 | 0:28 to 0:37 | Detect, design, build (2x) | "It detects a 3D model, designs an adapter that renders it, and writes the code in a sandbox with no network." | 21 |
| 6 | 0:37 to 0:43 | The adapter's code, expanded | "This is the code it wrote. It ran first time." | 10 |
| 7 | 0:43 to 0:46 | The dataset search starts | "Then it goes looking for test data," | 7 |
| 8 | 0:46 to 1:00 | Data search (16x) | "on the Hugging Face Hub and across the web with Linkup. It finds labelled ModelNet meshes, and a second set of objects it was never built around." | 28 |
| 9 | 1:00 to 1:08 | Verification (16x) | "Now Claude answers fifty held-out questions through the adapter, and again under controls." | 13 |
| 10 | 1:08 to 1:20 | The report card: installed | "One hundred percent. With another file's input: eight. Reading the raw file: thirty-six. Categories it never saw: seventy-two. Every bar cleared, so it installs." | 24 |
| 11 | 1:20 to 1:30 | Same question: "This is a chair" | "Same model, same question. Now it sees a chair, and says which sense carried the file. The model's weights never changed." | 22 |
| 12 | 1:30 to 1:38 | How it's built | "Under the hood: a frozen model, sandboxed adapters, and fixed rules that decide. Linkup finds the data. Databricks records the runs." | 21 |
| 13 | 1:38 to 1:50 | Figma: design tokens, then components | "We designed it in Figma. Every variable maps to the app's CSS, and every component has an installed and a refused state, with equal weight." | 25 |
| 14 | 1:50 to 1:54 | Coordinates: refused | "It also says no. Claude already reads coordinates." | 8 |
| 15 | 1:54 to 2:00 | Any file; how often it works | "Thirty-three formats, anything else from scratch. 3D: three for three. Sensorium." | 12 |

**Total:** about 250 words in 2:00.

## If a line runs long

- **Line 2:** "Your model reads text and images. Real work doesn't stop there. Sensorium gives your model a new sense, with proof first."
- **Line 8:** "on the Hugging Face Hub and the web with Linkup, and finds labelled meshes plus a set it never saw."
- **Line 10:** "One hundred percent, against eight with another file and thirty-six on the raw file. It installs."
- **Line 13:** "Designed in Figma: variables map to the app's CSS, and every state has a component."

## Facts behind the lines (all in `demo/fresh/records/` and `socket_pilot/artifacts/`)

- **The live 3D run:** Claude Sonnet 5.5, starting with no senses installed. `adapter:3d-glb` scored 100.0% on 50 held-out ModelNet40 meshes. Controls: another file's input 8%, no input 22%, raw file 36%, unseen categories 72% (chance 20%). It was installed. In the chat afterwards Claude said "This is a chair", read via the adapter.
- **The coordinates run:** the adapter scored 100%, but Claude reading the raw coordinates also scored 100%, so it was refused.
- **"Three for three":** the 3D goal installed in all three recorded runs (two with a chair, one with a guitar).
- **"Thirty-three formats":** the formats `agent/detect.py` recognises from a file's bytes. Any other file goes to an adapter designed from scratch, which is how the MIDI blind test ran (it was refused at 32%).
- **Databricks:** `integrations/databricks` writes runs, report cards and trace events to Delta tables and MLflow. This video shows no Databricks screen. If that matters for the prize, the team should record a clip from the workspace.
