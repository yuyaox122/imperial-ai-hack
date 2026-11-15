# Final demo video and Devpost gallery

- `sensorium_final.mp4`: 2:00 at 1920 by 1080, H.264, with a silent stereo track ready for dubbing. It follows the 2-minute brief:

| Time | Section |
|---|---|
| 0:00 to 0:15 | the problem, and who it's for |
| 0:15 to 1:30 | one live run of the whole task: a 3D model, from "file not received" to "This is a chair" |
| 1:30 to 1:50 | how it's built, with the bounty tools (Linkup, Figma, Databricks) |
| 1:50 to 2:00 | the coordinates refusal, the 33 formats it recognises, and how often it works |

  Sped-up parts are labelled on screen. The footage is the live session in `demo/fresh/`, recorded on 15 November from 10:48 to 10:59.
- `dub_script.md`: the voice-over, timed line by line to the video, with pronunciations and shorter fallbacks.
- `sensorium_final.srt`: the same lines as subtitles.

## Gallery for Devpost (`gallery/`, in this order)

| File | Caption to paste |
|---|---|
| `01_cover.png` | Sensorium: an agent that gives a model a sense it doesn't have, proves it works, and installs it, or refuses. |
| `02_the_problem.png` | Your model reads text and images; the world's data doesn't stop there. For teams who rely on one model but hold data it can't read. |
| `03_the_gap.png` | The gap: Claude Sonnet 5.5 can't read a 3D file, and the harness reports "file not received". |
| `04_agent_writes_code.png` | From one file and one sentence, the agent designs and writes the adapter, then runs it in a sandbox with no network. |
| `05_finds_its_own_data.png` | It finds its own test data: ModelNet40 meshes from the Hugging Face Hub, plus a set of categories it never saw. |
| `06_proof_before_install.png` | Proof before install: 100.0% on 50 held-out files, against another file's input (8%), no input (22%), the raw file (36%) and unseen categories (72%). |
| `07_the_new_sense.png` | The new sense: same model, same question, "This is a chair", read via the agent-written adapter. |
| `08_it_refuses.png` | It refuses when it should: Claude already reads coordinates (100% on the raw file), so nothing is installed. |
| `09_any_file_and_how_often.png` | 33 formats recognised from their bytes, anything else from scratch, and how often it worked in real runs. |
| `10_how_its_built.png` | How it's built: a frozen base model, sandboxed agent-written adapters, decisions by controls, plus Linkup, Figma and Databricks. |
| `11_figma_components.png` | Designed in Figma: every component has an installed and a refused state of equal weight. |
| `12_figma_tokens.png` | Designed in Figma: each variable carries its CSS name, so a design change is one line in the app. |
