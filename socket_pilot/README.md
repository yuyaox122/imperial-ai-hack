# Audio socket pilot

A go/no-go test for the hot-plug modality idea (issue #4): can a frozen text-and-image model be given a
new sense with no training of the model itself?

**Verdict so far:** the mechanism works, but in this form it transfers a label, and pasting retrieved text
into the prompt does the open-vocabulary job better. See [What it shows](#what-it-shows).

## Method

After [A Training-Free Socket for DNA in Frozen Language Models](https://arxiv.org/abs/2611.05401), which
tested DNA and protein only:

1. **Encoder.** A frozen CLAP audio encoder turns each clip into one 512-dimensional vector.
2. **Target.** For text naming the sound, take the host model's own mean-pooled hidden state after an
   early block.
3. **Plug.** One ridge regression maps the audio vector to that target, followed by a norm match. It is
   closed-form: no gradient passes through the host.
4. **Socket.** The prompt contains four placeholder tokens. After the chosen block, their hidden states
   are replaced by the plug's output and the forward pass continues.

Everything is frozen. The host is Qwen3.5 (0.8B, 2B), which already takes text, images and video.

## Test

ESC-50: 2,000 five-second clips, 50 sound classes, five folds. Folds 1 to 3 fit the plug, fold 4 picks the
socket layer, and fold 5 (400 clips) is scored once. Each item is a five-option multiple-choice question,
scored by the next-token logit of the option letters, so chance is 20%.

- **Identify:** which of five sounds is in the clip.
- **Category:** what kind of thing makes the sound. The options never name the sound, so the host has to
  use what it knows about it.

Controls: audio swapped for a clip of another class, a zero write, no write, the nearest fitting clip's
target (lookup), the host's own text representation of the true class (ceiling), and CLAP's own zero-shot
choice among the five options with no language model.

## Results

Accuracy in percent on the held-out fold, with 95% bootstrap intervals.

### Paired plug (all 50 classes seen during fitting)

| Condition | Qwen3.5-0.8B identify | Qwen3.5-0.8B category | Qwen3.5-2B identify | Qwen3.5-2B category |
|---|---|---|---|---|
| Plug | 99.0 [98.0, 99.8] | 60.2 [55.5, 65.0] | 96.8 [94.8, 98.5] | 66.8 [62.5, 71.2] |
| Swapped audio | 19.2 [15.5, 23.2] | 21.2 [17.5, 25.5] | 19.0 [15.2, 23.0] | 20.5 [16.5, 24.8] |
| No write | 23.5 [19.5, 27.8] | 21.5 [17.5, 25.8] | 23.2 [19.2, 27.8] | 21.2 [17.2, 25.2] |
| Nearest fitting clip (lookup) | 97.8 [96.0, 99.0] | 61.5 [56.8, 66.2] | 96.0 [93.8, 98.0] | 66.5 [61.8, 71.0] |
| Own text (ceiling) | 99.8 [99.2, 100.0] | 60.8 [56.0, 65.5] | 99.5 [98.8, 100.0] | 66.8 [62.3, 71.2] |
| CLAP zero-shot, no language model | 98.2 [97.0, 99.5] | | 98.2 [97.0, 99.5] | |

The socket layer chosen on the dev fold was block 1 on the 0.8B model and block 2 on the 2B. On both, the
host reads the slots up to about block 4 and stops reading them from block 5.

On Qwen3.5-4B only the dev sweep ran (200 clips) before the run was stopped to free memory: the plug
scored 98.0 to 99.5 at blocks 0 to 4, against 20.0 to 23.0 for swapped audio. There is no test-fold number
for the 4B.

### Text-only plug (no paired audio, so every class is unseen)

The plug is fitted on 12,000 AudioCaps captions as text only: CLAP's text embedding of a caption is mapped
to the host's representation of that caption, then applied to audio embeddings.

| Condition | 0.8B identify | 0.8B category | 2B identify | 2B category |
|---|---|---|---|---|
| Text-only plug | 53.2 [48.0, 58.0] | 31.8 [27.3, 36.2] | 52.0 [47.0, 57.0] | 45.0 [40.0, 50.0] |
| Swapped audio | 19.8 [16.0, 23.8] | 20.0 [16.5, 24.0] | 20.0 [16.2, 24.0] | 25.0 [21.0, 29.2] |
| Nearest caption written into the socket | 76.5 [72.2, 80.5] | 52.8 [47.8, 57.8] | 75.8 [71.2, 79.8] | 57.5 [52.5, 62.3] |
| Nearest caption pasted into the prompt as text (no socket) | 83.2 [79.5, 86.8] | 66.0 [61.5, 70.5] | not run | not run |

On the 80 test clips from ten classes held out of the paired plug's fit:

| Plug | 0.8B | 2B |
|---|---|---|
| Paired plug, fitted on the other 40 classes | 37.5 [27.5, 48.8] | 43.8 [32.5, 55.0] |
| Text-only plug | 62.5 [52.5, 72.5] | 70.0 [60.0, 80.0] |

Free-text answers through the text-only plug are generic on the 0.8B model, for example "A low, continuous
hum." for a dog. On the 2B model about a third of twelve samples name the sound (dog, church bells, rain)
and the rest are wrong or vague.

### Music (GTZAN, 10 genres), Qwen3.5-0.8B

Per genre, 60 tracks fit the plug, 20 pick the layer and the rest are scored. Each track is its middle 10 seconds.

GTZAN holds 14 duplicate recordings (byte-identical files). The first version of `run_music.py` ignored them, so
5 of its 199 held-out tracks had an identical copy among the fitting or dev tracks, and one recording filed under
two genres (metal.00058 = rock.00016) could reach the "held out" rock-or-classical test as a fitted metal track.
The current script keeps each duplicate group in one split and leaves that recording out of everything; the
first-version column is reproduced exactly with `--keep-duplicates`. Both columns are measured runs
(`artifacts/music_qwen35_0p8b.json` and `artifacts/music_qwen35_0p8b_dedup.json`). The two splits also differ in
which tracks are scored and in the option order, so the change between the columns is not all due to the
duplicates; every change lies within the 95% intervals.

| Genre, five options, held-out tracks of seen genres | First version (199 tracks, block 0) | De-duplicated (196 tracks, block 1) |
|---|---|---|
| Plug | 84.9 [79.4, 89.9] | 83.7 [78.6, 88.8] |
| Swapped audio | 11.1 [7.0, 15.6] | 14.3 [9.7, 19.4] |
| Nearest fitting track (lookup) | 79.9 [73.9, 85.4] | 78.1 [71.9, 83.7] |
| Own text (ceiling) | 100.0 | 100.0 |
| No write | 22.6 [17.1, 28.6] | 23.5 [17.9, 29.6] |
| CLAP zero-shot, no language model | 49.2 [42.2, 56.3] | 51.0 [43.9, 58.2] |

| Rock or classical, two options (chance 50) | First version | De-duplicated |
|---|---|---|
| Plug fitted on all ten genres, held-out tracks | 100.0 (40 tracks) | 94.9 [87.2, 100.0] (39) |
| Same, audio swapped | 0.0 (40) | 5.1 [0.0, 12.8] (39) |
| Plug fitted with rock and classical held out | 53.5 [46.5, 60.5] (200) | 51.8 [44.7, 58.8] (199) |
| Text-only plug | 53.5 [46.5, 60.0] (200) | 50.3 [43.7, 57.3] (199) |
| Plug fitted on environmental sounds only (ESC-50) | 52.0 [45.0, 59.0] (200) | 50.3 [43.2, 57.8] (199) |
| No write | 52.0 [45.5, 59.0] (200) | 49.2 [42.2, 56.3] (199) |
| Nearest caption pasted into the prompt as text | 66.5 [60.0, 72.5] (200) | 63.8 [57.3, 70.4] (199) |
| CLAP zero-shot, no language model | 98.5 [96.5, 100.0] (200) | 98.5 [96.5, 100.0] (199) |

An unseen track of a seen genre is classified reliably; an unseen genre is at chance, although CLAP's own
vector separates the two genres at 98.5%. GTZAN is known to repeat artists across tracks, so "held-out
track" is weaker than "held-out artist".

### Vision transplant (Qwen3.5-2B into SmolLM2-1.7B, no image training)

`transplant.py` gives a text-only model another model's reading of an image. The donor, Qwen3.5-2B, looks
at the image with its own vision. Its hidden states at the image positions, about 138 vectors per image,
go through a ridge map into the recipient's input embeddings. The map is fitted on 8,000 words that both
tokenizers hold as a single token, and on no image.

The test is COCO val2017 images with one dominant object (at least 15% of the image area). Each question
offers that object's category and four categories absent from the image. 100 images pick the layers and
400 are scored once.

| Condition | Accuracy |
|---|---|
| SmolLM2 with transplanted vision | 62.5 [57.8, 67.2] |
| SmolLM2, image swapped | 27.8 [23.5, 32.0] |
| SmolLM2, no image | 24.0 [19.8, 28.2] |
| Qwen3.5-2B's own answer (native vision) | 97.8 [96.2, 99.0] |

The setting chosen on the dev images takes the donor's states after block 19 of 24 and writes them as the
recipient's input embeddings, norm-matched. The vision tower's raw output (the donor's input layer)
transfers nothing: 17% to 27% on dev. Writing after the recipient's blocks 1 to 3 is worse than writing at
its input.

Free-text answers land near the subject but read as lists of related things: "a variety of animals,
including a bear, a whale, and a deer" for a bear, "buses, trains, and bicycles" for a bus.

Two limits. By block 19 the donor's language model has already read the image, so this passes on the
donor's reading, not raw vision. And the donor could simply write a caption for the recipient to read; we
did not measure that baseline here, and the donor's own 97.8% suggests it would score higher.

## What it shows

- **The mechanism works on Qwen3.5.** A closed-form map, fitted in about a second or less, makes the model
  answer questions about a sound it was never trained to hear, about 78 to 80 points above swapped audio.
- **What crosses the socket is the label.** Lookup of the nearest fitting clip matches the plug, and CLAP
  alone identifies the sound as well. The language model's contribution is the category question, where
  the plug reaches the host's own text ceiling.
- **Open vocabulary is not there.** With no paired audio the plug beats chance and beats the paired plug
  on unseen classes, but it loses to retrieval, and retrieval pasted as plain text beats both.
- **The limit is the map, not the vector's size.** The plug can only output blends of the targets it was
  fitted on. CLAP's 512 numbers hold the rock-or-classical distinction; the plug does not carry it across
  unless those genres were in the fitting set.
- **A map fitted on words alone can carry a whole token sequence between models.** A text-only model
  reached 62.5% on image questions through another model's late-layer states, against 24.0% with no
  image, with no image used in fitting. It conveys the subject's neighbourhood, not detail.

## Run it

Requires `uv`, a local Qwen3.5 snapshot, and ESC-50 unpacked in its own directory (GTZAN and the AudioCaps
`train.csv` for the music and caption runs). Run these from `socket_pilot/`. The harness's
`scripts/setup_machine.py rebuild` runs the three embedding commands for you.

```bash
uv sync
.venv/bin/python -I clap_embed.py --esc50 <ESC-50-master> --out artifacts/esc50_clap.npz
.venv/bin/python -I clap_embed_gtzan.py --gtzan <GTZAN genres dir> --out artifacts/gtzan_clap.npz
.venv/bin/python run_pilot.py --host <qwen3.5 snapshot> --out artifacts/results.json
# music genres (needs all three embedding files); --keep-duplicates reproduces the first-version column
.venv/bin/python run_music.py --host <qwen3.5 snapshot> --out artifacts/music.json
# open-vocabulary test and the plain-text baseline
.venv/bin/python -I clap_text.py --captions <audiocaps train.csv> --out artifacts/audiocaps_text_clap.npz
.venv/bin/python run_textplug.py --host <qwen3.5 snapshot> --out artifacts/textplug.json
.venv/bin/python run_cascade.py --host <qwen3.5 snapshot> --out artifacts/cascade.json
# vision transplant: build the COCO question set, then run donor -> recipient
.venv/bin/python -I coco_prep.py --ann <dir with COCO val2017 annotation json> --out <set dir>
.venv/bin/python transplant.py --donor <qwen3.5 snapshot> --recipient <smollm2 snapshot> --set <set dir> \
    --out artifacts/transplant.json --n-dev 100 --n-test 400
```

Timings on an M4 Pro with 24 GB: fitting the plug, penalty search included, takes 0.5 to 1.0 seconds;
embedding the 2,000 clips takes under three minutes including the encoder download; a full pilot run
takes about 5 minutes on the 0.8B model and 12 minutes on the 2B.

The results above are these files in `artifacts/`:

| File | Section |
|---|---|
| `results_qwen35_0p8b.json`, `results_qwen35_2b.json` | Paired plug (`run_pilot.py`) |
| `textplug_qwen35_0p8b.json`, `textplug_qwen35_2b.json` | Text-only plug, nearest caption written into the socket, and the 80 unseen-class clips (`run_textplug.py`) |
| `cascade_qwen35_0p8b.json` | Nearest caption pasted into the prompt as text (`run_cascade.py`) |
| `music_qwen35_0p8b.json`, `music_qwen35_0p8b_dedup.json` | Music, first version and de-duplicated (`run_music.py`) |
| `transplant_qwen35_2b_to_smollm2.json` | Vision transplant (`transplant.py`) |

## Harness runs (sense/, 14 and 15 November)

Runs of the harness in `sense/`, `agent/` and `eval/`, not of the pilot scripts, on the same M4 Pro, saved in
`artifacts/` by `scripts/demo_cli.py --save`, `scripts/score_free_text.py` and `eval/triage.py --save`, except
where a row names a one-off script. The `run_*.json` records are in the format `sense.runs.import_record` reads,
so they can be replayed (`POST /api/replay` after importing them). An imported record replays its trace and
cards only: it brings its sense back only when it is imported on the machine that made it while that very sense
is still installed there.

| File | What it holds |
|---|---|
| `run_qwen3.5-0.8b_gtzan_closed.json` | GTZAN, closed scope, after the duplicate fix (#41): `vector:gtzan` installed at 79.5% |
| `run_qwen3.5-0.8b_esc50_open.json` | ESC-50, open scope: `vector:esc50` refused (51.2% on unseen sounds), `text:captions` installed at 80.5% |
| `run_qwen3.5-0.8b_gtzan-folder_closed.json` | the GTZAN genre folders through `sense.datasets.from_folder` (50 clips per genre): 77.0% |
| `run_openai-chat_esc50_open.json` | the API model `openai-chat` (gpt-5.4-mini), ESC-50 open scope, 50 items: `text:captions` at 90.0%, with API usage and five chats; recorded by a one-off script (its `note` says how) |
| `free_text_qwen3.5-0.8b_gtzan_vector.json` | free-text answers through `vector:gtzan`: the reply names the true genre and no other in 37.0% of 100 clips |
| `free_text_qwen3.5-0.8b_esc50_vector.json` | free-text answers through `vector:esc50`: 45.0% under strict matching ("Clock" for clock tick counts as no label) |
| `run_auto_claude-sonnet_3d.json` | agent-written adapter, real mode through `POST /api/auto` (15 November): Claude Sonnet 5.5, a ModelNet40 guitar uploaded as `mystery.glb`, goal "Tell me what object this 3D model is", no examples. `gpt-6.1-terra` wrote an adapter that renders views (1 attempt); the agent found ModelNet40 meshes in a folder-labelled Hub repository (8 objects, 160 files) and, as the unseen set, codepawl/scanned-objects-glb (8 categories, CC BY 4.0). Installed `adapter:3d-glb` at 100.0% [92.9, 100] on 50 questions (swapped 12%, no input 20%, raw file 42%, unseen 64% at 20% chance), in 362 s |
| `run_auto_openai-chat_3d.json` | the same with `openai-chat` (`gpt-6.1-terra`) as the base model and another guitar: installed `adapter:3d-glb` at 100.0% (swapped 18%, no input 28%, raw file 58%, unseen 72%) |
| `run_auto_claude-sonnet_video.json` | Claude Sonnet 5.5, a UCF101 horse-riding clip as `mystery.mp4`, "What action is happening in this video?": a frames adapter; HMDB51 (8 actions) found on the Hub, and 8 other HMDB51 actions from another upload of the same corpus as the unseen set. Installed `adapter:video-mp4` at 78.0% [64.8, 87.3] (swapped 8%, no input 20%, unseen 92%) |
| `run_auto_claude-sonnet_molecule.json` | Claude Sonnet 5.5, vanillin's SMILES as `mystery.smi`, "What does this molecule smell like?": an RDKit-descriptor text adapter; Hari5115/molecular-odor-dataset (8 odour labels, MIT). Installed `adapter:molecule-smiles` at 58.0% [44.2, 70.6], 12 points above the 46% the model scored on the raw SMILES (swapped 10%, no input 14%); no second source was found, so unseen categories were not measured. An earlier, unrecorded run on the same data refused at 8 points over native |
| `run_auto_claude-sonnet_coordinates.json` | Claude Sonnet 5.5, a coordinate pair as `mystery.txt`, "Which country is this place in?": the nearest-places adapter scored 100.0% but the model reading the raw coordinates scored 98%, so nothing was installed ("the model already reads this file as it is"); unseen (8 other countries from Lilambd/world-cities) 100% |
| `run_auto_claude-sonnet_midi-first.json` | blind test, Claude Sonnet 5.5, a script-written MIDI file as `mystery.mid` (no metadata), "What genre of music is this?": detect said unknown bytes; the agent asked to install `mido` (installer verdict ok, approved), wrote an adapter that ran first time, then found only audio genre sets, whose MP3 files the adapter failed on (50 of 50), so nothing was verified (fixed in PR #103) |
| `run_auto_claude-sonnet_midi.json` | the same file and goal after the format-word search fix: the agent tried `projectlosangeles/lakh-clean-midi-genres` first (MIDI in a tar archive, labels in a separate index, labels predicted by a classifier), could not read probes from it, and asked for labelled examples; no card |
| `run_auto_claude-sonnet_midi-examples.json` | the same file and goal with an examples folder: 60 piano MIDI files from ADL Piano MIDI in 5 genres (one per artist, renamed; kept local, no licence stated): 32.0% [20.8, 45.8] on 50 held-out files, chance 20%, swapped 10%, no input 20%; below the 35% bar, refused |
| `run_auto_qwen3.5-0.8b_coordinates.json` | the same file and goal on local Qwen3.5-0.8B: 24.4% on 160 questions, swapped 21.9%, raw file 23.8% (chance 20%); refused, the small model cannot use the adapter's text |
| `run_auto_qwen3.5-0.8b_music-genres.json` | the first real autonomous run (15 November, through `POST /api/auto` on the real-mode server): Qwen3.5-0.8B, a GTZAN rock clip uploaded as `mystery clip.wav`, goal "Make it recognise music genres"; live Linkup research and dataset search, `gpt-6.1-terra` for the need step and planner. Both fetched candidates failed with HTTP 502 from the Hugging Face dataset viewer, so the run fell back to the pre-wired GTZAN set. Installed `vector:gtzan` at 79.5% [73.4, 84.5] (swapped 12.5%, no input 22.0%, lookup 84.5%, specialist alone 53.5%) over `text:gtzan` at 47.5%, in 110 s |
| `run_openai-chat_esc50_closed.json` | scenario 3 through the agent: `openai-chat` on ESC-50, closed scope, 50 items. The loop tried text and tool (vector is unavailable to an API model); `tool:esc50` was installed at 96.0% (swapped 22.0%, no input 18.0%) over `text:esc50` at 92.0%. The tool card's answers were served from the answer cache filled by the run in `tool_openai-chat_closed.json` (same questions and clips). In chat afterwards the model called `listen` and named chainsaw and church bells, and called a rooster clip a hen, as CLAP ranked it; those chats were observed, not stored in this record |
| `tool_openai-chat_closed.json` | the tool bridge with `openai-chat`, `verify` called directly by a one-off script (closed scope, 50 items): `tool:esc50` 96.0% and `tool:gtzan` 52.0%, the same as the specialist-alone control (96.0% and 50.0%), with the API usage and one chat each in which the model called `listen` |
| `triage_qwen3.5-0.8b_esc50.json` | forest triage (`eval/triage.py`), Qwen3.5-0.8B through `text:captions`, the first ("ranger") wording on ESC-50 fold 5, run before any dev-fold run: the sense flagged 369 of 400 and no input flagged 400 (all "chainsaw"); this failed control is what led to choosing the wording on fold 4 |
| `triage_qwen3.5-0.8b_esc50_dev_ranger.json`, `triage_qwen3.5-0.8b_esc50_dev_neutral.json`, `triage_qwen3.5-0.8b_esc50_dev_yesno.json`, `triage_qwen3.5-0.8b_esc50_dev_chat.json` | the four wordings on fold 4 (dev): 372, 295, 212 and 21 flagged, against 400, 400, 400 and 0 with no input; chat was chosen |
| `triage_qwen3.5-0.8b_esc50_test_chat.json` | the chosen chat wording on fold 5: 26 flagged, 13 of 24 targets caught (50.0% precision), none with no input; CLAP lookup 24 flagged, 23 caught |
| `triage_openai-chat_esc50_dev_chat.json`, `triage_openai-chat_esc50_test_chat.json` | the same chat wording with `openai-chat` through `text:captions`: fold 4 22 flagged, 16 caught; fold 5 23 flagged, 14 of 24 caught (60.9% precision), none with no input |

Fit time in the harness: the closed-form fit of one block (ridge with its penalty search, plus the host's label
targets) took 0.50 to 0.66 s on GTZAN's 597 fitting clips and 1.13 to 1.25 s on ESC-50's 1,200
(`fit_seconds_by_layer` in the free-text files). The agent's whole fit step took 23.2 s and 25.9 s, because it
fits five candidate blocks and scores each on 200 dev questions.

## Models and data used

- Host: `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B` (frozen). Transplant recipient:
  `HuggingFaceTB/SmolLM2-1.7B-Instruct` (frozen).
- Encoder: `laion/clap-htsat-unfused` (frozen).
- Data: [ESC-50](https://github.com/karolpiczak/ESC-50) audio and labels; caption text from
  [AudioCaps](https://github.com/cdjkim/audiocaps) (`train.csv`, text only);
  [GTZAN](https://huggingface.co/datasets/marsyas/gtzan) music clips and genre labels;
  [COCO](https://cocodataset.org) val2017 images, object annotations and captions.
