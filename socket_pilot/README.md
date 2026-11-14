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

Per genre, 60 tracks fit the plug, 20 pick the layer and 20 are scored. Each track is its middle 10 seconds.

| Genre, five options, 199 held-out tracks of seen genres | Accuracy |
|---|---|
| Plug | 84.9 [79.4, 89.9] |
| Swapped audio | 11.1 [7.0, 15.6] |
| Nearest fitting track (lookup) | 79.9 [73.9, 85.4] |
| Own text (ceiling) | 100.0 |
| No write | 22.6 [17.1, 28.6] |
| CLAP zero-shot, no language model | 49.2 [42.2, 56.3] |

| Rock or classical, two options (chance 50) | Tracks | Accuracy |
|---|---|---|
| Plug fitted on all ten genres, held-out tracks | 40 | 100.0 |
| Same, audio swapped | 40 | 0.0 |
| Plug fitted with rock and classical held out | 200 | 53.5 [46.5, 60.5] |
| Text-only plug | 200 | 53.5 [46.5, 60.0] |
| Plug fitted on environmental sounds only (ESC-50) | 200 | 52.0 [45.0, 59.0] |
| No write | 200 | 52.0 [45.5, 59.0] |
| Nearest caption pasted into the prompt as text | 200 | 66.5 [60.0, 72.5] |
| CLAP zero-shot, no language model | 200 | 98.5 [96.5, 100.0] |

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

Requires `uv`, a local Qwen3.5 snapshot, and ESC-50 unpacked in its own directory.

```bash
uv sync
.venv/bin/python -I clap_embed.py --esc50 <ESC-50-master> --out artifacts/esc50_clap.npz
.venv/bin/python run_pilot.py --host <qwen3.5 snapshot> --out artifacts/results.json
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

## Harness runs (sense/, 14 November)

Runs of the harness in `sense/` and `agent/`, not of the pilot scripts, on the same M4 Pro, saved in
`artifacts/` by `scripts/demo_cli.py --save` and `scripts/score_free_text.py`. The run records are in the format
`sense.runs.import_record` reads, so they can be replayed (`POST /api/replay` after importing them).

| File | What it holds |
|---|---|
| `run_qwen3.5-0.8b_gtzan_closed.json` | GTZAN, closed scope, after the duplicate fix (#41): `vector:gtzan` installed at 79.5% |
| `run_qwen3.5-0.8b_esc50_open.json` | ESC-50, open scope: `vector:esc50` refused (51.2% on unseen sounds), `text:captions` installed at 80.5% |
| `run_qwen3.5-0.8b_gtzan-folder_closed.json` | the GTZAN genre folders through `sense.datasets.from_folder` (50 clips per genre): 77.0% |
| `run_openai-chat_esc50_open.json` | the API model `openai-chat` (gpt-5.4-mini), ESC-50 open scope, 50 items: `text:captions` at 90.0%, with API usage and five chats |
| `free_text_qwen3.5-0.8b_gtzan_vector.json` | free-text answers through `vector:gtzan`: the reply names the true genre and no other in 37.0% of 100 clips |
| `free_text_qwen3.5-0.8b_esc50_vector.json` | free-text answers through `vector:esc50`: 45.0% under strict matching ("Clock" for clock tick counts as no label) |

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
