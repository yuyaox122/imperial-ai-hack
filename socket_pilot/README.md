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

### Text-only plug (no paired audio, so every class is unseen), Qwen3.5-0.8B

The plug is fitted on 12,000 AudioCaps captions as text only: CLAP's text embedding of a caption is mapped
to the host's representation of that caption, then applied to audio embeddings.

| Condition | Identify | Category |
|---|---|---|
| Text-only plug | 53.2 [48.0, 58.0] | 31.8 [27.3, 36.2] |
| Swapped audio | 19.8 [16.0, 23.8] | 20.0 [16.5, 24.0] |
| Nearest caption written into the socket | 76.5 [72.2, 80.5] | 52.8 [47.8, 57.8] |
| Nearest caption pasted into the prompt as text (no socket) | 83.2 [79.5, 86.8] | 66.0 [61.5, 70.5] |

On the 80 test clips from ten classes held out of the paired plug's fit, the paired plug scores
37.5 [27.5, 48.8] and the text-only plug 62.5 [52.5, 72.5].

Free-text answers through the text-only plug are generic, for example "A low, continuous hum." for a dog.

## What it shows

- **The mechanism works on Qwen3.5.** A closed-form map, fitted in about a second or less, makes the model
  answer questions about a sound it was never trained to hear, about 78 to 80 points above swapped audio.
- **What crosses the socket is the label.** Lookup of the nearest fitting clip matches the plug, and CLAP
  alone identifies the sound as well. The language model's contribution is the category question, where
  the plug reaches the host's own text ceiling.
- **Open vocabulary is not there.** With no paired audio the plug beats chance and beats the paired plug
  on unseen classes, but it loses to retrieval, and retrieval pasted as plain text beats both.

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
```

Timings on an M4 Pro with 24 GB: fitting the plug, penalty search included, takes 0.5 to 1.0 seconds;
embedding the 2,000 clips takes under three minutes including the encoder download; a full pilot run
takes about 5 minutes on the 0.8B model and 12 minutes on the 2B.

## Models and data used

- Host: `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B` (frozen).
- Encoder: `laion/clap-htsat-unfused` (frozen).
- Data: [ESC-50](https://github.com/karolpiczak/ESC-50) audio and labels; caption text from
  [AudioCaps](https://github.com/cdjkim/audiocaps) (`train.csv`, text only).
