"""Embed GTZAN music clips (10 genres x 100 tracks, 30 s each) with the frozen CLAP audio encoder.

Self-contained: the only script that reads the downloaded dataset. Run as
`python -I clap_embed_gtzan.py --gtzan <genres dir> --out <file.npz>`.
CLAP takes 10 s of audio, so each track is represented by its middle 10 seconds.
"""

import argparse
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import ClapModel, ClapProcessor

SR = 48_000
WINDOW_S = 10


def as_tensor(out):
    return out if torch.is_tensor(out) else out.pooler_output


def load_middle(path: Path) -> np.ndarray:
    wav, fs = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    half = WINDOW_S * fs // 2
    mid = len(wav) // 2
    wav = wav[max(0, mid - half) : mid + half]
    if fs != SR:
        g = gcd(SR, fs)
        wav = resample_poly(wav, SR // g, fs // g)
    return wav.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtzan", required=True, help="directory holding one sub-directory per genre")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="laion/clap-htsat-unfused")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    root = Path(args.gtzan)
    genres = sorted(p.name for p in root.iterdir() if p.is_dir())
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        model = ClapModel.from_pretrained(args.model, use_safetensors=True)
    except Exception:
        model = ClapModel.from_pretrained(args.model)
    model = model.to(device).eval()
    proc = ClapProcessor.from_pretrained(args.model)

    clips, labels, files = [], [], []
    for g, genre in enumerate(genres):
        for path in sorted((root / genre).glob("*.wav")):
            try:
                clips.append(load_middle(path))
            except Exception as err:  # GTZAN ships one corrupt file
                print(f"skipped {path.name}: {type(err).__name__}")
                continue
            labels.append(g)
            files.append(path.name)

    embs = []
    for start in range(0, len(clips), args.batch):
        batch = clips[start : start + args.batch]
        try:
            inputs = proc(audio=batch, sampling_rate=SR, return_tensors="pt", padding=True)
        except TypeError:
            inputs = proc(audios=batch, sampling_rate=SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            embs.append(as_tensor(model.get_audio_features(**{k: v.to(device) for k, v in inputs.items()})).float().cpu().numpy())
        print(f"audio {min(start + args.batch, len(clips))}/{len(clips)}", flush=True)

    text_inputs = proc(text=[f"{g} music" for g in genres], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_emb = as_tensor(model.get_text_features(**{k: v.to(device) for k, v in text_inputs.items()})).float().cpu().numpy()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, emb=np.concatenate(embs), text_emb=text_emb, y=np.array(labels), names=np.array(genres), files=np.array(files))
    print("saved", args.out, len(labels), "clips;", genres)


if __name__ == "__main__":
    main()
