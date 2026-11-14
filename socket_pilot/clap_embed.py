"""Embed every ESC-50 clip with a frozen CLAP audio encoder.

Self-contained on purpose: it is the only script that reads the downloaded dataset, so run it as
`python -I clap_embed.py --esc50 <dir> --out <file.npz>` with the dataset in its own directory.
"""

import argparse
import csv
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import ClapModel, ClapProcessor

SR = 48_000  # CLAP's input rate


def as_tensor(out):
    """transformers returns either a tensor or an output object, depending on version."""
    return out if torch.is_tensor(out) else out.pooler_output


def load_clip(path: Path) -> np.ndarray:
    wav, fs = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if fs != SR:
        g = gcd(SR, fs)
        wav = resample_poly(wav, SR // g, fs // g)
    return wav.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--esc50", required=True, help="path to the unpacked ESC-50-master directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="laion/clap-htsat-unfused")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    root = Path(args.esc50)
    with open(root / "meta" / "esc50.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    names = [""] * 50
    for r in rows:
        names[int(r["target"])] = r["category"].replace("_", " ")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        model = ClapModel.from_pretrained(args.model, use_safetensors=True)
    except Exception as err:  # repo without safetensors weights
        print(f"safetensors load failed ({type(err).__name__}); falling back to the default loader")
        model = ClapModel.from_pretrained(args.model)
    model = model.to(device).eval()
    proc = ClapProcessor.from_pretrained(args.model)

    embs = []
    for start in range(0, len(rows), args.batch):
        wavs = [load_clip(root / "audio" / r["filename"]) for r in rows[start : start + args.batch]]
        try:
            inputs = proc(audio=wavs, sampling_rate=SR, return_tensors="pt", padding=True)
        except TypeError:
            inputs = proc(audios=wavs, sampling_rate=SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            feats = as_tensor(model.get_audio_features(**{k: v.to(device) for k, v in inputs.items()}))
        embs.append(feats.float().cpu().numpy())
        print(f"audio {min(start + args.batch, len(rows))}/{len(rows)}", flush=True)
    emb = np.concatenate(embs)

    text_inputs = proc(text=[f"the sound of {n}" for n in names], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_emb = as_tensor(model.get_text_features(**{k: v.to(device) for k, v in text_inputs.items()}))
    text_emb = text_emb.float().cpu().numpy()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        emb=emb,
        text_emb=text_emb,
        y=np.array([int(r["target"]) for r in rows]),
        fold=np.array([int(r["fold"]) for r in rows]),
        names=np.array(names),
        files=np.array([r["filename"] for r in rows]),
    )
    print("saved", args.out, "audio", emb.shape, "text", text_emb.shape)


if __name__ == "__main__":
    main()
