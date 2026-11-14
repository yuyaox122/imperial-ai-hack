"""Embed caption text with CLAP's frozen text tower, for the text-only plug.

Self-contained: reads a downloaded captions CSV (a `caption` column), so run it as
`python -I clap_text.py --captions <file.csv> --out <file.npz>`.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor


def as_tensor(out):
    return out if torch.is_tensor(out) else out.pooler_output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="laion/clap-htsat-unfused")
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.captions, newline="") as fh:
        texts = sorted({r["caption"].strip() for r in csv.DictReader(fh) if r["caption"].strip()})
    order = np.random.default_rng(args.seed).permutation(len(texts))[: args.limit]
    texts = [texts[i] for i in order]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        model = ClapModel.from_pretrained(args.model, use_safetensors=True)
    except Exception:
        model = ClapModel.from_pretrained(args.model)
    model = model.to(device).eval()
    proc = ClapProcessor.from_pretrained(args.model)

    embs = []
    for start in range(0, len(texts), args.batch):
        inputs = proc(text=texts[start : start + args.batch], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embs.append(as_tensor(model.get_text_features(**{k: v.to(device) for k, v in inputs.items()})).float().cpu().numpy())
    emb = np.concatenate(embs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, text_emb=emb, texts=np.array(texts))
    print("saved", args.out, emb.shape)


if __name__ == "__main__":
    main()
