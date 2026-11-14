"""The baseline any socket has to beat: no socket at all.

Retrieve the nearest caption for the clip with CLAP and paste it into the prompt as plain text.
Same held-out ESC-50 fold and question formats as the socket runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from host import Host
from run_pilot import LETTERS, acc_ci, category_items, identify_items


def unit(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--emb", default="artifacts/esc50_clap.npz")
    ap.add_argument("--captions", default="artifacts/audiocaps_text_clap.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    esc = np.load(args.emb, allow_pickle=False)
    y, fold, names = esc["y"], esc["fold"], [str(n) for n in esc["names"]]
    audio = unit(esc["emb"].astype(np.float64))
    caps = np.load(args.captions, allow_pickle=False)
    cap_emb, cap_texts = unit(caps["text_emb"].astype(np.float64)), [str(t) for t in caps["texts"]]

    test_idx = np.where(fold == 5)[0]
    nearest = (audio[test_idx] @ cap_emb.T).argmax(1)
    swap = np.array([int(test_idx[y[test_idx] != y[i]][rng.integers((y[test_idx] != y[i]).sum())]) for i in test_idx])
    nearest_swap = (audio[swap] @ cap_emb.T).argmax(1)

    host = Host(args.host)
    option_ids = [host.token_id(letter) for letter in LETTERS]
    marker = "<<CLIP>>"
    id_prompts, id_ans, _ = identify_items(test_idx, y, names, marker, rng)
    cat_prompts, cat_ans = category_items(test_idx, y, marker, rng)

    results = {}
    for task, prompts, ans in [("identify", id_prompts, id_ans), ("category", cat_prompts, cat_ans)]:
        results[task] = {}
        for name, picks in {"nearest caption as text": nearest, "swapped audio": nearest_swap}.items():
            filled = [p.replace(marker, f'"{cap_texts[c]}"') for p, c in zip(prompts, picks)]
            results[task][name] = acc_ci(host.score(filled, option_ids).argmax(1) == ans, rng)

    print(f"=== text cascade, held-out test fold, n={len(test_idx)} ===")
    for task in results:
        print(f"[{task}]  accuracy %  (95% CI); chance is 20")
        for name, (m, lo, hi) in results[task].items():
            print(f"  {name:26s} {m:5.1f}  [{lo:5.1f}, {hi:5.1f}]")
    Path(args.out).write_text(json.dumps({"host": Path(args.host).parents[1].name, "test": results}, indent=2))


if __name__ == "__main__":
    main()
