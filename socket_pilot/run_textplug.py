"""Open-vocabulary test: can the plug be fitted with no paired audio at all?

CLAP puts audio and text in one space, so the plug is fitted on caption TEXT only
(CLAP text embedding of a caption -> the host's own representation of that caption) and then
applied to CLAP AUDIO embeddings. No ESC-50 label or audio-text pair is used for fitting, so
every ESC-50 class is unseen. Compared with:
  paired, classes held out : the pilot's paired plug, fitted on 40 classes, tested on the other 10
  nearest caption          : write the host's representation of the closest caption (retrieval)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from host import Host
from run_pilot import LETTERS, acc_ci, category_items, fit_plug, identify_items


def unit(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--emb", default="artifacts/esc50_clap.npz")
    ap.add_argument("--captions", default="artifacts/audiocaps_text_clap.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0,1,2,3")
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--dev-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    t_start = time.time()
    rng = np.random.default_rng(args.seed)
    esc = np.load(args.emb, allow_pickle=False)
    y, fold, names = esc["y"], esc["fold"], [str(n) for n in esc["names"]]
    audio = unit(esc["emb"].astype(np.float64))
    class_text = unit(esc["text_emb"].astype(np.float64))
    caps = np.load(args.captions, allow_pickle=False)
    cap_emb, cap_texts = unit(caps["text_emb"].astype(np.float64)), [str(t) for t in caps["texts"]]

    fit_idx = np.where(fold <= 3)[0]
    dev_idx = rng.permutation(np.where(fold == 4)[0])[: args.dev_n]
    test_idx = np.where(fold == 5)[0]

    # Close the modality gap with unlabeled statistics only: z-score each modality on its own.
    Xt = (cap_emb - cap_emb.mean(0)) / (cap_emb.std(0) + 1e-8)
    a_mu, a_sd = audio[fit_idx].mean(0), audio[fit_idx].std(0) + 1e-8  # audio only, no labels
    Xa = (audio - a_mu) / a_sd

    host = Host(args.host, n_slots=args.slots)
    option_ids = [host.token_id(letter) for letter in LETTERS]
    print(f"host {Path(args.host).parents[1].name}; {len(cap_texts)} captions; layers {layers}")
    cap_reps = host.text_reps(cap_texts, layers=layers)  # [n_caps, n_layers, d]
    cls = host.text_reps(names + [f"the sound of {n}" for n in names], layers=layers)
    class_target = (cls[:50] + cls[50:]) / 2
    print(f"caption representations ready in {time.time() - t_start:.0f}s")

    def swap_partner(idx):
        return np.array([int(idx[y[idx] != y[i]][rng.integers((y[idx] != y[i]).sum())]) for i in idx])

    # ---- choose the socket layer on dev clips
    dev_prompts, dev_ans, _ = identify_items(dev_idx, y, names, host.slots, rng)
    dev_swap = swap_partner(dev_idx)
    plugs, sweep = {}, []
    for k, layer in enumerate(layers):
        plugs[layer], lam = fit_plug(Xt, cap_reps[:, k])
        row = {"layer": layer, "lambda": lam}
        for name, src in {"plug": dev_idx, "swap": dev_swap}.items():
            pred = host.score(dev_prompts, option_ids, layer=layer, slot_vecs=plugs[layer](Xa[src])).argmax(1)
            row[name] = round(100 * float((pred == dev_ans).mean()), 1)
        sweep.append(row)
        print(f"layer {layer}: text-only plug {row['plug']:5.1f}  swap {row['swap']:5.1f}  lam {lam:g}", flush=True)
    L = max(sweep, key=lambda r: r["plug"] - r["swap"])["layer"]
    k = layers.index(L)
    plug = plugs[L]
    print(f"socket layer chosen on dev: {L}")

    # ---- held-out test fold
    nearest_cap = (audio[test_idx] @ cap_emb.T).argmax(1)
    test_swap = swap_partner(test_idx)
    conditions = {
        "text-only plug": plug(Xa[test_idx]),
        "swapped audio": plug(Xa[test_swap]),
        "nearest caption": cap_reps[nearest_cap, k],
        "own text (ceiling)": class_target[y[test_idx], k],
        "no write": None,
    }
    id_prompts, id_ans, id_opts = identify_items(test_idx, y, names, host.slots, rng)
    cat_prompts, cat_ans = category_items(test_idx, y, host.slots, rng)
    results, correct = {}, {}
    for task, prompts, ans in [("identify", id_prompts, id_ans), ("category", cat_prompts, cat_ans)]:
        results[task] = {}
        for name, vecs in conditions.items():
            correct[task, name] = host.score(prompts, option_ids, layer=L, slot_vecs=vecs).argmax(1) == ans
            results[task][name] = acc_ci(correct[task, name], rng)
    sims = np.take_along_axis(audio[test_idx] @ class_text.T, id_opts, axis=1)
    results["identify"]["CLAP zero-shot (no LM)"] = acc_ci(sims.argmax(1) == id_ans, rng)

    # ---- the paired plug cannot do this: hold 10 classes out of its fit, test on them
    held = np.arange(0, 50, 5)
    fit_seen = fit_idx[~np.isin(y[fit_idx], held)]
    emb_raw = esc["emb"].astype(np.float64)
    Xp = (emb_raw - emb_raw[fit_seen].mean(0)) / (emb_raw[fit_seen].std(0) + 1e-8)
    paired, _ = fit_plug(Xp[fit_seen], class_target[y[fit_seen], k])
    sub = np.isin(y[test_idx], held)
    held_prompts = [p for p, s in zip(id_prompts, sub) if s]
    pred = host.score(held_prompts, option_ids, layer=L, slot_vecs=paired(Xp[test_idx][sub])).argmax(1)
    results["identify, 10 classes unseen by the paired plug"] = {
        "paired plug (40 classes)": acc_ci(pred == id_ans[sub], rng),
        "text-only plug": acc_ci(correct["identify", "text-only plug"][sub], rng),
    }

    # ---- free-text samples
    samples = []
    for j in rng.permutation(len(test_idx))[:12]:
        said = host.generate(
            f"Audio clip: {host.slots}\nIn a few words, what sound is in the audio clip?",
            layer=L,
            slot_vec=plug(Xa[test_idx[j : j + 1]])[0],
        )
        samples.append({"truth": names[y[test_idx[j]]], "nearest_caption": cap_texts[nearest_cap[j]], "host_says": said})

    print(f"\n=== held-out test fold, n={len(test_idx)}, socket layer {L} ===")
    for task in results:
        print(f"[{task}]  accuracy %  (95% CI); chance is 20")
        for name, (m, lo, hi) in results[task].items():
            print(f"  {name:26s} {m:5.1f}  [{lo:5.1f}, {hi:5.1f}]")
    print("\nfree text (truth | host says):")
    for s in samples:
        print(f"  {s['truth']:18s} | {s['host_says']!r}")

    out = {
        "host": Path(args.host).parents[1].name,
        "encoder": "laion/clap-htsat-unfused",
        "n_captions": len(cap_texts),
        "socket_layer": L,
        "dev_sweep": sweep,
        "test": results,
        "samples": samples,
        "seconds": round(time.time() - t_start, 1),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out} in {out['seconds']}s")


if __name__ == "__main__":
    main()
