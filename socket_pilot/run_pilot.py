"""Go/no-go pilot: plug a frozen CLAP audio encoder into a frozen Qwen3.5 with one ridge regression.

Method (after arXiv 2611.05401, "A Training-Free Socket for DNA in Frozen Language Models"):
  target  = the host's own mean-pooled hidden state, after block L, for text naming the sound
  plug    = closed-form ridge map from the audio embedding to that target, then a norm match
  socket  = replace the hidden state at the slot positions after block L, and let the host continue

Scored on five-way multiple choice over held-out ESC-50 clips (folds 1-3 fit, fold 4 dev, fold 5 test).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from host import Host

LETTERS = "ABCDE"
CATEGORIES = [  # ESC-50 groups its 50 classes into five blocks of ten, in this order
    "an animal",
    "nature or water",
    "a human body sound",
    "a household or indoor object",
    "an outdoor or urban source",
]
LAMBDAS = [1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6]


# ---------------------------------------------------------------- the plug
def choose_lambda(X: np.ndarray, T: np.ndarray, k: int = 5, seed: int = 0) -> float:
    """Ridge penalty by k-fold cross-validated R^2 on the fitting clips only."""
    order = np.random.default_rng(seed).permutation(len(X))
    folds = np.array_split(order, k)
    eye = np.eye(X.shape[1])
    best, best_r2 = LAMBDAS[0], -np.inf
    for lam in LAMBDAS:
        ss_res = ss_tot = 0.0
        for f in folds:
            tr = np.setdiff1d(order, f)
            mu = T[tr].mean(0)
            W = np.linalg.solve(X[tr].T @ X[tr] + lam * eye, X[tr].T @ (T[tr] - mu))
            ss_res += ((T[f] - (mu + X[f] @ W)) ** 2).sum()
            ss_tot += ((T[f] - mu) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        if r2 > best_r2:
            best, best_r2 = lam, r2
    return best


def fit_plug(X: np.ndarray, T: np.ndarray):
    """Closed-form ridge map with the paper's norm match. X is already z-scored."""
    X, T = X.astype(np.float64), T.astype(np.float64)
    lam = choose_lambda(X, T)
    mu = T.mean(0)
    W = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ (T - mu))
    rho = np.linalg.norm(T - mu, axis=1).mean()

    def plug(x: np.ndarray) -> np.ndarray:
        dev = x.astype(np.float64) @ W
        dev = dev / np.maximum(np.linalg.norm(dev, axis=1, keepdims=True), 1e-12)
        return (mu + rho * dev).astype(np.float32)

    return plug, lam


# ---------------------------------------------------------------- the questions
def identify_items(idx, y, names, slots, rng):
    prompts, answers, options = [], [], []
    for i in idx:
        wrong = rng.choice([c for c in range(len(names)) if c != y[i]], 4, replace=False)
        opts = [int(y[i]), *map(int, wrong)]
        rng.shuffle(opts)
        lines = "\n".join(f"{LETTERS[k]}. {names[c]}" for k, c in enumerate(opts))
        prompts.append(
            f"Audio clip: {slots}\nQuestion: Which sound is in the audio clip?\n{lines}\nAnswer with the letter only."
        )
        answers.append(opts.index(int(y[i])))
        options.append(opts)
    return prompts, np.array(answers), np.array(options)


def category_items(idx, y, slots, rng):
    """One hop past identity: the options never name the sound, only what kind of thing makes it."""
    prompts, answers = [], []
    for i in idx:
        order = rng.permutation(5)
        lines = "\n".join(f"{LETTERS[k]}. {CATEGORIES[c]}" for k, c in enumerate(order))
        prompts.append(
            f"Audio clip: {slots}\nQuestion: What kind of thing produces the sound in the audio clip?\n{lines}\n"
            "Answer with the letter only."
        )
        answers.append(int(np.where(order == y[i] // 10)[0][0]))
    return prompts, np.array(answers)


def acc_ci(correct: np.ndarray, rng, n: int = 2000):
    boots = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(n)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return round(100 * float(correct.mean()), 1), round(100 * float(lo), 1), round(100 * float(hi), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="path to a local Qwen3.5 snapshot")
    ap.add_argument("--emb", default="artifacts/esc50_clap.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--max-layer", type=int, default=12, help="sweep socket layers 0..max-layer-1")
    ap.add_argument("--dev-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t_start = time.time()
    rng = np.random.default_rng(args.seed)
    data = np.load(args.emb, allow_pickle=False)
    emb, y, fold, names = data["emb"].astype(np.float64), data["y"], data["fold"], [str(n) for n in data["names"]]
    text_emb = data["text_emb"].astype(np.float64)

    fit_idx = np.where(fold <= 3)[0]
    dev_idx = rng.permutation(np.where(fold == 4)[0])[: args.dev_n]
    test_idx = np.where(fold == 5)[0]
    mu_x, sd_x = emb[fit_idx].mean(0), emb[fit_idx].std(0) + 1e-8
    X = (emb - mu_x) / sd_x

    host = Host(args.host, n_slots=args.slots)
    option_ids = [host.token_id(letter) for letter in LETTERS]
    print(f"host {Path(args.host).parents[1].name}: {host.n_layers} blocks, slot {host.slot_text!r} x{args.slots}")

    # the host's own representation of each class, per layer: mean over two text forms
    reps = host.text_reps(names + [f"the sound of {n}" for n in names])
    class_target = (reps[:50] + reps[50:]) / 2  # [50, layers, d]

    def swap_partner(idx):
        """For each clip, another clip of the same split from a different class."""
        out = []
        for i in idx:
            pool = idx[y[idx] != y[i]]
            out.append(int(pool[rng.integers(len(pool))]))
        return np.array(out)

    def nearest_fit_clip(idx):
        a = X[idx] / np.linalg.norm(X[idx], axis=1, keepdims=True)
        b = X[fit_idx] / np.linalg.norm(X[fit_idx], axis=1, keepdims=True)
        return fit_idx[(a @ b.T).argmax(1)]

    # ---- pick the socket layer on dev clips
    dev_prompts, dev_ans, _ = identify_items(dev_idx, y, names, host.slots, rng)
    dev_swap = swap_partner(dev_idx)
    sweep = []
    for layer in range(min(args.max_layer, host.n_layers)):
        plug, lam = fit_plug(X[fit_idx], class_target[y[fit_idx], layer])
        row = {"layer": layer, "lambda": lam}
        for name, vecs in {
            "plug": plug(X[dev_idx]),
            "swap": plug(X[dev_swap]),
            "own_text": class_target[y[dev_idx], layer],
        }.items():
            pred = host.score(dev_prompts, option_ids, layer=layer, slot_vecs=vecs).argmax(1)
            row[name] = round(100 * float((pred == dev_ans).mean()), 1)
        sweep.append(row)
        print(f"layer {layer:2d}  plug {row['plug']:5.1f}  swap {row['swap']:5.1f}  own-text {row['own_text']:5.1f}  lam {lam:g}", flush=True)
    best = max(sweep, key=lambda r: r["plug"] - r["swap"])
    L = best["layer"]
    print(f"socket layer chosen on dev: {L}")

    # ---- one pass on the held-out test fold at that layer
    plug, lam = fit_plug(X[fit_idx], class_target[y[fit_idx], L])
    test_swap = swap_partner(test_idx)
    conditions = {
        "plug": plug(X[test_idx]),
        "swapped audio": plug(X[test_swap]),
        "zero write": np.zeros_like(class_target[y[test_idx], L]),
        "nearest fit clip": class_target[y[nearest_fit_clip(test_idx)], L],
        "own text (ceiling)": class_target[y[test_idx], L],
        "no write": None,
    }
    results = {}
    id_prompts, id_ans, id_opts = identify_items(test_idx, y, names, host.slots, rng)
    cat_prompts, cat_ans = category_items(test_idx, y, host.slots, rng)
    for task, prompts, ans in [("identify", id_prompts, id_ans), ("category", cat_prompts, cat_ans)]:
        results[task] = {}
        correct = {}
        for name, vecs in conditions.items():
            pred = host.score(prompts, option_ids, layer=L, slot_vecs=vecs).argmax(1)
            correct[name] = pred == ans
            results[task][name] = acc_ci(correct[name], rng)
        gap = correct["plug"].astype(float) - correct["swapped audio"].astype(float)
        boots = [gap[rng.integers(0, len(gap), len(gap))].mean() for _ in range(2000)]
        results[task]["gap plug - swapped"] = (
            round(100 * float(gap.mean()), 1),
            round(100 * float(np.percentile(boots, 2.5)), 1),
            round(100 * float(np.percentile(boots, 97.5)), 1),
        )

    # reference: what the encoder alone can do on the same five options (no language model)
    a = emb[test_idx] / np.linalg.norm(emb[test_idx], axis=1, keepdims=True)
    t = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
    sims = np.take_along_axis(a @ t.T, id_opts, axis=1)
    results["identify"]["CLAP zero-shot (no LM)"] = acc_ci(sims.argmax(1) == id_ans, rng)

    print(f"\n=== held-out test fold, n={len(test_idx)}, socket layer {L}, ridge lambda {lam:g} ===")
    for task in results:
        print(f"[{task}]  accuracy %  (95% CI); chance is 20")
        for name, (m, lo, hi) in results[task].items():
            print(f"  {name:24s} {m:5.1f}  [{lo:5.1f}, {hi:5.1f}]")
    out = {
        "host": Path(args.host).parents[1].name,
        "encoder": "laion/clap-htsat-unfused",
        "slots": args.slots,
        "socket_layer": L,
        "n_layers": host.n_layers,
        "ridge_lambda": lam,
        "n_fit": int(len(fit_idx)),
        "n_test": int(len(test_idx)),
        "dev_sweep": sweep,
        "test": results,
        "seconds": round(time.time() - t_start, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out} in {out['seconds']}s")


if __name__ == "__main__":
    main()
