"""Music test: does the audio socket carry genre, and how does it do on music it was not fitted on?

GTZAN, 10 genres x 100 tracks. Per genre: 60 tracks fit the plug, 20 pick the layer, 20 are scored.
Two questions:
  genre, five options : true genre plus four others, on held-out tracks of seen genres
  rock or classical   : two options, under plugs that have seen less and less
      all genres seen           paired plug fitted on all ten genres (held-out tracks only)
      rock, classical held out  paired plug fitted on the other eight genres
      text-only                 plug fitted on caption text, no music and no labels
      environmental sounds only the ESC-50 plug from the first pilot

GTZAN holds 14 duplicate recordings (byte-identical files, so their CLAP embeddings have cosine 1). By default
each duplicate group goes to one split whole, and the one recording filed under two genres (metal.00058 =
rock.00016) is left out of everything, including the rock-or-classical tests. --keep-duplicates reproduces the
first version of this script, whose split put 5 of the 200 held-out tracks next to an identical copy in fit or dev.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from host import Host
from run_pilot import LETTERS, acc_ci, fit_plug

DISPLAY = {"hiphop": "hip hop"}


def unit(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def zscore(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return (a - ref.mean(0)) / (ref.std(0) + 1e-8)


DUPLICATE_COSINE = 0.999  # the same rule as sense.datasets: at or above this, two clips are one recording


def duplicate_groups(emb: np.ndarray, members: np.ndarray) -> list[list[int]]:
    """Groups of `members` (indices into emb) whose embeddings are near-identical, joined transitively."""
    u = unit(emb[members])
    parent = list(range(len(members)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(np.triu(u @ u.T >= DUPLICATE_COSINE, k=1))):
        a, b = find(int(i)), find(int(j))
        if a != b:
            parent[max(a, b)] = min(a, b)
    groups: dict[int, list[int]] = {}
    for i in range(len(members)):
        groups.setdefault(find(i), []).append(int(members[i]))
    return sorted(groups.values(), key=lambda g: g[0])


def split_tracks(emb: np.ndarray, y: np.ndarray, n_genres: int, rng: np.random.Generator, keep_duplicates: bool):
    """Per genre: 60 tracks fit, 20 dev, the rest test, in a seeded order. Unless keep_duplicates, a recording under
    two genres is left out of everything and each group of duplicates goes to one split whole; with no duplicates
    the result, and the rng's state afterwards, are the same as with keep_duplicates.
    Returns fit, dev, test (index arrays), the left-out indices and the number of duplicate groups kept together."""
    left_out: list[int] = []
    if not keep_duplicates:
        u = unit(emb)
        across = np.triu((u @ u.T >= DUPLICATE_COSINE) & (y[:, None] != y[None, :]), k=1)
        left_out = sorted({int(i) for pair in zip(*np.nonzero(across)) for i in pair})
    usable = ~np.isin(np.arange(len(y)), left_out)
    fit_idx, dev_idx, test_idx = [], [], []
    kept_together = 0
    for g in range(n_genres):
        members = np.where((y == g) & usable)[0]
        if keep_duplicates:
            order = rng.permutation(members)
            fit_idx += list(order[:60])
            dev_idx += list(order[60:80])
            test_idx += list(order[80:])
            continue
        groups = duplicate_groups(emb, members)
        kept_together += sum(len(group) > 1 for group in groups)
        fit: list[int] = []
        dev: list[int] = []
        test: list[int] = []
        for k in rng.permutation(len(groups)):
            (fit if len(fit) < 60 else dev if len(dev) < 20 else test).extend(groups[k])
        fit_idx += fit
        dev_idx += dev
        test_idx += test
    return (*map(lambda a: np.array(a, dtype=np.int64), (fit_idx, dev_idx, test_idx)), left_out, kept_together)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--gtzan", default="artifacts/gtzan_clap.npz")
    ap.add_argument("--esc", default="artifacts/esc50_clap.npz")
    ap.add_argument("--captions", default="artifacts/audiocaps_text_clap.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0,1,2,3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="split as the first version did, ignoring GTZAN's duplicate recordings (for comparison)")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    t_start = time.time()
    rng = np.random.default_rng(args.seed)
    gt = np.load(args.gtzan, allow_pickle=False)
    emb, y = gt["emb"].astype(np.float64), gt["y"]
    names = [DISPLAY.get(str(n), str(n)) for n in gt["names"]]
    genre_text = unit(gt["text_emb"].astype(np.float64))
    rock, classical = names.index("rock"), names.index("classical")

    fit_idx, dev_idx, test_idx, left_out, kept_together = split_tracks(emb, y, len(names), rng, args.keep_duplicates)
    usable = ~np.isin(np.arange(len(y)), left_out)
    split = {
        "duplicates": "kept as in the first version" if args.keep_duplicates else "each group in one split",
        "left_out": [str(gt["files"][i]) for i in left_out],
        "duplicate_groups_kept_in_one_split": kept_together,
        "sizes": {"fit": len(fit_idx), "dev": len(dev_idx), "test": len(test_idx)},
    }
    print(f"split: {split}")

    host = Host(args.host)
    ids5 = [host.token_id(letter) for letter in LETTERS]
    ids2 = ids5[:2]
    reps = host.text_reps(names + [f"{n} music" for n in names], layers=layers)
    n_g = len(names)
    target = (reps[:n_g] + reps[n_g:]) / 2  # [genres, layers, d]
    print(f"host {Path(args.host).parents[1].name}; genres {names}")

    def five_way(idx):
        prompts, answers, options = [], [], []
        for i in idx:
            wrong = rng.choice([c for c in range(n_g) if c != y[i]], 4, replace=False)
            opts = [int(y[i]), *map(int, wrong)]
            rng.shuffle(opts)
            lines = "\n".join(f"{LETTERS[k]}. {names[c]}" for k, c in enumerate(opts))
            prompts.append(
                f"Audio clip: {host.slots}\nQuestion: Which genre of music is in the audio clip?\n{lines}\n"
                "Answer with the letter only."
            )
            answers.append(opts.index(int(y[i])))
            options.append(opts)
        return prompts, np.array(answers), np.array(options)

    def two_way(idx, slot_text):
        prompts, answers = [], []
        for i in idx:
            first_rock = bool(rng.integers(2))
            a, b = ("rock", "classical") if first_rock else ("classical", "rock")
            prompts.append(
                f"Audio clip: {slot_text}\nQuestion: Which genre of music is in the audio clip?\nA. {a}\nB. {b}\n"
                "Answer with the letter only."
            )
            answers.append(0 if (y[i] == rock) == first_rock else 1)
        return prompts, np.array(answers)

    X = zscore(emb, emb[fit_idx])

    # ---- socket layer, chosen on dev tracks with the all-genres paired plug
    dev_prompts, dev_ans, _ = five_way(dev_idx)
    sweep = []
    for k, layer in enumerate(layers):
        plug, _ = fit_plug(X[fit_idx], target[y[fit_idx], k])
        acc = float((host.score(dev_prompts, ids5, layer=layer, slot_vecs=plug(X[dev_idx])).argmax(1) == dev_ans).mean())
        sweep.append({"layer": layer, "plug": round(100 * acc, 1)})
        print(f"layer {layer}: paired plug {100 * acc:5.1f} on dev", flush=True)
    L = max(sweep, key=lambda r: r["plug"])["layer"]
    k = layers.index(L)
    print(f"socket layer chosen on dev: {L}")
    results = {}

    # ---- five-way genre on held-out tracks of seen genres
    plug_all, _ = fit_plug(X[fit_idx], target[y[fit_idx], k])
    prompts, ans, opts = five_way(test_idx)
    swap = np.array([int(test_idx[y[test_idx] != y[i]][rng.integers((y[test_idx] != y[i]).sum())]) for i in test_idx])
    nearest = fit_idx[(unit(X[test_idx]) @ unit(X[fit_idx]).T).argmax(1)]
    five = {}
    for name, vecs in {
        "plug": plug_all(X[test_idx]),
        "swapped audio": plug_all(X[swap]),
        "nearest fit track": target[y[nearest], k],
        "own text (ceiling)": target[y[test_idx], k],
        "no write": None,
    }.items():
        five[name] = acc_ci(host.score(prompts, ids5, layer=L, slot_vecs=vecs).argmax(1) == ans, rng)
    sims = np.take_along_axis(unit(emb[test_idx]) @ genre_text.T, opts, axis=1)
    five["CLAP zero-shot (no LM)"] = acc_ci(sims.argmax(1) == ans, rng)
    results[f"genre, five options, held-out tracks (n={len(test_idx)})"] = five

    # ---- rock or classical
    rc_test = test_idx[np.isin(y[test_idx], [rock, classical])]
    rc_all = np.where(np.isin(y, [rock, classical]) & usable)[0]
    two = {}

    p, a = two_way(rc_test, host.slots)
    other = np.array([int(rc_test[y[rc_test] != y[i]][rng.integers((y[rc_test] != y[i]).sum())]) for i in rc_test])
    two[f"all genres seen (n={len(rc_test)})"] = acc_ci(host.score(p, ids2, layer=L, slot_vecs=plug_all(X[rc_test])).argmax(1) == a, rng)
    two[f"all genres seen, audio swapped (n={len(rc_test)})"] = acc_ci(host.score(p, ids2, layer=L, slot_vecs=plug_all(X[other])).argmax(1) == a, rng)

    p, a = two_way(rc_all, host.slots)
    n_all = len(rc_all)
    two[f"no write (n={n_all})"] = acc_ci(host.score(p, ids2).argmax(1) == a, rng)

    seen8 = fit_idx[~np.isin(y[fit_idx], [rock, classical])]
    X8 = zscore(emb, emb[seen8])
    plug8, _ = fit_plug(X8[seen8], target[y[seen8], k])
    two[f"rock, classical held out (n={n_all})"] = acc_ci(host.score(p, ids2, layer=L, slot_vecs=plug8(X8[rc_all])).argmax(1) == a, rng)

    caps = np.load(args.captions, allow_pickle=False)
    cap_emb, cap_texts = unit(caps["text_emb"].astype(np.float64)), [str(t) for t in caps["texts"]]
    cap_reps = host.text_reps(cap_texts, layers=[L])[:, 0]
    plug_text, _ = fit_plug(zscore(cap_emb, cap_emb), cap_reps)
    audio_u = unit(emb)
    Xa = zscore(audio_u, audio_u[fit_idx])  # audio statistics only, no labels
    two[f"text-only plug (n={n_all})"] = acc_ci(host.score(p, ids2, layer=L, slot_vecs=plug_text(Xa[rc_all])).argmax(1) == a, rng)

    esc = np.load(args.esc, allow_pickle=False)
    e_emb, e_y, e_fold = esc["emb"].astype(np.float64), esc["y"], esc["fold"]
    e_names = [str(n) for n in esc["names"]]
    e_reps = host.text_reps(e_names + [f"the sound of {n}" for n in e_names], layers=[L])[:, 0]
    e_fit = e_fold <= 3
    plug_esc, _ = fit_plug(zscore(e_emb[e_fit], e_emb[e_fit]), ((e_reps[:50] + e_reps[50:]) / 2)[e_y[e_fit]])
    two[f"environmental sounds only (n={n_all})"] = acc_ci(
        host.score(p, ids2, layer=L, slot_vecs=plug_esc(zscore(emb[rc_all], e_emb[e_fit]))).argmax(1) == a, rng
    )

    nearest_cap = (audio_u[rc_all] @ cap_emb.T).argmax(1)
    p_text, a_text = two_way(rc_all, "<<CLIP>>")
    filled = [q.replace("<<CLIP>>", f'"{cap_texts[c]}"') for q, c in zip(p_text, nearest_cap)]
    two[f"nearest caption pasted as text, no socket (n={n_all})"] = acc_ci(host.score(filled, ids2).argmax(1) == a_text, rng)

    zs = (audio_u[rc_all] @ genre_text[[rock, classical]].T).argmax(1)
    two[f"CLAP zero-shot, no LM (n={n_all})"] = acc_ci(zs == (y[rc_all] != rock).astype(int), rng)
    results["rock or classical, two options (chance 50)"] = two

    print(f"\n=== GTZAN, socket layer {L} ===")
    for task, rows in results.items():
        print(f"[{task}]  accuracy %  (95% CI)")
        for name, (m, lo, hi) in rows.items():
            print(f"  {name:54s} {m:5.1f}  [{lo:5.1f}, {hi:5.1f}]")
    out = {"host": Path(args.host).parents[1].name, "socket_layer": L, "split": split, "dev_sweep": sweep, "test": results,
           "seconds": round(time.time() - t_start, 1)}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out} in {out['seconds']}s")


if __name__ == "__main__":
    main()
