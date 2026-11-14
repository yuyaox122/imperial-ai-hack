"""Vision transplant: can a text-only model see through another model's eyes, with no image training?

Donor     a frozen vision-language model (Qwen3.5). Its vision tower turns an image into a sequence of
          vectors that its own language model reads like word embeddings (about one per 1,024 pixels).
Recipient a frozen text-only model (SmolLM2). It has never seen an image.
Map       closed-form ridge regression from the donor's representation space to the recipient's,
          fitted on TEXT ONLY: tokens and words both models know. No image is used to fit it.
Test      push the donor's image-token states through the map into the recipient, and ask the recipient
          five-option questions about COCO images. Chance is 20%.

Two places to transplant, both swept on a dev split:
  donor layer      take the image-position states at the donor's input (0) or after a later block
  recipient layer  write them as input embeddings ("in") or over the hidden states after an early block
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

LETTERS = "ABCDE"
DEV = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
DTYPE = torch.bfloat16 if DEV.type == "cuda" else torch.float16
FIT_DEV = DEV if DEV.type == "cuda" else torch.device("cpu")  # the closed-form fit needs float64, which MPS lacks


def same_length_batches(lengths, batch):
    by_len = {}
    for i, n in enumerate(lengths):
        by_len.setdefault(n, []).append(i)
    for idx in by_len.values():
        for s in range(0, len(idx), batch):
            yield idx[s : s + batch]


# ------------------------------------------------------------------ donor
class Donor:
    def __init__(self, path: str):
        self.proc = AutoProcessor.from_pretrained(path)
        self.tok = self.proc.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=DTYPE).to(DEV).eval()
        self.lm = self.model.model.language_model
        self.n_layers = len(self.lm.layers)
        self.embed = self.model.get_input_embeddings()
        self.image_id = self.model.config.image_token_id

    def _inputs(self, image: Image.Image, text: str):
        msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]
        kw = dict(tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        try:
            return self.proc.apply_chat_template(msgs, enable_thinking=False, **kw).to(DEV)
        except TypeError:
            return self.proc.apply_chat_template(msgs, **kw).to(DEV)

    @torch.no_grad()
    def see(self, image: Image.Image, question: str, option_ids: list[int], layers: list[int]):
        """One pass: the donor's own answer, and its states at the image positions.
        Layer 0 is the language model's input (the vision tower's output); layer k is after block k-1."""
        inputs = self._inputs(image, question)
        out = self.model(**inputs, output_hidden_states=True)
        mask = inputs["input_ids"][0] == self.image_id
        states = {k: out.hidden_states[k][0, mask].to(torch.float16).cpu() for k in layers}
        return out.logits[0, -1, option_ids].float().cpu().numpy(), states

    @torch.no_grad()
    def text_states(self, texts: list[str], layers: list[int], batch: int = 256) -> torch.Tensor:
        return _text_states(self.tok, self.lm, texts, layers, batch)


# ------------------------------------------------------------------ recipient
class Recipient:
    def __init__(self, path: str):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=DTYPE).to(DEV).eval()
        self.lm = self.model.model
        self.n_layers = len(self.lm.layers)
        self.embed = self.model.get_input_embeddings()
        self.filler_id = self.tok.encode(" image", add_special_tokens=False)[0]

    def _wrap(self, question: str, with_image: bool):
        marker = "<<IMG>>"
        body = f"Image: {marker}\n{question}" if with_image else question
        text = self.tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)
        if not with_image:
            return self.tok.encode(text, add_special_tokens=False), []
        pre, post = text.split(marker)
        return self.tok.encode(pre, add_special_tokens=False), self.tok.encode(post, add_special_tokens=False)

    def _embeds(self, question: str, soft: torch.Tensor | None, layer: int | None):
        pre, post = self._wrap(question, soft is not None)
        ids = lambda x: self.embed(torch.tensor(x, device=DEV))  # noqa: E731
        if soft is None:
            return ids(pre)[None], None
        soft = soft.to(DEV, DTYPE)
        middle = soft if layer is None else self.embed(torch.full((len(soft),), self.filler_id, device=DEV))
        x = torch.cat([ids(pre), middle, ids(post)])[None]
        if layer is None:
            return x, None
        start, end, full = len(pre), len(pre) + len(soft), x.shape[1]

        def hook(_m, _i, output):
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] == full:
                hs[0, start:end] = soft.to(hs.dtype)
            return output

        return x, self.lm.layers[layer].register_forward_hook(hook)

    @torch.no_grad()
    def score(self, question: str, option_ids: list[int], soft=None, layer=None) -> np.ndarray:
        x, handle = self._embeds(question, soft, layer)
        try:
            return self.model(inputs_embeds=x).logits[0, -1, option_ids].float().cpu().numpy()
        finally:
            if handle is not None:
                handle.remove()

    @torch.no_grad()
    def say(self, question: str, soft=None, layer=None, max_new_tokens: int = 30) -> str:
        x, handle = self._embeds(question, soft, layer)
        try:
            out = self.model.generate(inputs_embeds=x, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.eos_token_id)
        finally:
            if handle is not None:
                handle.remove()
        return self.tok.decode(out[0], skip_special_tokens=True).strip()

    @torch.no_grad()
    def text_states(self, texts: list[str], layers: list[int], batch: int = 256) -> torch.Tensor:
        return _text_states(self.tok, self.lm, texts, layers, batch)


@torch.no_grad()
def _text_states(tok, lm, texts, layers, batch) -> torch.Tensor:
    """Mean-pooled state of each text at the requested layers (0 = input embeddings): [n, len(layers), d]."""
    enc = [tok.encode(t, add_special_tokens=False) for t in texts]
    out = None
    for idx in same_length_batches([len(e) for e in enc], batch):
        hs = lm(input_ids=torch.tensor([enc[i] for i in idx], device=DEV), output_hidden_states=True).hidden_states
        if out is None:
            out = torch.zeros(len(texts), len(layers), hs[0].shape[-1], dtype=torch.float32)
        for k, layer in enumerate(layers):
            out[idx, k] = hs[layer].float().mean(1).cpu()
    return out


# ------------------------------------------------------------------ the map
def fit_map(X: torch.Tensor, T: torch.Tensor, seed: int = 0) -> dict:
    """Ridge regression X -> T in closed form; the penalty is picked on a 10% holdout of the text pairs."""
    X, T = X.to(FIT_DEV, torch.float64), T.to(FIT_DEV, torch.float64)
    mu_x, mu_t = X.mean(0), T.mean(0)
    Xc, Tc = X - mu_x, T - mu_t
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed)).to(FIT_DEV)
    ho, tr = perm[: len(X) // 10], perm[len(X) // 10 :]
    eye = torch.eye(X.shape[1], device=FIT_DEV, dtype=torch.float64)
    G, B = Xc[tr].T @ Xc[tr], Xc[tr].T @ Tc[tr]
    scale = G.diagonal().mean()
    best = (-1e9, None)
    for rel in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
        W = torch.linalg.solve(G + rel * scale * eye, B)
        r2 = 1 - ((Tc[ho] - Xc[ho] @ W) ** 2).sum() / ((Tc[ho] - Tc[ho].mean(0)) ** 2).sum()
        if r2.item() > best[0]:
            best = (r2.item(), rel)
    W = torch.linalg.solve(Xc.T @ Xc + best[1] * (Xc.T @ Xc).diagonal().mean() * eye, Xc.T @ Tc)
    return {"W": W, "mu_x": mu_x, "mu_t": mu_t, "r2": best[0], "rel_lambda": best[1], "dev_norm": Tc.norm(dim=1).median()}


def apply_map(m: dict, v: torch.Tensor, match_norm: bool) -> torch.Tensor:
    dev = (v.to(FIT_DEV, torch.float64) - m["mu_x"]) @ m["W"]
    if match_norm:
        dev = dev / dev.norm(dim=1, keepdim=True).clamp_min(1e-9) * m["dev_norm"]
    return (m["mu_t"] + dev).float()


def shared_words(donor_tok, recip_tok, limit: int, seed: int) -> list[str]:
    """Word-initial alphabetic strings that are a single token in both vocabularies."""

    def table(tok):
        out = {}
        for token, i in tok.get_vocab().items():
            s = tok.convert_tokens_to_string([token])
            if s.startswith(" ") and s[1:].isalpha() and len(s) >= 4 and s[1:].islower():
                out[s] = i
        return out

    a, b = table(donor_tok), table(recip_tok)
    words = sorted(set(a) & set(b))
    random.Random(seed).shuffle(words)
    return words[:limit]


# ------------------------------------------------------------------ the questions
def load_items(set_dir: Path, n: int):
    """The question set built by coco_prep.py: items.json plus an images/ directory."""
    items = json.loads((set_dir / "items.json").read_text())[:n]
    for it in items:
        it["file"] = str(set_dir / "images" / it["file"])
    return items


def load_image(path: str, side: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side))
    return im


def acc_ci(correct: np.ndarray, rng, n: int = 2000):
    boots = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(n)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return round(100 * float(correct.mean()), 1), round(100 * float(lo), 1), round(100 * float(hi), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", required=True)
    ap.add_argument("--recipient", required=True)
    ap.add_argument("--set", required=True, help="directory written by coco_prep.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-dev", type=int, default=150)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--side", type=int, default=448)
    ap.add_argument("--words", type=int, default=8000)
    ap.add_argument("--donor-layers", default="0,2,4,8,12,16,20")
    ap.add_argument("--recipient-layers", default="in,1,2,3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maps-only", action="store_true", help="fit the text-only maps, report them and stop")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    donor, recip = Donor(args.donor), Recipient(args.recipient)
    d_layers = [int(x) for x in args.donor_layers.split(",") if int(x) <= donor.n_layers]
    r_layers = [None if x == "in" else int(x) for x in args.recipient_layers.split(",")]
    print(f"donor {donor.n_layers} blocks, recipient {recip.n_layers} blocks, device {DEV}", flush=True)

    # ---- the map, from text only
    words = shared_words(donor.tok, recip.tok, args.words, args.seed)
    r_state_layers = [0 if x is None else x + 1 for x in r_layers]  # hidden_states index of each write point
    Xw = donor.text_states(words, d_layers)
    Tw = recip.text_states(words, r_state_layers)
    maps = {}
    for i, dl in enumerate(d_layers):
        for j, rl in enumerate(r_layers):
            maps[dl, rl] = fit_map(Xw[:, i], Tw[:, j], args.seed)
    print(f"{len(words)} shared words; text-only maps fitted in {time.time() - t0:.0f}s", flush=True)
    for (dl, rl), m in maps.items():
        print(f"  donor layer {dl:2d} -> recipient {'in' if rl is None else rl}: holdout R2 {m['r2']:.3f}")
    if args.maps_only:
        return

    # ---- the donor looks at every image once
    items = load_items(Path(args.set), args.n_dev + args.n_test)
    d_opts = [donor.tok.encode(letter, add_special_tokens=False)[0] for letter in LETTERS]
    r_opts = [recip.tok.encode(letter, add_special_tokens=False)[0] for letter in LETTERS]
    states, donor_ok = [], []
    for it in items:
        logits, st = donor.see(load_image(it["file"], args.side), it["question"], d_opts, d_layers)
        states.append(st)
        donor_ok.append(int(logits.argmax()) == it["answer"])
    n_tok = int(np.mean([len(s[d_layers[0]]) for s in states]))
    print(f"donor read {len(items)} images, about {n_tok} tokens each, in {time.time() - t0:.0f}s", flush=True)
    dev, test = list(range(args.n_dev)), list(range(args.n_dev, len(items)))

    def run(idx, dl, rl, match, source=None):
        ok = []
        for i in idx:
            src = i if source is None else source[i]
            soft = apply_map(maps[dl, rl], states[src][dl], match)
            ok.append(int(recip.score(items[i]["question"], r_opts, soft=soft, layer=rl).argmax()) == items[i]["answer"])
        return np.array(ok)

    # ---- sweep on dev
    sweep = []
    for dl in d_layers:
        for rl in r_layers:
            for match in (False, True):
                acc = float(run(dev, dl, rl, match).mean())
                sweep.append({"donor_layer": dl, "recipient_layer": "in" if rl is None else rl, "match_norm": match, "acc": round(100 * acc, 1)})
        row = [f"{s['acc']:5.1f}" for s in sweep if s["donor_layer"] == dl]
        print(f"dev  donor layer {dl:2d}: " + " ".join(row) + f"   (recipient {args.recipient_layers}, each raw then norm-matched)", flush=True)
    best = max(sweep, key=lambda s: s["acc"])
    dl, match = best["donor_layer"], best["match_norm"]
    rl = None if best["recipient_layer"] == "in" else best["recipient_layer"]
    print(f"best on dev: {best}")

    # ---- held-out test
    swap = {i: test[(k + len(test) // 2) % len(test)] for k, i in enumerate(test)}
    blind = np.array([int(recip.score(items[i]["question"], r_opts).argmax()) == items[i]["answer"] for i in test])
    results = {
        "recipient with transplanted vision": acc_ci(run(test, dl, rl, match), rng),
        "recipient, image swapped": acc_ci(run(test, dl, rl, match, source=swap), rng),
        "recipient, no image": acc_ci(blind, rng),
        "donor's own answer (native vision)": acc_ci(np.array(donor_ok)[test], rng),
    }
    samples = []
    for i in test[:10]:
        soft = apply_map(maps[dl, rl], states[i][dl], match)
        samples.append({"truth": items[i]["truth"], "caption": items[i]["caption"], "recipient_says": recip.say("Describe the image in one sentence.", soft=soft, layer=rl)})

    print(f"\n=== COCO, {len(test)} held-out images, five options, chance 20 ===")
    for name, (m, lo, hi) in results.items():
        print(f"  {name:38s} {m:5.1f}  [{lo:5.1f}, {hi:5.1f}]")
    print("\nfree text (main object | recipient says):")
    for s in samples:
        print(f"  {s['truth']:14s} | {s['recipient_says']!r}")
    out = {
        "donor": Path(args.donor).parents[1].name,
        "recipient": Path(args.recipient).parents[1].name,
        "n_words": len(words),
        "image_tokens": n_tok,
        "best": best,
        "dev_sweep": sweep,
        "map_r2": {f"{a}->{'in' if b is None else b}": round(m["r2"], 3) for (a, b), m in maps.items()},
        "test": results,
        "samples": samples,
        "seconds": round(time.time() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out} in {out['seconds']}s")


if __name__ == "__main__":
    main()
