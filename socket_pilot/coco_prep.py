"""Build the image question set for transplant.py from COCO val2017.

Picks images with one clearly dominant object, writes a five-option question for each (the dominant
object plus four categories absent from the image), and downloads only those images.

Self-contained: it reads the downloaded annotation files, so run it as
`python -I coco_prep.py --ann <dir with instances_val2017.json, captions_val2017.json> --out <dir>`.
"""

import argparse
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LETTERS = "ABCDE"
BASE = "http://images.cocodataset.org/val2017/"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=650)
    ap.add_argument("--min-frac", type=float, default=0.15, help="dominant object's share of the image area")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ann = Path(args.ann)
    inst = json.loads((ann / "instances_val2017.json").read_text())
    caps = json.loads((ann / "captions_val2017.json").read_text())
    cats = {c["id"]: c["name"] for c in inst["categories"]}
    images = {im["id"]: im for im in inst["images"]}
    caption = {}
    for c in caps["annotations"]:
        caption.setdefault(c["image_id"], c["caption"].strip())
    present, biggest = {}, {}
    for a in inst["annotations"]:
        if a.get("iscrowd"):
            continue
        present.setdefault(a["image_id"], set()).add(a["category_id"])
        if a["area"] > biggest.get(a["image_id"], (0, None))[0]:
            biggest[a["image_id"]] = (a["area"], a["category_id"])

    rng = random.Random(args.seed)
    ids = sorted(i for i, (area, _) in biggest.items() if area / (images[i]["width"] * images[i]["height"]) >= args.min_frac)
    rng.shuffle(ids)
    items = []
    for i in ids[: args.n]:
        truth = biggest[i][1]
        opts = [truth, *rng.sample(sorted(set(cats) - present[i]), 4)]
        rng.shuffle(opts)
        lines = "\n".join(f"{LETTERS[k]}. {cats[c]}" for k, c in enumerate(opts))
        items.append(
            {
                "file": images[i]["file_name"],
                "question": f"Question: Which of these is in the image?\n{lines}\nAnswer with the letter only.",
                "answer": opts.index(truth),
                "truth": cats[truth],
                "caption": caption.get(i, ""),
            }
        )

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    def fetch(name: str) -> None:
        dest = out / "images" / name
        if not dest.exists():
            urllib.request.urlretrieve(BASE + name, dest)

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(fetch, [it["file"] for it in items]))
    (out / "items.json").write_text(json.dumps(items, indent=1))
    print(f"{len(items)} items from {len(ids)} eligible images; images in {out / 'images'}")


if __name__ == "__main__":
    main()
