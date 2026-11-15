"""Rebuild the demo inputs of the recorded adapter runs from their public sources, so a demo can repeat them.

Each input is written as `mystery.<ext>` in its own folder, so the file name gives nothing away:

  <out>/3d-guitar/mystery.glb   ModelNet40 guitar_0235 (test split), run_auto_claude-sonnet_3d.json
  <out>/3d-chair/mystery.glb    ModelNet40 chair_0946 (test split), run_auto_claude-sonnet_3d-repeat.json
  <out>/video/mystery.mp4       UCF101 v_HorseRiding_g24_c06, first 10 s, run_auto_claude-sonnet_video*.json
  <out>/molecule/mystery.smi    vanillin, COc1cc(C=O)ccc1O, run_auto_claude-sonnet_molecule*.json

Sources, pinned to the revisions the runs used:
  meshes  https://huggingface.co/datasets/naderalfares/ModelNet40_Auto_aligned (ModelNet, Princeton: free for
          academic research; cite Wu et al., 3D ShapeNets, CVPR 2015). OFF is converted to GLB with the same code as
          the runs' inputs: fan-triangulated, rotated from Z-up to glTF's Y-up, positions only.
  video   https://huggingface.co/datasets/sumith2425/UCF101 (UCF101, Soomro, Zamir and Shah 2012: distributed by UCF
          CRCV for research; no explicit licence). Transcoded with ffmpeg: video only, first 10 s, even size, yuv420p,
          libx264 (veryfast, crf 26), bit-exact flags.

Each file's sha256 is compared with the input the recorded run used. The meshes and the SMILES match byte for byte; the
video can differ when another ffmpeg or libx264 version encodes it, which changes the bytes but not the clip. Nothing
is written inside the repository unless you ask for it. Needs network access, and ffmpeg on PATH for the video.

Usage: .venv/bin/python scripts/fetch_demo_inputs.py [--out ~/.cache/sense-harness/demo-inputs] [--only 3d-chair ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from array import array
from pathlib import Path

HF = "https://huggingface.co"
MESH_REPO, MESH_REV = "naderalfares/ModelNet40_Auto_aligned", "aeb0af95f3ddc6503b5613c28ebcece569e3c2e2"
VIDEO_REPO, VIDEO_REV = "sumith2425/UCF101", "1112f14679d5c678584b59ed86fc93a3dd6f52d4"

# name -> (kind, source path or text, sha256 of the input the recorded run used)
INPUTS = {
    "3d-guitar": ("mesh", "guitar/test/guitar_0235.off",
                  "7bc313f45ea9aec1415b22f566ba5164a66ae9da3f2c167de982222d566ded25"),
    "3d-chair": ("mesh", "chair/test/chair_0946.off",
                 "66d6d674b69ad41f8382c10bf039ef7048da337ffdcc446ca2b12a11ebd43e14"),
    "video": ("video", "train/HorseRiding/v_HorseRiding_g24_c06.avi",
              "06cc18aa436eeaa41c1af049717b0b9ac75f2688aba689b2327d185208246ef0"),
    "molecule": ("smiles", "COc1cc(C=O)ccc1O",
                 "ce59d14178f6b96f7e68160f4616288407abec9462a82056470b0cb920fe5598"),
}
EXT = {"mesh": ".glb", "video": ".mp4", "smiles": ".smi"}


def download(repo: str, rev: str, path: str) -> bytes:
    url = f"{HF}/datasets/{repo}/resolve/{rev}/{urllib.parse.quote(path)}"
    req = urllib.request.Request(url, headers={"User-Agent": "sensorium-demo-inputs"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def read_off(text: str) -> tuple[list[float], list[int]]:
    """Parse OFF text into flat xyz floats and fan-triangulated indices (ModelNet's "OFF<n> <m> 0" quirk included)."""
    lines = [s for s in (raw.split("#", 1)[0].strip() for raw in text.splitlines()) if s]
    if not lines or not lines[0].startswith("OFF"):
        raise ValueError("not an OFF file")
    rest, i = lines[0][3:].strip(), 1
    if not rest:
        rest, i = lines[1], 2
    nv, nf = (int(t) for t in rest.split()[:2])
    if nv < 3 or nf < 1 or len(lines) < i + nv + nf:
        raise ValueError("bad OFF counts")
    verts: list[float] = []
    for s in lines[i:i + nv]:
        x, y, z = (float(t) for t in s.split()[:3])
        if not all(math.isfinite(c) for c in (x, y, z)):
            raise ValueError("non-finite vertex")
        verts += (x, y, z)
    tris: list[int] = []
    for s in lines[i + nv:i + nv + nf]:
        t = s.split()
        n = int(t[0])
        idx = [int(v) for v in t[1:1 + n]]
        if len(idx) != n or n < 3 or min(idx) < 0 or max(idx) >= nv:
            raise ValueError("bad face")
        for k in range(1, n - 1):
            a, b, c = idx[0], idx[k], idx[k + 1]
            if a != b and b != c and a != c:
                tris += (a, b, c)
    if not tris:
        raise ValueError("no triangles")
    return verts, tris


def _le(a: array) -> bytes:
    if sys.byteorder != "little":
        a = array(a.typecode, a)
        a.byteswap()
    return a.tobytes()


def glb_bytes(verts: list[float], tris: list[int]) -> bytes:
    """A binary glTF holding one mesh, rotated from Z-up to Y-up (x, z, -y)."""
    rot = [0.0] * len(verts)
    rot[0::3], rot[1::3], rot[2::3] = verts[0::3], verts[2::3], [-y for y in verts[1::3]]
    pos = array("f", rot)
    f32 = list(pos)  # min/max must match the stored float32 values exactly
    nv = len(f32) // 3
    mins, maxs = [min(f32[k::3]) for k in range(3)], [max(f32[k::3]) for k in range(3)]
    idx, ctype = (array("H", tris), 5123) if nv < 65536 else (array("I", tris), 5125)
    pos_b, idx_b = _le(pos), _le(idx)
    idx_pad = b"\x00" * (-len(idx_b) % 4)
    gltf = {
        "asset": {"version": "2.0", "generator": "fetch_3d.py"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
        "buffers": [{"byteLength": len(pos_b) + len(idx_b) + len(idx_pad)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_b), "target": 34962},
            {"buffer": 0, "byteOffset": len(pos_b), "byteLength": len(idx_b), "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": nv, "type": "VEC3", "min": mins, "max": maxs},
            {"bufferView": 1, "componentType": ctype, "count": len(tris), "type": "SCALAR"},
        ],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    binc = pos_b + idx_b + idx_pad
    total = 12 + 8 + len(js) + 8 + len(binc)
    return (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(js), b"JSON") + js
            + struct.pack("<I4s", len(binc), b"BIN\x00") + binc)


def transcode(avi: bytes, dest: Path) -> None:
    ffmpeg = shutil.which("ffmpeg") or shutil.which("ffmpeg", path="/opt/homebrew/bin:/usr/local/bin")
    if not ffmpeg:
        raise SystemExit("ffmpeg is needed for the video input (brew install ffmpeg, or apt install ffmpeg)")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "clip.avi"
        src.write_bytes(avi)
        cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-t", "10.0",
               "-map", "0:v:0", "-an", "-sn", "-dn", "-map_metadata", "-1", "-map_chapters", "-1",
               "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
               "-fflags", "+bitexact", "-flags:v", "+bitexact", "-movflags", "+faststart", str(dest)]
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode != 0:
            raise SystemExit(f"ffmpeg failed: {run.stderr.strip()[:300]}")


def build(name: str, out: Path) -> tuple[Path, bool]:
    kind, source, sha = INPUTS[name]
    dest = out / name / f"mystery{EXT[kind]}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if kind == "mesh":
        dest.write_bytes(glb_bytes(*read_off(download(MESH_REPO, MESH_REV, source).decode("ascii"))))
    elif kind == "video":
        transcode(download(VIDEO_REPO, VIDEO_REV, source), dest)
    else:
        dest.write_text(source + "\n")
    return dest, hashlib.sha256(dest.read_bytes()).hexdigest() == sha


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=Path("~/.cache/sense-harness/demo-inputs"))
    ap.add_argument("--only", nargs="*", choices=sorted(INPUTS), help="build only these inputs")
    args = ap.parse_args()
    out = args.out.expanduser()
    for name in args.only or list(INPUTS):
        dest, same = build(name, out)
        note = "identical to the recorded run's input" if same else (
            "differs in bytes from the recorded run's input" + (" (another ffmpeg build; the clip is the same)"
                                                                  if INPUTS[name][0] == "video" else ""))
        print(f"{name:10} {dest}  ({note})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
