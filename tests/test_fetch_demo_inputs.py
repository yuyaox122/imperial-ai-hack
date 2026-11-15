"""scripts/fetch_demo_inputs.py, offline: the OFF to GLB conversion that rebuilds the 3D demo inputs byte for byte."""
from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fetch_demo_inputs.py"
_spec = importlib.util.spec_from_file_location("fetch_demo_inputs", _PATH)
fetch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch)


def test_an_off_with_modelnets_header_quirk_becomes_a_y_up_glb():
    # ModelNet writes the counts on the OFF line itself; a quad fans into two triangles
    verts, tris = fetch.read_off("OFF4 2 0\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n4 0 1 2 3\n3 0 1 3\n")
    assert len(verts) == 12 and tris == [0, 1, 2, 0, 2, 3, 0, 1, 3]
    glb = fetch.glb_bytes(verts, tris)
    magic, version, total = struct.unpack_from("<4sII", glb)
    assert (magic, version, total) == (b"glTF", 2, len(glb))
    length, kind = struct.unpack_from("<I4s", glb, 12)
    gltf = json.loads(glb[20:20 + length])
    assert kind == b"JSON" and gltf["accessors"][0]["count"] == 4 and gltf["accessors"][1]["count"] == 9
    # Z-up to glTF's Y-up: (x, y, z) -> (x, z, -y)
    assert gltf["accessors"][0]["min"] == [0.0, 0.0, -1.0] and gltf["accessors"][0]["max"] == [1.0, 1.0, 0.0]


@pytest.mark.parametrize("text", ["PLY\n", "OFF\n2 1 0\n0 0 0\n1 1 1\n3 0 1 2\n", "OFF\n3 1 0\n0 0 0\n1 0 0\nnan 0 0\n3 0 1 2\n"])
def test_a_file_that_is_not_a_usable_off_is_refused(text):
    with pytest.raises(ValueError):
        fetch.read_off(text)


def test_each_input_names_its_source_and_the_hash_of_the_recorded_runs_input():
    assert set(fetch.INPUTS) == {"3d-guitar", "3d-chair", "video", "molecule"}
    for kind, source, sha in fetch.INPUTS.values():
        assert kind in fetch.EXT and source and len(sha) == 64
