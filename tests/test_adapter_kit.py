"""sense/adapter_kit.py, fully offline: generated videos, meshes, molecules, coordinates and a tiny gazetteer.

Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_adapter_kit.py

Nothing here touches the network or the real Natural Earth file: the gazetteer is a small GeoJSON fixture in
tmp_path, and SENSE_GAZETTEER is pointed at it (or at a missing file) for every test. Optional libraries (trimesh,
matplotlib, rdkit) and ffmpeg are skipped cleanly when absent.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import math
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sense import adapter_kit as ak

ROOT = Path(__file__).resolve().parent.parent
KIT = ROOT / "sense" / "adapter_kit.py"


@pytest.fixture(autouse=True)
def offline_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSE_FAKE", "1")
    monkeypatch.setenv("SENSE_GAZETTEER", str(tmp_path / "no-gazetteer-here.txt"))


# ============================================================================= isolation
def test_kit_imports_alone_under_python_isolated(tmp_path):
    """The harness copies the file alone into a sandbox: it must import with no repository on the path."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    shutil.copy(KIT, sandbox / "adapter_kit.py")
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(sandbox)!r})\n"
        "import adapter_kit\n"
        "assert adapter_kit.__file__.startswith(sys.path[0]), adapter_kit.__file__\n"
        "assert not any(m == 'sense' or m.startswith('sense.') for m in sys.modules), 'imported the repo'\n"
        "print(adapter_kit.read_coordinates('51.5, -0.12'))\n"
    )
    proc = subprocess.run([sys.executable, "-I", "-c", code], cwd=sandbox, capture_output=True, text=True,
                          timeout=120, env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)})
    assert proc.returncode == 0, proc.stderr
    assert "(51.5, -0.12)" in proc.stdout


EAGER_IMPORTS = {"__future__", "io", "json", "math", "os", "re", "shutil", "subprocess", "tempfile", "xml",
                 "pathlib", "numpy", "PIL"}


def test_kit_imports_only_vetted_modules_and_no_network():
    allowed = EAGER_IMPORTS | {"gzip", "trimesh", "matplotlib", "mpl_toolkits", "rdkit", "scipy"}
    tree = ast.parse(KIT.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import would need the repository"
            roots.add(node.module.split(".")[0])
    assert roots <= allowed, roots - allowed
    top_level = {alias.name.split(".")[0] for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
                 for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module)])}
    assert top_level <= EAGER_IMPORTS, "optional libraries must be imported lazily"


def test_docstring_lists_every_public_function():
    doc = ak.__doc__
    for name in ak.__all__:
        if callable(getattr(ak, name)):
            assert f"{name}(" in doc, name


# ============================================================================= video
def _ffmpeg():
    try:
        return ak._find_tool("ffmpeg")
    except ValueError:
        pytest.skip("ffmpeg is not installed")


def _make(path: Path, *args: str) -> Path | None:
    cmd = [_ffmpeg(), "-nostdin", "-v", "error", "-y", *args, str(path)]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    return path if proc.returncode == 0 and path.is_file() and path.stat().st_size else None


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    try:
        ak._find_tool("ffmpeg")
        ak._find_tool("ffprobe")
    except ValueError:
        pytest.skip("ffmpeg/ffprobe are not installed")
    d = tmp_path_factory.mktemp("videos")
    src = ["-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=2"]
    out = {
        "mp4": _make(d / "clip.mp4", *src, "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-shortest",
                     "-pix_fmt", "yuv420p"),
        "webm": _make(d / "clip.webm", *src, "-c:v", "libvpx-vp9", "-b:v", "200k"),
        "mov": _make(d / "clip.mov", *src, "-pix_fmt", "yuv420p"),
        "mkv": _make(d / "clip.mkv", *src, "-c:v", "mpeg4"),
        "avi": _make(d / "clip.avi", *src, "-c:v", "mpeg4"),
        "gif": _make(d / "clip.gif", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=1"),
        "wide": _make(d / "wide.mp4", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=10:duration=1",
                      "-pix_fmt", "yuv420p"),
        "audio": _make(d / "tone.wav", "-f", "lavfi", "-i", "sine=frequency=440:duration=1"),
    }
    return out


def _need(videos, key):
    if videos[key] is None:
        pytest.skip(f"this ffmpeg cannot write the {key} test clip")
    return videos[key]


def test_video_frames_mp4_count_size_and_order(videos):
    frames = ak.video_frames(_need(videos, "mp4"), n=6)
    assert len(frames) == 6
    assert all(isinstance(f, Image.Image) and f.mode == "RGB" for f in frames)
    assert all(f.size == (320, 240) for f in frames), "small clips are not upscaled"
    times = [f.info["time_s"] for f in frames]
    assert times == sorted(times) and 0 < times[0] < 0.4 and 1.6 < times[-1] < 2.0
    first, last = (np.asarray(frames[i], dtype=float) for i in (0, -1))
    assert np.abs(first - last).mean() > 1.0, "frames from different times should differ"


def test_video_frames_downscales_long_side(videos):
    frames = ak.video_frames(_need(videos, "wide"), n=3, size=128)
    assert len(frames) == 3
    assert all(max(f.size) <= 128 for f in frames)
    assert frames[0].size == (128, 72)


@pytest.mark.parametrize("key", ["webm", "mov", "mkv", "avi", "gif"])
def test_video_frames_other_containers(videos, key):
    path = _need(videos, key)
    frames = ak.video_frames(path, n=4, size=384)
    assert len(frames) == 4
    assert all(max(f.size) <= 384 and f.mode == "RGB" for f in frames)
    summary = ak.video_summary(path)
    assert summary["width"] in (320, 160) and summary["height"] in (240, 120)
    assert summary["duration"] == pytest.approx(2.0 if key != "gif" else 1.0, abs=0.15)


def test_video_frames_never_returns_more_frames_than_exist(videos):
    frames = ak.video_frames(_need(videos, "gif"), n=50)  # the gif has 10 frames
    assert 1 <= len(frames) <= 10


def test_video_summary_mp4(videos):
    s = ak.video_summary(_need(videos, "mp4"))
    assert s["duration"] == pytest.approx(2.0, abs=0.1)
    assert s["fps"] == pytest.approx(25.0, abs=0.1)
    assert (s["width"], s["height"]) == (320, 240)
    assert s["has_audio"] is True and s["audio_codec"]
    assert isinstance(s["codec"], str) and s["codec"]
    assert abs(s["n_frames"] - 50) <= 2
    json.dumps(s)


def test_video_errors(videos, tmp_path):
    with pytest.raises(ValueError, match="no such file"):
        ak.video_frames(tmp_path / "missing.mp4")
    junk = tmp_path / "junk.mp4"
    junk.write_text("this is not a video")
    with pytest.raises(ValueError):
        ak.video_frames(junk)
    with pytest.raises(ValueError, match="audio"):
        ak.video_summary(_need(videos, "audio"))
    with pytest.raises(ValueError):
        ak.video_frames(_need(videos, "mp4"), n=0)


def test_still_image_gives_one_frame(tmp_path):
    p = tmp_path / "still.png"
    Image.new("RGB", (900, 300), "red").save(p)
    frames = ak.video_frames(p, n=6, size=300)
    assert len(frames) == 1 and frames[0].size == (300, 100)


# ============================================================================= contact sheet
def test_contact_sheet_grid_width_and_labels(tmp_path):
    colours = ["red", "green", "blue", "yellow", "purple", "orange"]
    images = [Image.new("RGB", (300, 200), c) for c in colours]
    buf = io.BytesIO()
    images[1].save(buf, format="PNG")
    path = tmp_path / "x.jpg"
    images[2].save(path)
    sheet = ak.contact_sheet([images[0], buf.getvalue(), path, *images[3:]])
    assert sheet.mode == "RGB" and sheet.width <= 1024
    assert sheet.width > sheet.height  # 3 x 2 grid of landscape tiles
    small = ak.contact_sheet(images, cols=2, size=400)
    assert small.width <= 400
    px = np.asarray(sheet)
    assert px[6:12, 6:12].mean() < 60, "tile 1 carries a dark index label in its corner"
    assert ak.contact_sheet(images, labels=list("abcdef")).width <= 1024
    with pytest.raises(ValueError):
        ak.contact_sheet([])
    with pytest.raises(ValueError):
        ak.contact_sheet(images, labels=["only one"])
    with pytest.raises(ValueError):
        ak.contact_sheet([b"not an image"])


# ============================================================================= meshes
def _irregular_tetrahedron(trimesh):
    vertices = np.array([[0, 0, 0], [1.0, 0, 0], [0.3, 0.7, 0], [0.2, 0.1, 0.5]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
    mesh = trimesh.Trimesh(vertices, faces)
    mesh.fix_normals()
    return mesh


@pytest.fixture(scope="module")
def meshes(tmp_path_factory):
    trimesh = pytest.importorskip("trimesh")
    c = trimesh.creation
    d = tmp_path_factory.mktemp("meshes")
    shapes = {
        "box": c.box(extents=(1.0, 1.0, 1.0)),
        "slab": c.box(extents=(2.0, 1.0, 0.5)),
        "sphere": c.icosphere(subdivisions=3),
        "cylinder": c.cylinder(radius=0.5, height=3.0, sections=48),
        "tetra": _irregular_tetrahedron(trimesh),
    }
    if hasattr(c, "torus"):
        shapes["torus"] = c.torus(major_radius=1.0, minor_radius=0.3)
    files = {}
    for name, mesh in shapes.items():
        for ext in ("obj", "stl", "ply", "glb", "off"):
            path = d / f"{name}.{ext}"
            mesh.export(path)
            files[(name, ext)] = path
        ascii_stl = d / f"{name}_ascii.stl"
        ascii_stl.write_text(trimesh.exchange.stl.export_stl_ascii(mesh))
        files[(name, "stl_ascii")] = ascii_stl
    return {"dir": d, "files": files, "shapes": shapes, "trimesh": trimesh}


@pytest.mark.parametrize("ext", ["obj", "stl", "stl_ascii", "ply", "glb", "off"])
def test_load_mesh_formats_centre_and_scale(meshes, ext):
    mesh = ak.load_mesh(meshes["files"][("slab", ext)])
    assert isinstance(mesh, meshes["trimesh"].Trimesh)
    assert float(mesh.extents.max()) == pytest.approx(1.0)
    assert np.allclose(mesh.bounds.mean(0), 0, atol=1e-9)
    assert sorted(mesh.metadata["kit_original_extents"], reverse=True) == pytest.approx([2.0, 1.0, 0.5])
    assert mesh.is_watertight


def test_load_mesh_sniffs_format_without_extension(meshes, tmp_path):
    for ext in ("stl", "ply", "glb", "obj"):
        renamed = tmp_path / f"upload_{ext}.bin"
        shutil.copy(meshes["files"][("box", ext)], renamed)
        assert len(ak.load_mesh(renamed).faces) == 12, ext


def test_load_mesh_merges_scene_geometry(meshes, tmp_path):
    trimesh = meshes["trimesh"]
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(extents=(1, 1, 1)))
    scene.add_geometry(trimesh.creation.box(extents=(1, 1, 1)),
                       transform=trimesh.transformations.translation_matrix([3, 0, 0]))
    path = tmp_path / "two.glb"
    scene.export(path)
    mesh = ak.load_mesh(path)
    assert len(mesh.faces) == 24
    assert sorted(mesh.metadata["kit_original_extents"], reverse=True)[0] == pytest.approx(4.0)
    assert ak.mesh_summary(mesh)["components"] == 2


def test_load_mesh_errors(meshes, tmp_path):
    with pytest.raises(ValueError, match="no such file"):
        ak.load_mesh(tmp_path / "missing.stl")
    text = tmp_path / "notes.txt"
    text.write_text("hello, this is not a mesh\n")
    with pytest.raises(ValueError):
        ak.load_mesh(text)
    cloud = tmp_path / "cloud.ply"
    meshes["trimesh"].PointCloud(np.random.default_rng(0).random((20, 3))).export(cloud)
    with pytest.raises(ValueError):
        ak.load_mesh(cloud)


def test_box_summary(meshes):
    s = ak.mesh_summary(meshes["files"][("box", "stl")])
    assert s["watertight"] and s["vertices"] == 8 and s["faces"] == 12
    assert min(s["extent_ratios"]) > 0.95, "a cube has three similar extents"
    assert s["solidity"] > 0.95 and s["bbox_fill"] > 0.95
    assert s["genus_estimate"] == 0 and s["euler_number"] == 2
    assert s["volume"] == pytest.approx(1.0, rel=1e-6) and s["area"] == pytest.approx(6.0, rel=1e-6)
    assert s["mirror_planes"] >= 3 and s["mirror_symmetry"] > 0.9
    assert s["rotational_order"] == 4
    assert "watertight" in s["description"] and "genus 0" in s["description"]
    json.dumps(s)


def test_slab_summary_ratios(meshes):
    s = ak.mesh_summary(meshes["files"][("slab", "ply")])
    assert s["extent_ratios"] == pytest.approx([1.0, 0.5, 0.25], abs=1e-3)
    assert s["original_extents"] == pytest.approx([2.0, 1.0, 0.5], abs=1e-3)
    assert s["volume"] == pytest.approx(1.0, rel=1e-6)
    assert s["rotational_order"] == 2


@pytest.mark.parametrize("ext", ["obj", "stl", "ply", "glb"])
def test_torus_has_genus_one(meshes, ext):
    if ("torus", ext) not in meshes["files"]:
        pytest.skip("trimesh.creation.torus is not available")
    s = ak.mesh_summary(meshes["files"][("torus", ext)])
    assert s["watertight"]
    assert s["euler_number"] == 0 and s["genus_estimate"] == 1
    assert s["solidity"] < 0.8, "the hole makes a torus fill much less than its convex hull"
    assert "1 through-hole" in s["description"]


def test_symmetry_scores(meshes):
    sphere = ak.mesh_summary(meshes["files"][("sphere", "obj")])
    assert sphere["rotational_order"] == "continuous" and sphere["mirror_symmetry"] > 0.9
    cylinder = ak.mesh_summary(meshes["files"][("cylinder", "obj")])
    assert cylinder["rotational_order"] == "continuous" and cylinder["rotational_axis"] == "file z axis"
    tetra = ak.mesh_summary(meshes["files"][("tetra", "obj")])
    assert tetra["rotational_order"] == 1 and tetra["mirror_planes"] == 0
    assert tetra["mirror_symmetry"] < 0.5 and tetra["rotational_symmetry"] < 0.5
    assert "no clear rotational symmetry" in tetra["description"]


def test_open_mesh_reports_boundary_loops(meshes):
    trimesh = meshes["trimesh"]
    box = trimesh.creation.box(extents=(1, 1, 1))
    open_box = trimesh.Trimesh(box.vertices, box.faces[:-2])  # drop one square face
    s = ak.mesh_summary(open_box)
    assert not s["watertight"] and s["boundary_loops"] == 1
    assert s["volume"] is None and s["solidity"] is None
    assert s["genus_estimate"] == 0


def test_mesh_views_renders(meshes):
    pytest.importorskip("matplotlib")
    views = ak.mesh_views(meshes["files"][("tetra", "stl")], views=4, size=192)
    assert len(views) == 4
    arrays = [np.asarray(v, dtype=float) for v in views]
    for v, a in zip(views, arrays):
        assert v.size == (192, 192) and v.mode == "RGB"
        assert a[0, 0].tolist() == [255.0, 255.0, 255.0], "white background"
        assert (a.min(2) < 240).mean() > 0.05, "the object is visible"
        assert (a[[0, -1]].min(2) < 250).sum() == 0, "the object fits in the frame"
    for i in range(4):
        for j in range(i + 1, 4):
            assert np.abs(arrays[i] - arrays[j]).mean() > 1.0, f"views {i} and {j} of an asymmetric mesh differ"


def test_mesh_views_accepts_mesh_and_does_not_change_it(meshes):
    pytest.importorskip("matplotlib")
    mesh = meshes["shapes"]["slab"].copy()
    before = mesh.vertices.copy()
    views = ak.mesh_views(mesh, views=2, size=96)
    assert len(views) == 2 and views[0].size == (96, 96)
    assert np.array_equal(mesh.vertices, before)


def test_matplotlib_cache_dir_fallback(tmp_path, monkeypatch):
    import os
    import tempfile

    for name in ("MPLCONFIGDIR", "XDG_CACHE_HOME"):
        monkeypatch.setenv(name, "placeholder")  # records the original so teardown restores it
        monkeypatch.delenv(name)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    ak._matplotlib_cache_dir()
    shared = os.environ["MPLCONFIGDIR"]
    assert Path(shared).parent == tmp_path and Path(shared).is_dir(), "no cache under HOME: use a stable temp dir"
    monkeypatch.delenv("MPLCONFIGDIR")
    (home / ".matplotlib").mkdir()
    (home / ".matplotlib" / "fontlist-v390.json").write_text("{}")
    ak._matplotlib_cache_dir()
    assert "MPLCONFIGDIR" not in os.environ, "a cache under HOME is left for matplotlib to find"
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "chosen"))
    ak._matplotlib_cache_dir()
    assert os.environ["MPLCONFIGDIR"] == str(tmp_path / "chosen"), "an explicit choice is kept"


def test_mesh_views_decimates_big_meshes(meshes, tmp_path):
    pytest.importorskip("matplotlib")
    trimesh = meshes["trimesh"]
    big = trimesh.creation.icosphere(subdivisions=6)  # 81,920 faces
    v, f = ak._cluster_decimate(np.asarray(big.vertices), np.asarray(big.faces), 20_000)
    assert 1000 < len(f) <= 20_000 and f.max() < len(v)
    path = tmp_path / "big.glb"
    big.export(path)
    views = ak.mesh_views(path, views=2, size=128)
    assert len(views) == 2


# ============================================================================= molecules
@pytest.fixture
def chem():
    return pytest.importorskip("rdkit.Chem")


def test_ethyl_acetate_has_an_ester(chem):
    s = ak.molecule_summary("CCOC(C)=O")
    assert s["formula"] == "C4H8O2"
    assert s["functional_groups"] == {"ester": 1}
    assert s["molecular_weight"] == pytest.approx(88.11, abs=0.01)
    assert s["heavy_atoms_by_element"] == {"C": 4, "O": 2}
    json.dumps(s)


def test_ethanethiol_has_a_thiol(chem):
    s = ak.molecule_summary("CCS")
    assert s["functional_groups"] == {"thiol": 1}
    assert "thiol" in s["description"]


def test_vanillin_groups(chem):
    s = ak.molecule_summary("O=Cc1ccc(O)c(OC)c1")
    groups = s["functional_groups"]
    for name in ("aldehyde", "phenol", "ether", "aromatic_ring"):
        assert groups.get(name) == 1, name
    assert "ester" not in groups and "ketone" not in groups and "alcohol" not in groups
    assert s["aromatic_rings"] == 1 and s["h_bond_donors"] == 1
    assert s["canonical_smiles"] == chem.MolToSmiles(chem.MolFromSmiles("COc1cc(C=O)ccc1O"))


@pytest.mark.parametrize("smiles, group", [
    ("CC(=O)O", "carboxylic_acid"), ("CC(C)=O", "ketone"), ("CCO", "alcohol"), ("CN", "amine_primary"),
    ("Nc1ccccc1", "amine_primary"), ("CCNCC", "amine_secondary"), ("CCN(CC)CC", "amine_tertiary"),
    ("CC(N)=O", "amide"), ("CSC", "thioether"), ("CSSC", "disulfide"), ("CC#N", "nitrile"),
    ("[O-][N+](=O)c1ccccc1", "nitro"), ("ClC(Cl)Cl", "halide"), ("C=CC", "alkene"), ("CC#CC", "alkyne"),
    ("O=C1CCCO1", "lactone"), ("COC", "ether"),
])
def test_functional_groups(chem, smiles, group):
    assert group in ak.molecule_summary(smiles)["functional_groups"]


def test_amide_is_not_an_amine_and_ester_not_an_ether(chem):
    assert set(ak.molecule_summary("CC(=O)NC")["functional_groups"]) == {"amide"}
    assert "ether" not in ak.molecule_summary("CCOC(C)=O")["functional_groups"]


def test_terpene_like(chem):
    limonene = ak.molecule_summary("CC1=CCC(CC1)C(=C)C")
    assert limonene["terpene_like"] and limonene["functional_groups"]["isoprene_unit"] >= 2
    assert "terpene-like" in limonene["description"]
    assert not ak.molecule_summary("CCCCCCCCCC")["terpene_like"]


def test_read_molecule_formats(chem, tmp_path):
    vanillin = chem.MolFromSmiles("O=Cc1ccc(O)c(OC)c1")
    want = chem.MolToSmiles(vanillin)
    block = chem.MolToMolBlock(vanillin)
    (tmp_path / "v.mol").write_text(block)
    (tmp_path / "v.sdf").write_text(block + "$$$$\n" + chem.MolToMolBlock(chem.MolFromSmiles("CCO")) + "$$$$\n")
    (tmp_path / "v.smi").write_text("\n\nO=Cc1ccc(O)c(OC)c1 vanillin\nCCO ethanol\n")
    (tmp_path / "header.smi").write_text("SMILES Name\nO=Cc1ccc(O)c(OC)c1 vanillin\n")
    sources = [tmp_path / "v.mol", tmp_path / "v.sdf", tmp_path / "v.smi", tmp_path / "header.smi",
               str(tmp_path / "v.sdf"), block, "O=Cc1ccc(O)c(OC)c1", b"O=Cc1ccc(O)c(OC)c1"]
    if chem.inchi.INCHI_AVAILABLE:
        inchi = chem.MolToInchi(vanillin)
        (tmp_path / "v.inchi").write_text(inchi + "\n")
        sources += [inchi, tmp_path / "v.inchi"]
    for src in sources:
        assert chem.MolToSmiles(ak.read_molecule(src)) == want, src
    assert ak.molecule_summary(tmp_path / "v.sdf")["formula"] == "C8H8O3"


ETHANOL_MOL2 = """@<TRIPOS>MOLECULE
ethanol
 9 8 0 0 0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
      1 C1         -0.0100    1.0100    0.0100 C.3     1  UNL1        0.0000
      2 C2          1.5100    1.0100    0.0100 C.3     1  UNL1        0.0000
      3 O1          2.0000    2.3500    0.0100 O.3     1  UNL1        0.0000
      4 H1         -0.3800    0.0000    0.0100 H       1  UNL1        0.0000
      5 H2         -0.3800    1.5200    0.9000 H       1  UNL1        0.0000
      6 H3         -0.3800    1.5200   -0.8800 H       1  UNL1        0.0000
      7 H4          1.8800    0.5000    0.9000 H       1  UNL1        0.0000
      8 H5          1.8800    0.5000   -0.8800 H       1  UNL1        0.0000
      9 H6          2.9600    2.3500    0.0100 H       1  UNL1        0.0000
@<TRIPOS>BOND
     1     1     2    1
     2     2     3    1
     3     1     4    1
     4     1     5    1
     5     1     6    1
     6     2     7    1
     7     2     8    1
     8     3     9    1
"""


def test_read_molecule_mol2_and_pdb(chem, tmp_path):
    (tmp_path / "ethanol.mol2").write_text(ETHANOL_MOL2)
    assert chem.MolToSmiles(ak.read_molecule(tmp_path / "ethanol.mol2")) == "CCO"
    pdb = chem.MolToPDBBlock(chem.MolFromSmiles("CCS"))
    assert ak.molecule_summary(pdb)["functional_groups"] == {"thiol": 1}


def test_read_molecule_errors(chem, tmp_path):
    for bad in ["this is not a molecule", "", "   \n", "C1CC"]:
        with pytest.raises(ValueError):
            ak.read_molecule(bad)
    with pytest.raises(ValueError, match="no such file"):
        ak.read_molecule(str(tmp_path / "missing.sdf"))


def test_molecule_image(chem):
    try:
        img = ak.molecule_image("O=Cc1ccc(O)c(OC)c1", size=300)
    except ValueError as exc:
        pytest.skip(f"no rdkit drawing backend: {exc}")
    assert img.size == (300, 300) and img.mode == "RGB"
    a = np.asarray(img)
    assert (a.min(2) < 128).mean() > 0.005, "something is drawn"


# ============================================================================= coordinates
LONDON = (51.5074, -0.1278)


@pytest.mark.parametrize("text", [
    "51.5074, -0.1278", "51.5074 -0.1278", "51.5074\t-0.1278\n", "geo:51.5074,-0.1278",
    "lat=51.5074 lon=-0.1278", "Latitude: 51.5074, Longitude: -0.1278", "51.5074N 0.1278W",
    "name,lon,lat\nLondon,-0.1278,51.5074\n", "latitude;longitude\n51.5074;-0.1278",
])
def test_plain_coordinates(text):
    assert ak.read_coordinates(text) == [pytest.approx(LONDON)]


@pytest.mark.parametrize("text", [
    "51°30'26.6\"N 0°07'40.1\"W", "51°30′26.6″N, 0°07′40.1″W", "51 30 26.6 N 0 7 40.1 W",
    "N51 30.4433 W0 7.6683", "0°07'40.1\"W 51°30'26.6\"N", "51°30'26.6\", -0°07'40.1\"",
])
def test_dms_coordinates(text):
    (lat, lon), = ak.read_coordinates(text)
    assert lat == pytest.approx(51.50739, abs=1e-4) and lon == pytest.approx(-0.12781, abs=1e-4)


def test_dms_southern_and_eastern():
    assert ak.read_coordinates("33°51'54\"S 151°12'36\"E") == [pytest.approx((-33.865, 151.21))]


def test_geojson_is_lon_lat():
    point = {"type": "Point", "coordinates": [-0.1278, 51.5074]}
    assert ak.read_coordinates(json.dumps(point)) == [pytest.approx(LONDON)]
    # The trap: read lat-first this would be a valid point (0, 95); read correctly the latitude 95 is invalid.
    with pytest.raises(ValueError, match="latitude"):
        ak.read_coordinates(json.dumps({"type": "Point", "coordinates": [0, 95]}))
    collection = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": point},
        {"type": "Feature", "properties": {}, "geometry": {"type": "LineString",
                                                          "coordinates": [[2.35, 48.85], [2.36, 48.86, 35.0]]}},
        {"type": "Feature", "properties": {}, "geometry": {"type": "MultiPoint", "coordinates": [[10, 20]]}},
    ]}
    assert ak.read_coordinates(json.dumps(collection)) == [
        pytest.approx(LONDON), pytest.approx((48.85, 2.35)), pytest.approx((48.86, 2.36)), pytest.approx((20, 10))]
    feature = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [151.21, -33.87]}}
    assert ak.read_coordinates(json.dumps(feature)) == [pytest.approx((-33.87, 151.21))]


def test_gpx_and_kml_files(tmp_path):
    gpx = tmp_path / "walk.gpx"
    gpx.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">
  <wpt lat="51.5" lon="-0.12"><name>start</name></wpt>
  <rte><rtept lat="51.6" lon="-0.2"/></rte>
  <trk><trkseg><trkpt lat="51.51" lon="-0.13"><ele>12</ele></trkpt><trkpt lat="51.52" lon="-0.14"/></trkseg></trk>
</gpx>""")
    assert ak.read_coordinates(gpx) == [(51.5, -0.12), (51.6, -0.2), (51.51, -0.13), (51.52, -0.14)]
    kml = tmp_path / "places.kml"
    kml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <Placemark><Point><coordinates>-0.1278,51.5074,0</coordinates></Point></Placemark>
  <Placemark><LineString><coordinates>
    2.35,48.85,10 2.36,48.86
  </coordinates></LineString></Placemark>
</Document></kml>""")
    assert ak.read_coordinates(str(kml)) == [pytest.approx(LONDON), (48.85, 2.35), (48.86, 2.36)]


def test_coordinate_errors(tmp_path):
    for bad in ["91, 0", "0, 181", "", "no numbers here", '{"type": "Point"}', "<gpx><broken"]:
        with pytest.raises(ValueError):
            ak.read_coordinates(bad)
    with pytest.raises(ValueError, match="no such file"):
        ak.read_coordinates(str(tmp_path / "missing.gpx"))


def test_haversine():
    assert ak.haversine_km(51.5074, -0.1278, 48.8566, 2.3522) == pytest.approx(343.5, abs=1.5)
    assert ak.haversine_km(0, 0, 0, 0) == 0.0


GAZETTEER_ROWS = [
    # name, country (adm0name), iso_a2, admin1 (adm1name), population (pop_max), featurecla, lat, lon
    ("London", "United Kingdom", "GB", "Westminster", 8567000, "Admin-0 capital", 51.50194, -0.11867),
    ("Oxford", "United Kingdom", "GB", "Oxfordshire", 192796, "Populated place", 51.77042, -1.24999),
    ("Reading", "United Kingdom", "GB", "Reading", 369804, "Populated place", 51.47, -0.98),
    ("Paris", "France", "FR", "Île-de-France", 9904000, "Admin-0 capital", 48.86864, 2.33139),
    ("Sydney", "Australia", "AU", "New South Wales", 4630000, "Admin-1 capital", -33.92001, 151.18518),
    ("Singapore", "Singapore", "SG", None, 5183700, "Admin-0 capital", 1.29303, 103.85582),
    ("Pristina", "Kosovo", "-99", "Pristina", 465186, "Admin-0 capital", 42.66671, 21.16598),
]


def _feature(name, country, iso, admin1, pop, cla, lat, lon):
    # Natural Earth's simple places carry these fields among about thirty; the Point geometry is lon, lat.
    props = {"scalerank": 3, "featurecla": cla, "name": name, "nameascii": name, "adm0name": country,
             "iso_a2": iso, "adm1name": admin1, "latitude": lat, "longitude": lon, "pop_max": pop, "pop_min": pop}
    return {"type": "Feature", "properties": props, "geometry": {"type": "Point", "coordinates": [lon, lat]}}


def _write_gazetteer(path: Path, rows=GAZETTEER_ROWS) -> Path:
    collection = {"type": "FeatureCollection", "name": "ne_10m_populated_places_simple",
                  "features": [_feature(*row) for row in rows]}
    path.write_text(json.dumps(collection), encoding="utf-8")
    return path


@pytest.fixture
def gazetteer(tmp_path):
    return _write_gazetteer(tmp_path / "ne_10m_populated_places_simple.geojson")


def test_nearest_places(gazetteer):
    places = ak.nearest_places(51.72, -1.20, k=2, gazetteer=gazetteer)
    assert [p["name"] for p in places] == ["Oxford", "Reading"]
    oxford = places[0]
    assert oxford["country"] == "United Kingdom" and oxford["country_code"] == "GB"
    assert oxford["admin1"] == "Oxfordshire" and oxford["population"] == 192796
    assert oxford["feature_class"] == "Populated place"
    assert 5 < oxford["distance_km"] < 10
    assert oxford["bearing"] in ("S", "SSE", "SE"), "the query point lies south-east of Oxford"
    assert set(oxford) >= {"name", "country", "country_code", "population", "distance_km", "bearing"}
    assert len(ak.nearest_places(0, 0, k=50, gazetteer=gazetteer)) == len(GAZETTEER_ROWS)
    json.dumps(places)


def test_nearest_places_reads_geometry_and_cleans_codes(gazetteer):
    pristina = ak.nearest_places(42.67, 21.17, k=1, gazetteer=gazetteer)[0]
    assert pristina["name"] == "Pristina" and pristina["country"] == "Kosovo"
    assert pristina["country_code"] is None, "Natural Earth's -99 means no ISO code"
    assert ak.nearest_places(1.3, 103.86, k=1, gazetteer=gazetteer)[0]["admin1"] is None
    collection = json.loads(gazetteer.read_text())
    collection["features"][0]["geometry"]["coordinates"] = [10.0, 10.0]  # the geometry wins over the properties
    gazetteer.write_text(json.dumps(collection))
    moved = ak.nearest_places(10.0, 10.0, k=1, gazetteer=gazetteer)[0]
    assert moved["name"] == "London" and moved["distance_km"] == 0.0


def test_nearest_places_uses_env_and_reloads_changed_file(gazetteer, monkeypatch):
    monkeypatch.setenv("SENSE_GAZETTEER", str(gazetteer))
    assert ak.nearest_places(-33.9, 151.2, k=1)[0]["name"] == "Sydney"
    extra = ("Bondi Beachside", "Australia", "AU", "New South Wales", 20000, "Populated place", -33.89, 151.27)
    _write_gazetteer(gazetteer, GAZETTEER_ROWS + [extra])
    assert ak.nearest_places(-33.89, 151.27, k=1)[0]["name"] == "Bondi Beachside"


def test_nearest_places_missing_or_wrong_gazetteer(tmp_path):
    with pytest.raises(ValueError, match="fetch_gazetteer"):
        ak.nearest_places(51.5, -0.1)
    with pytest.raises(ValueError, match="fetch_gazetteer"):
        ak.nearest_places(51.5, -0.1, gazetteer=tmp_path / "nope.geojson")
    geonames_style = tmp_path / "cities15000.txt"
    geonames_style.write_text("2643743\tLondon\tLondon\t\t51.50853\t-0.12574\tP\tPPLC\tGB\n")
    with pytest.raises(ValueError, match="Natural Earth"):
        ak.nearest_places(51.5, -0.1, gazetteer=geonames_style)
    with pytest.raises(ValueError):
        ak.nearest_places(123, 0, gazetteer=tmp_path / "nope.geojson")


def test_coordinate_summary(gazetteer, monkeypatch):
    monkeypatch.setenv("SENSE_GAZETTEER", str(gazetteer))
    s = ak.coordinate_summary(*LONDON)
    assert s["hemisphere"] == "northern, western" and s["climate_band"] == "temperate"
    assert s["nearest_places"][0]["name"] == "London"
    assert "London (Westminster, United Kingdom)" in s["description"] and s["dms"].startswith("51°30'")
    assert "Natural Earth" in s["description"]
    json.dumps(s)
    assert ak.coordinate_summary(1.3, 103.8)["climate_band"] == "tropical"
    assert ak.coordinate_summary(78.2, 15.6)["climate_band"] == "polar"
    assert ak.coordinate_summary(-33.9, 151.2)["hemisphere"] == "southern, eastern"
    assert "remote" in ak.coordinate_summary(0.0, -140.0)["description"]


def test_coordinate_summary_without_gazetteer():
    s = ak.coordinate_summary(10.0, 10.0)
    assert s["nearest_places"] is None and "fetch_gazetteer" in s["gazetteer_note"]
    assert "tropical" in s["description"]


# ============================================================================= scripts/fetch_gazetteer.py
def _load_fetch_script():
    spec = importlib.util.spec_from_file_location("fetch_gazetteer", ROOT / "scripts" / "fetch_gazetteer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fetch_gazetteer_offline_and_idempotent(tmp_path, monkeypatch, capsys):
    fg = _load_fetch_script()
    fixture = _write_gazetteer(tmp_path / "fixture.geojson")
    calls = []

    def fake_download(url, target, timeout=120.0):
        calls.append(url)
        shutil.copy(fixture, target)
        return fixture.stat().st_size

    monkeypatch.setattr(fg, "_download", fake_download)
    monkeypatch.setattr(fg, "MIN_FEATURES", 3)  # the real file has about 7,300
    dest = tmp_path / "gaz"
    assert fg.main(["--dest", str(dest)]) == 0
    final = dest / "ne_10m_populated_places_simple.geojson"
    assert final.read_bytes() == fixture.read_bytes()
    source = (dest / "SOURCE.txt").read_text()
    assert fg.URL in source and "public domain" in source and f"features: {len(GAZETTEER_ROWS)}" in source
    assert "GeoNames" in source and "0 km" in source, "SOURCE.txt says why the gazetteer is not GeoNames"
    assert fg.main(["--dest", str(dest)]) == 0 and len(calls) == 1, "second run does nothing"
    assert "already present" in capsys.readouterr().out
    assert fg.main(["--dest", str(dest), "--force"]) == 0 and len(calls) == 2
    assert ak.nearest_places(51.5, -0.12, k=1, gazetteer=final)[0]["name"] == "London"


def test_fetch_gazetteer_rejects_wrong_data(tmp_path, monkeypatch, capsys):
    fg = _load_fetch_script()
    geonames_zip = tmp_path / "cities15000.zip"
    with zipfile.ZipFile(geonames_zip, "w") as z:
        z.writestr("cities15000.txt", "2643743\tLondon\tLondon\t\t51.50853\t-0.12574\n")
    monkeypatch.setattr(fg, "_download", lambda url, target, timeout=120.0: shutil.copy(geonames_zip, target))
    dest = tmp_path / "gaz"
    assert fg.main(["--dest", str(dest)]) == 1
    assert "not valid GeoJSON" in capsys.readouterr().err
    assert not (dest / "ne_10m_populated_places_simple.geojson").exists() and not (dest / "SOURCE.txt").exists()
    assert [p.name for p in dest.iterdir()] == [], "no temporary file is left behind"


def test_fetch_gazetteer_help():
    proc = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "fetch_gazetteer.py"), "--help"],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0
    text = " ".join(proc.stdout.split())  # argparse wraps lines
    assert "Natural Earth" in text and "public domain" in text and "--dest" in text and "GeoNames" in text


def test_math_helpers_are_plain_python():
    out = ak._plain({"a": np.float32(1.5), "b": np.int64(2), "c": np.array([1, 2]), "d": np.bool_(True),
                     "e": float("nan")})
    assert out == {"a": 1.5, "b": 2, "c": [1, 2], "d": True, "e": None}
    assert not math.isnan(json.loads(json.dumps(out))["a"])
