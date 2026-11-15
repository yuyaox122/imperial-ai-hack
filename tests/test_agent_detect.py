"""Modality detection: magic bytes first, then a look at the content.

Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_agent_detect.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from agent import detect as detect_module
from agent.detect import detect, is_text

CASES = [
    (b"RIFF\x24\x00\x00\x00WAVEfmt ", "clip.wav", "audio", "wav"),
    (b"fLaC\x00\x00\x00\x22", "a.flac", "audio", "flac"),
    (b"OggS\x00\x02", "a.ogg", "audio", "ogg"),
    (b"ID3\x04\x00\x00", "song.mp3", "audio", "mp3"),
    (b"\xff\xfb\x90\x64", "song.mp3", "audio", "mp3"),
    (b"\x89PNG\r\n\x1a\n", "x.png", "image", "png"),
    (b"\xff\xd8\xff\xe0", "x.jpg", "image", "jpeg"),
    (b"hello world\n", "notes.txt", "text", "text"),
    (b"t,ax,ay,az\n0,0.1,0.2,9.8\n", "walk.csv", "other", "csv"),
    (b">seq1\nACGTACGT\n", "gene.fasta", "other", "fasta"),
    (b"\x00\x01\x02\x03\xfe\xfd", "blob.bin", "other", "unknown"),
]


@pytest.mark.parametrize("data,name,modality,fmt", CASES)
def test_detect(tmp_path, data, name, modality, fmt):
    p = tmp_path / name
    p.write_bytes(data)
    d = detect(p)
    assert (d.modality, d.format) == (modality, fmt) and d.detail


def test_bytes_win_over_the_extension(tmp_path):
    p = tmp_path / "really_mp3.wav"
    p.write_bytes(b"ID3\x04\x00\x00")
    assert detect(p).format == "mp3" and detect(p).how == "magic bytes"


@pytest.mark.parametrize("make", ["missing", "directory", "empty"])
def test_unreadable_input_is_a_clear_error(tmp_path, make):
    p = tmp_path / "x.wav"
    if make == "directory":
        p.mkdir()
    if make == "empty":
        p.write_bytes(b"")
    with pytest.raises(ValueError, match="x.wav"):
        detect(p)


# ----------------------------------------------------------------------------- fix round 1
@pytest.mark.parametrize("codec,bom", [("utf-16-le", b"\xff\xfe"), ("utf-16-be", b"\xfe\xff")])
def test_utf16_text_with_a_byte_order_mark_is_text(tmp_path, codec, bom):
    p = tmp_path / "notes.txt"
    p.write_bytes(bom + "hello world\nsecond line\n".encode(codec))
    d = detect(p)
    assert (d.modality, d.format, d.how) == ("text", "text", "content") and "UTF-16" in d.detail


def test_utf16le_csv_is_csv(tmp_path):
    p = tmp_path / "walk.csv"
    p.write_bytes(b"\xff\xfe" + "t,ax,ay,az\n0,0.1,0.2,9.8\n".encode("utf-16-le"))
    assert (detect(p).modality, detect(p).format) == ("other", "csv")


def test_a_note_starting_with_the_letters_id3_is_text(tmp_path):
    p = tmp_path / "note.txt"
    p.write_bytes(b"ID3 tags are stored at the start of an MP3 file.\n")
    assert (detect(p).modality, detect(p).format) == ("text", "text")


def test_ff_fe_followed_by_bytes_that_are_not_utf16_text_is_not_mp3(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"\xff\xfe\x00\xdc\x00\xdc\x12\x34")  # lone low surrogates: not valid UTF-16
    d = detect(p)
    assert d.format != "mp3" and (d.modality, d.format) == ("other", "unknown")


def test_aac_adts_sync_is_not_reported_as_mp3(tmp_path):
    p = tmp_path / "a.aac"
    p.write_bytes(b"\xff\xf1\x50\x80\x02\x1f\xfc")
    assert detect(p).format != "mp3"


@pytest.mark.parametrize("second", [0xFB, 0xFA, 0xF3, 0xF2, 0xE3])
def test_every_layer3_frame_sync_is_mp3(tmp_path, second):
    p = tmp_path / "song.bin"
    p.write_bytes(bytes([0xFF, second, 0x90, 0x64]))
    assert detect(p).format == "mp3"


def test_fasta_with_a_utf8_byte_order_mark_is_fasta(tmp_path):
    p = tmp_path / "gene.fasta"
    p.write_bytes(b"\xef\xbb\xbf>seq1\nACGTACGT\n")
    assert detect(p).format == "fasta"


def test_a_short_file_ending_in_a_cut_off_character_is_unknown(tmp_path):
    p = tmp_path / "one.bin"
    p.write_bytes(b"\xc3")
    assert (detect(p).modality, detect(p).format) == ("other", "unknown")


def test_a_character_cut_off_by_the_read_window_is_still_text(tmp_path):
    p = tmp_path / "long.txt"
    p.write_bytes(("é" * 3000).encode())  # byte 4,096 falls inside a two-byte character
    assert detect(p).format == "text"


# ----------------------------------------------------------------------------- release: every format the specialist reads
@pytest.mark.parametrize("sf_format,subtype,fmt", [
    ("AIFF", "PCM_16", "aiff"), ("AIFF", "FLOAT", "aiff"),  # FLOAT is written as AIFC
    ("AU", "PCM_16", "au"), ("RF64", "PCM_16", "wav"), ("WAV", "PCM_16", "wav"),
])
def test_audio_the_specialist_reads_is_audio(tmp_path, sf_format, subtype, fmt):
    sf = pytest.importorskip("soundfile")
    import numpy as np
    p = tmp_path / "clip.bin"  # the extension is never consulted
    sf.write(p, np.zeros(800, dtype=np.float32), 8000, format=sf_format, subtype=subtype)
    d = detect(p)
    assert (d.modality, d.format, d.how) == ("audio", fmt, "magic bytes")


def test_text_that_starts_like_an_au_header_stays_text(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_bytes(b".snd files are Sun audio; this note is plain text.\n")
    assert (detect(p).modality, detect(p).format) == ("text", "text")


# ----------------------------------------------------------------------------- video, 3d, molecule, geo
FFMPEG = shutil.which("ffmpeg") or next((p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg")
                                         if Path(p).is_file()), None)


def _ffmpeg(out: Path, *args: str) -> Path:
    if FFMPEG is None:
        pytest.skip("ffmpeg is not installed")
    done = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args, str(out)],
                          capture_output=True, timeout=60)
    if done.returncode != 0 or not out.is_file():
        pytest.skip(f"this ffmpeg cannot write {out.name}: {done.stderr.decode(errors='replace')[-200:]}")
    return out


TESTSRC = ["-f", "lavfi", "-i", "testsrc=duration=0.3:size=32x32:rate=10"]


@pytest.mark.parametrize("name,args,fmt", [
    ("clip.mp4", ["-c:v", "mpeg4"], "mp4"),
    ("clip.mov", ["-c:v", "mjpeg", "-f", "mov"], "mov"),
    ("clip.mkv", ["-c:v", "mpeg4"], "mkv"),
    ("clip.webm", ["-c:v", "libvpx"], "webm"),
    ("clip.avi", ["-c:v", "mpeg4"], "avi"),
    ("clip.gif", [], "gif"),
    ("clip.3gp", ["-c:v", "mpeg4", "-f", "3gp"], "mp4"),
])
def test_video_from_ffmpeg_is_video(tmp_path, name, args, fmt):
    made = _ffmpeg(tmp_path / name, *TESTSRC, *args)
    renamed = made.rename(tmp_path / "upload.bin")  # the extension is never consulted
    d = detect(renamed)
    assert (d.modality, d.format, d.how) == ("video", fmt, "magic bytes"), d
    assert d.detail.startswith("upload.bin starts with the bytes of")


def test_an_m4a_stays_what_it_was(tmp_path):
    made = _ffmpeg(tmp_path / "voice.m4a", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.3", "-c:a", "aac")
    assert made.read_bytes()[8:12] == b"M4A "
    assert (detect(made).modality, detect(made).format) == ("other", "unknown")


def test_a_gif_with_one_frame_is_an_image(tmp_path):
    from PIL import Image

    still = tmp_path / "still.gif"
    Image.new("RGB", (16, 16), "red").save(still)
    assert (detect(still).modality, detect(still).format, detect(still).how) == ("image", "gif", "magic bytes")
    frames = [Image.new("RGB", (16, 16), c) for c in ("red", "green", "blue")]
    moving = tmp_path / "moving.gif"
    frames[0].save(moving, save_all=True, append_images=frames[1:], duration=100)
    assert (detect(moving).modality, detect(moving).format) == ("video", "gif")
    assert detect_module._gif_frames(moving, enough=10) == 3 and detect_module._gif_frames(still, enough=10) == 1


@pytest.fixture(scope="module")
def box():
    trimesh = pytest.importorskip("trimesh")
    return trimesh.creation.box(extents=(1, 2, 3))


@pytest.mark.parametrize("file_type,fmt,how", [
    ("glb", "glb", "magic bytes"), ("ply", "ply", "content"), ("off", "off", "content"),
    ("stl", "stl", "content"), ("stl_ascii", "stl", "content"), ("obj", "obj", "content"),
])
def test_meshes_are_3d(tmp_path, box, file_type, fmt, how):
    p = tmp_path / "shape.bin"
    data = box.export(file_type=file_type)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    d = detect(p)
    assert (d.modality, d.format, d.how) == ("3d", fmt, how), d


def test_ascii_ply_gltf_json_and_a_large_obj_are_3d(tmp_path, box):
    ply = tmp_path / "a.ply"
    ply.write_bytes(box.export(file_type="ply", encoding="ascii"))
    assert (detect(ply).modality, detect(ply).format) == ("3d", "ply") and "format ascii" in detect(ply).detail

    gltf = tmp_path / "scene.json"
    gltf.write_text(json.dumps({"asset": {"version": "2.0", "generator": "test"}, "scenes": [{"nodes": [0]}],
                                "nodes": [{"mesh": 0}]}, indent=2))
    assert (detect(gltf).modality, detect(gltf).format) == ("3d", "gltf")

    big = tmp_path / "big.obj"  # every vertex first: the faces start past the first 4,096 bytes
    big.write_text("# many vertices\n" + "".join(f"v {k} {k % 7} {k % 3}\n" for k in range(2000)) + "f 1 2 3\n")
    assert (detect(big).modality, detect(big).format) == ("3d", "obj")
    points = tmp_path / "points.obj"
    points.write_text("".join(f"v {k} 0 0\n" for k in range(10)))  # vertices without a single face
    assert detect(points).modality == "text"


def test_molecules(tmp_path):
    Chem = pytest.importorskip("rdkit.Chem")
    vanillin = Chem.MolFromSmiles("COc1cc(C=O)ccc1O")
    cases = {
        "a.mol": (Chem.MolToMolBlock(vanillin), "mol"),
        "a.sdf": (Chem.MolToMolBlock(vanillin) + "$$$$\n" + Chem.MolToMolBlock(Chem.MolFromSmiles("CCO")) + "$$$$\n",
                  "sdf"),
        "v3000.mol": (Chem.MolToV3KMolBlock(vanillin), "mol"),
        "a.inchi": (Chem.MolToInchi(vanillin) + "\n", "inchi"),
        "a.smi": ("COc1cc(C=O)ccc1O\n", "smiles"),
        "few.smi": ("CCO\nc1ccccc1\nCN(C)C\n", "smiles"),
        "salt.smi": ("[Na+].[Cl-]\n", None),  # no organic-subset atom outside brackets
    }
    for name, (content, fmt) in cases.items():
        p = tmp_path / name
        p.write_text(content)
        d = detect(p)
        if fmt is None:
            assert d.modality == "text", (name, d)
        else:
            assert (d.modality, d.format, d.how) == ("molecule", fmt, "content"), (name, d)


@pytest.mark.parametrize("content", [
    "Hello\n", "Thanks, see you soon.\n", "CCCCC1\n", "C(C\n", "OK\n", "NO WAY\n",
    "The molecule is CCO, also called ethanol.\n",
])
def test_words_and_prose_are_not_smiles(tmp_path, content):
    p = tmp_path / "note.txt"
    p.write_text(content)
    assert detect(p).format != "smiles"


def test_places(tmp_path):
    point = {"type": "Point", "coordinates": [-0.1276, 51.5072]}
    cases = {
        "point.json": (json.dumps(point), "geojson"),
        "fc.json": (json.dumps({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"name": "x"}, "geometry": point}]}, indent=1), "geojson"),
        "track.gpx": ('<?xml version="1.0"?>\n<gpx version="1.1" creator="t"><trk><trkseg>'
                      '<trkpt lat="51.5" lon="-0.12"/></trkseg></trk></gpx>\n', "gpx"),
        "doc.kml": ('<?xml version="1.0" encoding="UTF-8"?>\n<!-- made by hand -->\n'
                    '<kml xmlns="http://www.opengis.net/kml/2.2"><Placemark><Point><coordinates>-0.12,51.5'
                    '</coordinates></Point></Placemark></kml>\n', "kml"),
        "one.txt": ("51.3608, 7.4717\n", "latlon"),
        "spaces.txt": ("33.4484 -112.0740\n", "latlon"),
        "letters.txt": ("51.5072° N, 0.1276° W\n", "latlon"),
        "lon_first.txt": ("E 7.47 N 51.36\n", "latlon"),
        "three.txt": ("51.36, 7.47\n27.25, 33.81\n-33.86; 151.21\n", "latlon"),
    }
    for name, (content, fmt) in cases.items():
        p = tmp_path / name
        p.write_text(content)
        d = detect(p)
        assert (d.modality, d.format, d.how) == ("geo", fmt, "content"), (name, d)
    assert "(51.3608, 7.4717)" in detect(tmp_path / "one.txt").detail
    assert "(51.36, 7.47)" in detect(tmp_path / "lon_first.txt").detail
    assert "(51.5072, -0.1276)" in detect(tmp_path / "letters.txt").detail


@pytest.mark.parametrize("content,expected", [
    ("12 34\n", ("text", "text")),  # two whole numbers are not a place
    ("95.5, 10.2\n", ("text", "text")),  # no latitude is 95.5
    ("51.5 N 0.1 N\n", ("text", "text")),  # two latitudes
    ("-51.5 S, 0.1 W\n", ("text", "text")),  # a sign and a hemisphere letter
    ('{"type": "Point", "name": "no coordinates"}\n', ("text", "text")),
    ('{"type": "Point", "coordinates": [1, 2], "x": 1}\n', ("geo", "geojson")),
    ('{"name": "a", "b": 1, "c": 2}\n', ("other", "csv")),  # JSON that is not GeoJSON: as before
    ("<html><body>gpx files hold tracks</body></html>\n", ("text", "text")),
])
def test_what_is_not_a_place_stays_as_before(tmp_path, content, expected):
    p = tmp_path / "x.txt"
    p.write_text(content)
    assert (detect(p).modality, detect(p).format) == expected


def test_a_two_column_numeric_csv_of_many_rows_is_csv(tmp_path):
    p = tmp_path / "table.csv"
    p.write_text("lat,lon\n" + "".join(f"{10 + k * 0.01:.4f},{20 + k * 0.02:.4f}\n" for k in range(500)))
    assert (detect(p).modality, detect(p).format) == ("other", "csv")
    headerless = tmp_path / "values.csv"
    headerless.write_text("".join(f"{k}.5,{k * 2}.25\n" for k in range(20)))
    assert (detect(headerless).modality, detect(headerless).format) == ("other", "csv")


def test_prose_stays_text(tmp_path):
    p = tmp_path / "story.txt"
    p.write_text("It was a bright cold day in April, and the clocks were striking thirteen.\n" * 3)
    assert (detect(p).modality, detect(p).format) == ("text", "text")
    names = tmp_path / "names.txt"
    names.write_text("Smith, Anna\nJones, Bob\nBrown, Carl\n")  # one comma per line, but not numbers
    assert (detect(names).modality, detect(names).format) == ("text", "text")


def test_is_text(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("51.5, -0.12\n")
    b = tmp_path / "b.bin"
    b.write_bytes(bytes(range(256)))
    assert is_text(p) and not is_text(b)
