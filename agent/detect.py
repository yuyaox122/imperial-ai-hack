"""Work out what kind of file the agent was given, from its bytes rather than its name.

Public surface (do not rename):
  Detection                  modality, format, how, detail; to_dict() for the trace
  detect(path) -> Detection  ValueError (naming the path) for a missing path, a directory or an empty file
  is_text(path) -> bool      whether the file's first 4,096 bytes read as text, the way detect reads them

Modalities and their formats:
  audio     wav, aiff, au, flac, ogg, mp3
  image     png, jpeg, gif (a GIF with one frame)
  video     mp4, mov, webm, mkv, avi, gif (an animated GIF)
  3d        glb, gltf, ply, off, stl, obj
  molecule  mol, sdf, inchi, smiles
  geo       geojson, gpx, kml, latlon
  text      text
  other     csv, fasta, unknown
`how` is "magic bytes" when a signature at a fixed place decided, else "content". The file extension is never
consulted.

Magic bytes, in this order:
  wav (RIFF....WAVE, or RF64....WAVE for files over 4 GB), aiff (FORM....AIFF or FORM....AIFC), au (".snd" with a
  big-endian encoding code 1-27 at byte 12, so text that starts ".snd" stays text), flac (fLaC), ogg (OggS), png,
  jpeg, and mp3: an ID3v2.2-2.4 tag ("ID3" then version byte 2, 3 or 4) or an MPEG Layer III frame sync (FF, then a
  byte with the top three bits set, layer bits 01 and a version that is not reserved: FF FB, FF FA, FF F3, FF F2,
  FF E3 and the like; not FF FE or AAC FF F1).
  video, ISO base media: "ftyp" at byte 4. It is video when its major brand is a video brand (isom, iso2 to iso6,
  mp41, mp42, avc1, "M4V ", M4VH, M4VP, "qt  ", mmp4, MSNV, "f4v ", 3gp*, 3g2*), or when the major brand is not
  a still-image brand (heic, mif1, avif and the like) and a compatible brand listed in the ftyp box is a video
  brand; mov for the QuickTime brand "qt  ", else mp4. An ftyp box naming "M4A ", "M4B " or "M4P " (audio and
  audiobooks) is not video and stays what it was before video was known: other, unknown.
  video, Matroska: the EBML magic 1A 45 DF A3 with DocType "webm" (webm) or "matroska" (mkv). An audio-only WebM or
  Matroska file reads as video too: the tracks are not parsed.
  video, AVI: RIFF....AVI .
  GIF87a or GIF89a: its blocks are walked and its image descriptors counted; more than one is an animated GIF
  (video, gif), one is a still picture (image, gif).
  3d, GLB: "glTF" followed by version 1 or 2.
Then, from the structure of the bytes (how "content"):
  stl (binary): an 80-byte header, then a little-endian uint32 triangle count, with a file size of exactly
  84 + 50 * count (count at least 1). Checked before any text test: an STL header often starts with "solid".
  ply: the first line is "ply" and the second starts "format " (ASCII or binary PLY).
Otherwise the first 4,096 bytes are read as text: UTF-16 when they start with a byte-order mark (FF FE or FE FF),
else UTF-8 (a UTF-8 byte-order mark is skipped). A character cut off at the end is tolerated only when the
4,096-byte window was filled. Text containing NUL is not text, and a file that is not text is other, unknown.
Text is tested in this order:
  geojson: a JSON object (the whole file is parsed when it is at most 32 MB; a larger one is matched by its first
    bytes) whose "type" is Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon or
    GeometryCollection with coordinates, or a Feature or FeatureCollection with at least one such geometry.
  gltf: a JSON object with {"asset": {"version": ...}}.
  gpx, kml: XML whose root element is <gpx> or <kml> (after any XML declaration, comments and doctype).
  mol, sdf: a counts line (digits first, ending V2000 or V3000) in the first 8 lines; sdf when a "$$$$" line
    ends a record (looked for in the first MB), else mol. Without a counts line there, "$$$$" and "M  END" lines
    together still make sdf.
  inchi: the first non-empty line starts "InChI=".
  off: the first line that is not a comment is OFF (or COFF, NOFF, CNOFF, STOFF), with the three vertex, face
    and edge counts on that line or the next.
  stl (ASCII): the first line starts "solid" and a "facet normal" line follows within the first 2,048 characters.
  obj: more than half of the lines that are not comments start with an OBJ keyword (v, vn, vt, vp, f, o, g, s, l,
    p, usemtl, mtllib), with at least 3 "v" lines (the first with three numbers) and at least one "f" line
    (looked for in the first 64 MB when the first 4,096 bytes hold none).
  fasta: a first line starting with ">" (as before).
  csv: a first line with two or more commas (as before).
  latlon: 1 to 5 non-empty lines, the whole file, each holding two numbers that are a latitude (-90 to 90) and a
    longitude (-180 to 180), separated by a comma, a semicolon or spaces, each optionally with a degree sign, and
    with N/S/E/W letters either after the numbers ("51.5° N, 0.1° W") or before both ("N 51.5 W 0.1"). A letter
    takes an unsigned number; S and W make it negative; E/W first means longitude first; without letters,
    latitude comes first. Each line needs a decimal point, a letter or a degree sign, so "12 34" is not a place.
  csv: at least 3 non-empty lines that all hold the same number (1 or more) of commas, with a number in every
    field of every line after the first (a numeric table of one comma per line, which the first-line rule
    misses).
  smiles: 1 to 10 non-empty lines, the whole file, each one token of the SMILES alphabet (organic-subset atoms,
    aromatic atoms, bracket atoms, bonds, branches and ring closures), of at least 2 characters, with at least one
    organic-subset atom outside brackets, balanced parentheses and ring-closure digits in pairs; when rdkit can be
    imported, the first line must parse as a molecule.
  text: anything else.
"""

from __future__ import annotations

import codecs
import importlib.util
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

HEAD_BYTES = 64  # enough for every fixed-place signature of the audio and image formats
TEXT_BYTES = 4096  # how much is decoded to decide whether the file is text
JSON_MAX_BYTES = 32 * 1024 * 1024  # a JSON file up to this size is parsed whole
SCAN_BYTES = 64 * 1024 * 1024  # how far an OBJ file is searched for its first face
SDF_SCAN_BYTES = 1024 * 1024  # how far a MOL/SDF file is searched for a "$$$$" record end
MAX_LATLON_LINES = 5
MAX_SMILES_LINES = 10

_SIGNATURES = [  # (prefix, modality, format, name used in the detail sentence)
    (b"fLaC", "audio", "flac", "a FLAC"),
    (b"OggS", "audio", "ogg", "an Ogg"),
    (b"\x89PNG\r\n\x1a\n", "image", "png", "a PNG"),
    (b"\xff\xd8\xff", "image", "jpeg", "a JPEG"),
]
_VIDEO_BRANDS = {
    b"isom", b"iso2", b"iso3", b"iso4", b"iso5", b"iso6", b"mp41", b"mp42", b"avc1", b"M4V ", b"M4VH", b"M4VP",
    b"qt  ", b"mmp4", b"MSNV", b"f4v ",
}
_AUDIO_BRANDS = {b"M4A ", b"M4B ", b"M4P "}
_STILL_BRANDS = {b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx", b"mif1", b"msf1", b"avif", b"avis", b"crx ",
                 b"jp2 ", b"jpx ", b"jpm "}
_GEOMETRIES = {"Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon"}
_GEOJSON_TYPES = _GEOMETRIES | {"GeometryCollection", "Feature", "FeatureCollection"}
_OFF_HEADS = {"OFF", "COFF", "NOFF", "CNOFF", "STOFF"}
_OBJ_KEYS = {"v", "vn", "vt", "vp", "f", "o", "g", "s", "l", "p", "usemtl", "mtllib"}
_NUMBER = re.compile(r"^\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\s*$")
_DEGREES = r"[+-]?\d{1,3}(?:\.\d+)?"
_SEP = r"\s*(?:[,;]|\s)\s*"
_LATLON = [  # "51.5 N, 0.1 W" (letters after the numbers, or none) and "N 51.5 W 0.1" (letters before)
    re.compile(rf"^\s*(?P<an>{_DEGREES})\s*(?P<ad>°)?\s*(?P<al>[NSEW])?{_SEP}"
               rf"(?P<bn>{_DEGREES})\s*(?P<bd>°)?\s*(?P<bl>[NSEW])?\s*$"),
    re.compile(rf"^\s*(?P<al>[NSEW])\s*(?P<an>{_DEGREES})\s*(?P<ad>°)?{_SEP}"
               rf"(?P<bl>[NSEW])\s*(?P<bn>{_DEGREES})\s*(?P<bd>°)?\s*$"),
]
_SMILES = re.compile(r"(?:Cl|Br|[BCNOPSFI]|[bcnops]|\[[^\[\]\s]{1,20}\]|%\d{2}|\d|[-=#$:/\\().*])+")
_ORGANIC = re.compile(r"Cl|Br|[BCNOPSFIbcnops]")


@dataclass(frozen=True)
class Detection:
    modality: str  # "audio" | "image" | "video" | "3d" | "molecule" | "geo" | "text" | "other"
    format: str  # see the module docstring: "wav" ... "mp4" ... "obj" ... "smiles" ... "latlon" ... "unknown"
    how: str  # "magic bytes" | "content"
    detail: str  # one sentence for the trace

    def to_dict(self) -> dict:
        return asdict(self)


def detect(path: Path) -> Detection:
    path = Path(path)
    if not path.exists():
        raise ValueError(f"{path} does not exist; give the path of a file to detect.")
    if not path.is_file():
        raise ValueError(f"{path} is not a file; give a single file, not a directory.")
    with path.open("rb") as f:
        head = f.read(TEXT_BYTES)
    if not head:
        raise ValueError(f"{path} is empty; give a file with some content.")
    return (
        _by_magic(path, head)
        or _by_structure(path, head)
        or _by_content(path, head, filled=len(head) == TEXT_BYTES)
    )


def is_text(path: Path) -> bool:
    """Whether the first TEXT_BYTES of the file read as text (UTF-8, or UTF-16 after a byte-order mark)."""
    with Path(path).open("rb") as f:
        head = f.read(TEXT_BYTES)
    return bool(head) and _as_text(head, filled=len(head) == TEXT_BYTES)[0] is not None


# ----------------------------------------------------------------------------- magic bytes
def _by_magic(path: Path, data: bytes) -> Detection | None:
    head = data[:HEAD_BYTES]
    if head[:4] in (b"RIFF", b"RF64") and head[8:12] == b"WAVE":
        return _magic(path, "audio", "wav", "a WAV")
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return _magic(path, "video", "avi", "an AVI")
    if head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
        return _magic(path, "audio", "aiff", "an AIFF")
    if head[:4] == b".snd" and len(head) >= 16 and 1 <= int.from_bytes(head[12:16], "big") <= 27:
        return _magic(path, "audio", "au", "a Sun AU")
    for prefix, modality, fmt, name in _SIGNATURES:
        if head.startswith(prefix):
            return _magic(path, modality, fmt, name)
    if head[:3] == b"ID3" and len(head) > 3 and head[3] in (2, 3, 4):
        return _magic(path, "audio", "mp3", "an MP3 (ID3 tag)")
    if len(head) >= 2 and head[0] == 0xFF and _is_layer3_sync(head[1]):
        return _magic(path, "audio", "mp3", "an MP3 (MPEG frame sync)")
    if head[4:8] == b"ftyp":
        return _iso_media(path, data)
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return _matroska(path, data)
    if head[:6] in (b"GIF87a", b"GIF89a"):
        frames = _gif_frames(path)
        if frames > 1:
            return _magic(path, "video", "gif", "an animated GIF (two or more frames)")
        return _magic(path, "image", "gif", "a GIF with a single frame")
    if head[:4] == b"glTF" and len(head) >= 8 and int.from_bytes(head[4:8], "little") in (1, 2):
        return _magic(path, "3d", "glb", "a binary glTF (GLB)")
    return None


def _is_layer3_sync(b1: int) -> bool:
    """Second byte of an MPEG audio frame header: sync bits set, layer III (01), version not reserved (01)."""
    return b1 & 0xE0 == 0xE0 and b1 & 0x06 == 0x02 and b1 & 0x18 != 0x08


def _magic(path: Path, modality: str, fmt: str, name: str) -> Detection:
    detail = f"{path.name} starts with the bytes of {name} file, so it is {modality}."
    return Detection(modality, fmt, "magic bytes", detail)


def _video_brand(brand: bytes) -> bool:
    return brand in _VIDEO_BRANDS or brand[:3] in (b"3gp", b"3g2")


def _iso_media(path: Path, data: bytes) -> Detection | None:
    """An ISO base media file (MP4, MOV, 3GP, M4A, HEIC...): video only for a video brand, see the docstring."""
    if len(data) < 16:
        return None
    major = data[8:12]
    size = int.from_bytes(data[:4], "big")
    end = min(len(data), size if 16 <= size <= len(data) else 64)
    compatible = [data[k : k + 4] for k in range(16, end - 3, 4)]
    if major in _AUDIO_BRANDS or any(b in _AUDIO_BRANDS for b in compatible):
        return None  # M4A, M4B, M4P: audio in an MP4 box, left as it was
    if _video_brand(major) or (major not in _STILL_BRANDS and any(_video_brand(b) for b in compatible)):
        brand = major.decode("latin-1").strip() or "?"
        if major == b"qt  ":
            return _magic(path, "video", "mov", "a QuickTime MOV (ISO media, brand 'qt')")
        return _magic(path, "video", "mp4", f"an MP4 (ISO media, brand '{brand}')")
    return None


def _ebml_size(data: bytes, k: int) -> tuple[int, int] | None:
    """An EBML variable-length size at data[k]: (value, bytes used), or None."""
    if k >= len(data) or data[k] == 0:
        return None
    first = data[k]
    length = 8 - first.bit_length() + 1
    if k + length > len(data):
        return None
    value = first & (0xFF >> length)
    for b in data[k + 1 : k + length]:
        value = (value << 8) | b
    return value, length


def _matroska(path: Path, data: bytes) -> Detection | None:
    k = data.find(b"\x42\x82", 4, 256)  # the DocType element of the EBML header
    size = _ebml_size(data, k + 2) if k >= 0 else None
    if size is None:
        return None
    value, used = size
    doctype = data[k + 2 + used : k + 2 + used + value].rstrip(b"\x00")
    if doctype == b"webm":
        return _magic(path, "video", "webm", "a WebM")
    if doctype == b"matroska":
        return _magic(path, "video", "mkv", "a Matroska (MKV)")
    return None


def _gif_frames(path: Path, enough: int = 2) -> int:
    """How many image descriptors the GIF holds, counting up to `enough` (the blocks are walked, not searched)."""

    def skip_sub_blocks(f) -> bool:
        while True:
            n = f.read(1)
            if not n:
                return False
            if n[0] == 0:
                return True
            f.seek(n[0], 1)

    frames = 0
    with path.open("rb") as f:
        header = f.read(13)
        if len(header) < 13:
            return frames
        if header[10] & 0x80:  # a global colour table follows the screen descriptor
            f.seek(3 * 2 ** ((header[10] & 0x07) + 1), 1)
        while frames < enough:
            block = f.read(1)
            if block == b"!":  # an extension: its label, then sub-blocks
                if not f.read(1) or not skip_sub_blocks(f):
                    break
            elif block == b",":  # an image descriptor: 9 bytes, a local colour table, the LZW code size, data
                frames += 1
                descriptor = f.read(9)
                if len(descriptor) < 9:
                    break
                if descriptor[8] & 0x80:
                    f.seek(3 * 2 ** ((descriptor[8] & 0x07) + 1), 1)
                if not f.read(1) or not skip_sub_blocks(f):
                    break
            else:  # the trailer ";", the end of the file, or a damaged block
                break
    return frames


# ----------------------------------------------------------------------------- structure
def _by_structure(path: Path, head: bytes) -> Detection | None:
    size = path.stat().st_size
    if size >= 134 and len(head) >= 84:
        count = int.from_bytes(head[80:84], "little")
        if count >= 1 and size == 84 + 50 * count:
            return Detection("3d", "stl", "content", (
                f"{path.name} has an 80-byte header and a triangle count ({count}) that matches its size of "
                f"{size} bytes, so it is a binary STL mesh."
            ))
    lines = head.split(b"\n", 2)
    if len(lines) >= 2 and lines[0].rstrip(b"\r") == b"ply" and lines[1].startswith(b"format "):
        kind = lines[1].split()[1].decode("ascii", "replace") if len(lines[1].split()) > 1 else "?"
        return Detection("3d", "ply", "content", (
            f"{path.name} starts with a PLY header (format {kind}), so it is a 3D mesh or point cloud."
        ))
    return None


# ----------------------------------------------------------------------------- text
def _by_content(path: Path, head: bytes, *, filled: bool) -> Detection:
    text, encoding = _as_text(head, filled=filled)
    if text is None:
        return Detection(
            "other", "unknown", "content", f"{path.name} matches no known signature and is not UTF-8 or UTF-16 text."
        )
    found = _structured_text(path, text, encoding, filled=filled)
    if found is not None:
        return found
    first = text.splitlines()[0] if text else ""
    if first.startswith(">"):
        detail = f"{path.name} is text whose first line starts with '>', so FASTA."
        return Detection("other", "fasta", "content", detail)
    if first.count(",") >= 2:
        return Detection(
            "other", "csv", "content", f"{path.name} is text whose first line has {first.count(',')} commas, so CSV."
        )
    found = _plain_values(path, text, filled=filled)
    if found is not None:
        return found
    return Detection("text", "text", "content", f"{path.name} is {encoding} text with no table or sequence header.")


def _lines(text: str, *, filled: bool) -> list[str]:
    """The text's lines, without the last one when the read window was filled (it may be cut)."""
    lines = text.splitlines()
    return lines[:-1] if filled and len(lines) > 1 else lines


def _structured_text(path: Path, text: str, encoding: str, *, filled: bool) -> Detection | None:
    name = path.name
    stripped = text.lstrip("﻿ \t\r\n")
    if stripped.startswith("{"):
        found = _json_kind(path, stripped, encoding)
        if found is not None:
            return found
    if stripped.startswith("<"):
        root = _xml_root(stripped)
        if root in ("gpx", "kml"):
            return Detection("geo", root, "content", (
                f"{name} is XML whose root element is <{root}>, so it is a {root.upper()} file of places or tracks."
            ))
    lines = _lines(text, filled=filled)
    counts = next((k for k, line in enumerate(lines[:8]) if _is_counts_line(line)), None)
    if counts is not None:
        version = lines[counts].rstrip()[-5:]
        fmt = "sdf" if _has_record_end(path, text) else "mol"
        return Detection("molecule", fmt, "content", (
            f"{name} has a molfile counts line ({version}) on line {counts + 1}, so it is a "
            f"{'SDF file of molecules' if fmt == 'sdf' else 'MOL file of a molecule'}."
        ))
    if any(line.strip() == "$$$$" for line in lines) and any(line.startswith("M  END") for line in lines):
        return Detection("molecule", "sdf", "content", f"{name} holds MOL records ended by '$$$$', so it is an SDF file.")
    first = next((line.strip() for line in lines if line.strip()), "")
    if first.startswith("InChI="):
        return Detection("molecule", "inchi", "content", f"{name} starts with 'InChI=', so it is an InChI identifier.")
    off = _off_header(lines)
    if off is not None:
        return Detection("3d", "off", "content", f"{name} starts with an {off} header and counts, so it is an OFF mesh.")
    if first.startswith("solid") and re.search(r"^\s*facet\s+normal\b", text[:2048], re.M):
        return Detection("3d", "stl", "content", (
            f"{name} starts with 'solid' followed by 'facet normal' lines, so it is an ASCII STL mesh."
        ))
    obj = _obj_counts(path, lines, filled=filled)
    if obj is not None:
        return Detection("3d", "obj", "content", (
            f"{name} is text made of OBJ statements ({obj[0]} vertex lines and faces in the first lines), so it is "
            "an OBJ mesh."
        ))
    return None


def _json_kind(path: Path, stripped: str, encoding: str) -> Detection | None:
    """geojson or gltf for a JSON object, else None."""
    obj = None
    size = path.stat().st_size
    if size <= JSON_MAX_BYTES:
        raw = path.read_bytes()
        codec = {"UTF-16": "utf-16"}.get(encoding, "utf-8-sig")
        try:
            obj = json.loads(raw.decode(codec))
        except (UnicodeDecodeError, ValueError, RecursionError):
            return None
        if not isinstance(obj, dict):
            return None
        if _is_geojson(obj):
            return Detection("geo", "geojson", "content", (
                f"{path.name} is a GeoJSON object (type {obj.get('type')}) with coordinates, so it holds places."
            ))
        asset = obj.get("asset")
        if isinstance(asset, dict) and "version" in asset:
            return Detection("3d", "gltf", "content", (
                f"{path.name} is a JSON object with a glTF asset version, so it is a glTF 3D scene."
            ))
        return None
    # too large to parse here: decide from the first bytes
    if re.search(r'"asset"\s*:\s*\{[^{}]*"version"\s*:', stripped):
        return Detection("3d", "gltf", "content", f"{path.name} starts as a glTF JSON object, so it is a glTF 3D scene.")
    kind = re.search(r'"type"\s*:\s*"(' + "|".join(sorted(_GEOJSON_TYPES)) + r')"', stripped)
    if kind and '"coordinates"' in stripped:
        return Detection("geo", "geojson", "content", (
            f"{path.name} starts as a GeoJSON object (type {kind.group(1)}) with coordinates, so it holds places."
        ))
    return None


def _is_geojson(obj: object, depth: int = 0) -> bool:
    if not isinstance(obj, dict) or depth > 4:
        return False
    kind = obj.get("type")
    if kind in _GEOMETRIES:
        coordinates = obj.get("coordinates")
        return isinstance(coordinates, list) and len(coordinates) > 0
    if kind == "GeometryCollection":
        return any(_is_geojson(g, depth + 1) for g in (obj.get("geometries") or [])[:1000])
    if kind == "Feature":
        return _is_geojson(obj.get("geometry"), depth + 1)
    if kind == "FeatureCollection":
        features = obj.get("features")
        return isinstance(features, list) and any(_is_geojson(f, depth + 1) for f in features[:1000])
    return False


def _xml_root(text: str) -> str | None:
    """The local name of the first element, after an XML declaration, comments and a doctype; lowercased."""
    rest = text
    for _ in range(50):
        if rest.startswith("<?"):
            end = rest.find("?>")
            rest = rest[end + 2 :].lstrip() if end >= 0 else ""
        elif rest.startswith("<!--"):
            end = rest.find("-->")
            rest = rest[end + 3 :].lstrip() if end >= 0 else ""
        elif rest.startswith("<!"):
            end = rest.find(">")
            rest = rest[end + 1 :].lstrip() if end >= 0 else ""
        else:
            break
    found = re.match(r"<(?:[A-Za-z_][\w.\-]*:)?([A-Za-z_][\w.\-]*)[\s>/]", rest)
    return found.group(1).lower() if found else None


def _is_counts_line(line: str) -> bool:
    line = line.rstrip()
    return line.endswith(("V2000", "V3000")) and re.match(r"^\s*\d", line) is not None


def _has_record_end(path: Path, text: str) -> bool:
    if re.search(r"^\$\$\$\$\s*$", text, re.M):
        return True
    with path.open("rb") as f:
        return re.search(rb"(?m)^\$\$\$\$\s*$", f.read(SDF_SCAN_BYTES)) is not None


def _off_header(lines: list[str]) -> str | None:
    body = [line.split() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if not body or body[0][0] not in _OFF_HEADS:
        return None
    counts = body[0][1:] or (body[1] if len(body) > 1 else [])
    return body[0][0] if len(counts) >= 3 and all(c.isdigit() for c in counts[:3]) else None


def _obj_counts(path: Path, lines: list[str], *, filled: bool) -> tuple[int, int] | None:
    """(vertex lines, face lines) when the lines read as an OBJ mesh, else None."""
    body = [line.split() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if not body:
        return None
    keys = [words[0] for words in body]
    vertices = [words for words in body if words[0] == "v"]
    faces = keys.count("f")
    if sum(k in _OBJ_KEYS for k in keys) * 2 <= len(keys) or len(vertices) < 3:
        return None
    if len(vertices[0]) < 4 or not all(_NUMBER.match(x) for x in vertices[0][1:4]):
        return None
    if faces == 0 and filled:
        faces = int(_has_face_later(path))
    return (len(vertices), faces) if faces >= 1 else None


def _has_face_later(path: Path) -> bool:
    """Whether an "f" line appears in the first SCAN_BYTES of the file (a large OBJ lists its vertices first)."""
    face = re.compile(rb"(?m)^[ \t]*f[ \t]")
    tail = b""
    with path.open("rb") as f:
        read = 0
        while read < SCAN_BYTES:
            chunk = f.read(1 << 20)
            if not chunk:
                return False
            read += len(chunk)
            if face.search(tail + chunk):
                return True
            tail = chunk[-64:]
    return False


def _plain_values(path: Path, text: str, *, filled: bool) -> Detection | None:
    """latlon, a numeric CSV of one comma per line, or SMILES."""
    name = path.name
    lines = [line.strip() for line in _lines(text, filled=filled) if line.strip()]
    if not lines:
        return None
    if not filled and len(lines) <= MAX_LATLON_LINES:
        points = [_latlon(line) for line in lines]
        if all(p is not None for p in points):
            shown = ", ".join(f"({lat:g}, {lon:g})" for lat, lon in points[:2])
            more = f" and {len(points) - 2} more" if len(points) > 2 else ""
            return Detection("geo", "latlon", "content", (
                f"{name} holds {len(points)} latitude and longitude pair{'' if len(points) == 1 else 's'} "
                f"{shown}{more}, so it is a list of places."
            ))
    if len(lines) >= 3:
        commas = {line.count(",") for line in lines}
        if len(commas) == 1 and min(commas) >= 1 and all(
            _NUMBER.match(field) for line in lines[1:] for field in line.split(",")
        ):
            k = min(commas)
            return Detection("other", "csv", "content", (
                f"{name} is text whose lines each hold {k} comma{'' if k == 1 else 's'} between numbers, so CSV."
            ))
    if not filled and len(lines) <= MAX_SMILES_LINES and all(_smiles_token(line) for line in lines):
        return Detection("molecule", "smiles", "content", (
            f"{name} holds {len(lines)} SMILES string{'' if len(lines) == 1 else 's'} ({lines[0][:60]}), so it "
            "describes molecules."
        ))
    return None


def _latlon(line: str) -> tuple[float, float] | None:
    found = next((m for m in (p.match(line) for p in _LATLON) if m), None)
    if found is None:
        return None
    sides = []
    for side in "ab":
        letter, number = found.group(f"{side}l"), found.group(f"{side}n")
        if letter and number.startswith(("+", "-")):
            return None  # "-51.5 N": a sign and a hemisphere letter at once
        sides.append((letter, float(number)))
    marked = any(letter for letter, _ in sides) or found.group("ad") or found.group("bd")
    if not marked and "." not in found.group("an") + found.group("bn"):
        return None  # two whole numbers and nothing else: not a place
    (l1, v1), (l2, v2) = sides
    if l1 in ("E", "W") or l2 in ("N", "S"):  # longitude first
        (l1, v1), (l2, v2) = (l2, v2), (l1, v1)
    if (l1 and l1 not in ("N", "S")) or (l2 and l2 not in ("E", "W")):
        return None
    lat = -v1 if l1 == "S" else v1
    lon = -v2 if l2 == "W" else v2
    return (lat, lon) if -90 <= lat <= 90 and -180 <= lon <= 180 else None


def _smiles_token(line: str) -> bool:
    if len(line) < 2 or not _SMILES.fullmatch(line):
        return False
    outside = re.sub(r"\[[^\]]*\]", "", line)
    if not _ORGANIC.search(outside) or outside.count("(") != outside.count(")"):
        return False
    rings = re.findall(r"%\d{2}|\d", outside)  # ring-closure labels open and close a ring: each comes in pairs
    if any(rings.count(label) % 2 for label in set(rings)):
        return False
    return _rdkit_parses(line)


def _rdkit_parses(smiles: str) -> bool:
    """True when rdkit cannot be imported (the alphabet checks stand alone), else whether rdkit parses it."""
    if importlib.util.find_spec("rdkit") is None:
        return True
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    try:
        return Chem.MolFromSmiles(smiles) is not None
    finally:
        RDLogger.EnableLog("rdApp.*")


def _as_text(head: bytes, *, filled: bool) -> tuple[str | None, str]:
    """(text, encoding name), or (None, "") if not text. A byte-order mark picks UTF-16 and is dropped; a
    character cut off at the end is tolerated only when the read window was filled (the file goes on)."""
    if head[:2] == b"\xff\xfe":
        codec, name, body = "utf-16-le", "UTF-16", head[2:]
    elif head[:2] == b"\xfe\xff":
        codec, name, body = "utf-16-be", "UTF-16", head[2:]
    else:
        codec, name, body = "utf-8", "UTF-8", head.removeprefix(codecs.BOM_UTF8)
    try:
        text = codecs.getincrementaldecoder(codec)().decode(body, final=not filled)
    except UnicodeDecodeError:
        return None, ""
    if "\x00" in text:
        return None, ""
    return text, name
