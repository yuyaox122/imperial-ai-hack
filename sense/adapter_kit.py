"""adapter_kit: vetted helpers for adapters that give a frozen base model a new sense.

An adapter is `adapt(path) -> {"text": str} | {"images": [...]}`. The harness copies this file alone next to
the adapter and runs it in an isolated subprocess (`python -I`, no network, secrets stripped, timeout), so this
module imports nothing from the repository and never touches the network. It needs only the standard library,
numpy and PIL at import time; trimesh, matplotlib (always the Agg backend) and rdkit are imported lazily inside
the functions that use them, and ffmpeg/ffprobe are run as subprocesses. Every function raises ValueError with
a plain message when the input cannot be read or a needed tool is missing. Images come back as RGB PIL images
of bounded size; summaries are small JSON-serialisable dicts (plain int, float, str, bool, list, dict, None).

    from adapter_kit import video_frames, contact_sheet
    def adapt(path):
        return {"images": [contact_sheet(video_frames(path, n=6))]}

Video (ffmpeg/ffprobe; animated GIF, APNG and animated WebP also work through PIL alone)
    video_frames(path, n=6, size=384) -> list[PIL.Image]
        n frames at the centres of n equal slices of the duration (fewer when the clip has fewer frames; a still
        image gives one frame), RGB, long side at most `size`, never upscaled. Each frame's .info["time_s"] is
        its timestamp in seconds. mp4, webm, mov, mkv, avi, gif and anything else ffmpeg decodes.
    video_summary(path) -> dict
        duration (s), fps, width, height, codec, has_audio, audio_codec, n_frames (estimate), container,
        size_bytes.

Images
    contact_sheet(images, cols=None, size=1024, labels=None) -> PIL.Image
        A white grid of the images (PIL images, PNG/JPEG bytes or file paths), at most `size` px wide, each
        tile marked with a small index number (1, 2, ...) or the given labels. cols defaults to ceil(sqrt(n)).

3D meshes (trimesh; matplotlib for rendering)
    load_mesh(path) -> trimesh.Trimesh
        OBJ, STL (ascii and binary), PLY, OFF, GLB/glTF (scene geometry merged with its transforms; glTF is
        Y-up, so it is turned to Z-up). The format is sniffed when the extension is missing or wrong. Vertices
        are merged, then the mesh is centred on its bounding-box centre and scaled so its longest side is 1.
        mesh.metadata["kit_original_extents"] keeps the extents in file units.
    mesh_views(path_or_mesh, views=4, size=256) -> list[PIL.Image]
        Flat-shaded orthographic renders from `views` evenly spaced azimuths (starting at 30 degrees) at 25
        degrees elevation, Z up: light grey, Lambert shading from a light fixed relative to the camera, white
        background, axes off, equal aspect, one zoom for all views so the object fills about 90% of the frame.
        Meshes above 20,000 faces are decimated by vertex clustering first. When MPLCONFIGDIR is unset and HOME
        has no matplotlib font cache, MPLCONFIGDIR is set to a per-user folder in the temp directory so the
        slow first-run font scan happens once, not on every sandboxed run.
    mesh_summary(path_or_mesh) -> dict
        vertices, faces, components, watertight, winding_consistent, non_manifold_edges, original_extents (file
        units, longest first), extent_ratios, area (file units), volume (file units), bbox_fill and solidity
        (volume / convex hull volume) - these three are None unless the mesh is watertight and consistently
        wound - euler_number, boundary_loops (gaps in the skin), genus_estimate (through-holes),
        principal_axis_ratios (from a surface sample), mirror_symmetry (0-1, best tested plane) and mirror_planes
        (planes scoring 0.8+), rotational_symmetry (0-1), rotational_order (an int, 1 for none, or "continuous")
        and rotational_axis, and `description`: one plain-English paragraph built only from these numbers (it
        never guesses what the object is). Symmetry is tested about the file axes and the principal axes through
        the surface centroid, on a fixed-seed sample, so results are repeatable.

Molecules (rdkit)
    read_molecule(path_or_text) -> rdkit.Chem.Mol
        A path or the text itself: SMILES (first token of the first non-empty line; a header line naming
        SMILES is skipped), InChI, MOL block, SDF (first record), Mol2 (first molecule) or PDB.
    molecule_summary(mol_or_path) -> dict
        canonical_smiles, formula, molecular_weight, logp, tpsa, h_bond_donors, h_bond_acceptors,
        rotatable_bonds, rings, aromatic_rings, heavy_atoms, heavy_atoms_by_element, formal_charge, fragments,
        functional_groups ({name: count}, only those present; names in FUNCTIONAL_GROUPS plus aromatic_ring),
        terpene_like, and a plain-English `description`.
    molecule_image(mol_or_path, size=300) -> PIL.Image
        A 2D structure drawing (rdkit.Chem.Draw); ValueError when rdkit has no drawing backend.

Coordinates (Natural Earth gazetteer, offline)
    read_coordinates(path_or_text) -> list[tuple[float, float]]
        (lat, lon) pairs in degrees, in file order, from: plain "lat, lon" / "lat lon" lines (latitude first;
        extra fields after the two numbers are ignored; CSV with a lat/lon header in either column order;
        "lat=.. lon=.." labels; geo: URIs), degrees-minutes-seconds with N/S/E/W (51°30'26"N 0°7'39"W,
        N51 30.4 W0 7.6, ...; either order), GeoJSON (Point, MultiPoint, LineString, MultiLineString, Polygon,
        MultiPolygon, GeometryCollection, Feature, FeatureCollection; GeoJSON stores lon, lat and this returns
        lat, lon), GPX (trkpt, wpt, rtept) and KML (<coordinates> lon,lat[,alt]; gx:coord). Text lines without
        two numbers are skipped; a latitude outside -90..90 or a longitude outside -180..180 is a ValueError.
    nearest_places(lat, lon, k=3, gazetteer=None) -> list[dict]
        The k nearest places in the gazetteer, nearest first, by haversine distance: name, country (adm0name),
        country_code (ISO 3166-1 alpha-2 from iso_a2; None where Natural Earth has none), admin1 (adm1name),
        population (pop_max), feature_class (featurecla, e.g. "Admin-0 capital"), distance_km, bearing (16-point
        compass direction of the query point as seen from the place, so "<distance_km> km <bearing> of <name>"
        reads naturally), bearing_deg, lat, lon. The gazetteer is Natural Earth 1:10m populated places (public
        domain, about 7,300 cities and towns): the GeoJSON at $SENSE_GAZETTEER, else
        ~/.cache/sense-harness/gazetteer/ne_10m_populated_places_simple.geojson (fetch it with
        scripts/fetch_gazetteer.py). It is loaded once per process and cached.
        Why not GeoNames: the coordinate probe labels are built from GeoNames cities15000, so a gazetteer taken
        from that same table would match every probe point at 0 km and verification would prove nothing. Keep
        the gazetteer independent of the probe labels' source.
    coordinate_summary(lat, lon, k=3, gazetteer=None) -> dict
        lat, lon, dms, hemisphere, climate_band (rough, by latitude only), nearest_places (None when the
        gazetteer is unavailable, with the reason in gazetteer_note) and a plain-English `description`.
    haversine_km(lat1, lon1, lat2, lon2) -> float
        Great-circle distance in kilometres on a 6371 km sphere.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = [
    "video_frames", "video_summary", "contact_sheet",
    "load_mesh", "mesh_views", "mesh_summary",
    "read_molecule", "molecule_summary", "molecule_image",
    "read_coordinates", "nearest_places", "coordinate_summary", "haversine_km",
    "FUNCTIONAL_GROUPS",
]

_TOOL_DIRS = ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin")
_MAX_TEXT_BYTES = 50_000_000


# ============================================================================= shared helpers
def _find_tool(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    for folder in _TOOL_DIRS:
        candidate = os.path.join(folder, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise ValueError(f"{name} is not installed (looked on PATH and in {', '.join(_TOOL_DIRS)}); "
                     f"install ffmpeg to read video")


def _existing_file(path) -> Path:
    p = Path(os.fspath(path)).expanduser()
    if not p.is_file():
        raise ValueError(f"no such file: {os.fspath(path)}")
    return p


def _looks_like_path(value) -> bool:
    if isinstance(value, os.PathLike):
        return True
    if isinstance(value, str) and value and len(value) < 4096 and "\n" not in value:
        try:
            return Path(value).expanduser().is_file()
        except (OSError, ValueError):
            return False
    return False


_TEXT_SUFFIXES = (".txt", ".csv", ".tsv", ".json", ".geojson", ".gpx", ".kml", ".smi", ".smiles", ".sdf",
                  ".mol", ".mol2", ".pdb", ".inchi", ".xyz", ".gz")


def _read_text(path_or_text, what: str) -> str:
    """Return the text of a file path, or the value itself when it is text (str or bytes)."""
    if isinstance(path_or_text, (bytes, bytearray)):
        return bytes(path_or_text).decode("utf-8", errors="replace")
    if isinstance(path_or_text, str) and "\n" not in path_or_text and not _looks_like_path(path_or_text):
        value = path_or_text.strip()
        if value.startswith(("/", "~", "./", "../")) or value.lower().endswith(_TEXT_SUFFIXES):
            raise ValueError(f"no such file: {value}")
    if _looks_like_path(path_or_text):
        p = _existing_file(path_or_text)
        if p.stat().st_size > _MAX_TEXT_BYTES:
            raise ValueError(f"{p.name} is too large to read as {what} ({p.stat().st_size} bytes)")
        data = p.read_bytes()
        if data[:2] == b"\x1f\x8b":
            import gzip
            try:
                data = gzip.decompress(data)
            except OSError as exc:
                raise ValueError(f"{p.name} looks gzipped but could not be decompressed: {exc}") from None
        return data.decode("utf-8-sig", errors="replace")
    if isinstance(path_or_text, str):
        return path_or_text
    raise ValueError(f"expected a file path or {what} text, got {type(path_or_text).__name__}")


def _to_image(item) -> Image.Image:
    """PIL image, PNG/JPEG bytes or a file path -> RGB PIL image."""
    try:
        if isinstance(item, Image.Image):
            img = item
        elif isinstance(item, (bytes, bytearray)):
            img = Image.open(io.BytesIO(bytes(item)))
        elif isinstance(item, (str, os.PathLike)):
            img = Image.open(_existing_file(item))
        elif isinstance(item, np.ndarray):
            img = Image.fromarray(item)
        else:
            raise ValueError(f"cannot use a {type(item).__name__} as an image")
        img.load()
    except ValueError:
        raise
    except Exception as exc:  # PIL raises several types for bad data
        raise ValueError(f"unreadable image: {exc}") from None
    if img.mode in ("RGBA", "LA", "P", "PA"):
        img = img.convert("RGBA")
        back = Image.new("RGB", img.size, "white")
        back.paste(img, mask=img.getchannel("A"))
        return back
    return img.convert("RGB") if img.mode != "RGB" else img


def _shrink(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB") if img.mode != "RGB" else img
    if max(img.size) > size:
        img = img.copy()
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
    return img


def _plain(value):
    """numpy scalars and arrays -> plain Python, recursively, so summaries are JSON-serialisable."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    return value


def _r(x, digits=3):
    return None if x is None else round(float(x), digits)


def _font(px: int):
    try:
        return ImageFont.load_default(size=px)
    except TypeError:  # older Pillow without sized default font
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


# ============================================================================= video
def _animated_pil(p: Path):
    """Open p with PIL when it is an image file (animated or still); None when PIL cannot read it."""
    try:
        img = Image.open(p)
    except Exception:
        return None
    if img.format not in ("GIF", "PNG", "WEBP", "JPEG", "MPO", "BMP", "TIFF"):
        return None
    return img


def _pil_durations(img) -> list[float]:
    n = getattr(img, "n_frames", 1)
    out = []
    for i in range(n):
        img.seek(i)
        d = img.info.get("duration") or 0
        out.append((d if d >= 20 else 100) / 1000.0)
    img.seek(0)
    return out


def _ffprobe(p: Path) -> dict:
    cmd = [_find_tool("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
           str(p)]
    try:
        proc = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL, timeout=60)
    except subprocess.TimeoutExpired:
        raise ValueError(f"ffprobe timed out reading {p.name}") from None
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise ValueError(f"ffprobe cannot read {p.name}: {msg[-1] if msg else 'unknown error'}")
    try:
        return json.loads(proc.stdout.decode("utf-8", "replace") or "{}")
    except json.JSONDecodeError:
        raise ValueError(f"ffprobe gave unreadable output for {p.name}") from None


def _rate(text) -> float | None:
    if not text or text in ("0/0", "N/A"):
        return None
    try:
        if "/" in str(text):
            a, b = str(text).split("/", 1)
            return float(a) / float(b) if float(b) else None
        return float(text)
    except ValueError:
        return None


def _float(x) -> float | None:
    try:
        f = float(x)
        return f if math.isfinite(f) and f > 0 else None
    except (TypeError, ValueError):
        return None


def _decode_duration(p: Path) -> float | None:
    """Duration by stream-copying the video to null and reading the last time= ffmpeg reports."""
    cmd = [_find_tool("ffmpeg"), "-nostdin", "-v", "info", "-i", str(p), "-map", "0:v:0", "-c", "copy",
           "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL, timeout=120)
    except subprocess.TimeoutExpired:
        return None
    times = re.findall(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr.decode("utf-8", "replace"))
    if not times:
        return None
    h, m, s = times[-1]
    return _float(int(h) * 3600 + int(m) * 60 + float(s))


def _probe_video(p: Path) -> dict:
    info = _ffprobe(p)
    streams = info.get("streams") or []
    fmt = info.get("format") or {}
    video = [s for s in streams if s.get("codec_type") == "video"
             and not (s.get("disposition") or {}).get("attached_pic")]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if not video:
        what = "only audio" if audio else "no video stream"
        raise ValueError(f"{p.name} has no video to show ({what})")
    v = video[0]
    rates = [_rate(v.get("avg_frame_rate")), _rate(v.get("r_frame_rate"))]
    fps = next((r for r in rates if r and 0 < r <= 1000), None)  # some containers report nonsense rates
    duration = _float(fmt.get("duration")) or _float(v.get("duration"))
    nb = None
    try:
        nb = int(v.get("nb_frames")) if v.get("nb_frames") not in (None, "N/A") else None
    except ValueError:
        nb = None
    if duration is None:
        duration = _decode_duration(p)
    if nb is None and duration and fps:
        nb = max(1, int(round(duration * fps)))
    return {
        "container": fmt.get("format_name"),
        "duration": duration,
        "fps": fps,
        "width": v.get("width"),
        "height": v.get("height"),
        "codec": v.get("codec_name"),
        "has_audio": bool(audio),
        "audio_codec": audio[0].get("codec_name") if audio else None,
        "n_frames": nb,
    }


def video_summary(path) -> dict:
    """Container facts for a video or animated image: duration, fps, size, codec, audio, frame count."""
    p = _existing_file(path)
    out = None
    img = _animated_pil(p)
    if img is not None:
        try:
            if getattr(img, "n_frames", 1) > 1:
                durations = _pil_durations(img)
                total = sum(durations)
                out = {"container": img.format.lower(), "duration": total,
                       "fps": len(durations) / total if total else None, "width": img.width, "height": img.height,
                       "codec": img.format.lower(), "has_audio": False, "audio_codec": None,
                       "n_frames": len(durations)}
        except Exception:
            out = None  # let ffprobe try
        finally:
            img.close()
    if out is None:
        out = _probe_video(p)
    out["size_bytes"] = p.stat().st_size
    out["duration"] = _r(out["duration"])
    out["fps"] = _r(out["fps"])
    return _plain(out)


def _ffmpeg_frame(p: Path, t: float, size: int, accurate: bool) -> Image.Image | None:
    scale = f"scale='min({size},iw)':'min({size},ih)':force_original_aspect_ratio=decrease"
    seek = ["-ss", f"{t:.3f}"]
    cmd = [_find_tool("ffmpeg"), "-nostdin", "-v", "error"]
    cmd += (["-i", str(p)] + seek) if accurate else (seek + ["-i", str(p)])
    cmd += ["-map", "0:v:0", "-an", "-sn", "-dn", "-frames:v", "1", "-vf", scale, "-f", "image2pipe",
            "-c:v", "png", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL, timeout=120)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    try:
        img = Image.open(io.BytesIO(proc.stdout))
        img.load()
        return img.convert("RGB")
    except Exception:
        return None


def video_frames(path, n: int = 6, size: int = 384) -> list:
    """n RGB frames evenly spaced over the clip (centres of n equal slices), long side at most `size`."""
    n = int(n)
    size = int(size)
    if n < 1 or n > 200:
        raise ValueError(f"n must be between 1 and 200, got {n}")
    if size < 16:
        raise ValueError(f"size must be at least 16 px, got {size}")
    p = _existing_file(path)

    img = _animated_pil(p)
    if img is not None:
        try:
            durations = _pil_durations(img)
            starts = np.concatenate([[0.0], np.cumsum(durations)[:-1]])
            total = len(durations)
            picks = sorted({min(total - 1, int((i + 0.5) * total / n)) for i in range(n)})
            frames = []
            for i in picks:
                img.seek(i)
                frame = _shrink(_to_image(img.copy()), size)
                frame.info["time_s"] = round(float(starts[i]), 3)
                frames.append(frame)
            return frames
        except Exception:
            pass  # a damaged GIF/WebP: let ffmpeg try
        finally:
            img.close()

    meta = _probe_video(p)
    duration = meta["duration"]
    if meta["n_frames"]:
        n = max(1, min(n, int(meta["n_frames"])))
    if not duration:
        times = [0.0]
    else:
        times = [duration * (i + 0.5) / n for i in range(n)]
    frames = []
    for t in times:
        frame = _ffmpeg_frame(p, t, size, accurate=False) or _ffmpeg_frame(p, t, size, accurate=True)
        if frame is None and t > 0.5:
            frame = _ffmpeg_frame(p, max(0.0, t - 0.5), size, accurate=True)
        if frame is not None:
            frame = _shrink(frame, size)
            frame.info["time_s"] = round(t, 3)
            frames.append(frame)
    if not frames:
        raise ValueError(f"ffmpeg could not decode any frame of {p.name}")
    return frames


# ============================================================================= images
def contact_sheet(images, cols: int | None = None, size: int = 1024, labels=None) -> Image.Image:
    """A white grid of the images, at most `size` px wide, each tile marked with a small index number."""
    items = list(images) if not isinstance(images, (Image.Image, bytes, str, os.PathLike)) else [images]
    if not items:
        raise ValueError("contact_sheet needs at least one image")
    if len(items) > 400:
        raise ValueError(f"contact_sheet takes at most 400 images, got {len(items)}")
    imgs = [_to_image(i) for i in items]
    n = len(imgs)
    cols = int(cols) if cols else math.ceil(math.sqrt(n))
    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols)
    size = max(64, int(size))
    gap = 4
    widest = max(i.width for i in imgs)
    cell_w = max(8, min(widest, (size - gap * (cols + 1)) // cols))
    aspect = max(i.height / i.width for i in imgs)
    cell_h = max(8, int(round(cell_w * aspect)))
    max_h = 2 * size  # keep tall inputs small too
    if rows * cell_h + gap * (rows + 1) > max_h:
        cell_h = max(8, (max_h - gap * (rows + 1)) // rows)
        cell_w = max(8, min(cell_w, int(cell_h / aspect)))
    sheet = Image.new("RGB", (cols * cell_w + gap * (cols + 1), rows * cell_h + gap * (rows + 1)), "white")
    draw = ImageDraw.Draw(sheet)
    font_px = max(9, min(18, cell_w // 10))
    font = _font(font_px)
    if labels is not None:
        labels = [str(x) for x in labels]
        if len(labels) != n:
            raise ValueError(f"got {len(labels)} labels for {n} images")
    for k, img in enumerate(imgs):
        r, c = divmod(k, cols)
        tile = img.copy()
        tile.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        x0 = gap + c * (cell_w + gap)
        y0 = gap + r * (cell_h + gap)
        sheet.paste(tile, (x0 + (cell_w - tile.width) // 2, y0 + (cell_h - tile.height) // 2))
        text = labels[k] if labels is not None else str(k + 1)
        box = draw.textbbox((0, 0), text, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        draw.rectangle([x0, y0, x0 + tw + 6, y0 + th + 6], fill="black")
        draw.text((x0 + 3 - box[0], y0 + 3 - box[1]), text, fill="white", font=font)
    return sheet


# ============================================================================= meshes
_MESH_TYPES = {"obj", "stl", "ply", "off", "glb", "gltf"}
_MAX_RENDER_FACES = 20_000


def _sniff_mesh_type(p: Path) -> str | None:
    """The mesh format from the file's signature, else its extension, else OBJ-looking text; None if unknown."""
    with open(p, "rb") as fh:
        head = fh.read(4096)
    size = p.stat().st_size
    if head[:4] == b"glTF":
        return "glb"
    if head[:3].lower() == b"ply":
        return "ply"
    stripped = head.lstrip()
    if stripped[:3] in (b"OFF",) or stripped[:4] in (b"COFF", b"NOFF"):
        return "off"
    if len(head) >= 84:
        count = int.from_bytes(head[80:84], "little")
        if count and 84 + 50 * count == size:
            return "stl"
    if stripped[:5].lower() == b"solid" and b"facet" in head:
        return "stl"
    if stripped[:1] == b"{" and b'"asset"' in head:
        return "gltf"
    ext = p.suffix.lower().lstrip(".")
    if ext in _MESH_TYPES:
        return ext
    text = head.decode("utf-8", "replace")
    if re.search(r"(?m)^\s*v\s+[-+.\d]", text) and (re.search(r"(?m)^\s*f\s+\d", text) or size > 4096):
        return "obj"
    return None


def _trimesh():
    try:
        import trimesh
    except ImportError:
        raise ValueError("trimesh is not installed, so meshes cannot be read") from None
    return trimesh


def _merge_scene(loaded):
    trimesh = _trimesh()
    if isinstance(loaded, trimesh.Scene):
        parts = [g for g in loaded.dump(concatenate=False) if isinstance(g, trimesh.Trimesh) and len(g.faces)]
        if not parts:
            raise ValueError("the file holds no triangle geometry (only points, lines or an empty scene)")
        return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    raise ValueError(f"the file holds a {type(loaded).__name__}, not a triangle mesh")


def _normalise(mesh, original_extents=None):
    """Copy of mesh with merged vertices, centred on its bounding box, longest side scaled to 1."""
    m = mesh.copy()
    m.remove_unreferenced_vertices()
    if len(m.faces) == 0 or len(m.vertices) < 3:
        raise ValueError("the mesh has no faces")
    if not np.all(np.isfinite(m.vertices)):
        raise ValueError("the mesh has non-finite vertex coordinates")
    m.merge_vertices(merge_tex=True, merge_norm=True)  # geometry only: split UV/normal seams are joined
    lo, hi = m.bounds
    ext = hi - lo
    longest = float(ext.max())
    if longest <= 0:
        raise ValueError("the mesh has zero size")
    m.apply_translation(-(lo + hi) / 2.0)
    m.apply_scale(1.0 / longest)
    m.metadata["kit_original_extents"] = [float(x) for x in (original_extents if original_extents is not None
                                                             else ext)]
    m.metadata["kit_scale"] = 1.0 / longest
    return m


def load_mesh(path):
    """Read a mesh file into one trimesh.Trimesh, centred and scaled so its longest side is 1."""
    trimesh = _trimesh()
    p = _existing_file(path)
    kind = _sniff_mesh_type(p)
    if kind is None:
        raise ValueError(f"{p.name} is not a mesh format this kit reads (OBJ, STL, PLY, OFF, GLB, glTF)")
    try:
        loaded = trimesh.load(str(p), file_type=kind, process=True)
    except Exception as exc:
        raise ValueError(f"trimesh cannot read {p.name} as {kind.upper()}: {exc}") from None
    mesh = _merge_scene(loaded)
    if kind in ("glb", "gltf"):  # glTF is Y-up; turn it to Z-up so renders stand upright
        mesh = mesh.copy()
        mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
    return _normalise(mesh)


def _as_mesh(path_or_mesh):
    trimesh = _trimesh()
    if isinstance(path_or_mesh, trimesh.Trimesh):
        if "kit_scale" in path_or_mesh.metadata and "kit_original_extents" in path_or_mesh.metadata:
            return path_or_mesh.copy()
        return _normalise(path_or_mesh)
    if isinstance(path_or_mesh, trimesh.Scene):
        return _normalise(_merge_scene(path_or_mesh))
    return load_mesh(path_or_mesh)


def _cluster_decimate(vertices: np.ndarray, faces: np.ndarray, max_faces: int):
    """Vertex-clustering decimation: snap vertices to a grid, merge each cell, drop collapsed faces."""
    if len(faces) <= max_faces:
        return vertices, faces
    lo = vertices.min(0)
    span = float((vertices.max(0) - lo).max()) or 1.0
    res = 256
    best = (vertices, faces)
    while res >= 4:
        keys = np.floor((vertices - lo) / span * res).astype(np.int64)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        counts = np.bincount(inverse).astype(float)
        merged = np.stack([np.bincount(inverse, weights=vertices[:, d]) for d in range(3)], 1) / counts[:, None]
        f = inverse[faces]
        keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
        f = f[keep]
        if len(f):
            _, first = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
            f = f[np.sort(first)]
        best = (merged, f)
        if len(f) <= max_faces:
            return best
        res = int(res * 0.7)
    v, f = best
    if len(f) > max_faces:  # pathological input: keep an even subset
        f = f[np.linspace(0, len(f) - 1, max_faces).astype(int)]
    return v, f


def _view_vector(azim_deg: float, elev_deg: float) -> np.ndarray:
    a, e = math.radians(azim_deg), math.radians(elev_deg)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def _matplotlib_cache_dir() -> None:
    """Give matplotlib a font cache that survives between sandboxed runs.

    Without a usable cache matplotlib rescans every system font (over 10 s on a Mac) and, when HOME is missing
    or read-only, does so again on every run in a fresh temporary directory. When MPLCONFIGDIR is unset and no
    cache exists under HOME, point it at one stable per-user directory in the system temp folder instead.
    """
    if os.environ.get("MPLCONFIGDIR"):
        return
    home = Path(os.path.expanduser("~"))
    xdg = os.environ.get("XDG_CACHE_HOME")
    homes = [home / ".matplotlib", home / ".cache" / "matplotlib"] + ([Path(xdg) / "matplotlib"] if xdg else [])
    try:
        if any(d.is_dir() and any(d.glob("fontlist-*.json")) for d in homes):
            return
    except OSError:
        pass
    uid = os.getuid() if hasattr(os, "getuid") else "user"
    shared = Path(tempfile.gettempdir()) / f"adapter-kit-matplotlib-{uid}"
    try:
        shared.mkdir(mode=0o700, exist_ok=True)
    except OSError:
        return
    if os.access(shared, os.W_OK):
        os.environ["MPLCONFIGDIR"] = str(shared)


def _refine_for_sorting(verts: np.ndarray, faces: np.ndarray):
    """Split large triangles of a coarse mesh so matplotlib's per-face painter's sort makes fewer mistakes."""
    if len(faces) >= 4000:
        return verts, faces
    try:
        from trimesh import remesh
        v2, f2 = remesh.subdivide_to_size(verts, faces, max_edge=0.06, max_iter=6)
    except Exception:
        return verts, faces
    return (v2, f2) if len(f2) <= 12_000 else (verts, faces)


def mesh_views(path_or_mesh, views: int = 4, size: int = 256) -> list:
    """Flat-shaded renders from evenly spaced azimuths at 25 degrees elevation; white background."""
    views = int(views)
    size = int(size)
    if views < 1 or views > 24:
        raise ValueError(f"views must be between 1 and 24, got {views}")
    if size < 32 or size > 2048:
        raise ValueError(f"size must be between 32 and 2048 px, got {size}")
    mesh = _as_mesh(path_or_mesh)
    _matplotlib_cache_dir()
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from mpl_toolkits.mplot3d import proj3d
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        raise ValueError("matplotlib is not installed, so meshes cannot be rendered") from None

    closed = bool(mesh.is_watertight and mesh.is_winding_consistent)
    if closed and mesh.volume < 0:
        mesh.invert()
    verts, faces = _cluster_decimate(np.asarray(mesh.vertices, float), np.asarray(mesh.faces), _MAX_RENDER_FACES)
    verts, faces = _refine_for_sorting(verts, faces)
    tris = verts[faces]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    ok = lengths > 1e-12
    tris, normals = tris[ok], normals[ok] / lengths[ok, None]
    if len(tris) == 0:
        raise ValueError("the mesh has only degenerate (zero-area) faces")
    used = np.unique(faces[ok])
    probe = verts[used] if len(used) <= 5000 else verts[used[np.linspace(0, len(used) - 1, 5000).astype(int)]]
    elev = 25.0
    base = np.array([0.82, 0.82, 0.84])
    azims = [30.0 + 360.0 * i / views for i in range(views)]

    def axes_for(azim, zoom):
        fig = Figure(figsize=(size / 100.0, size / 100.0), dpi=100, facecolor="white")
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes((0, 0, 1, 1), projection="3d")
        ax.set_facecolor("white")
        ax.set_proj_type("ortho")
        ax.view_init(elev=elev, azim=azim)
        for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            setter(-0.5, 0.5)
        ax.set_box_aspect((1, 1, 1), zoom=zoom)
        ax.set_axis_off()
        return fig, canvas, ax

    # One zoom for all views: at zoom 1 the +-0.5 cube (which holds the unit-scaled mesh) always fits, and the
    # projected footprint of the vertices says how much closer the camera can come and still show every view.
    reach = 1e-9
    for azim in azims:
        _, _, ax = axes_for(azim, 1.0)
        m = ax.get_proj()
        x, y, _ = proj3d.proj_transform(*probe.T, m)
        px = ax.transData.transform(np.c_[x, y])
        cx, cy = ax.transData.transform(np.c_[proj3d.proj_transform(0.0, 0.0, 0.0, m)[:2]].T)[0]
        reach = max(reach, float(np.abs(px[:, 0] - cx).max()), float(np.abs(px[:, 1] - cy).max()))
    zoom = float(np.clip(0.88 * (size / 2.0) / reach, 1.0, 6.0))

    out = []
    for azim in azims:
        eye = _view_vector(azim, elev)
        light = _view_vector(azim + 35.0, elev + 30.0)
        if closed:
            front = normals @ eye > -1e-6
            t, nrm = tris[front], normals[front]
        else:
            t, nrm = tris, normals
        shade = 0.35 + 0.65 * np.abs(nrm @ light)
        colours = np.clip(base[None, :] * shade[:, None], 0, 1)
        fig, canvas, ax = axes_for(azim, zoom)
        ax.add_collection3d(Poly3DCollection(t, facecolors=colours, edgecolors=colours, linewidths=0.15))
        canvas.draw()
        w, h = canvas.get_width_height()
        img = Image.frombuffer("RGBA", (w, h), bytes(canvas.buffer_rgba()), "raw", "RGBA", 0, 1).convert("RGB")
        img.info["azimuth"] = azim
        img.info["elevation"] = elev
        out.append(img)
    return out


def _sample_surface(mesh, n: int, seed: int) -> np.ndarray:
    tri = np.asarray(mesh.triangles, float)
    areas = np.asarray(mesh.area_faces, float)
    total = areas.sum()
    if total <= 0:
        return np.asarray(mesh.vertices, float)[: n]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(areas), size=n, p=areas / total)
    u = rng.random((n, 2))
    flip = u.sum(1) > 1
    u[flip] = 1 - u[flip]
    t = tri[idx]
    return t[:, 0] + u[:, :1] * (t[:, 1] - t[:, 0]) + u[:, 1:] * (t[:, 2] - t[:, 0])


def _nn_mean(a: np.ndarray, b: np.ndarray) -> float:
    """Mean distance from each point of a to its nearest point of b."""
    try:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(b).query(a, k=1)
        return float(np.mean(d))
    except ImportError:
        best = []
        bb = (b * b).sum(1)
        for start in range(0, len(a), 256):
            chunk = a[start:start + 256]
            d2 = (chunk * chunk).sum(1)[:, None] + bb[None, :] - 2 * chunk @ b.T
            best.append(np.sqrt(np.maximum(d2.min(1), 0)))
        return float(np.mean(np.concatenate(best)))


def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * (_nn_mean(a, b) + _nn_mean(b, a))


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + s * k + (1 - c) * (k @ k)


def _symmetry(points: np.ndarray, other: np.ndarray, centre: np.ndarray, axes: list[tuple[str, np.ndarray]]):
    """Mirror and rotational symmetry scores (0-1) of a surface sample about candidate axes through its centre.

    A score compares the chamfer distance between the transformed sample and an independent second sample with
    the chamfer distance of the two untouched samples (pure sampling noise), relative to the object's RMS radius.
    """
    p, q = points - centre, other - centre
    radius = float(np.sqrt((p * p).sum(1).mean())) or 1.0
    noise = _chamfer(p, q)

    def score(transformed):
        excess = max(0.0, _chamfer(transformed, q) - noise)
        return math.exp(-excess / (0.015 * radius))  # symmetric: excess < 0.002 R; asymmetric: > 0.02 R

    mirror = []
    for name, axis in axes:
        a = axis / np.linalg.norm(axis)
        reflect = np.eye(3) - 2 * np.outer(a, a)
        mirror.append((score(p @ reflect.T), name))
    best_rot = (0.0, 1, None)  # (score, order, axis name); order is an int or "continuous"

    def rank(entry):
        score_, order_, _ = entry
        return (2, 0, score_) if order_ == "continuous" else (1 if order_ > 1 else 0, order_, score_)

    for name, axis in axes:
        cont = min(score(p @ _rotation(axis, math.radians(deg)).T) for deg in (37.0, 151.0))
        if cont >= 0.8:
            candidate = (cont, "continuous", name)
        else:
            candidate = None
            for order in (8, 6, 5, 4, 3, 2):  # highest order that passes; a 4-fold shape fails 8, 6, 5
                s_order = score(p @ _rotation(axis, 2 * math.pi / order).T)
                if s_order >= 0.8:
                    candidate = (s_order, order, name)
                    break
            if candidate is None:
                candidate = (s_order, 1, name)  # s_order is the 2-fold (180 degree) score here
        if rank(candidate) > rank(best_rot):
            best_rot = candidate
    return mirror, best_rot


def _boundary_loops(mesh) -> int:
    edges = np.asarray(mesh.edges_sorted)
    if len(edges) == 0:
        return 0
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = uniq[counts == 1]
    if len(boundary) == 0:
        return 0
    parent: dict[int, int] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in boundary.tolist():
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return len({find(v) for v in parent})


def _shape_words(r2: float, r3: float) -> str:
    if r2 > 0.8 and r3 > 0.8:
        return "compact, about as deep and tall as it is long"
    if r2 < 0.45 and r3 < 0.45:
        return "elongated along one direction (rod-like proportions)"
    if r2 > 0.6 and r3 < 0.3:
        return "flat (plate-like proportions)"
    if r3 < 0.15:
        return "thin in one direction"
    return "moderately elongated"


def mesh_summary(path_or_mesh) -> dict:
    """Geometry facts for a mesh plus a plain-English description built only from those facts."""
    m = _as_mesh(path_or_mesh)
    scale = float(m.metadata.get("kit_scale", 1.0))
    original = sorted(m.metadata.get("kit_original_extents", list(m.extents)), reverse=True)
    extents = sorted((float(x) for x in m.extents), reverse=True)
    longest = extents[0] or 1.0
    ratios = [e / longest for e in extents]
    watertight = bool(m.is_watertight)
    winding = bool(m.is_winding_consistent)
    area_n = float(m.area)
    vol_n = abs(float(m.volume)) if watertight and winding else None
    bbox_vol_n = float(np.prod(m.extents)) or None
    solidity = None
    try:
        hull_vol = float(m.convex_hull.volume)
        if vol_n is not None and hull_vol > 0:
            solidity = min(1.0, vol_n / hull_vol)
    except Exception:
        hull_vol = None
    faces = np.asarray(m.faces)
    v_used = len(np.unique(faces))
    e_unique = len(m.edges_unique)
    euler = int(v_used - e_unique + len(faces))
    try:
        components = int(m.body_count)
    except Exception:
        components = 1
    loops = _boundary_loops(m)
    edge_counts = np.unique(np.asarray(m.edges_sorted), axis=0, return_counts=True)[1]
    non_manifold = int((edge_counts > 2).sum())
    genus = max(0, int(round((2 * components - euler - loops) / 2)))

    weights = np.asarray(m.area_faces, float)
    centre = (np.asarray(m.triangles_center, float) * weights[:, None]).sum(0) / (weights.sum() or 1.0)
    pts = _sample_surface(m, 1024, seed=0)
    other = _sample_surface(m, 1024, seed=1)
    centred = pts - centre
    evals, evecs = np.linalg.eigh(np.cov(centred.T))
    order = np.argsort(evals)[::-1]
    evals, evecs = np.maximum(evals[order], 0), evecs[:, order]
    pa = [1.0] + [math.sqrt(evals[i] / evals[0]) if evals[0] > 0 else 0.0 for i in (1, 2)]
    axes = [("file x axis", np.array([1.0, 0, 0])), ("file y axis", np.array([0, 1.0, 0])),
            ("file z axis", np.array([0, 0, 1.0]))]
    for i, label in enumerate(("longest principal axis", "middle principal axis", "shortest principal axis")):
        if all(abs(float(evecs[:, i] @ a)) < 0.99 for _, a in axes):
            axes.append((label, evecs[:, i]))
    mirror, (rot_score, rot_order, rot_axis) = _symmetry(pts, other, centre, axes)
    mirror_best = max(s for s, _ in mirror)
    mirror_planes = sum(1 for s, _ in mirror if s >= 0.8)

    out = {
        "vertices": int(len(m.vertices)),
        "faces": int(len(faces)),
        "components": components,
        "watertight": watertight,
        "winding_consistent": winding,
        "non_manifold_edges": non_manifold,
        "original_extents": [_r(x, 4) for x in original],
        "extent_ratios": [_r(x) for x in ratios],
        "area": _r(area_n / scale ** 2, 4),
        "volume": _r(vol_n / scale ** 3, 4) if vol_n is not None else None,
        "bbox_fill": _r(vol_n / bbox_vol_n) if vol_n is not None and bbox_vol_n else None,
        "solidity": _r(solidity),
        "euler_number": euler,
        "boundary_loops": loops,
        "genus_estimate": genus,
        "principal_axis_ratios": [_r(x) for x in pa],
        "mirror_symmetry": _r(mirror_best),
        "mirror_planes": int(mirror_planes),
        "rotational_symmetry": _r(rot_score),
        "rotational_order": rot_order,
        "rotational_axis": rot_axis,
    }

    s = []
    closed = "a closed (watertight)" if watertight else "an open (not watertight)"
    s.append(f"This is {closed} triangle mesh with {out['vertices']:,} vertices and {out['faces']:,} faces in "
             f"{components} connected part{'s' if components != 1 else ''}.")
    s.append(f"Its bounding box is {ratios[0]:.2f} : {ratios[1]:.2f} : {ratios[2]:.2f} from longest to shortest "
             f"side ({' x '.join(f'{x:.4g}' for x in original)} in file units), so it is "
             f"{_shape_words(ratios[1], ratios[2])}.")
    if solidity is not None:
        if solidity > 0.9:
            conv = "almost convex"
        elif solidity > 0.65:
            conv = "convex overall with some dents or concavities"
        else:
            conv = "strongly non-convex (large hollows, arms or gaps between parts)"
        s.append(f"It fills {solidity:.0%} of its convex hull and {out['bbox_fill']:.0%} of its bounding box, "
                 f"so it is {conv}.")
    elif loops:
        s.append(f"The surface is open ({loops} boundary loop{'s' if loops != 1 else ''}, that is gaps in the skin), "
                 f"so volume and solidity are not defined.")
    else:
        s.append("The surface does not enclose a volume cleanly (edges not shared by exactly two consistently "
                 "oriented faces), so volume and solidity are not given.")
    if watertight or loops == 0:
        if genus == 0:
            s.append(f"Euler number {euler} gives genus 0: no through-holes.")
        else:
            s.append(f"Euler number {euler} gives genus {genus}: {genus} through-hole{'s' if genus != 1 else ''} "
                     f"(tunnels right through the solid).")
    if non_manifold:
        s.append(f"{non_manifold} edges are shared by more than two faces (non-manifold), so the topology numbers "
                 f"are rough.")
    if mirror_planes:
        s.append(f"It is close to mirror-symmetric across {mirror_planes} of the {len(mirror)} tested planes "
                 f"(best score {mirror_best:.2f}).")
    else:
        s.append(f"It is not clearly mirror-symmetric across any tested plane (best score {mirror_best:.2f}).")
    if rot_order == "continuous":
        s.append(f"It looks rotationally symmetric for any angle about its {rot_axis} (a surface of revolution).")
    elif isinstance(rot_order, int) and rot_order > 1:
        s.append(f"It has approximately {rot_order}-fold rotational symmetry about its {rot_axis}.")
    else:
        s.append("It has no clear rotational symmetry.")
    out["description"] = " ".join(s)
    return _plain(out)


# ============================================================================= molecules
FUNCTIONAL_GROUPS = {
    "carboxylic_acid": "[CX3](=O)[OX2H1,OX1-]",
    "ester": "[#6][CX3](=O)[OX2H0][#6]",
    "lactone": "[#6]@[CX3;R](=O)@[OX2;R]@[#6]",
    "amide": "[NX3][CX3](=[OX1])",
    "aldehyde": "[$([CX3H1](=O)[#6]),$([CX3H2]=O)]",
    "ketone": "[#6][CX3](=O)[#6]",
    "alcohol": "[OX2H][CX4]",
    "phenol": "[OX2H]c",
    "ether": "[OD2;!$(O[#6]=[O,S,N])]([#6])[#6]",
    "amine_primary": "[NX3;H2;+0;!$(N[#6,#16]=[#7,#8,#16]);!$(N[#7,#8])][#6]",
    "amine_secondary": "[NX3;H1;+0;!$(N[#6,#16]=[#7,#8,#16]);!$(N[#7,#8])]([#6])[#6]",
    "amine_tertiary": "[NX3;H0;+0;!$(N[#6,#16]=[#7,#8,#16]);!$(N[#7,#8])]([#6])([#6])[#6]",
    "thiol": "[#6][SX2H1]",
    "thioether": "[#6][SX2;!$(S[#16])][#6]",
    "disulfide": "[#6][SX2][SX2][#6]",
    "nitrile": "[NX1]#[CX2]",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "halide": "[#6][F,Cl,Br,I]",
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
    "isoprene_unit": "[CX4H3,CX3H2;D1]~[#6](~[#6])~[#6]~[#6]",  # branched C5 skeleton (CH3 or =CH2 branch)
}
_GROUP_WORDS = {"carboxylic_acid": "carboxylic acid", "amine_primary": "primary amine",
                "amine_secondary": "secondary amine", "amine_tertiary": "tertiary amine",
                "isoprene_unit": "isoprene-like unit", "aromatic_ring": "aromatic ring",
                "thioether": "thioether (sulfide)"}
_COMPILED: dict = {}


def _rdkit():
    try:
        from rdkit import Chem, RDLogger
    except ImportError:
        raise ValueError("rdkit is not installed, so molecules cannot be read") from None
    RDLogger.DisableLog("rdApp.*")
    return Chem


def read_molecule(path_or_text):
    """Parse SMILES, InChI, a MOL block, SDF (first record), Mol2 or PDB from a path or the text itself."""
    Chem = _rdkit()
    if isinstance(path_or_text, Chem.Mol):
        return path_or_text
    text = _read_text(path_or_text, "molecule")
    stripped = text.strip()
    if not stripped:
        raise ValueError("the molecule input is empty")
    mol = None
    kind = None
    if "M  END" in text or re.search(r"V[23]000", text):
        kind = "MOL/SDF"
        block = text.split("$$$$", 1)[0]
        mol = Chem.MolFromMolBlock(block, sanitize=True, removeHs=True)
        if mol is None:
            mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
            if mol is not None:
                try:
                    mol.UpdatePropertyCache(strict=False)
                    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                except Exception:
                    mol = None
    elif stripped.startswith("InChI="):
        kind = "InChI"
        try:
            mol = Chem.MolFromInchi(stripped.split()[0])
        except Exception:
            mol = None
    elif "@<TRIPOS>MOLECULE" in text:
        kind = "Mol2"
        block = text.split("@<TRIPOS>MOLECULE")[1]
        mol = Chem.MolFromMol2Block("@<TRIPOS>MOLECULE" + block)
    elif re.search(r"(?m)^(ATOM  |HETATM)", text):
        kind = "PDB"
        mol = Chem.MolFromPDBBlock(text)
    else:
        kind = "SMILES"
        lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        if not lines:
            raise ValueError("the molecule input has no SMILES line")
        if "smiles" in lines[0].lower() and len(lines) > 1:
            lines = lines[1:]
        token = re.split(r"[\s,;]+", lines[0])[0]
        mol = Chem.MolFromSmiles(token)
        if mol is None:
            raise ValueError(f"could not parse {token[:80]!r} as SMILES (or InChI, MOL, SDF, Mol2, PDB)")
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"could not parse the input as {kind}")
    return mol


def _as_mol(mol_or_path):
    Chem = _rdkit()
    if isinstance(mol_or_path, Chem.Mol):
        return mol_or_path
    return read_molecule(mol_or_path)


def _count(mol, name: str) -> int:
    Chem = _rdkit()
    patt = _COMPILED.get(name)
    if patt is None:
        patt = _COMPILED[name] = Chem.MolFromSmarts(FUNCTIONAL_GROUPS[name])
    matches = mol.GetSubstructMatches(patt, uniquify=True, maxMatches=10_000)
    if name == "isoprene_unit":  # count disjoint C5 units only
        used: set[int] = set()
        n = 0
        for match in matches:
            if used.isdisjoint(match):
                used.update(match)
                n += 1
        return n
    if name in ("halide",):
        return len({m[1] for m in matches})
    if name in ("alcohol", "phenol", "thiol", "ether", "amine_primary", "amine_secondary", "amine_tertiary"):
        return len({m[0] for m in matches})
    return len(matches)


def molecule_summary(mol_or_path) -> dict:
    """Descriptors, element counts and SMARTS functional groups, plus a plain-English description."""
    Chem = _rdkit()
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    mol = _as_mol(mol_or_path)
    try:
        smiles = Chem.MolToSmiles(mol)
    except Exception:
        smiles = None
    elements: dict[str, int] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:
            elements[atom.GetSymbol()] = elements.get(atom.GetSymbol(), 0) + 1
    ordered = dict(sorted(elements.items(), key=lambda kv: (kv[0] not in ("C",), kv[0])))
    groups = {}
    for name in FUNCTIONAL_GROUPS:
        try:
            c = _count(mol, name)
        except Exception:
            c = 0
        if c:
            groups[name] = c
    ri = mol.GetRingInfo()
    aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
    if aromatic:
        groups["aromatic_ring"] = int(aromatic)
    carbons = elements.get("C", 0)
    only_cho = set(elements) <= {"C", "O"}
    terpene_like = bool(only_cho and carbons >= 10 and carbons % 5 == 0 and groups.get("isoprene_unit", 0) >= 2)
    out = {
        "canonical_smiles": smiles,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": _r(Descriptors.MolWt(mol), 2),
        "logp": _r(Crippen.MolLogP(mol), 2),
        "tpsa": _r(rdMolDescriptors.CalcTPSA(mol), 1),
        "h_bond_donors": int(Lipinski.NumHDonors(mol)),
        "h_bond_acceptors": int(Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "rings": int(ri.NumRings()),
        "aromatic_rings": int(aromatic),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "heavy_atoms_by_element": ordered,
        "formal_charge": int(Chem.GetFormalCharge(mol)),
        "fragments": len(Chem.GetMolFrags(mol)),
        "functional_groups": groups,
        "terpene_like": terpene_like,
    }

    elem_text = ", ".join(f"{v} {k}" for k, v in ordered.items())
    s = [f"A molecule with formula {out['formula']} (molecular weight {out['molecular_weight']:.2f}) and "
         f"{out['heavy_atoms']} heavy atoms ({elem_text})."]
    if out["fragments"] > 1:
        s.append(f"It has {out['fragments']} disconnected fragments (for example a salt or mixture).")
    if out["rings"]:
        s.append(f"It has {out['rings']} ring{'s' if out['rings'] != 1 else ''}, {aromatic} of them aromatic.")
    else:
        s.append("It has no rings (acyclic).")
    if groups:
        parts = [f"{_GROUP_WORDS.get(k, k.replace('_', ' '))} x{v}" for k, v in groups.items()]
        s.append("Functional groups found: " + ", ".join(parts) + ".")
    else:
        s.append("No listed functional groups were found (a plain hydrocarbon skeleton or an unusual group).")
    if terpene_like:
        s.append(f"Its {carbons} carbons and {groups['isoprene_unit']} branched C5 units are consistent with a "
                 f"terpene-like (isoprenoid) skeleton.")
    logp = out["logp"]
    lip = "very lipophilic" if logp > 5 else "lipophilic" if logp > 3 else "moderately lipophilic" if logp > 1 \
        else "fairly polar" if logp > -1 else "very polar"
    s.append(f"Calculated logP {logp:.2f} ({lip}), TPSA {out['tpsa']:.1f} square angstroms, "
             f"{out['h_bond_donors']} H-bond donor{'s' if out['h_bond_donors'] != 1 else ''} and "
             f"{out['h_bond_acceptors']} acceptor{'s' if out['h_bond_acceptors'] != 1 else ''}, "
             f"{out['rotatable_bonds']} rotatable bond{'s' if out['rotatable_bonds'] != 1 else ''}.")
    if out["formal_charge"]:
        s.append(f"Net formal charge {out['formal_charge']:+d}.")
    out["description"] = " ".join(s)
    return _plain(out)


def molecule_image(mol_or_path, size: int = 300) -> Image.Image:
    """A 2D drawing of the molecule as an RGB PIL image of size x size px."""
    size = int(size)
    if size < 64 or size > 2048:
        raise ValueError(f"size must be between 64 and 2048 px, got {size}")
    mol = _as_mol(mol_or_path)
    try:
        from rdkit.Chem import Draw
        img = Draw.MolToImage(mol, size=(size, size))
    except Exception as exc:
        raise ValueError(f"rdkit cannot draw molecules here (no drawing backend?): {exc}") from None
    return _to_image(img)


# ============================================================================= coordinates
_HEMI = re.compile(r"(?<![A-Za-z])([NSEWnsew])(?![A-Za-z])")
_PLAIN_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _check(lat: float, lon: float, where: str) -> tuple[float, float]:
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise ValueError(f"non-finite coordinate {where}")
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"latitude {lat} is outside -90..90 {where}")
    if not -180.0 <= lon <= 180.0:
        raise ValueError(f"longitude {lon} is outside -180..180 {where}")
    return (float(lat), float(lon))


def _dms_value(numbers: list[str], where: str) -> float:
    vals = [float(x.replace(",", ".")) for x in numbers]
    if not 1 <= len(vals) <= 3:
        raise ValueError(f"expected degrees[, minutes[, seconds]] {where}")
    deg = vals[0]
    sign = -1.0 if numbers[0].strip().startswith("-") else 1.0
    rest = 0.0
    if len(vals) > 1:
        if not 0 <= vals[1] < 60:
            raise ValueError(f"minutes {vals[1]} out of range {where}")
        rest += vals[1] / 60.0
    if len(vals) > 2:
        if not 0 <= vals[2] < 60:
            raise ValueError(f"seconds {vals[2]} out of range {where}")
        rest += vals[2] / 3600.0
    return sign * (abs(deg) + rest)


def _parse_hemisphere_line(line: str, where: str):
    clean = re.sub(r"[°º˚′’'″”\"`´:]", " ", line)
    tokens = re.findall(r"[-+]?\d+(?:\.\d+)?|(?<![A-Za-z])[NSEWnsew](?![A-Za-z])", clean)
    if not tokens:
        return None
    prefix = tokens[0].upper() in "NSEW"
    groups: list[tuple[str, list[str]]] = []
    current: list[str] = []
    letter = None
    for tok in tokens:
        if tok.upper() in ("N", "S", "E", "W"):
            if prefix:
                if letter is not None:
                    groups.append((letter, current))
                letter, current = tok.upper(), []
            else:
                groups.append((tok.upper(), current))
                current = []
        else:
            current.append(tok)
    if prefix and letter is not None:
        groups.append((letter, current))
    lat = lon = None
    for hemi, nums in groups:
        if not nums:
            continue
        value = abs(_dms_value(nums, where))
        if hemi in "NS":
            lat = value if hemi == "N" else -value
        else:
            lon = value if hemi == "E" else -value
    if lat is None or lon is None:
        return None
    return _check(lat, lon, where)


def _parse_plain_line(line: str, where: str, order: tuple[int, int] | None):
    labelled_lat = re.search(r"\blat(?:itude)?\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", line, re.I)
    labelled_lon = re.search(r"\b(?:lon|lng|long|longitude)\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", line, re.I)
    if labelled_lat and labelled_lon:
        return _check(float(labelled_lat.group(1)), float(labelled_lon.group(1)), where)
    if re.search(r"[°º′″]|\d'\s*\d", line):  # signed DMS without hemisphere letters: "51°30'26\", -0°7'39\""
        halves = [h for h in re.split(r"[,;\t]|\s{2,}", line) if h.strip()]
        if len(halves) == 2:
            nums = [re.findall(r"[-+]?\d+(?:\.\d+)?", re.sub(r"[°º˚′’'″”\"]", " ", h)) for h in halves]
            if all(nums):
                return _check(_dms_value(nums[0], where), _dms_value(nums[1], where), where)
    if order is not None:
        stripped = line.strip()
        fields = re.split(r"\s*[,;\t]\s*", stripped) if re.search(r"[,;\t]", stripped) else stripped.split()
        try:
            return _check(float(fields[order[0]]), float(fields[order[1]]), where)
        except (IndexError, ValueError):
            return None
    fields = [f for f in re.split(r"[,;\t]|\s+", line.strip()) if f]
    nums = []
    for f in fields:
        if re.fullmatch(_PLAIN_NUM, f):
            nums.append(float(f))
            if len(nums) == 2:
                break
    if len(nums) < 2:
        return None
    return _check(nums[0], nums[1], where)


def _header_order(line: str) -> tuple[int, int] | None:
    fields = [f.strip().strip('"\'').lower() for f in re.split(r"[,;\t]|\s+", line.strip()) if f.strip()]
    lat_i = next((i for i, f in enumerate(fields) if f in ("lat", "latitude", "y")), None)
    lon_i = next((i for i, f in enumerate(fields) if f in ("lon", "lng", "long", "longitude", "x")), None)
    if lat_i is None or lon_i is None:
        return None
    return (lat_i, lon_i)


def _geojson_points(obj, out: list) -> None:
    if isinstance(obj, list):
        for item in obj:
            _geojson_points(item, out)
        return
    if not isinstance(obj, dict):
        return
    kind = obj.get("type")
    if kind == "FeatureCollection":
        for f in obj.get("features") or []:
            _geojson_points(f, out)
    elif kind == "Feature":
        _geojson_points(obj.get("geometry"), out)
    elif kind == "GeometryCollection":
        for g in obj.get("geometries") or []:
            _geojson_points(g, out)
    elif kind in ("Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon"):
        def walk(c):
            if isinstance(c, list) and len(c) >= 2 and all(isinstance(v, (int, float)) for v in c[:2]):
                out.append(_check(float(c[1]), float(c[0]), f"in GeoJSON {kind} (positions are lon, lat)"))
            elif isinstance(c, list):
                for item in c:
                    walk(item)
        walk(obj.get("coordinates"))
    else:
        keys = {k.lower(): k for k in obj}
        la = next((keys[k] for k in ("lat", "latitude") if k in keys), None)
        lo = next((keys[k] for k in ("lon", "lng", "long", "longitude") if k in keys), None)
        if la and lo:
            out.append(_check(float(obj[la]), float(obj[lo]), "in JSON object"))


def _json_points(text: str) -> list:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"input looks like JSON but does not parse: {exc}") from None
    out: list = []
    is_geojson = isinstance(data, dict) and "type" in data
    if not is_geojson and isinstance(data, list) and data and all(
            isinstance(p, list) and len(p) >= 2 and all(isinstance(v, (int, float)) for v in p[:2]) for p in data):
        return [_check(float(p[0]), float(p[1]), "in JSON array (read as lat, lon)") for p in data]
    _geojson_points(data, out)
    return out


def _xml_points(text: str) -> list:
    try:
        root = ET.fromstring(text.encode("utf-8"))
    except ET.ParseError as exc:
        raise ValueError(f"input looks like XML (GPX/KML) but does not parse: {exc}") from None
    out = []
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1] if isinstance(el.tag, str) else ""
        if tag in ("trkpt", "wpt", "rtept") and "lat" in el.attrib and "lon" in el.attrib:
            out.append(_check(float(el.attrib["lat"]), float(el.attrib["lon"]), f"in GPX <{tag}>"))
        elif tag == "coordinates" and el.text:
            for tup in el.text.split():
                parts = tup.split(",")
                if len(parts) >= 2:
                    out.append(_check(float(parts[1]), float(parts[0]), "in KML <coordinates> (lon,lat)"))
        elif tag == "coord" and el.text:  # gx:coord "lon lat alt"
            parts = el.text.split()
            if len(parts) >= 2:
                out.append(_check(float(parts[1]), float(parts[0]), "in KML <gx:coord> (lon lat)"))
    return out


def read_coordinates(path_or_text) -> list:
    """(lat, lon) pairs from text lines, DMS, GeoJSON (lon, lat order), GPX or KML; ranges validated."""
    text = _read_text(path_or_text, "coordinate")
    body = text.strip()
    if not body:
        raise ValueError("the coordinate input is empty")
    if body[0] in "{[":
        points = _json_points(body)
        source = "JSON"
    elif body.startswith("<") or "<gpx" in body[:2000] or "<kml" in body[:2000]:
        points = _xml_points(body)
        source = "XML"
    else:
        points = []
        order = None
        source = "text"
        for no, raw in enumerate(body.splitlines(), 1):
            line = re.sub(r"^geo:", "", raw.strip(), flags=re.I)
            if not line or line.startswith(("#", "//")):
                continue
            where = f"on line {no}"
            if not points and order is None:
                order = _header_order(line)
                if order is not None:
                    continue
            pt, dms_error = None, None
            if _HEMI.search(re.sub(r"\d[eE][-+]?\d", "", line)):
                try:
                    pt = _parse_hemisphere_line(line, where)
                except ValueError as exc:  # a stray standalone letter ("Point E 51.5 -0.1"): try plain numbers
                    dms_error = exc
            if pt is None:
                pt = _parse_plain_line(line, where, order)
            if pt is None and dms_error is not None:
                raise dms_error
            if pt is not None:
                points.append(pt)
    if not points:
        raise ValueError(f"no coordinates found in the {source} input")
    return [(round(a, 7), round(b, 7)) for a, b in points]


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (numpy arrays broadcast)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    d = 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return float(d) if np.ndim(d) == 0 else d


_COMPASS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")
_GAZETTEER_FILE = "ne_10m_populated_places_simple.geojson"
_GAZETTEER_CACHE: dict = {}


def _gazetteer_path(gazetteer=None) -> Path:
    if gazetteer:
        return Path(os.fspath(gazetteer)).expanduser()
    env = os.environ.get("SENSE_GAZETTEER")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "sense-harness" / "gazetteer" / _GAZETTEER_FILE


def _text_or_none(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text in ("", "-99", "None") else text


def _load_gazetteer(path: Path) -> dict:
    """Natural Earth populated places (GeoJSON FeatureCollection of points) as numpy columns, cached per file."""
    if not path.is_file():
        raise ValueError(f"gazetteer file not found at {path}; run `python scripts/fetch_gazetteer.py` to download "
                         f"Natural Earth populated places (or set SENSE_GAZETTEER to its {_GAZETTEER_FILE})")
    st = path.stat()
    key = (str(path.resolve()), st.st_mtime_ns, st.st_size)
    cached = _GAZETTEER_CACHE.get(key)
    if cached is not None:
        return cached
    if st.st_size > 200_000_000:
        raise ValueError(f"gazetteer file {path} is too large ({st.st_size} bytes)")
    try:
        data = json.loads(path.read_bytes().decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        data = None
    if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
        raise ValueError(f"gazetteer file {path} is not a GeoJSON FeatureCollection; expected Natural Earth "
                         f"populated places ({_GAZETTEER_FILE}, from scripts/fetch_gazetteer.py)")
    cols: dict[str, list] = {k: [] for k in ("name", "country", "country_code", "admin1", "population",
                                             "feature_class", "lat", "lon")}
    for feature in data.get("features") or []:
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates") if geom.get("type") == "Point" else None
        try:
            if coords and len(coords) >= 2:
                la, lo = float(coords[1]), float(coords[0])  # GeoJSON order is lon, lat
            else:
                la, lo = float(props["latitude"]), float(props["longitude"])
            _check(la, lo, "")
        except (KeyError, TypeError, ValueError):
            continue
        name = _text_or_none(props.get("name")) or _text_or_none(props.get("nameascii"))
        if not name:
            continue
        try:
            population = int(props.get("pop_max") or 0)
        except (TypeError, ValueError):
            population = 0
        cols["name"].append(name)
        cols["country"].append(_text_or_none(props.get("adm0name")))
        cols["country_code"].append(_text_or_none(props.get("iso_a2")))
        cols["admin1"].append(_text_or_none(props.get("adm1name")))
        cols["population"].append(max(0, population))
        cols["feature_class"].append(_text_or_none(props.get("featurecla")))
        cols["lat"].append(la)
        cols["lon"].append(lo)
    if not cols["name"]:
        raise ValueError(f"gazetteer file {path} has no named point features")
    table = dict(cols, population=np.array(cols["population"], dtype=np.int64), lat=np.array(cols["lat"]),
                 lon=np.array(cols["lon"]))
    _GAZETTEER_CACHE.clear()
    _GAZETTEER_CACHE[key] = table
    return table


def _bearing(lat1, lon1, lat2, lon2) -> float:
    """Initial bearing in degrees from point 1 to point 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def nearest_places(lat: float, lon: float, k: int = 3, gazetteer=None) -> list:
    """The k nearest gazetteer places, nearest first, with distance and compass direction."""
    lat, lon = _check(float(lat), float(lon), "for nearest_places")
    k = int(k)
    if k < 1 or k > 100:
        raise ValueError(f"k must be between 1 and 100, got {k}")
    table = _load_gazetteer(_gazetteer_path(gazetteer))
    d = haversine_km(lat, lon, table["lat"], table["lon"])
    k = min(k, len(d))
    idx = np.argpartition(d, k - 1)[:k] if k < len(d) else np.arange(len(d))
    idx = idx[np.argsort(d[idx], kind="stable")]
    out = []
    for i in idx.tolist():
        b = _bearing(table["lat"][i], table["lon"][i], lat, lon)
        out.append({"name": table["name"][i], "country": table["country"][i],
                    "country_code": table["country_code"][i], "admin1": table["admin1"][i],
                    "population": int(table["population"][i]), "feature_class": table["feature_class"][i],
                    "distance_km": round(float(d[i]), 1), "bearing": _COMPASS[int((b + 11.25) // 22.5) % 16],
                    "bearing_deg": round(b, 1), "lat": float(table["lat"][i]), "lon": float(table["lon"][i])})
    return out


def _dms(value: float, pos: str, neg: str) -> str:
    hemi = pos if value >= 0 else neg
    v = abs(value)
    d = int(v)
    m = int((v - d) * 60)
    s = (v - d - m / 60.0) * 3600
    if s >= 59.95:
        s, m = 0.0, m + 1
    if m >= 60:
        m, d = 0, d + 1
    return f"{d}°{m:02d}'{s:04.1f}\"{hemi}"


def _km(d: float) -> str:
    return f"{d:.1f} km" if d < 10 else f"{d:,.0f} km"


def coordinate_summary(lat: float, lon: float, k: int = 3, gazetteer=None) -> dict:
    """Hemisphere, a rough latitude climate band and the nearest places, with a plain-English description."""
    lat, lon = _check(float(lat), float(lon), "for coordinate_summary")
    a = abs(lat)
    if a < 23.44:
        band = "tropical"
    elif a < 35:
        band = "subtropical"
    elif a < 55:
        band = "temperate"
    elif a < 66.56:
        band = "subpolar"
    else:
        band = "polar"
    ns = "northern" if lat >= 0 else "southern"
    ew = "eastern" if lon >= 0 else "western"
    out = {"lat": round(lat, 6), "lon": round(lon, 6), "dms": f"{_dms(lat, 'N', 'S')} {_dms(lon, 'E', 'W')}",
           "hemisphere": f"{ns}, {ew}", "climate_band": band, "nearest_places": None, "gazetteer_note": None}
    try:
        out["nearest_places"] = nearest_places(lat, lon, k=k, gazetteer=gazetteer)
    except ValueError as exc:
        out["gazetteer_note"] = str(exc)
    s = [f"The point {out['dms']} ({lat:.5f}, {lon:.5f}) lies in the {ns} and {ew} hemispheres, at a latitude "
         f"whose rough climate band is {band} (latitude alone; altitude, oceans and local geography change this)."]
    places = out["nearest_places"]
    if places:
        first = places[0]
        region = ", ".join(x for x in (first["admin1"], first["country"] or first["country_code"]) if x)
        where = f"{first['name']} ({region})" if region else first["name"]
        pop = f", population {first['population']:,}" if first["population"] else ""
        s.append(f"The nearest place in the gazetteer (Natural Earth populated places: about 7,300 cities and "
                 f"towns) is {where}{pop}; the point is {_km(first['distance_km'])} {first['bearing']} of it.")
        if len(places) > 1:
            rest = "; ".join(f"{p['name']} ({p['country'] or p['country_code'] or 'unknown country'}) "
                             f"{_km(p['distance_km'])}" for p in places[1:])
            s.append(f"Next nearest: {rest}.")
        if first["distance_km"] > 300:
            s.append("No gazetteer place lies within 300 km, so the point is remote (for example open ocean, "
                     "desert, mountains or polar ice).")
    elif out["gazetteer_note"]:
        s.append("No gazetteer is available, so nearby places are unknown.")
    out["description"] = " ".join(s)
    return _plain(out)
