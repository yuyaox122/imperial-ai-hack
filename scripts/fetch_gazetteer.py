"""Download the Natural Earth populated-places gazetteer that sense/adapter_kit.py uses to name places near a point.

What it does
    Downloads ne_10m_populated_places_simple.geojson (Natural Earth 1:10m populated places: about 7,300 cities
    and towns worldwide, with name, country, ISO code, admin-1 region and population) from
    https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_populated_places_simple.geojson
    checks that it is a GeoJSON FeatureCollection of named points, and writes SOURCE.txt next to it with the URL,
    licence, download date, feature count and SHA-256. The same data is published as a shapefile at
    https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places_simple.zip; the kit reads the GeoJSON.

Why Natural Earth and not GeoNames
    The coordinate probe set (the labelled files the harness verifies a coordinate sense on) is built from
    GeoNames cities15000. If the adapter's gazetteer were that same table, every probe point would match a
    gazetteer entry at 0 km and verification would prove nothing. The gazetteer must come from an independent
    source, so it is Natural Earth.

Where the file goes
    ~/.cache/sense-harness/gazetteer/ by default, or --dest DIR. adapter_kit.nearest_places reads
    $SENSE_GAZETTEER when it is set, else ~/.cache/sense-harness/gazetteer/ne_10m_populated_places_simple.geojson,
    so point SENSE_GAZETTEER at <DIR>/ne_10m_populated_places_simple.geojson when you use --dest.

Idempotent
    When the GeoJSON and SOURCE.txt are already there it does nothing (exit 0); --force downloads again. The file
    is written under a temporary name and renamed into place only after it validates, so an interrupted run
    leaves no half-written gazetteer.

Licence
    Natural Earth data is in the public domain (https://www.naturalearthdata.com/about/terms-of-use/). No
    attribution is required; "Made with Natural Earth" is appreciated.

Usage
    python scripts/fetch_gazetteer.py [--dest DIR] [--force] [--url URL]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
       "ne_10m_populated_places_simple.geojson")
SHAPEFILE_URL = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places_simple.zip"
FILENAME = "ne_10m_populated_places_simple.geojson"
LICENCE = "public domain (https://www.naturalearthdata.com/about/terms-of-use/)"
ATTRIBUTION = "Made with Natural Earth (https://www.naturalearthdata.com/); no attribution required"
WHY = ("The coordinate probe labels are built from GeoNames cities15000. A gazetteer taken from that same table "
       "would match every probe point at 0 km and verification would prove nothing, so the adapter kit's "
       "gazetteer comes from an independent source: Natural Earth. Do not replace this file with GeoNames data.")
DEFAULT_DEST = Path.home() / ".cache" / "sense-harness" / "gazetteer"
MAX_BYTES = 128 * 1024 * 1024
MIN_FEATURES = 1000


def _download(url: str, target: Path, timeout: float = 120.0) -> int:
    request = urllib.request.Request(url, headers={"User-Agent": "sense-harness-fetch-gazetteer/1.0"})
    size = 0
    with urllib.request.urlopen(request, timeout=timeout) as response, open(target, "wb") as out:
        while True:
            chunk = response.read(1 << 16)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_BYTES:
                raise RuntimeError(f"download is larger than {MAX_BYTES} bytes; refusing to continue")
            out.write(chunk)
    return size


def _validate(path: Path) -> int:
    """Check the file is a GeoJSON FeatureCollection of named points with populations; return the feature count."""
    try:
        data = json.loads(path.read_bytes().decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"the download is not valid GeoJSON: {exc}") from None
    if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
        raise RuntimeError("the download is not a GeoJSON FeatureCollection")
    features = data.get("features") or []
    if len(features) < MIN_FEATURES:
        raise RuntimeError(f"the download has only {len(features)} features (expected about 7,300)")
    named = 0
    for f in features:
        geom, props = f.get("geometry") or {}, f.get("properties") or {}
        coords = geom.get("coordinates") or []
        if geom.get("type") == "Point" and len(coords) >= 2 and props.get("name") and "pop_max" in props:
            named += 1
    if named < 0.95 * len(features):
        raise RuntimeError(f"only {named} of {len(features)} features are named points with pop_max")
    return len(features)


def fetch(dest: Path, force: bool = False, url: str = URL) -> int:
    dest = dest.expanduser()
    final, source = dest / FILENAME, dest / "SOURCE.txt"
    if not force and final.is_file() and final.stat().st_size > 0 and source.is_file() \
            and FILENAME in source.read_text(encoding="utf-8", errors="replace"):
        print(f"already present: {final} (use --force to download again)")
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    fd, tmp_name = tempfile.mkstemp(prefix=".ne_places.", suffix=".tmp", dir=dest)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        size = _download(url, tmp)
        count = _validate(tmp)
        sha = hashlib.sha256(tmp.read_bytes()).hexdigest()
        os.chmod(tmp, 0o644)  # mkstemp makes 0600; sandboxed adapters may run as another user
        os.replace(tmp, final)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    text = (f"file: {FILENAME}\nurl: {url}\nalso published as a shapefile: {SHAPEFILE_URL}\n"
            f"downloaded: {today}\nbytes: {size}\nfeatures: {count}\nsha256: {sha}\n"
            f"licence: {LICENCE}\nattribution: {ATTRIBUTION}\n"
            f"fields used by sense/adapter_kit.py: name, adm0name (country), iso_a2 (country code), adm1name "
            f"(admin-1 region), pop_max (population), featurecla, latitude/longitude (or the Point geometry)\n"
            f"why not GeoNames: {WHY}\n")
    tmp_source = dest / ".SOURCE.txt.tmp"
    tmp_source.write_text(text, encoding="utf-8")
    os.replace(tmp_source, source)
    print(f"wrote {final} ({count:,} places, {size:,} bytes) and {source}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fetch_gazetteer.py",
        description="Download the Natural Earth 1:10m populated places (public domain; about 7,300 cities and "
                    "towns) for sense/adapter_kit.py's nearest_places and coordinate_summary. Natural Earth, not "
                    "GeoNames, because the coordinate probe labels come from GeoNames cities15000: a gazetteer "
                    "from the same table would match every probe at 0 km and prove nothing.",
        epilog=f"Writes {FILENAME} and SOURCE.txt (URL, licence, date, feature count, SHA-256) into the "
               f"destination. Does nothing when both are already there unless --force. adapter_kit reads "
               f"$SENSE_GAZETTEER or {DEFAULT_DEST / FILENAME}.")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help=f"directory to write into (default: {DEFAULT_DEST})")
    parser.add_argument("--force", action="store_true", help="download again even if the file is present")
    parser.add_argument("--url", default=URL, help="GeoJSON URL to fetch (default: the Natural Earth repository)")
    args = parser.parse_args(argv)
    try:
        return fetch(args.dest, force=args.force, url=args.url)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"fetch_gazetteer: failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
