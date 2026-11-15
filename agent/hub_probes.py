"""Find labelled probe files for a new kind of input on the Hugging Face Hub, and in table files on GitHub or Zenodo.

An adapter turns an input file (a 3D model, a video, a molecule, a pair of coordinates, ...) into text or images,
and the harness verifies it on labelled probe files. The labels must come from an independent source, so this
module finds them on the public Hub: it searches the dataset index, reads a dataset's columns through the public
dataset viewer, picks an input column and a label column, and writes a capped sample as one sub-folder per label
(the folder format agent.data.build_set and sense.adapters.probe_folder read). For some inputs the best labelled
data is a plain table file instead (an odour table as a CSV on GitHub, a city list with a country column), so
find_probes also asks a Linkup web search (agent.research) for datasets: the Hub ids it names join the Hub
candidates and its file URLs become table candidates, read with the same column, label and honesty rules. Many Hub
datasets keep their files in one folder per label without dataset-viewer support, so when a Hub candidate's viewer
fails, find_probes reads it as a folder-labelled repository (a FolderCandidate): its file list from the Hub tree API,
its labels from its folder names. The Hub search needs no API key; without the web search (no LINKUP_API_KEY, fake
mode, no network) the Hub is searched alone.

Public surface (do not rename):
  ProbesUnavailable                 raised when no usable probe set can be had; the message says why
  HubCandidate                      hub_id, title, downloads, likes, licence, tags, url, gated, score, found_by,
                                    revision; to_dict()
  TableCandidate                    url, title, licence, licence_source, label_hint, page_url, given_url, score;
                                    to_dict()
  FolderCandidate                   hub_id, title, licence, tags, url, revision, downloads, found_by, score;
                                    FolderCandidate.of(hub_candidate); to_dict()
  Mapping                           which columns are input and label (a folder: label_kind "folder" and
                                    folder_depth); to_dict()
  FetchReport                       folder, labels, counts, rows_seen, skipped, licence, source_url, notes, hub_id,
                                    label_values, bytes, licence_source
  SEARCH_SCHEMA, MAPPING_SCHEMA     structured-output schemas for agent.llm.ask_json
  WEB_SCHEMA                        the structured-output schema of the Linkup search (agent.research)
  search_hub(goal, modality, *, limit=20) -> list[HubCandidate]
  choose_mapping(candidate, goal, modality, *, wanted_labels=None, budget_s=None, min_per_label=6) -> Mapping
  fetch_probes(candidate, mapping, dest, *, wanted_labels=None, n_per_label=20, max_labels=8, min_per_label=6,
               budget_s=120, max_bytes=150_000_000, disjoint_from=()) -> FetchReport
  find_probes(goal, modality, dest_root, *, wanted_labels=None, emit=None, max_candidates=3, max_inspect=12,
              total_budget_s=None, exclude_sources=(), disjoint_from=(), **fetch_kwargs) -> (folder, info)
  safe_label(text, index) -> str
choose_mapping and fetch_probes take a HubCandidate, a TableCandidate or a FolderCandidate (choose_mapping's
min_per_label applies to a FolderCandidate only).

modality is "audio", "image", "video", "3d", "chemical", "coordinates" or "text", or an alias such as smell, taste,
molecule, smiles, location, place, gps or mesh; any other word still works with the generic rules.

search_hub asks agent.llm.ask_json for up to 5 short queries (SEARCH_SCHEMA); when it answers None (fake mode, no
key, any failure) or nonsense, a rule builds up to 6: the goal's first two content words together, then the
modality's own search words (odor/smell/olfaction for molecules, cities/geonames/country for coordinates,
modelnet/objects/shapenet for 3D, action/ucf101/kinetics/hmdb for video) interleaved with the goal's words. A word
that only names the input (molecule, place, video, 3d, ...) is searched only inside the first query. Each query is
one GET of https://huggingface.co/api/datasets?search=..&limit=..&full=true&sort=downloads. Results are merged and
deduplicated; gated, private and disabled datasets, those whose card turns the viewer off and ids that are not
plain owner/name are dropped. Order: relevance (2 for each topic word and 1 for each input word found in the
dataset's id or title, its words taken with camelCase split and whole so "ModelNet40" names modelnet, plus
MODALITY_TAG_BONUS for the Hub's modality tag of this modality and 1 for a *-classification task tag), then how many
queries found it, then downloads. The licence is cardData.license (with cardData.license_name and license_link in
brackets when the card gives them), else a "license:" tag, else None; revision is the record's sha (its commit).

choose_mapping reads the viewer's /splits (config "default" if there is one, split train, else test, validation)
and /first-rows (else the first /rows page) for the real features and sample values. It asks ask_json
(MAPPING_SCHEMA) which columns are input and label, then validates the answer against those features: the columns
exist, the label column is not an input, the label is a ClassLabel, a low-cardinality string, a list of strings, a
";", "|", "," or "/" delimited string list (for "," and "/" only when the parts also stand alone in the column) or a
JSON/Python list literal, with at least 2 values; a media input is one viewer media column (Audio, Image, Video,
Mesh) of the modality's type that links to https Hugging Face URLs; text inputs are string or number columns, for
a modality with typed input slots (SMILES; latitude and longitude) exactly one column per slot, named for it with
fitting values, so no extra column can carry the label; and a text template names exactly the input columns. Any
text mapping, the model's or the rule's, is refused when an input column determines the label (at most half as
many distinct values as rows, each with one label in 90% of its rows: a country name beside a country code). An
answer that fails is ignored and the rule decides: a media column whose type fits the modality, else string or
number columns named for it with fitting values (smiles... with SMILES-like strings; lat/latitude and
lon/lng/longitude within range; text...). The label is the best column named for the goal's topic (the goal's
words and the modality's label words such as genre, action, odor, descriptor, country, less the input words). It
ranks columns by score: the goal's focus (the first topic word after what/which/whose) beats its later words, a
column that only adds words to a bare topic column (a model's "qwen_emotion" beside "emotion", so the exact name
ranks above it) and one with prediction-like words (pred, model, score, prob, logit, ...) lose points, and ties go
to the column with fewer words, then the first. A generic column (label/class/category/target/tags, any
ClassLabel) or, last, a block of at least 3 columns holding only 0/1 ("multihot": one column per odour, say) counts
only when the dataset's own id or title names the topic, or its values name the goal's words or the wanted labels,
so an unrelated benchmark's labels are never taken for the goal's; the refusal says which ClassLabel or label
column was not trusted and which label columns hold bare integers. When the first rows show no usable label (a
split sorted by label), one /rows page from the middle of the split is read too. No usable mapping raises
ProbesUnavailable with the reason.

fetch_probes owns dest: it replaces it, deletes it again on any failure, and writes only
dest/<safe_label>/<row index><extension>, so no label or URL can place a file elsewhere. It reads up to 12 /rows
pages of 100 rows spread evenly over the split. If the viewer refuses /rows for good (an HTTP 500 such as its
Parquet scan-size limit, or a 4xx) only /first-rows is read, which the notes say; if page 0 failed for a passing
reason (a timeout, 429 or 5xx gateway error) the row count comes from /size and the other pages are still read.
Once a page fails even when retried, later pages get one try each. For media, reading /rows ends at half the
budget (each request's timeout is cut to that point) so the downloads keep the rest. Rows that repeat an earlier row's
input are merged into it (skipped["duplicate_input"]), and an input whose rows carry different labels (long-format
data) is treated as multi-labelled. Then labels are picked: the wanted ones (matched case-insensitively, with
space, "_" and "-" alike, against the dataset's own values; all must be present) or else the most frequent values
with at least min_per_label rows, at most max_labels, never a placeholder such as unknown, none or other, and never
a value that combines other values ("fruity, green" beside "fruity"). For list and multihot labels, and for inputs
merged from rows with different labels, only inputs carrying exactly one of the chosen labels are kept (other,
unchosen labels are ignored; skipped["multi_label"] or ["conflicting_labels"]), and the notes say so. Text inputs
are written from the template after a leak check: a row whose input text contains its label as a word (any case;
"hip hop" also as hip-hop or hiphop; a label of digits only not inside a decimal such as "1.5") is dropped, and more
than half of the chosen labels' rows leaking refuses the dataset. Media cells are downloaded DOWNLOAD_WORKERS at a
time, the label with the fewest files first, only from https URLs on huggingface.co or hf.co hosts (redirects too),
each at most FILE_MAX_BYTES and all written files together at most max_bytes, within budget_s; failures are skipped
and counted by reason. Each file then has the metadata that can name its label stripped: names and extras in a
GLB/glTF, OBJ and OFF comments and OBJ object, group and material names, PLY comment lines, an STL's solid name (or a
binary STL's header), a molfile's title and comment lines and SDF data fields, SMILES names, GeoJSON properties, KML
and GPX names, descriptions and metadata, PNG text chunks, JPEG EXIF/XMP/IPTC/comment segments, and for audio and
video an ffmpeg stream-copy remux without global, stream or chapter metadata (when ffmpeg is installed). A file
whose remaining text runs (ASCII or UTF-16, first and last SCAN_BYTES) still name its own label (such as a source
path ...\\drink\\...) is dropped as skipped["leak"], and more than half of the downloaded files leaking refuses the
dataset (checked early after LEAK_SAMPLE files). A file whose bytes were already written under the same label is skipped as a duplicate, and
identical bytes under two labels are removed from both (skipped["conflicting_labels"]). Labels with fewer than
min_per_label files are dropped; fewer than 2 left, or a wanted label missing, raises ProbesUnavailable. Labels
are always the dataset's own values, never generated.

fetch_probes' disjoint_from: labels never chosen (matched with case, spaces, "_", "-" and other separators and a
plural "s" ignored, so "Hip-Hop", "hip_hop" and "hiphops" are one; codes and names are not matched: "FR" is not
"France"). A row carrying one counts only for another label it carries, like any unchosen label, and at least 2
labels must be left; a wanted label that is also in disjoint_from is refused.

The web search. find_probes runs research.search_structured beside the Hub search, once per goal and modality in a
process (later calls reuse the answer: info["web"]["origin"] "memo"), with WEB_SCHEMA and include_domains
WEB_DOMAINS (huggingface.co, github.com, raw.githubusercontent.com, zenodo.org, gist.githubusercontent.com),
asking for up to 8 labelled datasets for the goal and the modality's input, each with a Hugging Face dataset id or
a direct download URL of a data file, its licence, its label column and its input column (MAX_WEB_ITEMS are read).
When it cannot answer (no LINKUP_API_KEY and nothing cached, fake mode, the network, no time, any failure) the Hub
is searched alone and info["web"]["reason"] says why. A Hub id (owner/name, or a huggingface.co/datasets/... URL)
joins the Hub candidates through its Hub record (/api/datasets/<id>), so gated, private and viewer-less datasets
are dropped as in the search; a file URL becomes a TableCandidate. File URLs must be https on github.com (a blob or
raw URL, rewritten to raw.githubusercontent.com), raw.githubusercontent.com, gist.githubusercontent.com, zenodo.org
(a .../files/<name> URL) or huggingface.co (a dataset blob or resolve URL), with no user, password or other port,
and name a .csv, .tsv, .tab, .json, .jsonl, .ndjson, .txt, .zip or .gz file (or no suffix); anything else, and any
table for a media modality (a table gives text inputs only), is refused into info["web"]["refused"] before any
request. Hub and table candidates are ranked together (_ranked: the Hub's relevance; a table's URL path and title
words count as an id and title; a web search's recommendation adds 1), and up to max_inspect // 2 of those the web
search named are always among the max_inspect inspected.

Table candidates. choose_mapping downloads the file once (_table_get: at most TABLE_MAX_BYTES, within
TABLE_TIMEOUT_S or the time left, https on the hosts above only, redirects too), unpacks a .zip (its first member
named like a table file, skipping folders, __MACOSX and hidden files) or a .gz, refusing over TABLE_MAX_BYTES
unpacked too, and parses it with the standard library: a CSV or TSV with a header row (dialect sniffed among ",",
tab, ";" and "|", checked against the rows), a JSON list of objects (or the one list of objects in a JSON object)
or JSON Lines; an HTML page is refused. At most MAX_TABLE_ROWS rows (fewer for wide tables: MAX_TABLE_CELLS) are
kept, spread evenly over the file, so a file sorted by label keeps every label; a CSV column whose every value is a
number becomes numbers (0/1 blocks, coordinates). Then the columns are chosen exactly as for a Hub dataset, on
MAPPING_ROWS rows spread over the table: the language model's mapping validated against the real columns, else the
rules (typed slots such as SMILES and latitude/longitude; a label column named for the goal's topic; a generic
label column or a block of 0/1 columns only when the source's own name, its host, path and archive member, never
the web search's title, names the topic), with the same refusal of an input column that determines the label.
fetch_probes then reads every kept row through the same steps as a Hub dataset's rows: duplicate inputs merged,
conflicting labels dropped, multi-label rows (list, list-literal, separator-joined and 0/1 labels) kept only with
exactly one chosen label, the leak check for text inputs, and the same folder layout written from the template
(row numbers count data rows from 0). The licence is always reported and never invented: the web search's answer
when it gave one, else the first line of the repository's LICENSE (LICENSE.md, LICENSE.txt, LICENCE, COPYING) file
for a GitHub file, the record's licence for a Zenodo file or the dataset card's for a Hugging Face file, else
"unknown"; FetchReport.licence_source says where it was read.

Folder-labelled repositories (FolderCandidate). The files come from the Hub tree API,
GET https://huggingface.co/api/datasets/<id>/tree/<revision>?recursive=true (no key), paged through its Link
header's rel="next" URL (which must stay on the same huggingface.co path), at most TREE_MAX_PAGES pages within
TREE_BUDGET_S; a cut listing is said in the notes. The revision is the candidate's commit (the search record's sha),
else the Hub record's, else "main" (said in the notes). The input files are those with the modality's extensions
(_FOLDER_EXT: 3d .off .obj .stl .ply .glb .gltf; video .mp4 .avi .mov .webm .mkv; audio .wav .flac .mp3 .ogg; image
.png .jpg .jpeg; molecule .mol .sdf .smi; coordinates .geojson .gpx .kml; text has none), skipping paths with
hidden, empty, "." or ".." parts. Each file's label is the nearest folder above it whose name is neither a split or
generic name (train, test, val, valid, validation, dev, data, files, raw, all, images, clips, ..., numbered folds,
parts and splits, numbers) nor an id (mostly digits, a hash or UUID, a numbered item such as chair_0001), so
ModelNet's <class>/<split>/... and UCF101's <split>/<Class>/... both label by class. The repository's label depth is
the depth most of those files share (a tie goes to the deeper); it must hold more than half of them and give at
least 2 labels with min_per_label files, else ProbesUnavailable says why (folder names that look like ids are
named). The labels are trusted only when the repository's id or title names the goal's topic (the same rule as for
a generic label column, so an unrelated repository's folders never label the goal) or the folder names name the
goal's words or the wanted labels. fetch_probes then takes the files at that depth (a file the list says is over
FILE_MAX_BYTES is skipped without a request), picks labels as for a viewer dataset (wanted, else most frequent;
disjoint_from with the same loose matching and country aliases), picks each label's files evenly spread over them
from a point seeded by the repository and revision, and downloads them only from
https://huggingface.co/datasets/<id>/resolve/<revision>/<path> (redirects to the Hub's CDN hosts allowed), with
the same per-file and total byte caps, time budget, metadata stripping and leak scan as viewer media, into
dest/<safe label>/<n><extension> (n counts the listed files; never the original file name). The source is the Hub
id (info["source"], what exclude_sources takes; "<id>@<revision>" works too) plus info["revision"], and
FetchReport.source_url is the tree URL at that revision. The licence is the card's or its licence tag's, as for a
Hub dataset; the notes say labels came from folder names and quote what the dataset card (README.md at the
revision, at most LICENCE_MAX_BYTES) says: the other sites it links to (whose own terms may apply to re-hosted
files, which the card's licence may not cover) and up to 3 of its lines on licences, terms, academic or
non-commercial use or copyright. Nothing about the terms is invented.

find_probes searches, then inspects candidates in rank order with choose_mapping (at most max_inspect: a look at
the columns costs two cached viewer calls, or a table's download, and most search hits do not fit) and runs
fetch_probes into dest_root/<owner>__<name> (a table: dest_root/web__<file stem>__<url hash>) for at most
max_candidates of those whose columns fit. When a Hub candidate's inspection fails for any reason other than a
missing label column (the viewer fails, or shows no input column of the modality, or its media cells link nowhere
usable), the same candidate is inspected as a folder-labelled repository in the same turn (no extra place among
max_inspect; its fetch counts toward max_candidates), and its tried reason gives both failures. The whole step has
one deadline, total_budget_s (default INSPECT_ALLOWANCE_S plus max_candidates fetch budgets): the searches and
inspections run within it and each fetch gets the smaller of its budget_s and the time left. exclude_sources skips
candidates whose Hub id, dataset page or file URL (a GitHub blob URL and its raw file alike) is listed, files in the
same folder as a listed file (other files of the same dataset) and files under a listed repository or folder URL;
disjoint_from goes to fetch_probes. For the unseen-categories control, the agent calls find_probes again after the
main probe set, with exclude_sources=[info["source"]] and disjoint_from=list(info["label_values"].values()), and gets
an independent set whose categories the adapter has not seen. emit (if given) is called with small
JSON-serialisable dicts {"action": "search" | "candidate" | "inspect" | "fetch", "status": "start" | "ok" | "fail",
...}; candidate, inspect and fetch events carry hub_id ("" for a table), source and kind ("hub" | "table" |
"folder": a folder attempt's events after the viewer's inspect fail), and the search's "ok" event the web search's
summary and the excluded sources. It returns the folder and an info dict (hub_id, url, title, licence,
licence_source, kind, source: the Hub id or table URL to pass to exclude_sources, revision: a folder's commit, else
None, labels, label_values, counts, mapping, rows_seen, skipped, queries, query_origin, tried: [{hub_id, reason}, or
for a table {hub_id: "", url, reason}], notes, web: {used, reason, origin, hub_ids, tables, refused}, excluded:
[{source, reason}], disjoint_from), or raises ProbesUnavailable naming every candidate's reason and, when it was not
used, why the web search was not.

Every Hub and viewer request and media download goes through _http_get (User-Agent "sense-harness", a timeout
checked before every read, a byte cap, https Hugging Face hosts only), a tree page through _http_get_page (the same,
also returning the Link header), and every table file and licence request through _table_get (the same, on the
table hosts above), so tests replace those three functions. A JSON request
answered HTTP 429, 502, 503 or 504 is retried after RETRY_WAITS (or the server's Retry-After, at most
MAX_RETRY_WAIT), never past the caller's deadline; an HTTP 500 is the viewer's own refusal and is not retried. The
Hub, the viewer, GitHub and Zenodo need no API key; the web search uses research's LINKUP_API_KEY when it is set.
"""

from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import http.client
import io
import itertools
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
import zlib
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from concurrent.futures import TimeoutError as _FutureTimeout
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable

from agent import llm, research

HUB_SEARCH = "https://huggingface.co/api/datasets"
VIEWER = "https://datasets-server.huggingface.co"
USER_AGENT = "sense-harness"
TIMEOUT_S = 20.0
ROWS_TIMEOUT_S = 40.0  # the viewer's /rows reads parquet on demand and can be slow on large splits
JSON_MAX_BYTES = 50_000_000
FILE_MAX_BYTES = 25_000_000
PAGE_ROWS = 100  # the most rows the dataset viewer returns in one page
MAX_PAGES = 12
MAX_QUERIES = 6
MAX_TEXT_CHARS = 20_000
DOWNLOAD_WORKERS = 4
SCAN_BYTES = 262_144  # containers keep their metadata near the start or the end: the leak scan reads this much of each
LEAK_SAMPLE = 12  # files scanned before a dataset whose files mostly name their label is refused early
GRACE_S = 10.0  # how long past the budget a running download may take before it is abandoned
FFMPEG_TIMEOUT_S = 60.0
INSPECT_ALLOWANCE_S = 120.0  # find_probes' default time for the search and inspections, on top of the fetches
TABLE_MAX_BYTES = 50_000_000  # a table file, downloaded or unpacked, is refused above this size
TABLE_TIMEOUT_S = 60.0  # one table download
LICENCE_MAX_BYTES = 200_000
LICENCE_TIMEOUT_S = 10.0
MAX_TABLE_ROWS = 20_000  # rows kept from one table, evenly spread over the file
MAX_TABLE_CELLS = 2_000_000  # fewer rows are kept from a wide table
MAX_TABLE_COLUMNS = 300
MAPPING_ROWS = 500  # rows of a table the column choice looks at, evenly spread
WEB_TIMEOUT_S = 30.0  # the Linkup search
MAX_WEB_ITEMS = 8  # datasets read from one web search answer
WEB_DOMAINS = ("huggingface.co", "github.com", "raw.githubusercontent.com", "zenodo.org",
               "gist.githubusercontent.com")
TABLE_HOSTS = frozenset(("github.com", "raw.githubusercontent.com", "gist.githubusercontent.com", "zenodo.org"))
TABLE_SUFFIXES = (".csv", ".tsv", ".tab", ".json", ".jsonl", ".ndjson")  # .txt is read too when named directly
TREE_MAX_PAGES = 25  # the Hub tree API answers 1,000 files and folders a page: a repository's first 25,000 are listed
TREE_BUDGET_S = 60.0  # listing one repository's files

_now = time.monotonic  # tests replace it to run the budget
_sleep = time.sleep  # tests replace it to skip the waits
RETRY_WAITS = (5.0, 15.0)  # waits before retrying a rate-limited or busy viewer or Hub request
MAX_RETRY_WAIT = 30.0
RETRY_REASONS = ("http_429", "http_502", "http_503", "http_504")  # passing: rate limit, gateway errors, busy
LIST_LITERAL = "list-literal"  # Mapping.label_separator of a label stored as a JSON or Python list literal

SEARCH_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
    "additionalProperties": False,
}

MAPPING_SCHEMA = {
    "type": "object",
    "properties": {
        "input_columns": {"type": "array", "items": {"type": "string"}},
        "template": {"type": "string"},
        "label_column": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["input_columns", "template", "label_column", "reason"],
    "additionalProperties": False,
}

# The first property's name is not one research's fake mode has canned answers for, so fake mode answers it empty.
WEB_SCHEMA = {
    "type": "object",
    "properties": {
        "labelled_datasets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "hub_id": {"type": "string"},
                    "download_url": {"type": "string"},
                    "licence": {"type": "string"},
                    "label_column": {"type": "string"},
                    "input_column": {"type": "string"},
                    "page_url": {"type": "string"},
                },
                "required": ["name", "hub_id", "download_url", "licence", "label_column"],
            },
        }
    },
    "required": ["labelled_datasets"],
}

_WEB_INPUTS = {
    "chemical": "a molecule given as a SMILES string",
    "coordinates": "a place given as latitude and longitude",
    "text": "a piece of text",
    "audio": "an audio file", "image": "an image file", "video": "a video file", "3d": "a 3D model file",
}

_WEB_QUERY = (
    "Find up to 8 different public labelled datasets for this goal: {goal} Each row must be one input, {input}, "
    "with the category the goal asks about as its label, from a source independent of any model, and each category "
    "must have many rows (one row per input, not one row per category). For each dataset give its name; either its "
    "Hugging Face dataset id (owner/name) or a direct download URL of the data file itself (a CSV, TSV, JSON or JSON "
    "Lines file, possibly zipped or gzipped: a raw file on GitHub such as "
    "https://raw.githubusercontent.com/<owner>/<repo>/<branch>/<path>, or a file on Zenodo), else an empty string; "
    "its licence as stated by the source, else an empty string; the column holding the label; the column holding "
    "the input; and the page that describes it."
)

_SEARCH_INSTRUCTION = (
    "Suggest up to 5 short search queries (one to three words each) for the Hugging Face Hub dataset search that "
    "would find a public dataset of labelled examples for this goal, where each example is an input of the given "
    "modality and its label is the category the goal asks about. The search matches words in dataset names, so "
    "prefer benchmark names and plain domain words. Return JSON with queries."
)

_MAPPING_INSTRUCTION = (
    "Choose the columns of this Hugging Face dataset that give labelled examples for the goal. input_columns: the "
    "column or columns holding the input itself (one media column, or text or number columns such as a SMILES string "
    "or latitude and longitude). template: for text inputs a format string over the input columns such as "
    "'{latitude}, {longitude}', else an empty string. label_column: the column holding the category the goal asks "
    "about; it must not be an input column and must not be derived from the input. Give a one-sentence reason. "
    "Return JSON with input_columns, template, label_column and reason."
)

_MEDIA_TYPES = ("Audio", "Image", "Video", "Mesh")
_MEDIA_EXT = {
    "Audio": (".wav", (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac")),
    "Image": (".jpg", (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff")),
    "Video": (".mp4", (".mp4", ".webm", ".avi", ".mkv", ".mov", ".m4v")),
    "Mesh": (".glb", (".glb", ".gltf", ".obj", ".ply", ".stl", ".off")),
    # the folder kind only: molecule and geometry files
    "Molecule": (".mol", (".mol", ".sdf", ".smi")),
    "Geo": (".geojson", (".geojson", ".gpx", ".kml")),
}
_FOLDER_EXT = {  # the files a folder-labelled repository may hold for each kind of input
    "Mesh": (".off", ".obj", ".stl", ".ply", ".glb", ".gltf"),
    "Video": (".mp4", ".avi", ".mov", ".webm", ".mkv"),
    "Audio": (".wav", ".flac", ".mp3", ".ogg"),
    "Image": (".png", ".jpg", ".jpeg"),
    "Molecule": (".mol", ".sdf", ".smi"),
    "Geo": (".geojson", ".gpx", ".kml"),
}
_STOP = frozenset("""
a an the this that these those it its is are was were be been being of in on at to for from by with as and or not
no do does did done what which who whom whose where when why how can could would should will shall may might must
make makes made tell me my you your i we our us let give gives say says show find recognise recognize identify
classify detect predict guess know learn sense new happening happen happens like kind type sort model models file
files data dataset input inputs please any some each into about there here thing things one two get gets
""".split())
_ASK = frozenset(("what", "which", "whose"))
_LABEL_WORDS = ("label", "labels", "class", "classes", "category", "categories", "target", "tag", "tags")
_NOT_LABEL = frozenset(("id", "idx", "index", "name", "names", "title", "url", "path", "file", "filename", "source",
                        "split", "description", "caption", "text", "prompt", "question", "date", "time", "timestamp",
                        "license", "author", "authors", "uid", "hash", "sha", "md5", "fold",
                        # a model's output beside the ground truth
                        "pred", "preds", "prediction", "predictions", "predicted", "model", "score", "scores",
                        "prob", "probs", "probability", "probabilities", "logit", "logits", "confidence"))


@dataclass(frozen=True)
class _Slot:
    words: tuple[str, ...]  # column-name words, best first
    check: str = "any"  # "any" | "smiles" | "lat" | "lon"


@dataclass(frozen=True)
class _Profile:
    words: tuple[str, ...]  # Hub search words
    media: tuple[str, ...] = ()  # viewer feature types accepted as the input
    slots: tuple[_Slot, ...] = ()  # text inputs: one slot per column
    ext: str = ".txt"
    tag: str | None = None  # the Hub's modality:<tag>
    label_words: tuple[str, ...] = ()  # column names that label this modality by themselves
    input_words: tuple[str, ...] = ()  # words that only name the input, never the label's topic
    folder: str | None = None  # the _FOLDER_EXT key of the files a folder-labelled repository may hold


_PROFILES = {
    "audio": _Profile(("audio", "sound"), media=("Audio",), tag="audio", label_words=("genre",),
                      input_words=("audio", "sound", "recording", "clip", "wav"), folder="Audio"),
    "image": _Profile(("image",), media=("Image",), tag="image", input_words=("image", "picture", "photo"),
                      folder="Image"),
    "video": _Profile(("action", "ucf101", "kinetics", "video", "hmdb"), media=("Video",), tag="video",
                      label_words=("action", "activity"), input_words=("video", "clip", "movie", "footage"),
                      folder="Video"),
    "3d": _Profile(("modelnet", "objects", "3d", "mesh", "shapenet"), media=("Mesh",), tag="3d",
                   label_words=("object", "shape"), input_words=("3d", "mesh", "model", "point", "cloud", "glb"),
                   folder="Mesh"),
    "chemical": _Profile(("odor", "smell", "olfaction", "odorant", "scent"),
                         slots=(_Slot(("smiles", "smile", "mol", "molecule"), "smiles"),), ext=".smi",
                         label_words=("odor", "odour", "descriptor", "smell", "scent", "taste", "flavor"),
                         input_words=("molecule", "molecular", "smiles", "chemical", "chemistry", "compound", "mol",
                                      "chem", "structure"), folder="Molecule"),
    "coordinates": _Profile(("cities", "city", "geonames", "country", "countries"),
                            slots=(_Slot(("latitude", "lat"), "lat"),
                                   _Slot(("longitude", "lng", "lon", "long"), "lon")),
                            tag="geospatial", label_words=("country", "continent", "region", "state"),
                            input_words=("place", "location", "coordinate", "latitude", "longitude", "lat", "lon",
                                         "gps", "geo", "point", "position"), folder="Geo"),
    "text": _Profile((), slots=(_Slot(("text", "sentence", "content", "document")),), tag="text",
                     input_words=("text", "sentence", "document")),
}
_ALIASES = {
    "sound": "audio", "music": "audio", "speech": "audio", "picture": "image", "photo": "image", "images": "image",
    "videos": "video", "clip": "video", "mesh": "3d", "3d model": "3d", "3d-model": "3d", "model3d": "3d",
    "pointcloud": "3d", "point cloud": "3d", "smell": "chemical", "taste": "chemical", "molecule": "chemical",
    "molecules": "chemical", "smiles": "chemical", "chemistry": "chemical", "olfaction": "chemical",
    "odor": "chemical", "odour": "chemical", "coordinate": "coordinates", "location": "coordinates",
    "geo": "coordinates", "gps": "coordinates", "latlon": "coordinates", "lat/lon": "coordinates",
    "geolocation": "coordinates", "place": "coordinates",
}
_PLACEHOLDERS = frozenset(("unknown", "none", "null", "nan", "n/a", "na", "other", "others", "misc", "-", "?"))
_SMILES = re.compile(r"^[A-Za-z0-9@+\-\[\]()=#$:/\\.%*~]+$")
_PARTS = re.compile(r"\s*[;|,/]\s*")  # separators that join several label values in one cell


class ProbesUnavailable(RuntimeError):
    """No usable probe set: the reason is in the message."""


class _RequestFailed(ProbesUnavailable):
    """A Hub or viewer request failed; `reason` is the _HttpError reason (http_500, timeout, network, ...)."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


class _HttpError(RuntimeError):
    """One request failed; `reason` is a short key such as host_refused, too_large, timeout, network or http_404."""

    def __init__(self, reason: str, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.reason = reason
        self.retry_after = retry_after  # seconds the server asked to wait (HTTP 429 or 503), if it said


@dataclass
class HubCandidate:
    hub_id: str
    title: str
    downloads: int
    likes: int
    licence: str | None
    tags: list[str]
    url: str
    gated: bool = False
    score: int = 0  # search relevance; ordering only
    found_by: str = "hub"  # "hub" (the Hub search), "web" (the web search named it) or "both"
    revision: str | None = None  # the repository's commit when the Hub record gave it

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(eq=False)
class FolderCandidate:
    """A Hub dataset repository whose files sit in one folder per label, read through the Hub's file list and
    resolve URLs rather than the dataset viewer: find_probes tries one when a HubCandidate's viewer fails."""

    hub_id: str
    title: str
    licence: str | None
    tags: list[str]
    url: str  # https://huggingface.co/datasets/<id>
    revision: str | None = None  # the commit the files are listed and downloaded at; pinned when first listed
    downloads: int = 0
    found_by: str = "hub"
    score: int = 0
    _files: tuple | None = field(default=None, repr=False)  # (files as (path, size), notes) once listed

    @classmethod
    def of(cls, candidate: HubCandidate) -> "FolderCandidate":
        return cls(candidate.hub_id, candidate.title, candidate.licence, list(candidate.tags), candidate.url,
                   candidate.revision, candidate.downloads, candidate.found_by, candidate.score)

    def to_dict(self) -> dict:
        return {"hub_id": self.hub_id, "title": self.title, "licence": self.licence, "tags": list(self.tags),
                "url": self.url, "revision": self.revision, "downloads": self.downloads,
                "found_by": self.found_by, "score": self.score}


@dataclass(eq=False)
class TableCandidate:
    """A labelled table file (CSV, TSV, JSON or JSON Lines, possibly in a .zip or .gz) that a web search named."""

    url: str  # the https download URL (a GitHub blob URL rewritten to its raw file)
    title: str  # the web search's name for it, else the file name: display and ordering only
    licence: str | None = None  # the web search's claim until fetch_probes looks one up; then always set
    licence_source: str | None = None  # where the licence was read
    label_hint: str = ""  # the label column the web search named: shown to the language model only
    page_url: str = ""  # the page the web search cited for it
    given_url: str = ""  # the URL exactly as the web search gave it
    score: int = 0  # search relevance; ordering only
    _loaded: tuple | None = field(default=None, repr=False)  # (columns, rows, member, notes) once downloaded

    def to_dict(self) -> dict:
        return {"url": self.url, "title": self.title, "licence": self.licence,
                "licence_source": self.licence_source, "label_hint": self.label_hint, "page_url": self.page_url,
                "given_url": self.given_url, "score": self.score}


@dataclass
class Mapping:
    config: str
    split: str
    input_columns: list[str]
    input_kind: str  # "media" | "text"
    template: str  # text: e.g. "{latitude}, {longitude}"; media: ""
    extension: str  # ".wav", ".jpg", ".mp4", ".glb", ".smi", ".txt", ...
    label_column: str  # "" for multihot and folder
    label_kind: str  # "classlabel" | "string" | "list" | "multihot" | "folder"
    label_names: list[str] | None = None  # ClassLabel names, the 0/1 columns of a multihot label, or label folders
    label_separator: str | None = None  # a "list" label in one string: ";", "|", ",", "/" or LIST_LITERAL
    media_type: str | None = None  # "Audio" | "Image" | "Video" | "Mesh" (a folder also "Molecule" | "Geo")
    origin: str = "rule"  # "llm" | "rule" (a mapping's origin; search queries record "llm+rule" or "rule")
    reason: str = ""
    folder_depth: int | None = None  # folder: which folder of a file's path names its label (0 = the top one)

    def to_dict(self) -> dict:
        out = asdict(self)
        if self.label_names is not None and len(self.label_names) > 50:
            out["label_names"] = self.label_names[:50] + [f"... {len(self.label_names) - 50} more"]
        return out


@dataclass
class FetchReport:
    folder: Path
    labels: list[str]  # folder names kept, most files first
    counts: dict[str, int]  # folder name -> files written
    rows_seen: int
    skipped: dict[str, int]  # reason -> rows or files skipped
    licence: str | None
    source_url: str
    notes: list[str]
    hub_id: str = ""
    label_values: dict[str, str] = field(default_factory=dict)  # folder name -> the dataset's own label value
    bytes: int = 0  # the kept files' total size
    licence_source: str | None = None  # where the licence was read: the Hub card, a LICENSE file, ...

    def to_dict(self) -> dict:
        return asdict(self) | {"folder": str(self.folder)}


# ----------------------------------------------------------------------------- HTTP
class _Redirects(urllib.request.HTTPRedirectHandler):
    def _ok(self, url: str) -> bool:
        return _allowed_url(url)

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not self._ok(newurl):
            raise _HttpError("host_refused", f"refused a redirect to {_host(newurl)}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _TableRedirects(_Redirects):
    def _ok(self, url: str) -> bool:
        return _table_url_ok(url)


def _http_get(url: str, *, timeout: float = TIMEOUT_S, max_bytes: int = JSON_MAX_BYTES) -> bytes:
    """GET an https Hugging Face URL: at most max_bytes, within timeout seconds, or _HttpError. The time left is
    checked before every read, each read returns as soon as some bytes arrive (read1), and the socket's own timeout
    is cut to the time left, so a server that trickles bytes cannot outlast the timeout."""
    if not _allowed_url(url):
        raise _HttpError("host_refused", f"only https Hugging Face URLs are fetched, not {_host(url)}")
    return _get(url, timeout, max_bytes, _Redirects)


def _http_get_page(url: str, *, timeout: float = TIMEOUT_S,
                   max_bytes: int = JSON_MAX_BYTES) -> tuple[bytes, str | None]:
    """_http_get that also returns the response's Link header, or None: the Hub tree API pages through it."""
    if not _allowed_url(url):
        raise _HttpError("host_refused", f"only https Hugging Face URLs are fetched, not {_host(url)}")
    seen: dict = {}
    body = _get(url, timeout, max_bytes, _Redirects, seen)
    return body, seen.get("link")


def _table_get(url: str, *, timeout: float = TABLE_TIMEOUT_S, max_bytes: int = TABLE_MAX_BYTES) -> bytes:
    """_http_get for table files and their licences: https URLs on Hugging Face, github.com,
    raw.githubusercontent.com, gist.githubusercontent.com or zenodo.org only, redirects too."""
    if not _table_url_ok(url):
        raise _HttpError("host_refused", f"only https URLs on Hugging Face, GitHub or Zenodo hosts are fetched, not "
                                         f"{_host(url)}")
    return _get(url, timeout, max_bytes, _TableRedirects)


def _get(url: str, timeout: float, max_bytes: int, redirects: type, seen: dict | None = None) -> bytes:
    if timeout <= 0:
        raise _HttpError("timeout", "no time left")
    deadline = _now() + timeout
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    try:
        with urllib.request.build_opener(redirects).open(request, timeout=timeout) as response:
            if seen is not None:
                seen["link"] = response.headers.get("Link")
            length = response.headers.get("Content-Length")
            if length and length.isdigit() and int(length) > max_bytes:
                raise _HttpError("too_large", f"{_host(url)} offers {int(length):,} bytes, over the {max_bytes:,} cap")
            read = getattr(response, "read1", None) or response.read
            chunks, size = [], 0
            while True:
                left = deadline - _now()
                if left <= 0:
                    raise _HttpError("timeout", f"{_host(url)} took over {timeout:.0f} s")
                _cut_socket_timeout(response, left)
                chunk = read(min(1 << 16, max_bytes + 1 - size))
                if not chunk:
                    return b"".join(chunks)
                size += len(chunk)
                chunks.append(chunk)
                if size > max_bytes:
                    raise _HttpError("too_large", f"{_host(url)} sent over the {max_bytes:,} byte cap")
    except _HttpError:
        raise
    except urllib.error.HTTPError as exc:
        detail = _error_detail(exc)
        wait_s = _number(exc.headers.get("Retry-After") if exc.headers else None)
        exc.close()
        raise _HttpError(f"http_{exc.code}", f"HTTP {exc.code} from {_host(url)}{': ' + detail if detail else ''}",
                         wait_s) from None
    except TimeoutError:
        raise _HttpError("timeout", f"{_host(url)} took over {timeout:.0f} s") from None
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise _HttpError("timeout", f"{_host(url)} took over {timeout:.0f} s") from None
        raise _HttpError("network", f"URLError reaching {_host(url)}") from None
    except (OSError, http.client.HTTPException, ValueError) as exc:
        raise _HttpError("network", f"{type(exc).__name__} reaching {_host(url)}") from None


def _cut_socket_timeout(response, seconds: float) -> None:
    """Best effort: make the response's next socket read wait at most `seconds` (urllib's timeout is per read)."""
    sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
    try:
        current = sock.gettimeout()
        sock.settimeout(max(0.05, min(seconds, current) if current else seconds))
    except (AttributeError, OSError, TypeError, ValueError):
        pass


def _error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read(2000)
        answer = json.loads(body)
        text = answer.get("error") if isinstance(answer, dict) else None
    except Exception:  # an error body is optional detail only
        return ""
    return re.sub(r"\s+", " ", text)[:200] if isinstance(text, str) else ""


def _allowed_url(url: object) -> bool:
    try:
        url.encode("ascii")
        parts = urllib.parse.urlsplit(url)
        host, port = (parts.hostname or "").lower(), parts.port
    except (UnicodeError, ValueError, AttributeError):
        return False
    return (parts.scheme == "https" and not parts.username and not parts.password and port in (None, 443)
            and (host in ("huggingface.co", "hf.co") or host.endswith((".huggingface.co", ".hf.co"))))


def _table_url_ok(url: object) -> bool:
    """An https URL on a Hugging Face host or on github.com, raw.githubusercontent.com, gist.githubusercontent.com or
    zenodo.org, with no user, password or port other than 443."""
    if _allowed_url(url):
        return True
    try:
        url.encode("ascii")
        parts = urllib.parse.urlsplit(url)
        host, port = (parts.hostname or "").lower(), parts.port
    except (UnicodeError, ValueError, AttributeError):
        return False
    return (parts.scheme == "https" and not parts.username and not parts.password and port in (None, 443)
            and host in TABLE_HOSTS)


def _host(url: object) -> str:
    try:
        return urllib.parse.urlsplit(str(url)).hostname or "an invalid URL"
    except ValueError:
        return "an invalid URL"


def _get_json(base: str, params: dict, *, timeout: float = TIMEOUT_S, what: str, deadline: float | None = None,
              retry: bool = True) -> object:
    """One JSON answer. Each attempt gets the smaller of timeout and the time left before deadline; a rate limit
    (HTTP 429), a gateway error (502, 504) or a busy server (503) is retried (unless retry is False) after
    RETRY_WAITS (or the server's Retry-After, at most MAX_RETRY_WAIT), but a wait that would end within 2 s of the
    deadline is not taken."""
    url = base + ("?" + urllib.parse.urlencode(params) if params else "")
    return _attempts(lambda seconds: json.loads(_http_get(url, timeout=seconds, max_bytes=JSON_MAX_BYTES)),
                     what=what, timeout=timeout, deadline=deadline, retry=retry)


def _attempts(request: Callable[[float], object], *, what: str, timeout: float, deadline: float | None,
              retry: bool) -> object:
    """request(seconds) (a JSON request and its parse), retried as _get_json says."""
    for pause_s in (RETRY_WAITS if retry else ()) + (None,):
        attempt = timeout if deadline is None else min(timeout, deadline - _now())
        if attempt <= 0:
            raise _RequestFailed("timeout", f"{what} failed: the time budget was spent")
        try:
            return request(attempt)
        except _HttpError as exc:
            pause = None if pause_s is None else min(max(exc.retry_after or 0.0, pause_s), MAX_RETRY_WAIT)
            if pause is None or exc.reason not in RETRY_REASONS or (deadline is not None
                                                                    and _now() + pause + 2 > deadline):
                raise _RequestFailed(exc.reason, f"{what} failed: {exc}") from None
            _sleep(pause)
        except (ValueError, UnicodeError):
            raise _RequestFailed("not_json", f"{what} did not answer JSON") from None
    raise AssertionError("unreachable")


# ----------------------------------------------------------------------------- words
def _profile(modality: str) -> _Profile | None:
    key = re.sub(r"\s+", " ", str(modality).strip().lower())
    return _PROFILES.get(_ALIASES.get(key, key))


def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("sses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _content_words(goal: str) -> list[str]:
    out = []
    for word in re.findall(r"[a-z0-9]+", str(goal).lower()):
        if word in _STOP or (len(word) < 3 and not re.search(r"\d", word)) or word.isdigit():
            continue
        word = _singular(word)
        if word not in out and word not in _STOP:
            out.append(word)
    return out


def _tokens(name: str) -> list[str]:
    """Column-name words: camelCase and separators split, lower case."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return [t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t]


def _name_hits(name: str, words) -> bool:
    for token in _tokens(name):
        for word in words:
            if token == word or (len(word) >= 5 and token.endswith(word)) or _singular(token) == word:
                return True
    return False


# ----------------------------------------------------------------------------- search
def search_hub(goal: str, modality: str, *, limit: int = 20) -> list[HubCandidate]:
    return _search(goal, modality, limit)[0]


def _search(goal: str, modality: str, limit: int,
            deadline: float | None = None) -> tuple[list[HubCandidate], list[str], str]:
    found, hits, errors, queries, origin = _hub_query(goal, modality, limit, deadline)
    if errors and len(errors) == len(queries):
        raise ProbesUnavailable(f"the Hub search failed for every query: {errors[0]}")
    return _ranked(list(found.values()), hits, queries, modality)[:limit], queries, origin


def _hub_query(goal: str, modality: str, limit: int, deadline: float | None):
    """(found: hub_id -> HubCandidate, hits: how many queries found each, errors, queries, query origin)."""
    # The model's queries come first, the rule's (goal words plus the modality's own words) after: in a real run the
    # model's terms alone ("shapenet ...") missed the one dataset with real meshes that the rule's terms find.
    asked = _llm_queries(llm.ask_json(_SEARCH_INSTRUCTION, {"goal": goal, "modality": modality}, SEARCH_SCHEMA))
    origin = "llm+rule" if asked else "rule"
    queries = list(dict.fromkeys([*asked, *_rule_queries(goal, modality)]))[:MAX_QUERIES + 5]
    if not queries:
        raise ProbesUnavailable(f"no search words could be taken from the goal {goal!r}")
    found: dict[str, HubCandidate] = {}
    hits: Counter = Counter()
    errors = []
    for query in queries:
        try:
            answer = _get_json(HUB_SEARCH, {"search": query, "limit": limit, "full": "true", "sort": "downloads"},
                               what=f"the Hub search for {query!r}", deadline=deadline)
        except ProbesUnavailable as exc:
            errors.append(str(exc))
            continue
        if not isinstance(answer, list):
            errors.append(f"the Hub search for {query!r} did not answer a list")
            continue
        for item in answer:
            candidate = _candidate(item)
            if candidate is not None:
                found.setdefault(candidate.hub_id, candidate)
                hits[candidate.hub_id] += 1
    return found, hits, errors, queries, origin


def _ranked(candidates: list, hits: Counter, queries: list[str], modality: str) -> list:
    """Hub and table candidates in one order: relevance (2 for each topic word and 1 for each input word of the
    queries found in the id or title, or a table's URL path and title; plus the Hub's modality and classification
    tag bonus; plus 1 for a web search's recommendation), then how many searches found it, then downloads."""
    profile = _profile(modality)
    words = {_singular(w) for q in queries for w in re.findall(r"[a-z0-9]+", q.lower()) if w not in _STOP}
    inputs = set(profile.input_words) if profile else set(_content_words(modality))
    for candidate in candidates:
        candidate.score = 2 * _relevance(candidate, words - inputs) + _relevance(candidate, words & inputs)
        if isinstance(candidate, TableCandidate):
            candidate.score += 1
        else:
            candidate.score += _tag_bonus(candidate.tags, profile) + (candidate.found_by != "hub")
    return sorted(candidates, key=lambda c: (-c.score, -hits[_source_of(c)], -getattr(c, "downloads", 0),
                                             _source_of(c)))


def _relevance(candidate, words: set[str]) -> int:
    """How many of the search words appear in the dataset's id or title (plurals and word stems alike)."""
    if isinstance(candidate, TableCandidate):
        tokens = set(_name_tokens(urllib.parse.urlsplit(candidate.url).path, candidate.title))
    else:
        tokens = set(_name_tokens(candidate.hub_id, candidate.title))
    return sum(1 for word in words if any(_same_word(token, word) for token in tokens))


def _name_tokens(*names: str) -> list[str]:
    """The words of names, with camelCase split and also whole ("ModelNet40" gives model, net40 and modelnet40, so
    it names modelnet)."""
    out = []
    for name in names:
        out += _tokens(name) + re.findall(r"[a-z0-9]+", str(name).lower())
    return out


def _source_of(candidate) -> str:
    """What names a candidate in exclude_sources, info["source"] and hits: a Hub id, or a table's URL."""
    return candidate.url if isinstance(candidate, TableCandidate) else candidate.hub_id


def _same_word(token: str, word: str) -> bool:
    if token == word or _singular(token) == word:
        return True
    if len(word) >= 4 and token.startswith(word):
        return True
    common = len(os.path.commonprefix([token, word]))
    return common >= 6 and common >= min(len(token), len(word)) - 2


def _llm_queries(answer: object) -> list[str]:
    raw = answer.get("queries") if isinstance(answer, dict) else None
    if not isinstance(raw, list):
        return []
    out = []
    for query in raw:
        if isinstance(query, str) and 0 < len(query.strip()) <= 60 and query.strip().isprintable():
            query = re.sub(r"\s+", " ", query.strip())
            if query.lower() not in (q.lower() for q in out):
                out.append(query)
    return out[:5]


def _rule_queries(goal: str, modality: str) -> list[str]:
    """The goal's first two content words together, then the modality's search words interleaved with the goal's
    words; a word that only names the input (molecule, video, 3d, ...) is searched only inside the first query."""
    profile = _profile(modality)
    inputs = set(profile.input_words) if profile else set(_content_words(modality))
    goal_words = _content_words(goal)
    queries = [" ".join(goal_words[:2])] if len(goal_words) >= 2 else []
    words = [w for w in goal_words if w not in inputs]
    extra = [w for w in (profile.words if profile else _content_words(modality)) if w not in inputs]
    if not words and not extra:
        words = goal_words or _content_words(modality)
    for i in range(max(len(words), len(extra))):
        queries += [w for w in (extra[i] if i < len(extra) else None, words[i] if i < len(words) else None) if w]
    out = []
    for query in queries:
        if query not in out:
            out.append(query)
    return out[:MAX_QUERIES]


MODALITY_TAG_BONUS = 4  # the Hub sets modality:<x> from the data's own feature types: the best sign of usable files


def _tag_bonus(tags: list[str], profile: _Profile | None) -> int:
    # Worth more than a name word: in a real run, datasets named for ShapeNet (point clouds, npz) outranked the one
    # tagged modality:3d that holds real meshes.
    bonus = MODALITY_TAG_BONUS if profile and profile.tag and f"modality:{profile.tag}" in tags else 0
    return bonus + (1 if any(t.startswith("task_categories:") and t.endswith("-classification") for t in tags) else 0)


_HUB_ID = re.compile(r"^[\w.-]+/[\w.-]+$")


def _candidate(item: object) -> HubCandidate | None:
    if not isinstance(item, dict):
        return None
    hub_id = item.get("id")
    if not isinstance(hub_id, str) or not _HUB_ID.match(hub_id) or {".", ".."} & set(hub_id.split("/")):
        return None
    if item.get("gated") or item.get("private") or item.get("disabled"):
        return None
    card = item.get("cardData") if isinstance(item.get("cardData"), dict) else {}
    if card.get("viewer") is False:
        return None
    tags = [t for t in item.get("tags") or [] if isinstance(t, str)] if isinstance(item.get("tags"), list) else []
    title = card.get("pretty_name") if isinstance(card.get("pretty_name"), str) else hub_id
    sha = item.get("sha")
    return HubCandidate(hub_id, title.strip()[:120] or hub_id, _int(item.get("downloads")), _int(item.get("likes")),
                        _licence(card, tags), tags[:40], f"https://huggingface.co/datasets/{hub_id}", False,
                        revision=sha if isinstance(sha, str) and _SHA.fullmatch(sha) else None)


_SHA = re.compile(r"[0-9a-f]{40}")


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _licence(card: dict, tags: list[str]) -> str | None:
    """cardData.license (with the card's license_name and license_link in brackets, which say what an "other"
    licence is), else a "license:" tag, else None."""
    value = card.get("license")
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value if isinstance(v, str) and v) or None
    if not (isinstance(value, str) and value.strip()):
        value = next((t.partition(":")[2] for t in tags if t.startswith("license:") and t.partition(":")[2]), None)
    if value is None:
        return None
    extra = [card[k].strip()[:200] for k in ("license_name", "license_link")
             if isinstance(card.get(k), str) and card[k].strip()]
    return value.strip() + (f" ({', '.join(extra)})" if extra else "")


# ----------------------------------------------------------------------------- choosing columns
@dataclass
class _Column:
    name: str
    kind: str  # "media" | "classlabel" | "string" | "number" | "list" | "other"
    media_type: str | None = None
    names: list[str] | None = None  # ClassLabel names (also for a list of ClassLabel)


@dataclass(frozen=True)
class _Named:
    """What the column choice knows of a source: its name (a Hub id, or a table's host and path), its own title (a
    Hub card's pretty_name, or a table's file name) and a web search's label column hint (shown to the model only)."""

    name: str
    title: str
    hint: str = ""


def choose_mapping(candidate: HubCandidate | TableCandidate | FolderCandidate, goal: str, modality: str, *,
                   wanted_labels: list[str] | None = None, budget_s: float | None = None,
                   min_per_label: int = 6) -> Mapping:
    deadline = None if budget_s is None else _now() + budget_s
    if isinstance(candidate, TableCandidate):
        return _choose_table(candidate, goal, modality, list(wanted_labels or []), deadline)
    if isinstance(candidate, FolderCandidate):
        return _choose_folder(candidate, goal, modality, list(wanted_labels or []), deadline, min_per_label)
    return _choose(candidate, goal, modality, list(wanted_labels or []), deadline)


def _choose(candidate: HubCandidate, goal: str, modality: str, wanted: list[str], deadline: float | None) -> Mapping:
    hub_id = candidate.hub_id
    named = _Named(hub_id, candidate.title)
    splits = _get_json(f"{VIEWER}/splits", {"dataset": hub_id}, what=f"the viewer's splits of {hub_id}",
                       deadline=deadline)
    config, split = _pick_split(splits, hub_id)
    params = {"dataset": hub_id, "config": config, "split": split}
    what = f"the viewer's rows of {hub_id}"
    try:  # the viewer precomputes first-rows, so it is quick and rarely fails
        first = _get_json(f"{VIEWER}/first-rows", params, what=what + " (first rows)", deadline=deadline)
        if not isinstance(first, dict) or not isinstance(first.get("rows"), list):
            raise ProbesUnavailable(f"{what} (first rows) did not answer a page of rows")
    except ProbesUnavailable:
        first = _rows_page(params, 0, what=what, timeout=ROWS_TIMEOUT_S, deadline=deadline)
    columns = _columns(first.get("features"), hub_id)
    rows = _rows(first)
    if not rows:
        raise ProbesUnavailable(f"the viewer shows no rows for {hub_id} ({config}/{split})")
    profile = _profile(modality)
    try:
        mapping = _map(named, goal, modality, columns, rows, profile, wanted)
    except _NoLabel:
        # A split sorted by label can show one label in its first rows: look once more, in the middle.
        try:
            total = _rows_page(params, 0, what=what, timeout=ROWS_TIMEOUT_S, length=1,
                               deadline=deadline).get("num_rows_total")
            if not isinstance(total, int) or total <= 2 * PAGE_ROWS:
                raise ProbesUnavailable("no more rows")
            rows += _rows(_rows_page(params, total // 2, what=what, timeout=ROWS_TIMEOUT_S, deadline=deadline))
        except ProbesUnavailable:
            raise _NoLabel(_map_error(named, goal, modality, columns, rows, profile, wanted)) from None
        mapping = _map(named, goal, modality, columns, rows, profile, wanted)
    mapping.config, mapping.split = config, split
    return mapping


class _NoLabel(ProbesUnavailable):
    """The columns fit the input but no label column was found in the rows looked at."""


def _map_error(named: _Named, goal, modality, columns, rows, profile, wanted) -> str:
    try:
        _map(named, goal, modality, columns, rows, profile, wanted)
    except ProbesUnavailable as exc:
        return str(exc)
    return f"{named.name} has no usable label column"


def _map(named: _Named, goal, modality, columns, rows, profile, wanted) -> Mapping:
    mapping = _llm_mapping(named, goal, modality, columns, rows, profile)
    if mapping is not None:
        return mapping
    return _rule_mapping(named, goal, columns, rows, profile, modality, wanted)


def _pick_split(answer: object, hub_id: str) -> tuple[str, str]:
    raw = answer.get("splits") if isinstance(answer, dict) else None
    splits = [s for s in raw or [] if isinstance(s, dict) and isinstance(s.get("config"), str)
              and isinstance(s.get("split"), str)] if isinstance(raw, list) else []
    if not splits:
        error = answer.get("error") if isinstance(answer, dict) else None
        raise ProbesUnavailable(f"the viewer lists no splits for {hub_id}" + (f": {str(error)[:200]}" if error else ""))
    configs = [s["config"] for s in splits]
    config = "default" if "default" in configs else configs[0]
    in_config = [s["split"] for s in splits if s["config"] == config]
    return config, next((s for s in ("train", "test", "validation") if s in in_config), in_config[0])


_PASSING = frozenset(RETRY_REASONS + ("timeout", "network"))  # failures that may not happen on the next request


def _first_page(params: dict, *, what: str, deadline: float,
                fallback_deadline: float) -> tuple[dict, str | None, str | None]:
    """(the first /rows page, None, None), else (/first-rows, why /rows failed, the failure's reason): the viewer
    refuses /rows for some splits (large row groups) while it still serves their first 100 rows. /rows gets the
    reading deadline; the quick, precomputed /first-rows gets the later fallback_deadline."""
    try:
        return _rows_page(params, 0, what=what, timeout=ROWS_TIMEOUT_S, deadline=deadline), None, None
    except ProbesUnavailable as exc:
        try:
            answer = _get_json(f"{VIEWER}/first-rows", params, what=what + " (first rows)",
                               deadline=fallback_deadline)
        except ProbesUnavailable:
            raise exc from None
        if not isinstance(answer, dict) or not isinstance(answer.get("rows"), list):
            raise exc from None
        return answer, str(exc), getattr(exc, "reason", None)


def _rows_page(params: dict, offset: int, *, what: str, timeout: float = TIMEOUT_S, length: int = PAGE_ROWS,
               deadline: float | None = None, retry: bool = True) -> dict:
    answer = _get_json(f"{VIEWER}/rows", params | {"offset": offset, "length": length}, what=what, timeout=timeout,
                       deadline=deadline, retry=retry)
    if not isinstance(answer, dict) or not isinstance(answer.get("rows"), list):
        raise ProbesUnavailable(f"{what} did not answer a page of rows")
    return answer


def _split_rows(params: dict, deadline: float) -> int | None:
    """The split's row count from the viewer's /size, or None."""
    try:
        answer = _get_json(f"{VIEWER}/size", {"dataset": params["dataset"]}, what="the viewer's size",
                           deadline=deadline)
    except ProbesUnavailable:
        return None
    size = answer.get("size") if isinstance(answer, dict) else None
    splits = size.get("splits") if isinstance(size, dict) else None
    for entry in splits if isinstance(splits, list) else []:
        if isinstance(entry, dict) and (entry.get("config"), entry.get("split")) == (params["config"],
                                                                                     params["split"]):
            rows = entry.get("num_rows")
            return rows if isinstance(rows, int) and not isinstance(rows, bool) and rows > 0 else None
    return None


def _rows(page: dict) -> list[tuple[int, dict, set]]:
    """(row index, row, truncated column names) for each well-formed row of a viewer page."""
    out = []
    for entry in page.get("rows") or []:
        if not isinstance(entry, dict):
            continue
        idx, row = entry.get("row_idx"), entry.get("row")
        if isinstance(idx, int) and not isinstance(idx, bool) and idx >= 0 and isinstance(row, dict):
            cut = entry.get("truncated_cells")
            out.append((idx, row, set(cut) if isinstance(cut, list) else set()))
    return out


def _columns(features: object, hub_id: str) -> dict[str, _Column]:
    out = {}
    for feature in features if isinstance(features, list) else []:
        if not isinstance(feature, dict) or not isinstance(feature.get("name"), str):
            continue
        name, kind = feature["name"], feature.get("type")
        kind = kind if isinstance(kind, dict) else {}
        out[name] = _column(name, kind)
    if not out:
        raise ProbesUnavailable(f"the viewer shows no columns for {hub_id}")
    return out


def _column(name: str, kind: dict) -> _Column:
    typ, dtype = kind.get("_type"), kind.get("dtype")
    if typ in _MEDIA_TYPES:
        return _Column(name, "media", media_type=typ)
    if typ == "ClassLabel" and isinstance(kind.get("names"), list):
        return _Column(name, "classlabel", names=[str(n) for n in kind["names"]])
    if typ == "Value" and dtype in ("string", "large_string"):
        return _Column(name, "string")
    if typ == "Value" and isinstance(dtype, str) and re.fullmatch(r"u?int\d+|float\d+|bool", dtype):
        return _Column(name, "number")
    if typ in ("List", "Sequence", "LargeList") and isinstance(kind.get("feature"), dict):
        inner = kind["feature"]
        if inner.get("_type") == "Value" and inner.get("dtype") in ("string", "large_string"):
            return _Column(name, "list")
        if inner.get("_type") == "ClassLabel" and isinstance(inner.get("names"), list):
            return _Column(name, "list", names=[str(n) for n in inner["names"]])
    return _Column(name, "other")


def _llm_mapping(named: _Named, goal, modality, columns, rows, profile) -> Mapping | None:
    data = {"goal": goal, "modality": modality, "dataset": named.name, "title": named.title,
            "columns": [_describe(col, rows) for col in list(columns.values())[:80]]}
    if named.hint:
        data["label_column_named_by_a_web_search"] = named.hint
    answer = llm.ask_json(_MAPPING_INSTRUCTION, data, MAPPING_SCHEMA, max_tokens=300)
    if not isinstance(answer, dict):
        return None
    inputs, template, label = answer.get("input_columns"), answer.get("template"), answer.get("label_column")
    if not (isinstance(inputs, list) and inputs and len(inputs) <= 4 and all(isinstance(c, str) for c in inputs)
            and len(set(inputs)) == len(inputs) and all(c in columns for c in inputs)
            and isinstance(label, str) and label in columns and label not in inputs and isinstance(template, str)):
        return None
    labelled = _label_fit(columns[label], rows)
    if labelled is None:
        return None
    kind, separator = labelled
    reason = answer.get("reason").strip()[:300] if isinstance(answer.get("reason"), str) else ""
    picked = [columns[c] for c in inputs]
    if picked[0].kind == "media":
        if len(picked) != 1 or (profile and profile.media and picked[0].media_type not in profile.media):
            return None
        if (profile and not profile.media) or _media_ext(picked[0], rows) is None:
            return None
        return Mapping("", "", inputs, "media", "", _media_ext(picked[0], rows), label, kind, columns[label].names,
                       separator, picked[0].media_type, "llm", reason)
    if any(col.kind not in ("string", "number") for col in picked) or (profile and profile.media):
        return None
    if profile and any(slot.check != "any" for slot in profile.slots) and not _fills_slots(inputs, profile.slots,
                                                                                            rows):
        return None
    template = template.strip() or ", ".join("{" + c + "}" for c in inputs)
    if set(re.findall(r"\{([^{}]*)\}", template)) != set(inputs):
        return None
    ext = profile.ext if profile else ".txt"
    mapping = Mapping("", "", inputs, "text", template, ext, label, kind, columns[label].names, separator, None, "llm",
                      reason)
    return None if _determining_input(mapping, rows) else mapping


def _fills_slots(inputs: list[str], slots: tuple[_Slot, ...], rows) -> bool:
    """Whether the input columns are exactly the profile's typed slots: one column per slot, named for it, with
    fitting values. So no extra column (a country name beside the coordinates, say) can carry the label."""
    if len(inputs) != len(slots):
        return False
    return any(all(_name_hits(name, slot.words) and _values_fit(name, rows, slot.check) for name, slot in
                   zip(order, slots)) for order in itertools.permutations(inputs))


def _determining_input(mapping: Mapping, rows) -> str | None:
    """An input column that gives the label away, else None: one with at most half as many distinct values as rows,
    whose rows with each value carry one label in at least 90% of them (a country name beside a country code)."""
    for name in mapping.input_columns:
        pairs = []
        for _, row, _ in rows:
            value, labels = row.get(name), _labels_of(row, mapping)
            if value is not None and not isinstance(value, (dict, list)) and str(value).strip() and labels:
                pairs.append((str(value).strip().lower(), tuple(labels)))
        if len(pairs) < 10 or len({labels for _, labels in pairs}) < 2:
            continue  # too few rows, or one label only: nothing to judge
        groups: dict[str, Counter] = {}
        for value, labels in pairs:
            groups.setdefault(value, Counter())[labels] += 1
        if 2 * len(groups) <= len(pairs) and sum(max(c.values()) for c in groups.values()) >= 0.9 * len(pairs):
            return name
    return None


def _describe(col: _Column, rows) -> dict:
    out = {"name": col.name, "kind": col.kind}
    if col.media_type:
        out["media_type"] = col.media_type
    if col.names:
        out["label_names"] = col.names[:20]
    elif col.kind != "media":
        out["examples"] = [str(row.get(col.name))[:80] for _, row, _ in rows[:3]]
    return out


def _rule_mapping(named: _Named, goal, columns, rows, profile, modality, wanted) -> Mapping:
    hub_id, names = named.name, ", ".join(list(columns)[:30])
    inputs, media_type, ext, template = _rule_inputs(columns, rows, profile, modality)
    if not inputs:
        if profile and profile.media:
            raise ProbesUnavailable(f"{hub_id} has no {'/'.join(profile.media)} column for {modality} "
                                    f"(columns: {names})")
        wanted_cols = " and ".join("/".join(slot.words) for slot in profile.slots) if profile else "text"
        raise ProbesUnavailable(f"{hub_id} has no input column for {modality}: wanted a column named like "
                                f"{wanted_cols} with fitting values (columns: {names})")
    if media_type and ext is None:
        raise ProbesUnavailable(f"{hub_id}'s {inputs[0]} cells do not link to https Hugging Face files "
                                f"(e.g. {_example_src(columns[inputs[0]], rows)})")
    topic = _topic_words(goal, modality, profile)
    on_topic = _names_topic(named, topic)
    weights = _topic_weights(goal, modality, profile)
    goal_words = [w for w in weights if w in _content_words(goal)]
    label = _rule_label(weights, columns, rows, set(inputs), on_topic, goal_words, wanted)
    if label is None:
        raise _NoLabel(
            f"{hub_id} has no label column named for the goal (a string or list column named like "
            f"{'/'.join(sorted(topic)[:8]) or 'the goal'} with at least 2 values)"
            + ("" if on_topic else ", and its name does not mention the goal's topic, so a generic label/class "
                                   "column, ClassLabel or block of 0/1 columns is not trusted")
            + _label_hints(columns, set(inputs), tuple(weights), on_topic)
            + f" (columns: {names})")
    column, kind, separator, label_names = label
    kind_of = "media" if media_type else "text"
    reason = f"rule: input {', '.join(inputs)}; label {column or 'the 0/1 columns'} ({kind})"
    mapping = Mapping("", "", inputs, kind_of, template, ext, column, kind, label_names, separator, media_type, "rule",
                      reason)
    if kind_of == "text" and (leaky := _determining_input(mapping, rows)):
        raise ProbesUnavailable(f"{hub_id}'s input column {leaky!r} determines the label {column!r} (few distinct "
                                "values, each with one label), so the input would give the label away")
    return mapping


def _names_topic(named: _Named, topic: set[str]) -> bool:
    """Whether a source's own name or title names one of the topic words: what lets a generic label column, a
    ClassLabel, a block of 0/1 columns or a repository's folder names label probes."""
    return any(_same_word(token, word) for token in _name_tokens(named.name, named.title) for word in topic)


def _label_hints(columns, inputs: set, topic: tuple, on_topic: bool) -> str:
    """Why plausible label columns were not used: an untrusted ClassLabel or label/class column, or integers."""
    out = ""
    if not on_topic:
        untrusted = [c for c in columns.values() if c.name not in inputs and (
            c.kind == "classlabel" or (c.kind in ("string", "list") and _name_hits(c.name, _LABEL_WORDS)))]
        if untrusted:
            out += "; present but not trusted: " + ", ".join(
                repr(c.name) + (f" (a ClassLabel: {', '.join(c.names[:6])}{', ...' if len(c.names) > 6 else ''})"
                                if c.kind == "classlabel" and c.names else "") for c in untrusted[:4])
    integers = [c.name for c in columns.values() if c.kind == "number" and c.name not in inputs
                and _name_hits(c.name, _LABEL_WORDS + topic)]
    if integers:
        out += f"; {', '.join(map(repr, integers[:4]))} hold numbers, not label names, so they are not used"
    return out


def _rule_inputs(columns, rows, profile, modality):
    """(input columns, media type or None, extension or None, template)."""
    media_types = profile.media if profile else tuple(t for t in _MEDIA_TYPES if t.lower() == str(modality).lower())
    if media_types:
        media = [c for c in columns.values() if c.kind == "media" and c.media_type in media_types]
        media.sort(key=lambda c: (_media_ext(c, rows) is None, not _name_hits(c.name, (c.media_type.lower(),))))
        if media:
            return [media[0].name], media[0].media_type, _media_ext(media[0], rows), ""
        return [], None, None, ""
    if not profile or not profile.slots:
        return [], None, None, ""
    picked = []
    for slot in profile.slots:
        best = None
        for rank, word in enumerate(slot.words):
            for col in columns.values():
                if col.kind in ("string", "number") and col.name not in picked and _name_hits(col.name, (word,)) \
                        and _values_fit(col.name, rows, slot.check):
                    best = col.name
                    break
            if best:
                break
        if best is None:
            return [], None, None, ""
        picked.append(best)
    return picked, None, profile.ext, ", ".join("{" + c + "}" for c in picked)


def _values_fit(name: str, rows, check: str) -> bool:
    values = [row.get(name) for _, row, _ in rows]
    present = [v for v in values if v is not None and str(v).strip()]
    if len(present) < max(1, len(values) // 2):
        return False
    if check == "smiles":
        return sum(1 for v in present if isinstance(v, str) and _SMILES.match(v.strip())) >= 0.8 * len(present)
    if check in ("lat", "lon"):
        limit = 90 if check == "lat" else 180
        numbers = [_number(v) for v in present]
        return all(n is not None and -limit <= n <= limit for n in numbers)
    return True


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _media_ext(col: _Column, rows) -> str | None:
    """The file extension of the column's first downloadable cell; None when no cell links to an allowed URL."""
    default, known = _MEDIA_EXT.get(col.media_type, (".bin", ()))
    for _, row, _ in rows:
        src = _src(row.get(col.name))
        if src and _allowed_url(src):
            suffix = PurePosixPath(urllib.parse.urlsplit(src).path).suffix.lower()
            return suffix if suffix in known else default
    return None


def _example_src(col: _Column, rows) -> str:
    for _, row, _ in rows:
        src = _src(row.get(col.name))
        if src:
            return src.split("::")[0][:60]
    return "no file links at all"


def _src(cell: object) -> str | None:
    """The file URL of a viewer media cell: a dict with "src", or a one-item list of one."""
    if isinstance(cell, list) and len(cell) == 1:
        cell = cell[0]
    src = cell.get("src") if isinstance(cell, dict) else None
    return src if isinstance(src, str) else None


def _topic_words(goal: str, modality: str, profile: _Profile | None, *, search_words: bool = True) -> set[str]:
    """Words that can name what the label is about: the goal's content words and the modality's label words (and,
    for naming a whole dataset, its search words), less the words that only name the input."""
    inputs = set(profile.input_words) if profile else set(_content_words(modality))
    words = set(_content_words(goal)) | set(profile.label_words if profile else ())
    if search_words and profile:
        words |= set(profile.words)
    return words - inputs


def _topic_weights(goal: str, modality: str, profile: _Profile | None) -> dict[str, int]:
    """The words that can name the label's column (the goal's content words and the modality's label words, less the
    input words), each with a weight: 4 for the goal's focus (the first of its words after what/which/whose, else
    its first word), 2 for its next word, 0 for the others and for the modality's label words. The focus outweighs
    the 3 points a name loses for a word such as "name", so "country_name" still beats "region" for "Which
    country ...?"."""
    inputs = set(profile.input_words) if profile else set(_content_words(modality))
    words = [w for w in _content_words(goal) if w not in inputs]
    raw = [_singular(w) for w in re.findall(r"[a-z0-9]+", str(goal).lower())]
    asked = next((i for i, w in enumerate(raw) if w in _ASK), None)
    if asked is not None:
        focus = next((w for w in raw[asked + 1:] if w in words), None)
        if focus:
            words = [focus] + [w for w in words if w != focus]
    weights = {w: 0 for w in (profile.label_words if profile else ()) if w not in inputs}
    for rank, word in enumerate(words):
        weights[word] = max(0, 4 - 2 * rank)
    return weights


def _rule_label(weights: dict[str, int], columns, rows, inputs: set, on_topic: bool, goal_words: list[str],
                wanted: list[str]):
    """(column, kind, separator, names) of the best label, or None. A column named for the goal's topic always
    counts; a generic one (label, class, category, any ClassLabel, a block of 0/1 columns) only when the dataset
    itself is named for the topic (on_topic) or its values name the goal's words or the wanted labels, so an
    unrelated benchmark's labels are never taken for the goal's. Score: 4 for a name with a topic word plus that
    word's weight (_topic_weights), 3 for a bare label word (label, class, ...) or 2 for one inside the name, -3 for
    a name word that marks no label or a model's output (id, name, pred, model, prob, ...), -3 for a longer variant
    of a bare topic column that is itself a usable label ("qwen_emotion" beside "emotion": a model's prediction
    beside the ground truth, so the exact name ranks above it), +1 for a ClassLabel. Ties go to the column with
    fewer words, then the first."""
    topic = tuple(weights)
    usable = [c for c in columns.values() if c.name not in inputs and c.kind in ("classlabel", "string", "list")]
    fits = {c.name: _label_fit(c, rows) for c in usable}
    bare = {w for c in usable if fits[c.name] for w in topic if [_singular(t) for t in _tokens(c.name)] == [w]}
    scored = []
    for position, col in enumerate(usable):
        tokens = _tokens(col.name)
        hits = [w for w in topic if _name_hits(col.name, (w,))]
        if not hits and not on_topic and not _values_name_goal(col, rows, goal_words, wanted):
            continue
        score = 0
        if hits:
            score = 4 + max(weights[w] for w in hits)
            if bare & set(hits) and not any([_singular(t) for t in tokens] == [w] for w in hits):
                score -= 3
        score += 3 if col.name.lower() in _LABEL_WORDS else 2 if _name_hits(col.name, _LABEL_WORDS) else 0
        score -= 3 if set(tokens) & _NOT_LABEL else 0
        if col.kind == "classlabel":
            score = max(score, 0) + 1
        if score > 0 and fits[col.name] is not None:
            fit = fits[col.name]
            scored.append(((score, -len(tokens), -position), (col.name, fit[0], fit[1], col.names)))
    if scored:
        return max(scored, key=lambda item: item[0])[1]
    if not on_topic:
        return None
    binary = [c.name for c in columns.values() if c.kind == "number" and c.name not in inputs
              and not set(_tokens(c.name)) & _NOT_LABEL and _binary(c.name, rows)]
    if len(binary) >= 3:
        return "", "multihot", None, binary
    return None


def _values_name_goal(col: _Column, rows, goal_words: list[str], wanted: list[str]) -> bool:
    """Whether a column's label values (its ClassLabel names, else the values in the rows) include the wanted
    labels (all of a ClassLabel's, at least one in the rows) or, with none wanted, two of the goal's words (one if
    the goal has only one): a ClassLabel cat/dog/... for "Is this a cat or a dog?"."""
    if col.names:
        values = col.names
    else:
        values = [v for _, row, _ in rows for v in (row.get(col.name) if isinstance(row.get(col.name), list)
                                                     else [row.get(col.name)]) if isinstance(v, str)]
    keys = {_key(v) for v in values} | {_singular(_key(v)) for v in values}
    if wanted:
        found = [w for w in wanted if _key(w) in keys]
        return len(found) == len(wanted) if col.names else bool(found)
    need = min(2, len(goal_words))
    return need > 0 and sum(1 for w in goal_words if w in keys) >= need


def _binary(name: str, rows) -> bool:
    values = [row.get(name) for _, row, _ in rows if row.get(name) is not None]
    return bool(values) and all(v in (0, 1) for v in values) and any(v == 1 for v in values)


def _label_fit(col: _Column, rows) -> tuple[str, str | None] | None:
    """(label kind, separator) when the column can label probes, else None. A string column is a "list" label when
    it holds JSON/Python list literals, or when ";" or "|" joins values in 30% of its cells; "," and "/" (which also
    occur inside single names such as "Korea, Republic of") count only when the joined parts are short and, in at
    least half of those cells, stand alone elsewhere in the column."""
    if col.kind == "classlabel":
        return ("classlabel", None) if col.names and len(col.names) >= 2 else None
    cells = [row.get(col.name) for _, row, _ in rows]
    present = [c for c in cells if c is not None and c != "" and c != []]
    if len(present) < max(2, len(cells) // 2):
        return None
    if col.kind == "list":
        if col.names is not None:
            return ("list", None) if len(col.names) >= 2 else None
        values = [v.strip() for c in present if isinstance(c, list) for v in c if isinstance(v, str) and v.strip()]
        return ("list", None) if len(set(values)) >= 2 else None
    if col.kind != "string":
        return None
    texts = [c.strip() for c in present if isinstance(c, str) and c.strip()]
    if not texts:
        return None
    literals = [_list_literal(t) for t in texts]
    if sum(v is not None for v in literals) >= 0.5 * len(texts):
        values = [v for parsed in literals if parsed for v in parsed]
        return ("list", LIST_LITERAL) if len(set(values)) >= 2 else None
    for separator in (";", "|", ",", "/"):
        joined = [t for t in texts if separator in t]
        if len(joined) < 0.3 * len(texts):
            continue
        values = [v.strip() for t in texts for v in t.split(separator) if v.strip()]
        if not 2 <= len(set(values)) <= 0.8 * len(values):
            continue
        if separator in (",", "/"):
            whole = {_key(t) for t in texts if separator not in t}
            alone = sum(1 for t in joined if any(_key(p) in whole for p in t.split(separator) if p.strip()))
            if max(map(len, values)) > 40 or alone < 0.5 * len(joined):
                continue
        return "list", separator
    distinct = len(set(texts))
    return ("string", None) if 2 <= distinct <= 0.6 * len(texts) and max(map(len, texts)) <= 100 else None


def _list_literal(text: str) -> list[str] | None:
    """The strings of a JSON or Python list literal such as "['fruity', 'green']", else None."""
    if not (text.startswith("[") and text.endswith("]")) or len(text) > 2000:
        return None
    try:
        value = json.loads(text)
    except ValueError:
        try:
            value = ast.literal_eval(text)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            return None
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        return None
    return [v.strip() for v in value if v.strip()]


# ----------------------------------------------------------------------------- fetching probes
def fetch_probes(candidate: HubCandidate | TableCandidate | FolderCandidate, mapping: Mapping, dest: Path, *,
                 wanted_labels: list[str] | None = None, n_per_label: int = 20, max_labels: int = 8,
                 min_per_label: int = 6, budget_s: float = 120, max_bytes: int = 150_000_000,
                 disjoint_from: list[str] | tuple = ()) -> FetchReport:
    dest = Path(dest)
    shutil.rmtree(dest, ignore_errors=True)  # dest belongs to this fetch: an older sample is replaced
    try:
        dest.mkdir(parents=True, exist_ok=True)
        start = _now()
        limits = _Limits(list(wanted_labels or []), n_per_label, max_labels, min_per_label, max_bytes,
                         _disjoint(disjoint_from, wanted_labels))
        if isinstance(candidate, TableCandidate):
            return _fetch_table(candidate, mapping, dest, limits, start + budget_s)
        if isinstance(candidate, FolderCandidate):
            return _fetch_folder(candidate, mapping, dest, limits, start + budget_s)
        return _fetch(candidate, mapping, dest, limits, start, start + budget_s)
    except ProbesUnavailable:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    except OSError as exc:
        shutil.rmtree(dest, ignore_errors=True)
        raise ProbesUnavailable(f"could not write probes under {dest}: {exc}") from exc
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        raise


@dataclass(frozen=True)
class _Limits:
    wanted: list[str]
    n_per_label: int
    max_labels: int
    min_per_label: int
    max_bytes: int
    disjoint: tuple[str, ...] = ()  # labels never to choose (disjoint_from), as given


def _disjoint(values, wanted) -> tuple[str, ...]:
    """disjoint_from as strings; a wanted label that is also in it is a contradiction."""
    out = tuple(str(v).strip() for v in values or () if isinstance(v, str) and v.strip())
    keys = _loose_keys(out)
    clash = [w for w in wanted or [] if _loose_hit(w, keys)]
    if clash:
        raise ProbesUnavailable(f"wanted labels {clash} are also in disjoint_from")
    return out


def _loose_keys(labels) -> set[str]:
    """Label keys with case, separators and a plural ending ignored ("Hip-Hop", "hip_hop" and "hiphops" alike),
    plus the other names of a country (see _country_aliases), so "CN" and "China" count as one label."""
    keys = set()
    for label in labels:
        key = re.sub(r"[\W_]+", "", str(label).lower())
        if key:
            keys |= {key, _singular(key)} | _country_aliases().get(key, set())
    return keys


def _loose_hit(label: str, keys: set[str]) -> bool:
    key = re.sub(r"[\W_]+", "", str(label).lower())
    return bool(key) and (key in keys or _singular(key) in keys or bool(_country_aliases().get(key, set()) & keys))


_country_alias_cache: dict[str, set[str]] | None = None


def _country_aliases() -> dict[str, set[str]]:
    """Loose key -> loose keys of the same country: its ISO 3166 alpha-2 code and Natural Earth's country name (with
    and without "of America" and "The"), read once from the gazetteer sense.adapter_kit uses (SENSE_GAZETTEER, else
    ~/.cache/sense-harness/gazetteer/ne_10m_populated_places_simple.geojson); {} when it is missing or unreadable.
    Used only to keep a second probe set's categories disjoint from the first one's, where one source labels
    countries by code and the other by name. A mistaken alias can only drop more labels, never let one through."""
    global _country_alias_cache
    if _country_alias_cache is None:
        from sense.adapters import country_aliases  # the same table verify_files uses for its unseen control

        _country_alias_cache = country_aliases()
    return _country_alias_cache


def _fetch(candidate, mapping, dest, limits: _Limits, start, deadline):
    hub_id = candidate.hub_id
    params = {"dataset": hub_id, "config": mapping.config, "split": mapping.split}
    what = f"the viewer's rows of {hub_id}"
    skipped: Counter = Counter()
    notes: list[str] = []
    # Media still has its downloads to do: reading /rows pages may use at most half the budget.
    reserve = (deadline - start) / 2 if mapping.input_kind == "media" else 5.0
    reading = deadline - reserve
    first, refused, reason = _first_page(params, what=what, deadline=reading, fallback_deadline=deadline)
    total = first.get("num_rows_total") if isinstance(first.get("num_rows_total"), int) else PAGE_ROWS
    if refused:
        total = _split_rows(params, reading) if reason in _PASSING else None
        if total is None:
            total = PAGE_ROWS
            notes.append(f"only the split's first rows could be read ({refused})")
        else:
            notes.append(f"the first rows page failed ({refused}), so the first rows came from /first-rows and the "
                         "split's size from /size")
    seen: dict[int, tuple[str, list[str]]] = {}  # row index -> (input text or URL, label values)
    read: set[int] = set()
    rows_seen = _take(first, mapping, seen, read, skipped)
    for offset in _offsets(total)[1:]:
        if reading - _now() < 1:
            notes.append(f"stopped reading rows at the time budget after {rows_seen} rows")
            break
        try:  # once a page has failed even when retried, the viewer is struggling: later pages get one try each
            page = _rows_page(params, offset, what=what, timeout=ROWS_TIMEOUT_S, deadline=reading,
                              retry=not skipped["page_failed"])
        except ProbesUnavailable as exc:
            skipped["page_failed"] += 1
            notes.append(f"a page at offset {offset} failed: {exc}"
                         + ("; later pages get one try each" if skipped["page_failed"] == 1 else ""))
            continue
        rows_seen += _take(page, mapping, seen, read, skipped)
    report = _write_probes(hub_id, mapping, dest, seen, rows_seen, skipped, notes, limits, deadline)
    report.licence, report.source_url, report.hub_id = candidate.licence, candidate.url, hub_id
    report.licence_source = "the Hub dataset card" if candidate.licence else None
    return report


def _write_probes(name: str, mapping: Mapping, dest: Path, seen: dict, rows_seen: int, skipped: Counter,
                  notes: list, limits: _Limits, deadline: float, seed: str | None = None) -> FetchReport:
    """The rows read (seen: row index -> (input, labels)) into dest/<label>/<row index><extension>: inputs merged,
    labels chosen, multi-labelled and leaking rows dropped, files written or downloaded. Each label's inputs are
    picked evenly spread over them, starting from a point that the seed (if any) and the label set. The caller
    fills in the licence and the source."""
    if not seen:
        raise ProbesUnavailable(f"none of the {rows_seen} rows read from {name} had both an input and a label")
    entries, mixed = _merge_inputs(seen, skipped, notes)
    multi = mapping.label_kind in ("list", "multihot") or bool(mixed)
    chosen = _choose_labels(entries, mapping, limits.wanted, limits.max_labels, limits.min_per_label, notes, name,
                            multi, limits.disjoint)
    by_label = _assign(entries, chosen, mapping, mixed, skipped, notes)
    if mapping.input_kind == "text":
        _leak_check(by_label, chosen, skipped, notes, name)
    folders = {value: safe_label(value, i) for i, value in enumerate(chosen)}
    n_per_label, min_per_label = limits.n_per_label, limits.min_per_label
    queues = {value: _spread(sorted(by_label[value]), n_per_label, _phase(seed, value)) for value in chosen}
    if mapping.input_kind == "text":
        counts, sizes = _write_texts(queues, folders, mapping, dest, n_per_label)
    else:
        counts, sizes, scanned = _download_all(queues, folders, mapping, dest, n_per_label, deadline,
                                               limits.max_bytes, skipped, notes)
        if scanned and skipped["leak"] * 2 > scanned:
            raise ProbesUnavailable(f"{name}'s files name their own label in their metadata in {skipped['leak']} "
                                    f"of {scanned} downloaded files, so they cannot test a sense")
    kept = {}
    for value in chosen:
        if counts[value] >= min_per_label:
            kept[value] = counts[value]
        else:
            shutil.rmtree(dest / folders[value], ignore_errors=True)
            notes.append(f"dropped label {value!r}: {counts[value]} files, fewer than {min_per_label}")
    missing = [w for w in limits.wanted if _key(w) not in {_key(v) for v in kept}]
    if missing:
        raise ProbesUnavailable(f"wanted labels without {min_per_label} probe files in {name}: {missing}")
    if len(kept) < 2:
        raise ProbesUnavailable(f"only {len(kept)} label(s) of {name} reached {min_per_label} probe files "
                                f"({sum(counts.values())} files written; skipped: {dict(skipped)})")
    order = sorted(kept, key=lambda v: -kept[v])
    return FetchReport(dest, [folders[v] for v in order], {folders[v]: kept[v] for v in order}, rows_seen,
                       dict(skipped), None, "", notes, "", {folders[v]: v for v in order},
                       sum(sizes[v] for v in order))


def _offsets(total: int) -> list[int]:
    """Page offsets: every page when the split fits in MAX_PAGES pages, else MAX_PAGES spread evenly over it."""
    if total <= PAGE_ROWS * MAX_PAGES:
        return list(range(0, max(total, 1), PAGE_ROWS))
    last = total - PAGE_ROWS
    return sorted({int(k * last / (MAX_PAGES - 1)) for k in range(MAX_PAGES)})


def _take(page: dict, mapping: Mapping, seen: dict, read: set, skipped: Counter) -> int:
    """Read one page's rows into seen; returns how many new rows it held."""
    return _take_rows(_rows(page), mapping, seen, read, skipped)


def _take_rows(rows, mapping: Mapping, seen: dict, read: set, skipped: Counter) -> int:
    n = 0
    for idx, row, truncated in rows:
        if idx in read:
            continue
        read.add(idx)
        n += 1
        if truncated & set(mapping.input_columns):
            skipped["truncated"] += 1
            continue
        value = _input_of(row, mapping)
        if value is None:
            skipped["no_input"] += 1
            continue
        labels = _labels_of(row, mapping)
        if not labels:
            skipped["no_label"] += 1
            continue
        seen[idx] = (value, labels)
    return n


def _input_of(row: dict, mapping: Mapping) -> str | None:
    if mapping.input_kind == "media":
        return _src(row.get(mapping.input_columns[0]))
    values = {}
    for name in mapping.input_columns:
        value = row.get(name)
        if value is None or isinstance(value, (dict, list)) or not str(value).strip():
            return None
        values[name] = str(value).strip()
    text = re.sub(r"\{([^{}]*)\}", lambda m: values.get(m.group(1), m.group(0)), mapping.template)
    return text if len(text) <= MAX_TEXT_CHARS else None


def _labels_of(row: dict, mapping: Mapping) -> list[str]:
    if mapping.label_kind == "multihot":
        return [name for name in mapping.label_names or [] if row.get(name) in (1, True)]
    cell = row.get(mapping.label_column)
    names = mapping.label_names
    if mapping.label_kind == "classlabel":
        return [names[cell]] if isinstance(cell, int) and not isinstance(cell, bool) and 0 <= cell < len(names) else []
    if mapping.label_kind == "string":
        return [cell.strip()] if isinstance(cell, str) and cell.strip() else []
    if isinstance(cell, str) and mapping.label_separator == LIST_LITERAL:
        cell = _list_literal(cell.strip()) or []
    elif isinstance(cell, str) and mapping.label_separator:
        cell = cell.split(mapping.label_separator)
    if not isinstance(cell, list):
        return []
    out = []
    for value in cell:
        if names is not None and isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(names):
            value = names[value]
        if isinstance(value, str) and value.strip() and value.strip() not in out:
            out.append(value.strip())
    return out


def _key(label: str) -> str:
    return re.sub(r"[\s_-]+", " ", str(label).strip().lower())


def _merge_inputs(seen: dict, skipped: Counter, notes: list) -> tuple[dict, set]:
    """One entry per distinct input (text or file URL): a row repeating an earlier row's input is merged into it,
    their labels joined. Returns the entries and the set of entries whose rows disagree on the label (long-format
    data, one row per input and label): those are multi-labelled, so a fetch keeps them only with exactly one of
    the chosen labels."""
    head_of: dict[str, int] = {}
    entries: dict[int, tuple[str, list[str]]] = {}
    mixed: set[int] = set()
    repeats = 0
    for idx in sorted(seen):
        value, labels = seen[idx]
        head = head_of.setdefault(value, idx)
        if head == idx:
            entries[idx] = (value, list(labels))
            continue
        repeats += 1
        known = entries[head][1]
        if set(labels) != set(known):
            mixed.add(head)
        entries[head] = (value, known + [v for v in labels if v not in known])
    if repeats:
        skipped["duplicate_input"] += repeats
        notes.append(f"{repeats} rows repeat an earlier row's input and were merged into it"
                     + (f"; {len(mixed)} inputs carry different labels in different rows" if mixed else ""))
    return entries, mixed


def _choose_labels(entries, mapping, wanted, max_labels, min_per_label, notes, hub_id, multi,
                   disjoint: tuple[str, ...] = ()) -> list[str]:
    rows_labels = [labels for _, labels in entries.values()]
    freq = Counter(v for labels in rows_labels for v in labels)
    if wanted:
        known = {}
        for value in list(mapping.label_names or []) + list(freq):
            known.setdefault(_key(value), value)
        missing = [w for w in wanted if _key(w) not in known]
        if missing:
            sample = ", ".join(sorted(freq, key=lambda v: -freq[v])[:15])
            raise ProbesUnavailable(f"wanted labels not among {hub_id}'s label values: {missing} (seen: {sample})")
        chosen = list(dict.fromkeys(known[_key(w)] for w in wanted))
        folders = [_folder(v) for v in chosen]
        if None in folders or len(set(folders)) < len(folders):
            raise ProbesUnavailable(f"wanted labels {chosen} have no distinct ASCII folder names")
        notes.append(f"labels are the wanted ones, matched to the dataset's values: {chosen}")
        return chosen
    observed = {_key(v) for v in freq}
    avoid, avoided = _loose_keys(disjoint), []
    chosen, taken, combined = [], set(), set()
    for value, count in sorted(freq.items(), key=lambda kv: -kv[1]):
        if len(chosen) >= max_labels or count < min_per_label:
            break
        if avoid and _loose_hit(value, avoid):
            avoided.append(value)
            continue
        folder = _folder(value)
        if folder is None or folder in taken:
            notes.append(f"skipped label {value!r}: its folder name would be empty or shared")
            continue
        if _key(value) in _PLACEHOLDERS:
            notes.append(f"skipped label {value!r}: it names no category")
            continue
        parts = _parts(value)
        if parts and (parts & (observed - {_key(value)}) or parts & combined):
            notes.append(f"skipped label {value!r}: it combines other label values ({', '.join(sorted(parts))}), so "
                         "its rows carry several labels")
            continue
        trial = chosen + [value]
        if multi and min(_exclusive(rows_labels, trial).values()) < min_per_label:
            continue
        chosen, taken, combined = trial, taken | {folder}, combined | parts
    if avoided:
        notes.append(f"labels in disjoint_from were not chosen: {', '.join(map(repr, avoided[:12]))}"
                     + (f" and {len(avoided) - 12} more" if len(avoided) > 12 else "")
                     + "; a row carrying one counts only for another label it carries, like any unchosen label")
    if len(chosen) < 2:
        raise ProbesUnavailable(f"{hub_id} has fewer than 2 label values with {min_per_label} usable rows among "
                                f"{len(entries)} inputs read"
                                + (" (inputs carrying exactly one label)" if multi else "")
                                + (f", once the {len(avoided)} labels in disjoint_from are left out" if avoided
                                   else ""))
    return chosen


def _parts(value: str) -> set[str]:
    """The values a cell such as "fruity, green" or "rock/pop" joins, if it joins two or more, else an empty set."""
    parts = {_key(p) for p in _PARTS.split(str(value)) if p.strip()}
    return parts if len(parts) >= 2 else set()


def _folder(value: str) -> str | None:
    folder = safe_label(value, 0)
    return None if folder == "label_0" and _key(value) != "label 0" else folder


def _exclusive(rows_labels: list[list[str]], chosen: list[str]) -> Counter:
    wanted = set(chosen)
    counts = Counter({v: 0 for v in chosen})
    for labels in rows_labels:
        hit = wanted.intersection(labels)
        if len(hit) == 1:
            counts[hit.pop()] += 1
    return counts


def _assign(entries, chosen, mapping, mixed, skipped, notes) -> dict[str, list[tuple[int, str]]]:
    """Each chosen label's inputs; an input carrying two or more chosen labels (in one row of a list or multihot
    label, or across rows: mixed) is dropped."""
    wanted = set(chosen)
    by_label: dict[str, list[tuple[int, str]]] = {v: [] for v in chosen}
    multi = conflicting = 0
    for idx, (value, labels) in entries.items():
        hit = wanted.intersection(labels)
        if len(hit) == 1:
            by_label[hit.pop()].append((idx, value))
        elif len(hit) > 1 and idx in mixed:
            conflicting += 1
        elif len(hit) > 1:
            multi += 1
    if mapping.label_kind in ("list", "multihot"):
        skipped["multi_label"] += multi
        notes.append(f"the dataset gives several labels per row: only rows carrying exactly one of the chosen labels "
                     f"are kept ({multi} rows with two or more were dropped); other labels a row carries are ignored")
    if mixed:
        skipped["conflicting_labels"] += conflicting
        notes.append(f"the same input appears in several rows with different labels: only inputs carrying exactly one "
                     f"of the chosen labels are kept ({conflicting} inputs with two or more were dropped)")
    return by_label


def _leak_check(by_label, chosen, skipped, notes, hub_id) -> None:
    total = leaked = 0
    for value in chosen:
        pattern = _word_pattern(value)
        clean = [(idx, text) for idx, text in by_label[value] if not pattern.search(text)]
        total += len(by_label[value])
        leaked += len(by_label[value]) - len(clean)
        by_label[value] = clean
    if leaked:
        skipped["leak"] += leaked
        notes.append(f"dropped {leaked} of {total} rows whose input text names their own label")
    if total and leaked * 2 > total:
        raise ProbesUnavailable(f"{hub_id}'s input text names the label in {leaked} of {total} rows, so it cannot "
                                "test a sense")


def _label_words(label: str) -> list[str]:
    return re.findall(r"[^\W_]+", str(label).lower())


def _word_pattern(label: str) -> re.Pattern:
    """The label as a whole word in any case, its words joined by any or no space, "_" or "-" ("hip hop" also
    matches hip-hop, hip_hop and hiphop). A label of digits only must not touch a "-" or a decimal point either, so
    "1" is not found in the coordinates "1.5, -2.75"."""
    words = _label_words(label)
    if not words:
        return re.compile(r"(?!)")
    body = r"[\s_\-]*".join(re.escape(w) for w in words)
    if all(w.isdigit() for w in words):
        return re.compile(rf"(?<![^\W_])(?<![.\-])(?:{body})(?![^\W_])(?!\.\d)", re.IGNORECASE)
    return re.compile(rf"(?<![^\W_])(?:{body})(?![^\W_])", re.IGNORECASE)


def _phase(seed: str | None, label: str) -> float:
    """Where in [0, 1) of the first step a label's evenly spaced picks start: 0 without a seed."""
    if seed is None:
        return 0.0
    return int(hashlib.sha256(f"{seed}\n{label}".encode()).hexdigest()[:8], 16) / 2 ** 32


def _spread(items: list, k: int, phase: float = 0.0) -> list:
    """k items evenly spaced through the list first (from phase, in [0, 1), of the first step), then the rest as
    spares for failed downloads."""
    if len(items) <= k:
        return list(items)
    picks = sorted({int((i + phase) * len(items) / k) for i in range(k)})
    chosen = set(picks)
    return [items[j] for j in picks] + [x for j, x in enumerate(items) if j not in chosen]


def _write_texts(queues, folders, mapping, dest, n_per_label) -> tuple[Counter, Counter]:
    counts, sizes = Counter({value: 0 for value in queues}), Counter()
    for value, queue in queues.items():
        for idx, text in queue[:n_per_label]:
            data = (text + "\n").encode("utf-8")
            _write(dest / folders[value] / f"{idx}{mapping.extension}", data)
            counts[value] += 1
            sizes[value] += len(data)
    return counts, sizes


def _download_all(queues, folders, mapping, dest, n_per_label, deadline, max_bytes, skipped, notes):
    """Download with DOWNLOAD_WORKERS at once, always for the label with the fewest files so far, until each label is
    full or out of rows, or the byte or time budget is spent. Each worker strips the file's metadata
    (_strip_metadata); then a file whose text still names its label is dropped as a leak, a file already written
    under the same label is a duplicate, and the same bytes under two labels are removed from both. A file that
    would take the written total over max_bytes is discarded, so the folder never holds more than max_bytes.
    Returns (counts, sizes, files scanned for leaks)."""
    pending = {value: deque(queue) for value, queue in queues.items()}
    counts, busy, sizes = Counter({value: 0 for value in queues}), Counter(), Counter()
    default, known = _MEDIA_EXT.get(mapping.media_type, (mapping.extension, ()))
    ffmpeg = _ffmpeg() if mapping.media_type in ("Audio", "Video") else None
    spent, stop, running = 0, None, {}
    written: dict[str, tuple[str, Path, int]] = {}  # sha256 -> (label, file, size)
    clashed: set[str] = set()
    scanned, cleaning = 0, Counter()

    def next_row():
        open_labels = [v for v in pending if pending[v] and counts[v] + busy[v] < n_per_label]
        if not open_labels:
            return None
        value = min(open_labels, key=lambda v: counts[v] + busy[v])
        return value, *pending[value].popleft()

    pool = ThreadPoolExecutor(DOWNLOAD_WORKERS)
    try:
        while True:
            while stop is None and len(running) < DOWNLOAD_WORKERS:
                if spent >= max_bytes:
                    stop = f"stopped downloading at the {max_bytes:,} byte cap"
                elif deadline - _now() < 1:
                    stop = "stopped downloading at the time budget"
                elif scanned >= LEAK_SAMPLE and skipped["leak"] * 2 > scanned:
                    stop = "stopped downloading: most files name their own label in their metadata"
                elif (job := next_row()) is None:
                    break
                elif not _allowed_url(job[2]):
                    skipped["host_refused"] += 1
                else:
                    value, idx, src = job
                    suffix = PurePosixPath(urllib.parse.urlsplit(src).path).suffix.lower()
                    ext = suffix if suffix in known else mapping.extension or default
                    busy[value] += 1
                    left = deadline - _now()
                    running[pool.submit(_download_one, src, ext, mapping.media_type, ffmpeg, min(TIMEOUT_S, left),
                                        min(FILE_MAX_BYTES, max_bytes - spent), left)] = (value, idx, ext)
            if not running:
                break
            done, _ = wait(running, timeout=max(0.5, deadline + GRACE_S - _now()), return_when=FIRST_COMPLETED)
            if not done:
                skipped["download_failed"] += len(running)
                stop = f"stopped downloading at the time budget ({len(running)} downloads abandoned)"
                break
            for future in done:
                value, idx, ext = running.pop(future)
                busy[value] -= 1
                try:
                    data, how = future.result()
                except _HttpError as exc:
                    skipped[exc.reason if exc.reason in ("too_large", "host_refused") else "download_failed"] += 1
                    continue
                if not data:
                    skipped["download_failed"] += 1
                    continue
                scanned += 1
                cleaning[how] += 1
                if _names_label(data, value):
                    skipped["leak"] += 1
                    continue
                digest = hashlib.sha256(data).hexdigest()
                if digest in clashed:
                    skipped["conflicting_labels"] += 1
                    continue
                if digest in written:
                    label, path, size = written[digest]
                    if label == value:
                        skipped["duplicate_input"] += 1
                        continue
                    path.unlink(missing_ok=True)  # the same file under two labels: keep neither
                    counts[label] -= 1
                    sizes[label] -= size
                    spent -= size
                    del written[digest]
                    clashed.add(digest)
                    skipped["conflicting_labels"] += 2
                    continue
                if spent + len(data) > max_bytes:
                    skipped["over_byte_cap"] += 1
                    stop = f"stopped downloading at the {max_bytes:,} byte cap"
                    continue
                target = dest / folders[value] / f"{idx}{ext}"
                _write(target, data)
                written[digest] = (value, target, len(data))
                counts[value] += 1
                sizes[value] += len(data)
                spent += len(data)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)  # an abandoned download never writes: only this loop does
    if cleaning["stripped"]:
        how = {"Audio": "tags, by an ffmpeg remux", "Video": "tags, by an ffmpeg remux",
               "Mesh": "glTF names and extras; OBJ/OFF comments and object names, PLY comments, STL solid names",
               "Image": "PNG text chunks, JPEG EXIF/XMP/comments",
               "Molecule": "molfile title and comment lines, SDF data fields, SMILES names",
               "Geo": "GeoJSON properties, KML/GPX names, descriptions and metadata"}.get(
            mapping.media_type, "names and tags")
        notes.append(f"stripped the metadata that can name a label from {cleaning['stripped']} files ({how})")
    if cleaning["unstripped"]:
        why = "ffmpeg is not installed" if mapping.media_type in ("Audio", "Video") and not ffmpeg else \
            "they could not be parsed or remuxed"
        notes.append(f"{cleaning['unstripped']} files kept their metadata ({why}); the leak scan still read them")
    if skipped["leak"]:
        notes.append(f"dropped {skipped['leak']} of {scanned} downloaded files whose metadata names their own label")
    if skipped["conflicting_labels"] or skipped["duplicate_input"]:
        notes.append(f"identical files: {skipped['duplicate_input']} repeats within a label skipped, "
                     f"{skipped['conflicting_labels']} under two labels removed")
    if stop:
        notes.append(stop)
    return counts, sizes, scanned


def _download_one(src: str, ext: str, media_type: str | None, ffmpeg: str | None, timeout: float, max_bytes: int,
                  strip_s: float) -> tuple[bytes, str]:
    data = _http_get(src, timeout=timeout, max_bytes=max_bytes)
    return _strip_metadata(data, ext, media_type, ffmpeg, strip_s) if data else (data, "kept")


def _write(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.part")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def safe_label(text: str, index: int) -> str:
    """Only a-z, 0-9 and inner "_", at most 40 characters, never empty: so never ".", ".." or a path."""
    out = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:40].strip("_")
    return out or f"label_{index}"


# ----------------------------------------------------------------------------- media metadata
def _names_label(data: bytes, label: str) -> bool:
    """Whether the file's text runs (ASCII or UTF-16, in its first and last SCAN_BYTES, where containers keep their
    metadata) name the label as a word: a source path ...\\drink\\clip.avi, an ID3 genre tag. A label shorter than
    3 characters or of digits only is not looked for (random bytes spell those too often); a 3-character one only
    inside runs of 6 or more."""
    letters = "".join(_label_words(label))
    if len(letters) < 3 or letters.isdigit():
        return False
    pattern = _word_pattern(label)
    head = data if len(data) <= 2 * SCAN_BYTES else data[:SCAN_BYTES] + b"\x00" + data[-SCAN_BYTES:]
    return any(pattern.search(run) for run in _text_runs(head, 6 if len(letters) < 4 else 3))


def _text_runs(data: bytes, shortest: int):
    for match in re.finditer(rb"[\x20-\x7e]{%d,}" % shortest, data):
        yield match.group().decode("ascii")
    for match in re.finditer(rb"(?:[\x20-\x7e]\x00){%d,}" % shortest, data):
        yield match.group().decode("utf-16-le")
    for match in re.finditer(rb"(?:\x00[\x20-\x7e]){%d,}" % shortest, data):
        yield match.group().decode("utf-16-be")


def _ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def _strip_metadata(data: bytes, ext: str, media_type: str | None, ffmpeg: str | None,
                    timeout_s: float) -> tuple[bytes, str]:
    """(bytes, how) with the metadata that can name a file's label removed and the content left as it is. how is
    "stripped", "kept" (a format with no known name carrier here; the leak scan still reads it) or "unstripped"
    (it could not be parsed, or ffmpeg is missing or failed)."""
    try:
        if data[:4] == b"glTF":
            out = _strip_glb(data)
        elif ext == ".gltf":
            out = _strip_gltf(data)
        elif data.startswith(_PNG_SIGNATURE):
            out = _strip_png(data)
        elif data[:3] == b"\xff\xd8\xff":
            out = _strip_jpeg(data)
        elif ext in _FILE_STRIPPERS:
            out = _FILE_STRIPPERS[ext](data)
        elif media_type in ("Audio", "Video"):
            out = _remux(data, ext, media_type, ffmpeg, timeout_s) if ffmpeg else None
        else:
            return data, "kept"
    except (ValueError, struct.error, RecursionError):
        out = None
    return (out, "stripped") if out else (data, "unstripped")


_GLB_JSON = 0x4E4F534A


def _strip_glb(data: bytes) -> bytes | None:
    """The GLB with "name" and "extras" removed from every object in its JSON chunk (scenes, nodes, meshes,
    materials, images, ...: they often name the object), its binary chunk as it was; None when malformed."""
    if len(data) < 20:
        return None
    version, length = struct.unpack_from("<II", data, 4)
    size, kind = struct.unpack_from("<II", data, 12)
    if version != 2 or length > len(data) or kind != _GLB_JSON or 20 + size > length:
        return None
    doc = _drop_names(json.loads(data[20:20 + size].rstrip(b"\x00 ")))
    if doc is None:
        return None
    body = json.dumps(doc, separators=(",", ":")).encode("ascii")
    body += b" " * (-len(body) % 4)
    rest = data[20 + size:length]
    return (b"glTF" + struct.pack("<II", 2, 20 + len(body) + len(rest)) + struct.pack("<II", len(body), _GLB_JSON)
            + body + rest)


def _strip_gltf(data: bytes) -> bytes | None:
    doc = _drop_names(json.loads(data))
    return None if doc is None else json.dumps(doc, separators=(",", ":")).encode("ascii")


def _drop_names(doc: object) -> dict | None:
    """The glTF document without any "name" or "extras" (extension data is left alone), or None if not a glTF."""
    if not isinstance(doc, dict) or not isinstance(doc.get("asset"), dict):
        return None

    def drop(node):
        if isinstance(node, dict):
            node.pop("name", None)
            node.pop("extras", None)
            for key, value in node.items():
                if key != "extensions":
                    drop(value)
        elif isinstance(node, list):
            for value in node:
                drop(value)

    drop(doc)
    return doc


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_TEXT = frozenset((b"tEXt", b"zTXt", b"iTXt", b"eXIf", b"tIME"))


def _strip_png(data: bytes) -> bytes | None:
    """The PNG without its text, EXIF and time chunks; None when malformed."""
    out, pos = [_PNG_SIGNATURE], len(_PNG_SIGNATURE)
    while pos + 12 <= len(data):
        length, kind = struct.unpack_from(">I4s", data, pos)
        end = pos + 12 + length
        if end > len(data):
            return None
        if kind not in _PNG_TEXT:
            out.append(data[pos:end])
        pos = end
        if kind == b"IEND":
            return b"".join(out)
    return None


# APP1 (EXIF, XMP), APP3-APP13 (IPTC and makers' notes) and APP15, and comments; APP0 (JFIF), APP2 (ICC colour
# profile) and APP14 (Adobe colour transform) stay because they change how the pixels decode.
_JPEG_DROP = frozenset({0xE1, *range(0xE3, 0xEE), 0xEF, 0xFE})


def _strip_jpeg(data: bytes) -> bytes | None:
    """The JPEG without its metadata segments (the image data after start-of-scan is copied as it is)."""
    out, pos = [b"\xff\xd8"], 2
    while pos + 4 <= len(data):
        if data[pos] != 0xFF:
            return None
        marker = data[pos + 1]
        if marker == 0xFF:  # fill byte
            pos += 1
            continue
        if marker == 0xDA:
            out.append(data[pos:])
            return b"".join(out)
        length = int.from_bytes(data[pos + 2:pos + 4], "big")
        end = pos + 2 + length
        if length < 2 or end > len(data):
            return None
        if marker not in _JPEG_DROP:
            out.append(data[pos:end])
        pos = end
    return None


def _remux(data: bytes, ext: str, media_type: str, ffmpeg: str, timeout_s: float) -> bytes | None:
    """The file remuxed by ffmpeg with its streams copied as they are but no global, stream or chapter metadata
    (title, comment, genre, handler or source-path tags), no encoder tag, and only its audio (and, for video, its
    video) streams; None when ffmpeg fails or there is no time."""
    if timeout_s < 1 or not re.fullmatch(r"\.[a-z0-9]{1,5}", ext):
        return None
    maps = ["-map", "0:a"] if media_type == "Audio" else ["-map", "0:v", "-map", "0:a?"]
    with tempfile.TemporaryDirectory(prefix="hub-probes-") as tmp:
        source, target = Path(tmp) / f"in{ext}", Path(tmp) / f"out{ext}"
        source.write_bytes(data)
        command = [ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(source), *maps, "-c", "copy",
                   "-map_metadata", "-1", "-map_metadata:s", "-1", "-map_chapters", "-1",
                   "-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact", str(target)]
        try:
            done = subprocess.run(command, capture_output=True, timeout=min(FFMPEG_TIMEOUT_S, timeout_s))
        except (OSError, subprocess.SubprocessError):
            return None
        if done.returncode != 0 or not target.is_file():
            return None
        return target.read_bytes() or None


# Mesh, molecule and geometry files keep names in plain text: an OBJ's comments and o/g lines, a PLY's comment
# lines, an STL's solid name (or a binary STL's 80-byte header), an OFF's comments, a molfile's title and comment
# lines and an SDF's data fields, a SMILES file's names, GeoJSON properties and KML/GPX names and descriptions. Each
# helper drops those and leaves the geometry or structure byte for byte; None when the file is not of its format.
def _lines_without(data: bytes, drop: Callable[[bytes], bool]) -> bytes:
    return b"".join(line for line in data.splitlines(keepends=True) if not drop(line.strip()))


def _cut_comments(data: bytes) -> bytes:
    """data with every "#" comment cut from its line; a line holding only a comment is dropped."""
    out = []
    for line in data.splitlines(keepends=True):
        if b"#" in line:
            body = line.split(b"#", 1)[0].rstrip()
            if not body.strip():
                continue
            line = body + line[len(line.rstrip(b"\r\n")):]
        out.append(line)
    return b"".join(out)


def _strip_obj(data: bytes) -> bytes | None:
    if not re.search(rb"(?m)^[ \t]*v[ \t]", data[:SCAN_BYTES]):
        return None
    return _lines_without(_cut_comments(data), lambda line: line.split(None, 1)[:1] in (
        [b"o"], [b"g"], [b"mtllib"], [b"usemtl"]))


def _strip_off(data: bytes) -> bytes | None:
    if not re.match(rb"[A-Z0-9]{0,4}OFF", data.lstrip()[:12]):
        return None
    return _cut_comments(data)


def _strip_ply(data: bytes) -> bytes | None:
    end = data.find(b"end_header")
    newline = data.find(b"\n", end) if data.startswith(b"ply") and end >= 0 else -1
    if newline < 0:
        return None
    header = _lines_without(data[:newline + 1], lambda line: line.startswith((b"comment", b"obj_info")))
    return header + data[newline + 1:]


def _strip_stl(data: bytes) -> bytes | None:
    if len(data) >= 84 and 84 + 50 * struct.unpack_from("<I", data, 80)[0] == len(data):
        return b"\x00" * 80 + data[80:]  # a binary STL: its free-text 80-byte header often names the part
    if data.lstrip()[:5].lower() == b"solid":
        return re.sub(rb"(?im)^([ \t]*(?:end)?solid)[ \t][^\r\n]*", rb"\1", data)
    return None


def _strip_mol(data: bytes) -> bytes | None:
    """The molfile with its title, program/timestamp and comment lines (the first three) blanked, up to M  END."""
    lines = data.splitlines(keepends=True)
    end = next((i for i, line in enumerate(lines) if line.strip() == b"M  END"), None)
    if end is None or end < 3:
        return None
    blank = [line[len(line.rstrip(b"\r\n")):] or b"\n" for line in lines[:3]]
    return b"".join(blank + lines[3:end + 1])


def _strip_sdf(data: bytes) -> bytes | None:
    """Each record's molfile stripped (_strip_mol) and its data fields (> <name> blocks after M  END) dropped."""
    out, record = [], []
    for line in data.splitlines(keepends=True) + [None]:  # None ends the last record
        if line is not None and line.strip() != b"$$$$":
            record.append(line)
            continue
        if b"".join(record).strip():
            mol = _strip_mol(b"".join(record))
            if mol is None:
                return None
            out.append(mol + (line or b""))
        record = []
    return b"".join(out) or None


def _strip_smi(data: bytes) -> bytes | None:
    """One SMILES string a line, without the names that may follow it."""
    lines = [line.split()[0] for line in data.decode("ascii").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if not lines or not all(_SMILES.match(smiles) for smiles in lines):
        return None
    return ("\n".join(lines) + "\n").encode("ascii")


_GEO_KEYS = frozenset(("type", "features", "geometry", "geometries", "coordinates", "bbox"))


def _strip_geojson(data: bytes) -> bytes | None:
    """The GeoJSON with only its geometry: every Feature's properties emptied, names, ids and other keys dropped."""
    doc = json.loads(data)
    if not isinstance(doc, dict) or not isinstance(doc.get("type"), str):
        return None

    def clean(node):
        if isinstance(node, dict):
            out = {key: clean(value) for key, value in node.items() if key in _GEO_KEYS}
            if node.get("type") == "Feature":
                out["properties"] = {}
            return out
        return [clean(value) for value in node] if isinstance(node, list) else node

    return json.dumps(clean(doc), separators=(",", ":")).encode("ascii")


_XML_NAMED = re.compile(rb"<((?:\w+:)?(?:name|description|desc|cmt|keywords|author|metadata|ExtendedData|Snippet|"
                        rb"copyright|link|address|phoneNumber))\b[^>]*?(?:/>|>.*?</\1\s*>)", re.S | re.I)


def _strip_xml(data: bytes) -> bytes | None:
    """A KML or GPX file without its comments and its name, description, metadata and link elements."""
    if not re.search(rb"<(?:\w+:)?(?:kml|gpx)\b", data[:4096]):
        return None
    return _XML_NAMED.sub(b"", re.sub(rb"<!--.*?-->", b"", data, flags=re.S))


_FILE_STRIPPERS = {".obj": _strip_obj, ".off": _strip_off, ".ply": _strip_ply, ".stl": _strip_stl,
                   ".mol": _strip_mol, ".sdf": _strip_sdf, ".smi": _strip_smi, ".geojson": _strip_geojson,
                   ".kml": _strip_xml, ".gpx": _strip_xml}


# ----------------------------------------------------------------------------- table files a web search names
_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_INTEGER = re.compile(r"[+-]?\d+")
_NO_LICENCE = frozenset(("", "unknown", "none", "null", "n/a", "na", "not specified", "unspecified", "not stated",
                         "not available", "unclear", "not found", "no license", "no licence", "see source"))
_NOT_IDS = frozenset(("n/a", "na/na", "none/none", "owner/name", "unknown/unknown", "not/applicable"))


def _hub_id_of(text: object) -> str | None:
    """The Hugging Face dataset id in a plain "owner/name" (or "datasets/owner/name"), or in an https
    huggingface.co/datasets/owner/name[/...] URL; else None."""
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        return None
    text = text.strip()
    if "://" in text:
        try:
            parts = urllib.parse.urlsplit(text)
            host = (parts.hostname or "").lower()
        except ValueError:
            return None
        bits = [b for b in parts.path.split("/") if b]
        if parts.scheme != "https" or host not in ("huggingface.co", "www.huggingface.co", "hf.co") or len(bits) < 3 \
                or bits[0] != "datasets":
            return None
        hub_id = f"{bits[1]}/{bits[2]}"
    else:
        hub_id = text[len("datasets/"):] if text.startswith("datasets/") else text
    if not _HUB_ID.match(hub_id) or {".", ".."} & set(hub_id.split("/")) or hub_id.lower() in _NOT_IDS:
        return None
    return hub_id


def _file_url(url: object) -> tuple[str | None, str | None]:
    """(the https download URL of a table file, None) or (None, why it is not one). A GitHub blob or raw page URL
    becomes its raw.githubusercontent.com file, a Hugging Face blob URL its resolve URL; the file must be named
    like a CSV, TSV, JSON or JSON Lines file, a .zip or .gz, or have no suffix."""
    if not isinstance(url, str) or not url.strip():
        return None, "no URL"
    url = url.strip()
    try:
        url.encode("ascii")
        parts = urllib.parse.urlsplit(url)
        host, port = (parts.hostname or "").lower(), parts.port
    except (UnicodeError, ValueError):
        return None, "not a plain ASCII URL"
    if parts.scheme != "https":
        return None, f"not an https URL ({parts.scheme or 'no scheme'})"
    if parts.username or parts.password or port not in (None, 443):
        return None, "a URL with a user, password or port"
    bits = [b for b in parts.path.split("/") if b]
    if host in ("github.com", "www.github.com"):
        if len(bits) < 5 or bits[2] not in ("blob", "raw"):
            return None, "a GitHub page, not a file (only blob and raw file URLs are read)"
        url = f"https://raw.githubusercontent.com/{bits[0]}/{bits[1]}/{'/'.join(bits[3:])}"
    elif host in ("huggingface.co", "www.huggingface.co", "hf.co"):
        if len(bits) < 6 or bits[0] != "datasets" or bits[3] not in ("blob", "resolve", "raw"):
            return None, "a Hugging Face page, not a dataset file"
        url = f"https://huggingface.co/datasets/{bits[1]}/{bits[2]}/resolve/{'/'.join(bits[4:])}"
    elif host == "zenodo.org":
        if "files" not in bits[:-1]:
            return None, "a Zenodo page, not a file"
        url = urllib.parse.urlunsplit(("https", host, parts.path, parts.query, ""))
    elif host in ("raw.githubusercontent.com", "gist.githubusercontent.com"):
        if len(bits) < 4:
            return None, "not a file"
        url = urllib.parse.urlunsplit(("https", host, parts.path, "", ""))
    else:
        return None, f"{host or 'no host'} is not an allowed host (Hugging Face, GitHub or Zenodo)"
    name = _file_name(url).lower()
    inner = name[:-3] if name.endswith(".gz") else name
    suffix = PurePosixPath(inner).suffix
    if suffix and suffix not in TABLE_SUFFIXES + (".txt", ".zip") and not re.fullmatch(r"\.\d+", suffix):
        return None, f"{name!r} is not a CSV, TSV, JSON or JSON Lines file, or a .zip or .gz of one"
    return url, None


def _file_name(url: str) -> str:
    bits = [b for b in urllib.parse.urlsplit(url).path.split("/") if b]
    if len(bits) >= 2 and bits[-1] == "content":  # zenodo.org/api/records/<id>/files/<name>/content
        bits = bits[:-1]
    return urllib.parse.unquote(bits[-1]) if bits else ""


def _table_display(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    return (parts.hostname or "") + urllib.parse.unquote(parts.path)[:200]


@dataclass
class _Web:
    used: bool
    reason: str | None = None  # why the web search was not used
    origin: str | None = None  # "live", "cache" (research's disk cache) or "memo" (an earlier answer this run)
    hub_ids: list[str] = field(default_factory=list)
    tables: list[TableCandidate] = field(default_factory=list)
    refused: list[dict] = field(default_factory=list)  # [{"url" or "hub_id", "reason"}]

    def summary(self) -> dict:
        return {"used": self.used, "reason": self.reason, "origin": self.origin, "hub_ids": list(self.hub_ids),
                "tables": [t.url for t in self.tables], "refused": [dict(r) for r in self.refused]}


_WEB_MEMO: dict[str, tuple[dict, str]] = {}  # query -> (answer data, origin): a goal searched again costs nothing


def _web_query(goal: str, modality: str) -> str:
    key = re.sub(r"\s+", " ", str(modality).strip().lower())
    what = _WEB_INPUTS.get(_ALIASES.get(key, key)) or f"a {key or 'file'} input"
    return _WEB_QUERY.format(goal=" ".join(str(goal).split())[:300], input=what)


def _web_search(goal: str, modality: str, deadline: float) -> _Web:
    """The web search's Hub ids and table files, or _Web(used=False, reason) when it cannot be had: no Linkup key,
    fake mode, the network, no time, or any failure. Never raises."""
    try:
        query = _web_query(goal, modality)
        memo = _WEB_MEMO.get(query)
        if memo is not None:
            data, origin = memo[0], "memo"
        else:
            left = deadline - _now()
            if left < 3:
                return _Web(False, "no time was left for the web search")
            try:
                found = research.search_structured(query, WEB_SCHEMA, include_domains=list(WEB_DOMAINS),
                                                   max_results=10, timeout=min(WEB_TIMEOUT_S, left))
            except research.ResearchUnavailable as exc:
                return _Web(False, f"the web search is unavailable: {exc}")
            if found.origin == "fake":
                return _Web(False, "fake mode (SENSE_FAKE) has no web search, so only the Hub was searched")
            data, origin = found.data, found.origin
            if len(_WEB_MEMO) >= 32:
                _WEB_MEMO.clear()
            _WEB_MEMO[query] = (data, origin)
        web = _web_items(data, modality)
        web.origin = origin
        return web
    except Exception as exc:  # the web search is optional: any failure leaves the Hub search alone
        return _Web(False, f"the web search failed: {type(exc).__name__}: {str(exc)[:200]}")


def _web_items(data: object, modality: str) -> _Web:
    items = data.get("labelled_datasets") if isinstance(data, dict) else None
    web = _Web(True)
    profile = _profile(modality)
    urls = set()
    for item in (items if isinstance(items, list) else [])[:MAX_WEB_ITEMS]:
        if not isinstance(item, dict):
            continue
        name, hint = _short(item.get("name"), 120), _short(item.get("label_column"), 80)
        given = _short(item.get("download_url"), 2000)
        hub_id = _hub_id_of(_short(item.get("hub_id"), 300)) or _hub_id_of(given)
        if hub_id and hub_id not in web.hub_ids:
            web.hub_ids.append(hub_id)
        if not given or (_hub_id_of(given) and _file_url(given)[0] is None):
            continue  # no file, or a Hugging Face dataset page: the Hub id covers it
        url, problem = _file_url(given)
        if problem is None and profile and profile.media:
            problem = f"a table file gives text inputs only, not {modality} files"
        if problem is not None:
            web.refused.append({"url": given[:300], "reason": problem})
            continue
        if url in urls:
            continue
        urls.add(url)
        licence = _web_licence(item.get("licence"))
        web.tables.append(TableCandidate(url, name or _file_name(url), licence,
                                         "the web search's answer" if licence else None, hint,
                                         _short(item.get("page_url"), 300), given))
    return web


def _short(value: object, limit: int) -> str:
    return " ".join(value.split())[:limit] if isinstance(value, str) else ""


def _web_licence(value: object) -> str | None:
    text = _short(value, 120)
    return None if text.lower().strip(" .") in _NO_LICENCE else text


# ----------------------------------------------------------------------------- reading a table
def _choose_table(table: TableCandidate, goal: str, modality: str, wanted: list[str],
                  deadline: float | None) -> Mapping:
    """Download and parse the table (once), then choose its columns as for a Hub dataset, on MAPPING_ROWS rows
    spread evenly over it. The source's own names (its host, path and archive member, never the web search's title)
    say whether it is about the goal's topic."""
    columns, rows, member, _ = _load_table(table, deadline)
    named = _Named(_table_display(table.url), member, table.label_hint)
    mapping = _map(named, goal, modality, columns, _even(rows, MAPPING_ROWS), _profile(modality), wanted)
    mapping.config, mapping.split = member, ""
    return mapping


def _fetch_table(table: TableCandidate, mapping: Mapping, dest: Path, limits: _Limits,
                 deadline: float) -> FetchReport:
    columns, rows, member, load_notes = _load_table(table, deadline)
    name = _table_display(table.url)
    needed = list(mapping.input_columns) + ([mapping.label_column] if mapping.label_column else []) + (
        list(mapping.label_names or []) if mapping.label_kind == "multihot" else [])
    if mapping.input_kind != "text" or any(column not in columns for column in needed):
        raise ProbesUnavailable(f"the mapping does not fit the table {name}: a table gives text inputs from its own "
                                "columns")
    skipped: Counter = Counter()
    notes = list(load_notes)
    seen: dict[int, tuple[str, list[str]]] = {}
    rows_seen = _take_rows(rows, mapping, seen, set(), skipped)
    report = _write_probes(name, mapping, dest, seen, rows_seen, skipped, notes, limits, deadline)
    table.licence, table.licence_source = _table_licence(table, deadline)
    report.licence, report.licence_source, report.source_url = table.licence, table.licence_source, table.url
    return report


def _load_table(table: TableCandidate, deadline: float | None):
    """(columns, rows as (row number, row, set()), archive member or "", notes), downloaded and parsed once."""
    if table._loaded is None:
        name = _table_display(table.url)
        if not _table_url_ok(table.url):
            raise ProbesUnavailable(f"{table.url[:120]!r} is not an https URL on a Hugging Face, GitHub or Zenodo "
                                    "host")
        left = TABLE_TIMEOUT_S if deadline is None else min(TABLE_TIMEOUT_S, deadline - _now())
        try:
            data = _table_get(table.url, timeout=left, max_bytes=TABLE_MAX_BYTES)
        except _HttpError as exc:
            raise _RequestFailed(exc.reason, f"downloading {name} failed: {exc}") from None
        body, inner, member = _unpack(data, _file_name(table.url), name)
        columns, rows, notes = _parse_table(body, inner, name)
        if member:
            notes.insert(0, f"read {member!r}, the first table file in the zip archive")
        table._loaded = (columns, rows, member, notes)
    return table._loaded


def _unpack(data: bytes, name: str, display: str) -> tuple[bytes, str, str]:
    """(file bytes, file name, zip member or ""): a .zip's first member named like a table file, a .gz unpacked;
    either at most TABLE_MAX_BYTES once unpacked."""
    lower = name.lower()
    if data[:4] == b"PK\x03\x04" or lower.endswith(".zip"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for entry in archive.infolist():
                    member = entry.filename
                    base = PurePosixPath(member).name
                    if entry.is_dir() or member.startswith("__MACOSX/") or base.startswith(".") \
                            or PurePosixPath(base.lower()).suffix not in TABLE_SUFFIXES:
                        continue
                    if entry.file_size > TABLE_MAX_BYTES:
                        raise ProbesUnavailable(f"{display}'s member {member!r} unpacks to {entry.file_size:,} "
                                                f"bytes, over the {TABLE_MAX_BYTES:,} byte cap")
                    with archive.open(entry) as handle:
                        body = handle.read(TABLE_MAX_BYTES + 1)
                    if len(body) > TABLE_MAX_BYTES:
                        raise ProbesUnavailable(f"{display}'s member {member!r} unpacks to over the "
                                                f"{TABLE_MAX_BYTES:,} byte cap")
                    return body, base, member
        except (zipfile.BadZipFile, zipfile.LargeZipFile, NotImplementedError, RuntimeError, EOFError, OSError,
                zlib.error, ValueError) as exc:
            raise ProbesUnavailable(f"{display} is not a readable zip archive: {type(exc).__name__}") from None
        raise ProbesUnavailable(f"{display} holds no CSV, TSV, JSON or JSON Lines file")
    if data[:2] == b"\x1f\x8b" or lower.endswith(".gz"):
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as handle:
                body = handle.read(TABLE_MAX_BYTES + 1)
        except (OSError, EOFError, zlib.error, ValueError) as exc:
            raise ProbesUnavailable(f"{display} is not a readable gzip file: {type(exc).__name__}") from None
        if len(body) > TABLE_MAX_BYTES:
            raise ProbesUnavailable(f"{display} unpacks to over the {TABLE_MAX_BYTES:,} byte cap")
        return body, name[:-3] if lower.endswith(".gz") else name, ""
    return data, name, ""


def _decode(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


class _Sampler:
    """Every step-th row, the step doubling whenever more than cap rows are kept: rows spread evenly over a file of
    any length, without holding all of it."""

    def __init__(self, cap: int):
        self.cap, self.step, self.kept, self.total = cap, 1, [], 0

    def add(self, idx: int, row) -> None:
        self.total += 1
        if idx % self.step:
            return
        self.kept.append((idx, row))
        if len(self.kept) > self.cap:
            self.step *= 2
            self.kept = [(i, r) for i, r in self.kept if i % self.step == 0]


def _row_cap(n_columns: int) -> int:
    return max(1_000, min(MAX_TABLE_ROWS, MAX_TABLE_CELLS // max(1, n_columns)))


def _parse_table(body: bytes, name: str, display: str):
    """(columns, rows, notes) of a CSV or TSV (header row; dialect sniffed), a JSON list of objects or JSON Lines
    file, read with the standard library. At most _row_cap rows are kept, spread evenly over the file; CSV columns
    whose every value is a number become numbers."""
    text = _decode(body)
    suffix = PurePosixPath(name.lower()).suffix
    start = text.lstrip()[:1]
    if start == "<":
        raise ProbesUnavailable(f"{display} is an HTML or XML page, not a data file")
    notes: list[str] = []
    if suffix in (".jsonl", ".ndjson") or (suffix != ".json" and start == "{"):
        names, sampler, bad = _json_rows(_json_lines(text), display)
        kind = "JSON Lines"
    elif suffix == ".json" or start == "[":
        records = _json_list(text, display)
        if records is None:  # a .json file holding one object per line
            names, sampler, bad = _json_rows(_json_lines(text), display)
            kind = "JSON Lines"
        else:
            names, sampler, bad = _json_rows(records, display)
            kind = "JSON"
    else:
        names, sampler, bad = _csv_rows(text, suffix, display, notes)
        kind = "TSV" if suffix in (".tsv", ".tab") else "CSV"
    if kind in ("CSV", "TSV"):
        rows = [(idx, dict(zip(names, (f.strip() or None for f in fields))), set()) for idx, fields in sampler.kept]
    else:
        rows = [(idx, {k: record[k] for k in names if k in record}, set()) for idx, record in sampler.kept]
    if not rows:
        raise ProbesUnavailable(f"{display} holds no data rows")
    if kind in ("CSV", "TSV"):
        _numbers(names, rows)
    columns = {n: _table_column(n, [row.get(n) for _, row, _ in rows]) for n in names}
    notes.insert(0, f"read {sampler.total:,} rows of {display} ({kind}, {len(names)} columns)"
                 + (f"; kept {len(rows):,} spread evenly over them" if len(rows) < sampler.total else "")
                 + (f"; skipped {bad:,} malformed rows" if bad else ""))
    return columns, rows, notes


def _dialect(sample: str, suffix: str):
    lines = sample.splitlines()
    probe = "\n".join((lines[:-1] if len(lines) > 2 else lines)[:50])  # the last line may be cut
    options = [csv.excel_tab] if suffix in (".tsv", ".tab") else []
    try:
        options.append(csv.Sniffer().sniff(probe, delimiters=",\t;|"))
    except csv.Error:
        pass
    options += [csv.excel, csv.excel_tab]
    for dialect in options:
        try:
            rows = [r for r in csv.reader(io.StringIO(probe, newline=""), dialect) if r]
        except csv.Error:
            continue
        if rows and len(rows[0]) >= 2 and sum(len(r) == len(rows[0]) for r in rows) >= 0.8 * len(rows):
            return dialect
    return options[0]


def _column_names(header: list) -> list[str]:
    names = []
    for i, raw in enumerate(header):
        name = " ".join(str(raw).split())[:100] or f"column_{i}"
        while name in names:
            name += "_2"
        names.append(name)
    return names


def _csv_rows(text: str, suffix: str, display: str, notes: list):
    reader = csv.reader(io.StringIO(text, newline=""), _dialect(text[:65_536], suffix))
    try:
        header = next(reader)
    except StopIteration:
        raise ProbesUnavailable(f"{display} is empty") from None
    except csv.Error as exc:
        raise ProbesUnavailable(f"{display} could not be read as CSV: {exc}") from None
    names = _column_names(header)[:MAX_TABLE_COLUMNS]
    if len(names) < 2:
        raise ProbesUnavailable(f"{display} has fewer than 2 columns")
    sampler, bad, idx = _Sampler(_row_cap(len(names))), 0, 0
    try:
        for fields in reader:
            if not any(f.strip() for f in fields):
                continue
            if len(fields) > len(header) and any(f.strip() for f in fields[len(header):]):
                bad += 1
                continue
            sampler.add(idx, (fields + [""] * len(names))[:len(names)])
            idx += 1
    except csv.Error as exc:
        notes.append(f"stopped reading at data row {idx}: {exc}")
    return names, sampler, bad


def _json_list(text: str, display: str) -> list | None:
    """The objects of a JSON list (or of the one list of objects in a JSON object); None when the text is not one
    JSON value (it may be JSON Lines)."""
    try:
        value = json.loads(text)
    except ValueError:
        return None
    if isinstance(value, dict):
        lists = [v for v in value.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v[:50])]
        value = lists[0] if len(lists) == 1 else None
    if not isinstance(value, list):
        raise ProbesUnavailable(f"{display} is JSON but not a list of objects")
    return value


def _json_lines(text: str):
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except ValueError:
            yield None


def _json_rows(records, display: str):
    names: list[str] = []
    known: set[str] = set()
    sampler, bad, idx = None, 0, 0
    for record in records:
        if not isinstance(record, dict):
            bad += 1
            continue
        for key in record:
            if key not in known and len(names) < MAX_TABLE_COLUMNS and isinstance(key, str):
                known.add(key)
                names.append(key)
        if sampler is None:
            sampler = _Sampler(_row_cap(len(record)))
        sampler.add(idx, record)
        idx += 1
    if sampler is None:
        raise ProbesUnavailable(f"{display} holds no JSON objects")
    return names, sampler, bad


def _numbers(names: list[str], rows) -> None:
    """Turn each CSV column whose every value is a number into ints or floats (so 0/1 blocks and coordinates read as
    numbers)."""
    for name in names:
        values = [row[name] for _, row, _ in rows if row.get(name) is not None]
        if values and all(_NUMBER.fullmatch(v) for v in values):
            for _, row, _ in rows:
                value = row.get(name)
                if value is not None:
                    row[name] = int(value) if _INTEGER.fullmatch(value) else float(value)


def _table_column(name: str, values: list) -> _Column:
    present = [v for v in values if v is not None and v != "" and v != []]
    if not present:
        return _Column(name, "other")
    if all(isinstance(v, (int, float)) for v in present):
        return _Column(name, "number")
    if all(isinstance(v, str) for v in present):
        return _Column(name, "string")
    if all(isinstance(v, list) and all(isinstance(x, str) for x in v) for v in present):
        return _Column(name, "list")
    return _Column(name, "other")


def _even(items: list, k: int) -> list:
    return list(items) if len(items) <= k else [items[int(i * len(items) / k)] for i in range(k)]


# ----------------------------------------------------------------------------- a table's licence
def _table_licence(table: TableCandidate, deadline: float) -> tuple[str, str]:
    """(licence, where it was read). The web search's answer when it gave one; else the first line of the
    repository's LICENSE file (GitHub), the record's licence (Zenodo) or the dataset card's licence (Hugging Face);
    else "unknown". Never invented."""
    if table.licence:
        return table.licence, table.licence_source or "the web search's answer"
    parts = urllib.parse.urlsplit(table.url)
    host, bits = (parts.hostname or "").lower(), [b for b in parts.path.split("/") if b]
    if host == "raw.githubusercontent.com" and len(bits) >= 4:
        ref = "/".join(bits[2:5]) if bits[2] == "refs" and len(bits) >= 6 else bits[2]
        base = f"https://raw.githubusercontent.com/{bits[0]}/{bits[1]}/{ref}/"
        for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE", "COPYING"):
            data = _licence_get(base + name, deadline, LICENCE_MAX_BYTES)
            line = _first_line(data) if data else ""
            if line:
                return line, f"the first line of {base + name}"
        return "unknown", f"no LICENSE file could be read in github.com/{bits[0]}/{bits[1]} at {ref}"
    if host == "zenodo.org":
        record = next((bits[i + 1] for i, b in enumerate(bits[:-1]) if b in ("records", "record")), None)
        if record and record.isdigit():
            data = _licence_get(f"https://zenodo.org/api/records/{record}", deadline, 5_000_000)
            licence = _zenodo_licence(data)
            if licence:
                return licence, f"the Zenodo record {record}"
        return "unknown", "the Zenodo record's licence could not be read"
    hub_id = _hub_id_of(table.url) if _allowed_url(table.url) else None
    if hub_id:
        try:
            record = _get_json(f"{HUB_SEARCH}/{hub_id}", {}, what=f"the Hub record of {hub_id}", deadline=deadline,
                               retry=False)
        except ProbesUnavailable:
            record = None
        found = _candidate(record) if isinstance(record, dict) else None
        if found and found.licence:
            return found.licence, f"the Hub dataset card of {hub_id}"
        return "unknown", f"the Hub dataset card of {hub_id} states no licence that could be read"
    return "unknown", "the source states no licence that could be read"


def _licence_get(url: str, deadline: float, max_bytes: int) -> bytes | None:
    left = min(LICENCE_TIMEOUT_S, deadline - _now())
    if left < 0.5:
        return None
    try:
        return _table_get(url, timeout=left, max_bytes=max_bytes)
    except _HttpError:
        return None


def _first_line(data: bytes) -> str:
    for line in _decode(data).splitlines():
        line = " ".join(line.strip().lstrip("#").split())
        if re.search(r"[A-Za-z0-9]", line):
            return line[:200]
    return ""


def _zenodo_licence(data: bytes | None) -> str | None:
    try:
        record = json.loads(data) if data else None
    except ValueError:
        return None
    licence = record.get("metadata", {}).get("license") if isinstance(record, dict) else None
    if isinstance(licence, dict):
        licence = licence.get("id") or licence.get("title")
    return _short(licence, 120) or None if isinstance(licence, str) else None


# ----------------------------------------------------------------------------- folder-labelled repositories
# Folders that never name a label: splits, generic containers and the input's own kind.
_GENERIC_FOLDERS = frozenset("""
train training trainset train_set train_data test testing testset test_set test_data val valid validation validate
valset val_set dev devset dev_set eval evaluation holdout heldout held_out trainval train_val train_test
data dataset datasets files file raw all full split splits subset subsets sample samples example examples
images image imgs img pictures photos videos video clips clip audio audios sounds sound wav wavs meshes mesh models
model shapes objects obj off ply stl glb gltf molecules mol sdf geojson kml gpx frames recordings media assets
content contents resources downloads download extracted unzipped original originals processed source sources src
output outputs input inputs public main master default
""".split())
_GENERIC_NUMBERED = re.compile(r"(?:train|training|test|testing|val|valid|validation|dev|eval|fold|part|split|batch|"
                               r"chunk|shard|subset|set|seq|session|run|version|v)[\W_]*\d+[a-z]?")
_NUMBERED_ITEM = re.compile(r"(?:class|label|category|cls|id|item|sample|group|subject|speaker|user)[\W_]*\d+")


def _generic_folder(name: str) -> bool:
    key = re.sub(r"[\s\-]+", "_", name.strip().lower())
    return key in _GENERIC_FOLDERS or key.isdigit() or bool(_GENERIC_NUMBERED.fullmatch(key))


def _id_like(name: str) -> bool:
    """A folder name that is an id, not a label: mostly digits (n01440764, 2019), a hash or UUID, or a numbered
    item (chair_0001, sample-07, class3)."""
    key = name.strip().lower()
    alnum = re.sub(r"[^a-z0-9]", "", key)
    digits = sum(c.isdigit() for c in alnum)
    return (not alnum or 2 * digits >= len(alnum)
            or (len(alnum) >= 8 and digits > 0 and re.fullmatch(r"[0-9a-f]+", alnum) is not None)
            or re.search(r"(?:[\W_]\d{2,}|\d{3,})$", key) is not None or _NUMBERED_ITEM.fullmatch(key) is not None)


def _folder_label(parts: list[str], ids: Counter) -> tuple[int, str] | None:
    """(depth, name) of the nearest folder above a file (parts: its path's parts) that can name a label: not a
    split or generic name and not an id (counted in ids); None when there is none."""
    for depth in range(len(parts) - 2, -1, -1):
        name = parts[depth]
        if _generic_folder(name):
            continue
        if _id_like(name):
            ids[name] += 1
            continue
        return depth, name
    return None


def _bad_path(path: str) -> bool:
    parts = path.split("/")
    return "\\" in path or any(p in ("", ".", "..", "__MACOSX") or p.startswith(".") for p in parts)


@dataclass
class _Layout:
    matching: list[tuple[str, int]]  # the files of the input's kind, as (path, size), sorted by path
    found: dict[int, tuple[int, str]]  # index in matching -> (depth, folder name) of the files with a label folder
    skipped: Counter
    ids: Counter  # id-like folder names passed over -> files


def _folder_labels(files: list[tuple[str, int]], exts: tuple[str, ...]) -> _Layout:
    matching = [(p, s) for p, s in files if PurePosixPath(p).suffix.lower() in exts]
    layout = _Layout(matching, {}, Counter(), Counter())
    for idx, (path, _) in enumerate(matching):
        if _bad_path(path):
            layout.skipped["bad_path"] += 1
            continue
        hit = _folder_label(path.split("/"), layout.ids)
        if hit is None:
            layout.skipped["no_label_folder"] += 1
        else:
            layout.found[idx] = hit
    return layout


def _folder_depth(name: str, layout: _Layout, exts: tuple[str, ...], min_files: int) -> tuple[int, Counter]:
    """The folder depth that labels the repository's files, and its folder names' file counts: the depth most of
    the files' nearest label folders share (a tie goes to the deeper), whose names give at least 2 labels with
    min_files files each; else ProbesUnavailable saying why."""
    kinds = "/".join(exts)
    if not layout.matching:
        raise ProbesUnavailable(f"{name} holds no {kinds} files")
    if not layout.found:
        raise ProbesUnavailable(
            f"none of {name}'s {len(layout.matching):,} {kinds} files sits in a folder that can name a label"
            + (f": the folder names look like ids (e.g. {', '.join(n for n, _ in layout.ids.most_common(4))})"
               if layout.ids else " (they sit in the top folder or in split or generic folders only)"))
    depths = Counter(depth for depth, _ in layout.found.values())
    depth, count = max(depths.items(), key=lambda item: (item[1], item[0]))
    if 2 * count <= len(layout.found):
        raise ProbesUnavailable(f"{name}'s files sit at no one folder depth: files per depth "
                                f"{dict(sorted(depths.items()))}, so its folders do not label them consistently")
    names = Counter(folder for d, folder in layout.found.values() if d == depth)
    if sum(1 for n in names.values() if n >= min_files) < 2:
        top = ", ".join(f"{folder} ({n})" for folder, n in names.most_common(6))
        raise ProbesUnavailable(f"fewer than 2 of {name}'s label folders hold {min_files} {kinds} files (files "
                                f"per folder at depth {depth}: {top})")
    return depth, names


def _resolve_url(hub_id: str, revision: str, path: str) -> str:
    return (f"https://huggingface.co/datasets/{hub_id}/resolve/{urllib.parse.quote(revision, safe='')}/"
            f"{urllib.parse.quote(path, safe='/')}")


_LINK_NEXT = re.compile(r'<([^>]*)>[^,<]*;\s*rel="?next"?', re.I)


def _next_link(header: str | None) -> str | None:
    """The rel="next" URL of a Link header, or None."""
    match = _LINK_NEXT.search(header) if isinstance(header, str) else None
    return match.group(1).strip() if match else None


def _tree_page(url: str, *, what: str, deadline: float) -> tuple[object, str | None]:
    def request(seconds):
        body, link = _http_get_page(url, timeout=seconds, max_bytes=JSON_MAX_BYTES)
        return json.loads(body), _next_link(link)

    return _attempts(request, what=what, timeout=TIMEOUT_S, deadline=deadline, retry=True)


def _list_folder(folder: FolderCandidate, deadline: float | None) -> tuple[list[tuple[str, int]], list[str]]:
    """The repository's files as (path, size), sorted by path, and notes: from the Hub tree API at its revision
    (pinned to a commit first when the candidate has none), paged through the Link header; at most
    TREE_MAX_PAGES pages within TREE_BUDGET_S. Listed once per candidate."""
    if folder._files is not None:
        return folder._files
    stop = _now() + TREE_BUDGET_S if deadline is None else min(deadline, _now() + TREE_BUDGET_S)
    notes: list[str] = []
    if not folder.revision:
        folder.revision = _pin_revision(folder.hub_id, stop)
        if folder.revision == "main":
            notes.append(f"{folder.hub_id}'s commit could not be read, so its files were listed and downloaded at "
                         "the main branch")
    url = f"{HUB_SEARCH}/{folder.hub_id}/tree/{urllib.parse.quote(folder.revision, safe='')}?recursive=true"
    path = urllib.parse.urlsplit(url).path
    what = f"the Hub's file list of {folder.hub_id}"
    files: list[tuple[str, int]] = []
    pages = 0
    while url:
        if pages >= TREE_MAX_PAGES or (pages and stop - _now() < 2):
            notes.append(f"listed only the first {pages} pages ({len(files):,} files) of {folder.hub_id}'s file list "
                         f"({'the page cap' if pages >= TREE_MAX_PAGES else 'the time budget'}), so its labels and "
                         "counts come from those")
            break
        try:
            answer, link = _tree_page(url, what=what, deadline=stop)
        except ProbesUnavailable as exc:
            if not pages:
                raise
            notes.append(f"the file list stopped after {pages} pages ({len(files):,} files): {exc}")
            break
        if not isinstance(answer, list):
            raise ProbesUnavailable(f"{what} did not answer a list")
        pages += 1
        for entry in answer:
            if isinstance(entry, dict) and entry.get("type") == "file" and isinstance(entry.get("path"), str):
                lfs = entry.get("lfs") if isinstance(entry.get("lfs"), dict) else {}
                files.append((entry["path"], _int(lfs.get("size")) or _int(entry.get("size"))))
        url = urllib.parse.urljoin(url, link) if link else None
        if url and (urllib.parse.urlsplit(url).hostname != "huggingface.co" or not _allowed_url(url)
                    or urllib.parse.urlsplit(url).path != path):
            notes.append(f"the file list's next page pointed elsewhere ({_host(url)}), so listing stopped after "
                         f"{pages} pages")
            break
    files.sort()
    folder._files = (files, notes)
    return folder._files


def _pin_revision(hub_id: str, deadline: float) -> str:
    """The repository's current commit from its Hub record, else "main"."""
    try:
        record = _get_json(f"{HUB_SEARCH}/{hub_id}", {}, what=f"the Hub record of {hub_id}", deadline=deadline,
                           retry=False)
    except ProbesUnavailable:
        return "main"
    sha = record.get("sha") if isinstance(record, dict) else None
    return sha if isinstance(sha, str) and _SHA.fullmatch(sha) else "main"


def _choose_folder(folder: FolderCandidate, goal: str, modality: str, wanted: list[str], deadline: float | None,
                   min_per_label: int) -> Mapping:
    """The folder kind's Mapping: the folder depth that labels the repository's files of the input's kind, trusted
    only when the repository's id or title names the goal's topic (as for a generic label column) or the folder
    names name the goal's words or the wanted labels."""
    profile = _profile(modality)
    kind = profile.folder if profile else None
    if kind is None:
        raise ProbesUnavailable(f"no file types are known for {modality!r}, so a folder-labelled repository cannot "
                                "be read for it")
    files, _ = _list_folder(folder, deadline)
    exts = _FOLDER_EXT[kind]
    layout = _folder_labels(files, exts)
    depth, names = _folder_depth(folder.hub_id, layout, exts, min_per_label)
    labels = [name for name, _ in names.most_common()]
    topic = _topic_words(goal, modality, profile)
    goal_words = [w for w in _topic_weights(goal, modality, profile) if w in _content_words(goal)]
    if not (_names_topic(_Named(folder.hub_id, folder.title), topic)
            or _values_name_goal(_Column("folders", "classlabel", names=labels), [], goal_words, wanted)):
        raise ProbesUnavailable(
            f"{folder.hub_id}'s {kind} files sit in folders named {', '.join(labels[:6])}"
            f"{', ...' if len(labels) > 6 else ''}, but its name does not mention the goal's topic "
            f"({'/'.join(sorted(topic)[:8]) or 'the goal'}), so its folder names are not trusted as labels")
    at_depth = [layout.matching[i][0] for i, (d, _) in sorted(layout.found.items()) if d == depth]
    ext = Counter(PurePosixPath(p).suffix.lower() for p in at_depth).most_common(1)[0][0]
    reason = (f"rule: {len(at_depth):,} {'/'.join(sorted({PurePosixPath(p).suffix.lower() for p in at_depth}))} "
              f"files labelled by the name of folder {depth + 1} of their path (e.g. {at_depth[0][:120]}), "
              f"{len(labels)} labels")
    return Mapping(folder.revision or "main", "", [], "media", "", ext, "", "folder", labels, None, kind, "rule",
                   reason, depth)


def _fetch_folder(folder: FolderCandidate, mapping: Mapping, dest: Path, limits: _Limits,
                  deadline: float) -> FetchReport:
    """The files at the mapping's folder depth, downloaded through the repository's resolve URLs at its revision
    into dest/<label>/<n><extension>, through the same label choice, caps, metadata stripping and leak scan as a
    viewer dataset's media."""
    name = folder.hub_id
    if mapping.label_kind != "folder" or mapping.folder_depth is None or mapping.media_type not in _FOLDER_EXT:
        raise ProbesUnavailable(f"the mapping does not fit the folder-labelled repository {name}")
    files, list_notes = _list_folder(folder, deadline)
    revision = folder.revision or "main"
    layout = _folder_labels(files, _FOLDER_EXT[mapping.media_type])
    skipped: Counter = Counter()
    seen: dict[int, tuple[str, list[str]]] = {}
    for idx, (depth, label) in layout.found.items():
        path, size = layout.matching[idx]
        if depth != mapping.folder_depth:
            skipped["other_folder_depth"] += 1
        elif size > FILE_MAX_BYTES:  # the file list gives each file's size: no request for one over the cap
            skipped["too_large"] += 1
        else:
            seen[idx] = (_resolve_url(name, revision, path), [label])
    skipped.update(layout.skipped)
    notes = list(list_notes) + [
        f"labels are the names of the folders the files sit in (folder {mapping.folder_depth + 1} of each path), "
        f"read from {name}'s file list at revision {revision}; files are written as <n><extension>, never under "
        "their own names"]
    notes += _card_notes(folder, revision, deadline)
    report = _write_probes(name, mapping, dest, seen, len(layout.matching), skipped, notes, limits, deadline,
                           seed=f"{name}@{revision}")
    report.licence, report.hub_id = folder.licence, name
    report.source_url = f"https://huggingface.co/datasets/{name}/tree/{urllib.parse.quote(revision, safe='')}"
    report.licence_source = "the Hub dataset card" if folder.licence else None
    return report


_TERMS = re.compile(r"licen[cs]|terms of use|terms and conditions|academic|non-?commercial|research (?:use|purpose|"
                    r"only)|copyright|all rights reserved", re.I)
_LINK = re.compile(r"https?://([A-Za-z0-9.-]+)")


def _card_notes(folder: FolderCandidate, revision: str, deadline: float) -> list[str]:
    """What the dataset card says about the files' terms, quoted, never invented: the licence it states (or that it
    states none), other sites it links to (a re-hosted dataset's own source, whose terms the card's licence may not
    cover) and up to 3 of its lines that mention a licence, terms, academic or non-commercial use or copyright."""
    if folder.licence:
        note = f"licence {folder.licence!r} is what the Hub dataset card or its licence tag states for this repository"
    else:
        note = "the Hub dataset card states no licence, so the files' terms are unknown"
    left = min(LICENCE_TIMEOUT_S, deadline - _now())
    try:
        text = _decode(_http_get(_resolve_url(folder.hub_id, revision, "README.md"), timeout=left,
                                 max_bytes=LICENCE_MAX_BYTES)) if left >= 0.5 else None
    except _HttpError:
        text = None
    if text is None:
        return [note + "; the card's text could not be read"]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    body = re.sub(r"\A\s*---\n.*?\n---[ \t]*(?:\n|\Z)", "", text, count=1, flags=re.S)  # the YAML front matter
    hosts = list(dict.fromkeys(h.lower().rstrip(".") for h in _LINK.findall(body)
                               if not re.search(r"(?:^|\.)(?:huggingface\.co|hf\.co)$", h.lower().rstrip("."))))
    if hosts:
        note += (f"; the card links to other sites ({', '.join(hosts[:3])}): if the files come from there, that "
                 "source's own terms may apply, which this licence may not cover")
    lines = body.splitlines()
    quoted = []
    for i, line in enumerate(lines):
        if len(quoted) < 3 and _TERMS.search(line):
            if line.lstrip().startswith("#"):  # a heading such as "## License": quote its section's first line
                text = next((s.strip() for s in lines[i + 1:i + 6] if s.strip()), "")
                if not text or text.startswith("#"):
                    continue
                line = f"{line.strip().lstrip('#').strip()}: {text}"
            quoted.append(" ".join(line.split())[:200])
    return [note] + [f"the dataset card says: {line!r}" for line in quoted]


# ----------------------------------------------------------------------------- the whole step
class _Exclusions:
    """exclude_sources: Hub ids (also as owner/name@revision), dataset pages, file URLs (GitHub blob or raw alike),
    folders or repositories. A
    candidate is excluded when its id or URL is one of them, when it is a file in the same folder as an excluded file
    (another file of the same dataset), or when its URL lies under an excluded URL (a repository or folder)."""

    def __init__(self, values):
        self.keys, self.folders, self.prefixes = set(), set(), set()
        for value in values or ():
            if not isinstance(value, str) or not value.strip():
                continue
            value = value.strip().rstrip("/")
            self.keys.add(value.lower())
            pinned = re.fullmatch(r"([\w.-]+/[\w.-]+)@[\w./-]+", value)  # owner/name@revision
            hub_id = _hub_id_of(pinned.group(1) if pinned else value)
            if hub_id:
                self.keys.add(hub_id.lower())
            url, _ = _file_url(value)
            if url:
                self.keys.add(url.lower())
                self.folders.add(url.lower().rsplit("/", 1)[0])
            if "://" in value:
                self.prefixes.add(value.lower() + "/")
                bits = [b for b in urllib.parse.urlsplit(value).path.split("/") if b]
                if (urllib.parse.urlsplit(value).hostname or "").lower() in ("github.com", "www.github.com") \
                        and len(bits) >= 2:
                    self.prefixes.add(f"https://raw.githubusercontent.com/{bits[0]}/{bits[1]}/".lower())

    def why(self, candidate) -> str | None:
        if isinstance(candidate, TableCandidate):
            url = candidate.url.lower()
            keys = {url, candidate.given_url.strip().rstrip("/").lower()}
            hub_id = _hub_id_of(candidate.url)
            keys |= {hub_id.lower()} if hub_id else set()
            if keys & self.keys:
                return "in exclude_sources"
            if url.rsplit("/", 1)[0] in self.folders:
                return "a file in the same folder as an excluded file"
            if any(url.startswith(prefix) for prefix in self.prefixes):
                return "under an excluded URL"
            return None
        return "in exclude_sources" if {candidate.hub_id.lower(), candidate.url.lower()} & self.keys else None


def _as_folder(candidate: HubCandidate, modality: str) -> FolderCandidate | None:
    """The candidate as a folder-labelled repository, when files of the modality's kind are known."""
    profile = _profile(modality)
    return FolderCandidate.of(candidate) if profile and profile.folder else None


def _from_web(candidate) -> bool:
    return isinstance(candidate, TableCandidate) or candidate.found_by != "hub"


def _window(ranked: list, n: int) -> list:
    """The first n candidates in rank order, but with up to n // 2 of those a web search named among them."""
    web = [c for c in ranked if _from_web(c)][:n // 2]
    keep = {id(c) for c in web}
    for candidate in ranked:
        if len(keep) >= n:
            break
        keep.add(id(candidate))
    return [c for c in ranked if id(c) in keep]


def _folder_for(candidate) -> str:
    if isinstance(candidate, TableCandidate):
        stem = PurePosixPath(_file_name(candidate.url)).name.split(".")[0]
        return f"web__{safe_label(stem, 0)}__{hashlib.sha256(candidate.url.encode()).hexdigest()[:8]}"
    return candidate.hub_id.replace("/", "__")


def _tried(candidate, reason: str) -> dict:
    if isinstance(candidate, TableCandidate):
        return {"hub_id": "", "url": candidate.url, "reason": reason}
    return {"hub_id": candidate.hub_id, "reason": reason}


def _web_note(web: _Web) -> str:
    if not web.used:
        return f"; the web search was not used ({web.reason})"
    if not (web.hub_ids or web.tables):
        return "; the web search named no usable source"
    return ""


def _add_web_hub_ids(web: _Web, found: dict, hits: Counter, exclusions: _Exclusions, excluded: list,
                     deadline: float) -> None:
    """The Hub ids the web search named join the Hub candidates, from their Hub records (so gated, private and
    viewer-less datasets are dropped as in the search)."""
    known = {hub_id.lower(): hub_id for hub_id in found}
    for hub_id in web.hub_ids:
        if exclusions.keys & {hub_id.lower(), f"https://huggingface.co/datasets/{hub_id}".lower()}:
            excluded.append({"source": hub_id, "reason": "in exclude_sources"})
            continue
        if hub_id.lower() in known:
            same = found[known[hub_id.lower()]]
            same.found_by = "both" if same.found_by == "hub" else same.found_by
            hits[same.hub_id] += 1
            continue
        if deadline - _now() < 5:
            web.refused.append({"hub_id": hub_id, "reason": "no time was left to read its Hub record"})
            continue
        try:
            record = _get_json(f"{HUB_SEARCH}/{hub_id}", {}, what=f"the Hub record of {hub_id}", deadline=deadline)
        except ProbesUnavailable as exc:
            web.refused.append({"hub_id": hub_id, "reason": str(exc)})
            continue
        candidate = _candidate(record)
        if candidate is None:
            web.refused.append({"hub_id": hub_id, "reason": "gated, private, disabled, or its card turns the "
                                                            "viewer off"})
            continue
        candidate.found_by = "web"
        found[candidate.hub_id] = candidate
        known[candidate.hub_id.lower()] = candidate.hub_id
        hits[candidate.hub_id] += 1


# Words that name a file format or a processing step, never a category ("remesh", "dora sharp point ply" were folder
# names a real run took as labels). "off" is left out: it is also an action word ("take off").
_FORMAT_WORDS = frozenset("""
ply obj stl glb gltf fbx dae usd usdz mp4 avi mov webm mkv wav mp3 flac ogg png jpg jpeg tif tiff bmp smi sdf mol csv tsv
json txt npz npy h5 hdf5 parquet pkl zip remesh remeshed mesh meshes pointcloud pointclouds voxel voxels simplified
decimated normalized normalised aligned watertight manifold processed preprocessed augmented resampled sharp
""".split())
MIN_WEB_LABELS = 3  # a web probe set needs this many labels: two make a coin toss of every question
_FIT_SCHEMA = {
    "type": "object",
    "properties": {"answers_goal": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["answers_goal", "reason"], "additionalProperties": False,
}
_FIT_INSTRUCTION = (
    "A user wants a model to answer this goal about their files. A dataset's labels are listed. Say whether the "
    "labels are answers to the goal (categories of the kind the goal asks for, such as object names for 'what "
    "object is this'), not file formats, processing steps, ids or unrelated categories. Judge only the labels; "
    "never suggest labels."
)


def _probe_set_problem(report: FetchReport, goal: str, modality: str, min_per_label: int) -> str | None:
    """Why a fetched probe set is not a fair test, or None. Also keeps one file type: when the labels' files differ
    in extension, an extension could give the label away (probe copies keep it), so only the most common extension
    that every label has is kept (FetchReport.counts updated), and the set is refused when none is shared."""
    folder = Path(report.folder)
    by_label = {label: [p for p in (folder / label).iterdir() if p.is_file()] for label in report.counts
                if (folder / label).is_dir()}
    exts = {label: {p.suffix.lower() for p in files} for label, files in by_label.items()}
    shared = set.intersection(*exts.values()) if exts else set()
    if any(e != shared for e in exts.values()):
        if not shared:
            return "the labels' files are of different types, so a file type would give the label away"
        keep = max(shared, key=lambda e: sum(p.suffix.lower() == e for files in by_label.values() for p in files))
        for label, files in by_label.items():
            for p in files:
                if p.suffix.lower() != keep:
                    p.unlink()
            report.counts[label] = sum(1 for p in (folder / label).iterdir() if p.is_file())
        report.notes.append(f"only {keep} files were kept, so that no file type gives a label away")
        if min(report.counts.values()) < min_per_label:
            return (f"after keeping only {keep} files (one type for every label, so the type gives nothing away), "
                    f"a label has fewer than {min_per_label} files")
    labels = [report.label_values.get(name, name) for name in report.labels]
    junk = [label for label in labels if set(re.findall(r"[a-z]+", str(label).lower())) & _FORMAT_WORDS]
    if junk:
        return f"labels {junk[:4]} name file formats or processing steps, not categories"
    if len(labels) < MIN_WEB_LABELS:
        return f"only {len(labels)} labels; a fair test needs at least {MIN_WEB_LABELS}"
    answer = llm.ask_json(_FIT_INSTRUCTION, {"goal": goal, "modality": modality, "labels": labels[:16]}, _FIT_SCHEMA)
    if isinstance(answer, dict) and answer.get("answers_goal") is False:
        return f"the labels {labels[:6]} do not answer the goal (the agent's model: {str(answer.get('reason'))[:200]})"
    return None


def find_probes(goal: str, modality: str, dest_root: Path, *, wanted_labels: list[str] | None = None,
                emit: Callable[[dict], None] | None = None, max_candidates: int = 3, max_inspect: int = 12,
                total_budget_s: float | None = None, exclude_sources: list[str] | tuple = (),
                disjoint_from: list[str] | tuple = (), **fetch_kwargs) -> tuple[Path, dict]:
    say = emit or (lambda event: None)
    dest_root = Path(dest_root)
    fetch_budget = float(fetch_kwargs.pop("budget_s", 120))
    total = INSPECT_ALLOWANCE_S + max_candidates * fetch_budget if total_budget_s is None else float(total_budget_s)
    deadline = _now() + total
    wanted = list(wanted_labels or [])
    disjoint = _disjoint(disjoint_from, wanted)
    exclusions = _Exclusions(exclude_sources)
    say({"action": "search", "status": "start", "goal": str(goal)[:300], "modality": str(modality)})
    pool = ThreadPoolExecutor(1)
    try:  # the web search runs beside the Hub search
        web_job = pool.submit(_web_search, goal, modality, deadline)
        hub_error = None
        try:
            found, hits, errors, queries, origin = _hub_query(goal, modality, 20, deadline)
            if errors and len(errors) == len(queries):
                hub_error = ProbesUnavailable(f"the Hub search failed for every query: {errors[0]}")
        except ProbesUnavailable as exc:
            hub_error, found, hits, queries, origin = exc, {}, Counter(), [], "rule"
        try:
            web = web_job.result(timeout=max(1.0, min(WEB_TIMEOUT_S + 10, deadline - _now())))
        except _FutureTimeout:
            web = _Web(False, "the web search did not answer within the time budget")
    finally:
        pool.shutdown(wait=False)
    excluded: list[dict] = []
    _add_web_hub_ids(web, found, hits, exclusions, excluded, deadline)
    for table in web.tables:
        hits[table.url] += 1
    candidates = []
    for candidate in list(found.values()) + list(web.tables):
        why = exclusions.why(candidate)
        if why:
            excluded.append({"source": _source_of(candidate), "reason": why})
        else:
            candidates.append(candidate)
    ranked = [c for i, c in enumerate(_ranked(candidates, hits, queries, modality)) if i < 20 or _from_web(c)]
    window = _window(ranked, max_inspect)
    if not ranked and hub_error is not None:
        reason = str(hub_error) + _web_note(web)
        say({"action": "search", "status": "fail", "reason": reason})
        raise ProbesUnavailable(reason)
    web_info = web.summary()
    say({"action": "search", "status": "ok", "queries": queries, "origin": origin, "found": len(ranked),
         "top": [_source_of(c) for c in window], "web": web_info, "excluded": excluded})
    if not ranked:
        raise ProbesUnavailable(f"the Hub search found no open dataset for queries {queries}" + _web_note(web)
                                + (f"; {len(excluded)} other {'match was' if len(excluded) == 1 else 'matches were'} "
                                   f"left out because the run had already used {'it' if len(excluded) == 1 else 'them'}"
                                   if excluded else ""))
    tried, fetched, out_of_time = [], 0, False
    min_files = fetch_kwargs.get("min_per_label", 6)
    for candidate in window:
        if fetched >= max_candidates:
            break
        if deadline - _now() < 5:
            out_of_time = True
            break
        is_table = isinstance(candidate, TableCandidate)
        source = _source_of(candidate)
        tag = {"hub_id": "" if is_table else candidate.hub_id, "source": source, "kind": "table" if is_table else "hub"}
        about = {"title": candidate.title} if is_table else {"downloads": candidate.downloads,
                                                             "found_by": candidate.found_by}
        say({"action": "candidate", "status": "start"} | tag | about
            | {"licence": candidate.licence, "url": candidate.url})
        say({"action": "inspect", "status": "start"} | tag)
        target, viewer = candidate, ""  # what is fetched (the candidate, or its folder kind), why the viewer failed
        try:
            try:
                mapping = (_choose_table if is_table else _choose)(candidate, goal, modality, wanted, deadline)
            except ProbesUnavailable as exc:
                say({"action": "inspect", "status": "fail"} | tag | {"reason": str(exc)})
                # A Hub dataset whose viewer fails or shows no input column may still keep its files in one folder
                # per label: tried in the same candidate's turn (a viewer dataset without a label column is not).
                folder = None if is_table or isinstance(exc, _NoLabel) else _as_folder(candidate, modality)
                if folder is None or deadline - _now() < 5:
                    tried.append(_tried(candidate, f"inspect: {exc}"))
                    say({"action": "candidate", "status": "fail"} | tag | {"reason": str(exc)})
                    continue
                target, viewer, tag = folder, str(exc), tag | {"kind": "folder"}
                say({"action": "inspect", "status": "start"} | tag)
                try:
                    mapping = _choose_folder(folder, goal, modality, wanted, deadline, min_files)
                except ProbesUnavailable as exc2:
                    reason = f"inspect: {exc}; folder: {exc2}"
                    tried.append(_tried(candidate, reason))
                    say({"action": "inspect", "status": "fail"} | tag | {"reason": str(exc2)})
                    say({"action": "candidate", "status": "fail"} | tag | {"reason": reason})
                    continue
            say({"action": "inspect", "status": "ok"} | tag | {"mapping": mapping.to_dict()})
            fetched += 1
            dest = dest_root / _folder_for(target)
            say({"action": "fetch", "status": "start"} | tag | {"folder": str(dest)})
            try:
                report = fetch_probes(target, mapping, dest, wanted_labels=wanted_labels, disjoint_from=disjoint,
                                      budget_s=min(fetch_budget, max(0.0, deadline - _now())), **fetch_kwargs)
            except ProbesUnavailable as exc:
                reason = f"inspect: {viewer}; folder fetch: {exc}" if viewer else f"fetch: {exc}"
                tried.append(_tried(candidate, reason))
                say({"action": "fetch", "status": "fail"} | tag | {"reason": str(exc)})
                say({"action": "candidate", "status": "fail"} | tag | {"reason": reason})
                continue
        finally:
            if is_table:
                candidate._loaded = None  # the parsed rows are not needed again
            if isinstance(target, FolderCandidate):
                target._files = None  # nor the file list
        problem = _probe_set_problem(report, goal, modality, fetch_kwargs.get("min_per_label", 6))
        if problem:
            shutil.rmtree(report.folder, ignore_errors=True)
            tried.append(_tried(candidate, f"set: {problem}"))
            say({"action": "fetch", "status": "fail"} | tag | {"reason": problem})
            say({"action": "candidate", "status": "fail"} | tag | {"reason": problem})
            continue
        say({"action": "fetch", "status": "ok"} | tag | {"counts": dict(report.counts), "rows_seen": report.rows_seen,
                                                         "skipped": dict(report.skipped), "bytes": report.bytes})
        say({"action": "candidate", "status": "ok"} | tag | {"labels": list(report.labels),
                                                             "licence": report.licence})
        notes = ([f"the dataset viewer could not be used ({viewer}), so the files were read from the repository's "
                  "folders"] if viewer else []) + [mapping.reason] + list(report.notes)
        info = {"hub_id": tag["hub_id"], "url": candidate.url, "title": candidate.title, "licence": report.licence,
                "labels": list(report.labels), "label_values": dict(report.label_values),
                "counts": dict(report.counts), "mapping": mapping.to_dict(), "rows_seen": report.rows_seen,
                "skipped": dict(report.skipped), "queries": queries, "query_origin": origin, "tried": tried,
                "notes": notes, "kind": tag["kind"], "source": source,
                "revision": target.revision if isinstance(target, FolderCandidate) else None,
                "licence_source": report.licence_source, "web": web_info, "excluded": excluded,
                "disjoint_from": list(disjoint)}
        return report.folder, info
    reasons = "; ".join(f"{t['hub_id'] or t.get('url')} ({t['reason']})" for t in tried) or "none was tried"
    raise ProbesUnavailable("no candidate dataset gave usable probes: " + reasons
                            + (f"; the {total:.0f} s time budget ran out" if out_of_time else "") + _web_note(web))
