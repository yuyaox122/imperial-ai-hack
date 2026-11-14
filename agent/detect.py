"""Work out what kind of file the agent was given, from its bytes rather than its name.

Public surface (do not rename):
  Detection                  modality, format, how, detail; to_dict() for the trace
  detect(path) -> Detection  ValueError (naming the path) for a missing path, a directory or an empty file

Magic bytes in the first 64 bytes decide the audio and image formats:
  wav (RIFF....WAVE), flac (fLaC), ogg (OggS), mp3 (ID3, or an MPEG frame sync), png, jpeg.
Otherwise, when the first 4,096 bytes are UTF-8 text with no NUL byte: a first line starting with ">" is
fasta, a first line with two or more commas is csv, anything else is text. Anything else is unknown.
csv, fasta and unknown have modality "other". The file extension is never consulted.
"""

from __future__ import annotations

import codecs
from dataclasses import asdict, dataclass
from pathlib import Path

HEAD_BYTES = 64  # enough for every signature below
TEXT_BYTES = 4096  # how much is decoded to decide whether the file is text

_SIGNATURES = [  # (prefix, modality, format, name used in the detail sentence)
    (b"fLaC", "audio", "flac", "a FLAC"),
    (b"OggS", "audio", "ogg", "an Ogg"),
    (b"ID3", "audio", "mp3", "an MP3 (ID3 tag)"),
    (b"\x89PNG\r\n\x1a\n", "image", "png", "a PNG"),
    (b"\xff\xd8\xff", "image", "jpeg", "a JPEG"),
]


@dataclass(frozen=True)
class Detection:
    modality: str  # "audio" | "image" | "text" | "other"
    format: str  # "wav" | "flac" | "ogg" | "mp3" | "png" | "jpeg" | "text" | "csv" | "fasta" | "unknown"
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
    return _by_magic(path, head[:HEAD_BYTES]) or _by_content(path, head)


def _by_magic(path: Path, head: bytes) -> Detection | None:
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return _magic(path, "audio", "wav", "a WAV")
    for prefix, modality, fmt, name in _SIGNATURES:
        if head.startswith(prefix):
            return _magic(path, modality, fmt, name)
    if len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0:
        return _magic(path, "audio", "mp3", "an MP3 (MPEG frame sync)")
    return None


def _magic(path: Path, modality: str, fmt: str, name: str) -> Detection:
    detail = f"{path.name} starts with the bytes of {name} file, so it is {modality}."
    return Detection(modality, fmt, "magic bytes", detail)


def _by_content(path: Path, head: bytes) -> Detection:
    text = _as_text(head)
    if text is None:
        return Detection(
            "other", "unknown", "content", f"{path.name} matches no known signature and is not UTF-8 text."
        )
    first = text.splitlines()[0] if text else ""
    if first.startswith(">"):
        detail = f"{path.name} is text whose first line starts with '>', so FASTA."
        return Detection("other", "fasta", "content", detail)
    if first.count(",") >= 2:
        return Detection(
            "other", "csv", "content", f"{path.name} is text whose first line has {first.count(',')} commas, so CSV."
        )
    return Detection("text", "text", "content", f"{path.name} is UTF-8 text with no table or sequence header.")


def _as_text(head: bytes) -> str | None:
    """The bytes as UTF-8, allowing a character cut at the end of the window; None if not text."""
    if b"\x00" in head:
        return None
    try:
        return codecs.getincrementaldecoder("utf-8")().decode(head, final=False)
    except UnicodeDecodeError:
        return None
