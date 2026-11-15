"""Work out what kind of file the agent was given, from its bytes rather than its name.

Public surface (do not rename):
  Detection                  modality, format, how, detail; to_dict() for the trace
  detect(path) -> Detection  ValueError (naming the path) for a missing path, a directory or an empty file

Magic bytes in the first 64 bytes decide the audio and image formats:
  wav (RIFF....WAVE, or RF64....WAVE for files over 4 GB), aiff (FORM....AIFF or FORM....AIFC), au (".snd" with
  a big-endian encoding code 1-27 at byte 12, so text that starts ".snd" stays text),
  flac (fLaC), ogg (OggS), png, jpeg, and mp3: an ID3v2.2-2.4 tag ("ID3" then version
  byte 2, 3 or 4) or an MPEG Layer III frame sync (FF, then a byte with the top three bits set, layer bits 01
  and a version that is not reserved: FF FB, FF FA, FF F3, FF F2, FF E3 and the like; not FF FE or AAC FF F1).
Otherwise the first 4,096 bytes are read as text: UTF-16 when they start with a byte-order mark (FF FE or
FE FF), else UTF-8 (a UTF-8 byte-order mark is skipped). A character cut off at the end is tolerated only
when the 4,096-byte window was filled. Text containing NUL is not text. For text: a first line starting with
">" is fasta, a first line with two or more commas is csv, anything else is text. Anything else is unknown.
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
    (b"\x89PNG\r\n\x1a\n", "image", "png", "a PNG"),
    (b"\xff\xd8\xff", "image", "jpeg", "a JPEG"),
]


@dataclass(frozen=True)
class Detection:
    modality: str  # "audio" | "image" | "text" | "other"
    format: str  # "wav" | "aiff" | "au" | "flac" | "ogg" | "mp3" | "png" | "jpeg" | "text" | "csv" | "fasta" | "unknown"
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
    return _by_magic(path, head[:HEAD_BYTES]) or _by_content(path, head, filled=len(head) == TEXT_BYTES)


def _by_magic(path: Path, head: bytes) -> Detection | None:
    if head[:4] in (b"RIFF", b"RF64") and head[8:12] == b"WAVE":
        return _magic(path, "audio", "wav", "a WAV")
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
    return None


def _is_layer3_sync(b1: int) -> bool:
    """Second byte of an MPEG audio frame header: sync bits set, layer III (01), version not reserved (01)."""
    return b1 & 0xE0 == 0xE0 and b1 & 0x06 == 0x02 and b1 & 0x18 != 0x08


def _magic(path: Path, modality: str, fmt: str, name: str) -> Detection:
    detail = f"{path.name} starts with the bytes of {name} file, so it is {modality}."
    return Detection(modality, fmt, "magic bytes", detail)


def _by_content(path: Path, head: bytes, *, filled: bool) -> Detection:
    text, encoding = _as_text(head, filled=filled)
    if text is None:
        return Detection(
            "other", "unknown", "content", f"{path.name} matches no known signature and is not UTF-8 or UTF-16 text."
        )
    first = text.splitlines()[0] if text else ""
    if first.startswith(">"):
        detail = f"{path.name} is text whose first line starts with '>', so FASTA."
        return Detection("other", "fasta", "content", detail)
    if first.count(",") >= 2:
        return Detection(
            "other", "csv", "content", f"{path.name} is text whose first line has {first.count(',')} commas, so CSV."
        )
    return Detection("text", "text", "content", f"{path.name} is {encoding} text with no table or sequence header.")


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
