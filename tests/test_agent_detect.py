"""Modality detection: magic bytes first, then a look at the content.

Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_agent_detect.py
"""

from __future__ import annotations

import pytest

from agent.detect import detect

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
