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
