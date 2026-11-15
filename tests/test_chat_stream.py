"""Chat history and streaming (#60): the history contract, QwenModel's message building and streaming (without
weights: a character tokenizer and a stand-in for model.generate), FakeModel, POST /api/chat's history field and
POST /api/chat/stream, with model doubles and end to end in fake mode.

No weights, no downloads and no network. The OpenAI adapter's history and streaming are in tests/test_models_api.py.
Run: SENSE_FAKE=1 .venv/bin/python -m pytest -q tests/test_chat_stream.py
"""

from __future__ import annotations

import gc
import io
import json
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import anyio
import numpy as np
import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

from agent import jobs
from agent import planner as planner_mod
from sense import bridges, config, datasets, models, registry
from sense.contracts import FitRequest, JobState, ModelInfo, SenseInput
from sense.models import N_SLOTS, SLOT_MARKER
from sense.verify import verify
from server import app as server_app

HISTORY = [{"role": "user", "content": "What is this?"}, {"role": "assistant", "content": "A dog barking."}]


def parse_events(text: str) -> list[tuple[str, object]]:
    """Server-Sent Events as (event, data) pairs; every event must be exactly an event line and a data line."""
    assert text.endswith("\n\n")
    out = []
    for block in text[:-2].split("\n\n"):
        event, data = block.split("\n")
        assert event.startswith("event: ") and data.startswith("data: ")
        out.append((event[len("event: "):], json.loads(data[len("data: "):])))
    return out


# ----------------------------------------------------------------------------- the history contract
def test_chat_history_keeps_turns_and_refuses_anything_else():
    assert models.chat_history(None) == [] and models.chat_history([]) == []
    turns = [{"role": "user", "content": "a", "extra": 1}, {"role": "assistant", "content": ""}]
    assert models.chat_history(turns) == [{"role": "user", "content": "a"}, {"role": "assistant", "content": ""}]
    with pytest.raises(ValueError, match="must be a list of earlier turns, not dict"):
        models.chat_history({"role": "user", "content": "a"})
    for bad in ([{"role": "system", "content": "a"}], [{"role": "user", "content": 3}], [["user", "a"]],
                [{"role": "user"}], [{"content": "a"}]):
        with pytest.raises(ValueError, match='Turn 1 of the chat history is not a turn: each needs a "role"'):
            models.chat_history(bad)
    with pytest.raises(ValueError, match="Turn 2 "):
        models.chat_history([HISTORY[0], "hello"])


@pytest.mark.parametrize("pieces", [
    ["Hello", " world"],
    ["  ", "\n Hello", " there ", "\n", "world", " \n"],
    [" a b ", " ", " c "],
    ["", "x", ""],
    ["   ", "\n"],
    [],
])
def test_stripped_pieces_join_to_the_stripped_reply(pieces):
    out = list(models.stripped_pieces(pieces))
    assert "".join(out) == "".join(pieces).strip()
    assert all(out)  # no empty pieces


# ----------------------------------------------------------------------------- QwenModel, without weights
SLOT_ID = 1_000_000


class CharTokenizer:
    """One token per character, except the slot marker, which is one token (SLOT_ID), as in Qwen3.5."""

    def __init__(self):
        self.messages: list[list[dict]] = []

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        self.messages.append(msgs)
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in msgs) + "<assistant>"

    def encode(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            if text.startswith(SLOT_MARKER, i):
                ids.append(SLOT_ID)
                i += len(SLOT_MARKER)
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i)) for i in ids if int(i) != SLOT_ID)


class Generate:
    """Stands in for model.generate: produces the characters of `text` one token at a time, feeding the streamer
    and checking the stopping criteria as transformers does, and records the slot hooks in place while it runs and
    whether the model's lock was held."""

    def __init__(self, qwen, text: str, delay: float = 0.0, fail_at: int | None = None):
        self.qwen, self.text, self.delay, self.fail_at = qwen, text, delay, fail_at
        self.produced = 0
        self.hooks: list[int] = []
        self.locked: list[bool] = []

    def generate(self, input_ids, attention_mask, max_new_tokens, do_sample, streamer=None, stopping_criteria=None):
        import torch

        assert do_sample is False and attention_mask.shape == input_ids.shape
        self.hooks.append(sum(len(layer._forward_hooks) for layer in self.qwen.layers))
        self.locked.append(self.qwen._lock.locked())
        out = input_ids
        if streamer is not None:
            streamer.put(input_ids.cpu())
        for k, char in enumerate(self.text[:max_new_tokens]):
            if k == self.fail_at:
                raise RuntimeError("MPS backend out of memory")
            token = torch.tensor([ord(char)])
            out = torch.cat([out, token[None]], dim=1)
            self.produced += 1
            if streamer is not None:
                streamer.put(token)
            if stopping_criteria is not None and bool(stopping_criteria(out, None).all()):
                break
            time.sleep(self.delay)
        if streamer is not None:
            streamer.end()
        return out


@pytest.fixture
def qwen():
    """A QwenModel built without __init__, so no weights are loaded."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    model = object.__new__(models.QwenModel)
    model.info = ModelInfo(name="qwen-test", label="Qwen test", native=["text"], open_weights=True)
    model.device = "cpu"
    model.tok = CharTokenizer()
    model.layers = [torch.nn.Identity() for _ in range(3)]
    model.n_layers = 3
    model.hidden_size = 4
    model.slot_id = SLOT_ID
    model._lock = threading.Lock()
    return model


def vector_input() -> SenseInput:
    return SenseInput(kind="vector", modality="audio", source="a.wav", vector=np.ones(4, dtype=np.float32), layer=1)


def test_qwen_history_comes_first_without_slot_markers():
    nested = "<|vision_<|vision_pad|>pad|>"  # removing the inner marker once would leave a marker
    history = [
        {"role": "user", "content": f"Is {SLOT_MARKER * 2}it{nested} loud?"},
        {"role": "assistant", "content": "Yes."},
    ]
    assert models._chat_messages("And now?", history) == [
        {"role": "user", "content": "Is it loud?"},
        {"role": "assistant", "content": "Yes."},
        {"role": "user", "content": "And now?"},
    ]
    assert models._chat_messages("Hi") == [{"role": "user", "content": "Hi"}]


def test_qwen_render_puts_slots_only_in_the_final_message(qwen):
    history = [
        {"role": "user", "content": SLOT_MARKER * 5},
        {"role": "assistant", "content": f"It{SLOT_MARKER} rings."},
    ]
    rendered = qwen._render([vector_input()], "What now?", history)
    [msgs] = qwen.tok.messages
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"] == "It rings."
    slots = [i for i, t in enumerate(rendered.ids) if t == SLOT_ID]
    assert len(slots) == N_SLOTS
    final = len(qwen.tok.encode("".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in msgs[:2])))
    assert min(slots) > final  # every slot sits in the final user message
    [(positions, _, layer)] = rendered.writes
    assert positions == slots and layer == 1


def test_qwen_render_strips_a_nested_slot_marker_from_the_message_and_text_inputs(qwen):
    """Removing the marker once from "<|vision_<|vision_pad|>pad|>" would leave a marker, which the tokenizer
    turns into a slot token, and the slot count check would then fail the chat."""
    nested = "<|vision_<|vision_pad|>pad|>"
    rendered = qwen._render([], f"What is {nested} this?")
    assert SLOT_ID not in rendered.ids and rendered.writes == []
    assert qwen.tok.messages[-1][-1]["content"] == "What is  this?"
    text = SenseInput(kind="text", modality="audio", source="a.wav", text=f'sounds like "{nested}{SLOT_MARKER}"')
    rendered = qwen._render([text], f"{nested}What is it?")
    assert SLOT_ID not in rendered.ids
    assert qwen.tok.messages[-1][-1]["content"] == 'Audio clip: sounds like ""\nWhat is it?'
    rendered = qwen._render([vector_input()], f"What is {nested} this?")
    slots = [i for i, t in enumerate(rendered.ids) if t == SLOT_ID]
    assert len(slots) == N_SLOTS and rendered.writes[0][0] == slots


def test_qwen_slot_count_is_still_checked(qwen):
    qwen.tok.apply_chat_template = lambda msgs, **kwargs: SLOT_MARKER + msgs[-1]["content"]  # a template adding one
    with pytest.raises(RuntimeError, match="Expected 4 slot tokens in the prompt, found 5"):
        qwen._render([vector_input()], "What now?", HISTORY)


def test_qwen_chat_stream_joins_to_chat_with_the_same_slot_hooks(qwen):
    text = "  It is a dog,\nbarking twice. \n"
    qwen.model = Generate(qwen, text)
    reply = qwen.chat("What is it?", [vector_input()], max_new_tokens=64, history=HISTORY)
    assert reply == text.strip()
    [from_chat] = qwen.tok.messages
    assert from_chat[:2] == HISTORY and [m["role"] for m in from_chat] == ["user", "assistant", "user"]
    pieces = list(qwen.chat_stream("What is it?", [vector_input()], max_new_tokens=64, history=HISTORY))
    assert len(pieces) > 1 and "".join(pieces) == reply
    assert qwen.model.hooks == [1, 1]  # the slot hook was in place for both, and removed after each
    assert sum(len(layer._forward_hooks) for layer in qwen.layers) == 0
    assert qwen.tok.messages[-1] == from_chat  # the same messages for both
    assert qwen.model.locked == [True, True]  # both generated holding the model's lock
    assert not qwen._lock.locked()


def test_qwen_chat_stream_respects_max_new_tokens(qwen):
    qwen.model = Generate(qwen, "one two three four five")
    assert "".join(qwen.chat_stream("Count", [], max_new_tokens=7)) == "one two"


def test_closing_a_qwen_stream_early_stops_generation_and_frees_the_lock(qwen):
    text = "word " * 400
    qwen.model = Generate(qwen, text, delay=0.002)
    stream = qwen.chat_stream("Talk", [], max_new_tokens=len(text))
    assert next(stream) == "word"
    stream.close()
    assert qwen.model.produced < len(text)  # stopped at the next token instead of running on
    assert not qwen._lock.locked()
    assert not [t for t in threading.enumerate() if t.name == "qwen-chat-stream"]  # the worker was joined


def test_an_error_in_qwen_generation_reaches_the_reader_after_the_pieces(qwen):
    text = "Hello there, world"
    qwen.model = Generate(qwen, text, fail_at=12)
    pieces = []
    with pytest.raises(RuntimeError, match="out of memory"):
        for piece in qwen.chat_stream("Hi", [], max_new_tokens=64):
            pieces.append(piece)
    assert "".join(pieces) == text[:12].strip()
    assert not qwen._lock.locked()
    assert sum(len(layer._forward_hooks) for layer in qwen.layers) == 0


def test_qwen_chat_stream_refuses_a_tool_input_like_chat(qwen):
    qwen.model = Generate(qwen, "never")
    tool = SenseInput(kind="tool", modality="audio", source="a.wav")
    with pytest.raises(ValueError, match="cannot call tools"):
        list(qwen.chat_stream("Hi", [tool]))
    assert qwen.model.hooks == [] and not qwen._lock.locked()


# ----------------------------------------------------------------------------- FakeModel
def fake_model():
    return models.FakeModel(ModelInfo(name="fake", label="Fake", native=["text"], open_weights=True))


def test_fake_model_says_how_many_earlier_turns_it_saw():
    model = fake_model()
    plain = model.chat("Hi", [])
    assert "earlier turn" not in plain and model.chat("Hi", [], history=[]) == plain
    assert model.chat("Hi", [], history=HISTORY) == f"{plain} (2 earlier turns)"
    assert model.chat("Hi", [], history=HISTORY[:1]).endswith("(1 earlier turn)")
    with pytest.raises(ValueError, match="not a turn"):
        model.chat("Hi", [], history=[{"role": "robot", "content": "beep"}])


def test_fake_chat_stream_joins_to_chat():
    model = fake_model()
    pieces = list(model.chat_stream("Which genre is it?", [], max_new_tokens=96, history=HISTORY))
    assert len(pieces) > 2 and all(pieces)
    assert "".join(pieces) == model.chat("Which genre is it?", [], max_new_tokens=96, history=HISTORY)
    assert "".join(model.chat_stream("Hi there you", [], max_new_tokens=3)) == model.chat("Hi there you", [], 3)


# ----------------------------------------------------------------------------- HTTP, with model doubles
INFO = ModelInfo(name="m1", label="Model one", native=["text"], open_weights=True)
INFO2 = ModelInfo(name="m2", label="Model two", native=["text"], open_weights=True)


class Bridge:
    kind, modality, model_name, name = "vector", "audio", "m1", "vector:test"

    def prepare(self, path: Path) -> SenseInput:
        assert path.is_file()  # the upload must exist while the bridge reads it
        return SenseInput(kind="vector", modality="audio", detail="fake embedding")


class Plain:
    """A model without history or chat_stream, like the doubles written before either existed."""

    info = INFO

    def chat(self, message, inputs, max_new_tokens=96):
        return f"{message}|{len(inputs)}"


class Recording:
    """Takes history and records what it was given."""

    info = INFO

    def __init__(self):
        self.histories: list = []

    def chat(self, message, inputs, max_new_tokens=96, history=None):
        self.histories.append(history)
        return f"{message}|{len(history or [])}"


class Streaming(Recording):
    """Streams the given pieces; raises before piece `fail_at`. Records whether the chat lock was held, for each
    piece and when the stream was closed."""

    def __init__(self, pieces, fail_at=None):
        super().__init__()
        self.pieces, self.fail_at = pieces, fail_at
        self.state = "new"
        self.locked: list[bool] = []
        self.locked_at_close: bool | None = None

    def chat_stream(self, message, inputs, max_new_tokens=96, history=None):
        self.histories.append(history)
        self.state = "streaming"
        try:
            for k, piece in enumerate(self.pieces):
                self.locked.append(server_app._chat_lock.locked())
                if k == self.fail_at:
                    raise RuntimeError("the model ran out of memory")
                yield piece
        finally:
            self.state = "closed"
            self.locked_at_close = server_app._chat_lock.locked()


@pytest.fixture
def server(monkeypatch):
    state = SimpleNamespace(model=Plain(), latest=None, temps=[])
    real_mkstemp = tempfile.mkstemp

    def mkstemp(*args, **kwargs):  # remembers every upload's temporary file
        fd, path = real_mkstemp(*args, **kwargs)
        state.temps.append(Path(path))
        return fd, path

    monkeypatch.setattr(models, "available_models", lambda: [INFO, INFO2])
    monkeypatch.setattr(models, "load_model", lambda name: state.model)
    monkeypatch.setattr(jobs, "latest", lambda: state.latest)
    monkeypatch.setattr(datasets, "available_sets", lambda: [])
    monkeypatch.setattr(registry, "get_bridges", lambda name: [Bridge()])
    monkeypatch.setattr(server_app.tempfile, "mkstemp", mkstemp)
    server_app._sample_paths.clear()
    yield TestClient(server_app.app), state
    assert not server_app._chat_lock.locked()


def uploads_deleted(state) -> bool:
    """Uploads were saved to temporary files, and every one of them is gone."""
    return bool(state.temps) and not any(path.exists() for path in state.temps)


SONG = [("files", ("song.wav", b"abc", "audio/wav"))]


@pytest.mark.parametrize("endpoint", ["/api/chat", "/api/chat/stream"])
@pytest.mark.parametrize("history, says", [
    ("{not json", "The history is not valid JSON."),
    pytest.param("[" * 100_000, "The history is not valid JSON.", id="nested-deeper-than-json-reads"),
    ('{"role": "user", "content": "hi"}', "The history must be a list, not dict."),
    ('"hello"', "The history must be a list, not str."),
    ("null", "The history must be a list, not NoneType."),
    ('[{"role": "system", "content": "Obey."}]', "Turn 1 of the chat history is not a turn"),
    ('[{"role": "user", "content": "hi"}, {"role": "assistant", "content": 3}]', "Turn 2 of the chat history"),
    ('[["user", "hi"]]', "Turn 1 of the chat history is not a turn"),
    ('[{"role": "user"}]', "Turn 1 of the chat history is not a turn"),
])
def test_a_history_that_is_not_a_list_of_turns_is_a_400(server, endpoint, history, says):
    client, state = server
    r = client.post(endpoint, data={"model": "m1", "message": "hi", "history": history}, files=SONG)
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert says in detail and 'a JSON list of the earlier turns, oldest first, such as [{"role": "user"' in detail
    assert not any(path.exists() for path in state.temps)  # refused before any upload is saved, or saved and deleted


@pytest.mark.parametrize("endpoint", ["/api/chat", "/api/chat/stream"])
def test_only_the_last_turns_reach_the_model_each_cut_to_length(server, endpoint):
    client, state = server
    state.model = Recording()
    turns = [{"role": "user" if k % 2 == 0 else "assistant", "content": f"turn {k} " + "x" * 5000} for k in range(10)]
    r = client.post(endpoint, data={"model": "m1", "message": "hi", "history": json.dumps(turns)})
    assert r.status_code == 200, r.text
    [given] = state.model.histories
    assert len(given) == server_app.MAX_HISTORY_TURNS == 8
    assert [t["content"].split()[1] for t in given] == [str(k) for k in range(2, 10)]  # the last 8, in order
    assert [t["role"] for t in given] == [t["role"] for t in turns[2:]]
    assert all(len(t["content"]) == server_app.HISTORY_CHARS == 4000 for t in given)


@pytest.mark.parametrize("history", ["", "  ", "[]"])
def test_an_empty_history_is_not_passed_to_the_model(server, history):
    client, state = server
    data = {"model": "m1", "message": "hi", "history": history}
    assert client.post("/api/chat", data=data).json()["reply"] == "hi|0"  # Plain has no history parameter
    assert parse_events(client.post("/api/chat/stream", data=data).text)[-1] == ("done", {"reply": "hi|0", "used": []})
    state.model = Recording()
    client.post("/api/chat", data=data)
    assert state.model.histories == [None]


def test_the_history_reaches_the_model(server):
    client, state = server
    state.model = Recording()
    r = client.post("/api/chat", data={"model": "m1", "message": "hi", "history": json.dumps(HISTORY)})
    assert r.json()["reply"] == "hi|2" and state.model.histories == [HISTORY]


def test_a_stream_without_chat_stream_sends_the_reply_as_one_token(server):
    client, state = server
    r = client.post("/api/chat/stream", data={"model": "m1", "message": "hi"}, files=SONG)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    used = [{"bridge": "vector:test", "kind": "vector", "source": "song.wav", "detail": "fake embedding"}]
    assert parse_events(r.text) == [
        ("used", used), ("token", {"text": "hi|1"}), ("done", {"reply": "hi|1", "used": used}),
    ]
    assert uploads_deleted(state)
    plain = client.post("/api/chat", data={"model": "m1", "message": "hi"}, files=SONG).json()
    assert plain == {"reply": "hi|1", "used": used}  # the same reply and used as the stream's done event


def test_a_stream_sends_each_piece_in_order_while_holding_the_chat_lock(server):
    client, state = server
    state.model = Streaming(["Hel", "lo", "", " world", "\n"])
    r = client.post("/api/chat/stream", data={"model": "m1", "message": "hi", "history": json.dumps(HISTORY)})
    events = parse_events(r.text)
    assert [name for name, _ in events] == ["used", "token", "token", "token", "token", "done"]
    assert [data["text"] for name, data in events if name == "token"] == ["Hel", "lo", " world", "\n"]  # none empty
    assert events[-1] == ("done", {"reply": "Hello world\n", "used": []})
    assert state.model.histories == [HISTORY] and state.model.state == "closed"
    assert state.model.locked == [True] * 5


def test_a_model_error_mid_stream_is_an_error_event(server):
    client, state = server
    state.model = Streaming(["Hel", "lo", " world"], fail_at=2)
    r = client.post("/api/chat/stream", data={"model": "m1", "message": "hi"}, files=SONG)
    assert r.status_code == 200
    events = parse_events(r.text)
    assert [name for name, _ in events] == ["used", "token", "token", "error"]
    assert events[-1][1] == {"detail": "The model failed to answer: the model ran out of memory"}
    assert state.model.state == "closed" and uploads_deleted(state)


def test_a_model_without_chat_stream_that_fails_is_an_error_event(server):
    client, state = server

    class Failing(Plain):
        def chat(self, message, inputs, max_new_tokens=96):
            raise RuntimeError("no answer today")

    state.model = Failing()
    events = parse_events(client.post("/api/chat/stream", data={"model": "m1", "message": "hi"}).text)
    assert events == [("used", []), ("error", {"detail": "The model failed to answer: no answer today"})]


def test_errors_before_the_stream_are_ordinary_http_errors(server):
    client, state = server
    r = client.post("/api/chat/stream", data={"model": "nope", "message": "hi"})
    assert r.status_code == 400 and "Unknown model 'nope'" in r.json()["detail"]
    r = client.post("/api/chat/stream", data={"model": "m1", "message": "  "})
    assert r.status_code == 400 and "message is empty" in r.json()["detail"]
    r = client.post("/api/chat/stream", data={"model": "m1", "message": "hi", "sample_ids": ["nope"]}, files=SONG)
    assert r.status_code == 400 and "Unknown sample 'nope'" in r.json()["detail"]
    assert uploads_deleted(state)
    state.latest = JobState(id="j1", request=FitRequest(model="m2", dataset="gtzan"))
    r = client.post("/api/chat/stream", data={"model": "m1", "message": "hi"})
    assert r.status_code == 409 and "agent run on m2 is in progress" in r.json()["detail"]


def test_a_client_that_goes_away_frees_the_lock_and_deletes_the_uploads(server):
    _, state = server
    state.model = Streaming([f"piece {k} " for k in range(50)])
    upload = UploadFile(io.BytesIO(b"abc"), filename="song.wav")
    response = server_app.api_chat_stream(model="m1", message="hi", files=[upload], sample_ids=[], history="")
    assert response.media_type == "text/event-stream"

    async def read_then_leave():
        body = response.body_iterator
        chunks = [await body.__anext__() for _ in range(3)]
        await body.aclose()  # what the server does with the body when the client disconnects
        return chunks

    chunks = anyio.run(read_then_leave)
    assert chunks[0].startswith("event: used\n") and chunks[2] == 'event: token\ndata: {"text": "piece 1 "}\n\n'
    assert state.model.state == "closed"  # the model's stream was closed, not left running
    assert state.model.locked_at_close is True  # closed by the server while it held the chat lock, not afterwards
    assert not server_app._chat_lock.locked()
    assert uploads_deleted(state)


def test_uploads_are_deleted_even_if_the_body_is_never_read(server):
    _, state = server
    state.model = Streaming(["never sent"])
    upload = UploadFile(io.BytesIO(b"abc"), filename="song.wav")
    response = server_app.api_chat_stream(model="m1", message="hi", files=[upload], sample_ids=[], history="")
    assert state.temps and all(path.exists() for path in state.temps)  # kept for the stream
    del response  # what happens to a response whose body nobody reads
    gc.collect()
    assert uploads_deleted(state)
    assert state.model.state == "new"  # the model was never asked
    assert not server_app._chat_lock.locked()


def test_a_stream_holding_the_chat_lock_still_gets_threads_while_chats_wait_for_it(server, monkeypatch):
    """Each POST /api/chat waiting for the chat lock holds a thread of the shared pool. The stream that holds the
    lock must not need one of those threads to go on, or it could never free the lock and the server would hang."""
    _, state = server
    lock = threading.Lock()
    monkeypatch.setattr(server_app, "_chat_lock", lock)
    state.model = Streaming(["one ", "two ", "three"])
    response = server_app.api_chat_stream(model="m1", message="hi", files=[], sample_ids=[], history="")
    wedged: list[bool] = []

    def chat():  # what POST /api/chat does on a thread of the pool
        with lock:
            pass

    async def scenario():
        pool = anyio.to_thread.current_default_thread_limiter()
        pool.total_tokens = 2  # this event loop's pool only; small, so that a few chats fill it
        body = response.body_iterator
        chunks = [await body.__anext__(), await body.__anext__()]  # used, then the first piece: the lock is held
        assert lock.locked()
        async with anyio.create_task_group() as tg:
            for _ in range(pool.total_tokens + 1):  # more chats than the pool has threads
                tg.start_soon(anyio.to_thread.run_sync, chat)
            while pool.statistics().tasks_waiting < 1:  # every thread of the pool waits for the lock
                await anyio.sleep(0.01)

            async def watchdog():  # if the stream is stuck, free the lock by hand so that the test ends
                await anyio.sleep(10)
                wedged.append(True)
                lock.release()

            tg.start_soon(watchdog)
            chunks += [chunk async for chunk in body]
            tg.cancel_scope.cancel()  # stops the watchdog; the chats finish now that the lock is free
        return chunks

    chunks = anyio.run(scenario)
    assert not wedged, "the stream waited for a thread held by a chat that was waiting for the stream's lock"
    assert parse_events("".join(chunks))[-1] == ("done", {"reply": "one two three", "used": []})
    assert state.model.locked_at_close is True


# ----------------------------------------------------------------------------- end to end in fake mode
@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("SENSE_FAKE", "1")
    monkeypatch.setenv("SENSE_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SENSE_SETS_DIR", raising=False)
    monkeypatch.delenv("SENSE_API_VERIFY_ITEMS", raising=False)
    monkeypatch.delenv("SENSE_API_CACHE", raising=False)
    monkeypatch.setattr(jobs, "_jobs", {})
    monkeypatch.setattr(jobs, "_latest_id", None)
    monkeypatch.setattr(planner_mod, "_preferred", None)  # the rule planner, even if an LLM planner is registered
    server_app._sample_paths.clear()
    assert config.fake()
    return TestClient(server_app.app)


def run_job(client: TestClient, scope: str = "closed") -> dict:
    r = client.post("/api/fit", json={"model": "fake", "dataset": "gtzan", "modality": "audio", "scope": scope})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    for _ in range(400):
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["status"] != "running":
            return state
        time.sleep(0.02)
    raise AssertionError("the agent run did not finish within 8 seconds")


def sample(client: TestClient, label: str) -> dict:
    return next(s for s in client.get("/api/samples", params={"dataset": "gtzan"}).json() if s["label"] == label)


def test_a_sample_streams_through_the_installed_fake_sense(client):
    assert run_job(client)["installed"] == "vector:gtzan"
    rock = sample(client, "rock")
    data = {"model": "fake", "message": "Which genre?", "sample_ids": [rock["id"]], "history": json.dumps(HISTORY)}
    r = client.post("/api/chat/stream", data=data)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_events(r.text)
    names = [name for name, _ in events]
    assert names[0] == "used" and names[-1] == "done" and set(names[1:-1]) == {"token"} and len(names) > 4
    [used] = events[0][1]
    assert (used["bridge"], used["kind"], used["source"]) == ("vector:gtzan", "vector", rock["name"])
    reply = events[-1][1]["reply"]
    assert reply == "".join(payload["text"] for name, payload in events if name == "token")
    assert "rock" in reply and reply.endswith("(2 earlier turns)")
    assert client.post("/api/chat", data=data).json() == {"reply": reply, "used": events[-1][1]["used"]}


def test_a_tool_sense_streams_and_done_says_what_the_model_did_with_the_tool(client):
    api_model = models.load_model("fake-api")
    gtzan = datasets.load_labelled_set("gtzan")
    bridge = bridges.make_tool_bridge(api_model, gtzan)
    registry.install(bridge, verify(api_model, bridge, gtzan, FitRequest(model="fake-api", dataset="gtzan")))
    rock = sample(client, "rock")
    data = {"model": "fake-api", "message": "Which genre?", "sample_ids": [rock["id"]]}
    events = parse_events(client.post("/api/chat/stream", data=data).text)
    [before] = events[0][1]
    assert before["kind"] == "tool" and before["detail"] == "the model can call the listen tool to hear this clip"
    done = events[-1]
    assert done[0] == "done"
    [after] = done[1]["used"]
    assert after["detail"].startswith("the model called listen 1 time: Nearest labels (Fake CLAP): rock ")
    assert "rock" in done[1]["reply"]
    assert client.post("/api/chat", data=data).json() == done[1]  # the same reply and used as POST /api/chat
