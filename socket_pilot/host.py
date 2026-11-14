"""A frozen Qwen3.5 host with a socket: read its text representations, and write vectors into slots."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

# The slot token only marks positions; its hidden state is overwritten at the socket layer.
SLOT_CANDIDATES = ["<|vision_pad|>", "<|box_start|>", "@"]


class Host:
    def __init__(self, path: str, n_slots: int = 4):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if self.device == "mps" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype).to(self.device).eval()
        self.layers = self.model.model.language_model.layers
        self.n_layers = len(self.layers)
        self.n_slots = n_slots

        cfg = self.model.config
        reserved = {getattr(cfg, "image_token_id", None), getattr(cfg, "video_token_id", None)}
        for cand in SLOT_CANDIDATES:
            ids = self.tok.encode(cand, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in reserved:
                self.slot_text, self.slot_id = cand, ids[0]
                break
        else:
            raise RuntimeError("no single-token slot marker found")

    @property
    def slots(self) -> str:
        return self.slot_text * self.n_slots

    def chat(self, user: str) -> str:
        msgs = [{"role": "user", "content": user}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def token_id(self, text: str) -> int:
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{text!r} is not a single token")
        return ids[0]

    @staticmethod
    def _same_length_batches(lengths: list[int], batch: int):
        """Batches of equal token length, so nothing is ever padded."""
        by_len: dict[int, list[int]] = {}
        for i, n in enumerate(lengths):
            by_len.setdefault(n, []).append(i)
        for idx in by_len.values():
            for s in range(0, len(idx), batch):
                yield idx[s : s + batch]

    @torch.no_grad()
    def text_reps(self, texts: list[str], layers: list[int] | None = None, batch: int = 64) -> np.ndarray:
        """Mean-pooled hidden state after each requested block (default: all), for each text:
        [n_texts, n_requested_layers, d]."""
        layers = list(range(self.n_layers)) if layers is None else layers
        enc = [self.tok.encode(t, add_special_tokens=False) for t in texts]
        out = None
        for idx in self._same_length_batches([len(e) for e in enc], batch):
            ids = torch.tensor([enc[i] for i in idx], device=self.device)
            hs = self.model.model(input_ids=ids, output_hidden_states=True).hidden_states
            if out is None:
                out = np.zeros((len(texts), len(layers), hs[0].shape[-1]), dtype=np.float32)
            for k, layer in enumerate(layers):
                out[idx, k] = hs[layer + 1].float().mean(dim=1).cpu().numpy()
        return out

    @torch.no_grad()
    def generate(self, prompt: str, layer: int | None = None, slot_vec: np.ndarray | None = None, max_new_tokens: int = 24) -> str:
        """Greedy free-text answer, with slot_vec written into the slots of the prompt."""
        ids = torch.tensor([self.tok.encode(self.chat(prompt), add_special_tokens=False)], device=self.device)
        handle = None
        if slot_vec is not None:
            cols = (ids[0] == self.slot_id).nonzero(as_tuple=True)[0]
            vec = torch.tensor(slot_vec, device=self.device)
            full_len = ids.shape[1]

            def hook(_mod, _inp, output):
                hs = output[0] if isinstance(output, tuple) else output
                if hs.shape[1] == full_len:  # only the prompt pass; later steps see cached states
                    hs[0, cols] = vec.to(hs.dtype)
                return output

            handle = self.layers[layer].register_forward_hook(hook)
        try:
            out = self.model.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False)
        finally:
            if handle is not None:
                handle.remove()
        return self.tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()

    @torch.no_grad()
    def score(
        self,
        prompts: list[str],
        option_ids: list[int],
        layer: int | None = None,
        slot_vecs: np.ndarray | None = None,
        batch: int = 16,
    ) -> np.ndarray:
        """Next-token logits over the option tokens. If slot_vecs is given, row i's vector replaces
        the hidden state at every slot position of prompt i, after block `layer`."""
        enc = [self.tok.encode(self.chat(p), add_special_tokens=False) for p in prompts]
        out = np.zeros((len(prompts), len(option_ids)), dtype=np.float32)
        for idx in self._same_length_batches([len(e) for e in enc], batch):
            ids = torch.tensor([enc[i] for i in idx], device=self.device)
            handle = None
            if slot_vecs is not None:
                rows, cols = (ids == self.slot_id).nonzero(as_tuple=True)
                vecs = torch.tensor(slot_vecs[idx], device=self.device)

                def hook(_mod, _inp, output, rows=rows, cols=cols, vecs=vecs):
                    hs = output[0] if isinstance(output, tuple) else output
                    hs[rows, cols] = vecs[rows].to(hs.dtype)
                    return output

                handle = self.layers[layer].register_forward_hook(hook)
            try:
                last = self.model.model(input_ids=ids).last_hidden_state[:, -1]
                logits = self.model.lm_head(last)[:, option_ids]
            finally:
                if handle is not None:
                    handle.remove()
            out[idx] = logits.float().cpu().numpy()
        return out
