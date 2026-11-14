# Plan: an agent that gives a model a new sense

Status: proposal from Michael, written by Michael's Claude on Saturday 14 November. It needs the team
lead's decision on the points under [Decisions needed](#decisions-needed) before anyone builds on it.

## The idea in one paragraph

A user picks a base model and hands it a file the model cannot read, for example an audio clip given to a
text-and-image model. An agent then works out how to give the model that sense, proves on held-out
examples that the sense works, and installs it, or refuses and says why. After that the user keeps
chatting with the same model, which can now use the new input. The interface is a chat harness in the
style of opencode: a model picker, a file drop, a panel of installed senses, and a trace of what the agent
did.

The track is Autonomous Agents, whose brief is "agents that research, write and run code, and reliably
finish real multi-step tasks". The multi-step task here is fit, verify, install. The checkable end state
is a report card with accuracy against controls.

## How a sense reaches a model

A bridge is one way of getting a modality into a base model. The harness keeps a registry of bridges per
model and modality.

| Bridge | How it works | Works with |
|---|---|---|
| Native | The model already accepts the input | Whatever the model supports, such as images for Claude and Qwen3.5 |
| Text | A specialist model describes or transcribes the input, and the text goes in the prompt | Any model |
| Tool | The specialist is exposed as a tool the base model calls when it wants to look or listen | Any model with tool use |
| Image | The input is rendered as an image, such as a spectrogram or sampled video frames | Any model with vision; untested by us |
| Vector | A closed-form map writes the specialist's embedding into the base model's early layers | Open-weight models only |

API-key models such as Claude cannot take the vector bridge, because an API exposes no hidden states.
Claude's API accepts text, images and PDF or plain-text documents; its current Files and Vision
documentation lists no audio or video input.

## What we have measured

All numbers are from `socket_pilot/` (draft PR #5), on held-out data, with five-option questions where
chance is 20% unless stated. Full tables are in `socket_pilot/README.md`.

| Finding | Number |
|---|---|
| Vector bridge, sounds it was fitted on (ESC-50), Qwen3.5-0.8B and 2B | 99.0% and 96.8%, against 19.2% and 19.0% with swapped audio |
| Time to fit the vector bridge | 0.5 to 1.3 seconds per block; about 25 seconds with the layer choice (5 blocks, each scored on 200 dev questions) |
| Vector bridge, music genre on held-out tracks of seen genres (GTZAN, de-duplicated split) | 83.7% on 196 tracks; rock or classical 94.9% on 39 tracks |
| Vector bridge, rock or classical when both genres were left out of fitting | 51.8%, where chance is 50% |
| Text bridge, sounds never fitted on (nearest caption pasted into the prompt), Qwen3.5-0.8B | 83.2% |
| Vector bridge fitted on caption text only, same unseen sounds | 53.2% |
| Text bridge, rock or classical | 63.8%, where chance is 50% |
| Model-to-model: text-only SmolLM2-1.7B reading Qwen3.5-2B's states for an image, map fitted on words only (COCO) | 62.5%, against 24.0% with no image; Qwen3.5-2B itself scores 97.8% |

What this means for the product:

- The vector bridge is fast and accurate for categories the user can supply examples of, and useless
  outside them. Nearest-neighbour lookup matches it, so it transfers a label, not a description.
- The text bridge is the better default for open-ended input, but it is only as good as its specialist:
  our caption-retrieval version is weak on music genre.
- The agent's job is therefore real: choose the bridge, run the controls, and refuse to install a bridge
  that fails them. The unseen-genre result is the case where it must refuse.

## What the agent does

Given a base model, a modality and some example files (with labels if the user has them):

1. Check what the model accepts natively and whether it is open-weight.
2. Pick candidate bridges and the specialist model for the modality.
3. Build each candidate. For the vector bridge this means fitting the map on the labelled examples.
4. Score each candidate on held-out examples against four controls: swapped input, no input, lookup of
   the nearest fitted example, and the text bridge.
5. Install the best candidate that beats the controls, or install the text bridge and state why the
   others failed.
6. Write a report card: accuracy, each control, sample size, verdict.

The orchestrator is Claude with tool use. Its tools wrap code that already exists in `socket_pilot/`
(`host.py` for writing into Qwen3.5, `run_pilot.py` for fitting and the controls).

## Three demo scenarios

1. **Seen categories, open model.** "Make Qwen3.5-2B recognise music genres." The agent fits the vector
   bridge, verifies it and installs it. The user plays a track and asks about it.
2. **Unseen categories, open model.** "Make it handle any sound." The vector bridge fails the controls,
   so the agent installs the text bridge and explains the failure.
3. **API model.** The same request with Claude as the base model. The vector bridge is unavailable, so the
   agent installs a tool bridge and Claude calls it.

## Proposed contracts

These are a starting point for the owners to settle in the first hour, not final.

```python
class BaseModel(Protocol):
    name: str
    native: set[str]          # e.g. {"text", "image"}
    open_weights: bool        # True allows the vector bridge
    def chat(self, messages: list[dict], inputs: list["SenseInput"]) -> Iterator[str]: ...

class Bridge(Protocol):
    kind: str                 # "native" | "text" | "tool" | "image" | "vector"
    modality: str             # e.g. "audio"
    def prepare(self, path: Path) -> "SenseInput": ...   # text, image bytes, or a vector

@dataclass
class ReportCard:
    bridge: str
    n: int
    accuracy: float
    controls: dict[str, float]   # swapped, none, lookup, text
    verdict: str                 # "installed" | "rejected: <reason>"
```

Backend to interface, over HTTP:

| Call | Purpose |
|---|---|
| `GET /models` | Base models and what each accepts |
| `GET /senses?model=` | Installed bridges with their report cards |
| `POST /chat` | Message plus files; streams the reply |
| `POST /fit` | Start an agent run for a model and modality; returns a job id |
| `GET /jobs/{id}` | Trace events and the final report card |

## Proposed workstreams

Only the lead assigns work. This split is a suggestion that keeps folders separate.

| Workstream | Folder | Contents |
|---|---|---|
| A. Harness core | `harness/` | Base model adapters (local Qwen3.5, Claude), bridge registry, vector and text bridges, the controls. Michael's Claude can take this, since it builds on `socket_pilot/`. |
| B. Agent | `agent/` | The Claude tool-use loop for fit, verify, install; trace events; report cards |
| C. Interface | `web/` | Chat, model picker, file drop, senses panel, trace view |
| D. Evaluation and demo | `eval/`, `demo/` | Held-out sets for the three scenarios, a recorded backup run, pitch, README and Devpost text |

## Milestones (London time)

| By | Deliverable |
|---|---|
| Sat 17:00 | Lead's decisions posted; owners in `AGENTS.md`; contracts agreed |
| Sat 19:00 | One vertical slice: the interface sends a clip to local Qwen3.5 through an installed vector bridge and shows the answer |
| Sat 21:00 | Agent runs fit, verify, install from the command line for scenario 1 (venue closes at 21:00) |
| Sun 01:00 | Feature freeze; all three scenarios pass twice in a row |
| Sun 10:30 | Recorded backup run; README lists models, APIs and datasets; repository made public |
| Sun 11:30 | Devpost submitted (submissions close at 12:00) |

## Constraints

- **Compute.** Michael's laptop (M4 Pro, 24 GB) runs Qwen3.5-0.8B and 2B; the 4B exhausted its memory.
  Qwen3.5 weights and SmolLM2 are already on that laptop. The remote 4090 host is not usable for this
  (its link is too slow to move data).
- **Rules.** All project code must be written during the hackathon. Pretrained models, public APIs and
  open-source libraries are allowed and must be listed. Devpost needs a public GitHub repository.
- **Credits.** Each participant has 50 USD of Anthropic API credit, which covers the orchestrator.

## Risks

- **Judges read it as model tooling, not an agent.** Lead the demo with the agent run and its report
  card, not with the chat.
- **The vector bridge is narrow.** Present it as one bridge among several and show the report card that
  rejects it in scenario 2.
- **Live demo failure.** Cache specialist embeddings and keep a replay mode and a recorded run.

## Decisions needed

From the team lead:

1. Does the team build this, and does it replace or combine with the other concept being scoped?
2. Main track: Autonomous Agents, as proposed, or Voice, Vision & Generative Media.
3. Who owns each workstream.
4. Whether to aim for sponsor prizes that need extra integration: Linkup for the agent's research step,
   Figma for the interface design.
