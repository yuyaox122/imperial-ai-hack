# An agent that gives a model a new sense

Imperial AI Hack 2026

## What it is

A user picks a base model and hands it a file the model cannot read, for example an audio clip given to a text-and-image model. An agent works out how to give the model that sense, proves on held-out examples that it works, and installs it, or refuses and says why. The user then keeps chatting with the same model. The interface is a chat harness with a model picker, a file drop, a panel of installed senses, and a trace of what the agent did.

## Status

As of Saturday 14 November 2026:

- The plan is approved by the team lead. It lives in `docs/plan.md` on [PR #5](https://github.com/yuyaox122/imperial-ai-hack/pull/5).
- The feasibility pilot is done. Its code and full result tables are in `socket_pilot/` on PR #5.
- A prototype is being built under [issue #6](https://github.com/yuyaox122/imperial-ai-hack/issues/6) on branch `6-mvp-prototype`.
- Track: Autonomous Agents is proposed; the lead has not yet confirmed it. Sponsor prizes the lead approved aiming for: Databricks and Linkup.
- Submissions close Sunday 15 November at 12:00.

## How a sense reaches a model

| Bridge | How it works | Works with |
|---|---|---|
| Native | The model already accepts the input. | Whatever the model supports |
| Text | A specialist model describes or transcribes the input, and the text goes in the prompt. | Any model |
| Tool | The specialist is exposed as a tool the base model calls. | Any model with tool use |
| Image | The input is rendered as an image, such as a spectrogram. | Any model with vision; untested |
| Vector | A closed-form map writes the specialist's embedding into the base model's early layers. | Open-weight models only |

## What has been measured

All figures are on held-out data. Questions have five options, so chance is 20% unless stated.

- Vector bridge on sound classes it was fitted on (ESC-50), with frozen Qwen3.5-0.8B and 2B: 99.0% and 96.8%, against 19.2% and 19.0% with swapped audio. Fitting takes 0.5 to 1.0 seconds.
- Music genre on held-out tracks of seen genres (GTZAN): 84.9%. Rock or classical when both genres were left out of fitting: 53.5%, where chance is 50%.
- Text bridge on sounds never fitted on: 83.2%, against 53.2% for a vector bridge fitted on caption text only.
- A text-only model (SmolLM2-1.7B) reading Qwen3.5-2B's internal states for an image, through a map fitted on words only: 62.5% on COCO questions, against 24.0% with no image.

The vector bridge is fast and accurate for categories the user can give examples of and fails outside them, so the agent's verification step, and its refusal when a bridge fails the controls, is the core of the product.

## Repository

- `AGENTS.md`: team rules for every agent.
- `tools/board`: the team message board, pinned issue #1.
- `design/`: interface mockup and board diagrams.
- `docs/`: the plan, once PR #5 merges.
- Application code lands from PR #5 and issue #6.

## Working in this repo

- Read `AGENTS.md` first.
- One issue per piece of work, branch `<issue#>-<slug>`, PR with `Closes #N`.
- Post on the board with `tools/board post "[Name's Agent] ..."` and read it with `tools/board read -n 20`.

## Models and data

- Models: Qwen/Qwen3.5-0.8B, Qwen/Qwen3.5-2B, HuggingFaceTB/SmolLM2-1.7B-Instruct, laion/clap-htsat-unfused.
- Data: ESC-50, GTZAN, AudioCaps caption text, COCO val2017.
