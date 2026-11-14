# How Figma drove the interface

The web interface in `web/` (`index.html`, `app.js`, `style.css`) has had two designs, and both live in the team Figma file
[Imperial-AI-Hackathon](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=0-1).

- **Version 2, current (#17).** Michael's Henad design system, drawn as a minimal FUI.
  Henad's tokens are a Figma variable collection whose code syntax is the CSS in `web/style.css`.
  The running build was captured back into Figma with every colour bound to those variables, and its parts became Figma components.
- **Version 1 (#11, PR #15).** Built from the team's Figma design system and the three screens drawn in Figma.
  Those pages are unchanged in the file, and their notes follow version 2 below.

In both versions the behaviour and the HTTP contract (`server/app.py`) are unchanged.
Every request `app.js` makes and every field it reads are the ones the server already provides, and the page shows only data the API returns.

## Version 2: the Henad redesign

Issue #17. Branch `17-henad-redesign`.

### In the Figma file

| Page | Node | What it holds |
|---|---|---|
| Henad · Design system | [55:75](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=55-75) | Henad · Tokens: the Henad variable collection as swatches, each with its CSS name |
| Henad · Design system | [55:749](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=55-749) | Henad · Components: Graph node (Installed, Refused, Not installed), Held-out check, Status badge (five states), Callout (three), Agent step (Done, Refused), Metric row (four kinds), Band (Header, Status line) |
| Henad · Screens | [50:2](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=50-2) | Henad 01 · Ready, with the New sense window open |
| Henad · Screens | [49:2](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=49-2) | Henad 02 · Sense installed, with a chat turn |
| Henad · Screens | [47:2](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=47-2) | Henad 03 · Sense refused, fallback installed |
| Local variables | collection "Henad", mode Dark | 131 colour variables |

### How Figma was used

1. **Tokens.** Henad's `tokens.json` (from Michael's claude.ai Design System artifact) became the variable collection "Henad".
   The 77 MCS entries are primitives, hidden from the pickers.
   The 54 roles, hue roles, app tokens and data colours alias them and are scoped to fills, text or strokes.
   Every variable's web code syntax is the custom property in `web/style.css`: `Roles/page` is `var(--page)`, `MCS/gray/950` is `var(--mcs-gray-950)`.
2. **Build.** `web/style.css` declares the same names with the same values, so a change in Figma maps to one line of CSS.
3. **Back into Figma.** The running build (fake mode, real API data) was captured into the Henad · Screens page in three states with the Figma MCP server (`generate_figma_design`).
   The capture bound colours to the Henad variables by their code syntax.
   A Plugin API pass (`use_figma`) then bound the remaining 198 paints by value, so every solid colour in the three screens is a Henad variable and none is a raw hex.
   The same pass redrew the corner brackets and the bridge group's end ticks, which CSS draws with gradients, as one-pixel rectangles on `Roles/rule-dark`, tiled the grid images, and replaced Iosevka Micfong with IBM Plex Mono, Henad's fallback, so teammates without Michael's font see no missing fonts.
4. **Components.** The parts of the screens became component sets with variants on the Henad · Design system page, with Henad's rule for each in its description.

### What changed in the interface

- **One look, Henad.** Colours are MCS entries named by role. Corners are square, there are no shadows, and parts are separated by one-pixel rules.
  Henad's app is dark only, so the light mode went with the Figma theme and its switch.
- **The band.** The header and a new status line sit on Henad's black band, with micfong.space's 32 px grid and a crosshair every 128 px fading in from the left.
  Labels on the band knock the grid out behind them.
- **Panes** are numbered 01 Chat, 02 Sense graph, 03 Agent run, 04 Report card and 05 Senses, with the index in the faint text colour as Henad numbers its sections.
- **The sense graph** sits on a fainter copy of the grid. Nodes are ruled boxes with an index (N01 to N04), square ports and right-angled wires in data colours.
  A node's head strip takes its state's subtle tint and its title the strong colour, as a callout's title strip does.
  The selected node carries the accent bar, as Henad's current tab does.
- **States** use Henad's app colours: blue for running and the primary action, green for finished, red for refused or failed, orange for a warning.
  The accent, orange 500, marks only focus, the current tab, the selected node, a new sense and links.
- **Status badges** fill with the hue's 900 and write in its 200. **Callouts** carry the agent's reason, the agent's verdict in the chat and every error.
- **The readout** puts the headline accuracy inside corner brackets, the inward ends of Henad's code block turned into four corners.
- **Icons** are Material Design Icons 7.4.47, the set Henad's app uses, for the controls as well as the modalities.
- **Type** is IBM Plex Sans with `ss01`, `ss02`, tabular figures and the slashed zero, which is what Henad Sans bakes in.
  Numbers and code use Iosevka Micfong where it is installed, else IBM Plex Mono.

New parts, all from data the API already returns:

- a status line: whether `/api/models` answered, the model and labelled set names, the job id, state, event count and elapsed time, and the number of senses;
- a one-pixel progress line along the band's lower edge, the share of the six steps that have finished, blue while running and green or red at the end;
- Graph and Event log tabs in place of the Full log button.

### Where the build departs from Henad, and why

| Henad says | The build | Why |
|---|---|---|
| App body text 12.5 px | 13 px | The demo is read on a projector |
| No style larger than the 26 px page title | The headline accuracy is 34 px | It is the one number the demo reads out |
| Henad Sans and Henad Mono | IBM Plex Sans and Iosevka Micfong, else IBM Plex Mono | The Henad fonts are in the artifact, not the repo. Committing font files to a repo that goes public is Michael's call |
| Wire colours | Kept from version 1 | Henad leaves data colours to the data |

The API gaps listed under version 1 still stand, and version 2 adds none.

### Evidence

The three Henad frames are the build itself, captured in fake mode, so they match it by construction.
To reproduce a state, run the server in fake mode and open http://127.0.0.1:8020:

- **Henad 01.** Press New sense.
- **Henad 02.** Press New sense, keep "These categories only", press Run agent, then attach the blues sample clip and ask about it.
- **Henad 03.** Press New sense, choose "Any sound", press Run agent.

```bash
uv sync
SENSE_FAKE=1 SENSE_HOME=$(mktemp -d) .venv/bin/python -m uvicorn server.app:app --port 8020
```

## Version 1: built from the team design system

Issue #11. Branch `11-interface-figma`. Refs #6.
Tokens, components and screens were read with the Figma MCP server (`get_metadata`, `get_variable_defs`,
`get_design_context`, `get_screenshot`, and read-only Plugin API scripts through `use_figma`), then written as
plain HTML, CSS and JavaScript with no framework and no build step.
Version 2 replaced this look. The Figma pages below are unchanged, and the build is in git history at PR #15.

### What is in the Figma file

| Page | Node | What it holds |
|---|---|---|
| Cover | [1:6](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=1-6) | Title banner, IBM Plex Sans Bold on black |
| Cover | [18:2](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=18-2) | Modality icons (MDI): 20 modalities, the Other fallback, 9 reserve icons, alternatives considered |
| Cover | [13:2](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=13-2) | Main components `Modality/*` behind 18:2 |
| Design system | [19:52](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=19-52) | Components, with variables (Primitives, Color in Dark and Light, Space & radius), text and effect styles |
| Screens & prototype | [31:721](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=31-721) | 01 · Agent at work |
| Screens & prototype | [27:51](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=27-51) | 02 · Sense installed |
| Screens & prototype | [32:1205](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=32-1205) | 03 · Sense refused |
| Screens & prototype | [34:1779](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=34-1779) | 02 · Sense installed · Light |

### Frames implemented

All three screens share one layout, and the interface reproduces it:
a 56 px top bar, then the chat panel (320 px), the centre (sense graph over the agent run strip) and the report panel (336 px).
Which screen is on show depends on the run: while the agent works the page is screen 01, after an install it is screen 02,
and after a refusal with a fallback it is screen 03. The Light frame (34:1779) is reached through the Light mode of the Color
variables, which the page follows when the operating system asks for a light theme.

#### Tokens

Every Figma variable has a web code syntax such as `var(--color-bg-app)`, and `style.css` declares the same custom properties
with the same values, so a value changed in Figma maps to one line of CSS.

| Figma collection | CSS |
|---|---|
| Primitives (`ink/*`, `gray/*`, `violet/*`, `green/*`, ...) | `--ink-1000` to `--teal-600-a40` |
| Color, mode Dark | `--color-bg-*`, `--color-border-*`, `--color-text-*`, `--color-accent-*`, `--color-status-*`, `--color-modality-*`, `--color-bridge-*` in `:root` |
| Color, mode Light | the same names inside `@media (prefers-color-scheme: light)` |
| Space & radius | `--space-2` to `--space-32`, `--radius-xs` to `--radius-full` |
| Text styles (Display/Metric, Heading/L to S, Body/M and S, Label/M and S, Caption, Overline, Mono/S) | the `font` shorthands in each component rule, and `.t-*` classes |
| Effect styles (Elevation/Card, Elevation/Popover, Glow/Installed) | `--elevation-card`, `--elevation-popover`, `--glow-installed` |

#### Components

| Figma component | Node | In the code |
|---|---|---|
| Icons (Lucide, 16 px) | [19:57](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=19-57) | `ICONS` in `app.js`, path data exported from the components with `exportAsync` |
| Tag | [19:235](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=19-235) | `.tag--*`, `tag()` |
| Button | [19:260](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=19-260) | `.btn--primary`, `--secondary`, `--soft`, `--ghost` |
| Sense chip | [19:279](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=19-279) | `.chip`, `renderChips()` |
| Toggle | [19:287](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=19-287) | `.switch` (the theme switch) |
| Port row | [19:294](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=19-294) | `.port-row`, `portRow()` |
| Metric row | [22:126](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=22-126) | `.metric--*`, `metricRow()`; bar width and chance tick from the card |
| Agent step | [22:177](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=22-177) | `.step--done`, `--running`, `--pending`, `--refused`, `renderSteps()` |
| Chat message | [22:204](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=22-204) | `.msg--user`, `--agent`, `--model`, `appendTurn()` |
| Result card | [22:229](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=22-229) | `.result`, `.result--refused`, `postAgentResult()` |
| Attachment, Suggestion, Composer, Drop zone | [22:233](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=22-233) | `.attachment`, `.composer`, `.drop-zone` |
| Sense row | [22:327](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=22-327) | `.sense-row`, `.sense-row--refused`, `renderSenses()` |
| Node | [23:487](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=23-487) | `.node`, `.node--running`, `--installed`, `--refused`, `--skipped`, `--model`, `graphNode()` |
| Top bar | [23:492](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=23-492) | `.topbar`, `.model-picker` |
| Modality icons | [13:2](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=13-2) | `MODALITY_ICONS` and `MODALITY_ALIASES`; a key the table does not know gets Other |

### Where the build differs from the screens, and why

The interface shows only what the API returns. Where a screen holds a placeholder the API cannot fill, it is left out:

| In the screens | In the build | Why |
|---|---|---|
| Breadcrumb "Hackathon team / Sound check", the chat's session name, the edit icon | Left out | There are no teams or chat sessions |
| Team avatars, Share, the "…" menu on the report | Left out | Not built |
| Replay mode toggle, Replay run | Left out; the toggle's place holds the theme switch | There is no replay API |
| Input file node with waveform, "Detected: audio" | Left out | A run starts from a labelled set, not from a dropped file |
| CLAP encoder node (specialist, latency, "picked by agent") | Left out | The trace does not name the specialist (see API gaps) |
| Image bridge "skipped" | Shown only for bridges the planner named | The planner reports the kinds it tries, not the ones it ruled out |
| Step titles "Detect the sense", "Find a specialist" and code such as `sniff(file)` | The loop's six real steps (Check the base model, Choose bridges, Build bridges, Verify on held-out, Install or refuse, Install) and a summary of each step's data, for example `rules: vector, text` | The steps and their data come from the trace |
| "212 of 400 clips scored" while verifying | "scoring…", then "2 of 2 scored" | Verification reports per bridge, not per clip |
| Toggles on installed senses | A Remove button | The API can remove a sense but not disable one |
| Try next suggestions such as "expect refusal" | Sample clips from `/api/samples` in that place | A suggestion would predict an outcome the API has not produced |
| The New sense form | A popover opened by New sense and Add a sense | The screens have no form, but a run needs a labelled set and a scope |
| "CLAP alone, no LLM" | "Specialist alone, no LLM" | Same reason as the CLAP node |

Two rules map API values onto the design's states:

- A bridge the API marks `rejected` is drawn Refused when the planner tried it before the installed bridge, or when nothing was installed (screen 03). A bridge planned after the installed one was not needed, so it is drawn as a plain candidate tagged "not installed", as screen 02 draws its text bridge.
- When the planner's first bridge was refused and a later one installed, the run is tagged "refused · fallback installed", the chat card names the refused bridge with the agent's reason, and the report opens on its card (screen 03).

The report shows the new `interval` field as "95% CI low to high" in the headline note and the `specialist_alone` control as a reference row, as the screens do.

### API gaps

Data the screens show that the API does not return. None of these block the interface; each would let it show more.

1. The specialist each bridge uses (name, latency, why it was picked), in the `fit` or `plan` event data. This would fill the CLAP encoder node and name the specialist-alone row.
2. The bridge kinds the planner considered and ruled out, with the reason, so the graph can draw them as Skipped.
3. Progress inside `verify` (clips scored so far), for the "212 of 400" counter of screen 01.
4. Enabling and disabling an installed sense without removing it, for the Sense row toggle.
5. The audio file's duration and channel layout, for the Attachment meta ("Audio · 5.0 s") and the input node footer.
6. A replay endpoint, if the team wants Replay mode.

### The modality icon set

Michael asked for a predefined set of modality icons from [Material Design Icons](https://pictogrammers.com/library/mdi/)
(7.4.47, Apache-2.0) with one icon for every other modality. The set was chosen from the 7,447 icons in the catalogue, drawn
into Figma as components (13:2), documented with the alternatives considered (18:2), and then used in the code: the drop zone
tiles, the attachment tiles and the sample clips take their icon from `MODALITY_ICONS`. The set is visual only. Any modality
string from the server picks an icon by key or alias, and anything else gets Other.

### The Henad theme

Version 1 ended with Henad as a second theme behind a switch in the top bar, remapping the Figma tokens over the same markup.
Version 2 above replaced both the switch and the Figma look with a redesign in Henad.

### Compared with the earlier mockup

`design/sensorium/sensorium.html` (with `preview.png`) is a single static frame of the same idea, generated by `gen.py`.
The Figma file grew out of it and keeps its structure (top bar, chat, sense graph, agent run strip, report card),
but differs in four ways:

- It is a design system: tokens with code syntax, text and effect styles, and components with variants, where the mockup had hard-coded colours.
- It has three screens for the three states of a run (working, installed, refused with a fallback) and a Light variant, where the mockup showed one finished run.
- The chat panel is narrower (320 px against 400 px) and so is the report panel (336 px against 380 px), which gives the graph more room.
- Its refusal story is explicit: refused nodes are dashed red, the report opens on the refused card, and the run tag says "refused · fallback installed".

### Evidence

Three side-by-side images, Figma frame on the left and the version 1 build on the right, were made for the presentation and sent to Michael.
They show version 1, so reproducing them needs the build as of PR #15, commit `bc6c76c`, for example in a separate worktree:

1. **Screen 02, sense installed.** Open [27:51](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=27-51). In the build, press New sense, keep "These categories only", press Run agent, then attach the blues sample clip and ask about it. The graph, the agent run strip, the report card with its controls and chance tick, the result card and the model's "heard via vector bridge" tag match the frame, with the build's numbers from the API.
2. **Screen 03, sense refused.** Open [32:1205](https://www.figma.com/design/9DxXU1h0aHjw8Mc8vUwl8O/Imperial-AI-Hackathon?node-id=32-1205). In the build, press New sense, choose "Any sound", press Run agent. The vector bridge node is Refused (dashed red), the text bridge Installed, the run is tagged "refused · fallback installed", step 5 carries the refusal, and the chat card reads "Vector bridge refused" with the agent's reason.
3. **Two themes, one markup.** Turn on Henad style in the top bar. Every component keeps its Figma structure and takes Henad's tokens.
