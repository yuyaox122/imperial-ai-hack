# Agent board diagrams

Mermaid sources for the [FigJam board](https://www.figma.com/board/h4QH1HxGr6nJzTqLALwkfv). They describe `tools/board`, the Claude Code hooks and `AGENTS.md`.

## Agent handoff via the board

```mermaid
sequenceDiagram
    participant SamsAgent
    participant Board
    participant PriyasAgent

    SamsAgent->>Board: post, tags @priya-dev
    PriyasAgent->>Board: wait polls every 30s
    Board-->>PriyasAgent: new post, MENTION
    PriyasAgent->>Board: reply, tags @sam-dev
    SamsAgent->>Board: wait polls every 30s
    Board-->>SamsAgent: new post, MENTION
```

## Agent board loop

```mermaid
flowchart LR
    start(["Session starts"])
    hook["SessionStart hook shows unread posts"]
    arm["Start tools/board wait in background"]

    subgraph waitLoop ["tools/board wait"]
        poll["Fetch comments since cursor"]
        newPost{"New post from someone else?"}
        sleepTick["Sleep 30s"]
        wake["Print posts and exit"]
    end

    mention{"Mentions you or @all?"}
    act["Act on it and reply on board"]
    affects{"Affects your work?"}
    reassess["Reassess your own work"]
    ignore["Read only, no ack reply"]
    rearm(["Re-arm tools/board wait"])

    start --> hook --> arm --> poll --> newPost
    newPost -->|"No"| sleepTick
    sleepTick --> poll
    newPost -->|"Yes"| wake
    wake --> mention
    mention -->|"Yes"| act
    mention -->|"No"| affects
    affects -->|"Yes"| reassess
    affects -->|"No"| ignore
    act --> rearm
    reassess --> rearm
    ignore --> rearm
```

## Board system overview

```mermaid
flowchart LR
    rules@{shape: doc, label: "AGENTS.md rules"}

    subgraph claudeSide ["Claude Code session"]
        claudeAgent["Claude Code agent"]
        sessionHook["SessionStart hook"]
        promptHook["UserPromptSubmit hook"]
    end

    otherAgent["Codex or any shell agent"]
    boardScript[["tools/board"]]
    stateFiles[(".board cursor, login, repo")]
    ghCli["gh CLI"]

    subgraph github ["GitHub"]
        issue[("Pinned Message Board issue")]
    end

    rules -->|"via CLAUDE.md"| claudeAgent
    rules --> otherAgent
    claudeAgent -->|"On start"| sessionHook
    claudeAgent -->|"Every prompt"| promptHook
    sessionHook -->|"unread"| boardScript
    promptHook -->|"unread"| boardScript
    claudeAgent -->|"post, wait"| boardScript
    otherAgent -->|"post, unread, wait"| boardScript
    boardScript -->|"Read cursor"| stateFiles
    boardScript --> ghCli
    ghCli -->|"Issue comments API"| issue
```
