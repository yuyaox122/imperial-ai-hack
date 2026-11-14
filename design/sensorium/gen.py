#!/usr/bin/env python3
"""Sensorium main screen: give a model a new sense (docs/plan.md, PR #5). Writes a static 1920x1080 HTML mockup."""
import sys

WF, HF = 1920, 1080
TOP_H = 56
LEFT_W, RIGHT_W = 400, 380
CX = LEFT_W
CW = WF - LEFT_W - RIGHT_W  # 1140
CH = HF - TOP_H             # 1024

C = {
    "bg": "#0E1015", "panel": "#141720", "panel2": "#181C27", "border": "#242938",
    "node": "#1A1E2A", "nodeBorder": "#2C3245", "canvas": "#0B0D12",
    "text": "#E7E9F0", "text2": "#9BA1B4", "muted": "#626980",
    "accent": "#7C5CFF", "accentSoft": "#2A2350", "green": "#34D399", "red": "#F87171", "amber": "#F59E0B",
}
TYPE = {"audio": "#F59E0B", "text": "#34D399", "image": "#38BDF8", "embedding": "#A78BFA", "data": "#F472B6"}
CAT = {
    "Input": ("#94A3B8", "wave"), "Data": ("#94A3B8", "db"), "Agent": ("#A78BFA", "sparkle"),
    "Specialist": ("#F59E0B", "mic"), "Bridge": ("#2DD4BF", "plug"), "Base model": ("#34D399", "cpu"),
}

ICONS = {
    "wave": '<path d="M2 12h2M6 8v8M10 4v16M14 7v10M18 10v4M22 12h0"/>',
    "db": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5"/><path d="M3 12c0 1.7 4 3 9 3s9-1.3 9-3"/>',
    "sparkle": '<path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z"/><path d="M19 17l.7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7z"/>',
    "mic": '<rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 10v1a7 7 0 0 0 14 0v-1"/><path d="M12 18v4"/>',
    "plug": '<path d="M9 2v6M15 2v6"/><path d="M6 8h12v4a6 6 0 0 1-12 0z"/><path d="M12 18v4"/>',
    "cpu": '<rect x="5" y="5" width="14" height="14" rx="2"/><path d="M9 9h6v6H9zM9 2v3M15 2v3M9 19v3M15 19v3M19 9h3M19 15h3M2 9h3M2 15h3"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>', "minus": '<path d="M5 12h14"/>',
    "pointer": '<path d="m4 4 7 17 2.5-7.5L21 11z"/>',
    "move": '<path d="M5 9l-3 3 3 3M9 5l3-3 3 3M15 19l-3 3-3-3M19 9l3 3-3 3M2 12h20M12 2v20"/>',
    "chev": '<path d="m6 9 6 6 6-6"/>',
    "share": '<circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><path d="m8.6 13.5 6.8 4M15.4 6.5l-6.8 4"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>', "x": '<path d="M18 6 6 18M6 6l12 12"/>',
    "arrowR": '<path d="M5 12h14M13 6l6 6-6 6"/>', "arrowUp": '<path d="M12 19V5M5 12l7-7 7 7"/>',
    "clip": '<path d="m21.4 11-8.5 8.5a5.5 5.5 0 0 1-7.8-7.8l8.5-8.5a3.7 3.7 0 0 1 5.2 5.2l-8.5 8.5a1.8 1.8 0 0 1-2.6-2.6l7.8-7.8"/>',
    "image": '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-5-5L5 21"/>',
    "video": '<rect x="2" y="6" width="14" height="12" rx="2"/><path d="m22 8-6 4 6 4V8z"/>',
    "table": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18M9 3v18"/>',
    "msg": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "replay": '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>',
    "edit": '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/>',
    "download": '<path d="M12 3v12M7 10l5 5 5-5"/><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>',
    "globe": '<circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15 15 0 0 1 0 20M12 2a15 15 0 0 0 0 20"/>',
}


def icon(name, size=16, color=None, sw=2):
    color = color or C["text2"]
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')


def play_icon(size=12, color="#fff"):
    return f'<svg width="{size}" height="{size}" viewBox="0 0 24 24"><path d="M6 4l14 8-14 8z" fill="{color}"/></svg>'


def dots3(color):
    d = "".join(f'<circle cx="{x}" cy="12" r="1.8" fill="{color}"/>' for x in (5, 12, 19))
    return f'<svg width="16" height="16" viewBox="0 0 24 24">{d}</svg>'


def rgba(hex_, a):
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def waveform(w, h, color, bars=34, seed=7):
    vals, x = [], seed
    for i in range(bars):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        env = 0.35 + 0.65 * (1 if 6 < i < 13 or 19 < i < 26 else 0.3)
        vals.append(max(0.12, env * ((x % 100) / 100)))
    bw = w / bars
    rects = "".join(
        f'<rect x="{i * bw + bw * 0.2:.1f}" y="{(h - v * h) / 2:.1f}" width="{bw * 0.6:.1f}" height="{v * h:.1f}" rx="1" fill="{color}"/>'
        for i, v in enumerate(vals))
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">{rects}</svg>'


# ---------------- graph ----------------
NW = 184
HEAD, PAD, ROW, FOOT = 50, 6, 24, 30
COL = [24, 252, 480, 708, 936]


def badge(text, col):
    return f'<span class="nbadge" style="background:{col}">{text}</span>'


# id -> dict
NODES = {
    "disc": dict(title="Find a specialist", cat="Agent", sub="Agent step 3 · Linkup", x=COL[0], y=84,
                 ins=[], outs=[], body="disc", body_h=116, chips=["3 sources cited"], status=("done", "6.2 s")),
    "file": dict(title="dog_bark_17.wav", cat="Input", sub="Detected: audio · 5.0 s", x=COL[0], y=330,
                 ins=[], outs=[("audio", "audio")], body="wave", body_h=44, chips=["mono · 44.1 kHz"], status=None),
    "clap": dict(title="CLAP encoder", cat="Specialist", sub="Audio ↔ text · chosen", x=COL[1], y=330,
                 ins=[("audio", "audio")], outs=[("caption", "text"), ("embedding", "embedding")],
                 chips=["from model card"], status=("done", "120 ms")),
    "ex": dict(title="Labelled examples", cat="Data", sub="ESC-50 · sound events", x=COL[1], y=560,
               ins=[], outs=[("fit set", "data"), ("held out", "data")], chips=["1,600 fit", "400 test"], status=None),
    "textb": dict(title="Text bridge", cat="Bridge", sub="Caption into the prompt", x=COL[2], y=140,
                  ins=[("caption", "text")], outs=[("prompt", "text")], chips=["any model"], status=("score", "83.2%")),
    "vecb": dict(title="Vector bridge", cat="Bridge", sub="Closed-form map · layer 4", x=COL[2], y=330,
                 ins=[("embedding", "embedding"), ("examples", "data")], outs=[("states", "embedding")],
                 chips=["fit 0.8 s"], status=("best", "96.8%"), win=True),
    "verify": dict(title="Held-out check", cat="Agent", sub="Agent step 5 · 4 controls", x=COL[3], y=380,
                   ins=[("text", "text"), ("vector", "embedding"), ("held out", "data")], outs=[("install", "embedding")],
                   body="bars", body_h=104, chips=["n = 400", "5-way"], status=("done", "28.4 s")),
    "base": dict(title="Qwen3.5-2B", cat="Base model", sub="Local · open weights", x=COL[4], y=332,
                 ins=[("text · native", "text"), ("image · native", "image"), ("audio · new", "audio")],
                 outs=[("reply", "text")], chips=["M4 Pro"], status=("ready", "Ready"), sel=True),
}
NATIVE = {("base", "text · native"), ("base", "image · native")}

WIRES = [
    ("file", "audio", "clap", "audio"),
    ("clap", "caption", "textb", "caption"),
    ("clap", "embedding", "vecb", "embedding"),
    ("ex", "fit set", "vecb", "examples"),
    ("ex", "held out", "verify", "held out"),
    ("textb", "prompt", "verify", "text"),
    ("vecb", "states", "verify", "vector"),
    ("verify", "install", "base", "audio · new"),
]


def rows_of(n):
    return max(len(n["ins"]), len(n["outs"]))


def ports_h(n):
    r = rows_of(n)
    return PAD + r * ROW + PAD if r else 0


def node_h(n):
    return 1 + HEAD + ports_h(n) + n.get("body_h", 0) + FOOT + 1


def port_xy(nid, port, side):
    n = NODES[nid]
    plist = n["outs"] if side == "out" else n["ins"]
    i = [p[0] for p in plist].index(port)
    x = n["x"] + (NW if side == "out" else 0)
    y = n["y"] + 1 + HEAD + PAD + i * ROW + ROW // 2
    return x, y


def ptype(nid, port, side):
    n = NODES[nid]
    return dict(n["outs"] if side == "out" else n["ins"])[port]


def bez(x1, y1, x2, y2):
    dx = max(48, (x2 - x1) * 0.5)
    return f"M{x1} {y1} C{x1 + dx} {y1}, {x2 - dx} {y2}, {x2} {y2}"


CONNECTED = {(a, ap, "out") for a, ap, _, _ in WIRES} | {(b, bp, "in") for _, _, b, bp in WIRES}


def status_html(kind, label):
    if kind == "done":
        return f'<span class="st"><span class="sdot" style="background:{C["green"]}"></span>{label}</span>'
    if kind == "best":
        return (f'<span class="st" style="background:{rgba(C["green"], .14)};color:{C["green"]};padding:0 7px;height:18px;border-radius:999px">'
                f'{icon("check", 11, C["green"], 3)}{label}</span>')
    if kind == "score":
        return f'<span class="st" style="color:{C["text"]}">{label}</span>'
    return f'<span class="st"><span class="sdot" style="background:{C["green"]}"></span>{label}</span>'


def body_disc():
    rows = [
        ("check", C["green"], "CLAP", "audio ↔ text", C["text"]),
        ("minus", C["muted"], "BEATs", "audio only, no text side", C["text2"]),
        ("x", C["red"], "Whisper v3", "speech only", C["text2"]),
    ]
    h = "".join(f'<div class="drow">{icon(i, 12, c, 3)}<span style="color:{tc};font-weight:600">{n}</span>'
                f'<span style="color:{C["muted"]};overflow:hidden;white-space:nowrap;text-overflow:ellipsis">{r}</span></div>'
                for i, c, n, r, tc in rows)
    return (f'<div class="nbody" style="height:116px;padding:8px 10px">{h}'
            f'<div style="margin-top:6px;font-size:10px;color:{C["muted"]};display:flex;align-items:center;gap:5px">'
            f'{icon("db", 11, C["muted"])}Fallback: pre-wired registry</div></div>')


def body_bars():
    rows = [("Vector", 96.8, C["green"]), ("Lookup", 97.8, "#6B7280"), ("Text", 83.2, TYPE["text"]), ("Swapped", 19.0, C["red"])]
    h = "".join(
        f'<div class="brow"><span class="bl">{n}</span><div class="bt"><div style="width:{v}%;background:{c}"></div>'
        f'<span class="chance"></span></div><span class="bv">{v:.1f}</span></div>' for n, v, c in rows)
    return (f'<div class="nbody" style="height:104px;padding:8px 10px">{h}'
            f'<div style="font-size:9px;color:{C["muted"]};margin-top:2px">Dashed line: chance 20%</div></div>')


def body_wave():
    return f'<div class="nbody" style="height:44px;padding:6px 10px">{waveform(164, 32, rgba(TYPE["audio"], .85))}</div>'


def node_html(nid, n):
    col, ic = CAT[n["cat"]]
    h = node_h(n)
    if n.get("sel"):
        shadow = f"0 0 0 1px {C['accent']}, 0 0 0 6px {rgba(C['accent'], .22)}, 0 12px 32px rgba(0,0,0,.45)"
        border = C["accent"]
    elif n.get("win"):
        shadow = f"0 0 0 1px {C['green']}, 0 0 0 5px {rgba(C['green'], .14)}, 0 10px 28px rgba(0,0,0,.4)"
        border = C["green"]
    else:
        shadow, border = "0 8px 24px rgba(0,0,0,.35)", C["nodeBorder"]
    o = [f'<div class="node" style="left:{n["x"]}px;top:{n["y"]}px;height:{h}px;border-color:{border};box-shadow:{shadow}">']
    if n.get("win"):
        o.append(f'<span class="ntag" style="background:{C["green"]};color:#06281C">{icon("check", 10, "#06281C", 3.2)}Installed</span>')
    if n.get("sel"):
        o.append(f'<span class="ntag" style="background:{C["accent"]};color:#fff">Base model</span>')
    o.append(f'<div class="nhead"><div class="nicon" style="background:{rgba(col, .16)}">{icon(ic, 15, col)}</div>'
             f'<div class="nt"><div class="ntitle">{n["title"]}</div><div class="nsub">{n["sub"]}</div></div></div>')
    r = rows_of(n)
    if r:
        o.append('<div class="nports">')
        for i in range(r):
            o.append('<div class="prow">')
            if i < len(n["ins"]):
                pn = n["ins"][i][0]
                hl = ' style="color:#F8C35A;font-weight:600"' if pn == "audio · new" else ""
                o.append(f'<span class="pl"{hl}>{pn}</span>')
            if i < len(n["outs"]):
                o.append(f'<span class="pl" style="margin-left:auto;text-align:right">{n["outs"][i][0]}</span>')
            o.append('</div>')
        o.append('</div>')
    if n.get("body") == "disc":
        o.append(body_disc())
    elif n.get("body") == "bars":
        o.append(body_bars())
    elif n.get("body") == "wave":
        o.append(body_wave())
    for side, plist in (("in", n["ins"]), ("out", n["outs"])):
        for i, (pn, pt) in enumerate(plist):
            top = HEAD + PAD + i * ROW + ROW // 2 - 5
            pos = "left:-6px" if side == "in" else "right:-6px"
            conn = (nid, pn, side) in CONNECTED
            native = (nid, pn) in NATIVE
            fill = TYPE[pt] if conn or native else C["node"]
            bc = C["node"] if conn or native else TYPE[pt]
            o.append(f'<span class="pdot" style="{pos};top:{top}px;background:{fill};border-color:{bc}"></span>')
    st = status_html(*n["status"]) if n.get("status") else ""
    o.append('<div class="nfoot">' + "".join(f'<span class="chip">{c}</span>' for c in n["chips"])
             + f'<span style="margin-left:auto">{st}</span></div>')
    o.append('</div>')
    return "".join(o)


def canvas_html():
    o = []
    # group: candidate bridges
    o.append(f'<div class="group" style="left:464px;top:104px;width:216px;height:612px;border-color:{rgba("#2DD4BF", .35)};background:{rgba("#2DD4BF", .03)}">'
             f'<div class="glabel"><span class="sdot" style="background:#2DD4BF"></span>Candidate bridges</div>'
             f'<div class="skipped">{icon("image", 13, C["muted"])}<div><div style="color:{C["text2"]};font-weight:600">Image bridge</div>'
             f'<div>Skipped · untested on audio</div></div></div></div>')
    paths = []
    for a, ap, b, bp in WIRES:
        x1, y1 = port_xy(a, ap, "out")
        x2, y2 = port_xy(b, bp, "in")
        t = ptype(a, ap, "out")
        w = 3 if b == "base" else 2
        paths.append(f'<path d="{bez(x1, y1, x2, y2)}" stroke="{TYPE[t]}" stroke-width="{w}" fill="none" stroke-linecap="round" opacity=".9"/>')
    # agent action: discovery -> CLAP (dashed)
    d = NODES["disc"]
    sx, sy = d["x"] + NW // 2, d["y"] + node_h(d)
    tx, ty = NODES["clap"]["x"] + NW // 2, NODES["clap"]["y"]
    paths.append(f'<path d="M{sx} {sy} C{sx} {sy + 40}, {tx} {ty - 40}, {tx} {ty}" stroke="{TYPE["embedding"]}" stroke-width="1.6" stroke-dasharray="5 5" fill="none"/>')
    paths.append(f'<path d="M{tx - 5} {ty - 8} L{tx} {ty - 1} L{tx + 5} {ty - 8}" stroke="{TYPE["embedding"]}" stroke-width="1.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')
    o.append(f'<svg class="wires" width="{CW}" height="{CH}" viewBox="0 0 {CW} {CH}">{"".join(paths)}</svg>')
    mx, my = (sx + tx) // 2, (sy + ty) // 2
    o.append(f'<span class="elabel" style="left:{mx - 6}px;top:{my - 10}px">picks CLAP</span>')
    for nid, n in NODES.items():
        o.append(node_html(nid, n))

    # senses strip (top-left)
    senses = [("Text", "native", TYPE["text"], False), ("Image", "native", TYPE["image"], False),
              ("Audio", "vector · 96.8%", TYPE["audio"], True)]
    chips = "".join(
        f'<span class="schip{" new" if new else ""}"><span class="sdot" style="background:{c}"></span>'
        f'<b>{t}</b><span style="color:{C["muted"]}">{m}</span>'
        + (f'<span class="newtag">NEW</span>' if new else "") + '</span>' for t, m, c, new in senses)
    o.append(f'<div class="senses"><span style="color:{C["text2"]};font-weight:600;white-space:nowrap">Qwen3.5-2B can use</span>{chips}'
             '</div>')
    # toolbar (top-right)
    tools = [("pointer", True), ("move", False)]
    tb = "".join(f'<span class="tbtn{" on" if on else ""}">{icon(n, 15, "#fff" if on else C["text2"])}</span>' for n, on in tools)
    o.append(f'<div class="toolbar">{tb}<span class="tsep"></span>'
             f'<span class="tlbl">{icon("plus", 14, C["text"])}Add model</span>'
             f'<span class="tlbl soft">{icon("sparkle", 14, "#C9BDFF")}Let the agent choose</span><span class="tsep"></span>'
             f'<span class="tbtn">{icon("minus", 14)}</span><span style="color:{C["text"]};width:34px;text-align:center">90%</span>'
             f'<span class="tbtn">{icon("plus", 14)}</span></div>')
    # legend under toolbar
    leg = "".join(f'<span class="legi"><span class="sdot" style="background:{c}"></span>{t.capitalize()}</span>' for t, c in TYPE.items())
    leg += f'<span class="legi"><svg width="16" height="4"><path d="M0 2h16" stroke="{TYPE["embedding"]}" stroke-width="1.6" stroke-dasharray="4 3"/></svg>Agent action</span>'
    o.append(f'<div class="legend">{leg}</div>')

    # agent trace (bottom)
    steps = [
        ("Detect the sense", "dog_bark_17.wav is audio, 5.0 s, mono.", "sniff(file)", "0.1 s"),
        ("Check the base model", "Reads text and images. Open weights, so a vector bridge is allowed.", "caps(model)", "0.2 s"),
        ("Find a specialist", "Compared 3 cited sources. Chose CLAP. Registry kept as fallback.", "linkup.search(…)", "6.2 s"),
        ("Build bridges", "Fitted the vector map on 1,600 clips. Text bridge ready. Image skipped.", "fit(layer=4)", "1.9 s"),
        ("Verify on held-out", "400 clips against swapped, no-input, lookup and text controls.", "run_controls(n=400)", "28.4 s"),
        ("Install or refuse", "Installed the vector bridge. Text bridge kept for unseen sounds.", "install(vector)", "0.3 s"),
    ]
    cols = []
    for i, (t, dsc, code, tm) in enumerate(steps, 1):
        cols.append(f'''<div class="step">
          <div style="display:flex;align-items:center;gap:8px"><span class="snum">{icon("check", 11, "#06281C", 3.4)}</span>
            <span style="font-size:10px;color:{C['muted']};font-weight:600">STEP {i}</span><span style="margin-left:auto;font-size:10px;color:{C['text2']}">{tm}</span></div>
          <div class="stitle">{t}</div><div class="sdesc">{dsc}</div><div class="scode">{code}</div></div>''')
    o.append(f'''<div class="trace">
      <div class="thead">
        <span style="font-size:13px;font-weight:600;color:{C['text']}">Agent run</span>
        <span class="okpill">{icon("check", 11, C['green'], 3)}Sense installed</span>
        <span style="color:{C['text2']}">Give Qwen3.5-2B hearing · Claude orchestrator · 6 steps · 41 s · $0.06</span>
        <span class="btn ghost sm" style="margin-left:auto">{icon("replay", 13, C['text'])}Replay run</span>
        <span class="btn ghost sm">{icon("msg", 13, C['text'])}Full log</span>
      </div>
      <div class="steps">{"".join(cols)}</div></div>''')
    return "".join(o)


# ---------------- left: chat ----------------
def chat_html():
    return f'''<div class="chat">
      <div class="phead">
        <span style="font-size:14px;font-weight:600;color:{C['text']}">Chat</span>
        <span style="color:{C['muted']}">Sound check</span>
        <span class="iconbtn" style="margin-left:auto">{icon("edit", 14)}</span>
      </div>
      <div class="msgs">
        <div class="umsg">
          <div class="file">{icon("wave", 16, TYPE['audio'])}<div><div style="color:{C['text']};font-weight:600">dog_bark_17.wav</div>
            <div style="color:{C['muted']};font-size:11px">Audio · 5.0 s</div></div>{play_icon(10, C['text2'])}</div>
          <div>What is making this sound?</div>
        </div>
        <div class="amsg">
          <div class="who">{icon("sparkle", 13, "#C9BDFF")}<span style="color:#C9BDFF">Agent</span><span style="color:{C['muted']}">· 41 s</span></div>
          <div>Qwen3.5-2B can't hear audio yet, so I gave it a new sense before answering.</div>
          <div class="installed">
            <div style="display:flex;align-items:center;gap:6px">{icon("check", 12, C['green'], 3)}<b style="color:{C['text']}">Audio installed</b>
              <span style="margin-left:auto;color:{C['green']};font-weight:600">96.8%</span></div>
            <div style="color:{C['text2']};font-size:11px;margin-top:4px">Vector bridge via CLAP · 400 held-out clips · beats swapped audio (19.0%) and the text bridge (83.2%)</div>
            <div style="color:#C9BDFF;font-size:11px;margin-top:6px;font-weight:600">Open report card →</div>
          </div>
        </div>
        <div class="bmsg">
          <div class="who">{icon("cpu", 13, C['green'])}<span style="color:{C['text']}">Qwen3.5-2B</span><span class="via">heard via audio sense</span></div>
          <div>A dog barking.</div>
        </div>
        <div class="umsg"><div>Will it work on sounds outside ESC-50?</div></div>
        <div class="amsg">
          <div class="who">{icon("sparkle", 13, "#C9BDFF")}<span style="color:#C9BDFF">Agent</span></div>
          <div>Not with the vector bridge: it drops to 53.2% on sounds it was never fitted on. For those I'll route the clip through the text bridge instead, which scored 83.2%.</div>
        </div>
      </div>
      <div class="composer">
        <div style="font-size:11px;color:{C['muted']};font-weight:600;margin-bottom:8px">TRY NEXT</div>
        <div style="display:flex;flex-direction:column;gap:6px;margin-bottom:14px">
          <div class="sugg">{icon("wave", 14, TYPE['audio'])}<span>Teach it music genres it hasn't heard</span><span class="kind bad" style="margin-left:auto">expect refusal</span></div>
          <div class="sugg">{icon("cpu", 14, C['text2'])}<span>Same request with Claude as the base model</span><span class="kind" style="margin-left:auto">tool bridge</span></div>
        </div>
        <div class="drop">{icon("download", 18, C['text2'])}
          <div><div style="color:{C['text']};font-weight:600">Drop any file</div>
          <div style="color:{C['muted']};font-size:11px;margin-top:2px">The agent works out which sense the model needs</div></div>
          <div style="margin-left:auto;display:flex;gap:4px">
            <span class="fmt">{icon("wave", 13, TYPE['audio'])}</span><span class="fmt">{icon("image", 13, TYPE['image'])}</span>
            <span class="fmt">{icon("video", 13, C['text2'])}</span><span class="fmt">{icon("table", 13, C['text2'])}</span></div>
        </div>
        <div class="input">
          <span style="color:{C['muted']};flex:1">Ask Qwen3.5-2B anything…</span>
          <span class="iconbtn plain">{icon("clip", 15)}</span>
          <span class="send">{icon("arrowUp", 15, "#fff", 2.6)}</span>
        </div>
      </div>
    </div>'''


# ---------------- right: report card + senses ----------------
def report_html():
    bars = [("Vector bridge", "installed", 96.8, C["green"], True), ("Nearest-clip lookup", "control", 97.8, "#6B7280", False),
            ("Text bridge", "control", 83.2, TYPE["text"], False), ("Swapped audio", "control", 19.0, C["red"], False)]
    b = "".join(f'''<div class="rbar">
        <div style="display:flex;align-items:center;gap:6px"><span style="color:{C['text'] if win else C['text2']};font-weight:{600 if win else 500}">{n}</span>
          <span class="kind{' win' if win else ''}">{k}</span><span style="margin-left:auto;color:{C['text']};font-weight:600">{v:.1f}%</span></div>
        <div class="rtrack"><div style="width:{v}%;background:{c}"></div><span class="rchance"></span></div></div>''' for n, k, v, c, win in bars)
    senses = [
        ("Text", "Native", None, TYPE["text"], "native"),
        ("Image", "Native", None, TYPE["image"], "native"),
        ("Audio · trained sounds", "Vector bridge · CLAP", "96.8%", TYPE["audio"], "on"),
        ("Audio · new sounds", "Text bridge · fallback", "83.2%", TYPE["audio"], "on"),
        ("Music · unseen genres", "Vector bridge refused", "53.5%", C["red"], "refused"),
    ]
    srows = []
    for t, sub, score, col, state in senses:
        if state == "native":
            right = f'<span class="kind">native</span>'
        elif state == "on":
            right = f'<span style="color:{C["text"]};font-weight:600">{score}</span><span class="sw on"><span></span></span>'
        else:
            right = f'<span style="color:{C["red"]};font-weight:600">{score}</span><span class="kind bad">refused</span>'
        srows.append(f'<div class="srow"><span class="sdot" style="background:{col}"></span><div style="flex:1;min-width:0">'
                     f'<div style="color:{C["text"]};font-weight:500">{t}</div><div style="color:{C["muted"]};font-size:11px;margin-top:2px">{sub}</div></div>'
                     f'<div style="display:flex;align-items:center;gap:8px">{right}</div></div>')
    return f'''<div class="right">
      <div class="phead">
        <span style="font-size:14px;font-weight:600;color:{C['text']}">Report card</span>
        <span class="okpill">{icon("check", 11, C['green'], 3)}Installed</span>
        <span class="iconbtn" style="margin-left:auto">{dots3(C['text2'])}</span>
      </div>
      <div class="rsec">
        <div style="color:{C['text2']}">Audio → Qwen3.5-2B · vector bridge, layer 4</div>
        <div style="display:flex;align-items:baseline;gap:10px;margin-top:10px">
          <span style="font-size:40px;font-weight:700;color:{C['text']};letter-spacing:-.02em">96.8%</span>
          <span style="color:{C['text2']}">on 400 held-out clips</span></div>
        <div style="color:{C['muted']};font-size:11px;margin-top:2px">Five-option questions · chance is 20%</div>
        <div style="margin-top:16px">{b}</div>
        <div class="rnote">{icon("sparkle", 13, "#F8C35A")}<div>Matches nearest-clip lookup, so it passes on the label, not a description. Sounds it wasn't fitted on go to the text bridge.</div></div>
      </div>
      <div class="rsec" style="border-bottom:none">
        <div style="display:flex;align-items:center;margin-bottom:6px"><span style="font-size:12px;font-weight:600;color:{C['text']}">Senses on Qwen3.5-2B</span>
          <span style="margin-left:auto;color:{C['muted']};font-size:11px">4 active · 1 refused</span></div>
        {"".join(srows)}
      </div>
      <div class="rfoot">
        <span class="btn ghost">{icon("download", 13, C['text'])}Export report</span>
        <span class="btn primary" style="margin-left:auto">{icon("plus", 13, "#fff", 2.6)}Add a sense</span>
      </div>
    </div>'''


def topbar_html():
    team = [("Y", "#7C5CFF"), ("M", "#38BDF8"), ("J", "#34D399"), ("A", "#F59E0B")]
    av = "".join(f'<span class="avatar" style="background:{c};margin-left:{-5 if i else 0}px">{t}</span>' for i, (t, c) in enumerate(team))
    return f'''<div class="topbar">
      <div class="logo">{icon("sparkle", 16, "#fff", 2.2)}</div>
      <span style="font-weight:700;color:{C['text']};font-size:14px">Sensorium</span>
      <span style="color:{C['muted']}">/</span><span style="color:{C['text2']}">Hackathon team</span>
      <span style="color:{C['muted']}">/</span><span style="color:{C['text']};font-weight:500">Sound check</span>
      <div class="picker">
        <div class="nicon" style="width:26px;height:26px;background:{rgba(C['green'], .16)}">{icon("cpu", 14, C['green'])}</div>
        <div><div style="color:{C['text']};font-weight:600;font-size:13px">Qwen3.5-2B</div>
          <div style="color:{C['muted']};font-size:10px">Local · open weights</div></div>
        <div style="display:flex;gap:4px;margin-left:8px">
          <span class="mchip"><span class="sdot" style="background:{TYPE['text']}"></span>Text</span>
          <span class="mchip"><span class="sdot" style="background:{TYPE['image']}"></span>Image</span>
          <span class="mchip new"><span class="sdot" style="background:{TYPE['audio']}"></span>Audio</span></div>
        {icon("chev", 15, C['text2'])}
      </div>
      <div style="margin-left:auto;display:flex;align-items:center;gap:12px">
        <span style="display:flex;align-items:center;gap:7px;color:{C['text2']}">Replay mode<span class="sw"><span></span></span></span>
        <div style="display:flex">{av}</div>
        <span class="btn ghost">{icon("share", 13, C['text'])}Share</span>
        <span class="btn primary">{icon("plus", 13, "#fff", 2.6)}New sense</span>
      </div>
    </div>'''


CSS = f'''
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{WF}px;height:{HF}px;background:{C['bg']};font-family:Inter,"Helvetica Neue",Arial,sans-serif;font-size:12px;color:{C['text']};-webkit-font-smoothing:antialiased}}
b{{font-weight:600}}
.app{{position:relative;width:{WF}px;height:{HF}px;overflow:hidden;background:{C['bg']}}}
.sdot{{width:7px;height:7px;border-radius:50%;flex:none;display:inline-block}}
.btn{{height:32px;padding:0 12px;border-radius:8px;display:flex;align-items:center;gap:7px;font-size:12px;font-weight:600;white-space:nowrap}}
.btn.sm{{height:28px;padding:0 10px;font-size:11px}}
.btn.ghost{{background:{C['panel2']};border:1px solid {C['border']};color:{C['text']}}}
.btn.primary{{background:{C['accent']};color:#fff}}
.iconbtn{{width:28px;height:28px;border-radius:7px;display:flex;align-items:center;justify-content:center;background:{C['panel2']};border:1px solid {C['border']};flex:none}}
.iconbtn.plain{{background:transparent;border:none}}
.okpill{{height:20px;padding:0 8px;border-radius:999px;background:{rgba(C['green'], .13)};color:{C['green']};font-size:11px;font-weight:600;display:flex;align-items:center;gap:4px;white-space:nowrap}}
.sw{{position:relative;width:28px;height:16px;border-radius:999px;background:#2E3446;display:inline-block;flex:none}}
.sw span{{position:absolute;left:2px;top:2px;width:12px;height:12px;border-radius:50%;background:#fff}}
.sw.on{{background:{C['accent']}}}
.sw.on span{{left:14px}}
.kind{{height:16px;padding:0 6px;border-radius:4px;background:#232838;color:{C['text2']};font-size:10px;display:inline-flex;align-items:center;white-space:nowrap}}
.kind.win{{background:{rgba(C['green'], .14)};color:{C['green']}}}
.kind.bad{{background:{rgba(C['red'], .14)};color:{C['red']}}}

.topbar{{position:absolute;left:0;top:0;width:{WF}px;height:{TOP_H}px;display:flex;align-items:center;gap:10px;padding:0 16px;background:{C['panel']};border-bottom:1px solid {C['border']};font-size:13px}}
.logo{{width:28px;height:28px;border-radius:8px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#7C5CFF,#F59E0B)}}
.picker{{position:absolute;left:50%;transform:translateX(-50%);height:40px;display:flex;align-items:center;gap:10px;padding:0 12px 0 7px;border-radius:10px;background:{C['bg']};border:1px solid {C['border']}}}
.mchip{{height:20px;padding:0 7px;border-radius:999px;background:#1D2130;color:{C['text2']};font-size:10px;font-weight:500;display:flex;align-items:center;gap:5px}}
.mchip.new{{background:{rgba(C['amber'], .14)};color:#F8C35A;box-shadow:inset 0 0 0 1px {rgba(C['amber'], .4)}}}
.avatar{{width:26px;height:26px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#0E1015;border:2px solid {C['panel']}}}

.chat{{position:absolute;left:0;top:{TOP_H}px;width:{LEFT_W}px;height:{CH}px;background:{C['panel']};border-right:1px solid {C['border']}}}
.phead{{height:52px;display:flex;align-items:center;gap:10px;padding:0 16px;border-bottom:1px solid {C['border']}}}
.msgs{{padding:16px;display:flex;flex-direction:column;gap:14px;font-size:13px;line-height:19px}}
.umsg{{align-self:flex-end;max-width:300px;background:{C['accentSoft']};border-radius:12px 12px 4px 12px;padding:10px 12px;color:{C['text']};display:flex;flex-direction:column;gap:8px}}
.file{{display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:8px;background:{rgba('#000000', .25)};font-size:12px;line-height:16px}}
.amsg,.bmsg{{max-width:340px;color:{C['text']};display:flex;flex-direction:column;gap:6px}}
.who{{display:flex;align-items:center;gap:6px;font-size:11px;font-weight:600;line-height:14px}}
.via{{height:16px;padding:0 6px;border-radius:4px;background:{rgba(C['amber'], .14)};color:#F8C35A;font-size:10px;font-weight:500;display:inline-flex;align-items:center}}
.installed{{margin-top:2px;padding:10px 12px;border-radius:10px;border:1px solid {rgba(C['green'], .35)};background:{rgba(C['green'], .06)};font-size:12px;line-height:16px}}
.composer{{position:absolute;left:0;right:0;bottom:0;padding:14px 16px 16px;border-top:1px solid {C['border']};background:{C['panel']}}}
.drop{{height:62px;border-radius:10px;border:1.5px dashed #3A4158;display:flex;align-items:center;gap:12px;padding:0 12px;background:{C['bg']};font-size:12px}}
.sugg{{height:36px;border-radius:9px;border:1px solid {C['border']};background:{C['panel2']};display:flex;align-items:center;gap:9px;padding:0 10px;font-size:12px;color:{C['text']}}}
.fmt{{width:26px;height:26px;border-radius:7px;background:#1D2130;display:flex;align-items:center;justify-content:center}}
.input{{margin-top:10px;height:44px;border-radius:10px;border:1px solid {C['border']};background:{C['bg']};display:flex;align-items:center;gap:8px;padding:0 6px 0 12px;font-size:13px}}
.send{{width:32px;height:32px;border-radius:8px;background:{C['accent']};display:flex;align-items:center;justify-content:center}}

.canvas{{position:absolute;left:{CX}px;top:{TOP_H}px;width:{CW}px;height:{CH}px;overflow:hidden;background:{C['canvas']}}}
.wires{{position:absolute;left:0;top:0}}
.group{{position:absolute;border:1px dashed;border-radius:16px}}
.glabel{{position:absolute;left:14px;top:10px;display:flex;align-items:center;gap:7px;font-size:11px;font-weight:600;color:{C['text2']}}}
.skipped{{position:absolute;left:16px;right:16px;bottom:14px;height:44px;border-radius:9px;border:1px dashed #3A4158;display:flex;align-items:center;gap:9px;padding:0 10px;font-size:10px;color:{C['muted']};line-height:13px}}
.elabel{{position:absolute;height:20px;padding:0 7px;border-radius:999px;background:{C['panel']};border:1px solid {rgba(TYPE['embedding'], .45)};color:#C9BDFF;font-size:10px;font-weight:600;display:flex;align-items:center;white-space:nowrap}}
.node{{position:absolute;width:{NW}px;background:{C['node']};border:1px solid {C['nodeBorder']};border-radius:12px}}
.ntag{{position:absolute;right:10px;top:-10px;height:19px;padding:0 7px;border-radius:999px;font-size:10px;font-weight:700;display:flex;align-items:center;gap:3px;z-index:2}}
.nhead{{height:{HEAD}px;display:flex;align-items:center;gap:9px;padding:0 10px;border-bottom:1px solid #262B3B}}
.nicon{{width:28px;height:28px;border-radius:8px;display:flex;align-items:center;justify-content:center;flex:none}}
.nt{{flex:1;min-width:0}}
.ntitle{{font-size:12px;font-weight:600;color:{C['text']};white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.nsub{{font-size:10px;color:{C['muted']};margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.nports{{padding:{PAD}px 14px}}
.prow{{height:{ROW}px;display:flex;align-items:center}}
.pl{{font-size:11px;color:{C['text2']};white-space:nowrap}}
.pdot{{position:absolute;width:10px;height:10px;border-radius:50%;border:2px solid}}
.nbody{{border-top:1px solid #262B3B}}
.drow{{height:26px;display:flex;align-items:center;gap:6px;font-size:10px}}
.brow{{height:20px;display:flex;align-items:center;gap:6px;font-size:10px}}
.bl{{width:44px;color:{C['text2']}}}
.bt{{position:relative;flex:1;height:6px;border-radius:999px;background:#2A3042}}
.bt > div{{height:100%;border-radius:999px}}
.chance{{position:absolute;left:20%;top:-3px;width:0;height:12px;border-left:1px dashed {C['text2']}}}
.bv{{width:28px;text-align:right;color:{C['text']};font-weight:600}}
.nfoot{{height:{FOOT}px;border-top:1px solid #262B3B;display:flex;align-items:center;gap:5px;padding:0 10px}}
.chip{{height:18px;padding:0 6px;border-radius:5px;background:#232838;color:{C['text2']};font-size:10px;display:flex;align-items:center;white-space:nowrap}}
.st{{display:flex;align-items:center;gap:5px;font-size:10px;font-weight:600;color:{C['text2']};white-space:nowrap}}
.senses{{position:absolute;left:16px;top:16px;height:40px;padding:0 6px 0 12px;border-radius:10px;background:{C['panel']};border:1px solid {C['border']};display:flex;align-items:center;gap:6px}}
.schip{{height:28px;padding:0 10px;border-radius:7px;border:1px solid {C['border']};display:flex;align-items:center;gap:6px;font-size:11px;color:{C['text']};white-space:nowrap}}
.schip.new{{border-color:{rgba(C['amber'], .5)};background:{rgba(C['amber'], .08)}}}
.newtag{{height:15px;padding:0 4px;border-radius:3px;background:{C['amber']};color:#2A1A00;font-size:9px;font-weight:800;display:flex;align-items:center}}
.toolbar{{position:absolute;right:16px;top:16px;height:40px;padding:0 4px;border-radius:10px;background:{C['panel']};border:1px solid {C['border']};display:flex;align-items:center;gap:2px}}
.tbtn{{width:30px;height:30px;border-radius:7px;display:flex;align-items:center;justify-content:center}}
.tbtn.on{{background:{C['accent']}}}
.tsep{{width:1px;height:18px;background:{C['border']};margin:0 5px}}
.tlbl{{height:30px;padding:0 10px;border-radius:7px;display:flex;align-items:center;gap:6px;font-size:12px;font-weight:600;color:{C['text']};white-space:nowrap}}
.tlbl.soft{{background:{C['accentSoft']};color:#C9BDFF}}
.legend{{position:absolute;right:16px;top:66px;height:28px;padding:0 10px;border-radius:8px;background:{rgba('#141720', .9)};border:1px solid {C['border']};display:flex;align-items:center;gap:12px}}
.legi{{display:flex;align-items:center;gap:5px;font-size:10px;color:{C['text2']}}}
.trace{{position:absolute;left:16px;right:16px;bottom:16px;height:248px;border-radius:14px;background:{C['panel']};border:1px solid {C['border']};box-shadow:0 12px 32px rgba(0,0,0,.45)}}
.thead{{height:50px;display:flex;align-items:center;gap:10px;padding:0 14px;border-bottom:1px solid {C['border']}}}
.steps{{display:flex;gap:8px;padding:12px}}
.step{{flex:1;height:170px;border-radius:10px;background:{C['node']};border:1px solid #262B3B;padding:12px}}
.snum{{width:18px;height:18px;border-radius:50%;background:{C['green']};display:flex;align-items:center;justify-content:center}}
.stitle{{font-size:12px;font-weight:600;color:{C['text']};margin-top:10px}}
.sdesc{{font-size:11px;line-height:16px;color:{C['text2']};margin-top:5px;height:48px}}
.scode{{margin-top:10px;height:24px;border-radius:6px;background:{C['bg']};border:1px solid #262B3B;font-family:"JetBrains Mono",Menlo,monospace;font-size:10px;color:#C9BDFF;display:flex;align-items:center;padding:0 8px;white-space:nowrap;overflow:hidden}}

.right{{position:absolute;right:0;top:{TOP_H}px;width:{RIGHT_W}px;height:{CH}px;background:{C['panel']};border-left:1px solid {C['border']}}}
.rsec{{padding:16px 18px;border-bottom:1px solid {C['border']}}}
.rbar{{margin-bottom:12px}}
.rtrack{{position:relative;margin-top:6px;height:8px;border-radius:999px;background:#2A3042}}
.rtrack > div{{height:100%;border-radius:999px}}
.rchance{{position:absolute;left:20%;top:-4px;height:16px;border-left:1.5px dashed {C['text2']}}}
.rnote{{margin-top:4px;padding:10px 12px;border-radius:9px;background:{rgba(C['amber'], .08)};border:1px solid {rgba(C['amber'], .25)};display:flex;gap:8px;font-size:11px;line-height:16px;color:{C['text2']}}}
.srow{{height:52px;display:flex;align-items:center;gap:10px;border-bottom:1px solid #1F2433;font-size:12px}}
.srow:last-child{{border-bottom:none}}
.rfoot{{position:absolute;left:0;right:0;bottom:0;height:64px;display:flex;align-items:center;padding:0 18px;border-top:1px solid {C['border']}}}
'''


def main(path):
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>Sensorium – give a model a new sense</title><style>{CSS}</style></head>
<body><div class="app">
{topbar_html()}
{chat_html()}
<div class="canvas">{canvas_html()}</div>
{report_html()}
</div></body></html>'''
    with open(path, "w") as f:
        f.write(html)
    print(f"wrote {path} ({len(html)} bytes)")


if __name__ == "__main__":
    main(sys.argv[1])
