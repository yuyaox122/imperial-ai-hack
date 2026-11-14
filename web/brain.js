/* Sense brain: the base model drawn as a field of neurons. Each data colour (--wire-* in style.css) has a
   region; a region lights when the model has a sense of that kind. app.js decides the levels from API
   data, this file only draws them. Exposes window.SenseBrain = { mount, regionFor, REGIONS }. */
(function () {
  "use strict";

  const REGIONS = {
    frontal:    { role: "Language", token: "text" },
    occipital:  { role: "Vision",   token: "image" },
    temporal:   { role: "Hearing",  token: "audio" },
    parietal:   { role: "Data",     token: "data" },
    cerebellum: { role: "Other",    token: "embedding" },
    stem:       { role: "Input",    token: "input" },
  };
  const TOKEN_REGION = { text: "frontal", image: "occipital", audio: "temporal", data: "parietal", input: "stem" };
  const regionFor = (token) => TOKEN_REGION[token] || "cerebellum";

  // Side view, facing left; coordinates in a unit square.
  const inE = (x, y, cx, cy, rx, ry) => ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1;
  function regionAt(x, y) {
    if (inE(x, y, 0.5, 0.6, 0.27, 0.12) && y > 0.53) return "temporal";
    if (inE(x, y, 0.5, 0.42, 0.42, 0.31)) return x < 0.37 ? "frontal" : x > 0.74 ? "occipital" : "parietal";
    if (inE(x, y, 0.76, 0.73, 0.15, 0.1)) return "cerebellum";
    if (y > 0.72 && inE(x, y, 0.62, 0.86, 0.045, 0.13)) return "stem";
    return null;
  }
  const LABEL_AT = { frontal: [0.2, 0.3], parietal: [0.55, 0.2], occipital: [0.87, 0.4], temporal: [0.47, 0.64], cerebellum: [0.8, 0.77], stem: [0.62, 0.95] };

  function seeded(seed) { return () => (seed = (seed * 16807) % 2147483647) / 2147483647; }

  function build(n) {
    const r = seeded(7), pts = [];
    while (pts.length < n) {
      const x = r(), y = r(), g = regionAt(x, y);
      if (g) pts.push({ x, y, g, s: 0.6 + r() * 1.1, ph: r() * 6.28 });
    }
    const links = [];
    for (let i = 0; i < pts.length; i++) {
      const near = [];
      for (let j = 0; j < pts.length; j++) {
        if (i === j) continue;
        const d = (pts[i].x - pts[j].x) ** 2 + (pts[i].y - pts[j].y) ** 2;
        if (d < 0.0028) near.push([d, j]);
      }
      near.sort((a, b) => a[0] - b[0]);
      for (const [, j] of near.slice(0, 3)) if (i < j) links.push([i, j]);
    }
    return { pts, links };
  }

  function cssVar(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }
  function rgba(hex, a) {
    const m = /^#?([0-9a-f]{6})$/i.exec(hex);
    if (!m) return hex;
    const n = parseInt(m[1], 16);
    return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
  }

  // Brightness per level; "off" regions stay grey.
  const LEVEL = { off: 0.3, refused: 0.5, native: 0.6, running: 0.85, installed: 1 };

  function mount(canvas, opts) {
    opts = opts || {};
    const ctx = canvas.getContext("2d");
    const { pts, links } = build(opts.n || 700);
    const still = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let levels = {}, colours = {}, hover = null, W = 0, H = 0, SX = 0, SY = 0, ox = 0, oy = 0;
    const pulses = [];

    function readColours() {
      const off = cssVar("--rule-dark", "#616161"), err = cssVar("--app-error", "#e5484d");
      for (const g in REGIONS) {
        const lv = levels[g] || "off";
        colours[g] = lv === "off" ? off : lv === "refused" ? err : cssVar(`--wire-${REGIONS[g].token}`, "#9e9e9e");
      }
    }
    function size() {
      const b = canvas.getBoundingClientRect(), d = window.devicePixelRatio || 1;
      if (!b.width || !b.height) return;
      canvas.width = b.width * d; canvas.height = b.height * d;
      ctx.setTransform(d, 0, 0, d, 0, 0);
      W = b.width; H = b.height;
      // Fill wide cards: stretch sideways up to 1.3x rather than leave empty margins.
      SY = Math.min(H / 1.02, W / 1.08); SX = Math.min(W / 1.08, SY * 1.3); ox = (W - SX) / 2; oy = (H - SY) / 2;
    }
    const at = (p) => [ox + p.x * SX, oy + p.y * SY];
    const bright = (g) => LEVEL[levels[g] || "off"] + (g === hover ? 0.25 : 0);

    function frame(now) {
      requestAnimationFrame(frame);
      if (document.hidden || !W) return;
      const t = now / 1000;
      ctx.clearRect(0, 0, W, H);
      ctx.lineWidth = 1;
      for (const [i, j] of links) {
        const a = pts[i], b = pts[j], [x1, y1] = at(a), [x2, y2] = at(b);
        ctx.strokeStyle = rgba(colours[a.g], 0.05 + 0.13 * bright(a.g));
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      }
      for (const p of pts) {
        const [x, y] = at(p), lv = levels[p.g] || "off", b = bright(p.g);
        const flicker = still ? 1 : lv === "running" ? 0.35 + 0.65 * Math.abs(Math.sin(t * 3 + p.ph)) : lv === "installed" ? 0.65 + 0.35 * Math.sin(t * 2 + p.ph) : 1;
        ctx.fillStyle = rgba(colours[p.g], Math.min(1, (0.2 + 0.7 * b) * flicker));
        ctx.beginPath(); ctx.arc(x, y, p.s * (0.9 + 0.6 * b), 0, 6.283); ctx.fill();
      }
      if (!still) {
        if (Math.random() < 0.6) {
          const l = links[(Math.random() * links.length) | 0], lv = levels[pts[l[0]].g];
          if (lv === "installed" || lv === "running") pulses.push({ l, t: 0 });
        }
        for (let k = pulses.length - 1; k >= 0; k--) {
          const q = pulses[k]; q.t += 0.035;
          if (q.t > 1) { pulses.splice(k, 1); continue; }
          const [x1, y1] = at(pts[q.l[0]]), [x2, y2] = at(pts[q.l[1]]);
          ctx.fillStyle = "#ffffff";
          ctx.fillRect(x1 + (x2 - x1) * q.t - 1, y1 + (y2 - y1) * q.t - 1, 2, 2);
        }
      }
      ctx.font = `600 10px ${cssVar("--font-mono", "monospace")}`;
      ctx.textAlign = "center";
      for (const g in LABEL_AT) {
        if ((levels[g] || "off") === "off" && g !== hover) continue;
        ctx.fillStyle = rgba(colours[g], g === hover ? 1 : 0.8);
        ctx.fillText(REGIONS[g].role.toUpperCase(), ox + LABEL_AT[g][0] * SX, oy + LABEL_AT[g][1] * SY);
      }
    }

    function pick(e) {
      const b = canvas.getBoundingClientRect();
      return regionAt((e.clientX - b.left - ox) / SX, (e.clientY - b.top - oy) / SY);
    }
    canvas.addEventListener("mousemove", (e) => {
      const g = pick(e);
      canvas.style.cursor = g && opts.clickable && opts.clickable(g) ? "pointer" : "default";
      hover = g;
      if (opts.onHover) opts.onHover(g, e);
    });
    canvas.addEventListener("mouseleave", () => { hover = null; if (opts.onHover) opts.onHover(null); });
    canvas.addEventListener("click", (e) => { const g = pick(e); if (g && opts.onClick) opts.onClick(g); });
    if (window.ResizeObserver) new ResizeObserver(size).observe(canvas); else window.addEventListener("resize", size);
    size(); readColours(); requestAnimationFrame(frame);

    return {
      /** levels: { region: "native" | "installed" | "running" | "refused" }; missing regions are off. */
      update(next) { levels = next || {}; readColours(); },
    };
  }

  window.SenseBrain = { mount, regionFor, REGIONS };
})();
