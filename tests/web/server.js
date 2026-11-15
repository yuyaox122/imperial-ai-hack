import http from "node:http";
import { readFile } from "node:fs/promises";

// Only serve the actual UI assets. Test-specific API answers are intercepted in Playwright.
const assets = new Map([
  ["/", ["../../web/index.html", "text/html"]],
  ["/static/app.js", ["../../web/app.js", "text/javascript"]],
  ["/static/brain.js", ["../../web/brain.js", "text/javascript"]],
  ["/static/style.css", ["../../web/style.css", "text/css"]],
  ["/static/favicon.svg", ["../../web/favicon.svg", "image/svg+xml"]],
]);
http.createServer(async (req, res) => {
  if (req.url === "/test-stream") {
    req.resume();
    res.setHeader("Content-Type", "text/event-stream");
    const frame = Buffer.from('event: used\r\ndata: []\r\n\r\nevent: token\r\ndata: {"text":"Dog 🐕"}\r\n\r\n');
    const cut = frame.indexOf(Buffer.from("🐕")) + 2;
    res.write(frame.subarray(0, cut));
    const token = setTimeout(() => res.write(frame.subarray(cut)), 50);
    const done = setTimeout(() => res.end('event: done\r\ndata: {"reply":"Dog 🐕"}\r\n\r\n'), 1500);
    res.on("close", () => { clearTimeout(token); clearTimeout(done); });
    return;
  }
  const asset = assets.get(req.url);
  if (!asset) { res.writeHead(404).end(); return; }
  res.setHeader("Content-Type", asset[1]);
  res.end(await readFile(new URL(asset[0], import.meta.url)));
}).listen(8766, "127.0.0.1");
