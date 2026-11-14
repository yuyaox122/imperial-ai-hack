"""Private stdlib HTTP worker. Run as a script, supervised by research._http.

Only this process performs DNS, TLS and HTTP I/O. Its parent owns the wall-clock
budget and kills/reaps it on expiry. stdout carries bounded response bytes;
stderr carries a small JSON failure record. Request credentials arrive on stdin.
"""
from __future__ import annotations

import http.client
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


class ResearchUnavailable(RuntimeError):
    """No usable research answer is available."""


class _RedirectGuard(urllib.request.HTTPRedirectHandler):
    """Keep urllib's redirect rules, without draining unbounded redirect bodies."""

    def http_error_302(self, req, fp, code, msg, headers):
        # Each request uses a new connection. There is no need to drain its body
        # for reuse. urllib's default handler would call fp.read() without a cap.
        fp.close()
        return super().http_error_302(req, io.BytesIO(), code, msg, headers)

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        try:
            old, new = urllib.parse.urlsplit(req.full_url), urllib.parse.urlsplit(newurl)
            same_place = (old.hostname, old.port) == (new.hostname, new.port)
        except ValueError:
            raise ResearchUnavailable("refused a redirect to an invalid URL") from None
        if new.scheme not in ("http", "https"):
            raise ResearchUnavailable(f"refused a redirect to a {new.scheme or 'schemeless'} URL")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and (not same_place or (old.scheme, new.scheme) == ("https", "http")):
            redirected.remove_header("Authorization")
        return redirected


def fetch(method, url, *, body=None, headers=None, timeout, max_bytes, sink):
    """Stream at most max_bytes + one overflow sentinel to sink.

    Per-read timeouts are a secondary guard. The supervising process enforces
    the deadline even inside DNS, headers, redirects and chunk framing reads.
    """
    deadline = time.monotonic() + timeout
    data = json.dumps(body).encode() if body is not None else None
    all_headers = {"User-Agent": "sense-agent/0.1", "Accept": "application/json, */*"}
    if data is not None:
        all_headers["Content-Type"] = "application/json"
    all_headers.update(headers or {})
    request = urllib.request.Request(url, data=data, headers=all_headers, method=method)
    received = 0
    try:
        with urllib.request.build_opener(_RedirectGuard).open(request, timeout=timeout) as response:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ResearchUnavailable("response deadline reached")
                # read1 may close fp and its socket on the final nonempty read
                # of a Content-Length response. Never retain that socket.
                sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
                if sock is not None:
                    sock.settimeout(remaining)
                chunk = response.read1(min(64 * 1024, max_bytes + 1 - received))
                received += len(chunk)
                sink(chunk)
                if received > max_bytes:
                    raise ResearchUnavailable(f"the answer is over the {max_bytes:,} byte limit")
                if time.monotonic() >= deadline:
                    raise ResearchUnavailable("response deadline reached")
                if not chunk:
                    # HTTPResponse.read1 does not itself reject early EOF for
                    # a Content-Length body; do not publish a truncated file.
                    if getattr(response, "length", None):
                        raise http.client.IncompleteRead(b"", response.length)
                    break
    except Exception as exc:
        exc.bytes_received = received + len(getattr(exc, "partial", b""))
        # A failed read may have consumed bytes internally before raising.
        exc.transfer_incomplete = not isinstance(exc, urllib.error.HTTPError)
        raise


def main():
    try:
        request = json.load(sys.stdin)
        def emit(chunk):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        fetch(**request, sink=emit)
    except Exception as exc:
        record = {"message": str(exc)[:1000], "bytes_received": getattr(exc, "bytes_received", 0),
                  "transfer_incomplete": getattr(exc, "transfer_incomplete", True)}
        if isinstance(exc, urllib.error.HTTPError):
            record["status"] = exc.code
            exc.close()  # Error bodies are never an unbudgeted transfer.
        sys.stderr.write(json.dumps(record))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
