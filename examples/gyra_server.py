"""Tiny local Gyra server: POST /v1/decide with {"state": ..., "questions": {...}} (the arguments of Laya's
system_one) and get Laya's answer object back. Standard library only (plus laya).

    python gyra_server.py --model RomanRG008/gyra --port 8765
    curl -s localhost:8765/v1/decide -d '{"state": "Command: ls\\nOutput:\\n(no output)",
      "questions": {"ok": {"type": "noul", "instructions": "Judging from the output, did the command succeed?"}}}'

Binds 127.0.0.1 by default. There is no authentication: do not expose it to a network.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

import laya


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="RomanRG008/gyra")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default=None, help="cpu or cuda (default: Laya's choice)")
    a = ap.parse_args()
    model, lock = (laya.load(a.model, device=a.device) if a.device else laya.load(a.model)), Lock()
    model.system_one("warm up", {"q": {"type": "noul", "instructions": "Is this a test?"}})   # fail at start, not on a request

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path.rstrip("/") != "/v1/decide":
                return self._send(404, {"error": "use POST /v1/decide"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                with lock:
                    out = model.system_one(body["state"], body["questions"])
                self._send(200, out)
            except (KeyError, ValueError, TypeError) as e:
                self._send(422, {"error": str(e)})
            except Exception as e:  # keep serving; the caller sees why
                self._send(500, {"error": "%s: %s" % (type(e).__name__, e)})

        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    print("Gyra listening on http://%s:%d/v1/decide" % (a.host, a.port), flush=True)
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
