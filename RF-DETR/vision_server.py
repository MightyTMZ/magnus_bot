"""Serve the live board over localhost so the arm controller can read it.

The camera and RF-DETR (torch 2.14 + CUDA) live in RF-DETR/.venv, but the arm controller runs
in the lerobot venv (torch<2.12), so the two talk over HTTP instead of sharing a process. The
model runs on every frame; the server keeps the last few readings and serves them as JSON.

    .venv/Scripts/python.exe vision_server.py          # headless
    .venv/Scripts/python.exe vision_server.py --show   # plus the annotated debug window

Endpoints:
    GET /health                    {"ok", "frame_id", "fps", "camera_ok"}
    GET /board                     newest reading, possibly mid-motion (see "stable")
    GET /board?fresh=1&timeout=8   waits until the last --stable-frames readings agree and were
                                   ALL captured after the request arrived; 504 if not in time

Only one process can hold the camera: stop Chessdetect.py before starting this.
"""

import argparse
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
from board import build_pieces, draw_grid, draw_points, load_homography, prep_frame
from detector import detect, load_model

MAX_FRESH_TIMEOUT = 60.0


def color_of(name):
    if name.startswith("white"):
        return "W"
    if name.startswith("black"):
        return "B"
    return "?"  # Roboflow's "pieces" superclass or a stray label: occupied, color unknown


class BoardState:
    """The last few readings, shared between the capture loop and the HTTP handlers."""

    def __init__(self, stable_frames):
        self.stable_frames = stable_frames
        self.cond = threading.Condition()
        self.recent = deque(maxlen=stable_frames)  # (frame_id, timestamp, pieces, occupancy)
        self.frame_id = 0
        self.fps = 0.0
        self.camera_ok = False

    def publish(self, pieces):
        occupancy = {sq: color_of(name) for sq, name in pieces.items()}
        with self.cond:
            self.frame_id += 1
            self.recent.append((self.frame_id, time.time(), pieces, occupancy))
            self.cond.notify_all()

    def _stable(self, after_frame=0):
        """Newest reading if the last N agree on occupancy and all came after ``after_frame``.

        Caller holds ``self.cond``. Occupancy (not class names) decides stability, so a piece
        flickering between, say, white-rook and white-queen does not block a reading.
        """
        if len(self.recent) < self.stable_frames or self.recent[0][0] <= after_frame:
            return None
        first = self.recent[0][3]
        if any(r[3] != first for r in self.recent):
            return None
        return self.recent[-1]

    def snapshot(self):
        with self.cond:
            if not self.recent:
                return None
            return self._payload(self.recent[-1], stable=self._stable() is not None)

    def wait_fresh(self, timeout):
        deadline = time.monotonic() + timeout
        with self.cond:
            start = self.frame_id
            while True:
                hit = self._stable(after_frame=start)
                if hit is not None:
                    return self._payload(hit, stable=True)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.cond.wait(remaining)

    def _payload(self, reading, stable):
        frame_id, timestamp, pieces, occupancy = reading
        return {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "stable": stable,
            "stable_frames": self.stable_frames,
            "pieces": pieces,
            "occupancy": occupancy,
        }


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if url.path == "/health":
                self._send(200, {"ok": True, "frame_id": state.frame_id,
                                 "fps": round(state.fps, 1), "camera_ok": state.camera_ok})
            elif url.path == "/board":
                self._board(query)
            else:
                self._send(404, {"error": f"unknown path {url.path}"})

        def _board(self, query):
            if query.get("fresh", ["0"])[0].lower() not in ("1", "true", "yes"):
                payload = state.snapshot()
                if payload is None:
                    self._send(503, {"error": "no frames yet", "camera_ok": state.camera_ok})
                else:
                    self._send(200, payload)
                return
            try:
                timeout = min(float(query.get("timeout", ["8"])[0]), MAX_FRESH_TIMEOUT)
            except ValueError:
                self._send(400, {"error": "timeout must be a number of seconds"})
                return
            payload = state.wait_fresh(timeout)
            if payload is None:
                self._send(504, {"error": f"no stable board within {timeout:g}s",
                                 "camera_ok": state.camera_ok, "latest": state.snapshot()})
            else:
                self._send(200, payload)

        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):  # keep the console for the capture loop
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Serve the live board over localhost.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--camera", type=int, default=0, help="DirectShow index (iVCam is 0)")
    parser.add_argument("--stable-frames", type=int, default=5,
                        help="identical consecutive readings needed for a stable board")
    parser.add_argument("--show", action="store_true", help="open the annotated debug window")
    args = parser.parse_args()

    model = load_model()
    H = load_homography()
    state = BoardState(args.stable_frames)

    try:
        server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    except OSError as e:
        raise SystemExit(f"cannot listen on {args.host}:{args.port} ({e}); is another server running?") from e
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Serving the board on http://{args.host}:{args.port}/board  (Ctrl+C, or q in the window, to stop)")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    box_annotator = None
    if args.show:
        import supervision as sv

        box_annotator = sv.BoxAnnotator()

    failures = 0
    last = time.monotonic()
    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                state.camera_ok = False
                if failures % 50 == 0:
                    print(f"camera {args.camera} read failed; is iVCam running?")
                failures += 1
                time.sleep(0.1)
                continue
            failures = 0
            state.camera_ok = True

            frame = prep_frame(raw)
            dets = detect(model, frame)
            state.publish(build_pieces(dets, H))

            now = time.monotonic()
            if now > last:
                state.fps = 0.9 * state.fps + 0.1 / (now - last)
            last = now

            if args.show:
                annotated = box_annotator.annotate(frame.copy(), dets)
                annotated = draw_grid(annotated, H)
                if len(dets):
                    annotated = draw_points(annotated, dets, H)
                snap = state.snapshot()
                stable = bool(snap and snap["stable"])
                cv2.putText(annotated, f"{'STABLE' if stable else 'moving'}  {state.fps:.0f} fps",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 200, 0) if stable else (0, 165, 255), 2, cv2.LINE_AA)
                cv2.imshow("vision server", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        server.shutdown()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
