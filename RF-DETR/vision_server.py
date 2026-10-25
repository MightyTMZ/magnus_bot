"""Serve the live board over localhost so the arm controller can read it.

The camera and RF-DETR (torch 2.14 + CUDA) live in RF-DETR/.venv, but the arm controller runs
in the lerobot venv (torch<2.12), so the two talk over HTTP instead of sharing a process. The
model runs on every frame; the server keeps the last few readings and serves them as JSON.

It also finds the green-painted fingertip (``--marker-hsv``) and reports where it is on the board,
so the reach probe can measure where the arm really went. And it refuses to report a board unless
the checkerboard is actually in view: iVCam's "waiting for connection" screen would otherwise read
as an empty board.

    .venv/Scripts/python.exe vision_server.py          # headless
    .venv/Scripts/python.exe vision_server.py --show   # plus the annotated debug window
    then open http://127.0.0.1:8765/ in a browser for a live view

Endpoints:
    GET /                          live view (annotated frame + status), refreshed in the browser
    GET /frame.jpg                 the latest annotated frame
    GET /health                    {"ok", "frame_id", "fps", "camera_ok", "board_visible", "contrast"}
    GET /board                     newest reading, possibly mid-motion (see "stable")
    GET /board?fresh=1&timeout=8   waits until the last --stable-frames readings agree and were
                                   ALL captured after the request arrived; 504 if not in time

Positions ("marker") are in board squares: x toward the h-file, y toward rank 8, origin at a1's
outer corner. Only one process can hold the camera: stop Chessdetect.py before starting this.
"""

import argparse
import json
import os
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from board import (
    CORNERS_PATH,
    CROP_PATH,
    board_to_pixel,
    build_pieces,
    draw_grid,
    draw_points,
    load_crop,
    load_homography,
    pixel_to_board,
    prep_frame,
)
from detector import detect, load_model

MAX_FRESH_TIMEOUT = 60.0
# OpenCV HSV (H 0-180). Measured 2026-10-24 on the rig: the painted tips are yellow-green (H 32-47,
# S 25-127, V 154-231); the board's dark squares are blue-green (H 83, S 80) and the gripper's
# yellow band is H 27-29, so hue separates them. Re-measure if the paint or lighting changes.
DEFAULT_MARKER_HSV = "33,55,120,55,255,255"
MARKER_GROUP_PX = 80  # green blobs this close to the biggest one count as the same fingertip
OPEN_KERNEL = np.ones((3, 3), np.uint8)
VIEW_PAGE = b"""<!doctype html><meta charset=utf-8><title>magnus_bot camera</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px system-ui,sans-serif}
img{display:block;margin:0 auto;max-width:100vw;max-height:90vh}p{text-align:center}</style>
<img id=f src="/frame.jpg"><p id=s>connecting...</p>
<script>
const f=document.getElementById('f'), s=document.getElementById('s');
setInterval(()=>{f.src='/frame.jpg?t='+Date.now()},250);
setInterval(async()=>{try{const h=await(await fetch('/health')).json();
s.textContent=(h.board_visible?'board visible':'BOARD NOT VISIBLE')+' | '+h.fps+' fps | frame '+h.frame_id
+' | contrast '+h.contrast}catch(e){s.textContent='server not reachable'}},500);
</script>"""


def color_of(name):
    if name.startswith("white"):
        return "W"
    if name.startswith("black"):
        return "B"
    return "?"  # Roboflow's "pieces" superclass or a stray label: occupied, color unknown


def parse_hsv(text):
    values = [int(v) for v in text.split(",")]
    if len(values) != 6:
        raise ValueError(f"--marker-hsv needs 6 numbers (h,s,v low then high), got {text!r}")
    return np.array(values[:3], np.uint8), np.array(values[3:], np.uint8)


def find_marker(frame, H, lo, hi, min_area):
    """The green fingertip, or None: its lowest visible point (nearest the board, so the least
    parallax) in pixels and in board squares, and the blob's area in pixels."""
    mask = cv2.morphologyEx(cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lo, hi),
                            cv2.MORPH_OPEN, OPEN_KERNEL)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[biggest, cv2.CC_STAT_AREA] < min_area:
        return None
    # Both jaw tips are painted: pool the blobs near the biggest (one blob when the jaws are shut).
    group = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area
             and np.hypot(*(centroids[i] - centroids[biggest])) <= MARKER_GROUP_PX]
    area = int(sum(stats[i, cv2.CC_STAT_AREA] for i in group))
    ys, xs = np.nonzero(np.isin(labels, group))
    # The camera sits beyond rank 1 looking down the board, so lower points image lower: the
    # bottom of the blob is the part of the tip closest to the board surface.
    y = float(np.percentile(ys, 95))
    x = float(np.median(xs[ys >= np.percentile(ys, 85)]))
    bx, by = pixel_to_board(x, y, H)
    return {"px": [round(x, 1), round(y, 1)], "board": [round(bx, 4), round(by, 4)], "area": area}


class BoardCheck:
    """Is the checkerboard really in view? Compares dark and light square brightness.

    a1 is dark. Pieces cover some squares, so each color uses a low/high percentile of its squares
    rather than the mean; a uniform image (a placeholder screen, a covered lens) scores ~0.
    """

    def __init__(self, H, shape):
        h, w = shape[:2]
        self.centers = []  # (x, y, dark)
        for f in range(8):
            for r in range(8):
                x, y = board_to_pixel(f + 0.5, r + 0.5, H)
                if 3 <= x < w - 3 and 3 <= y < h - 3:
                    self.centers.append((int(x), int(y), (f + r) % 2 == 0))

    def contrast(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dark, light = [], []
        for x, y, is_dark in self.centers:
            (dark if is_dark else light).append(float(gray[y - 3:y + 4, x - 3:x + 4].mean()))
        if len(dark) < 8 or len(light) < 8:
            return 0.0
        return round(float(abs(np.percentile(light, 75) - np.percentile(dark, 25))) / 255.0, 3)


class Calibration:
    """crop.json + corners.json as ONE pair, reloaded whenever Calibrate.py rewrites them.

    Calibrate.py saves the crop first and the corners only after the four clicks, so while
    crop.json is newer than corners.json a recalibration is half done (or was abandoned):
    ``ok`` is then False and nothing should be published, because a new crop with old corners
    maps every pixel to the wrong square.
    """

    def __init__(self):
        self.version = 0
        self._stamp = None
        self.reload_if_changed()

    @staticmethod
    def _mtimes():
        crop = os.path.getmtime(CROP_PATH) if os.path.exists(CROP_PATH) else 0.0
        return crop, os.path.getmtime(CORNERS_PATH)

    def reload_if_changed(self):
        """Reload both files if either changed on disk; True if something was reloaded."""
        stamp = self._mtimes()
        if stamp == self._stamp:
            return False
        self._stamp = stamp
        self.crop, self.H = load_crop(), load_homography()
        self.ok = stamp[1] >= stamp[0]
        self.version += 1
        return True


@dataclass
class Reading:
    frame_id: int
    timestamp: float
    pieces: dict
    occupancy: dict
    marker: dict | None


class BoardState:
    """The last few readings, shared between the capture loop and the HTTP handlers."""

    def __init__(self, stable_frames):
        self.stable_frames = stable_frames
        self.cond = threading.Condition()
        self.recent = deque(maxlen=stable_frames)
        self.frame_id = 0
        self.fps = 0.0
        self.camera_ok = False
        self.board_visible = False
        self.calibration_ok = True
        self.contrast = 0.0
        self.latest_view = None  # (frame, detections, marker, H) for /frame.jpg

    def reset(self):
        """Forget recent readings (they were mapped with a calibration that just changed)."""
        with self.cond:
            self.recent.clear()

    def publish(self, pieces, marker=None):
        occupancy = {sq: color_of(name) for sq, name in pieces.items()}
        with self.cond:
            self.frame_id += 1
            self.recent.append(Reading(self.frame_id, time.time(), pieces, occupancy, marker))
            self.cond.notify_all()

    def _stable(self, after_frame=0):
        """Newest reading if the last N agree on occupancy and all came after ``after_frame``.

        Caller holds ``self.cond``. Occupancy (not class names) decides stability, so a piece
        flickering between, say, white-rook and white-queen does not block a reading.
        """
        if len(self.recent) < self.stable_frames or self.recent[0].frame_id <= after_frame:
            return None
        first = self.recent[0].occupancy
        if any(r.occupancy != first for r in self.recent):
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
        """Caller holds ``self.cond``. The marker is the median over the recent window."""
        seen = [r.marker["board"] for r in self.recent if r.marker]
        marker = None
        if seen:
            marker = [round(statistics.median(p[0] for p in seen), 4),
                      round(statistics.median(p[1] for p in seen), 4)]
        return {
            "frame_id": reading.frame_id,
            "timestamp": reading.timestamp,
            "stable": stable,
            "stable_frames": self.stable_frames,
            "board_visible": self.board_visible,
            "pieces": reading.pieces,
            "occupancy": reading.occupancy,
            "marker": marker,
            "marker_seen": len(seen),
        }


def render(view, box_annotator):
    """The latest frame with boxes, the board grid, piece points and the fingertip marker."""
    frame, dets, marker, H = view
    out = box_annotator.annotate(frame.copy(), dets) if box_annotator is not None else frame.copy()
    out = draw_grid(out, H)
    if len(dets):
        out = draw_points(out, dets, H)
    if marker:
        x, y = (int(v) for v in marker["px"])
        cv2.circle(out, (x, y), 9, (255, 0, 255), 2)
        bx, by = marker["board"]
        cv2.putText(out, f"tip ({bx:.2f}, {by:.2f}) sq", (x + 12, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)
    return out


def make_handler(state, box_annotator=None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                self._send(200, VIEW_PAGE, "text/html; charset=utf-8")
            elif url.path == "/frame.jpg":
                self._frame()
            elif url.path == "/health":
                self._json(200, {"ok": True, "frame_id": state.frame_id, "fps": round(state.fps, 1),
                                 "camera_ok": state.camera_ok, "board_visible": state.board_visible,
                                 "calibration_ok": state.calibration_ok, "contrast": state.contrast})
            elif url.path == "/board":
                self._board(parse_qs(url.query))
            else:
                self._json(404, {"error": f"unknown path {url.path}"})

        def _frame(self):
            view = state.latest_view
            if view is None:
                self._json(503, {"error": "no frames yet"})
                return
            ok, jpg = cv2.imencode(".jpg", render(view, box_annotator), [cv2.IMWRITE_JPEG_QUALITY, 85])
            self._send(200, jpg.tobytes(), "image/jpeg")

        def _board(self, query):
            if query.get("fresh", ["0"])[0].lower() not in ("1", "true", "yes"):
                payload = state.snapshot()
                if payload is None:
                    self._json(503, {"error": "no frames yet", "camera_ok": state.camera_ok})
                else:
                    self._json(200, payload)
                return
            try:
                timeout = min(float(query.get("timeout", ["8"])[0]), MAX_FRESH_TIMEOUT)
            except ValueError:
                self._json(400, {"error": "timeout must be a number of seconds"})
                return
            payload = state.wait_fresh(timeout)
            if payload is None:
                if not state.calibration_ok:
                    why = "recalibration half done (crop.json newer than corners.json): finish Calibrate.py"
                elif not state.board_visible:
                    why = "board not visible (camera disconnected or blocked?)"
                else:
                    why = "no stable board"
                self._json(504, {"error": f"{why} within {timeout:g}s", "camera_ok": state.camera_ok,
                                 "board_visible": state.board_visible, "latest": state.snapshot()})
            else:
                self._json(200, payload)

        def _json(self, code, body):
            self._send(code, json.dumps(body).encode(), "application/json")

        def _send(self, code, data, content_type):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
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
    parser.add_argument("--marker-hsv", default=DEFAULT_MARKER_HSV,
                        help="fingertip color range, OpenCV HSV: h,s,v low then h,s,v high")
    parser.add_argument("--marker-min-area", type=int, default=30, help="smallest fingertip blob, px")
    parser.add_argument("--min-contrast", type=float, default=0.08,
                        help="dark/light square contrast below which the board counts as not visible")
    parser.add_argument("--show", action="store_true", help="open the annotated debug window")
    args = parser.parse_args()
    marker_lo, marker_hi = parse_hsv(args.marker_hsv)

    import supervision as sv

    model = load_model()
    calib = Calibration()
    state = BoardState(args.stable_frames)
    box_annotator = sv.BoxAnnotator()

    try:
        server = ThreadingHTTPServer((args.host, args.port), make_handler(state, box_annotator))
    except OSError as e:
        raise SystemExit(f"cannot listen on {args.host}:{args.port} ({e}); is another server running?") from e
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Serving the board on http://{args.host}:{args.port}/  (Ctrl+C, or q in the window, to stop)")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    check = None
    failures = 0
    hidden_since = None
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

            if calib.reload_if_changed():
                check = None
                state.reset()
                state.calibration_ok = calib.ok
                print("calibration reloaded" if calib.ok else
                      "recalibration half done (crop.json newer than corners.json): not reporting a board")
            H = calib.H
            frame = prep_frame(raw, calib.crop)
            if check is None:
                check = BoardCheck(H, frame.shape)
            state.contrast = check.contrast(frame)
            state.board_visible = bool(state.contrast >= args.min_contrast)  # plain bool: JSON-safe
            dets = detect(model, frame)
            marker = find_marker(frame, H, marker_lo, marker_hi, args.marker_min_area)
            state.latest_view = (frame, dets, marker, H)
            if state.board_visible and calib.ok:
                hidden_since = None
                state.publish(build_pieces(dets, H), marker)
            elif hidden_since is None:
                hidden_since = time.monotonic()
                print(f"board not visible (contrast {state.contrast}); is the phone connected to iVCam?")

            now = time.monotonic()
            if now > last:
                state.fps = 0.9 * state.fps + 0.1 / (now - last)
            last = now

            if args.show:
                annotated = render(state.latest_view, box_annotator)
                label = "board visible" if state.board_visible else "BOARD NOT VISIBLE"
                cv2.putText(annotated, f"{label}  {state.fps:.0f} fps", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 200, 0) if state.board_visible else (0, 0, 255), 2, cv2.LINE_AA)
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
