import json
import os
import subprocess
import time
from collections import deque

import chess
import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRSmall

from board import build_board, draw_grid, draw_points, load_homography, prep_frame, CORNERS_PATH
from moves import find_move, is_rook_sacrifice

CLIP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RookSacrifice.mp4")
# Spoken line in RookSacrifice.mp4. Playback reads that file and does not write another.
ROOK_TRANSCRIPT = "He sacrifices the rook!"

game = chess.Board()
history = deque(maxlen=3)
moves_played = []
notice = None
seen_illegal = None
quote = None

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "model", "weights_2.0.pt")

# Class ids from the Roboflow COCO export of chess-full-kntuv/1.
# Id 0 is Roboflow's superclass; "bishop" is a stray label with one annotation.
CLASS_NAMES = [
    "pieces",
    "bishop",
    "black-bishop",
    "black-king",
    "black-knight",
    "black-pawn",
    "black-queen",
    "black-rook",
    "white-bishop",
    "white-king",
    "white-knight",
    "white-pawn",
    "white-queen",
    "white-rook",
]

CONFIDENCE = 0.35
SEND_EVERY = 0.5

print(f"Loading RF-DETR Small weights from {WEIGHTS}")
model = RFDETRSmall(pretrain_weights=WEIGHTS)
try:
    model.optimize_for_inference()
except Exception as e:  # optional speed-up; plain inference still works
    print(f"optimize_for_inference skipped: {e}")


def detect(frame):
    """Run the local model on a BGR OpenCV frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = model.predict(rgb, threshold=CONFIDENCE)
    dets.data["class_name"] = np.array(
        [CLASS_NAMES[i] if 0 <= i < len(CLASS_NAMES) else str(i) for i in dets.class_id]
    )
    return dets


box_annotator = sv.BoxAnnotator()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detections = sv.Detections.empty()
last_sent = 0
checked_frame = False

H = load_homography()
with open(CORNERS_PATH) as f:
    saved_corners = json.load(f)


def move_line(moves):
    parts = []
    for i, san in enumerate(moves):
        if i % 2 == 0:
            parts.append(f"{i // 2 + 1}. {san}")
        else:
            parts.append(san)
    return parts


def play_rook_clip():
    """Play the clip's audio. ffplay reads the mp4 in place and writes nothing."""
    subprocess.Popen(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", CLIP],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def fit_text(text, font, scale, thickness, max_width):
    while scale > 0.35:
        width = cv2.getTextSize(text, font, scale, thickness)[0][0]
        if width <= max_width:
            return scale
        scale -= 0.05
    return scale


def draw_moves(frame, moves, notice=None, quote=None):
    """Draw the game's moves along the bottom of the frame. SAN already includes captures (exd4)."""
    h, w = frame.shape[:2]
    bar_h = 84 if quote and not notice else 52
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    if notice:
        cv2.putText(
            frame,
            notice,
            (12, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    parts = move_line(moves) or ["waiting for a move"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    move_y = h - 46 if quote else h - 16
    while len(parts) > 1:
        text = "   ".join(parts)
        width = cv2.getTextSize(text, font, scale, thickness)[0][0]
        if width <= w - 24:
            break
        parts = parts[1:]
    text = "   ".join(parts)
    cv2.putText(
        frame,
        text,
        (12, move_y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    if quote:
        q_scale = fit_text(quote, font, 0.6, 2, w - 24)
        cv2.putText(
            frame,
            quote,
            (12, h - 14),
            font,
            q_scale,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
    return frame


f = 0

while True:
    f += 1
    print(f"Frame {f}")
    ref, frame = cap.read()

    if not ref:
        break

    # Same rotate + crop as Calibrate.py, so corners.json and the detections
    # share one coordinate space.
    frame = prep_frame(frame)

    if not checked_frame:
        h, w = frame.shape[:2]
        print(f"Frame size: {w}x{h}")
        xs = [c[0] for c in saved_corners]
        ys = [c[1] for c in saved_corners]
        if max(xs) >= w or max(ys) >= h or max(xs) < w * 0.5 or max(ys) < h * 0.5:
            print("corners.json does not match this frame. Run Calibrate.py and click the four corners again.")
        checked_frame = True

    now = time.time()
    if now - last_sent > SEND_EVERY:
        detections = detect(frame)
        last_sent = now
        board = build_board(detections, H)
        history.append(board)
        if len(history) == 3 and history[0] == history[1] == history[2]:
            move, illegal = find_move(game, board)
            if move is not None:
                san = game.san(move)
                moves_played.append(san)
                print(san)
                if is_rook_sacrifice(game, move):
                    quote = ROOK_TRANSCRIPT
                    print(quote)
                    try:
                        play_rook_clip()
                    except OSError:
                        print("Could not play RookSacrifice.mp4")
                else:
                    quote = None
                game.push(move)
                history.clear()
                notice = None
                seen_illegal = None
            elif illegal:
                notice = "illegal move"
                if board != seen_illegal:
                    print("illegal move")
                    seen_illegal = board
            else:
                notice = None
                seen_illegal = None

    annotated = box_annotator.annotate(frame.copy(), detections)
    annotated = draw_grid(annotated, H)
    if len(detections) and "class_name" in detections.data:
        annotated = draw_points(annotated, detections, H)

    for (x, y) in saved_corners:
        cv2.circle(annotated, (x, y), 8, (255, 0, 255), -1)

    annotated = draw_moves(annotated, moves_played, notice, quote)

    cv2.imshow("Chess Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
