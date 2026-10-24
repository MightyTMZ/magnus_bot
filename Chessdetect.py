import json
import os
import time
from collections import deque

import chess
import cv2
import supervision as sv
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

from board import build_board, draw_grid, draw_points, load_homography
from moves import find_move

game = chess.Board()
history = deque(maxlen=3)
moves_played = []
notice = None
seen_illegal = None

load_dotenv()

MODEL_ID = "chess-full-kntuv/1"

CONFIDENCE = 0.35
SEND_EVERY = 0.5

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"],
)

box_annotator = sv.BoxAnnotator()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detections = sv.Detections.empty()
last_sent = 0
checked_frame = False

H = load_homography()
with open("corners.json") as f:
    saved_corners = json.load(f)


def move_line(moves):
    parts = []
    for i, san in enumerate(moves):
        if i % 2 == 0:
            parts.append(f"{i // 2 + 1}. {san}")
        else:
            parts.append(san)
    return parts


def draw_moves(frame, moves, notice=None):
    """Draw the game's moves along the bottom of the frame. SAN already includes captures (exd4)."""
    h, w = frame.shape[:2]
    bar_h = 52
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
        (12, h - 16),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return frame


while True:
    ref, frame = cap.read()

    if not ref:
        break

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
        result = client.infer(frame, model_id=MODEL_ID)
        detections = sv.Detections.from_inference(result)
        detections = detections[detections.confidence >= CONFIDENCE]
        last_sent = now
        board = build_board(detections, H)
        history.append(board)
        if len(history) == 3 and history[0] == history[1] == history[2]:
            move, illegal = find_move(game, board)
            if move is not None:
                san = game.san(move)
                moves_played.append(san)
                print(san)
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

    annotated = draw_moves(annotated, moves_played, notice)

    cv2.imshow("Chess Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
