import cv2
import json


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
corners = []
printed = False

def on_click(event, x,y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
        corners.append([x,y])
        print(f"Added corner {len(corners)}: {x}, {y}")
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', on_click)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    if not printed:
        h, w = frame.shape[:2]
        print(f"Calibrate frame size: {w}x{h}")
        printed = True
    for (x, y) in corners:
        cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)

    if len(corners) == 4:
        break

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if len(corners) == 4:
    with open('corners.json', 'w') as f:
        json.dump(corners, f)
    print(f"Corners saved to corners.json: {corners}")
else:
    print("Calibration cancelled, need exactly 4 corners.")