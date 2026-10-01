import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

VIDEO_PATH = "videos/match.mp4"
OUTPUT_PATH = "outputs/tracking_v3.mp4"

# Football-specific detector
model = YOLO("models/football-player-detection.pt")

# Roboflow model class IDs
BALL = 0
GOALKEEPER = 1
PLAYER = 2
REFEREE = 3

tracker = sv.ByteTrack(
    minimum_consecutive_frames=3
)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    result = model(
        frame,
        imgsz=1280,
        conf=0.25,
        device="mps",
        verbose=False
    )[0]

    detections = sv.Detections.from_ultralytics(result)

    # 第一版先保留 player + goalkeeper
    mask = np.isin(
        detections.class_id,
        [PLAYER, GOALKEEPER]
    )

    detections = detections[mask]

    # ByteTrack
    detections = tracker.update_with_detections(detections)

    if detections.tracker_id is not None:

        for box, track_id in zip(
            detections.xyxy,
            detections.tracker_id
        ):
            x1, y1, x2, y2 = map(int, box)

            # 小綠框
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # 小 ID
            cv2.putText(
                frame,
                f"#{int(track_id)}",
                (x1, max(y1 - 4, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

    writer.write(frame)

cap.release()
writer.release()

print("Saved:", OUTPUT_PATH)