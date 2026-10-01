import cv2
import supervision as sv
from ultralytics import YOLO

VIDEO_PATH = "videos/match.mp4"
OUTPUT_PATH = "outputs/pitch_debug_v2.mp4"
MODEL_PATH = "models/football-pitch-detection.pt"

model = YOLO(MODEL_PATH)

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
        device="mps",
        verbose=False
    )[0]

    keypoints = sv.KeyPoints.from_ultralytics(result)

    annotated = frame.copy()

    if len(keypoints.xy) > 0:

        points = keypoints.xy[0]

        if keypoints.confidence is not None:
            confs = keypoints.confidence[0]
        else:
            confs = [1.0] * len(points)

        for i, ((x, y), conf) in enumerate(zip(points, confs)):

            # 只保留可信點
            if conf < 0.5:
                continue

            if x <= 1 or y <= 1:
                continue

            x = int(x)
            y = int(y)

            cv2.circle(
                annotated,
                (x, y),
                5,
                (255, 0, 255),
                -1
            )

            cv2.putText(
                annotated,
                f"{i} {conf:.2f}",
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 0, 255),
                1,
                cv2.LINE_AA
            )

    writer.write(annotated)

cap.release()
writer.release()

print("Saved:", OUTPUT_PATH)