import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from sklearn.cluster import KMeans
from collections import defaultdict

VIDEO_PATH = "videos/match.mp4"
OUTPUT_PATH = "outputs/team_tracking.mp4"
MODEL_PATH = "models/football-player-detection.pt"

BALL = 0
GOALKEEPER = 1
PLAYER = 2
REFEREE = 3

model = YOLO(MODEL_PATH)

print("Model classes:", model.names)


def get_jersey_feature(frame, box):
    """
    擷取球員上半身球衣區域，
    回傳 Lab 色彩空間的平均顏色。
    """
    x1, y1, x2, y2 = map(int, box)

    h = y2 - y1
    w = x2 - x1

    # torso / jersey 區域
    crop = frame[
        y1 + int(h * 0.15): y1 + int(h * 0.55),
        x1 + int(w * 0.20): x2 - int(w * 0.20)
    ]

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # 去除草地綠色像素
    green_mask = cv2.inRange(
        hsv,
        np.array([30, 40, 30]),
        np.array([95, 255, 255])
    )

    valid_mask = green_mask == 0

    if valid_mask.sum() < 10:
        return None

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

    pixels = lab[valid_mask]

    return np.median(pixels, axis=0)


# =========================================================
# PASS 1：蒐集球衣顏色
# =========================================================

print("\n[1/2] Collecting jersey colors...")

features = []

cap = cv2.VideoCapture(VIDEO_PATH)

frame_id = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 每 5 frame 抽一次，減少運算
    if frame_id % 5 != 0:
        frame_id += 1
        continue

    result = model(
        frame,
        imgsz=1280,
        conf=0.25,
        device="mps",
        verbose=False
    )[0]

    detections = sv.Detections.from_ultralytics(result)

    if detections.class_id is not None:

        mask = detections.class_id == PLAYER
        players = detections[mask]

        for box in players.xyxy:

            feature = get_jersey_feature(frame, box)

            if feature is not None:
                features.append(feature)

    frame_id += 1

cap.release()

features = np.array(features)

print("Collected:", len(features), "jersey samples")

if len(features) < 20:
    raise RuntimeError("Not enough player samples for team clustering.")


# =========================================================
# KMeans：分成兩隊
# =========================================================

kmeans = KMeans(
    n_clusters=2,
    random_state=0,
    n_init=20
)

kmeans.fit(features)

centers = kmeans.cluster_centers_

# Lab 第一維 L = brightness
# 讓較亮的一隊固定成 Team A
bright_cluster = np.argmax(centers[:, 0])
dark_cluster = 1 - bright_cluster

TEAM_A_CENTER = centers[bright_cluster]
TEAM_B_CENTER = centers[dark_cluster]

print("Team A center:", TEAM_A_CENTER)
print("Team B center:", TEAM_B_CENTER)


def classify_team(feature):

    if feature is None:
        return None

    dist_a = np.linalg.norm(feature - TEAM_A_CENTER)
    dist_b = np.linalg.norm(feature - TEAM_B_CENTER)

    return 0 if dist_a < dist_b else 1


# =========================================================
# PASS 2：Tracking + Team Classification
# =========================================================

print("\n[2/2] Running tracking + team classification...")

tracker = sv.ByteTrack(
    minimum_consecutive_frames=3
)

team_votes = defaultdict(lambda: [0, 0])

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

    # Player + Goalkeeper
    mask = np.isin(
        detections.class_id,
        [PLAYER, GOALKEEPER]
    )

    detections = detections[mask]

    detections = tracker.update_with_detections(detections)

    if detections.tracker_id is not None:

        for box, track_id, class_id in zip(
            detections.xyxy,
            detections.tracker_id,
            detections.class_id
        ):

            track_id = int(track_id)
            class_id = int(class_id)

            x1, y1, x2, y2 = map(int, box)

            # -------------------------
            # Goalkeeper
            # -------------------------

            if class_id == GOALKEEPER:

                color = (0, 255, 255)
                label = f"GK #{track_id}"

            # -------------------------
            # Player
            # -------------------------

            else:

                feature = get_jersey_feature(frame, box)

                team = classify_team(feature)

                if team is not None:
                    team_votes[track_id][team] += 1

                # 使用歷史多數決
                votes = team_votes[track_id]

                stable_team = int(np.argmax(votes))

                if stable_team == 0:

                    color = (255, 80, 40)
                    label = f"A #{track_id}"

                else:

                    color = (40, 40, 255)
                    label = f"B #{track_id}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 4, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA
            )

    writer.write(frame)

cap.release()
writer.release()

print("\nDone!")
print("Saved:", OUTPUT_PATH)