import cv2
import numpy as np
import supervision as sv

from ultralytics import YOLO
from sklearn.cluster import KMeans
from collections import defaultdict

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration


VIDEO_PATH = "videos/match.mp4"
OUTPUT_PATH = "outputs/radar_v1.mp4"

PLAYER_MODEL_PATH = "models/football-player-detection.pt"
PITCH_MODEL_PATH = "models/football-pitch-detection.pt"

BALL = 0
GOALKEEPER = 1
PLAYER = 2
REFEREE = 3

CONFIG = SoccerPitchConfiguration()

player_model = YOLO(PLAYER_MODEL_PATH)
pitch_model = YOLO(PITCH_MODEL_PATH)


# =========================================================
# Jersey feature
# =========================================================

def get_jersey_feature(frame, box):

    x1, y1, x2, y2 = map(int, box)

    h = y2 - y1
    w = x2 - x1

    crop = frame[
        y1 + int(h * 0.15):y1 + int(h * 0.55),
        x1 + int(w * 0.20):x2 - int(w * 0.20)
    ]

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # remove grass
    green_mask = cv2.inRange(
        hsv,
        np.array([30, 40, 30]),
        np.array([95, 255, 255])
    )

    valid = green_mask == 0

    if valid.sum() < 10:
        return None

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

    return np.median(lab[valid], axis=0)


# =========================================================
# PASS 1 - Learn team colors
# =========================================================

print("[1/2] Learning team colors...")

features = []

cap = cv2.VideoCapture(VIDEO_PATH)

frame_idx = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    if frame_idx % 10 != 0:
        frame_idx += 1
        continue

    result = player_model(
        frame,
        imgsz=1280,
        conf=0.25,
        device="mps",
        verbose=False
    )[0]

    detections = sv.Detections.from_ultralytics(result)

    players = detections[
        detections.class_id == PLAYER
    ]

    for box in players.xyxy:

        feature = get_jersey_feature(frame, box)

        if feature is not None:
            features.append(feature)

    frame_idx += 1

cap.release()

features = np.array(features)

kmeans = KMeans(
    n_clusters=2,
    n_init=20,
    random_state=0
)

kmeans.fit(features)

centers = kmeans.cluster_centers_

# brighter jersey = A
bright_cluster = np.argmax(centers[:, 0])
dark_cluster = 1 - bright_cluster

TEAM_A_CENTER = centers[bright_cluster]
TEAM_B_CENTER = centers[dark_cluster]


def classify_team(feature):

    if feature is None:
        return None

    da = np.linalg.norm(feature - TEAM_A_CENTER)
    db = np.linalg.norm(feature - TEAM_B_CENTER)

    return 0 if da < db else 1


# =========================================================
# PASS 2
# Tracking + Pitch + Homography + Radar
# =========================================================

print("[2/2] Building football radar...")

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

# 如果某幀 keypoints 失敗，沿用上一幀
last_H = None

pitch_vertices = np.array(
    CONFIG.vertices,
    dtype=np.float32
)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    annotated = frame.copy()

    # =====================================================
    # 1. Pitch Keypoints
    # =====================================================

    pitch_result = pitch_model(
        frame,
        device="mps",
        verbose=False
    )[0]

    keypoints = sv.KeyPoints.from_ultralytics(
        pitch_result
    )

    H = None

    if len(keypoints.xy) > 0:

        pts = keypoints.xy[0]

        if keypoints.confidence is not None:
            conf = keypoints.confidence[0]
        else:
            conf = np.ones(len(pts))

        valid = (
            (pts[:, 0] > 1) &
            (pts[:, 1] > 1) &
            (conf >= 0.5)
        )

        source = pts[valid].astype(np.float32)
        target = pitch_vertices[valid].astype(np.float32)

        # 至少四點才能做 Homography
        if len(source) >= 4:

            H_candidate, inliers = cv2.findHomography(
                source,
                target,
                cv2.RANSAC,
                5.0
            )

            if H_candidate is not None:

                # 至少 4 個 RANSAC inliers
                if inliers is None or inliers.sum() >= 4:
                    H = H_candidate
                    last_H = H_candidate

    # keypoint failure → fallback
    if H is None:
        H = last_H


    # =====================================================
    # 2. Player Detection
    # =====================================================

    result = player_model(
        frame,
        imgsz=1280,
        conf=0.25,
        device="mps",
        verbose=False
    )[0]

    detections = sv.Detections.from_ultralytics(
        result
    )

    keep = np.isin(
        detections.class_id,
        [PLAYER, GOALKEEPER]
    )

    detections = detections[keep]

    detections = tracker.update_with_detections(
        detections
    )


    # 儲存目前 frame 給 Radar 的資料
    radar_positions = []
    radar_teams = []


    if detections.tracker_id is not None:

        for box, track_id, class_id in zip(
            detections.xyxy,
            detections.tracker_id,
            detections.class_id
        ):

            track_id = int(track_id)
            class_id = int(class_id)

            x1, y1, x2, y2 = map(int, box)

            # =================================================
            # Team
            # =================================================

            if class_id == GOALKEEPER:

                team_id = 2

                box_color = (0, 255, 255)
                label = f"GK #{track_id}"

            else:

                feature = get_jersey_feature(
                    frame,
                    box
                )

                team = classify_team(feature)

                if team is not None:
                    team_votes[track_id][team] += 1

                team_id = int(
                    np.argmax(team_votes[track_id])
                )

                if team_id == 0:

                    box_color = (255, 80, 40)
                    label = f"A #{track_id}"

                else:

                    box_color = (40, 40, 255)
                    label = f"B #{track_id}"


            # =================================================
            # Draw player
            # =================================================

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )

            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 4, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                box_color,
                1,
                cv2.LINE_AA
            )


            # =================================================
            # Bottom-center → pitch coordinate
            # =================================================

            if H is not None:

                px = (x1 + x2) / 2
                py = y2

                image_point = np.array(
                    [[[px, py]]],
                    dtype=np.float32
                )

                pitch_point = cv2.perspectiveTransform(
                    image_point,
                    H
                )[0][0]

                radar_positions.append(
                    pitch_point
                )

                radar_teams.append(
                    team_id
                )


    # =====================================================
    # 3. Draw Radar
    # =====================================================

    if H is not None and len(radar_positions) > 0:

        radar_positions = np.array(
            radar_positions
        )

        radar_teams = np.array(
            radar_teams
        )

        radar = draw_pitch(
            config=CONFIG
        )

        # Team A
        radar = draw_points_on_pitch(
            config=CONFIG,
            xy=radar_positions[radar_teams == 0],
            face_color=sv.Color.from_hex("#00BFFF"),
            radius=18,
            pitch=radar
        )

        # Team B
        radar = draw_points_on_pitch(
            config=CONFIG,
            xy=radar_positions[radar_teams == 1],
            face_color=sv.Color.from_hex("#FF6347"),
            radius=18,
            pitch=radar
        )

        # GK
        radar = draw_points_on_pitch(
            config=CONFIG,
            xy=radar_positions[radar_teams == 2],
            face_color=sv.Color.from_hex("#FFD700"),
            radius=18,
            pitch=radar
        )


        # =================================================
        # Resize radar
        # =================================================

        radar = sv.resize_image(
            radar,
            (width // 3, height // 3)
        )

        radar_h, radar_w, _ = radar.shape

        rect = sv.Rect(
            x=width - radar_w - 20,
            y=height - radar_h - 20,
            width=radar_w,
            height=radar_h
        )

        annotated = sv.draw_image(
            annotated,
            radar,
            opacity=0.78,
            rect=rect
        )


    writer.write(annotated)


cap.release()
writer.release()

print("")
print("Done!")
print("Saved:", OUTPUT_PATH)