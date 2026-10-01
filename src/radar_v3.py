import os
import csv
import cv2
import numpy as np
import supervision as sv

from ultralytics import YOLO
from sklearn.cluster import KMeans
from collections import defaultdict

from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch
)
from sports.configs.soccer import SoccerPitchConfiguration


# ============================================================
# Paths
# ============================================================

VIDEO_PATH = "videos/match.mp4"

OUTPUT_VIDEO_PATH = "outputs/radar_v3.mp4"
OUTPUT_CSV_PATH = "outputs/tracking_data.csv"

PLAYER_MODEL_PATH = "models/football-player-detection.pt"
PITCH_MODEL_PATH = "models/football-pitch-detection.pt"


# ============================================================
# Classes
# ============================================================

BALL = 0
GOALKEEPER = 1
PLAYER = 2
REFEREE = 3


# ============================================================
# Models / Config
# ============================================================

CONFIG = SoccerPitchConfiguration()

player_model = YOLO(PLAYER_MODEL_PATH)
pitch_model = YOLO(PITCH_MODEL_PATH)

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Parameters
# ============================================================

KP_CONF_THRESHOLD = 0.60

KP_ALPHA = 0.60

KP_MAX_GAP = 10

PLAYER_ALPHA = 0.45

RANSAC_THRESHOLD = 250.0

MIN_KEYPOINTS = 6

MIN_INLIER_RATIO = 0.60

MAX_H_JUMP = 500.0

PLAYER_IMGSZ = 1280

PLAYER_CONF = 0.25


# ============================================================
# Jersey feature
# ============================================================

def get_jersey_feature(frame, box):

    x1, y1, x2, y2 = map(int, box)

    h = y2 - y1
    w = x2 - x1

    if h <= 0 or w <= 0:
        return None

    xa = max(
        0,
        x1 + int(w * 0.20)
    )

    xb = min(
        frame.shape[1],
        x2 - int(w * 0.20)
    )

    ya = max(
        0,
        y1 + int(h * 0.15)
    )

    yb = min(
        frame.shape[0],
        y1 + int(h * 0.55)
    )

    crop = frame[
        ya:yb,
        xa:xb
    ]

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV
    )

    green_mask = cv2.inRange(
        hsv,
        np.array([30, 40, 30]),
        np.array([95, 255, 255])
    )

    valid = green_mask == 0

    if valid.sum() < 10:
        return None

    lab = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2LAB
    )

    feature = np.median(
        lab[valid],
        axis=0
    )

    return feature.astype(
        np.float32
    )


# ============================================================
# PASS 1
# Learn team colors
# ============================================================

print("")
print("======================================")
print("[1/2] Learning team colors")
print("======================================")

features = []

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    raise RuntimeError(
        f"Cannot open video: {VIDEO_PATH}"
    )


frame_idx = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    # 每 10 frame 抽一次
    if frame_idx % 10 != 0:

        frame_idx += 1

        continue


    result = player_model(
        frame,
        imgsz=PLAYER_IMGSZ,
        conf=PLAYER_CONF,
        device="mps",
        verbose=False
    )[0]


    detections = (
        sv.Detections.from_ultralytics(
            result
        )
    )


    if detections.class_id is not None:

        players = detections[
            detections.class_id == PLAYER
        ]


        for box in players.xyxy:

            feature = (
                get_jersey_feature(
                    frame,
                    box
                )
            )

            if feature is not None:

                features.append(
                    feature
                )


    frame_idx += 1


cap.release()


features = np.asarray(
    features,
    dtype=np.float32
)


print(
    f"Collected {len(features)} jersey samples"
)


if len(features) < 20:

    raise RuntimeError(
        "Not enough jersey samples."
    )


# ============================================================
# KMeans team clustering
# ============================================================

kmeans = KMeans(
    n_clusters=2,
    n_init=20,
    random_state=0
)

kmeans.fit(
    features
)


centers = (
    kmeans.cluster_centers_
)


# Lab L channel = brightness
bright_cluster = int(
    np.argmax(
        centers[:, 0]
    )
)

dark_cluster = (
    1 - bright_cluster
)


TEAM_A_CENTER = (
    centers[bright_cluster]
)

TEAM_B_CENTER = (
    centers[dark_cluster]
)


print(
    "Team A center:",
    TEAM_A_CENTER
)

print(
    "Team B center:",
    TEAM_B_CENTER
)


def classify_team(feature):

    if feature is None:
        return None


    dist_a = np.linalg.norm(
        feature - TEAM_A_CENTER
    )

    dist_b = np.linalg.norm(
        feature - TEAM_B_CENTER
    )


    return 0 if dist_a < dist_b else 1


# ============================================================
# PASS 2
# Tracking + Radar + CSV
# ============================================================

print("")
print("======================================")
print("[2/2] Tracking + Radar + CSV")
print("======================================")


tracker = sv.ByteTrack(
    minimum_consecutive_frames=3
)


team_votes = defaultdict(
    lambda: [0, 0]
)


smoothed_player_positions = {}

smoothed_kps = {}

kp_last_seen = {}


# 所有要寫入 CSV 的資料
tracking_rows = []


cap = cv2.VideoCapture(
    VIDEO_PATH
)


if not cap.isOpened():

    raise RuntimeError(
        f"Cannot open video: {VIDEO_PATH}"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)


writer = cv2.VideoWriter(
    OUTPUT_VIDEO_PATH,
    cv2.VideoWriter_fourcc(
        *"mp4v"
    ),
    fps,
    (
        width,
        height
    )
)


if not writer.isOpened():

    raise RuntimeError(
        f"Cannot create: {OUTPUT_VIDEO_PATH}"
    )


# ============================================================
# Pitch coordinate reference
# ============================================================

pitch_vertices = np.asarray(
    CONFIG.vertices,
    dtype=np.float32
)


PITCH_X_MIN = float(
    np.min(
        pitch_vertices[:, 0]
    )
)

PITCH_X_MAX = float(
    np.max(
        pitch_vertices[:, 0]
    )
)

PITCH_Y_MIN = float(
    np.min(
        pitch_vertices[:, 1]
    )
)

PITCH_Y_MAX = float(
    np.max(
        pitch_vertices[:, 1]
    )
)


# ============================================================
# Homography memory
# ============================================================

last_H = None

frame_idx = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    annotated = frame.copy()


    # ========================================================
    # 1. Pitch keypoints
    # ========================================================

    pitch_result = pitch_model(
        frame,
        device="mps",
        verbose=False
    )[0]


    keypoints = (
        sv.KeyPoints.from_ultralytics(
            pitch_result
        )
    )


    H = None

    valid_kp_count = 0

    inlier_count = 0

    jump_value = None

    used_fallback = False


    if len(keypoints.xy) > 0:

        pts = keypoints.xy[0]


        if keypoints.confidence is not None:

            conf = (
                keypoints.confidence[0]
            )

        else:

            conf = np.ones(
                len(pts),
                dtype=np.float32
            )


        source_list = []

        target_list = []


        # ====================================================
        # Keypoint filtering + EMA
        # ====================================================

        for i, (
            (x, y),
            confidence
        ) in enumerate(
            zip(
                pts,
                conf
            )
        ):


            if confidence < KP_CONF_THRESHOLD:
                continue


            if x <= 1 or y <= 1:
                continue


            current = np.array(
                [x, y],
                dtype=np.float32
            )


            if (
                i in smoothed_kps
                and
                i in kp_last_seen
                and
                frame_idx - kp_last_seen[i]
                <= KP_MAX_GAP
            ):

                smoothed_kps[i] = (

                    KP_ALPHA
                    * current

                    +

                    (1.0 - KP_ALPHA)
                    * smoothed_kps[i]
                )

            else:

                smoothed_kps[i] = (
                    current
                )


            kp_last_seen[i] = (
                frame_idx
            )


            source_list.append(
                smoothed_kps[i]
            )

            target_list.append(
                pitch_vertices[i]
            )


        valid_kp_count = len(
            source_list
        )


        # ====================================================
        # Homography
        # ====================================================

        if valid_kp_count >= MIN_KEYPOINTS:

            source = np.asarray(
                source_list,
                dtype=np.float32
            )

            target = np.asarray(
                target_list,
                dtype=np.float32
            )


            H_candidate, inliers = (
                cv2.findHomography(
                    source,
                    target,
                    cv2.RANSAC,
                    RANSAC_THRESHOLD
                )
            )


            if (
                H_candidate is not None
                and
                inliers is not None
            ):

                inlier_count = int(
                    inliers.sum()
                )


                min_inliers = max(
                    5,
                    int(
                        np.ceil(
                            valid_kp_count
                            * MIN_INLIER_RATIO
                        )
                    )
                )


                if inlier_count >= min_inliers:

                    accept_H = True


                    # =========================================
                    # Homography jump detection
                    # =========================================

                    if last_H is not None:


                        probe = np.array(
                            [
                                [
                                    [
                                        width * 0.50,
                                        height * 0.55
                                    ]
                                ],

                                [
                                    [
                                        width * 0.30,
                                        height * 0.70
                                    ]
                                ],

                                [
                                    [
                                        width * 0.70,
                                        height * 0.70
                                    ]
                                ]
                            ],
                            dtype=np.float32
                        )


                        try:

                            old_pitch = (
                                cv2.perspectiveTransform(
                                    probe,
                                    last_H
                                )
                                .reshape(
                                    -1,
                                    2
                                )
                            )


                            new_pitch = (
                                cv2.perspectiveTransform(
                                    probe,
                                    H_candidate
                                )
                                .reshape(
                                    -1,
                                    2
                                )
                            )


                            distances = (
                                np.linalg.norm(
                                    new_pitch
                                    - old_pitch,
                                    axis=1
                                )
                            )


                            jump_value = float(
                                np.mean(
                                    distances
                                )
                            )


                            if not np.isfinite(
                                jump_value
                            ):

                                accept_H = False


                            elif jump_value > MAX_H_JUMP:

                                accept_H = False


                        except cv2.error:

                            accept_H = False


                    if accept_H:

                        H = H_candidate

                        last_H = (
                            H_candidate
                        )


    # ========================================================
    # Homography fallback
    # ========================================================

    if H is None:

        H = last_H

        if last_H is not None:

            used_fallback = True


    # ========================================================
    # 2. Player Detection
    # ========================================================

    result = player_model(
        frame,
        imgsz=PLAYER_IMGSZ,
        conf=PLAYER_CONF,
        device="mps",
        verbose=False
    )[0]


    detections = (
        sv.Detections.from_ultralytics(
            result
        )
    )


    if detections.class_id is not None:

        keep = np.isin(
            detections.class_id,
            [
                PLAYER,
                GOALKEEPER
            ]
        )

        detections = (
            detections[
                keep
            ]
        )


    # ========================================================
    # ByteTrack
    # ========================================================

    detections = (
        tracker.update_with_detections(
            detections
        )
    )


    radar_positions = []

    radar_teams = []


    # ========================================================
    # Process players
    # ========================================================

    if detections.tracker_id is not None:


        for (
            box,
            track_id,
            class_id
        ) in zip(
            detections.xyxy,
            detections.tracker_id,
            detections.class_id
        ):


            track_id = int(
                track_id
            )

            class_id = int(
                class_id
            )


            x1, y1, x2, y2 = map(
                int,
                box
            )


            # =================================================
            # Team
            # =================================================

            if class_id == GOALKEEPER:


                team_id = 2

                team_name = "GK"

                box_color = (
                    0,
                    255,
                    255
                )

                label = (
                    f"GK #{track_id}"
                )


            else:


                feature = (
                    get_jersey_feature(
                        frame,
                        box
                    )
                )


                team = (
                    classify_team(
                        feature
                    )
                )


                if team is not None:

                    team_votes[
                        track_id
                    ][team] += 1


                team_id = int(
                    np.argmax(
                        team_votes[
                            track_id
                        ]
                    )
                )


                if team_id == 0:

                    team_name = "A"

                    box_color = (
                        255,
                        80,
                        40
                    )

                    label = (
                        f"A #{track_id}"
                    )


                else:

                    team_name = "B"

                    box_color = (
                        40,
                        40,
                        255
                    )

                    label = (
                        f"B #{track_id}"
                    )


            # =================================================
            # Bounding box
            # =================================================

            cv2.rectangle(
                annotated,
                (
                    x1,
                    y1
                ),
                (
                    x2,
                    y2
                ),
                box_color,
                2
            )


            cv2.putText(
                annotated,
                label,
                (
                    x1,
                    max(
                        y1 - 4,
                        15
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                box_color,
                1,
                cv2.LINE_AA
            )


            # =================================================
            # Bottom-center
            # =================================================

            if H is not None:


                px = (
                    x1 + x2
                ) / 2.0

                py = float(
                    y2
                )


                image_point = np.array(
                    [
                        [
                            [
                                px,
                                py
                            ]
                        ]
                    ],
                    dtype=np.float32
                )


                try:

                    pitch_point = (
                        cv2.perspectiveTransform(
                            image_point,
                            H
                        )[0][0]
                    )

                except cv2.error:

                    continue


                if not np.all(
                    np.isfinite(
                        pitch_point
                    )
                ):

                    continue


                # =================================================
                # Clamp pitch coordinate
                # =================================================

                pitch_point[0] = np.clip(
                    pitch_point[0],
                    PITCH_X_MIN,
                    PITCH_X_MAX
                )


                pitch_point[1] = np.clip(
                    pitch_point[1],
                    PITCH_Y_MIN,
                    PITCH_Y_MAX
                )


                # =================================================
                # Position smoothing
                # =================================================

                if (
                    track_id
                    in smoothed_player_positions
                ):

                    smoothed_player_positions[
                        track_id
                    ] = (

                        PLAYER_ALPHA
                        * pitch_point

                        +

                        (1.0 - PLAYER_ALPHA)
                        * smoothed_player_positions[
                            track_id
                        ]
                    )

                else:

                    smoothed_player_positions[
                        track_id
                    ] = (
                        pitch_point.copy()
                    )


                pitch_point = (
                    smoothed_player_positions[
                        track_id
                    ]
                )


                pitch_x_cm = float(
                    pitch_point[0]
                )

                pitch_y_cm = float(
                    pitch_point[1]
                )


                # =================================================
                # Save data for CSV
                # =================================================

                tracking_rows.append(
                    {
                        "frame": frame_idx,

                        "time_sec": (
                            frame_idx / fps
                        ),

                        "track_id": track_id,

                        "team": team_name,

                        "class_id": class_id,

                        "image_x": px,

                        "image_y": py,

                        "pitch_x_cm": pitch_x_cm,

                        "pitch_y_cm": pitch_y_cm,

                        "pitch_x_m": (
                            pitch_x_cm / 100.0
                        ),

                        "pitch_y_m": (
                            pitch_y_cm / 100.0
                        ),

                        "homography_fallback": (
                            used_fallback
                        )
                    }
                )


                radar_positions.append(
                    pitch_point
                )


                radar_teams.append(
                    team_id
                )


    # ========================================================
    # 3. Radar
    # ========================================================

    if (
        H is not None
        and
        len(radar_positions) > 0
    ):


        radar_positions = np.asarray(
            radar_positions,
            dtype=np.float32
        )


        radar_teams = np.asarray(
            radar_teams,
            dtype=np.int32
        )


        radar = draw_pitch(
            config=CONFIG
        )


        # Team A
        radar = draw_points_on_pitch(
            config=CONFIG,

            xy=radar_positions[
                radar_teams == 0
            ],

            face_color=sv.Color.from_hex(
                "#00BFFF"
            ),

            radius=18,

            pitch=radar
        )


        # Team B
        radar = draw_points_on_pitch(
            config=CONFIG,

            xy=radar_positions[
                radar_teams == 1
            ],

            face_color=sv.Color.from_hex(
                "#FF6347"
            ),

            radius=18,

            pitch=radar
        )


        # Goalkeeper
        radar = draw_points_on_pitch(
            config=CONFIG,

            xy=radar_positions[
                radar_teams == 2
            ],

            face_color=sv.Color.from_hex(
                "#FFD700"
            ),

            radius=18,

            pitch=radar
        )


        radar = sv.resize_image(
            radar,
            (
                width // 3,
                height // 3
            )
        )


        radar_h, radar_w = (
            radar.shape[:2]
        )


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


    # ========================================================
    # Debug
    # ========================================================

    if frame_idx % 30 == 0:


        jump_text = (

            "N/A"

            if jump_value is None

            else f"{jump_value:.1f} cm"
        )


        print(

            f"frame={frame_idx:04d} | "

            f"valid_kp={valid_kp_count:02d} | "

            f"inliers={inlier_count:02d} | "

            f"jump={jump_text} | "

            f"fallback={used_fallback}"
        )


    writer.write(
        annotated
    )


    frame_idx += 1


# ============================================================
# Finish video
# ============================================================

cap.release()

writer.release()


# ============================================================
# Save CSV
# ============================================================

csv_columns = [

    "frame",

    "time_sec",

    "track_id",

    "team",

    "class_id",

    "image_x",

    "image_y",

    "pitch_x_cm",

    "pitch_y_cm",

    "pitch_x_m",

    "pitch_y_m",

    "homography_fallback"
]


with open(
    OUTPUT_CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as csvfile:


    writer_csv = csv.DictWriter(

        csvfile,

        fieldnames=csv_columns
    )


    writer_csv.writeheader()


    writer_csv.writerows(
        tracking_rows
    )


print("")
print("======================================")
print("DONE")
print("======================================")

print(
    f"Video: {OUTPUT_VIDEO_PATH}"
)

print(
    f"CSV:   {OUTPUT_CSV_PATH}"
)

print(
    f"Rows:  {len(tracking_rows)}"
)