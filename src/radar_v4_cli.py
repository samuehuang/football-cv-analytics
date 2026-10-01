import os
import csv
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import supervision as sv

from ultralytics import YOLO
from sklearn.cluster import KMeans

from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch,
)

from sports.configs.soccer import (
    SoccerPitchConfiguration,
)


# ============================================================
# Device
# ============================================================

def resolve_device(requested):

    if requested != "auto":
        return requested

    try:
        import torch

        if torch.cuda.is_available():
            return "0"

        if (
            hasattr(torch.backends, "mps")
            and
            torch.backends.mps.is_available()
        ):
            return "mps"

    except Exception:
        pass

    return "cpu"


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Radar V4: player tracking, team classification, "
        "pitch homography and radar visualization."
    )
)

parser.add_argument(
    "--video",
    default="videos/match.mp4",
)

parser.add_argument(
    "--player-model",
    default="models/football-player-detection.pt",
)

parser.add_argument(
    "--pitch-model",
    default="models/football-pitch-detection.pt",
)

parser.add_argument(
    "--output-video",
    default="outputs/radar_v4.mp4",
)

parser.add_argument(
    "--output-csv",
    default="outputs/tracking_data_v4.csv",
)

parser.add_argument(
    "--device",
    default="auto",
    help="auto, cpu, mps, 0, 1, ...",
)

args = parser.parse_args()


VIDEO_PATH = args.video

PLAYER_MODEL_PATH = (
    args.player_model
)

PITCH_MODEL_PATH = (
    args.pitch_model
)

OUTPUT_VIDEO_PATH = (
    args.output_video
)

OUTPUT_CSV_PATH = (
    args.output_csv
)

DEVICE = resolve_device(
    args.device
)


# ============================================================
# Validate paths
# ============================================================

Path(
    OUTPUT_VIDEO_PATH
).parent.mkdir(
    parents=True,
    exist_ok=True,
)

Path(
    OUTPUT_CSV_PATH
).parent.mkdir(
    parents=True,
    exist_ok=True,
)


for path, label in [

    (
        VIDEO_PATH,
        "video",
    ),

    (
        PLAYER_MODEL_PATH,
        "player model",
    ),

    (
        PITCH_MODEL_PATH,
        "pitch model",
    ),
]:

    if not Path(path).is_file():

        raise FileNotFoundError(
            f"Missing {label}: {path}"
        )


print("")

print(
    "======================================"
)

print(
    "Radar V4 CLI"
)

print(
    "======================================"
)

print(
    f"Video:        {VIDEO_PATH}"
)

print(
    f"Player model: {PLAYER_MODEL_PATH}"
)

print(
    f"Pitch model:  {PITCH_MODEL_PATH}"
)

print(
    f"Device:       {DEVICE}"
)

print(
    f"Output CSV:   {OUTPUT_CSV_PATH}"
)

print(
    f"Output video: {OUTPUT_VIDEO_PATH}"
)


# ============================================================
# Class IDs
# ============================================================

BALL = 0

GOALKEEPER = 1

PLAYER = 2

REFEREE = 3


# ============================================================
# Models
# ============================================================

CONFIG = (
    SoccerPitchConfiguration()
)

player_model = YOLO(
    PLAYER_MODEL_PATH
)

pitch_model = YOLO(
    PITCH_MODEL_PATH
)


# ============================================================
# Parameters
# ============================================================

KP_CONF_THRESHOLD = 0.60

MIN_KEYPOINTS = 6

MIN_INLIER_RATIO = 0.60

RANSAC_THRESHOLD = 250.0

MAX_MEAN_REPROJECTION_ERROR = (
    30.0
)

MAX_MEDIAN_REPROJECTION_ERROR = (
    20.0
)

MAX_FALLBACK_FRAMES = 3

PLAYER_ALPHA = 0.45

PLAYER_IMGSZ = 1280

PLAYER_CONF = 0.25


# ============================================================
# Jersey feature
# ============================================================

def get_jersey_feature(
    frame,
    box,
):

    x1, y1, x2, y2 = map(
        int,
        box,
    )

    h = y2 - y1

    w = x2 - x1


    if (
        h <= 0
        or
        w <= 0
    ):

        return None


    xa = max(
        0,
        x1 + int(
            w * 0.20
        ),
    )

    xb = min(
        frame.shape[1],
        x2 - int(
            w * 0.20
        ),
    )

    ya = max(
        0,
        y1 + int(
            h * 0.15
        ),
    )

    yb = min(
        frame.shape[0],
        y1 + int(
            h * 0.55
        ),
    )


    crop = frame[
        ya:yb,
        xa:xb,
    ]


    if crop.size == 0:

        return None


    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV,
    )


    green_mask = cv2.inRange(

        hsv,

        np.array(
            [
                30,
                40,
                30,
            ]
        ),

        np.array(
            [
                95,
                255,
                255,
            ]
        ),
    )


    valid = (
        green_mask == 0
    )


    if valid.sum() < 10:

        return None


    lab = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2LAB,
    )


    return np.median(
        lab[valid],
        axis=0,
    ).astype(
        np.float32
    )


# ============================================================
# PASS 1
# Learn team colors
# ============================================================

print("")

print(
    "======================================"
)

print(
    "[1/2] Learning team colors"
)

print(
    "======================================"
)


features = []


cap = cv2.VideoCapture(
    VIDEO_PATH
)


if not cap.isOpened():

    raise RuntimeError(
        f"Cannot open video: "
        f"{VIDEO_PATH}"
    )


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

        imgsz=PLAYER_IMGSZ,

        conf=PLAYER_CONF,

        device=DEVICE,

        verbose=False,

    )[0]


    detections = (
        sv.Detections
        .from_ultralytics(
            result
        )
    )


    if (
        detections.class_id
        is not None
    ):

        players = detections[
            detections.class_id
            ==
            PLAYER
        ]


        for box in players.xyxy:

            feature = (
                get_jersey_feature(
                    frame,
                    box,
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
    dtype=np.float32,
)


print(
    f"Collected "
    f"{len(features)} "
    f"jersey samples"
)


if len(features) < 20:

    raise RuntimeError(
        "Not enough jersey samples."
    )


# ============================================================
# KMeans Team Classification
# ============================================================

kmeans = KMeans(

    n_clusters=2,

    n_init=20,

    random_state=0,

)


kmeans.fit(
    features
)


centers = (
    kmeans.cluster_centers_
)


bright_cluster = int(
    np.argmax(
        centers[:, 0]
    )
)


dark_cluster = (
    1
    -
    bright_cluster
)


TEAM_A_CENTER = (
    centers[
        bright_cluster
    ]
)


TEAM_B_CENTER = (
    centers[
        dark_cluster
    ]
)


print(
    "Team A center:",
    TEAM_A_CENTER,
)


print(
    "Team B center:",
    TEAM_B_CENTER,
)


def classify_team(
    feature,
):

    if feature is None:

        return None


    dist_a = np.linalg.norm(
        feature
        -
        TEAM_A_CENTER
    )


    dist_b = np.linalg.norm(
        feature
        -
        TEAM_B_CENTER
    )


    return (
        0
        if dist_a < dist_b
        else 1
    )


# ============================================================
# PASS 2
# ============================================================

print("")

print(
    "======================================"
)

print(
    "[2/2] Radar V4"
)

print(
    "Raw KP + RANSAC + Reprojection Check"
)

print(
    "======================================"
)


tracker = sv.ByteTrack(
    minimum_consecutive_frames=3
)


team_votes = defaultdict(
    lambda: [
        0,
        0,
    ]
)


smoothed_player_positions = {}


tracking_rows = []


# ============================================================
# Video
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)


if not cap.isOpened():

    raise RuntimeError(
        f"Cannot open video: "
        f"{VIDEO_PATH}"
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
        height,
    ),
)


if not writer.isOpened():

    raise RuntimeError(
        f"Cannot create "
        f"{OUTPUT_VIDEO_PATH}"
    )


# ============================================================
# Pitch coordinates
# ============================================================

pitch_vertices = np.asarray(

    CONFIG.vertices,

    dtype=np.float32,

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
# Homography state
# ============================================================

last_H = None

fallback_streak = 0

frame_idx = 0


# ============================================================
# Main loop
# ============================================================

while True:

    ret, frame = cap.read()


    if not ret:

        break


    annotated = (
        frame.copy()
    )


    # ========================================================
    # Pitch Keypoint Detection
    # ========================================================

    pitch_result = pitch_model(

        frame,

        device=DEVICE,

        verbose=False,

    )[0]


    keypoints = (
        sv.KeyPoints
        .from_ultralytics(
            pitch_result
        )
    )


    H = None

    valid_kp_count = 0

    inlier_count = 0

    mean_reprojection_error = (
        None
    )

    median_reprojection_error = (
        None
    )

    used_fallback = False


    # ========================================================
    # Current-frame raw pitch keypoints
    # ========================================================

    if len(
        keypoints.xy
    ) > 0:

        pts = (
            keypoints.xy[0]
        )


        if (
            keypoints.keypoint_confidence
            is not None
        ):

            conf = (
                keypoints
                .keypoint_confidence[0]
            )

        else:

            conf = np.ones(
                len(pts),
                dtype=np.float32,
            )


        valid = (

            (
                pts[:, 0]
                >
                1
            )

            &

            (
                pts[:, 1]
                >
                1
            )

            &

            (
                conf
                >=
                KP_CONF_THRESHOLD
            )
        )


        source = (
            pts[valid]
            .astype(
                np.float32
            )
        )


        target = (
            pitch_vertices[valid]
            .astype(
                np.float32
            )
        )


        valid_kp_count = len(
            source
        )


        # ====================================================
        # Homography
        # ====================================================

        if (
            valid_kp_count
            >=
            MIN_KEYPOINTS
        ):

            (
                H_candidate,
                inliers,
            ) = cv2.findHomography(

                source,

                target,

                cv2.RANSAC,

                RANSAC_THRESHOLD,

            )


            if (
                H_candidate
                is not None

                and

                inliers
                is not None
            ):

                inlier_mask = (
                    inliers
                    .reshape(-1)
                    .astype(bool)
                )


                inlier_count = int(
                    inlier_mask.sum()
                )


                min_inliers = max(

                    5,

                    int(
                        np.ceil(
                            valid_kp_count
                            *
                            MIN_INLIER_RATIO
                        )
                    ),
                )


                if (
                    inlier_count
                    >=
                    min_inliers
                ):

                    try:

                        H_inverse = (
                            np.linalg.inv(
                                H_candidate
                            )
                        )


                        target_inliers = (

                            target[
                                inlier_mask
                            ]

                            .reshape(
                                -1,
                                1,
                                2,
                            )
                        )


                        source_inliers = (
                            source[
                                inlier_mask
                            ]
                        )


                        projected_image = (

                            cv2
                            .perspectiveTransform(

                                target_inliers,

                                H_inverse,

                            )

                            .reshape(
                                -1,
                                2,
                            )
                        )


                        errors = np.linalg.norm(

                            projected_image
                            -
                            source_inliers,

                            axis=1,
                        )


                        mean_reprojection_error = float(

                            np.mean(
                                errors
                            )
                        )


                        median_reprojection_error = float(

                            np.median(
                                errors
                            )
                        )


                        if (

                            np.isfinite(
                                mean_reprojection_error
                            )

                            and

                            np.isfinite(
                                median_reprojection_error
                            )

                            and

                            mean_reprojection_error
                            <=
                            MAX_MEAN_REPROJECTION_ERROR

                            and

                            median_reprojection_error
                            <=
                            MAX_MEDIAN_REPROJECTION_ERROR

                        ):

                            H = (
                                H_candidate
                            )


                            last_H = (
                                H_candidate
                            )


                            fallback_streak = 0


                    except (
                        cv2.error,
                        np.linalg.LinAlgError,
                    ):

                        pass


    # ========================================================
    # Homography fallback
    # ========================================================

    if H is None:

        if (

            last_H
            is not None

            and

            fallback_streak
            <
            MAX_FALLBACK_FRAMES

        ):

            H = last_H

            fallback_streak += 1

            used_fallback = True

        else:

            H = None

            fallback_streak += 1


    # ========================================================
    # Player Detection
    # ========================================================

    result = player_model(

        frame,

        imgsz=PLAYER_IMGSZ,

        conf=PLAYER_CONF,

        device=DEVICE,

        verbose=False,

    )[0]


    detections = (
        sv.Detections
        .from_ultralytics(
            result
        )
    )


    if (
        detections.class_id
        is not None
    ):

        keep = np.isin(

            detections.class_id,

            [
                PLAYER,
                GOALKEEPER,
            ],
        )


        detections = (
            detections[
                keep
            ]
        )


    # ========================================================
    # Tracking
    # ========================================================

    detections = (
        tracker
        .update_with_detections(
            detections
        )
    )


    radar_positions = []

    radar_teams = []


    # ========================================================
    # Process Tracks
    # ========================================================

    if (
        detections.tracker_id
        is not None
    ):

        for (
            box,
            track_id,
            class_id,
        ) in zip(

            detections.xyxy,

            detections.tracker_id,

            detections.class_id,

        ):

            track_id = int(
                track_id
            )


            class_id = int(
                class_id
            )


            x1, y1, x2, y2 = map(
                int,
                box,
            )


            # ================================================
            # Team classification
            # ================================================

            if (
                class_id
                ==
                GOALKEEPER
            ):

                team_id = 2

                team_name = "GK"

                box_color = (
                    0,
                    255,
                    255,
                )

                label = (
                    f"GK #{track_id}"
                )


            else:

                feature = (
                    get_jersey_feature(
                        frame,
                        box,
                    )
                )


                team = (
                    classify_team(
                        feature
                    )
                )


                if (
                    team
                    is not None
                ):

                    team_votes[
                        track_id
                    ][
                        team
                    ] += 1


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
                        40,
                    )

                    label = (
                        f"A #{track_id}"
                    )

                else:

                    team_name = "B"

                    box_color = (
                        40,
                        40,
                        255,
                    )

                    label = (
                        f"B #{track_id}"
                    )


            # ================================================
            # Draw player
            # ================================================

            cv2.rectangle(

                annotated,

                (
                    x1,
                    y1,
                ),

                (
                    x2,
                    y2,
                ),

                box_color,

                2,

            )


            cv2.putText(

                annotated,

                label,

                (
                    x1,

                    max(
                        y1 - 4,
                        15,
                    ),
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.40,

                box_color,

                1,

                cv2.LINE_AA,

            )


            # ================================================
            # Player -> pitch coordinates
            # ================================================

            if H is None:

                continue


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
                            py,
                        ]
                    ]
                ],

                dtype=np.float32,

            )


            try:

                pitch_point = (

                    cv2
                    .perspectiveTransform(

                        image_point,

                        H,

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


            pitch_x = float(
                pitch_point[0]
            )


            pitch_y = float(
                pitch_point[1]
            )


            inside_pitch = (

                PITCH_X_MIN
                <=
                pitch_x
                <=
                PITCH_X_MAX

                and

                PITCH_Y_MIN
                <=
                pitch_y
                <=
                PITCH_Y_MAX

            )


            if not inside_pitch:

                continue


            pitch_point = np.array(

                [
                    pitch_x,
                    pitch_y,
                ],

                dtype=np.float32,

            )


            # ================================================
            # Player position smoothing
            # ================================================

            if (
                track_id
                in
                smoothed_player_positions
            ):

                smoothed_player_positions[
                    track_id
                ] = (

                    PLAYER_ALPHA
                    *
                    pitch_point

                    +

                    (
                        1.0
                        -
                        PLAYER_ALPHA
                    )

                    *
                    smoothed_player_positions[
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


            # ================================================
            # CSV
            # ================================================

            tracking_rows.append(
                {
                    "frame":
                        frame_idx,

                    "time_sec":
                        frame_idx
                        /
                        fps,

                    "track_id":
                        track_id,

                    "team":
                        team_name,

                    "class_id":
                        class_id,

                    "image_x":
                        px,

                    "image_y":
                        py,

                    "pitch_x_cm":
                        pitch_x_cm,

                    "pitch_y_cm":
                        pitch_y_cm,

                    "pitch_x_m":
                        pitch_x_cm
                        /
                        100.0,

                    "pitch_y_m":
                        pitch_y_cm
                        /
                        100.0,

                    "valid_keypoints":
                        valid_kp_count,

                    "inliers":
                        inlier_count,

                    "mean_reprojection_error_px":
                        mean_reprojection_error,

                    "median_reprojection_error_px":
                        median_reprojection_error,

                    "homography_fallback":
                        used_fallback,
                }
            )


            radar_positions.append(
                pitch_point
            )


            radar_teams.append(
                team_id
            )


    # ========================================================
    # Radar
    # ========================================================

    if (

        H is not None

        and

        len(
            radar_positions
        ) > 0

    ):

        radar_positions = np.asarray(

            radar_positions,

            dtype=np.float32,

        )


        radar_teams = np.asarray(

            radar_teams,

            dtype=np.int32,

        )


        radar = draw_pitch(
            config=CONFIG
        )


        radar = draw_points_on_pitch(

            config=CONFIG,

            xy=radar_positions[
                radar_teams == 0
            ],

            face_color=(
                sv.Color
                .from_hex(
                    "#00BFFF"
                )
            ),

            radius=18,

            pitch=radar,

        )


        radar = draw_points_on_pitch(

            config=CONFIG,

            xy=radar_positions[
                radar_teams == 1
            ],

            face_color=(
                sv.Color
                .from_hex(
                    "#FF6347"
                )
            ),

            radius=18,

            pitch=radar,

        )


        radar = draw_points_on_pitch(

            config=CONFIG,

            xy=radar_positions[
                radar_teams == 2
            ],

            face_color=(
                sv.Color
                .from_hex(
                    "#FFD700"
                )
            ),

            radius=18,

            pitch=radar,

        )


        radar = sv.resize_image(

            radar,

            (
                width // 3,
                height // 3,
            ),
        )


        radar_h, radar_w = (
            radar.shape[:2]
        )


        rect = sv.Rect(

            x=(
                width
                -
                radar_w
                -
                20
            ),

            y=(
                height
                -
                radar_h
                -
                20
            ),

            width=radar_w,

            height=radar_h,

        )


        annotated = sv.draw_image(

            annotated,

            radar,

            opacity=0.78,

            rect=rect,

        )


    # ========================================================
    # Debug
    # ========================================================

    if frame_idx % 30 == 0:

        mean_text = (

            "N/A"

            if
            mean_reprojection_error
            is None

            else

            (
                f"{mean_reprojection_error:.2f}px"
            )
        )


        median_text = (

            "N/A"

            if
            median_reprojection_error
            is None

            else

            (
                f"{median_reprojection_error:.2f}px"
            )
        )


        print(

            f"frame={frame_idx:04d} | "

            f"kp={valid_kp_count:02d} | "

            f"inliers={inlier_count:02d} | "

            f"mean_err={mean_text} | "

            f"median_err={median_text} | "

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

    "valid_keypoints",

    "inliers",

    "mean_reprojection_error_px",

    "median_reprojection_error_px",

    "homography_fallback",

]


with open(

    OUTPUT_CSV_PATH,

    "w",

    newline="",

    encoding="utf-8",

) as csvfile:

    csv_writer = csv.DictWriter(

        csvfile,

        fieldnames=csv_columns,

    )


    csv_writer.writeheader()


    csv_writer.writerows(
        tracking_rows
    )


print("")

print(
    "======================================"
)

print(
    "DONE"
)

print(
    "======================================"
)


print(
    f"Video: "
    f"{OUTPUT_VIDEO_PATH}"
)


print(
    f"CSV:   "
    f"{OUTPUT_CSV_PATH}"
)


print(
    f"Rows:  "
    f"{len(tracking_rows)}"
)
