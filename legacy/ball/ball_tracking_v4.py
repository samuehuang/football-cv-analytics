import os
import csv
from collections import deque

import cv2
import numpy as np
import supervision as sv

from ultralytics import YOLO

from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch,
)

from sports.configs.soccer import (
    SoccerPitchConfiguration,
)


# ============================================================
# Paths
# ============================================================

VIDEO_PATH = "videos/match.mp4"

BALL_MODEL_PATH = "models/football-ball-detection.pt"
PITCH_MODEL_PATH = "models/football-pitch-detection.pt"

OUTPUT_VIDEO = "outputs/ball_tracking_v4.mp4"
OUTPUT_CSV = "outputs/ball_tracking_v4.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Device / Models
# ============================================================

DEVICE = "mps"

ball_model = YOLO(
    BALL_MODEL_PATH
)

pitch_model = YOLO(
    PITCH_MODEL_PATH
)

CONFIG = SoccerPitchConfiguration()


# ============================================================
# Detection Parameters
# ============================================================

# Full-frame detector
FULL_CONF = 0.08
FULL_IMGSZ = 1280


# ------------------------------------------------------------
# 3 x 2 tiled inference
#
# V4 每一幀都跑。
# 比 V3 慢，但是現在目標是 precision。
# ------------------------------------------------------------

TILE_ROWS = 2
TILE_COLS = 3

TILE_OVERLAP = 0.20

TILE_CONF = 0.06
TILE_IMGSZ = 960


# ------------------------------------------------------------
# Full / Tile candidate clustering
# ------------------------------------------------------------

CLUSTER_CENTER_PX = 22.0


# ============================================================
# Multi-scale Validation
# ============================================================

# Candidate 被視為 consensus：
#
# 1. full-frame + 至少一個 tile 都看到
#
# OR
#
# 2. 至少兩個 overlapping tiles 都看到
#
#
# Consensus 可以接受比較低 confidence。
# Single-scale candidate 必須更高 confidence。


CONSENSUS_MIN_CONF = 0.12

SINGLE_MIN_CONF = 0.35


# ============================================================
# Reacquisition
# ============================================================

# Track 丟失後重新抓球，比正常 tracking 嚴格很多。

REACQUIRE_CONSENSUS_CONF = 0.20

REACQUIRE_SINGLE_CONF = 0.65


# 重新取得球不能一幀就相信。
# 必須下一幀再次看到合理位置。

REACQUIRE_CONFIRM_MAX_GAP = 2


# image-space reacquire sanity
REACQUIRE_IMAGE_SPEED_PX_S = 2200.0


# pitch-space sanity
REACQUIRE_PITCH_SPEED_MPS = 60.0


# ============================================================
# Temporal Association
# ============================================================

# 最多容忍連續 5 frame 沒抓到球，
# 仍然沿用舊 trajectory 做 prediction。

MAX_ASSOCIATION_GAP = 5


# 只用來移除明顯 teleport。
#
# 60 m/s = 216 km/h
# 已經非常寬鬆。
MAX_BALL_SPEED_MPS = 60.0


# ------------------------------------------------------------
# Prediction gates
#
# V4 主要使用 image-space prediction。
#
# pitch-space 次要，
# 因為空中球不符合 ground-plane homography。
# ------------------------------------------------------------

BASE_IMAGE_GATE_PX = 110.0

EXTRA_IMAGE_GATE_PX_PER_GAP = 70.0


BASE_PITCH_GATE_M = 4.5

EXTRA_PITCH_GATE_M_PER_GAP = 2.5


# ------------------------------------------------------------
# Association scoring
# ------------------------------------------------------------

IMAGE_WEIGHT = 0.50

PITCH_WEIGHT = 0.20

CONFIDENCE_WEIGHT = 0.15

SUPPORT_WEIGHT = 0.15


# ============================================================
# Homography Parameters
# ============================================================

KP_CONF_THRESHOLD = 0.60

MIN_KEYPOINTS = 6

MIN_INLIER_RATIO = 0.60

RANSAC_THRESHOLD = 250.0


MAX_MEAN_REPROJECTION_ERROR = 30.0

MAX_MEDIAN_REPROJECTION_ERROR = 20.0


MAX_FALLBACK_FRAMES = 3


# ============================================================
# Pitch Geometry
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
# Homography
# ============================================================

def estimate_homography(
    frame,
    last_H,
    fallback_streak,
):

    result = pitch_model(
        frame,
        device=DEVICE,
        verbose=False,
    )[0]


    keypoints = (
        sv.KeyPoints.from_ultralytics(
            result
        )
    )


    H = None

    valid_kp_count = 0

    inlier_count = 0

    mean_error = None

    median_error = None

    used_fallback = False


    # ========================================================
    # Current frame keypoints
    # ========================================================

    if len(keypoints.xy) > 0:

        pts = keypoints.xy[0]


        if (
            keypoints.confidence
            is not None
        ):

            conf = (
                keypoints
                .confidence[0]
            )

        else:

            conf = np.ones(
                len(pts),
                dtype=np.float32,
            )


        valid = (

            (pts[:, 0] > 1)

            &

            (pts[:, 1] > 1)

            &

            (
                conf
                >= KP_CONF_THRESHOLD
            )
        )


        source = (
            pts[valid]
            .astype(np.float32)
        )


        target = (
            pitch_vertices[valid]
            .astype(np.float32)
        )


        valid_kp_count = len(
            source
        )


        # ====================================================
        # RANSAC
        # ====================================================

        if (
            valid_kp_count
            >= MIN_KEYPOINTS
        ):

            H_candidate, inliers = (
                cv2.findHomography(
                    source,
                    target,
                    cv2.RANSAC,
                    RANSAC_THRESHOLD,
                )
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
                    )
                )


                if (
                    inlier_count
                    >= min_inliers
                ):

                    try:

                        H_inv = (
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
                                2
                            )
                        )


                        source_inliers = (
                            source[
                                inlier_mask
                            ]
                        )


                        projected_image = (
                            cv2.perspectiveTransform(
                                target_inliers,
                                H_inv,
                            )
                            .reshape(
                                -1,
                                2
                            )
                        )


                        errors = (
                            np.linalg.norm(
                                projected_image
                                -
                                source_inliers,
                                axis=1,
                            )
                        )


                        mean_error = float(
                            np.mean(
                                errors
                            )
                        )


                        median_error = float(
                            np.median(
                                errors
                            )
                        )


                        if (

                            np.isfinite(
                                mean_error
                            )

                            and

                            np.isfinite(
                                median_error
                            )

                            and

                            mean_error
                            <= MAX_MEAN_REPROJECTION_ERROR

                            and

                            median_error
                            <= MAX_MEDIAN_REPROJECTION_ERROR

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
    # Homography short fallback
    # ========================================================

    if H is None:

        if (

            last_H is not None

            and

            fallback_streak
            < MAX_FALLBACK_FRAMES

        ):

            H = last_H

            fallback_streak += 1

            used_fallback = True

        else:

            fallback_streak += 1


    return (

        H,

        last_H,

        fallback_streak,

        used_fallback,

        valid_kp_count,

        inlier_count,

        mean_error,

        median_error,
    )


# ============================================================
# Generate Tiles
# ============================================================

def generate_tiles(
    frame_width,
    frame_height,
):

    base_w = int(
        np.ceil(
            frame_width
            /
            TILE_COLS
        )
    )


    base_h = int(
        np.ceil(
            frame_height
            /
            TILE_ROWS
        )
    )


    overlap_x = int(
        base_w
        *
        TILE_OVERLAP
    )


    overlap_y = int(
        base_h
        *
        TILE_OVERLAP
    )


    tiles = []


    for row in range(
        TILE_ROWS
    ):

        for col in range(
            TILE_COLS
        ):

            x1 = max(
                0,
                col * base_w
                -
                overlap_x,
            )


            y1 = max(
                0,
                row * base_h
                -
                overlap_y,
            )


            x2 = min(
                frame_width,
                (col + 1) * base_w
                +
                overlap_x,
            )


            y2 = min(
                frame_height,
                (row + 1) * base_h
                +
                overlap_y,
            )


            tiles.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                )
            )


    return tiles


# ============================================================
# Full-frame raw detections
# ============================================================

def raw_full_detections(
    frame,
):

    result = ball_model(

        frame,

        imgsz=FULL_IMGSZ,

        conf=FULL_CONF,

        device=DEVICE,

        verbose=False,

    )[0]


    boxes = result.boxes


    if (
        boxes is None
        or
        len(boxes) == 0
    ):

        return []


    xyxy = (
        boxes.xyxy
        .detach()
        .cpu()
        .numpy()
    )


    confidences = (
        boxes.conf
        .detach()
        .cpu()
        .numpy()
    )


    detections = []


    for box, conf in zip(
        xyxy,
        confidences,
    ):

        x1, y1, x2, y2 = box


        detections.append(
            {
                "box":
                    np.asarray(
                        box,
                        dtype=np.float32,
                    ),

                "confidence":
                    float(
                        conf
                    ),

                "image_x":
                    float(
                        (
                            x1 + x2
                        )
                        /
                        2.0
                    ),

                "image_y":
                    float(
                        (
                            y1 + y2
                        )
                        /
                        2.0
                    ),

                "source":
                    "full",
            }
        )


    return detections


# ============================================================
# Tiled raw detections
# ============================================================

def raw_tiled_detections(
    frame,
):

    height, width = (
        frame.shape[:2]
    )


    tiles = generate_tiles(
        width,
        height,
    )


    detections = []


    for tile_idx, (
        tx1,
        ty1,
        tx2,
        ty2,

    ) in enumerate(tiles):


        crop = frame[
            ty1:ty2,
            tx1:tx2,
        ]


        if crop.size == 0:

            continue


        result = ball_model(

            crop,

            imgsz=TILE_IMGSZ,

            conf=TILE_CONF,

            device=DEVICE,

            verbose=False,

        )[0]


        boxes = result.boxes


        if (
            boxes is None
            or
            len(boxes) == 0
        ):

            continue


        xyxy = (
            boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )


        confidences = (
            boxes.conf
            .detach()
            .cpu()
            .numpy()
        )


        for local_box, conf in zip(
            xyxy,
            confidences,
        ):

            x1, y1, x2, y2 = (
                local_box
            )


            full_box = np.array(
                [
                    x1 + tx1,
                    y1 + ty1,
                    x2 + tx1,
                    y2 + ty1,
                ],
                dtype=np.float32,
            )


            fx1, fy1, fx2, fy2 = (
                full_box
            )


            detections.append(
                {

                    "box":
                        full_box,

                    "confidence":
                        float(conf),

                    "image_x":
                        float(
                            (
                                fx1 + fx2
                            )
                            /
                            2.0
                        ),

                    "image_y":
                        float(
                            (
                                fy1 + fy2
                            )
                            /
                            2.0
                        ),

                    "source":
                        f"tile_{tile_idx}",
                }
            )


    return detections


# ============================================================
# Project image point -> pitch
# ============================================================

def project_center(
    image_x,
    image_y,
    H,
):

    if H is None:

        return None


    point = np.array(
        [
            [
                [
                    image_x,
                    image_y
                ]
            ]
        ],
        dtype=np.float32,
    )


    try:

        pitch = (
            cv2.perspectiveTransform(
                point,
                H,
            )[0][0]
        )


    except cv2.error:

        return None


    if not np.all(
        np.isfinite(
            pitch
        )
    ):

        return None


    px = float(
        pitch[0]
    )

    py = float(
        pitch[1]
    )


    if not (

        PITCH_X_MIN
        <= px
        <= PITCH_X_MAX

        and

        PITCH_Y_MIN
        <= py
        <= PITCH_Y_MAX

    ):

        return None


    return (
        px,
        py
    )


# ============================================================
# Multi-scale candidate clustering
# ============================================================

def cluster_multiscale_candidates(
    raw_detections,
    H,
):

    if not raw_detections:

        return []


    raw_sorted = sorted(

        raw_detections,

        key=lambda d:
            d["confidence"],

        reverse=True,
    )


    clusters = []


    # ========================================================
    # Cluster nearby detections
    # ========================================================

    for det in raw_sorted:

        center = np.array(
            [
                det["image_x"],
                det["image_y"]
            ],
            dtype=np.float32,
        )


        matched_cluster = None


        for cluster in clusters:

            cluster_center = np.array(
                [
                    cluster["image_x"],
                    cluster["image_y"]
                ],
                dtype=np.float32,
            )


            distance = float(
                np.linalg.norm(
                    center
                    -
                    cluster_center
                )
            )


            if (
                distance
                <= CLUSTER_CENTER_PX
            ):

                matched_cluster = (
                    cluster
                )

                break


        if matched_cluster is None:

            clusters.append(
                {

                    "members":
                        [det],

                    "image_x":
                        det["image_x"],

                    "image_y":
                        det["image_y"],
                }
            )


        else:

            matched_cluster[
                "members"
            ].append(
                det
            )


            weights = np.array(
                [
                    max(
                        member[
                            "confidence"
                        ],
                        1e-3,
                    )

                    for member
                    in matched_cluster[
                        "members"
                    ]
                ],
                dtype=np.float32,
            )


            xs = np.array(
                [
                    member["image_x"]

                    for member
                    in matched_cluster[
                        "members"
                    ]
                ]
            )


            ys = np.array(
                [
                    member["image_y"]

                    for member
                    in matched_cluster[
                        "members"
                    ]
                ]
            )


            matched_cluster[
                "image_x"
            ] = float(
                np.average(
                    xs,
                    weights=weights,
                )
            )


            matched_cluster[
                "image_y"
            ] = float(
                np.average(
                    ys,
                    weights=weights,
                )
            )


    # ========================================================
    # Convert clusters -> candidates
    # ========================================================

    candidates = []


    for cluster in clusters:

        members = (
            cluster["members"]
        )


        best_member = max(

            members,

            key=lambda member:
                member["confidence"],
        )


        sources = sorted(
            set(
                member["source"]

                for member
                in members
            )
        )


        full_support = (
            "full"
            in sources
        )


        tile_sources = [
            source
            for source in sources
            if source.startswith(
                "tile_"
            )
        ]


        tile_support_count = len(
            tile_sources
        )


        consensus = (

            (
                full_support

                and

                tile_support_count >= 1
            )

            or

            tile_support_count >= 2
        )


        projected = project_center(

            cluster["image_x"],

            cluster["image_y"],

            H,
        )


        if projected is None:

            continue


        pitch_x_cm, pitch_y_cm = (
            projected
        )


        candidates.append(
            {

                "box":
                    best_member[
                        "box"
                    ],

                "confidence":
                    float(
                        best_member[
                            "confidence"
                        ]
                    ),

                "image_x":
                    float(
                        cluster[
                            "image_x"
                        ]
                    ),

                "image_y":
                    float(
                        cluster[
                            "image_y"
                        ]
                    ),

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

                "sources":
                    "+".join(
                        sources
                    ),

                "support_count":
                    len(
                        sources
                    ),

                "full_support":
                    full_support,

                "tile_support_count":
                    tile_support_count,

                "consensus":
                    consensus,
            }
        )


    return candidates


# ============================================================
# Prediction from accepted history
# ============================================================

def predict_from_history(
    history,
    frame_idx,
    fps,
):

    last = history[-1]


    frame_gap = (

        frame_idx

        -

        int(
            last["frame"]
        )
    )


    dt = (
        frame_gap
        /
        fps
    )


    predicted_image = np.array(
        [
            last["image_x"],
            last["image_y"]
        ],
        dtype=np.float32,
    )


    predicted_pitch = np.array(
        [
            last["pitch_x_m"],
            last["pitch_y_m"]
        ],
        dtype=np.float32,
    )


    # ========================================================
    # Constant velocity
    # ========================================================

    if len(history) >= 2:

        previous = (
            history[-2]
        )


        previous_gap = (

            int(
                last["frame"]
            )

            -

            int(
                previous["frame"]
            )
        )


        previous_dt = (
            previous_gap
            /
            fps
        )


        if (

            previous_dt > 0

            and

            previous_gap
            <= MAX_ASSOCIATION_GAP

        ):

            image_velocity = (

                np.array(
                    [
                        last["image_x"],
                        last["image_y"]
                    ],
                    dtype=np.float32,
                )

                -

                np.array(
                    [
                        previous["image_x"],
                        previous["image_y"]
                    ],
                    dtype=np.float32,
                )

            ) / previous_dt


            pitch_velocity = (

                np.array(
                    [
                        last["pitch_x_m"],
                        last["pitch_y_m"]
                    ],
                    dtype=np.float32,
                )

                -

                np.array(
                    [
                        previous["pitch_x_m"],
                        previous["pitch_y_m"]
                    ],
                    dtype=np.float32,
                )

            ) / previous_dt


            # Only trust velocity
            # if pitch speed is still sane.
            if (
                float(
                    np.linalg.norm(
                        pitch_velocity
                    )
                )
                <=
                MAX_BALL_SPEED_MPS
            ):

                predicted_image = (

                    predicted_image

                    +

                    image_velocity
                    *
                    dt
                )


                predicted_pitch = (

                    predicted_pitch

                    +

                    pitch_velocity
                    *
                    dt
                )


    return (

        frame_gap,

        dt,

        predicted_image,

        predicted_pitch,
    )


# ============================================================
# Candidate eligibility during normal tracking
# ============================================================

def candidate_detection_eligible(
    candidate,
):

    if candidate[
        "consensus"
    ]:

        return (
            candidate[
                "confidence"
            ]
            >=
            CONSENSUS_MIN_CONF
        )


    return (
        candidate[
            "confidence"
        ]
        >=
        SINGLE_MIN_CONF
    )


# ============================================================
# Active-track Association
# ============================================================

def associate_active_track(
    candidates,
    frame_idx,
    fps,
    history,
):

    if len(history) == 0:

        return (
            None,
            "no_history",
            None,
            None,
            None,
        )


    (
        frame_gap,
        dt,
        predicted_image,
        predicted_pitch,

    ) = predict_from_history(
        history,
        frame_idx,
        fps,
    )


    if frame_gap <= 0:

        return (
            None,
            "invalid_gap",
            None,
            None,
            None,
        )


    if (
        frame_gap
        >
        MAX_ASSOCIATION_GAP
    ):

        return (
            None,
            "track_lost",
            None,
            None,
            None,
        )


    last = history[-1]


    last_pitch = np.array(
        [
            last["pitch_x_m"],
            last["pitch_y_m"]
        ],
        dtype=np.float32,
    )


    # ========================================================
    # Dynamic gates
    # ========================================================

    image_gate = (

        BASE_IMAGE_GATE_PX

        +

        EXTRA_IMAGE_GATE_PX_PER_GAP
        *
        (
            frame_gap - 1
        )
    )


    pitch_gate = (

        BASE_PITCH_GATE_M

        +

        EXTRA_PITCH_GATE_M_PER_GAP
        *
        (
            frame_gap - 1
        )
    )


    options = []


    for candidate in candidates:

        if not candidate_detection_eligible(
            candidate
        ):

            continue


        image_position = np.array(
            [
                candidate["image_x"],
                candidate["image_y"]
            ],
            dtype=np.float32,
        )


        pitch_position = np.array(
            [
                candidate["pitch_x_m"],
                candidate["pitch_y_m"]
            ],
            dtype=np.float32,
        )


        image_residual = float(
            np.linalg.norm(
                image_position
                -
                predicted_image
            )
        )


        pitch_residual = float(
            np.linalg.norm(
                pitch_position
                -
                predicted_pitch
            )
        )


        distance_from_last = float(
            np.linalg.norm(
                pitch_position
                -
                last_pitch
            )
        )


        estimated_speed = (
            distance_from_last
            /
            dt
        )


        # ====================================================
        # Hard teleport gate
        # ====================================================

        if (
            estimated_speed
            >
            MAX_BALL_SPEED_MPS
        ):

            continue


        # ====================================================
        # Single-scale candidates need stronger agreement
        # ====================================================

        if candidate[
            "consensus"
        ]:

            image_limit = (
                image_gate
            )

            pitch_limit = (
                pitch_gate
            )

        else:

            image_limit = (
                image_gate
                *
                0.55
            )

            pitch_limit = (
                pitch_gate
                *
                0.65
            )


        if (
            image_residual
            >
            image_limit
        ):

            continue


        if (
            pitch_residual
            >
            pitch_limit
        ):

            continue


        # ====================================================
        # Scoring
        # ====================================================

        image_cost = (
            image_residual
            /
            max(
                image_limit,
                1.0,
            )
        )


        pitch_cost = (
            pitch_residual
            /
            max(
                pitch_limit,
                0.5,
            )
        )


        confidence_cost = (
            1.0
            -
            candidate[
                "confidence"
            ]
        )


        support_cost = (

            0.0

            if candidate[
                "consensus"
            ]

            else

            1.0
        )


        score = (

            IMAGE_WEIGHT
            *
            image_cost

            +

            PITCH_WEIGHT
            *
            pitch_cost

            +

            CONFIDENCE_WEIGHT
            *
            confidence_cost

            +

            SUPPORT_WEIGHT
            *
            support_cost
        )


        options.append(
            (
                score,

                candidate,

                estimated_speed,

                image_residual,

                pitch_residual,
            )
        )


    if not options:

        return (
            None,
            "rejected_active",
            None,
            None,
            None,
        )


    options.sort(
        key=lambda item:
            item[0]
    )


    (
        score,
        candidate,
        speed,
        image_residual,
        pitch_residual,

    ) = options[0]


    return (

        candidate,

        "tracked",

        float(score),

        float(speed),

        (
            float(
                image_residual
            ),

            float(
                pitch_residual
            ),
        ),
    )


# ============================================================
# Reacquisition eligibility
# ============================================================

def candidate_reacquire_eligible(
    candidate,
):

    if candidate[
        "consensus"
    ]:

        return (
            candidate[
                "confidence"
            ]
            >=
            REACQUIRE_CONSENSUS_CONF
        )


    return (
        candidate[
            "confidence"
        ]
        >=
        REACQUIRE_SINGLE_CONF
    )


# ============================================================
# Select provisional reacquisition candidate
# ============================================================

def choose_reacquire_seed(
    candidates,
):

    eligible = [

        candidate

        for candidate
        in candidates

        if candidate_reacquire_eligible(
            candidate
        )
    ]


    if not eligible:

        return None


    # ========================================================
    # Priority:
    #
    # 1. consensus
    # 2. more detector support
    # 3. confidence
    # ========================================================

    eligible.sort(
        key=lambda candidate:
        (
            0
            if candidate[
                "consensus"
            ]
            else 1,

            -candidate[
                "support_count"
            ],

            -candidate[
                "confidence"
            ],
        )
    )


    return eligible[0]


# ============================================================
# Confirm reacquisition using next frame
# ============================================================

def confirm_pending_reacquire(
    candidates,
    pending,
    frame_idx,
    fps,
):

    if pending is None:

        return None


    frame_gap = (

        frame_idx

        -

        int(
            pending[
                "frame"
            ]
        )
    )


    if (
        frame_gap <= 0

        or

        frame_gap
        >
        REACQUIRE_CONFIRM_MAX_GAP
    ):

        return None


    dt = (
        frame_gap
        /
        fps
    )


    pending_image = np.array(
        [
            pending["image_x"],
            pending["image_y"]
        ],
        dtype=np.float32,
    )


    pending_pitch = np.array(
        [
            pending["pitch_x_m"],
            pending["pitch_y_m"]
        ],
        dtype=np.float32,
    )


    options = []


    for candidate in candidates:

        if not candidate_reacquire_eligible(
            candidate
        ):

            continue


        image_position = np.array(
            [
                candidate["image_x"],
                candidate["image_y"]
            ],
            dtype=np.float32,
        )


        pitch_position = np.array(
            [
                candidate["pitch_x_m"],
                candidate["pitch_y_m"]
            ],
            dtype=np.float32,
        )


        image_distance = float(
            np.linalg.norm(
                image_position
                -
                pending_image
            )
        )


        pitch_distance = float(
            np.linalg.norm(
                pitch_position
                -
                pending_pitch
            )
        )


        image_speed = (
            image_distance
            /
            dt
        )


        pitch_speed = (
            pitch_distance
            /
            dt
        )


        if (
            image_speed
            >
            REACQUIRE_IMAGE_SPEED_PX_S
        ):

            continue


        if (
            pitch_speed
            >
            REACQUIRE_PITCH_SPEED_MPS
        ):

            continue


        score = (

            0.55
            *
            (
                image_speed
                /
                REACQUIRE_IMAGE_SPEED_PX_S
            )

            +

            0.20
            *
            (
                pitch_speed
                /
                REACQUIRE_PITCH_SPEED_MPS
            )

            +

            0.15
            *
            (
                1.0
                -
                candidate[
                    "confidence"
                ]
            )

            +

            0.10
            *
            (
                0.0

                if candidate[
                    "consensus"
                ]

                else

                1.0
            )
        )


        options.append(
            (
                score,
                candidate
            )
        )


    if not options:

        return None


    options.sort(
        key=lambda item:
            item[0]
    )


    return options[0][1]


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


total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


writer = cv2.VideoWriter(

    OUTPUT_VIDEO,

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
        f"Cannot create video: "
        f"{OUTPUT_VIDEO}"
    )


# ============================================================
# State
# ============================================================

last_H = None

fallback_streak = 0


# Last two accepted observations
history = deque(
    maxlen=2
)


# Visualization trail
image_trail = deque(
    maxlen=30
)


# Provisional reacquisition
pending_reacquire = None


segment_id = -1


rows = []


frame_idx = 0


# ============================================================
# Statistics
# ============================================================

accepted_frames = 0

consensus_accepted_frames = 0

single_accepted_frames = 0

missing_frames = 0

pending_frames = 0

reacquisition_count = 0

homography_fallback_frames = 0


print("")
print(
    "========================================"
)

print(
    "BALL TRACKING V4"
)

print(
    "Multi-scale Consensus + Temporal Gating"
)

print(
    "========================================"
)

print("")

print(
    "Frames:",
    total_frames
)

print(
    "FPS:",
    fps
)

print(
    "NOTE: V4 runs tiled inference every frame."
)


# ============================================================
# Main Loop
# ============================================================

while True:

    ret, frame = cap.read()


    if not ret:

        break


    annotated = (
        frame.copy()
    )


    # ========================================================
    # 1. Homography
    # ========================================================

    (
        H,

        last_H,

        fallback_streak,

        used_homography_fallback,

        valid_kp_count,

        inlier_count,

        mean_error,

        median_error,

    ) = estimate_homography(
        frame,
        last_H,
        fallback_streak,
    )


    if (
        used_homography_fallback
    ):

        homography_fallback_frames += 1


    # ========================================================
    # 2. FULL + TILED every frame
    # ========================================================

    full_raw = (
        raw_full_detections(
            frame
        )
    )


    tile_raw = (
        raw_tiled_detections(
            frame
        )
    )


    candidates = (
        cluster_multiscale_candidates(
            full_raw
            +
            tile_raw,
            H,
        )
    )


    consensus_candidates = sum(

        1

        for candidate
        in candidates

        if candidate[
            "consensus"
        ]
    )


    # ========================================================
    # 3. Association state
    # ========================================================

    ball = None

    mode = "missing"

    association_score = None

    estimated_speed = None

    image_residual = None

    pitch_residual = None

    new_segment = False


    # ========================================================
    # Is old trajectory still active?
    # ========================================================

    active_track = False


    if len(history) > 0:

        gap_from_last = (

            frame_idx

            -

            int(
                history[-1][
                    "frame"
                ]
            )
        )


        active_track = (

            gap_from_last
            <=
            MAX_ASSOCIATION_GAP
        )


    # ========================================================
    # 4. Normal active tracking
    # ========================================================

    if active_track:

        (
            ball,

            mode,

            association_score,

            estimated_speed,

            residuals,

        ) = associate_active_track(
            candidates,
            frame_idx,
            fps,
            history,
        )


        if residuals is not None:

            (
                image_residual,
                pitch_residual
            ) = residuals


        if ball is not None:

            pending_reacquire = None


    # ========================================================
    # 5. Lost track:
    # two-frame confirmed reacquisition
    # ========================================================

    if (
        ball is None

        and

        not active_track
    ):

        confirmed = (
            confirm_pending_reacquire(
                candidates,
                pending_reacquire,
                frame_idx,
                fps,
            )
        )


        if confirmed is not None:

            ball = confirmed

            mode = (
                "reacquired_confirmed"
            )

            new_segment = True

            pending_reacquire = None

            reacquisition_count += 1


        else:

            seed = (
                choose_reacquire_seed(
                    candidates
                )
            )


            if seed is not None:

                pending_reacquire = {
                    **seed,

                    "frame":
                        frame_idx,
                }


                mode = (
                    "pending_reacquire"
                )


                pending_frames += 1


            else:

                pending_reacquire = None

                mode = (
                    "waiting_reacquire"
                )


    # ========================================================
    # Active trajectory:
    #
    # candidate rejected → leave missing.
    #
    # DO NOT instantly jump to another object.
    # ========================================================

    if (
        ball is None

        and

        active_track
    ):

        mode = (
            "short_gap_missing"
        )


    accepted = (
        ball is not None
    )


    # ========================================================
    # New trajectory segment
    # ========================================================

    if (
        accepted

        and

        new_segment
    ):

        segment_id += 1

        history.clear()

        image_trail.clear()


    # First accepted ball
    if (
        accepted

        and

        segment_id < 0
    ):

        segment_id = 0


    # ========================================================
    # Accepted ball
    # ========================================================

    if accepted:

        accepted_frames += 1


        if ball[
            "consensus"
        ]:

            consensus_accepted_frames += 1

        else:

            single_accepted_frames += 1


        # Update association history
        history.append(
            {

                "frame":
                    frame_idx,

                "image_x":
                    ball[
                        "image_x"
                    ],

                "image_y":
                    ball[
                        "image_y"
                    ],

                "pitch_x_m":
                    ball[
                        "pitch_x_m"
                    ],

                "pitch_y_m":
                    ball[
                        "pitch_y_m"
                    ],
            }
        )


        # Visual trail
        image_trail.append(
            (
                frame_idx,

                ball[
                    "image_x"
                ],

                ball[
                    "image_y"
                ],
            )
        )


    else:

        missing_frames += 1


    # ========================================================
    # 6. Draw all candidate centers
    #
    # Cyan = multi-scale consensus
    # Gray = single-scale
    # ========================================================

    for candidate in candidates:

        if candidate[
            "consensus"
        ]:

            color = (
                255,
                220,
                0
            )

        else:

            color = (
                150,
                150,
                150
            )


        cv2.circle(

            annotated,

            (
                int(
                    candidate[
                        "image_x"
                    ]
                ),

                int(
                    candidate[
                        "image_y"
                    ]
                ),
            ),

            4,

            color,

            1,
        )


    # ========================================================
    # Draw accepted ball
    # ========================================================

    if accepted:

        x1, y1, x2, y2 = (
            ball["box"]
        )


        cv2.rectangle(

            annotated,

            (
                int(x1),
                int(y1)
            ),

            (
                int(x2),
                int(y2)
            ),

            (
                0,
                255,
                255
            ),

            3,
        )


        cv2.circle(

            annotated,

            (
                int(
                    ball[
                        "image_x"
                    ]
                ),

                int(
                    ball[
                        "image_y"
                    ]
                ),
            ),

            7,

            (
                0,
                255,
                255
            ),

            -1,
        )


        support_label = (

            "CONSENSUS"

            if ball[
                "consensus"
            ]

            else

            "SINGLE"
        )


        cv2.putText(

            annotated,

            (
                f"BALL "
                f"{ball['confidence']:.2f} "
                f"[{support_label}]"
            ),

            (
                int(x1),

                max(
                    int(y1) - 10,
                    20
                ),
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (
                0,
                255,
                255
            ),

            2,

            cv2.LINE_AA,
        )


    # ========================================================
    # Pending reacquisition
    #
    # Magenta = candidate not trusted yet
    # ========================================================

    if (
        pending_reacquire
        is not None

        and

        not accepted
    ):

        cv2.circle(

            annotated,

            (
                int(
                    pending_reacquire[
                        "image_x"
                    ]
                ),

                int(
                    pending_reacquire[
                        "image_y"
                    ]
                ),
            ),

            8,

            (
                255,
                0,
                255
            ),

            2,
        )


        cv2.putText(

            annotated,

            "PENDING",

            (
                int(
                    pending_reacquire[
                        "image_x"
                    ]
                )
                +
                10,

                int(
                    pending_reacquire[
                        "image_y"
                    ]
                )
                -
                8,
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            (
                255,
                0,
                255
            ),

            2,

            cv2.LINE_AA,
        )


    # ========================================================
    # 7. Trail
    # ========================================================

    trail = list(
        image_trail
    )


    for i in range(
        1,
        len(trail)
    ):

        f1, x1, y1 = (
            trail[
                i - 1
            ]
        )


        f2, x2, y2 = (
            trail[i]
        )


        if (
            f2 - f1
            <=
            MAX_ASSOCIATION_GAP
        ):

            cv2.line(

                annotated,

                (
                    int(x1),
                    int(y1)
                ),

                (
                    int(x2),
                    int(y2)
                ),

                (
                    0,
                    255,
                    255
                ),

                2,
            )


    # ========================================================
    # 8. Radar
    # ========================================================

    if accepted:

        radar = draw_pitch(
            config=CONFIG
        )


        ball_xy = np.array(
            [
                [
                    ball[
                        "pitch_x_cm"
                    ],

                    ball[
                        "pitch_y_cm"
                    ],
                ]
            ],
            dtype=np.float32,
        )


        radar = draw_points_on_pitch(

            config=CONFIG,

            xy=ball_xy,

            face_color=(
                sv.Color.from_hex(
                    "#FFD700"
                )
            ),

            radius=20,

            pitch=radar,
        )


        radar = sv.resize_image(

            radar,

            (
                width // 3,
                height // 3
            ),
        )


        radar_h, radar_w = (
            radar.shape[:2]
        )


        rect = sv.Rect(

            x=
                width
                -
                radar_w
                -
                20,

            y=
                height
                -
                radar_h
                -
                20,

            width=
                radar_w,

            height=
                radar_h,
        )


        annotated = sv.draw_image(

            annotated,

            radar,

            opacity=0.80,

            rect=rect,
        )


    # ========================================================
    # 9. Overlay
    # ========================================================

    if accepted:

        status_color = (
            0,
            255,
            255
        )

    else:

        status_color = (
            0,
            120,
            255
        )


    cv2.putText(

        annotated,

        (
            f"Ball V4: "
            f"{mode} | "
            f"segment={segment_id}"
        ),

        (
            20,
            35
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        status_color,

        2,

        cv2.LINE_AA,
    )


    cv2.putText(

        annotated,

        (
            f"Full raw="
            f"{len(full_raw)} | "

            f"Tile raw="
            f"{len(tile_raw)} | "

            f"Merged="
            f"{len(candidates)} | "

            f"Consensus="
            f"{consensus_candidates}"
        ),

        (
            20,
            65
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.50,

        (
            255,
            255,
            255
        ),

        1,

        cv2.LINE_AA,
    )


    # ========================================================
    # 10. CSV
    # ========================================================

    rows.append(
        {

            "frame":
                frame_idx,

            "time_sec":
                frame_idx
                /
                fps,

            "full_raw_candidates":
                len(
                    full_raw
                ),

            "tile_raw_candidates":
                len(
                    tile_raw
                ),

            "merged_candidates":
                len(
                    candidates
                ),

            "consensus_candidates":
                consensus_candidates,

            "accepted":
                accepted,

            "association_mode":
                mode,

            "segment_id":
                (
                    segment_id

                    if accepted

                    else

                    None
                ),

            "confidence":
                (
                    ball[
                        "confidence"
                    ]

                    if accepted

                    else

                    None
                ),

            "consensus":
                (
                    ball[
                        "consensus"
                    ]

                    if accepted

                    else

                    None
                ),

            "support_count":
                (
                    ball[
                        "support_count"
                    ]

                    if accepted

                    else

                    None
                ),

            "sources":
                (
                    ball[
                        "sources"
                    ]

                    if accepted

                    else

                    None
                ),

            "image_x":
                (
                    ball[
                        "image_x"
                    ]

                    if accepted

                    else

                    None
                ),

            "image_y":
                (
                    ball[
                        "image_y"
                    ]

                    if accepted

                    else

                    None
                ),

            "pitch_x_m":
                (
                    ball[
                        "pitch_x_m"
                    ]

                    if accepted

                    else

                    None
                ),

            "pitch_y_m":
                (
                    ball[
                        "pitch_y_m"
                    ]

                    if accepted

                    else

                    None
                ),

            "estimated_speed_mps":
                estimated_speed,

            "image_prediction_residual_px":
                image_residual,

            "pitch_prediction_residual_m":
                pitch_residual,

            "association_score":
                association_score,

            "valid_keypoints":
                valid_kp_count,

            "inliers":
                inlier_count,

            "mean_reprojection_error_px":
                mean_error,

            "median_reprojection_error_px":
                median_error,

            "homography_fallback":
                used_homography_fallback,
        }
    )


    # ========================================================
    # Debug
    # ========================================================

    if (
        frame_idx % 30
        ==
        0
    ):

        print(

            f"frame={frame_idx:04d} | "

            f"accepted={accepted} | "

            f"mode={mode} | "

            f"full={len(full_raw)} | "

            f"tile={len(tile_raw)} | "

            f"merged={len(candidates)} | "

            f"consensus={consensus_candidates}"
        )


    writer.write(
        annotated
    )


    frame_idx += 1


# ============================================================
# Finish
# ============================================================

cap.release()

writer.release()


# ============================================================
# Save CSV
# ============================================================

columns = [

    "frame",

    "time_sec",

    "full_raw_candidates",

    "tile_raw_candidates",

    "merged_candidates",

    "consensus_candidates",

    "accepted",

    "association_mode",

    "segment_id",

    "confidence",

    "consensus",

    "support_count",

    "sources",

    "image_x",

    "image_y",

    "pitch_x_m",

    "pitch_y_m",

    "estimated_speed_mps",

    "image_prediction_residual_px",

    "pitch_prediction_residual_m",

    "association_score",

    "valid_keypoints",

    "inliers",

    "mean_reprojection_error_px",

    "median_reprojection_error_px",

    "homography_fallback",
]


with open(

    OUTPUT_CSV,

    "w",

    newline="",

    encoding="utf-8",

) as f:

    csv_writer = csv.DictWriter(

        f,

        fieldnames=columns,
    )


    csv_writer.writeheader()


    csv_writer.writerows(
        rows
    )


# ============================================================
# Summary
# ============================================================

accepted_rate = (

    100.0
    *
    accepted_frames
    /
    frame_idx

    if frame_idx > 0

    else 0.0
)


consensus_rate = (

    100.0
    *
    consensus_accepted_frames
    /
    accepted_frames

    if accepted_frames > 0

    else 0.0
)


print("")
print(
    "========================================"
)

print(
    "BALL TRACKING V4 SUMMARY"
)

print(
    "========================================"
)

print("")

print(
    "Frames:",
    frame_idx
)

print(
    "Accepted ball frames:",
    accepted_frames
)

print(
    "Accepted coverage:",
    f"{accepted_rate:.1f}%"
)

print("")

print(
    "Consensus accepted:",
    consensus_accepted_frames
)

print(
    "Single-scale accepted:",
    single_accepted_frames
)

print(
    "Consensus share:",
    f"{consensus_rate:.1f}%"
)

print("")

print(
    "Pending-reacquire frames:",
    pending_frames
)

print(
    "Confirmed reacquisitions:",
    reacquisition_count
)

print(
    "Missing frames:",
    missing_frames
)

print(
    "Trajectory segments:",
    (
        segment_id + 1

        if segment_id >= 0

        else 0
    )
)

print(
    "Homography fallback frames:",
    homography_fallback_frames
)


print("")
print(
    "Video:",
    OUTPUT_VIDEO
)

print(
    "CSV:",
    OUTPUT_CSV
)
