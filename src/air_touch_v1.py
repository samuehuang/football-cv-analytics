import os
import math

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO


# ============================================================
# Paths
# ============================================================

VIDEO_PATH = "videos/match.mp4"

PLAYER_MODEL_PATH = "models/football-player-detection.pt"

PLAYER_TRACK_CSV = "outputs/tracking_data_clean_v2.csv"
POSSESSION_CSV = "outputs/possession_v4_frames.csv"

OUTPUT_CSV = "outputs/air_touch_v1.csv"
OUTPUT_VIDEO = "outputs/air_touch_v1_qa.mp4"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Device
# ============================================================

DEVICE = "mps"


# ============================================================
# Football model classes
# ============================================================

BALL_CLASS = 0
GOALKEEPER_CLASS = 1
PLAYER_CLASS = 2
REFEREE_CLASS = 3


# ============================================================
# Detection
# ============================================================

PLAYER_CONF = 0.25
PLAYER_IMGSZ = 1280


# ============================================================
# Track ↔ detection box matching
#
# tracking CSV image_x/image_y is treated as the player
# ground / foot anchor.
#
# YOLO bbox bottom-center should therefore be nearby.
# ============================================================

MAX_TRACK_BOX_MATCH_PX = 55.0


# ============================================================
# Ball / player interaction zone
# ============================================================

# Horizontal expansion around bbox
BOX_X_MARGIN_RATIO = 0.18

# Slight vertical expansion
BOX_Y_MARGIN_RATIO = 0.10


# Normalized bbox regions:
#
# y = 0.0  -> head/top
# y = 1.0  -> feet/bottom
#
# An aerial touch can happen around:
# head / chest / shoulder.
AIR_ZONE_Y_MAX = 0.62


# ============================================================
# Trajectory-change evidence
# ============================================================

# Look 2 frames before and 2 frames after the candidate.
VELOCITY_OFFSET = 2


# Strong direction change
MIN_DIRECTION_CHANGE_DEG = 18.0


# Or meaningful speed change
MIN_SPEED_CHANGE_RATIO = 0.22


# Require ball displacement around touch to be real.
MIN_VECTOR_LENGTH_PX = 4.0


# ============================================================
# Candidate ball states
# ============================================================

ALLOWED_BALL_STATES = {
    "InTransit",
    "AerialContest",
}


# ============================================================
# Load
# ============================================================

print("")
print("Loading data...")


tracks = pd.read_csv(
    PLAYER_TRACK_CSV
)

possession = pd.read_csv(
    POSSESSION_CSV
)


# ============================================================
# Normalize player tracking
# ============================================================

for column in [
    "frame",
    "track_id",
    "image_x",
    "image_y",
]:

    tracks[column] = pd.to_numeric(
        tracks[column],
        errors="coerce"
    )


tracks = tracks.dropna(
    subset=[
        "frame",
        "track_id",
        "image_x",
        "image_y",
    ]
).copy()


tracks["frame"] = (
    tracks["frame"]
    .astype(int)
)


tracks = tracks[
    tracks["team"].isin(
        [
            "A",
            "B"
        ]
    )
].copy()


# ============================================================
# Reconstruct stable team assignment
#
# Same principle as possession_v3/v4.
# ============================================================

track_stats = []


for track_id, group in tracks.groupby(
    "track_id"
):

    counts = (
        group["team"]
        .value_counts()
    )


    if len(counts) == 0:
        continue


    dominant_team = counts.index[0]

    dominant_share = (
        counts.iloc[0]
        /
        len(group)
    )


    if dominant_share >= 0.70:

        track_stats.append(
            {
                "track_id":
                    track_id,

                "stable_team":
                    dominant_team,
            }
        )


stable_team_map = {
    row["track_id"]:
        row["stable_team"]

    for row in track_stats
}


tracks["stable_team"] = (
    tracks["track_id"]
    .map(
        stable_team_map
    )
)


tracks = tracks[
    tracks["stable_team"].notna()
].copy()


# Do not relabel contaminated rows.
tracks = tracks[
    tracks["team"]
    ==
    tracks["stable_team"]
].copy()


tracks_by_frame = {

    int(frame):
        group.copy()

    for frame, group
    in tracks.groupby("frame")
}


# ============================================================
# Normalize possession / ball data
# ============================================================

numeric_columns = [
    "frame",
    "time_sec",
    "ball_image_x",
    "ball_image_y",
    "ball_speed_mps",
]


for column in numeric_columns:

    possession[column] = pd.to_numeric(
        possession[column],
        errors="coerce"
    )


possession = possession.dropna(
    subset=["frame"]
).copy()


possession["frame"] = (
    possession["frame"]
    .astype(int)
)


possession_by_frame = (
    possession
    .set_index("frame")
)


# ============================================================
# Ball position lookup
# ============================================================

def get_ball_point(frame):

    if frame not in possession_by_frame.index:

        return None


    row = possession_by_frame.loc[
        frame
    ]


    if isinstance(
        row,
        pd.DataFrame
    ):

        row = row.iloc[0]


    x = row[
        "ball_image_x"
    ]

    y = row[
        "ball_image_y"
    ]


    if (
        pd.isna(x)
        or
        pd.isna(y)
    ):

        return None


    return np.array(
        [
            float(x),
            float(y)
        ],
        dtype=np.float32
    )


# ============================================================
# Velocity / deflection
# ============================================================

def calculate_deflection(frame):

    before_a = get_ball_point(
        frame - VELOCITY_OFFSET
    )

    before_b = get_ball_point(
        frame
    )

    after_a = get_ball_point(
        frame
    )

    after_b = get_ball_point(
        frame + VELOCITY_OFFSET
    )


    if any(
        point is None
        for point in [
            before_a,
            before_b,
            after_a,
            after_b,
        ]
    ):

        return {
            "valid": False,
            "direction_change_deg": np.nan,
            "speed_change_ratio": np.nan,
            "before_length_px": np.nan,
            "after_length_px": np.nan,
        }


    v_before = (
        before_b
        -
        before_a
    )


    v_after = (
        after_b
        -
        after_a
    )


    before_length = float(
        np.linalg.norm(
            v_before
        )
    )


    after_length = float(
        np.linalg.norm(
            v_after
        )
    )


    if (
        before_length
        <
        MIN_VECTOR_LENGTH_PX

        or

        after_length
        <
        MIN_VECTOR_LENGTH_PX
    ):

        return {
            "valid": False,
            "direction_change_deg": np.nan,
            "speed_change_ratio": np.nan,
            "before_length_px": before_length,
            "after_length_px": after_length,
        }


    cosine = float(
        np.dot(
            v_before,
            v_after
        )
        /
        (
            before_length
            *
            after_length
        )
    )


    cosine = float(
        np.clip(
            cosine,
            -1.0,
            1.0
        )
    )


    angle = math.degrees(
        math.acos(
            cosine
        )
    )


    speed_change_ratio = abs(
        after_length
        -
        before_length
    ) / max(
        before_length,
        1e-6
    )


    return {
        "valid": True,

        "direction_change_deg":
            float(angle),

        "speed_change_ratio":
            float(speed_change_ratio),

        "before_length_px":
            before_length,

        "after_length_px":
            after_length,
    }


# ============================================================
# YOLO player detection
# ============================================================

model = YOLO(
    PLAYER_MODEL_PATH
)


def detect_player_boxes(frame):

    result = model(
        frame,

        imgsz=PLAYER_IMGSZ,

        conf=PLAYER_CONF,

        classes=[
            GOALKEEPER_CLASS,
            PLAYER_CLASS,
        ],

        device=DEVICE,

        verbose=False
    )[0]


    boxes = result.boxes


    detections = []


    if (
        boxes is None
        or
        len(boxes) == 0
    ):

        return detections


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


    classes = (
        boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )


    for box, confidence, class_id in zip(
        xyxy,
        confidences,
        classes
    ):

        x1, y1, x2, y2 = box


        bottom_x = (
            x1 + x2
        ) / 2.0

        bottom_y = y2


        detections.append(
            {
                "box":
                    np.asarray(
                        [
                            x1,
                            y1,
                            x2,
                            y2
                        ],
                        dtype=np.float32
                    ),

                "bottom_x":
                    float(bottom_x),

                "bottom_y":
                    float(bottom_y),

                "confidence":
                    float(confidence),

                "class_id":
                    int(class_id),
            }
        )


    return detections


# ============================================================
# Match detection boxes to existing track IDs
# ============================================================

def match_tracks_to_boxes(
    frame_tracks,
    detections
):

    if (
        frame_tracks is None
        or
        len(frame_tracks) == 0
        or
        len(detections) == 0
    ):

        return []


    possible_matches = []


    track_rows = list(
        frame_tracks.iterrows()
    )


    for track_local_index, (_, track) in enumerate(
        track_rows
    ):

        tx = float(
            track["image_x"]
        )

        ty = float(
            track["image_y"]
        )


        for det_index, detection in enumerate(
            detections
        ):

            distance = float(
                np.hypot(
                    tx
                    -
                    detection["bottom_x"],

                    ty
                    -
                    detection["bottom_y"]
                )
            )


            if (
                distance
                <=
                MAX_TRACK_BOX_MATCH_PX
            ):

                possible_matches.append(
                    (
                        distance,
                        track_local_index,
                        det_index
                    )
                )


    possible_matches.sort(
        key=lambda item:
            item[0]
    )


    used_tracks = set()
    used_detections = set()

    matches = []


    for (
        distance,
        track_local_index,
        det_index
    ) in possible_matches:

        if (
            track_local_index
            in
            used_tracks
        ):

            continue


        if (
            det_index
            in
            used_detections
        ):

            continue


        used_tracks.add(
            track_local_index
        )

        used_detections.add(
            det_index
        )


        _, track = (
            track_rows[
                track_local_index
            ]
        )


        detection = (
            detections[
                det_index
            ]
        )


        matches.append(
            {
                "track_id":
                    float(
                        track[
                            "track_id"
                        ]
                    ),

                "team":
                    str(
                        track[
                            "stable_team"
                        ]
                    ),

                "box":
                    detection[
                        "box"
                    ],

                "box_confidence":
                    detection[
                        "confidence"
                    ],

                "match_distance_px":
                    distance,
            }
        )


    return matches


# ============================================================
# Is ball in player's aerial interaction zone?
# ============================================================

def aerial_zone_metrics(
    ball_x,
    ball_y,
    box
):

    x1, y1, x2, y2 = box


    width = max(
        float(
            x2 - x1
        ),
        1.0
    )

    height = max(
        float(
            y2 - y1
        ),
        1.0
    )


    expanded_x1 = (
        x1
        -
        width
        *
        BOX_X_MARGIN_RATIO
    )

    expanded_x2 = (
        x2
        +
        width
        *
        BOX_X_MARGIN_RATIO
    )


    expanded_y1 = (
        y1
        -
        height
        *
        BOX_Y_MARGIN_RATIO
    )

    expanded_y2 = y2


    inside_expanded_box = (

        expanded_x1
        <=
        ball_x
        <=
        expanded_x2

        and

        expanded_y1
        <=
        ball_y
        <=
        expanded_y2
    )


    normalized_x = (
        (
            ball_x
            -
            x1
        )
        /
        width
    )


    normalized_y = (
        (
            ball_y
            -
            y1
        )
        /
        height
    )


    aerial_zone = (

        inside_expanded_box

        and

        normalized_y
        <=
        AIR_ZONE_Y_MAX
    )


    # distance to approximate upper-body / head center
    head_center_x = (
        x1 + x2
    ) / 2.0

    head_center_y = (
        y1
        +
        0.22
        *
        height
    )


    head_distance = float(
        np.hypot(
            ball_x
            -
            head_center_x,

            ball_y
            -
            head_center_y
        )
    )


    return {
        "inside_expanded_box":
            inside_expanded_box,

        "aerial_zone":
            aerial_zone,

        "normalized_x":
            float(
                normalized_x
            ),

        "normalized_y":
            float(
                normalized_y
            ),

        "head_distance_px":
            head_distance,
    }


# ============================================================
# Video
# ============================================================

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
        f"Cannot create output: "
        f"{OUTPUT_VIDEO}"
    )


# ============================================================
# Main loop
# ============================================================

rows = []

frame_idx = 0


print("")
print(
    "========================================"
)

print(
    "AIR TOUCH V1"
)

print(
    "========================================"
)

print("")


while True:

    ret, frame = cap.read()


    if not ret:

        break


    annotated = frame.copy()


    if frame_idx not in possession_by_frame.index:

        writer.write(
            annotated
        )

        frame_idx += 1

        continue


    possession_row = (
        possession_by_frame
        .loc[
            frame_idx
        ]
    )


    if isinstance(
        possession_row,
        pd.DataFrame
    ):

        possession_row = (
            possession_row.iloc[0]
        )


    ball_state = str(
        possession_row[
            "ball_state_v4"
        ]
    )


    # Only evaluate aerial/transit states
    if (
        ball_state
        not in
        ALLOWED_BALL_STATES
    ):

        writer.write(
            annotated
        )

        frame_idx += 1

        continue


    ball_x = possession_row[
        "ball_image_x"
    ]

    ball_y = possession_row[
        "ball_image_y"
    ]


    if (
        pd.isna(ball_x)
        or
        pd.isna(ball_y)
    ):

        writer.write(
            annotated
        )

        frame_idx += 1

        continue


    ball_x = float(
        ball_x
    )

    ball_y = float(
        ball_y
    )


    # ========================================================
    # Player boxes
    # ========================================================

    detections = detect_player_boxes(
        frame
    )


    frame_tracks = (
        tracks_by_frame.get(
            frame_idx,
            None
        )
    )


    matched_players = (
        match_tracks_to_boxes(
            frame_tracks,
            detections
        )
    )


    # ========================================================
    # Ball trajectory deflection
    # ========================================================

    deflection = calculate_deflection(
        frame_idx
    )


    trajectory_change = False


    if deflection["valid"]:

        trajectory_change = (

            deflection[
                "direction_change_deg"
            ]
            >=
            MIN_DIRECTION_CHANGE_DEG

            or

            deflection[
                "speed_change_ratio"
            ]
            >=
            MIN_SPEED_CHANGE_RATIO
        )


    candidates = []


    # ========================================================
    # Player interaction candidates
    # ========================================================

    for player in matched_players:

        metrics = aerial_zone_metrics(

            ball_x,
            ball_y,

            player[
                "box"
            ]
        )


        if not metrics[
            "aerial_zone"
        ]:

            continue


        candidate = {
            "frame":
                frame_idx,

            "time_sec":
                frame_idx / fps,

            "team":
                player["team"],

            "track_id":
                player["track_id"],

            "ball_state_v4":
                ball_state,

            "possession_before":
                str(
                    possession_row[
                        "possession_v4"
                    ]
                ),

            "ball_speed_mps":
                possession_row[
                    "ball_speed_mps"
                ],

            "normalized_ball_x":
                metrics[
                    "normalized_x"
                ],

            "normalized_ball_y":
                metrics[
                    "normalized_y"
                ],

            "head_distance_px":
                metrics[
                    "head_distance_px"
                ],

            "direction_change_deg":
                deflection[
                    "direction_change_deg"
                ],

            "speed_change_ratio":
                deflection[
                    "speed_change_ratio"
                ],

            "trajectory_change":
                trajectory_change,

            "box_match_distance_px":
                player[
                    "match_distance_px"
                ],

            "box_confidence":
                player[
                    "box_confidence"
                ],
        }


        candidates.append(
            (
                candidate,
                player
            )
        )


    # ========================================================
    # Choose strongest candidate
    #
    # Require trajectory change.
    # ========================================================

    confirmed_candidates = [

        (
            candidate,
            player
        )

        for candidate, player
        in candidates

        if candidate[
            "trajectory_change"
        ]
    ]


    # Nearest to head/upper-body center
    confirmed_candidates.sort(
        key=lambda item:
            item[0][
                "head_distance_px"
            ]
    )


    confirmed = None
    confirmed_player = None


    if len(
        confirmed_candidates
    ) > 0:

        (
            confirmed,
            confirmed_player

        ) = confirmed_candidates[0]


        confirmed[
            "air_touch"
        ] = True


        rows.append(
            confirmed
        )


    # ========================================================
    # Draw ball
    # ========================================================

    cv2.circle(
        annotated,

        (
            int(ball_x),
            int(ball_y)
        ),

        10,

        (
            0,
            255,
            255
        ),

        3
    )


    # ========================================================
    # Draw candidate boxes
    # ========================================================

    for candidate, player in candidates:

        x1, y1, x2, y2 = (
            player[
                "box"
            ]
        )


        color = (
            0,
            165,
            255
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

            color,

            2
        )


        cv2.putText(
            annotated,

            (
                f"AIR ZONE "
                f"{player['team']} "
                f"#{int(player['track_id'])}"
            ),

            (
                int(x1),
                max(
                    20,
                    int(y1) - 8
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            color,

            2,

            cv2.LINE_AA
        )


    # ========================================================
    # Confirmed air touch
    # ========================================================

    if (
        confirmed is not None
        and
        confirmed_player is not None
    ):

        x1, y1, x2, y2 = (
            confirmed_player[
                "box"
            ]
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
                0
            ),

            4
        )


        cv2.putText(
            annotated,

            (
                f"AIR TOUCH "
                f"{confirmed['team']} "
                f"#{int(confirmed['track_id'])}"
            ),

            (
                int(x1),
                max(
                    30,
                    int(y1) - 15
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (
                0,
                255,
                0
            ),

            2,

            cv2.LINE_AA
        )


        cv2.putText(
            annotated,

            (
                f"angle="
                f"{confirmed['direction_change_deg']:.1f}deg | "
                f"speed-change="
                f"{confirmed['speed_change_ratio']:.2f}"
            ),

            (
                int(x1),
                max(
                    55,
                    int(y1) - 42
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            (
                0,
                255,
                0
            ),

            2,

            cv2.LINE_AA
        )


    # ========================================================
    # Overlay
    # ========================================================

    cv2.putText(
        annotated,

        (
            f"AIR TOUCH QA | "
            f"Ball state={ball_state}"
        ),

        (
            20,
            35
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (
            255,
            255,
            255
        ),

        2,

        cv2.LINE_AA
    )


    if deflection["valid"]:

        cv2.putText(
            annotated,

            (
                f"Direction change="
                f"{deflection['direction_change_deg']:.1f} deg | "
                f"Speed change="
                f"{deflection['speed_change_ratio']:.2f}"
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

            2,

            cv2.LINE_AA
        )


    writer.write(
        annotated
    )


    if frame_idx % 50 == 0:

        print(
            f"frame={frame_idx:04d} | "
            f"state={ball_state} | "
            f"matched_players={len(matched_players)} | "
            f"air_zone={len(candidates)} | "
            f"confirmed={confirmed is not None}"
        )


    frame_idx += 1


# ============================================================
# Finish
# ============================================================

cap.release()
writer.release()


# ============================================================
# CSV
# ============================================================

if len(rows) > 0:

    result = pd.DataFrame(
        rows
    )


    result.to_csv(
        OUTPUT_CSV,
        index=False
    )

else:

    result = pd.DataFrame()


# ============================================================
# Group adjacent frames into touch events
# ============================================================

events = []


if len(result) > 0:

    result = (
        result
        .sort_values("frame")
        .reset_index(drop=True)
    )


    current = None


    for _, row in result.iterrows():

        if current is None:

            current = {
                "team":
                    row["team"],

                "track_id":
                    row["track_id"],

                "start_frame":
                    int(
                        row["frame"]
                    ),

                "end_frame":
                    int(
                        row["frame"]
                    ),

                "max_direction_change_deg":
                    row[
                        "direction_change_deg"
                    ],

                "max_speed_change_ratio":
                    row[
                        "speed_change_ratio"
                    ],
            }

            continue


        same_player = (

            row["team"]
            ==
            current["team"]

            and

            float(
                row["track_id"]
            )
            ==
            float(
                current[
                    "track_id"
                ]
            )
        )


        close_frame = (

            int(
                row["frame"]
            )
            -
            current[
                "end_frame"
            ]
            <=
            2
        )


        if (
            same_player
            and
            close_frame
        ):

            current[
                "end_frame"
            ] = int(
                row["frame"]
            )


            current[
                "max_direction_change_deg"
            ] = max(
                current[
                    "max_direction_change_deg"
                ],

                row[
                    "direction_change_deg"
                ]
            )


            current[
                "max_speed_change_ratio"
            ] = max(
                current[
                    "max_speed_change_ratio"
                ],

                row[
                    "speed_change_ratio"
                ]
            )


        else:

            events.append(
                current
            )


            current = {
                "team":
                    row["team"],

                "track_id":
                    row["track_id"],

                "start_frame":
                    int(
                        row["frame"]
                    ),

                "end_frame":
                    int(
                        row["frame"]
                    ),

                "max_direction_change_deg":
                    row[
                        "direction_change_deg"
                    ],

                "max_speed_change_ratio":
                    row[
                        "speed_change_ratio"
                    ],
            }


    if current is not None:

        events.append(
            current
        )


# ============================================================
# Print
# ============================================================

print("")
print(
    "========================================"
)

print(
    "AIR TOUCH V1 SUMMARY"
)

print(
    "========================================"
)

print("")

print(
    "Confirmed air-touch frames:",
    len(result)
)

print(
    "Grouped air-touch events:",
    len(events)
)

print("")


if len(events) > 0:

    event_df = pd.DataFrame(
        events
    )


    event_df[
        "start_time_sec"
    ] = (
        event_df[
            "start_frame"
        ]
        /
        fps
    )


    event_df[
        "end_time_sec"
    ] = (
        event_df[
            "end_frame"
        ]
        /
        fps
    )


    print(
        event_df
        .round(2)
        .to_string(
            index=False
        )
    )


print("")

print(
    "CSV:",
    OUTPUT_CSV
)

print(
    "QA Video:",
    OUTPUT_VIDEO
)
