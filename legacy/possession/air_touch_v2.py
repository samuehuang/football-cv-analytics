import os
import math

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

VIDEO_PATH = "videos/match.mp4"

PLAYER_MODEL_PATH = "models/football-player-detection.pt"

PLAYER_TRACK_CSV = "outputs/tracking_data_clean_v2.csv"
POSSESSION_CSV = "outputs/possession_v4_frames.csv"

OUTPUT_CSV = "outputs/air_touch_v2.csv"
OUTPUT_REJECTED_CSV = "outputs/air_touch_v2_rejected.csv"
OUTPUT_VIDEO = "outputs/air_touch_v2_qa.mp4"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

DEVICE = "mps"


# ============================================================
# PLAYER MODEL CLASSES
# ============================================================

GOALKEEPER_CLASS = 1
PLAYER_CLASS = 2


# ============================================================
# DETECTION
# ============================================================

PLAYER_CONF = 0.25
PLAYER_IMGSZ = 1280


# ============================================================
# STABLE TEAM
# ============================================================

MIN_DOMINANT_TEAM_SHARE = 0.70


# ============================================================
# TRACK <-> BBOX MATCH
# ============================================================

MAX_TRACK_BOX_MATCH_PX = 55.0


# ============================================================
# UPPER BODY INTERACTION REGION
# ============================================================

BOX_X_MARGIN_RATIO = 0.18
BOX_Y_MARGIN_RATIO = 0.10

# 0.0 = top/head
# 1.0 = feet
AIR_ZONE_Y_MAX = 0.58


# ============================================================
# TRAJECTORY
# ============================================================

VELOCITY_OFFSET = 2

MIN_VECTOR_LENGTH_PX = 4.0


# ============================================================
# STRONG DEFLECTION RULE
#
# Speed change ALONE is NOT enough.
#
# Rule 1:
# direction >= 25 deg
#
# OR
#
# Rule 2:
# direction >= 18 deg
# AND
# speed change >= 0.35
# ============================================================

STRONG_DIRECTION_CHANGE_DEG = 25.0

COMBINED_DIRECTION_CHANGE_DEG = 18.0
COMBINED_SPEED_CHANGE_RATIO = 0.35


# ============================================================
# VERY STRONG AIR TOUCH OVERRIDE
#
# Important regression:
#
# B #10 at frame 281:
# speed ~= 20.91 m/s
# direction change ~= 137.56 deg
#
# This is strong enough evidence of an airborne touch that
# later reception-like geometry must NOT cancel it.
# ============================================================

VERY_STRONG_MIN_SPEED_MPS = 18.0

VERY_STRONG_DIRECTION_CHANGE_DEG = 60.0


# ============================================================
# RECEPTION SUPPRESSION
#
# If apparent upper-body interaction is followed by the SAME
# player becoming a reception candidate / controller, it is
# probably a reception sequence rather than an air touch.
#
# EXCEPTION:
# very strong air-touch evidence overrides this suppression.
# ============================================================

RECEPTION_LOOKAHEAD_FRAMES = 7


# ============================================================
# AIR TOUCH BALL STATES
# ============================================================

ALLOWED_BALL_STATES = {
    "InTransit",
    "AerialContest",
}


# ============================================================
# POSITIVE REGRESSION
# ============================================================

POSITIVE_REGRESSION_FRAME = 281
POSITIVE_REGRESSION_TEAM = "B"
POSITIVE_REGRESSION_ID = 10


# ============================================================
# LOAD DATA
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
# NORMALIZE PLAYER TRACKS
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
# STABLE TEAM RECONSTRUCTION
# ============================================================

stable_team_map = {}


for track_id, group in tracks.groupby(
    "track_id"
):

    counts = (
        group["team"]
        .value_counts()
    )


    if len(counts) == 0:

        continue


    dominant_team = (
        counts.index[0]
    )


    dominant_share = (
        counts.iloc[0]
        /
        len(group)
    )


    if (
        dominant_share
        >=
        MIN_DOMINANT_TEAM_SHARE
    ):

        stable_team_map[
            track_id
        ] = dominant_team


tracks["stable_team"] = (
    tracks["track_id"]
    .map(
        stable_team_map
    )
)


tracks = tracks[
    tracks[
        "stable_team"
    ].notna()
].copy()


# Never relabel contaminated rows.
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
# NORMALIZE POSSESSION
# ============================================================

numeric_columns = [

    "frame",
    "time_sec",

    "ball_image_x",
    "ball_image_y",

    "ball_speed_mps",

    "controller_id_v4",

    "reception_candidate_id",
    "reception_candidate_count",
]


for column in numeric_columns:

    if column in possession.columns:

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
# HELPERS
# ============================================================

def same_player_id(
    a,
    b
):

    if (
        pd.isna(a)
        or
        pd.isna(b)
    ):

        return False


    return (
        abs(
            float(a)
            -
            float(b)
        )
        <
        0.1
    )


def get_possession_row(
    frame
):

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


    return row


def get_ball_point(
    frame
):

    row = get_possession_row(
        frame
    )


    if row is None:

        return None


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
# TRAJECTORY DEFLECTION
# ============================================================

def calculate_deflection(
    frame
):

    p_before = get_ball_point(
        frame - VELOCITY_OFFSET
    )

    p_touch = get_ball_point(
        frame
    )

    p_after = get_ball_point(
        frame + VELOCITY_OFFSET
    )


    if any(
        point is None

        for point in [
            p_before,
            p_touch,
            p_after
        ]
    ):

        return {
            "valid":
                False,

            "direction_change_deg":
                np.nan,

            "speed_change_ratio":
                np.nan,

            "before_length_px":
                np.nan,

            "after_length_px":
                np.nan,
        }


    v_before = (
        p_touch
        -
        p_before
    )


    v_after = (
        p_after
        -
        p_touch
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
            "valid":
                False,

            "direction_change_deg":
                np.nan,

            "speed_change_ratio":
                np.nan,

            "before_length_px":
                before_length,

            "after_length_px":
                after_length,
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


    direction_change = math.degrees(
        math.acos(
            cosine
        )
    )


    speed_change_ratio = (

        abs(
            after_length
            -
            before_length
        )

        /

        max(
            before_length,
            1e-6
        )
    )


    return {
        "valid":
            True,

        "direction_change_deg":
            float(
                direction_change
            ),

        "speed_change_ratio":
            float(
                speed_change_ratio
            ),

        "before_length_px":
            before_length,

        "after_length_px":
            after_length,
    }


# ============================================================
# STRONG DEFLECTION
# ============================================================

def strong_air_deflection(
    deflection
):

    if not deflection[
        "valid"
    ]:

        return False


    direction = deflection[
        "direction_change_deg"
    ]

    speed_change = deflection[
        "speed_change_ratio"
    ]


    # --------------------------------------------------------
    # Rule 1
    # Strong direction change.
    # --------------------------------------------------------

    if (
        direction
        >=
        STRONG_DIRECTION_CHANGE_DEG
    ):

        return True


    # --------------------------------------------------------
    # Rule 2
    # Moderate direction change + meaningful speed change.
    # --------------------------------------------------------

    if (

        direction
        >=
        COMBINED_DIRECTION_CHANGE_DEG

        and

        speed_change
        >=
        COMBINED_SPEED_CHANGE_RATIO

    ):

        return True


    # Speed change alone is NOT sufficient.
    return False


# ============================================================
# VERY STRONG AIR TOUCH
# ============================================================

def very_strong_air_touch_evidence(
    ball_speed,
    deflection
):

    if not deflection[
        "valid"
    ]:

        return False


    if pd.isna(
        ball_speed
    ):

        return False


    return (

        float(
            ball_speed
        )
        >=
        VERY_STRONG_MIN_SPEED_MPS

        and

        deflection[
            "direction_change_deg"
        ]
        >=
        VERY_STRONG_DIRECTION_CHANGE_DEG
    )


# ============================================================
# FUTURE RECEPTION / CONTROL CHECK
# ============================================================

def followed_by_same_player_control(
    frame,
    team,
    track_id
):

    for future_frame in range(

        frame + 1,

        frame
        +
        RECEPTION_LOOKAHEAD_FRAMES
        +
        1
    ):

        row = get_possession_row(
            future_frame
        )


        if row is None:

            continue


        # ====================================================
        # Reception candidate
        # ====================================================

        candidate_team = str(
            row.get(
                "reception_candidate_team",
                None
            )
        )


        candidate_id = row.get(
            "reception_candidate_id",
            np.nan
        )


        candidate_count = row.get(
            "reception_candidate_count",
            0
        )


        if (

            candidate_team
            ==
            team

            and

            same_player_id(
                candidate_id,
                track_id
            )

            and

            pd.notna(
                candidate_count
            )

            and

            float(
                candidate_count
            )
            >
            0

        ):

            return True


        # ====================================================
        # Confirmed controller
        # ====================================================

        controller_team = str(
            row.get(
                "controller_team_v4",
                None
            )
        )


        controller_id = row.get(
            "controller_id_v4",
            np.nan
        )


        if (

            controller_team
            ==
            team

            and

            same_player_id(
                controller_id,
                track_id
            )

        ):

            return True


    return False


# ============================================================
# PLAYER DETECTOR
# ============================================================

model = YOLO(
    PLAYER_MODEL_PATH
)


def detect_player_boxes(
    frame
):

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


    for box, confidence in zip(
        xyxy,
        confidences
    ):

        x1, y1, x2, y2 = box


        detections.append(
            {
                "box":
                    np.asarray(
                        box,
                        dtype=np.float32
                    ),

                "bottom_x":
                    float(
                        (
                            x1 + x2
                        )
                        /
                        2.0
                    ),

                "bottom_y":
                    float(
                        y2
                    ),

                "confidence":
                    float(
                        confidence
                    ),
            }
        )


    return detections


# ============================================================
# MATCH TRACK <-> BBOX
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


    track_rows = list(
        frame_tracks.iterrows()
    )


    possible = []


    for local_track_index, (_, track) in enumerate(
        track_rows
    ):

        track_x = float(
            track["image_x"]
        )

        track_y = float(
            track["image_y"]
        )


        for detection_index, detection in enumerate(
            detections
        ):

            distance = float(
                np.hypot(

                    track_x
                    -
                    detection[
                        "bottom_x"
                    ],

                    track_y
                    -
                    detection[
                        "bottom_y"
                    ]
                )
            )


            if (
                distance
                <=
                MAX_TRACK_BOX_MATCH_PX
            ):

                possible.append(
                    (
                        distance,
                        local_track_index,
                        detection_index
                    )
                )


    possible.sort(
        key=lambda item:
            item[0]
    )


    used_tracks = set()
    used_detections = set()

    matches = []


    for (
        distance,
        local_track_index,
        detection_index

    ) in possible:

        if (
            local_track_index
            in
            used_tracks
        ):

            continue


        if (
            detection_index
            in
            used_detections
        ):

            continue


        used_tracks.add(
            local_track_index
        )

        used_detections.add(
            detection_index
        )


        _, track = track_rows[
            local_track_index
        ]


        detection = detections[
            detection_index
        ]


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
# AERIAL INTERACTION REGION
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


    inside = (

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
            ball_x - x1
        )
        /
        width
    )


    normalized_y = (
        (
            ball_y - y1
        )
        /
        height
    )


    upper_body = (

        inside

        and

        normalized_y
        <=
        AIR_ZONE_Y_MAX
    )


    upper_body_center_x = (
        x1 + x2
    ) / 2.0


    upper_body_center_y = (
        y1
        +
        0.25
        *
        height
    )


    upper_distance = float(
        np.hypot(

            ball_x
            -
            upper_body_center_x,

            ball_y
            -
            upper_body_center_y
        )
    )


    return {
        "inside":
            inside,

        "upper_body":
            upper_body,

        "normalized_x":
            float(
                normalized_x
            ),

        "normalized_y":
            float(
                normalized_y
            ),

        "upper_distance_px":
            upper_distance,
    }


# ============================================================
# VIDEO SETUP
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
# MAIN
# ============================================================

confirmed_rows = []
rejected_rows = []

frame_idx = 0


print("")
print(
    "========================================"
)

print(
    "AIR TOUCH V2"
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


    possession_row = get_possession_row(
        frame_idx
    )


    if possession_row is None:

        writer.write(
            annotated
        )

        frame_idx += 1

        continue


    ball_state = str(
        possession_row[
            "ball_state_v4"
        ]
    )


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


    ball_speed = possession_row[
        "ball_speed_mps"
    ]


    # ========================================================
    # PLAYER BBOX
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


    players = match_tracks_to_boxes(
        frame_tracks,
        detections
    )


    # ========================================================
    # TRAJECTORY
    # ========================================================

    deflection = calculate_deflection(
        frame_idx
    )


    strong_deflection = (
        strong_air_deflection(
            deflection
        )
    )


    very_strong = (
        very_strong_air_touch_evidence(
            ball_speed,
            deflection
        )
    )


    # ========================================================
    # CONTACT CANDIDATES
    # ========================================================

    candidates = []


    for player in players:

        metrics = aerial_zone_metrics(

            ball_x,
            ball_y,

            player[
                "box"
            ]
        )


        if not metrics[
            "upper_body"
        ]:

            continue


        followed_by_control = (
            followed_by_same_player_control(

                frame_idx,

                player[
                    "team"
                ],

                player[
                    "track_id"
                ]
            )
        )


        reject_reason = None


        # ----------------------------------------------------
        # No valid trajectory
        # ----------------------------------------------------

        if not deflection[
            "valid"
        ]:

            reject_reason = (
                "no_valid_trajectory"
            )


        # ----------------------------------------------------
        # Not enough deflection evidence
        # ----------------------------------------------------

        elif not strong_deflection:

            reject_reason = (
                "weak_deflection"
            )


        # ----------------------------------------------------
        # Reception suppression
        #
        # IMPORTANT:
        # VERY STRONG air-touch evidence overrides this.
        # ----------------------------------------------------

        elif (
            followed_by_control
            and
            not very_strong
        ):

            reject_reason = (
                "followed_by_same_player_reception"
            )


        confirmed = (
            reject_reason
            is None
        )


        row = {
            "frame":
                frame_idx,

            "time_sec":
                frame_idx / fps,

            "team":
                player[
                    "team"
                ],

            "track_id":
                player[
                    "track_id"
                ],

            "ball_state_v4":
                ball_state,

            "possession_v4":
                possession_row[
                    "possession_v4"
                ],

            "ball_speed_mps":
                ball_speed,

            "normalized_ball_x":
                metrics[
                    "normalized_x"
                ],

            "normalized_ball_y":
                metrics[
                    "normalized_y"
                ],

            "upper_distance_px":
                metrics[
                    "upper_distance_px"
                ],

            "direction_change_deg":
                deflection[
                    "direction_change_deg"
                ],

            "speed_change_ratio":
                deflection[
                    "speed_change_ratio"
                ],

            "strong_deflection":
                strong_deflection,

            "very_strong_air_touch":
                very_strong,

            "followed_by_same_player_control":
                followed_by_control,

            "confirmed_air_touch":
                confirmed,

            "reject_reason":
                reject_reason,

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
                row,
                player
            )
        )


    # ========================================================
    # MULTIPLE CANDIDATES
    #
    # Choose player nearest to upper-body interaction center.
    # ========================================================

    confirmed_candidates = [

        (
            row,
            player
        )

        for row, player
        in candidates

        if row[
            "confirmed_air_touch"
        ]
    ]


    confirmed_candidates.sort(
        key=lambda item:
            item[0][
                "upper_distance_px"
            ]
    )


    selected = None
    selected_player = None


    if len(
        confirmed_candidates
    ) > 0:

        (
            selected,
            selected_player

        ) = confirmed_candidates[0]


        confirmed_rows.append(
            selected
        )


    # ========================================================
    # SAVE REJECTED
    # ========================================================

    for row, _ in candidates:

        if not row[
            "confirmed_air_touch"
        ]:

            rejected_rows.append(
                row
            )


    # ========================================================
    # DRAW BALL
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
    # DRAW CANDIDATES
    # ========================================================

    for row, player in candidates:

        x1, y1, x2, y2 = (
            player[
                "box"
            ]
        )


        if row[
            "confirmed_air_touch"
        ]:

            color = (
                0,
                255,
                0
            )

        else:

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


        if row[
            "confirmed_air_touch"
        ]:

            if row[
                "very_strong_air_touch"
            ]:

                label = (
                    f"VERY STRONG AIR TOUCH "
                    f"{row['team']} "
                    f"#{int(row['track_id'])}"
                )

            else:

                label = (
                    f"AIR TOUCH "
                    f"{row['team']} "
                    f"#{int(row['track_id'])}"
                )

        else:

            label = (
                f"REJECT "
                f"{row['reject_reason']}"
            )


        cv2.putText(
            annotated,

            label,

            (
                int(x1),
                max(
                    22,
                    int(y1) - 8
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.42,

            color,

            2,

            cv2.LINE_AA
        )


    # ========================================================
    # OVERLAY
    # ========================================================

    cv2.putText(
        annotated,

        (
            f"AIR TOUCH V2 | "
            f"state={ball_state}"
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


    if deflection[
        "valid"
    ]:

        cv2.putText(
            annotated,

            (
                f"Direction="
                f"{deflection['direction_change_deg']:.1f} deg | "
                f"Speed-change="
                f"{deflection['speed_change_ratio']:.2f} | "
                f"Strong={strong_deflection} | "
                f"VeryStrong={very_strong}"
            ),

            (
                20,
                65
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.46,

            (
                255,
                255,
                255
            ),

            2,

            cv2.LINE_AA
        )


    if pd.notna(
        ball_speed
    ):

        cv2.putText(
            annotated,

            (
                f"Ball speed="
                f"{float(ball_speed):.2f} m/s"
            ),

            (
                20,
                93
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.46,

            (
                255,
                255,
                255
            ),

            2,

            cv2.LINE_AA
        )


    # ========================================================
    # REGRESSION FRAME 281
    # ========================================================

    if (
        frame_idx
        ==
        POSITIVE_REGRESSION_FRAME
    ):

        regression_ok = False


        if selected is not None:

            regression_ok = (

                selected[
                    "team"
                ]
                ==
                POSITIVE_REGRESSION_TEAM

                and

                same_player_id(
                    selected[
                        "track_id"
                    ],
                    POSITIVE_REGRESSION_ID
                )
            )


        regression_text = (

            "REGRESSION PASS | B #10 HEADER"

            if regression_ok

            else

            "REGRESSION FAIL | B #10 HEADER"
        )


        regression_color = (

            (
                0,
                255,
                0
            )

            if regression_ok

            else

            (
                0,
                0,
                255
            )
        )


        cv2.rectangle(
            annotated,

            (
                20,
                110
            ),

            (
                650,
                160
            ),

            regression_color,

            -1
        )


        cv2.putText(
            annotated,

            regression_text,

            (
                35,
                144
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (
                0,
                0,
                0
            ),

            2,

            cv2.LINE_AA
        )


    writer.write(
        annotated
    )


    frame_idx += 1


# ============================================================
# FINISH VIDEO
# ============================================================

cap.release()
writer.release()


# ============================================================
# DATAFRAMES
# ============================================================

confirmed_df = pd.DataFrame(
    confirmed_rows
)


rejected_df = pd.DataFrame(
    rejected_rows
)


confirmed_df.to_csv(
    OUTPUT_CSV,
    index=False
)


rejected_df.to_csv(
    OUTPUT_REJECTED_CSV,
    index=False
)


# ============================================================
# GROUP CONFIRMED FRAMES INTO EVENTS
# ============================================================

events = []


if len(
    confirmed_df
) > 0:

    confirmed_df = (
        confirmed_df
        .sort_values("frame")
        .reset_index(drop=True)
    )


    current = None


    for _, row in (
        confirmed_df.iterrows()
    ):

        if current is None:

            current = {
                "team":
                    row[
                        "team"
                    ],

                "track_id":
                    row[
                        "track_id"
                    ],

                "start_frame":
                    int(
                        row[
                            "frame"
                        ]
                    ),

                "end_frame":
                    int(
                        row[
                            "frame"
                        ]
                    ),

                "max_direction_change_deg":
                    row[
                        "direction_change_deg"
                    ],

                "max_speed_change_ratio":
                    row[
                        "speed_change_ratio"
                    ],

                "max_ball_speed_mps":
                    row[
                        "ball_speed_mps"
                    ],

                "has_very_strong_frame":
                    bool(
                        row[
                            "very_strong_air_touch"
                        ]
                    ),
            }

            continue


        same_player = (

            row[
                "team"
            ]
            ==
            current[
                "team"
            ]

            and

            same_player_id(
                row[
                    "track_id"
                ],
                current[
                    "track_id"
                ]
            )
        )


        close_frame = (

            int(
                row[
                    "frame"
                ]
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
                row[
                    "frame"
                ]
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


            if pd.notna(
                row[
                    "ball_speed_mps"
                ]
            ):

                if pd.isna(
                    current[
                        "max_ball_speed_mps"
                    ]
                ):

                    current[
                        "max_ball_speed_mps"
                    ] = row[
                        "ball_speed_mps"
                    ]

                else:

                    current[
                        "max_ball_speed_mps"
                    ] = max(

                        current[
                            "max_ball_speed_mps"
                        ],

                        row[
                            "ball_speed_mps"
                        ]
                    )


            current[
                "has_very_strong_frame"
            ] = (

                current[
                    "has_very_strong_frame"
                ]

                or

                bool(
                    row[
                        "very_strong_air_touch"
                    ]
                )
            )


        else:

            events.append(
                current
            )


            current = {
                "team":
                    row[
                        "team"
                    ],

                "track_id":
                    row[
                        "track_id"
                    ],

                "start_frame":
                    int(
                        row[
                            "frame"
                        ]
                    ),

                "end_frame":
                    int(
                        row[
                            "frame"
                        ]
                    ),

                "max_direction_change_deg":
                    row[
                        "direction_change_deg"
                    ],

                "max_speed_change_ratio":
                    row[
                        "speed_change_ratio"
                    ],

                "max_ball_speed_mps":
                    row[
                        "ball_speed_mps"
                    ],

                "has_very_strong_frame":
                    bool(
                        row[
                            "very_strong_air_touch"
                        ]
                    ),
            }


    if current is not None:

        events.append(
            current
        )


# ============================================================
# REGRESSION CHECK
# ============================================================

positive_match = False


if len(
    confirmed_df
) > 0:

    positive_rows = confirmed_df[
        confirmed_df[
            "frame"
        ]
        ==
        POSITIVE_REGRESSION_FRAME
    ]


    for _, row in (
        positive_rows.iterrows()
    ):

        if (

            row[
                "team"
            ]
            ==
            POSITIVE_REGRESSION_TEAM

            and

            same_player_id(
                row[
                    "track_id"
                ],
                POSITIVE_REGRESSION_ID
            )
        ):

            positive_match = True
            break


# ============================================================
# SUMMARY
# ============================================================

print("")
print(
    "========================================"
)

print(
    "AIR TOUCH V2 SUMMARY"
)

print(
    "========================================"
)

print("")

print(
    "Confirmed air-touch frames:",
    len(
        confirmed_df
    )
)

print(
    "Rejected interaction frames:",
    len(
        rejected_df
    )
)

print(
    "Grouped confirmed events:",
    len(
        events
    )
)


print("")

print(
    "B #10 @ frame 281 regression:",
    (
        "PASS"
        if positive_match
        else
        "FAIL"
    )
)


if len(
    events
) > 0:

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


    print("")

    print(
        event_df
        .round(2)
        .to_string(
            index=False
        )
    )


# ============================================================
# REJECTION REASONS
# ============================================================

if len(
    rejected_df
) > 0:

    print("")

    print(
        "Rejected reasons:"
    )

    print(
        rejected_df[
            "reject_reason"
        ]
        .value_counts()
        .to_string()
    )


# ============================================================
# VERY STRONG EVENTS
# ============================================================

if len(
    confirmed_df
) > 0:

    very_strong_df = confirmed_df[
        confirmed_df[
            "very_strong_air_touch"
        ]
        ==
        True
    ]


    print("")

    print(
        "Very-strong air-touch frames:",
        len(
            very_strong_df
        )
    )


    if len(
        very_strong_df
    ) > 0:

        print("")

        print(
            very_strong_df[
                [
                    "frame",
                    "time_sec",
                    "team",
                    "track_id",
                    "ball_speed_mps",
                    "direction_change_deg",
                    "speed_change_ratio",
                    "followed_by_same_player_control",
                ]
            ]
            .round(2)
            .to_string(
                index=False
            )
        )


print("")

print(
    "Confirmed CSV:",
    OUTPUT_CSV
)

print(
    "Rejected CSV:",
    OUTPUT_REJECTED_CSV
)

print(
    "QA Video:",
    OUTPUT_VIDEO
)