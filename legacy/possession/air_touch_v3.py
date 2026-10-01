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

OUTPUT_INTERACTION_CSV = "outputs/air_touch_v3_interactions.csv"
OUTPUT_EVENT_CSV = "outputs/air_touch_v3_events.csv"
OUTPUT_CONFIRMED_CSV = "outputs/air_touch_v3_confirmed_events.csv"
OUTPUT_VIDEO = "outputs/air_touch_v3_qa.mp4"

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

# normalized bbox Y:
# 0.0 = top of head
# 1.0 = feet
AIR_ZONE_Y_MAX = 0.58


# ============================================================
# BALL TRAJECTORY
# ============================================================

VELOCITY_OFFSET = 2

MIN_VECTOR_LENGTH_PX = 4.0


# ============================================================
# STRONG DEFLECTION
#
# Speed change ALONE cannot create AirTouch.
# ============================================================

STRONG_DIRECTION_CHANGE_DEG = 25.0

COMBINED_DIRECTION_CHANGE_DEG = 18.0
COMBINED_SPEED_CHANGE_RATIO = 0.35


# ============================================================
# VERY STRONG AIR TOUCH
#
# Current high-confidence header example:
#
# B #10 @ frame 281
# speed ~= 20.91 m/s
# direction ~= 137.56 deg
#
# Very-strong evidence overrides reception-like geometry.
# ============================================================

VERY_STRONG_MIN_SPEED_MPS = 18.0
VERY_STRONG_DIRECTION_CHANGE_DEG = 60.0


# ============================================================
# RECEPTION LOOKAHEAD
# ============================================================

RECEPTION_LOOKAHEAD_FRAMES = 7


# ============================================================
# EVENT GROUPING
#
# Same player interaction frames separated by <= 2 frames
# belong to the same physical interaction event.
# ============================================================

MAX_EVENT_FRAME_GAP = 2


# ============================================================
# BALL STATES
# ============================================================

ALLOWED_BALL_STATES = {
    "InTransit",
    "AerialContest",
}


# ============================================================
# REGRESSION TESTS
# ============================================================

# Negative:
# A #6 around 8.12s is ball bounce / reception,
# NOT AirTouch.
NEGATIVE_FRAME = 203
NEGATIVE_TEAM = "A"
NEGATIVE_TRACK_ID = 6


# Positive:
# B #10 at 11.24s is real header / air touch.
POSITIVE_FRAME = 281
POSITIVE_TEAM = "B"
POSITIVE_TRACK_ID = 10


# ============================================================
# LOAD
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
# NORMALIZE PLAYER TRACK DATA
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
            "B",
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


    dominant_team = counts.index[0]

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


# Never relabel inconsistent rows.
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
# NORMALIZE POSSESSION DATA
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
    subset=[
        "frame"
    ]
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

    if (
        frame
        not in
        possession_by_frame.index
    ):

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


    x = row.get(
        "ball_image_x",
        np.nan
    )

    y = row.get(
        "ball_image_y",
        np.nan
    )


    if (
        pd.isna(x)
        or
        pd.isna(y)
    ):

        return None


    return np.asarray(
        [
            float(x),
            float(y),
        ],
        dtype=np.float32
    )


# ============================================================
# TRAJECTORY DEFLECTION
# ============================================================

def calculate_deflection(
    frame
):

    before = get_ball_point(
        frame
        -
        VELOCITY_OFFSET
    )

    touch = get_ball_point(
        frame
    )

    after = get_ball_point(
        frame
        +
        VELOCITY_OFFSET
    )


    if any(
        p is None

        for p in [
            before,
            touch,
            after,
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


    vector_before = (
        touch
        -
        before
    )


    vector_after = (
        after
        -
        touch
    )


    length_before = float(
        np.linalg.norm(
            vector_before
        )
    )


    length_after = float(
        np.linalg.norm(
            vector_after
        )
    )


    if (
        length_before
        <
        MIN_VECTOR_LENGTH_PX

        or

        length_after
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
                length_before,

            "after_length_px":
                length_after,
        }


    cosine = float(

        np.dot(
            vector_before,
            vector_after
        )

        /

        (
            length_before
            *
            length_after
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
            length_after
            -
            length_before
        )

        /

        max(
            length_before,
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
            length_before,

        "after_length_px":
            length_after,
    }


# ============================================================
# STRONG DEFLECTION
# ============================================================

def is_strong_deflection(
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


    if (
        direction
        >=
        STRONG_DIRECTION_CHANGE_DEG
    ):

        return True


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


    return False


# ============================================================
# VERY STRONG AIR TOUCH
# ============================================================

def is_very_strong_air_touch(
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
# FUTURE SAME-PLAYER RECEPTION / CONTROL
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


        # ----------------------------------------------------
        # Reception candidate
        # ----------------------------------------------------

        candidate_team = row.get(
            "reception_candidate_team",
            None
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
            str(candidate_team)
            ==
            str(team)

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


        # ----------------------------------------------------
        # Confirmed controller
        # ----------------------------------------------------

        controller_team = row.get(
            "controller_team_v4",
            None
        )


        controller_id = row.get(
            "controller_id_v4",
            np.nan
        )


        if (
            str(controller_team)
            ==
            str(team)

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

        verbose=False,

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
                        [
                            x1,
                            y1,
                            x2,
                            y2,
                        ],
                        dtype=np.float32
                    ),

                "bottom_x":
                    float(
                        (
                            x1
                            +
                            x2
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
# TRACK <-> DETECTION MATCHING
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


    possible_matches = []


    for track_local_index, (_, track) in enumerate(
        track_rows
    ):

        tx = float(
            track[
                "image_x"
            ]
        )

        ty = float(
            track[
                "image_y"
            ]
        )


        for detection_index, detection in enumerate(
            detections
        ):

            distance = float(
                np.hypot(

                    tx
                    -
                    detection[
                        "bottom_x"
                    ],

                    ty
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

                possible_matches.append(
                    (
                        distance,
                        track_local_index,
                        detection_index,
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
        detection_index

    ) in possible_matches:

        if (
            track_local_index
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
            track_local_index
        )

        used_detections.add(
            detection_index
        )


        _, track = track_rows[
            track_local_index
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
                    float(
                        distance
                    ),
            }
        )


    return matches


# ============================================================
# UPPER-BODY ZONE
# ============================================================

def upper_body_metrics(
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
        BOX_X_MARGIN_RATIO
        *
        width
    )


    expanded_x2 = (
        x2
        +
        BOX_X_MARGIN_RATIO
        *
        width
    )


    expanded_y1 = (
        y1
        -
        BOX_Y_MARGIN_RATIO
        *
        height
    )


    expanded_y2 = y2


    inside_expanded = (

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

        inside_expanded

        and

        normalized_y
        <=
        AIR_ZONE_Y_MAX
    )


    upper_center_x = (
        x1
        +
        0.5
        *
        width
    )


    upper_center_y = (
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
            upper_center_x,

            ball_y
            -
            upper_center_y
        )
    )


    return {
        "inside_expanded":
            inside_expanded,

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
# FIRST PASS:
# BUILD FRAME-LEVEL INTERACTION CANDIDATES
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)


if not cap.isOpened():

    raise RuntimeError(
        f"Cannot open video: "
        f"{VIDEO_PATH}"
    )


fps = float(
    cap.get(
        cv2.CAP_PROP_FPS
    )
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


interaction_rows = []

frame_idx = 0


print("")
print(
    "========================================"
)

print(
    "AIR TOUCH V3 - PASS 1"
)

print(
    "Frame-level interaction extraction"
)

print(
    "========================================"
)

print("")


while True:

    ret, frame = cap.read()


    if not ret:

        break


    possession_row = get_possession_row(
        frame_idx
    )


    if possession_row is None:

        frame_idx += 1
        continue


    ball_state = str(
        possession_row.get(
            "ball_state_v4",
            ""
        )
    )


    if (
        ball_state
        not in
        ALLOWED_BALL_STATES
    ):

        frame_idx += 1
        continue


    ball_x = possession_row.get(
        "ball_image_x",
        np.nan
    )


    ball_y = possession_row.get(
        "ball_image_y",
        np.nan
    )


    if (
        pd.isna(ball_x)
        or
        pd.isna(ball_y)
    ):

        frame_idx += 1
        continue


    ball_x = float(
        ball_x
    )

    ball_y = float(
        ball_y
    )


    ball_speed = possession_row.get(
        "ball_speed_mps",
        np.nan
    )


    detections = detect_player_boxes(
        frame
    )


    frame_tracks = tracks_by_frame.get(
        frame_idx,
        None
    )


    matched_players = match_tracks_to_boxes(
        frame_tracks,
        detections
    )


    deflection = calculate_deflection(
        frame_idx
    )


    strong_deflection = is_strong_deflection(
        deflection
    )


    very_strong = is_very_strong_air_touch(
        ball_speed,
        deflection
    )


    for player in matched_players:

        zone = upper_body_metrics(

            ball_x,
            ball_y,

            player[
                "box"
            ]
        )


        if not zone[
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


        x1, y1, x2, y2 = (
            player[
                "box"
            ]
        )


        interaction_rows.append(
            {
                "frame":
                    frame_idx,

                "time_sec":
                    frame_idx
                    /
                    fps,

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
                    possession_row.get(
                        "possession_v4",
                        None
                    ),

                "ball_speed_mps":
                    ball_speed,

                "ball_image_x":
                    ball_x,

                "ball_image_y":
                    ball_y,

                "bbox_x1":
                    float(x1),

                "bbox_y1":
                    float(y1),

                "bbox_x2":
                    float(x2),

                "bbox_y2":
                    float(y2),

                "normalized_ball_x":
                    zone[
                        "normalized_x"
                    ],

                "normalized_ball_y":
                    zone[
                        "normalized_y"
                    ],

                "upper_distance_px":
                    zone[
                        "upper_distance_px"
                    ],

                "trajectory_valid":
                    deflection[
                        "valid"
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

                "box_match_distance_px":
                    player[
                        "match_distance_px"
                    ],

                "box_confidence":
                    player[
                        "box_confidence"
                    ],
            }
        )


    if (
        frame_idx
        %
        100
        ==
        0
    ):

        print(
            f"frame={frame_idx:04d} | "
            f"interactions={len(interaction_rows)}"
        )


    frame_idx += 1


cap.release()


# ============================================================
# INTERACTION DATAFRAME
# ============================================================

interactions = pd.DataFrame(
    interaction_rows
)


if len(
    interactions
) == 0:

    print("")
    print(
        "No interaction candidates found."
    )

    interactions.to_csv(
        OUTPUT_INTERACTION_CSV,
        index=False
    )

    raise SystemExit


interactions = (
    interactions
    .sort_values(
        [
            "team",
            "track_id",
            "frame",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# EVENT GROUPING
#
# IMPORTANT:
# We group ALL upper-body interaction candidates first.
#
# We do NOT decide AirTouch at frame level anymore.
# ============================================================

temporary_events = []


for (
    team,
    track_id
), group in interactions.groupby(
    [
        "team",
        "track_id",
    ],
    sort=False
):

    group = (
        group
        .sort_values(
            "frame"
        )
    )


    current_indices = []

    previous_frame = None


    for row_index, row in group.iterrows():

        frame = int(
            row[
                "frame"
            ]
        )


        if (
            previous_frame
            is None
        ):

            current_indices = [
                row_index
            ]


        elif (
            frame
            -
            previous_frame
            <=
            MAX_EVENT_FRAME_GAP
        ):

            current_indices.append(
                row_index
            )


        else:

            temporary_events.append(
                {
                    "team":
                        team,

                    "track_id":
                        float(
                            track_id
                        ),

                    "member_indices":
                        current_indices.copy(),
                }
            )


            current_indices = [
                row_index
            ]


        previous_frame = frame


    if len(
        current_indices
    ) > 0:

        temporary_events.append(
            {
                "team":
                    team,

                "track_id":
                    float(
                        track_id
                    ),

                "member_indices":
                    current_indices.copy(),
            }
        )


# ============================================================
# BUILD EVENT FEATURES
# ============================================================

event_rows = []


for event in temporary_events:

    member_df = interactions.loc[
        event[
            "member_indices"
        ]
    ].copy()


    start_frame = int(
        member_df[
            "frame"
        ].min()
    )


    end_frame = int(
        member_df[
            "frame"
        ].max()
    )


    has_strong = bool(
        member_df[
            "strong_deflection"
        ].any()
    )


    has_very_strong = bool(
        member_df[
            "very_strong_air_touch"
        ].any()
    )


    has_reception_evidence = bool(
        member_df[
            "followed_by_same_player_control"
        ].any()
    )


    valid_direction = pd.to_numeric(
        member_df[
            "direction_change_deg"
        ],
        errors="coerce"
    )


    valid_speed_change = pd.to_numeric(
        member_df[
            "speed_change_ratio"
        ],
        errors="coerce"
    )


    valid_ball_speed = pd.to_numeric(
        member_df[
            "ball_speed_mps"
        ],
        errors="coerce"
    )


    max_direction = (
        valid_direction.max()
        if valid_direction.notna().any()
        else np.nan
    )


    max_speed_change = (
        valid_speed_change.max()
        if valid_speed_change.notna().any()
        else np.nan
    )


    max_ball_speed = (
        valid_ball_speed.max()
        if valid_ball_speed.notna().any()
        else np.nan
    )


    min_upper_distance = (
        member_df[
            "upper_distance_px"
        ].min()
    )


    # ========================================================
    # EVENT-LEVEL CLASSIFICATION
    #
    # Priority:
    #
    # 1. VeryStrong
    #    -> AirTouch
    #
    # 2. Reception evidence anywhere in event
    #    -> Reception/Bounce
    #
    # 3. Strong trajectory deflection
    #    -> AirTouch
    #
    # 4. Otherwise weak interaction
    # ========================================================

    if has_very_strong:

        event_class = (
            "AirTouch"
        )

        confirmed_air_touch = True

        decision_reason = (
            "very_strong_air_touch"
        )


    elif has_reception_evidence:

        event_class = (
            "ReceptionOrBounce"
        )

        confirmed_air_touch = False

        decision_reason = (
            "event_contains_same_player_reception"
        )


    elif has_strong:

        event_class = (
            "AirTouch"
        )

        confirmed_air_touch = True

        decision_reason = (
            "strong_deflection_without_reception"
        )


    else:

        event_class = (
            "WeakInteraction"
        )

        confirmed_air_touch = False

        decision_reason = (
            "weak_deflection"
        )


    event_rows.append(
        {
            "team":
                event[
                    "team"
                ],

            "track_id":
                event[
                    "track_id"
                ],

            "start_frame":
                start_frame,

            "end_frame":
                end_frame,

            "start_time_sec":
                start_frame
                /
                fps,

            "end_time_sec":
                end_frame
                /
                fps,

            "duration_frames":
                end_frame
                -
                start_frame
                +
                1,

            "interaction_frame_count":
                len(
                    member_df
                ),

            "max_direction_change_deg":
                max_direction,

            "max_speed_change_ratio":
                max_speed_change,

            "max_ball_speed_mps":
                max_ball_speed,

            "min_upper_distance_px":
                min_upper_distance,

            "has_strong_deflection":
                has_strong,

            "has_very_strong_frame":
                has_very_strong,

            "has_reception_evidence":
                has_reception_evidence,

            "event_class":
                event_class,

            "confirmed_air_touch":
                confirmed_air_touch,

            "decision_reason":
                decision_reason,

            "_member_indices":
                event[
                    "member_indices"
                ],
        }
    )


# ============================================================
# SORT EVENTS CHRONOLOGICALLY
# ============================================================

events = pd.DataFrame(
    event_rows
)


events = (
    events
    .sort_values(
        [
            "start_frame",
            "team",
            "track_id",
        ]
    )
    .reset_index(
        drop=True
    )
)


events[
    "event_id"
] = np.arange(
    1,
    len(events)
    +
    1
)


# ============================================================
# MAP EVENT DECISION BACK TO INTERACTION FRAMES
# ============================================================

interactions[
    "event_id"
] = np.nan


interactions[
    "event_class"
] = None


interactions[
    "event_confirmed_air_touch"
] = False


interactions[
    "event_decision_reason"
] = None


for _, event in events.iterrows():

    member_indices = event[
        "_member_indices"
    ]


    interactions.loc[
        member_indices,
        "event_id"
    ] = int(
        event[
            "event_id"
        ]
    )


    interactions.loc[
        member_indices,
        "event_class"
    ] = event[
        "event_class"
    ]


    interactions.loc[
        member_indices,
        "event_confirmed_air_touch"
    ] = bool(
        event[
            "confirmed_air_touch"
        ]
    )


    interactions.loc[
        member_indices,
        "event_decision_reason"
    ] = event[
        "decision_reason"
    ]


# Remove private list column from CSV output.
events_output = events.drop(
    columns=[
        "_member_indices"
    ]
).copy()


confirmed_events = events_output[
    events_output[
        "confirmed_air_touch"
    ]
    ==
    True
].copy()


# ============================================================
# SAVE CSV
# ============================================================

interactions.to_csv(
    OUTPUT_INTERACTION_CSV,
    index=False
)


events_output.to_csv(
    OUTPUT_EVENT_CSV,
    index=False
)


confirmed_events.to_csv(
    OUTPUT_CONFIRMED_CSV,
    index=False
)


# ============================================================
# REGRESSION HELPERS
# ============================================================

def find_event_for_regression(
    frame,
    team,
    track_id
):

    matches = events_output[

        (
            events_output[
                "team"
            ]
            ==
            team
        )

        &

        (
            abs(
                events_output[
                    "track_id"
                ]
                -
                float(
                    track_id
                )
            )
            <
            0.1
        )

        &

        (
            events_output[
                "start_frame"
            ]
            <=
            frame
        )

        &

        (
            events_output[
                "end_frame"
            ]
            >=
            frame
        )
    ]


    if len(
        matches
    ) == 0:

        return None


    return matches.iloc[0]


negative_event = (
    find_event_for_regression(

        NEGATIVE_FRAME,

        NEGATIVE_TEAM,

        NEGATIVE_TRACK_ID
    )
)


positive_event = (
    find_event_for_regression(

        POSITIVE_FRAME,

        POSITIVE_TEAM,

        POSITIVE_TRACK_ID
    )
)


negative_pass = (

    negative_event
    is not None

    and

    not bool(
        negative_event[
            "confirmed_air_touch"
        ]
    )
)


positive_pass = (

    positive_event
    is not None

    and

    bool(
        positive_event[
            "confirmed_air_touch"
        ]
    )
)


# ============================================================
# SECOND PASS:
# QA VIDEO USING EVENT-LEVEL DECISIONS
# ============================================================

interaction_by_frame = {

    int(frame):
        group.copy()

    for frame, group
    in interactions.groupby(
        "frame"
    )
}


cap = cv2.VideoCapture(
    VIDEO_PATH
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
        f"Cannot create QA video: "
        f"{OUTPUT_VIDEO}"
    )


frame_idx = 0


print("")
print(
    "========================================"
)

print(
    "AIR TOUCH V3 - PASS 2"
)

print(
    "Event-level QA video"
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


    # ========================================================
    # BALL
    # ========================================================

    if possession_row is not None:

        ball_x = possession_row.get(
            "ball_image_x",
            np.nan
        )

        ball_y = possession_row.get(
            "ball_image_y",
            np.nan
        )


        if (
            pd.notna(
                ball_x
            )
            and
            pd.notna(
                ball_y
            )
        ):

            cv2.circle(
                annotated,

                (
                    int(
                        ball_x
                    ),
                    int(
                        ball_y
                    ),
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
    # INTERACTIONS
    # ========================================================

    frame_interactions = (
        interaction_by_frame.get(
            frame_idx,
            None
        )
    )


    if frame_interactions is not None:

        for _, row in frame_interactions.iterrows():

            x1 = int(
                row[
                    "bbox_x1"
                ]
            )

            y1 = int(
                row[
                    "bbox_y1"
                ]
            )

            x2 = int(
                row[
                    "bbox_x2"
                ]
            )

            y2 = int(
                row[
                    "bbox_y2"
                ]
            )


            confirmed = bool(
                row[
                    "event_confirmed_air_touch"
                ]
            )


            event_class = str(
                row[
                    "event_class"
                ]
            )


            if confirmed:

                color = (
                    0,
                    255,
                    0
                )


                label = (

                    f"AIR TOUCH EVENT "
                    f"{row['team']} "
                    f"#{int(row['track_id'])} "
                    f"E{int(row['event_id'])}"
                )


            elif (
                event_class
                ==
                "ReceptionOrBounce"
            ):

                color = (
                    0,
                    165,
                    255
                )


                label = (

                    f"RECEPTION/BOUNCE "
                    f"{row['team']} "
                    f"#{int(row['track_id'])} "
                    f"E{int(row['event_id'])}"
                )


            else:

                color = (
                    180,
                    180,
                    180
                )


                label = (

                    f"WEAK INTERACTION "
                    f"{row['team']} "
                    f"#{int(row['track_id'])} "
                    f"E{int(row['event_id'])}"
                )


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

                color,

                3
            )


            cv2.putText(
                annotated,

                label,

                (
                    x1,
                    max(
                        25,
                        y1
                        -
                        12
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.48,

                color,

                2,

                cv2.LINE_AA
            )


            cv2.putText(
                annotated,

                (
                    f"dir="
                    f"{row['direction_change_deg']:.1f} | "
                    f"speed="
                    f"{row['ball_speed_mps']:.1f} | "
                    f"future-control="
                    f"{bool(row['followed_by_same_player_control'])}"
                ),

                (
                    x1,
                    max(
                        45,
                        y1
                        -
                        34
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.38,

                color,

                1,

                cv2.LINE_AA
            )


    # ========================================================
    # TOP OVERLAY
    # ========================================================

    if possession_row is not None:

        ball_state = str(
            possession_row.get(
                "ball_state_v4",
                ""
            )
        )


        possession_team = str(
            possession_row.get(
                "possession_v4",
                ""
            )
        )


        cv2.putText(
            annotated,

            (
                f"AIR TOUCH V3 | "
                f"frame={frame_idx} | "
                f"time={frame_idx / fps:.2f}s | "
                f"state={ball_state} | "
                f"V4 possession={possession_team}"
            ),

            (
                20,
                35
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.58,

            (
                255,
                255,
                255
            ),

            2,

            cv2.LINE_AA
        )


    # ========================================================
    # NEGATIVE REGRESSION
    # ========================================================

    if (
        frame_idx
        ==
        NEGATIVE_FRAME
    ):

        text = (

            "NEGATIVE REGRESSION PASS | "
            "A #6 = RECEPTION / BOUNCE"

            if negative_pass

            else

            "NEGATIVE REGRESSION FAIL | "
            "A #6 SHOULD NOT BE AIR TOUCH"
        )


        color = (

            (
                0,
                255,
                0
            )

            if negative_pass

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
                75
            ),

            (
                820,
                125
            ),

            color,

            -1
        )


        cv2.putText(
            annotated,

            text,

            (
                35,
                108
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.60,

            (
                0,
                0,
                0
            ),

            2,

            cv2.LINE_AA
        )


    # ========================================================
    # POSITIVE REGRESSION
    # ========================================================

    if (
        frame_idx
        ==
        POSITIVE_FRAME
    ):

        text = (

            "POSITIVE REGRESSION PASS | "
            "B #10 = AIR TOUCH"

            if positive_pass

            else

            "POSITIVE REGRESSION FAIL | "
            "B #10 SHOULD BE AIR TOUCH"
        )


        color = (

            (
                0,
                255,
                0
            )

            if positive_pass

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
                75
            ),

            (
                780,
                125
            ),

            color,

            -1
        )


        cv2.putText(
            annotated,

            text,

            (
                35,
                108
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.60,

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


cap.release()
writer.release()


# ============================================================
# SUMMARY
# ============================================================

print("")
print(
    "========================================"
)

print(
    "AIR TOUCH V3 SUMMARY"
)

print(
    "========================================"
)

print("")

print(
    "Frame-level interaction candidates:",
    len(
        interactions
    )
)

print(
    "Grouped interaction events:",
    len(
        events_output
    )
)

print(
    "Confirmed AirTouch events:",
    len(
        confirmed_events
    )
)

print(
    "Rejected / non-AirTouch events:",
    len(
        events_output
    )
    -
    len(
        confirmed_events
    )
)


print("")

print(
    "NEGATIVE regression "
    "A #6 @ frame 203:",
    (
        "PASS"
        if negative_pass
        else
        "FAIL"
    )
)


if negative_event is not None:

    print(
        "  class:",
        negative_event[
            "event_class"
        ]
    )

    print(
        "  reason:",
        negative_event[
            "decision_reason"
        ]
    )

    print(
        "  event frames:",
        int(
            negative_event[
                "start_frame"
            ]
        ),
        "->",
        int(
            negative_event[
                "end_frame"
            ]
        )
    )


print("")

print(
    "POSITIVE regression "
    "B #10 @ frame 281:",
    (
        "PASS"
        if positive_pass
        else
        "FAIL"
    )
)


if positive_event is not None:

    print(
        "  class:",
        positive_event[
            "event_class"
        ]
    )

    print(
        "  reason:",
        positive_event[
            "decision_reason"
        ]
    )

    print(
        "  event frames:",
        int(
            positive_event[
                "start_frame"
            ]
        ),
        "->",
        int(
            positive_event[
                "end_frame"
            ]
        )
    )


# ============================================================
# EVENT TABLE
# ============================================================

print("")

if len(
    events_output
) > 0:

    display_columns = [
        "event_id",
        "team",
        "track_id",
        "start_frame",
        "end_frame",
        "start_time_sec",
        "end_time_sec",
        "interaction_frame_count",
        "max_ball_speed_mps",
        "max_direction_change_deg",
        "has_very_strong_frame",
        "has_reception_evidence",
        "event_class",
        "decision_reason",
    ]


    print(
        events_output[
            display_columns
        ]
        .round(2)
        .to_string(
            index=False
        )
    )


# ============================================================
# CLASS COUNTS
# ============================================================

print("")

print(
    "Event classes:"
)


print(
    events_output[
        "event_class"
    ]
    .value_counts()
    .to_string()
)


print("")

print(
    "Interaction CSV:",
    OUTPUT_INTERACTION_CSV
)

print(
    "Event CSV:",
    OUTPUT_EVENT_CSV
)

print(
    "Confirmed event CSV:",
    OUTPUT_CONFIRMED_CSV
)

print(
    "QA Video:",
    OUTPUT_VIDEO
)
