import os

import cv2
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

VIDEO_PATH = "videos/match.mp4"

PLAYER_TRACK_CSV = "outputs/tracking_data_clean_v2.csv"
POSSESSION_CSV = "outputs/possession_v5_1_frames.csv"
AIR_TOUCH_EVENT_CSV = "outputs/air_touch_v4_events.csv"

OUTPUT_VIDEO = "outputs/possession_v5_1_qa.mp4"
OUTPUT_REGRESSION_CSV = "outputs/possession_v5_1_regression_frames.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# REGRESSION FRAMES
# ============================================================

REGRESSION_INFO = {

    159: {
        "title":
            "GK LONG BALL / IN TRANSIT",

        "expected":
            "B | InTransit | controller NONE",
    },

    172: {
        "title":
            "A #1 + B #14 AERIAL DUEL",

        "expected":
            "Contested | AerialContest | LastTouch Unknown",
    },

    203: {
        "title":
            "A #6 BALL BOUNCE / RECEPTION",

        "expected":
            "NOT AirTouch | keep V4 estimation",
    },

    253: {
        "title":
            "AERIAL CONTEST",

        "expected":
            "Contested | controller NONE",
    },

    281: {
        "title":
            "B #10 HEADER",

        "expected":
            "B | InTransit | LastTouch B #10",
    },

    287: {
        "title":
            "POST-CONTEST FALLBACK",

        "expected":
            "AirTouch override ended | return to V4",
    },
}


# ============================================================
# PLAYER DISPLAY
# ============================================================

MIN_DOMINANT_TEAM_SHARE = 0.70

PLAYER_RADIUS = 8
NEAREST_RADIUS = 12

BALL_RADIUS = 9


# ============================================================
# LOAD DATA
# ============================================================

print("")
print("Loading player tracking...")

tracks = pd.read_csv(
    PLAYER_TRACK_CSV
)


print("Loading Possession V5.1...")

possession = pd.read_csv(
    POSSESSION_CSV
)


print("Loading AirTouch V4 events...")

events = pd.read_csv(
    AIR_TOUCH_EVENT_CSV
)


# ============================================================
# NORMALIZE TRACKING
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
# STABLE TEAM ASSIGNMENT
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


# Important:
# don't relabel contaminated rows.
tracks = tracks[
    tracks["team"]
    ==
    tracks["stable_team"]
].copy()


tracks_by_frame = {

    int(frame):
        group.copy()

    for frame, group
    in tracks.groupby(
        "frame"
    )
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
    "controller_id_v5",

    "last_touch_id_v5",

    "nearest_A_pitch_distance_m",
    "nearest_A_image_foot_distance_px",
    "nearest_B_pitch_distance_m",
    "nearest_B_image_foot_distance_px",

    "nearest_A_track_id",
    "nearest_B_track_id",

    "physical_event_id_v5",
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


possession = (
    possession
    .sort_values(
        "frame"
    )
    .reset_index(
        drop=True
    )
)


possession_by_frame = (
    possession
    .set_index(
        "frame"
    )
)


# ============================================================
# NORMALIZE AIR TOUCH EVENTS
# ============================================================

for column in [
    "physical_event_id",
    "start_frame",
    "end_frame",
    "last_touch_track_id",
]:

    if column in events.columns:

        events[column] = pd.to_numeric(
            events[column],
            errors="coerce"
        )


events = events.dropna(
    subset=[
        "start_frame",
        "end_frame",
        "final_class",
    ]
).copy()


events[
    "start_frame"
] = (
    events[
        "start_frame"
    ]
    .astype(int)
)


events[
    "end_frame"
] = (
    events[
        "end_frame"
    ]
    .astype(int)
)


events_by_frame = {}


for _, event in events.iterrows():

    start = int(
        event[
            "start_frame"
        ]
    )

    end = int(
        event[
            "end_frame"
        ]
    )


    for frame in range(
        start,
        end + 1
    ):

        events_by_frame.setdefault(
            frame,
            []
        ).append(
            event
        )


# ============================================================
# HELPERS
# ============================================================

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


def safe_text(
    value,
    default="NONE"
):

    if pd.isna(value):

        return default


    text = str(value).strip()


    if (
        text == ""
        or
        text.lower()
        in {
            "nan",
            "none",
        }
    ):

        return default


    return text


def safe_id_text(
    value
):

    if pd.isna(value):

        return "NONE"


    try:

        return str(
            int(
                float(value)
            )
        )

    except Exception:

        return str(value)


def format_controller(
    team,
    track_id
):

    team_text = safe_text(
        team
    )


    if (
        team_text
        not in
        {
            "A",
            "B",
        }
    ):

        return "NONE"


    if pd.isna(
        track_id
    ):

        return "NONE"


    return (
        f"{team_text} "
        f"#{safe_id_text(track_id)}"
    )


def format_last_touch(
    team,
    track_id,
    touch_type
):

    team_text = safe_text(
        team
    )


    type_text = safe_text(
        touch_type
    )


    if (
        team_text
        not in
        {
            "A",
            "B",
        }
    ):

        if (
            "Contest"
            in
            type_text
        ):

            return (
                "UNKNOWN "
                f"({type_text})"
            )


        return "UNKNOWN"


    if pd.isna(
        track_id
    ):

        return team_text


    return (
        f"{team_text} "
        f"#{safe_id_text(track_id)} "
        f"({type_text})"
    )


def get_track_row(
    frame_tracks,
    track_id
):

    if (
        frame_tracks is None
        or
        pd.isna(track_id)
    ):

        return None


    matches = frame_tracks[
        np.abs(
            frame_tracks[
                "track_id"
            ]
            -
            float(
                track_id
            )
        )
        <
        0.1
    ]


    if len(matches) == 0:

        return None


    return matches.iloc[0]


def put_text(
    image,
    text,
    x,
    y,
    scale=0.52,
    thickness=2,
    color=(255, 255, 255),
):

    cv2.putText(
        image,

        str(text),

        (
            int(x),
            int(y)
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        scale,

        color,

        thickness,

        cv2.LINE_AA
    )


# ============================================================
# VIDEO
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


# ============================================================
# QA
# ============================================================

frame_idx = 0

regression_rows = []


print("")
print(
    "========================================"
)

print(
    "POSSESSION V5.1 VISUAL QA"
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


    row = get_possession_row(
        frame_idx
    )


    if row is None:

        writer.write(
            annotated
        )

        frame_idx += 1

        continue


    # ========================================================
    # BASIC VALUES
    # ========================================================

    time_sec = row.get(
        "time_sec",
        frame_idx
        /
        fps
    )


    possession_v4 = safe_text(
        row.get(
            "possession_v4",
            None
        ),
        "Unknown"
    )


    possession_v5 = safe_text(
        row.get(
            "possession_v5",
            None
        ),
        "Unknown"
    )


    ball_state_v4 = safe_text(
        row.get(
            "ball_state_v4",
            None
        ),
        "Unknown"
    )


    ball_state_v5 = safe_text(
        row.get(
            "ball_state_v5",
            None
        ),
        "Unknown"
    )


    controller_v4 = format_controller(

        row.get(
            "controller_team_v4",
            None
        ),

        row.get(
            "controller_id_v4",
            np.nan
        )
    )


    controller_v5 = format_controller(

        row.get(
            "controller_team_v5",
            None
        ),

        row.get(
            "controller_id_v5",
            np.nan
        )
    )


    last_touch = format_last_touch(

        row.get(
            "last_touch_team_v5",
            None
        ),

        row.get(
            "last_touch_id_v5",
            np.nan
        ),

        row.get(
            "last_touch_type_v5",
            None
        )
    )


    source_v5 = safe_text(
        row.get(
            "source_v5",
            None
        ),
        "baseline_v4"
    )


    ball_speed = row.get(
        "ball_speed_mps",
        np.nan
    )


    # ========================================================
    # DRAW PLAYERS
    # ========================================================

    frame_tracks = (
        tracks_by_frame.get(
            frame_idx,
            None
        )
    )


    if frame_tracks is not None:

        for _, player in (
            frame_tracks.iterrows()
        ):

            x = int(
                player[
                    "image_x"
                ]
            )

            y = int(
                player[
                    "image_y"
                ]
            )


            team = str(
                player[
                    "stable_team"
                ]
            )


            track_id = int(
                player[
                    "track_id"
                ]
            )


            # Team A / B are intentionally visually distinct.
            if team == "A":

                color = (
                    255,
                    120,
                    0
                )

            else:

                color = (
                    0,
                    0,
                    255
                )


            cv2.circle(
                annotated,

                (
                    x,
                    y
                ),

                PLAYER_RADIUS,

                color,

                2
            )


            put_text(
                annotated,

                f"{team}#{track_id}",

                x + 7,
                y - 8,

                scale=0.38,

                color=color,

                thickness=1
            )


    # ========================================================
    # DRAW BALL
    # ========================================================

    ball_x = row.get(
        "ball_image_x",
        np.nan
    )


    ball_y = row.get(
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

        bx = int(
            ball_x
        )

        by = int(
            ball_y
        )


        cv2.circle(
            annotated,

            (
                bx,
                by
            ),

            BALL_RADIUS + 3,

            (
                0,
                0,
                0
            ),

            4
        )


        cv2.circle(
            annotated,

            (
                bx,
                by
            ),

            BALL_RADIUS,

            (
                0,
                255,
                255
            ),

            3
        )


        put_text(
            annotated,

            "BALL",

            bx + 13,
            by - 10,

            scale=0.40,

            color=(
                0,
                255,
                255
            )
        )


    # ========================================================
    # CONTROLLER
    # ========================================================

    controller_id_v5 = row.get(
        "controller_id_v5",
        np.nan
    )


    controller_track = get_track_row(
        frame_tracks,
        controller_id_v5
    )


    if controller_track is not None:

        cx = int(
            controller_track[
                "image_x"
            ]
        )

        cy = int(
            controller_track[
                "image_y"
            ]
        )


        cv2.circle(
            annotated,

            (
                cx,
                cy
            ),

            17,

            (
                0,
                255,
                0
            ),

            3
        )


        put_text(
            annotated,

            "CONTROLLER",

            cx + 15,
            cy + 18,

            scale=0.43,

            color=(
                0,
                255,
                0
            )
        )


    # ========================================================
    # LAST TOUCH PLAYER
    # ========================================================

    last_touch_id = row.get(
        "last_touch_id_v5",
        np.nan
    )


    last_touch_track = get_track_row(
        frame_tracks,
        last_touch_id
    )


    if (

        last_touch_track
        is not None

        and

        controller_track
        is None
    ):

        lx = int(
            last_touch_track[
                "image_x"
            ]
        )

        ly = int(
            last_touch_track[
                "image_y"
            ]
        )


        cv2.circle(
            annotated,

            (
                lx,
                ly
            ),

            18,

            (
                255,
                0,
                255
            ),

            3
        )


        put_text(
            annotated,

            "LAST TOUCH",

            lx + 15,
            ly + 18,

            scale=0.43,

            color=(
                255,
                0,
                255
            )
        )


    # ========================================================
    # NEAREST PLAYERS
    # ========================================================

    nearest_A_id = row.get(
        "nearest_A_track_id",
        np.nan
    )


    nearest_B_id = row.get(
        "nearest_B_track_id",
        np.nan
    )


    for nearest_id in [
        nearest_A_id,
        nearest_B_id,
    ]:

        nearest_track = get_track_row(
            frame_tracks,
            nearest_id
        )


        if nearest_track is None:

            continue


        nx = int(
            nearest_track[
                "image_x"
            ]
        )

        ny = int(
            nearest_track[
                "image_y"
            ]
        )


        cv2.circle(
            annotated,

            (
                nx,
                ny
            ),

            NEAREST_RADIUS,

            (
                255,
                255,
                255
            ),

            1
        )


    # ========================================================
    # TOP LEFT INFO PANEL
    # ========================================================

    panel_x1 = 15
    panel_y1 = 15

    panel_x2 = 690
    panel_y2 = 285


    overlay = annotated.copy()


    cv2.rectangle(
        overlay,

        (
            panel_x1,
            panel_y1
        ),

        (
            panel_x2,
            panel_y2
        ),

        (
            0,
            0,
            0
        ),

        -1
    )


    cv2.addWeighted(
        overlay,
        0.60,

        annotated,
        0.40,

        0,

        annotated
    )


    y = 42


    put_text(
        annotated,

        (
            f"POSSESSION V5.1 QA | "
            f"frame={frame_idx} | "
            f"time={time_sec:.2f}s"
        ),

        30,
        y,

        scale=0.58
    )


    y += 33


    put_text(
        annotated,

        f"TEAM POSSESSION: {possession_v5}",

        30,
        y,

        scale=0.67,
        color=(
            0,
            255,
            255
        )
    )


    y += 29


    put_text(
        annotated,

        f"BALL STATE: {ball_state_v5}",

        30,
        y,

        scale=0.57
    )


    y += 27


    put_text(
        annotated,

        f"CONTROLLER: {controller_v5}",

        30,
        y,

        scale=0.53
    )


    y += 27


    put_text(
        annotated,

        f"LAST TOUCH: {last_touch}",

        30,
        y,

        scale=0.53,

        color=(
            255,
            0,
            255
        )
    )


    y += 27


    put_text(
        annotated,

        f"SOURCE: {source_v5}",

        30,
        y,

        scale=0.49
    )


    y += 27


    if pd.notna(
        ball_speed
    ):

        speed_text = (
            f"{float(ball_speed):.2f} m/s"
        )

    else:

        speed_text = "N/A"


    put_text(
        annotated,

        f"BALL SPEED: {speed_text}",

        30,
        y,

        scale=0.49
    )


    # ========================================================
    # V4 -> V5 COMPARISON
    # ========================================================

    changed = (

        possession_v4
        !=
        possession_v5

        or

        ball_state_v4
        !=
        ball_state_v5

        or

        controller_v4
        !=
        controller_v5
    )


    if changed:

        comparison_text = (

            f"OVERRIDE | "
            f"V4 {possession_v4}/{ball_state_v4} "
            f"-> "
            f"V5 {possession_v5}/{ball_state_v5}"
        )


        cv2.rectangle(
            annotated,

            (
                15,
                300
            ),

            (
                820,
                350
            ),

            (
                0,
                140,
                255
            ),

            -1
        )


        put_text(
            annotated,

            comparison_text,

            30,
            333,

            scale=0.58,

            color=(
                0,
                0,
                0
            )
        )


    # ========================================================
    # PHYSICAL EVENT PANEL
    # ========================================================

    active_events = (
        events_by_frame.get(
            frame_idx,
            []
        )
    )


    event_y = 380


    for event in active_events:

        event_class = safe_text(
            event.get(
                "final_class",
                None
            )
        )


        participants = safe_text(
            event.get(
                "participants",
                None
            )
        )


        if (
            event_class
            ==
            "AirTouch"
        ):

            event_color = (
                0,
                255,
                0
            )


            event_text = (

                f"PHYSICAL EVENT: AIR TOUCH | "
                f"{participants} | "
                f"LAST TOUCH "
                f"{safe_text(event.get('last_touch_team'))}"
                f"#{safe_id_text(event.get('last_touch_track_id'))}"
            )


        elif (
            event_class
            ==
            "AerialContest"
        ):

            event_color = (
                255,
                0,
                255
            )


            event_text = (

                f"PHYSICAL EVENT: AERIAL CONTEST | "
                f"{participants} | "
                f"LAST TOUCH UNKNOWN"
            )


        elif (
            event_class
            ==
            "ReceptionOrBounce"
        ):

            event_color = (
                0,
                165,
                255
            )


            event_text = (

                f"PHYSICAL EVENT: "
                f"RECEPTION / BOUNCE | "
                f"{participants}"
            )


        else:

            event_color = (
                180,
                180,
                180
            )


            event_text = (

                f"PHYSICAL EVENT: "
                f"{event_class} | "
                f"{participants}"
            )


        cv2.rectangle(
            annotated,

            (
                15,
                event_y - 28
            ),

            (
                min(
                    width - 15,
                    980
                ),
                event_y + 12
            ),

            (
                0,
                0,
                0
            ),

            -1
        )


        put_text(
            annotated,

            event_text,

            30,
            event_y,

            scale=0.50,

            color=event_color
        )


        event_y += 48


    # ========================================================
    # NEAREST DISTANCES
    # ========================================================

    nearest_A_pitch = row.get(
        "nearest_A_pitch_distance_m",
        np.nan
    )


    nearest_A_image = row.get(
        "nearest_A_image_foot_distance_px",
        np.nan
    )


    nearest_B_pitch = row.get(
        "nearest_B_pitch_distance_m",
        np.nan
    )


    nearest_B_image = row.get(
        "nearest_B_image_foot_distance_px",
        np.nan
    )


    distance_y = height - 75


    if (
        pd.notna(
            nearest_A_pitch
        )

        or

        pd.notna(
            nearest_B_pitch
        )
    ):

        A_pitch_text = (
            f"{nearest_A_pitch:.2f}m"
            if pd.notna(
                nearest_A_pitch
            )
            else
            "N/A"
        )


        A_image_text = (
            f"{nearest_A_image:.1f}px"
            if pd.notna(
                nearest_A_image
            )
            else
            "N/A"
        )


        B_pitch_text = (
            f"{nearest_B_pitch:.2f}m"
            if pd.notna(
                nearest_B_pitch
            )
            else
            "N/A"
        )


        B_image_text = (
            f"{nearest_B_image:.1f}px"
            if pd.notna(
                nearest_B_image
            )
            else
            "N/A"
        )


        put_text(
            annotated,

            (
                f"Nearest A: "
                f"{A_pitch_text} / "
                f"{A_image_text}    "
                f"Nearest B: "
                f"{B_pitch_text} / "
                f"{B_image_text}"
            ),

            20,
            distance_y,

            scale=0.46,

            color=(
                255,
                255,
                255
            )
        )


    # ========================================================
    # REGRESSION BANNER
    # ========================================================

    if (
        frame_idx
        in
        REGRESSION_INFO
    ):

        info = REGRESSION_INFO[
            frame_idx
        ]


        banner_y1 = height - 155
        banner_y2 = height - 95


        cv2.rectangle(
            annotated,

            (
                15,
                banner_y1
            ),

            (
                width - 15,
                banner_y2
            ),

            (
                0,
                255,
                255
            ),

            -1
        )


        put_text(
            annotated,

            (
                f"REGRESSION | "
                f"{info['title']}"
            ),

            30,
            banner_y1 + 25,

            scale=0.56,

            color=(
                0,
                0,
                0
            )
        )


        put_text(
            annotated,

            (
                f"EXPECTED: "
                f"{info['expected']}"
            ),

            30,
            banner_y1 + 49,

            scale=0.48,

            color=(
                0,
                0,
                0
            )
        )


        regression_rows.append(
            {
                "frame":
                    frame_idx,

                "time_sec":
                    time_sec,

                "test":
                    info[
                        "title"
                    ],

                "expected":
                    info[
                        "expected"
                    ],

                "possession_v4":
                    possession_v4,

                "possession_v5":
                    possession_v5,

                "ball_state_v4":
                    ball_state_v4,

                "ball_state_v5":
                    ball_state_v5,

                "controller_v4":
                    controller_v4,

                "controller_v5":
                    controller_v5,

                "last_touch":
                    last_touch,

                "source_v5":
                    source_v5,
            }
        )


    # ========================================================
    # TIMELINE
    # ========================================================

    timeline_x1 = 20
    timeline_x2 = width - 20
    timeline_y = height - 28


    cv2.line(
        annotated,

        (
            timeline_x1,
            timeline_y
        ),

        (
            timeline_x2,
            timeline_y
        ),

        (
            180,
            180,
            180
        ),

        2
    )


    progress = (
        frame_idx
        /
        max(
            1,
            int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )
            -
            1
        )
    )


    current_x = int(

        timeline_x1

        +

        progress
        *
        (
            timeline_x2
            -
            timeline_x1
        )
    )


    cv2.circle(
        annotated,

        (
            current_x,
            timeline_y
        ),

        5,

        (
            0,
            255,
            255
        ),

        -1
    )


    # Regression markers on timeline.
    total_frames = max(
        1,
        int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )
        -
        1
    )


    for regression_frame in (
        REGRESSION_INFO.keys()
    ):

        rx = int(

            timeline_x1

            +

            regression_frame
            /
            total_frames
            *
            (
                timeline_x2
                -
                timeline_x1
            )
        )


        cv2.line(
            annotated,

            (
                rx,
                timeline_y - 6
            ),

            (
                rx,
                timeline_y + 6
            ),

            (
                0,
                0,
                255
            ),

            2
        )


    writer.write(
        annotated
    )


    frame_idx += 1


# ============================================================
# FINISH
# ============================================================

cap.release()
writer.release()


# ============================================================
# REGRESSION CSV
# ============================================================

regression_df = pd.DataFrame(
    regression_rows
)


regression_df.to_csv(
    OUTPUT_REGRESSION_CSV,
    index=False
)


# ============================================================
# PRINT
# ============================================================

print("")
print(
    "========================================"
)

print(
    "POSSESSION V5.1 VISUAL QA SUMMARY"
)

print(
    "========================================"
)

print("")

print(
    "QA Video:",
    OUTPUT_VIDEO
)

print(
    "Regression CSV:",
    OUTPUT_REGRESSION_CSV
)


print("")

print(
    "Regression frames:"
)


if len(
    regression_df
) > 0:

    print(
        regression_df[
            [
                "frame",
                "time_sec",
                "test",
                "possession_v4",
                "possession_v5",
                "ball_state_v5",
                "controller_v5",
                "last_touch",
                "source_v5",
            ]
        ]
        .to_string(
            index=False
        )
    )


print("")

print(
    "Visual QA priorities:"
)

print(
    "1. 6.36s  : long ball still belongs to B phase"
)

print(
    "2. 6.88s  : A #1 + B #14 must look like real aerial duel"
)

print(
    "3. 8.12s  : A #6 ground reception/bounce, NOT air touch"
)

print(
    "4. 10.12s : aerial contest, controller must be NONE"
)

print(
    "5. 11.24s : B #10 header, LastTouch B #10, controller NONE"
)

print(
    "6. 11.48s : AirTouch override already ended; V5 returns to V4"
)
