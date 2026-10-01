import os

import cv2
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

VIDEO_PATH = "videos/match.mp4"

PLAYER_CSV = "outputs/tracking_data_clean_v2.csv"
POSSESSION_CSV = "outputs/possession_v4_frames.csv"

OUTPUT_VIDEO = "outputs/possession_v4_qa.mp4"

OUTPUT_RECEPTION_CSV = "outputs/possession_v4_receptions.csv"
OUTPUT_TURNOVER_CSV = "outputs/possession_v4_turnovers.csv"
OUTPUT_AERIAL_CSV = "outputs/possession_v4_aerial_contests.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Stable-team settings
#
# Must match possession_v3.py / V4 input logic.
# ============================================================

MIN_DOMINANT_TEAM_SHARE = 0.70


# ============================================================
# Regression frames
# ============================================================

REGRESSION_FRAMES = {
    159: "V3 FALSE TURNOVER: must remain IN TRANSIT",
    253: "V3 FALSE CONTROL: should be AERIAL CONTEST / IN TRANSIT",
}


# ============================================================
# Colors (BGR)
# ============================================================

TEAM_A_COLOR = (255, 120, 40)
TEAM_B_COLOR = (40, 80, 255)

BALL_COLOR = (0, 255, 255)

CONTROLLER_COLOR = (0, 255, 0)

RECEPTION_COLOR = (255, 0, 255)

IN_TRANSIT_COLOR = (255, 0, 255)

AERIAL_COLOR = (0, 165, 255)

CONTESTED_COLOR = (0, 165, 255)

UNKNOWN_COLOR = (150, 150, 150)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)


# ============================================================
# Helpers
# ============================================================

def normalize_bool(series):

    return (
        series
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes"
            ]
        )
    )


def possession_color(state):

    if state == "A":
        return TEAM_A_COLOR

    if state == "B":
        return TEAM_B_COLOR

    if state == "Contested":
        return CONTESTED_COLOR

    return UNKNOWN_COLOR


def draw_panel(
    frame,
    x1,
    y1,
    x2,
    y2,
    alpha=0.68
):

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        BLACK,
        -1
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1.0 - alpha,
        0,
        frame
    )


def safe_int(value):

    if pd.isna(value):
        return None

    return int(
        round(
            float(value)
        )
    )


# ============================================================
# Load
# ============================================================

players = pd.read_csv(
    PLAYER_CSV
)

possession = pd.read_csv(
    POSSESSION_CSV
)


# ============================================================
# Normalize player data
# ============================================================

for column in [
    "frame",
    "track_id",
    "image_x",
    "image_y"
]:

    players[column] = pd.to_numeric(
        players[column],
        errors="coerce"
    )


players = players.dropna(
    subset=[
        "frame",
        "track_id",
        "image_x",
        "image_y"
    ]
).copy()


players["frame"] = (
    players["frame"]
    .astype(int)
)


players = players[
    players["team"]
    .isin(
        [
            "A",
            "B"
        ]
    )
].copy()


# ============================================================
# Stable team reconstruction
# ============================================================

track_stats = []


for track_id, group in players.groupby(
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


    dominant_count = int(
        counts.iloc[0]
    )


    total_count = int(
        len(group)
    )


    dominant_share = (
        dominant_count
        /
        total_count
    )


    track_stats.append(
        {
            "track_id":
                track_id,

            "dominant_team":
                dominant_team,

            "dominant_share":
                dominant_share,
        }
    )


track_stats = pd.DataFrame(
    track_stats
)


stable_stats = track_stats[
    track_stats[
        "dominant_share"
    ]
    >=
    MIN_DOMINANT_TEAM_SHARE
].copy()


stable_team_map = dict(
    zip(
        stable_stats[
            "track_id"
        ],

        stable_stats[
            "dominant_team"
        ]
    )
)


players["stable_team"] = (
    players["track_id"]
    .map(
        stable_team_map
    )
)


players = players[
    players[
        "stable_team"
    ].notna()
].copy()


# Do not relabel inconsistent rows.
players = players[
    players["team"]
    ==
    players["stable_team"]
].copy()


players["team"] = (
    players[
        "stable_team"
    ]
)


players_by_frame = {

    int(frame):
        group.copy()

    for frame, group
    in players.groupby("frame")
}


# ============================================================
# Normalize possession data
# ============================================================

possession["frame"] = pd.to_numeric(
    possession["frame"],
    errors="coerce"
)


possession = possession.dropna(
    subset=["frame"]
).copy()


possession["frame"] = (
    possession["frame"]
    .astype(int)
)


numeric_columns = [

    "time_sec",

    "ball_speed_mps",

    "ball_image_x",
    "ball_image_y",

    "nearest_A_id",
    "nearest_B_id",

    "nearest_A_pitch_distance_m",
    "nearest_B_pitch_distance_m",

    "nearest_A_image_foot_distance_px",
    "nearest_B_image_foot_distance_px",

    "controller_id_v4",

    "reception_candidate_id",
    "reception_candidate_count",

    "A_player_count",
    "B_player_count",
]


for column in numeric_columns:

    if column in possession.columns:

        possession[column] = pd.to_numeric(
            possession[column],
            errors="coerce"
        )


if "ball_usable_bool" in possession.columns:

    possession[
        "ball_usable_bool"
    ] = normalize_bool(
        possession[
            "ball_usable_bool"
        ]
    )

elif "ball_usable" in possession.columns:

    possession[
        "ball_usable_bool"
    ] = normalize_bool(
        possession[
            "ball_usable"
        ]
    )

else:

    possession[
        "ball_usable_bool"
    ] = False


possession_by_frame = (
    possession
    .set_index("frame")
)


MIN_FRAME = int(
    possession[
        "frame"
    ].min()
)

MAX_FRAME = int(
    possession[
        "frame"
    ].max()
)


# ============================================================
# Player lookup
# ============================================================

def find_player(
    frame_players,
    track_id
):

    if (
        frame_players is None
        or
        pd.isna(track_id)
    ):

        return None


    difference = np.abs(
        frame_players[
            "track_id"
        ]
        -
        float(track_id)
    )


    valid = difference.notna()


    if not valid.any():

        return None


    index = (
        difference[
            valid
        ]
        .idxmin()
    )


    if (
        difference.loc[
            index
        ]
        >
        0.1
    ):

        return None


    return frame_players.loc[
        index
    ]


# ============================================================
# Event extraction
# ============================================================

receptions = possession[
    possession[
        "possession_source_v4"
    ]
    ==
    "confirmed_reception"
].copy()


turnovers = possession[
    possession[
        "possession_source_v4"
    ]
    ==
    "confirmed_controlled_turnover"
].copy()


reception_columns = [

    "frame",
    "time_sec",

    "possession_v4",

    "controller_team_v4",
    "controller_id_v4",

    "ball_speed_mps",

    "nearest_A_pitch_distance_m",
    "nearest_A_image_foot_distance_px",

    "nearest_B_pitch_distance_m",
    "nearest_B_image_foot_distance_px",
]


reception_columns = [
    column
    for column in reception_columns
    if column in receptions.columns
]


receptions[
    reception_columns
].to_csv(
    OUTPUT_RECEPTION_CSV,
    index=False
)


turnover_columns = [

    "frame",
    "time_sec",

    "possession_v4",

    "controller_team_v4",
    "controller_id_v4",

    "ball_speed_mps",

    "nearest_A_pitch_distance_m",
    "nearest_A_image_foot_distance_px",

    "nearest_B_pitch_distance_m",
    "nearest_B_image_foot_distance_px",
]


turnover_columns = [
    column
    for column in turnover_columns
    if column in turnovers.columns
]


turnovers[
    turnover_columns
].to_csv(
    OUTPUT_TURNOVER_CSV,
    index=False
)


# ============================================================
# Aerial contest segments
# ============================================================

aerial_mask = (
    possession[
        "ball_state_v4"
    ]
    ==
    "AerialContest"
)


aerial_segments = []

start_frame = None
last_frame = None


for _, row in possession.iterrows():

    frame = int(
        row["frame"]
    )


    is_aerial = (
        str(
            row[
                "ball_state_v4"
            ]
        )
        ==
        "AerialContest"
    )


    if is_aerial:

        if start_frame is None:

            start_frame = frame

        last_frame = frame


    else:

        if start_frame is not None:

            aerial_segments.append(
                {
                    "start_frame":
                        start_frame,

                    "end_frame":
                        last_frame,

                    "start_time_sec":
                        start_frame / 25.0,

                    "end_time_sec":
                        last_frame / 25.0,

                    "duration_frames":
                        last_frame
                        -
                        start_frame
                        +
                        1,

                    "duration_sec":
                        (
                            last_frame
                            -
                            start_frame
                            +
                            1
                        )
                        /
                        25.0,
                }
            )


            start_frame = None
            last_frame = None


if start_frame is not None:

    aerial_segments.append(
        {
            "start_frame":
                start_frame,

            "end_frame":
                last_frame,

            "start_time_sec":
                start_frame / 25.0,

            "end_time_sec":
                last_frame / 25.0,

            "duration_frames":
                last_frame
                -
                start_frame
                +
                1,

            "duration_sec":
                (
                    last_frame
                    -
                    start_frame
                    +
                    1
                )
                /
                25.0,
        }
    )


aerial_df = pd.DataFrame(
    aerial_segments
)


aerial_df.to_csv(
    OUTPUT_AERIAL_CSV,
    index=False
)


# ============================================================
# Timeline
# ============================================================

timeline_states = {

    int(row["frame"]):
        str(
            row[
                "possession_v4"
            ]
        )

    for _, row
    in possession.iterrows()
}


def draw_timeline(
    frame,
    current_frame
):

    height, width = frame.shape[:2]


    margin = 30

    y1 = height - 42
    y2 = height - 19


    timeline_width = (
        width
        -
        margin * 2
    )


    cv2.rectangle(
        frame,
        (
            margin,
            y1
        ),
        (
            margin
            +
            timeline_width,
            y2
        ),
        (
            50,
            50,
            50
        ),
        -1
    )


    frame_range = max(
        MAX_FRAME
        -
        MIN_FRAME,
        1
    )


    for timeline_frame, state in (
        timeline_states.items()
    ):

        ratio = (
            timeline_frame
            -
            MIN_FRAME
        ) / frame_range


        x = int(
            margin
            +
            ratio
            *
            timeline_width
        )


        cv2.line(
            frame,
            (
                x,
                y1
            ),
            (
                x,
                y2
            ),
            possession_color(
                state
            ),
            3
        )


    if (
        MIN_FRAME
        <=
        current_frame
        <=
        MAX_FRAME
    ):

        ratio = (
            current_frame
            -
            MIN_FRAME
        ) / frame_range


        x = int(
            margin
            +
            ratio
            *
            timeline_width
        )


        cv2.line(
            frame,
            (
                x,
                y1 - 7
            ),
            (
                x,
                y2 + 7
            ),
            WHITE,
            2
        )


    cv2.putText(
        frame,
        "POSSESSION V4 TIMELINE",
        (
            margin,
            y1 - 7
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        WHITE,
        1,
        cv2.LINE_AA
    )


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
        f"Cannot create video: {OUTPUT_VIDEO}"
    )


# ============================================================
# Main loop
# ============================================================

frame_idx = 0


while True:

    ret, frame = cap.read()


    if not ret:

        break


    # ========================================================
    # Outside analysis
    # ========================================================

    if (
        frame_idx
        not in
        possession_by_frame.index
    ):

        draw_panel(
            frame,
            15,
            15,
            500,
            90
        )


        cv2.putText(
            frame,
            "POSSESSION V4 QA",
            (
                30,
                47
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            WHITE,
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            "Outside analysis range",
            (
                30,
                76
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            UNKNOWN_COLOR,
            2,
            cv2.LINE_AA
        )


        draw_timeline(
            frame,
            frame_idx
        )


        writer.write(
            frame
        )


        frame_idx += 1

        continue


    # ========================================================
    # Row
    # ========================================================

    row = possession_by_frame.loc[
        frame_idx
    ]


    if isinstance(
        row,
        pd.DataFrame
    ):

        row = row.iloc[0]


    possession_state = str(
        row[
            "possession_v4"
        ]
    )


    ball_state = str(
        row[
            "ball_state_v4"
        ]
    )


    raw_state = str(
        row[
            "raw_ball_state"
        ]
    )


    source = str(
        row[
            "possession_source_v4"
        ]
    )


    state_color = possession_color(
        possession_state
    )


    frame_players = (
        players_by_frame.get(
            frame_idx,
            None
        )
    )


    # ========================================================
    # Draw all players
    # ========================================================

    if frame_players is not None:

        for _, player in (
            frame_players.iterrows()
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
                    "team"
                ]
            )


            color = (
                TEAM_A_COLOR
                if team == "A"
                else TEAM_B_COLOR
            )


            cv2.circle(
                frame,
                (
                    x,
                    y
                ),
                4,
                color,
                -1
            )


    # ========================================================
    # Ball
    # ========================================================

    ball_x = None
    ball_y = None


    if (
        bool(
            row[
                "ball_usable_bool"
            ]
        )

        and

        pd.notna(
            row[
                "ball_image_x"
            ]
        )

        and

        pd.notna(
            row[
                "ball_image_y"
            ]
        )
    ):

        ball_x = int(
            row[
                "ball_image_x"
            ]
        )

        ball_y = int(
            row[
                "ball_image_y"
            ]
        )


        cv2.circle(
            frame,
            (
                ball_x,
                ball_y
            ),
            9,
            BALL_COLOR,
            -1
        )


        cv2.circle(
            frame,
            (
                ball_x,
                ball_y
            ),
            15,
            BALL_COLOR,
            2
        )


        cv2.putText(
            frame,
            "BALL",
            (
                ball_x + 13,
                ball_y - 8
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            BALL_COLOR,
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # Nearest A
    # ========================================================

    nearest_A_id = row[
        "nearest_A_id"
    ]


    A_player = find_player(
        frame_players,
        nearest_A_id
    )


    if A_player is not None:

        ax = int(
            A_player[
                "image_x"
            ]
        )

        ay = int(
            A_player[
                "image_y"
            ]
        )


        cv2.circle(
            frame,
            (
                ax,
                ay
            ),
            11,
            TEAM_A_COLOR,
            3
        )


        cv2.putText(
            frame,
            (
                f"A #{int(nearest_A_id)}"
            ),
            (
                ax + 13,
                ay - 7
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            TEAM_A_COLOR,
            2,
            cv2.LINE_AA
        )


        if ball_x is not None:

            cv2.line(
                frame,
                (
                    ax,
                    ay
                ),
                (
                    ball_x,
                    ball_y
                ),
                TEAM_A_COLOR,
                1
            )


    # ========================================================
    # Nearest B
    # ========================================================

    nearest_B_id = row[
        "nearest_B_id"
    ]


    B_player = find_player(
        frame_players,
        nearest_B_id
    )


    if B_player is not None:

        bx = int(
            B_player[
                "image_x"
            ]
        )

        by = int(
            B_player[
                "image_y"
            ]
        )


        cv2.circle(
            frame,
            (
                bx,
                by
            ),
            11,
            TEAM_B_COLOR,
            3
        )


        cv2.putText(
            frame,
            (
                f"B #{int(nearest_B_id)}"
            ),
            (
                bx + 13,
                by - 7
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            TEAM_B_COLOR,
            2,
            cv2.LINE_AA
        )


        if ball_x is not None:

            cv2.line(
                frame,
                (
                    bx,
                    by
                ),
                (
                    ball_x,
                    ball_y
                ),
                TEAM_B_COLOR,
                1
            )


    # ========================================================
    # Controller
    # ========================================================

    controller_id = row[
        "controller_id_v4"
    ]


    controller_team = str(
        row[
            "controller_team_v4"
        ]
    )


    controller = find_player(
        frame_players,
        controller_id
    )


    if controller is not None:

        cx = int(
            controller[
                "image_x"
            ]
        )

        cy = int(
            controller[
                "image_y"
            ]
        )


        cv2.circle(
            frame,
            (
                cx,
                cy
            ),
            19,
            CONTROLLER_COLOR,
            3
        )


        cv2.putText(
            frame,
            "CONTROLLER",
            (
                cx + 20,
                cy + 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            CONTROLLER_COLOR,
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # Reception candidate
    # ========================================================

    reception_id = row[
        "reception_candidate_id"
    ]


    reception_team = str(
        row[
            "reception_candidate_team"
        ]
    )


    reception_count = safe_int(
        row[
            "reception_candidate_count"
        ]
    )


    reception_player = find_player(
        frame_players,
        reception_id
    )


    if (
        reception_player is not None

        and

        reception_count is not None

        and

        reception_count > 0
    ):

        rx = int(
            reception_player[
                "image_x"
            ]
        )

        ry = int(
            reception_player[
                "image_y"
            ]
        )


        cv2.circle(
            frame,
            (
                rx,
                ry
            ),
            16,
            RECEPTION_COLOR,
            2
        )


        cv2.putText(
            frame,
            (
                f"RECEPTION "
                f"{reception_count}/3"
            ),
            (
                rx + 18,
                ry + 28
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            RECEPTION_COLOR,
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # Main panel
    # ========================================================

    draw_panel(
        frame,
        15,
        15,
        830,
        320
    )


    cv2.putText(
        frame,
        (
            f"TEAM POSSESSION: "
            f"{possession_state}"
        ),
        (
            30,
            49
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        state_color,
        2,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # Ball state
    # --------------------------------------------------------

    if ball_state == "InTransit":

        ball_state_color = (
            IN_TRANSIT_COLOR
        )

    elif ball_state == "AerialContest":

        ball_state_color = (
            AERIAL_COLOR
        )

    elif ball_state == "Controlled":

        ball_state_color = (
            CONTROLLER_COLOR
        )

    else:

        ball_state_color = (
            WHITE
        )


    cv2.putText(
        frame,
        (
            f"BALL STATE: "
            f"{ball_state}"
        ),
        (
            30,
            82
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        ball_state_color,
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        (
            f"Raw state: "
            f"{raw_state}"
        ),
        (
            30,
            111
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.49,
        WHITE,
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        (
            f"Source: "
            f"{source}"
        ),
        (
            30,
            139
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.49,
        WHITE,
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # Controller text
    # ========================================================

    if (
        controller is not None

        and

        controller_team
        in [
            "A",
            "B"
        ]
    ):

        controller_text = (
            f"CONTROLLER: "
            f"{controller_team} "
            f"#{int(controller_id)}"
        )

    else:

        controller_text = (
            "CONTROLLER: NONE"
        )


    cv2.putText(
        frame,
        controller_text,
        (
            30,
            169
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        CONTROLLER_COLOR,
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # Distances
    # ========================================================

    A_pitch = row[
        "nearest_A_pitch_distance_m"
    ]

    A_foot = row[
        "nearest_A_image_foot_distance_px"
    ]


    B_pitch = row[
        "nearest_B_pitch_distance_m"
    ]

    B_foot = row[
        "nearest_B_image_foot_distance_px"
    ]


    if (
        pd.notna(A_pitch)
        and
        pd.notna(A_foot)
        and
        pd.notna(nearest_A_id)
    ):

        A_text = (
            f"Nearest A #{int(nearest_A_id)} | "
            f"pitch={A_pitch:.2f}m | "
            f"foot={A_foot:.0f}px"
        )

    else:

        A_text = (
            "Nearest A: unavailable"
        )


    if (
        pd.notna(B_pitch)
        and
        pd.notna(B_foot)
        and
        pd.notna(nearest_B_id)
    ):

        B_text = (
            f"Nearest B #{int(nearest_B_id)} | "
            f"pitch={B_pitch:.2f}m | "
            f"foot={B_foot:.0f}px"
        )

    else:

        B_text = (
            "Nearest B: unavailable"
        )


    cv2.putText(
        frame,
        A_text,
        (
            30,
            199
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        TEAM_A_COLOR,
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        B_text,
        (
            30,
            228
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        TEAM_B_COLOR,
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # Ball speed
    # ========================================================

    ball_speed = row[
        "ball_speed_mps"
    ]


    if pd.notna(
        ball_speed
    ):

        speed_text = (
            f"Ball speed: "
            f"{ball_speed:.2f} m/s"
        )

    else:

        speed_text = (
            "Ball speed: unavailable"
        )


    cv2.putText(
        frame,
        speed_text,
        (
            30,
            258
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        BALL_COLOR,
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # Reception
    # ========================================================

    if (
        reception_count is not None
        and
        reception_count > 0
    ):

        reception_text = (
            f"Reception candidate: "
            f"{reception_team} "
            f"#{int(reception_id)} "
            f"{reception_count}/3"
        )

    else:

        reception_text = (
            "Reception candidate: NONE"
        )


    cv2.putText(
        frame,
        reception_text,
        (
            30,
            287
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        RECEPTION_COLOR,
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # State banners
    # ========================================================

    if ball_state == "InTransit":

        cv2.rectangle(
            frame,
            (
                width - 480,
                20
            ),
            (
                width - 20,
                70
            ),
            IN_TRANSIT_COLOR,
            -1
        )


        cv2.putText(
            frame,
            (
                f"IN TRANSIT | "
                f"retain {possession_state}"
            ),
            (
                width - 455,
                54
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.63,
            BLACK,
            2,
            cv2.LINE_AA
        )


    elif ball_state == "AerialContest":

        cv2.rectangle(
            frame,
            (
                width - 500,
                20
            ),
            (
                width - 20,
                70
            ),
            AERIAL_COLOR,
            -1
        )


        cv2.putText(
            frame,
            "AERIAL CONTEST",
            (
                width - 465,
                54
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            BLACK,
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # Reception confirmed banner
    # ========================================================

    if (
        source
        ==
        "confirmed_reception"
    ):

        cv2.rectangle(
            frame,
            (
                width // 2 - 240,
                340
            ),
            (
                width // 2 + 240,
                395
            ),
            GREEN,
            -1
        )


        cv2.putText(
            frame,
            (
                f"RECEPTION CONFIRMED "
                f"-> {possession_state}"
            ),
            (
                width // 2 - 215,
                377
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            BLACK,
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # Turnover confirmed banner
    # ========================================================

    if (
        source
        ==
        "confirmed_controlled_turnover"
    ):

        cv2.rectangle(
            frame,
            (
                width // 2 - 300,
                335
            ),
            (
                width // 2 + 300,
                400
            ),
            state_color,
            -1
        )


        cv2.putText(
            frame,
            (
                f"CONTROLLED TURNOVER "
                f"-> TEAM "
                f"{possession_state}"
            ),
            (
                width // 2 - 275,
                377
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            BLACK,
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # Regression banner
    # ========================================================

    if frame_idx in REGRESSION_FRAMES:

        regression_text = (
            REGRESSION_FRAMES[
                frame_idx
            ]
        )


        cv2.rectangle(
            frame,
            (
                20,
                height - 120
            ),
            (
                width - 20,
                height - 65
            ),
            RED,
            -1
        )


        cv2.putText(
            frame,
            (
                f"REGRESSION TEST | "
                f"{regression_text}"
            ),
            (
                40,
                height - 84
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            WHITE,
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # Frame/time
    # ========================================================

    cv2.putText(
        frame,
        (
            f"Frame {frame_idx} | "
            f"{frame_idx / fps:.2f}s"
        ),
        (
            width - 310,
            height - 61
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        WHITE,
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # Timeline
    # ========================================================

    draw_timeline(
        frame,
        frame_idx
    )


    writer.write(
        frame
    )


    frame_idx += 1


# ============================================================
# Finish
# ============================================================

cap.release()
writer.release()


# ============================================================
# Console
# ============================================================

print("")
print(
    "========================================"
)

print(
    "POSSESSION V4 VISUAL QA"
)

print(
    "========================================"
)

print("")

print(
    "QA Video:",
    OUTPUT_VIDEO
)

print("")

print(
    "Confirmed receptions:",
    len(receptions)
)


if len(receptions) > 0:

    print("")

    print(
        receptions[
            reception_columns
        ]
        .round(2)
        .to_string(
            index=False
        )
    )


print("")

print(
    "Confirmed controlled turnovers:",
    len(turnovers)
)


if len(turnovers) > 0:

    print("")

    print(
        turnovers[
            turnover_columns
        ]
        .round(2)
        .to_string(
            index=False
        )
    )


print("")

print(
    "Aerial contest segments:",
    len(aerial_df)
)


if len(aerial_df) > 0:

    print("")

    print(
        aerial_df
        .round(2)
        .to_string(
            index=False
        )
    )


print("")

print(
    "Regression frames:"
)

for frame, description in (
    REGRESSION_FRAMES.items()
):

    row = possession[
        possession["frame"]
        ==
        frame
    ]

    if len(row) == 0:

        continue


    row = row.iloc[0]


    print(
        f"frame={frame} | "
        f"time={row['time_sec']:.2f}s | "
        f"speed={row['ball_speed_mps']:.2f} | "
        f"ball_state={row['ball_state_v4']} | "
        f"possession={row['possession_v4']} | "
        f"controller={row['controller_team_v4']} "
        f"{row['controller_id_v4']} | "
        f"{description}"
    )


print("")

print(
    "Reception CSV:",
    OUTPUT_RECEPTION_CSV
)

print(
    "Turnover CSV:",
    OUTPUT_TURNOVER_CSV
)

print(
    "Aerial CSV:",
    OUTPUT_AERIAL_CSV
)
