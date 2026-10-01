import os
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PLAYER_CSV = "outputs/tracking_data_tactical.csv"
BALL_CSV = "outputs/ball_tracking_ensemble_trusted.csv"

OUTPUT_FRAME_CSV = "outputs/possession_v1_frames.csv"
OUTPUT_SUMMARY_CSV = "outputs/possession_v1_summary.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Parameters
# ============================================================

FPS = 25.0


# ------------------------------------------------------------
# Ball control distance
#
# Conservative first version.
# ------------------------------------------------------------

CONTROL_RADIUS_M = 2.5


# If both teams have players near the ball:
CONTESTED_RADIUS_M = 3.0

# Difference between nearest Team A and Team B distances
# smaller than this -> contested.
CONTESTED_DISTANCE_MARGIN_M = 0.75


# ------------------------------------------------------------
# Temporal smoothing
#
# A new possession needs several consecutive frames
# before replacing the previous team.
# ------------------------------------------------------------

SWITCH_CONFIRM_FRAMES = 4


# ============================================================
# Load
# ============================================================

players = pd.read_csv(
    PLAYER_CSV
)

ball = pd.read_csv(
    BALL_CSV
)


# ============================================================
# Find player pitch coordinate columns
# ============================================================

possible_x_columns = [
    "pitch_x_final_m",
    "pitch_x_clean_m",
    "pitch_x_m",
]

possible_y_columns = [
    "pitch_y_final_m",
    "pitch_y_clean_m",
    "pitch_y_m",
]


PLAYER_X = next(
    (
        c
        for c in possible_x_columns
        if c in players.columns
    ),
    None
)

PLAYER_Y = next(
    (
        c
        for c in possible_y_columns
        if c in players.columns
    ),
    None
)


if PLAYER_X is None or PLAYER_Y is None:

    raise RuntimeError(
        "Cannot find player pitch coordinate columns.\n"
        f"Available columns:\n{players.columns.tolist()}"
    )


print(
    "Using player coordinates:",
    PLAYER_X,
    PLAYER_Y
)


# ============================================================
# Ball usable normalization
# ============================================================

ball["ball_usable_bool"] = (
    ball["ball_usable"]
    .astype(str)
    .str.lower()
    .isin(["true", "1"])
)


# ============================================================
# Player normalization
# ============================================================

players["frame"] = pd.to_numeric(
    players["frame"],
    errors="coerce"
)

players[PLAYER_X] = pd.to_numeric(
    players[PLAYER_X],
    errors="coerce"
)

players[PLAYER_Y] = pd.to_numeric(
    players[PLAYER_Y],
    errors="coerce"
)


# ============================================================
# Ball normalization
# ============================================================

ball["frame"] = pd.to_numeric(
    ball["frame"],
    errors="coerce"
)

ball["pitch_x_trusted_m"] = pd.to_numeric(
    ball["pitch_x_trusted_m"],
    errors="coerce"
)

ball["pitch_y_trusted_m"] = pd.to_numeric(
    ball["pitch_y_trusted_m"],
    errors="coerce"
)


# ============================================================
# Only frames existing in tactical player dataset
# ============================================================

tactical_frames = sorted(
    players["frame"]
    .dropna()
    .astype(int)
    .unique()
)


ball_by_frame = (
    ball
    .set_index("frame")
)


# ============================================================
# Helpers
# ============================================================

def nearest_player_for_team(
    frame_players,
    team_name,
    ball_x,
    ball_y
):

    team_players = frame_players[
        frame_players["team"]
        ==
        team_name
    ].copy()


    if len(team_players) == 0:

        return {
            "distance": np.nan,
            "track_id": np.nan,
            "x": np.nan,
            "y": np.nan,
        }


    dx = (
        team_players[PLAYER_X]
        -
        ball_x
    )

    dy = (
        team_players[PLAYER_Y]
        -
        ball_y
    )


    distances = np.sqrt(
        dx ** 2
        +
        dy ** 2
    )


    valid = distances.notna()


    if not valid.any():

        return {
            "distance": np.nan,
            "track_id": np.nan,
            "x": np.nan,
            "y": np.nan,
        }


    nearest_index = (
        distances[valid]
        .idxmin()
    )


    row = team_players.loc[
        nearest_index
    ]


    return {
        "distance": float(
            distances.loc[
                nearest_index
            ]
        ),

        "track_id": row[
            "track_id"
        ],

        "x": row[
            PLAYER_X
        ],

        "y": row[
            PLAYER_Y
        ],
    }


# ============================================================
# Raw possession
# ============================================================

rows = []


for frame in tactical_frames:

    frame_players = players[
        players["frame"]
        ==
        frame
    ].copy()


    time_sec = (
        frame / FPS
    )


    # --------------------------------------------------------
    # Ball unavailable
    # --------------------------------------------------------

    if frame not in ball_by_frame.index:

        rows.append(
            {
                "frame": frame,
                "time_sec": time_sec,

                "ball_usable": False,

                "ball_x_m": np.nan,
                "ball_y_m": np.nan,

                "nearest_A_id": np.nan,
                "nearest_A_distance_m": np.nan,

                "nearest_B_id": np.nan,
                "nearest_B_distance_m": np.nan,

                "nearest_player_id": np.nan,
                "nearest_team": None,
                "nearest_distance_m": np.nan,

                "raw_possession": "Unknown",
            }
        )

        continue


    ball_row = ball_by_frame.loc[
        frame
    ]


    # Safety if duplicated
    if isinstance(
        ball_row,
        pd.DataFrame
    ):

        ball_row = ball_row.iloc[0]


    usable = bool(
        ball_row[
            "ball_usable_bool"
        ]
    )


    ball_x = ball_row[
        "pitch_x_trusted_m"
    ]

    ball_y = ball_row[
        "pitch_y_trusted_m"
    ]


    if (
        not usable
        or
        pd.isna(ball_x)
        or
        pd.isna(ball_y)
    ):

        rows.append(
            {
                "frame": frame,
                "time_sec": time_sec,

                "ball_usable": False,

                "ball_x_m": np.nan,
                "ball_y_m": np.nan,

                "nearest_A_id": np.nan,
                "nearest_A_distance_m": np.nan,

                "nearest_B_id": np.nan,
                "nearest_B_distance_m": np.nan,

                "nearest_player_id": np.nan,
                "nearest_team": None,
                "nearest_distance_m": np.nan,

                "raw_possession": "Unknown",
            }
        )

        continue


    # ========================================================
    # Nearest player from each team
    # ========================================================

    nearest_A = nearest_player_for_team(
        frame_players,
        "A",
        ball_x,
        ball_y
    )


    nearest_B = nearest_player_for_team(
        frame_players,
        "B",
        ball_x,
        ball_y
    )


    dA = nearest_A[
        "distance"
    ]

    dB = nearest_B[
        "distance"
    ]


    # ========================================================
    # Overall nearest
    # ========================================================

    if (
        pd.notna(dA)
        and
        (
            pd.isna(dB)
            or
            dA <= dB
        )
    ):

        nearest_team = "A"

        nearest_player_id = (
            nearest_A[
                "track_id"
            ]
        )

        nearest_distance = dA


    elif pd.notna(dB):

        nearest_team = "B"

        nearest_player_id = (
            nearest_B[
                "track_id"
            ]
        )

        nearest_distance = dB


    else:

        nearest_team = None

        nearest_player_id = np.nan

        nearest_distance = np.nan


    # ========================================================
    # Raw possession classification
    # ========================================================

    raw_possession = "Unknown"


    # --------------------------------------------------------
    # Contested
    # --------------------------------------------------------

    if (
        pd.notna(dA)
        and
        pd.notna(dB)
        and
        dA <= CONTESTED_RADIUS_M
        and
        dB <= CONTESTED_RADIUS_M
        and
        abs(
            dA - dB
        )
        <=
        CONTESTED_DISTANCE_MARGIN_M
    ):

        raw_possession = (
            "Contested"
        )


    # --------------------------------------------------------
    # Team A possession
    # --------------------------------------------------------

    elif (
        pd.notna(dA)
        and
        dA <= CONTROL_RADIUS_M
        and
        (
            pd.isna(dB)
            or
            dA < dB
        )
    ):

        raw_possession = "A"


    # --------------------------------------------------------
    # Team B possession
    # --------------------------------------------------------

    elif (
        pd.notna(dB)
        and
        dB <= CONTROL_RADIUS_M
        and
        (
            pd.isna(dA)
            or
            dB < dA
        )
    ):

        raw_possession = "B"


    rows.append(
        {
            "frame": frame,
            "time_sec": time_sec,

            "ball_usable": True,

            "ball_x_m": ball_x,
            "ball_y_m": ball_y,

            "nearest_A_id":
                nearest_A[
                    "track_id"
                ],

            "nearest_A_distance_m":
                dA,

            "nearest_B_id":
                nearest_B[
                    "track_id"
                ],

            "nearest_B_distance_m":
                dB,

            "nearest_player_id":
                nearest_player_id,

            "nearest_team":
                nearest_team,

            "nearest_distance_m":
                nearest_distance,

            "raw_possession":
                raw_possession,
        }
    )


result = pd.DataFrame(
    rows
)


# ============================================================
# Temporal possession smoothing / hysteresis
# ============================================================

result[
    "possession"
] = "Unknown"


current_team = None

candidate_team = None

candidate_count = 0


for i in range(len(result)):

    raw = result.loc[
        i,
        "raw_possession"
    ]


    # ========================================================
    # Unknown
    #
    # Do not invent possession.
    # ========================================================

    if raw == "Unknown":

        result.loc[
            i,
            "possession"
        ] = "Unknown"

        candidate_team = None
        candidate_count = 0

        continue


    # ========================================================
    # Contested
    # ========================================================

    if raw == "Contested":

        result.loc[
            i,
            "possession"
        ] = "Contested"

        candidate_team = None
        candidate_count = 0

        continue


    # ========================================================
    # First known possession
    # ========================================================

    if current_team is None:

        current_team = raw

        result.loc[
            i,
            "possession"
        ] = raw

        continue


    # ========================================================
    # Same team
    # ========================================================

    if raw == current_team:

        candidate_team = None
        candidate_count = 0

        result.loc[
            i,
            "possession"
        ] = current_team

        continue


    # ========================================================
    # Potential switch
    # ========================================================

    if candidate_team == raw:

        candidate_count += 1

    else:

        candidate_team = raw
        candidate_count = 1


    # Require consecutive evidence
    if (
        candidate_count
        >=
        SWITCH_CONFIRM_FRAMES
    ):

        current_team = raw

        candidate_team = None
        candidate_count = 0


    result.loc[
        i,
        "possession"
    ] = current_team


# ============================================================
# Save frame-level result
# ============================================================

result.to_csv(
    OUTPUT_FRAME_CSV,
    index=False
)


# ============================================================
# Summary
# ============================================================

counts = (
    result[
        "possession"
    ]
    .value_counts()
)


total_frames = len(
    result
)


known_team_frames = int(
    result[
        "possession"
    ]
    .isin(
        [
            "A",
            "B"
        ]
    )
    .sum()
)


A_frames = int(
    (
        result[
            "possession"
        ]
        ==
        "A"
    ).sum()
)


B_frames = int(
    (
        result[
            "possession"
        ]
        ==
        "B"
    ).sum()
)


contested_frames = int(
    (
        result[
            "possession"
        ]
        ==
        "Contested"
    ).sum()
)


unknown_frames = int(
    (
        result[
            "possession"
        ]
        ==
        "Unknown"
    ).sum()
)


if known_team_frames > 0:

    A_share_known = (
        A_frames
        /
        known_team_frames
        *
        100.0
    )

    B_share_known = (
        B_frames
        /
        known_team_frames
        *
        100.0
    )

else:

    A_share_known = 0.0
    B_share_known = 0.0


summary = pd.DataFrame(
    [
        {
            "team": "A",
            "frames": A_frames,
            "seconds": A_frames / FPS,
            "share_of_known_possession_pct":
                A_share_known,
        },

        {
            "team": "B",
            "frames": B_frames,
            "seconds": B_frames / FPS,
            "share_of_known_possession_pct":
                B_share_known,
        },

        {
            "team": "Contested",
            "frames": contested_frames,
            "seconds": contested_frames / FPS,
            "share_of_known_possession_pct":
                np.nan,
        },

        {
            "team": "Unknown",
            "frames": unknown_frames,
            "seconds": unknown_frames / FPS,
            "share_of_known_possession_pct":
                np.nan,
        },
    ]
)


summary.to_csv(
    OUTPUT_SUMMARY_CSV,
    index=False
)


# ============================================================
# Distance statistics
# ============================================================

valid_distances = (
    result[
        "nearest_distance_m"
    ]
    .dropna()
)


# ============================================================
# Print
# ============================================================

print("")
print(
    "========================================"
)

print(
    "POSSESSION V1 SUMMARY"
)

print(
    "========================================"
)

print("")

print(
    "Tactical frames:",
    total_frames
)

if total_frames > 0:

    print(
        "Window:",
        f"{result['time_sec'].min():.2f}s",
        "->",
        f"{result['time_sec'].max():.2f}s"
    )


print("")

print(
    "Team A:",
    A_frames,
    "frames",
    f"({A_frames / FPS:.2f}s)"
)

print(
    "Team B:",
    B_frames,
    "frames",
    f"({B_frames / FPS:.2f}s)"
)

print(
    "Contested:",
    contested_frames,
    "frames"
)

print(
    "Unknown:",
    unknown_frames,
    "frames"
)


print("")

print(
    "Known possession only:"
)

print(
    "Team A:",
    f"{A_share_known:.1f}%"
)

print(
    "Team B:",
    f"{B_share_known:.1f}%"
)


if len(valid_distances) > 0:

    print("")

    print(
        "Nearest-player distance median:",
        f"{valid_distances.median():.2f} m"
    )

    print(
        "Nearest-player distance P75:",
        f"{valid_distances.quantile(0.75):.2f} m"
    )

    print(
        "Nearest-player distance P95:",
        f"{valid_distances.quantile(0.95):.2f} m"
    )


print("")
print(
    "Raw possession:"
)

print(
    result[
        "raw_possession"
    ]
    .value_counts()
    .to_string()
)


print("")
print(
    "Smoothed possession:"
)

print(
    result[
        "possession"
    ]
    .value_counts()
    .to_string()
)


print("")

print(
    "Frame CSV:",
    OUTPUT_FRAME_CSV
)

print(
    "Summary CSV:",
    OUTPUT_SUMMARY_CSV
)
