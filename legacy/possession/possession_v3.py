import os
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PLAYER_CSV = "outputs/tracking_data_clean_v2.csv"
BALL_CSV = "outputs/ball_tracking_ensemble_trusted.csv"

OUTPUT_FRAME_CSV = "outputs/possession_v3_frames.csv"
OUTPUT_SUMMARY_CSV = "outputs/possession_v3_summary.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Parameters
# ============================================================

FPS = 25.0


# ============================================================
# Player frame quality
#
# V2 required exactly 10v10.
# V3 only requires enough stable players to make
# nearest-player reasoning meaningful.
# ============================================================

MIN_PLAYERS_PER_TEAM = 6


# ============================================================
# Track/team stability
#
# Track must belong to its dominant team sufficiently often.
# Rows that disagree with the dominant team are removed.
#
# This removes known team / ID contamination without
# throwing away the whole frame.
# ============================================================

MIN_DOMINANT_TEAM_SHARE = 0.70


# ============================================================
# Direct control
# ============================================================

# Pitch-space operational radius.
CONTROL_RADIUS_M = 4.0


# IMPORTANT:
# Player image_x/image_y in our tracking pipeline corresponds
# to the player's ground / foot anchor used for homography.
#
# Ball must ALSO be reasonably close to that foot anchor
# in image space.
#
# This rejects many aerial-ball projection mistakes.
FOOT_CONTROL_MAX_PX = 95.0


# ============================================================
# Contested
# ============================================================

CONTESTED_DISTANCE_MARGIN_M = 0.75

CONTESTED_FOOT_MARGIN_PX = 35.0


# ============================================================
# Temporal possession
# ============================================================

SWITCH_CONFIRM_FRAMES = 4


# Ball visible but nobody genuinely controls it.
#
# Covers:
# - pass in transit
# - aerial ball
# - loose ball
#
# 40 frames = 1.60 sec.
IN_FLIGHT_HOLD_FRAMES = 40


# Ball detection temporary disappearance.
#
# 8 frames = 0.32 sec.
BALL_MISSING_HOLD_FRAMES = 8


# Poor player-frame geometry.
PLAYER_QUALITY_HOLD_FRAMES = 5


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
# Detect player coordinate columns
# ============================================================

PLAYER_PITCH_X_OPTIONS = [
    "pitch_x_final_m",
    "pitch_x_clean_m",
    "pitch_x_m",
]

PLAYER_PITCH_Y_OPTIONS = [
    "pitch_y_final_m",
    "pitch_y_clean_m",
    "pitch_y_m",
]


PLAYER_X = next(
    (
        column
        for column in PLAYER_PITCH_X_OPTIONS
        if column in players.columns
    ),
    None
)

PLAYER_Y = next(
    (
        column
        for column in PLAYER_PITCH_Y_OPTIONS
        if column in players.columns
    ),
    None
)


if PLAYER_X is None or PLAYER_Y is None:

    raise RuntimeError(
        "Cannot find player pitch coordinate columns.\n"
        f"Available columns:\n{players.columns.tolist()}"
    )


if (
    "image_x" not in players.columns
    or
    "image_y" not in players.columns
):

    raise RuntimeError(
        "tracking_data_clean_v2.csv must contain "
        "image_x and image_y."
    )


print(
    "Using player pitch coordinates:",
    PLAYER_X,
    PLAYER_Y
)


# ============================================================
# Normalize player data
# ============================================================

players["frame"] = pd.to_numeric(
    players["frame"],
    errors="coerce"
)

players["track_id"] = pd.to_numeric(
    players["track_id"],
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

players["image_x"] = pd.to_numeric(
    players["image_x"],
    errors="coerce"
)

players["image_y"] = pd.to_numeric(
    players["image_y"],
    errors="coerce"
)


players = players.dropna(
    subset=[
        "frame",
        "track_id",
    ]
)


players["frame"] = (
    players["frame"]
    .astype(int)
)


# ============================================================
# Keep only A/B players
# ============================================================

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
# Stable team assignment per track
# ============================================================

track_team_stats = []


for track_id, group in players.groupby(
    "track_id"
):

    team_counts = (
        group["team"]
        .value_counts()
    )


    if len(team_counts) == 0:

        continue


    dominant_team = (
        team_counts.index[0]
    )


    dominant_count = int(
        team_counts.iloc[0]
    )


    total_count = int(
        len(group)
    )


    dominant_share = (
        dominant_count
        /
        total_count
    )


    track_team_stats.append(
        {
            "track_id":
                track_id,

            "dominant_team":
                dominant_team,

            "dominant_share":
                dominant_share,

            "rows":
                total_count,
        }
    )


track_team_stats = pd.DataFrame(
    track_team_stats
)


stable_track_stats = track_team_stats[
    track_team_stats[
        "dominant_share"
    ]
    >=
    MIN_DOMINANT_TEAM_SHARE
].copy()


stable_track_ids = set(
    stable_track_stats[
        "track_id"
    ].tolist()
)


dominant_team_map = dict(
    zip(
        stable_track_stats[
            "track_id"
        ],
        stable_track_stats[
            "dominant_team"
        ],
    )
)


# ============================================================
# Remove unstable tracks
# ============================================================

players = players[
    players["track_id"]
    .isin(
        stable_track_ids
    )
].copy()


# ============================================================
# Remove rows whose current team disagrees with stable team
#
# Do NOT relabel.
# We simply do not trust those rows.
# ============================================================

players[
    "stable_team"
] = players[
    "track_id"
].map(
    dominant_team_map
)


team_consistent_mask = (
    players["team"]
    ==
    players["stable_team"]
)


removed_inconsistent_rows = int(
    (
        ~team_consistent_mask
    ).sum()
)


players = players[
    team_consistent_mask
].copy()


players["team"] = (
    players["stable_team"]
)


# ============================================================
# Normalize ball data
# ============================================================

ball["frame"] = pd.to_numeric(
    ball["frame"],
    errors="coerce"
)


ball = ball.dropna(
    subset=["frame"]
)


ball["frame"] = (
    ball["frame"]
    .astype(int)
)


ball["ball_usable_bool"] = (
    ball["ball_usable"]
    .astype(str)
    .str.lower()
    .isin(
        [
            "true",
            "1"
        ]
    )
)


for column in [
    "pitch_x_trusted_m",
    "pitch_y_trusted_m",
    "image_x_trusted",
    "image_y_trusted",
    "trusted_speed_mps",
]:

    if column in ball.columns:

        ball[column] = pd.to_numeric(
            ball[column],
            errors="coerce"
        )


# ============================================================
# Index
# ============================================================

players_by_frame = {
    int(frame):
        group.copy()

    for frame, group
    in players.groupby("frame")
}


ball_by_frame = (
    ball
    .set_index("frame")
)


# ============================================================
# Full available frame range
# ============================================================

min_frame = max(
    int(players["frame"].min()),
    int(ball["frame"].min()),
)


max_frame = min(
    int(players["frame"].max()),
    int(ball["frame"].max()),
)


all_frames = range(
    min_frame,
    max_frame + 1
)


# ============================================================
# Nearest player helper
# ============================================================

def nearest_player(
    frame_players,
    team,
    ball_pitch_x,
    ball_pitch_y,
    ball_image_x,
    ball_image_y,
):

    team_players = frame_players[
        frame_players["team"]
        ==
        team
    ].copy()


    team_players = team_players.dropna(
        subset=[
            PLAYER_X,
            PLAYER_Y,
            "image_x",
            "image_y",
        ]
    )


    if len(team_players) == 0:

        return {
            "track_id": np.nan,

            "pitch_distance_m":
                np.nan,

            "image_foot_distance_px":
                np.nan,

            "direct_control":
                False,
        }


    # ========================================================
    # Pitch-space distance
    # ========================================================

    pitch_dx = (
        team_players[PLAYER_X]
        -
        ball_pitch_x
    )


    pitch_dy = (
        team_players[PLAYER_Y]
        -
        ball_pitch_y
    )


    pitch_distance = np.sqrt(
        pitch_dx ** 2
        +
        pitch_dy ** 2
    )


    # ========================================================
    # Image-space foot distance
    # ========================================================

    image_dx = (
        team_players["image_x"]
        -
        ball_image_x
    )


    image_dy = (
        team_players["image_y"]
        -
        ball_image_y
    )


    image_distance = np.sqrt(
        image_dx ** 2
        +
        image_dy ** 2
    )


    # ========================================================
    # Candidate score
    #
    # Pitch proximity first.
    # ========================================================

    valid = (
        pitch_distance.notna()
        &
        image_distance.notna()
    )


    if not valid.any():

        return {
            "track_id": np.nan,

            "pitch_distance_m":
                np.nan,

            "image_foot_distance_px":
                np.nan,

            "direct_control":
                False,
        }


    nearest_index = (
        pitch_distance[
            valid
        ]
        .idxmin()
    )


    p_dist = float(
        pitch_distance.loc[
            nearest_index
        ]
    )


    i_dist = float(
        image_distance.loc[
            nearest_index
        ]
    )


    row = team_players.loc[
        nearest_index
    ]


    direct = (
        p_dist
        <=
        CONTROL_RADIUS_M

        and

        i_dist
        <=
        FOOT_CONTROL_MAX_PX
    )


    return {
        "track_id":
            row["track_id"],

        "pitch_distance_m":
            p_dist,

        "image_foot_distance_px":
            i_dist,

        "direct_control":
            direct,
    }


# ============================================================
# PASS 1
# Direct evidence
# ============================================================

rows = []


projection_mismatch_count = 0


for frame in all_frames:

    time_sec = (
        frame / FPS
    )


    frame_players = players_by_frame.get(
        frame,
        None
    )


    # ========================================================
    # Player frame quality
    # ========================================================

    if frame_players is None:

        A_count = 0
        B_count = 0

    else:

        A_count = int(
            (
                frame_players["team"]
                ==
                "A"
            ).sum()
        )


        B_count = int(
            (
                frame_players["team"]
                ==
                "B"
            ).sum()
        )


    player_frame_good = (
        A_count
        >=
        MIN_PLAYERS_PER_TEAM

        and

        B_count
        >=
        MIN_PLAYERS_PER_TEAM
    )


    # ========================================================
    # Ball
    # ========================================================

    if frame not in ball_by_frame.index:

        ball_usable = False
        ball_row = None

    else:

        ball_row = ball_by_frame.loc[
            frame
        ]


        if isinstance(
            ball_row,
            pd.DataFrame
        ):

            ball_row = (
                ball_row.iloc[0]
            )


        ball_usable = bool(
            ball_row[
                "ball_usable_bool"
            ]
        )


    # ========================================================
    # Default output
    # ========================================================

    output = {
        "frame":
            frame,

        "time_sec":
            time_sec,

        "A_player_count":
            A_count,

        "B_player_count":
            B_count,

        "player_frame_good":
            player_frame_good,

        "ball_usable":
            ball_usable,

        "ball_quality":
            None,

        "ball_speed_mps":
            np.nan,

        "ball_pitch_x_m":
            np.nan,

        "ball_pitch_y_m":
            np.nan,

        "ball_image_x":
            np.nan,

        "ball_image_y":
            np.nan,

        "nearest_A_id":
            np.nan,

        "nearest_A_pitch_distance_m":
            np.nan,

        "nearest_A_image_foot_distance_px":
            np.nan,

        "nearest_B_id":
            np.nan,

        "nearest_B_pitch_distance_m":
            np.nan,

        "nearest_B_image_foot_distance_px":
            np.nan,

        "controller_team":
            None,

        "controller_id":
            np.nan,

        "direct_state":
            "Unknown",

        "projection_mismatch":
            False,
    }


    # ========================================================
    # Player data insufficient
    # ========================================================

    if not player_frame_good:

        output[
            "direct_state"
        ] = "PlayerQualityLow"

        rows.append(
            output
        )

        continue


    # ========================================================
    # Ball unavailable
    # ========================================================

    if (
        not ball_usable
        or
        ball_row is None
    ):

        output[
            "direct_state"
        ] = "BallMissing"

        rows.append(
            output
        )

        continue


    ball_pitch_x = ball_row[
        "pitch_x_trusted_m"
    ]

    ball_pitch_y = ball_row[
        "pitch_y_trusted_m"
    ]

    ball_image_x = ball_row[
        "image_x_trusted"
    ]

    ball_image_y = ball_row[
        "image_y_trusted"
    ]


    if (
        pd.isna(ball_pitch_x)
        or
        pd.isna(ball_pitch_y)
        or
        pd.isna(ball_image_x)
        or
        pd.isna(ball_image_y)
    ):

        output[
            "direct_state"
        ] = "BallMissing"

        rows.append(
            output
        )

        continue


    output[
        "ball_quality"
    ] = ball_row.get(
        "ball_quality",
        None
    )


    output[
        "ball_speed_mps"
    ] = ball_row.get(
        "trusted_speed_mps",
        np.nan
    )


    output[
        "ball_pitch_x_m"
    ] = ball_pitch_x


    output[
        "ball_pitch_y_m"
    ] = ball_pitch_y


    output[
        "ball_image_x"
    ] = ball_image_x


    output[
        "ball_image_y"
    ] = ball_image_y


    # ========================================================
    # Nearest A / B
    # ========================================================

    nearest_A = nearest_player(
        frame_players,
        "A",
        ball_pitch_x,
        ball_pitch_y,
        ball_image_x,
        ball_image_y,
    )


    nearest_B = nearest_player(
        frame_players,
        "B",
        ball_pitch_x,
        ball_pitch_y,
        ball_image_x,
        ball_image_y,
    )


    output[
        "nearest_A_id"
    ] = nearest_A[
        "track_id"
    ]


    output[
        "nearest_A_pitch_distance_m"
    ] = nearest_A[
        "pitch_distance_m"
    ]


    output[
        "nearest_A_image_foot_distance_px"
    ] = nearest_A[
        "image_foot_distance_px"
    ]


    output[
        "nearest_B_id"
    ] = nearest_B[
        "track_id"
    ]


    output[
        "nearest_B_pitch_distance_m"
    ] = nearest_B[
        "pitch_distance_m"
    ]


    output[
        "nearest_B_image_foot_distance_px"
    ] = nearest_B[
        "image_foot_distance_px"
    ]


    A_control = nearest_A[
        "direct_control"
    ]


    B_control = nearest_B[
        "direct_control"
    ]


    # ========================================================
    # Detect projection mismatch
    #
    # Pitch says somebody is close,
    # image says ball is nowhere near their feet.
    #
    # Typical aerial-ball case.
    # ========================================================

    pitch_close = False
    foot_far = True


    for nearest in [
        nearest_A,
        nearest_B
    ]:

        p = nearest[
            "pitch_distance_m"
        ]

        img = nearest[
            "image_foot_distance_px"
        ]


        if (
            pd.notna(p)
            and
            p <= CONTROL_RADIUS_M
        ):

            pitch_close = True


            if (
                pd.notna(img)
                and
                img
                <=
                FOOT_CONTROL_MAX_PX
            ):

                foot_far = False


    projection_mismatch = (
        pitch_close
        and
        foot_far
    )


    output[
        "projection_mismatch"
    ] = projection_mismatch


    if projection_mismatch:

        projection_mismatch_count += 1


    # ========================================================
    # Contested
    # ========================================================

    if (
        A_control
        and
        B_control
    ):

        pitch_diff = abs(
            nearest_A[
                "pitch_distance_m"
            ]
            -
            nearest_B[
                "pitch_distance_m"
            ]
        )


        foot_diff = abs(
            nearest_A[
                "image_foot_distance_px"
            ]
            -
            nearest_B[
                "image_foot_distance_px"
            ]
        )


        if (
            pitch_diff
            <=
            CONTESTED_DISTANCE_MARGIN_M

            or

            foot_diff
            <=
            CONTESTED_FOOT_MARGIN_PX
        ):

            output[
                "direct_state"
            ] = "Contested"


        elif (
            nearest_A[
                "image_foot_distance_px"
            ]
            <
            nearest_B[
                "image_foot_distance_px"
            ]
        ):

            output[
                "direct_state"
            ] = "A"

            output[
                "controller_team"
            ] = "A"

            output[
                "controller_id"
            ] = nearest_A[
                "track_id"
            ]


        else:

            output[
                "direct_state"
            ] = "B"

            output[
                "controller_team"
            ] = "B"

            output[
                "controller_id"
            ] = nearest_B[
                "track_id"
            ]


    # ========================================================
    # A control
    # ========================================================

    elif A_control:

        output[
            "direct_state"
        ] = "A"

        output[
            "controller_team"
        ] = "A"

        output[
            "controller_id"
        ] = nearest_A[
            "track_id"
        ]


    # ========================================================
    # B control
    # ========================================================

    elif B_control:

        output[
            "direct_state"
        ] = "B"

        output[
            "controller_team"
        ] = "B"

        output[
            "controller_id"
        ] = nearest_B[
            "track_id"
        ]


    # ========================================================
    # Ball visible, but nobody actually has it at their feet
    #
    # This is our InFlight / loose-ball state.
    # ========================================================

    else:

        output[
            "direct_state"
        ] = "InFlight"


    rows.append(
        output
    )


result = pd.DataFrame(
    rows
)


# ============================================================
# PASS 2
# Possession state machine
# ============================================================

result[
    "possession_v3"
] = "Unknown"


result[
    "possession_source"
] = "unknown"


result[
    "switch_candidate"
] = None


result[
    "switch_candidate_count"
] = 0


result[
    "in_flight_streak"
] = 0


current_team = None

switch_candidate = None
switch_count = 0

in_flight_streak = 0
missing_streak = 0
quality_streak = 0

confirmed_switches = 0


for i in range(len(result)):

    direct = result.loc[
        i,
        "direct_state"
    ]


    # ========================================================
    # Direct control by A/B
    # ========================================================

    if direct in [
        "A",
        "B"
    ]:

        in_flight_streak = 0
        missing_streak = 0
        quality_streak = 0


        # ----------------------------------------------------
        # First possession
        # ----------------------------------------------------

        if current_team is None:

            current_team = direct

            switch_candidate = None
            switch_count = 0


            result.loc[
                i,
                "possession_v3"
            ] = current_team


            result.loc[
                i,
                "possession_source"
            ] = "direct_control"


        # ----------------------------------------------------
        # Same team retains ball
        # ----------------------------------------------------

        elif direct == current_team:

            switch_candidate = None
            switch_count = 0


            result.loc[
                i,
                "possession_v3"
            ] = current_team


            result.loc[
                i,
                "possession_source"
            ] = "direct_control"


        # ----------------------------------------------------
        # Possible turnover
        # ----------------------------------------------------

        else:

            if (
                switch_candidate
                ==
                direct
            ):

                switch_count += 1

            else:

                switch_candidate = direct
                switch_count = 1


            if (
                switch_count
                >=
                SWITCH_CONFIRM_FRAMES
            ):

                current_team = direct

                confirmed_switches += 1

                switch_candidate = None
                switch_count = 0


                result.loc[
                    i,
                    "possession_v3"
                ] = current_team


                result.loc[
                    i,
                    "possession_source"
                ] = "confirmed_turnover"


            else:

                result.loc[
                    i,
                    "possession_v3"
                ] = current_team


                result.loc[
                    i,
                    "possession_source"
                ] = "pending_turnover"


    # ========================================================
    # In flight / pass / aerial ball / loose ball
    # ========================================================

    elif direct == "InFlight":

        switch_candidate = None
        switch_count = 0

        missing_streak = 0
        quality_streak = 0

        in_flight_streak += 1


        if (
            current_team is not None

            and

            in_flight_streak
            <=
            IN_FLIGHT_HOLD_FRAMES
        ):

            result.loc[
                i,
                "possession_v3"
            ] = current_team


            result.loc[
                i,
                "possession_source"
            ] = "carried_in_flight"


        else:

            result.loc[
                i,
                "possession_v3"
            ] = "Unknown"


            result.loc[
                i,
                "possession_source"
            ] = "in_flight_too_long"


    # ========================================================
    # Contested
    # ========================================================

    elif direct == "Contested":

        switch_candidate = None
        switch_count = 0

        in_flight_streak = 0
        missing_streak = 0
        quality_streak = 0


        result.loc[
            i,
            "possession_v3"
        ] = "Contested"


        result.loc[
            i,
            "possession_source"
        ] = "direct_contested"


        # NOTE:
        # current_team is deliberately NOT erased.
        #
        # If A had possession before the duel and A comes out
        # with the ball afterwards, continuity remains intact.


    # ========================================================
    # Ball missing
    # ========================================================

    elif direct == "BallMissing":

        switch_candidate = None
        switch_count = 0

        in_flight_streak = 0
        quality_streak = 0

        missing_streak += 1


        if (
            current_team is not None

            and

            missing_streak
            <=
            BALL_MISSING_HOLD_FRAMES
        ):

            result.loc[
                i,
                "possession_v3"
            ] = current_team


            result.loc[
                i,
                "possession_source"
            ] = "carried_ball_missing"


        else:

            result.loc[
                i,
                "possession_v3"
            ] = "Unknown"


            result.loc[
                i,
                "possession_source"
            ] = "ball_missing_too_long"


    # ========================================================
    # Player data not good enough
    # ========================================================

    elif direct == "PlayerQualityLow":

        switch_candidate = None
        switch_count = 0

        in_flight_streak = 0
        missing_streak = 0

        quality_streak += 1


        if (
            current_team is not None

            and

            quality_streak
            <=
            PLAYER_QUALITY_HOLD_FRAMES
        ):

            result.loc[
                i,
                "possession_v3"
            ] = current_team


            result.loc[
                i,
                "possession_source"
            ] = "carried_low_player_quality"


        else:

            result.loc[
                i,
                "possession_v3"
            ] = "Unknown"


            result.loc[
                i,
                "possession_source"
            ] = "player_quality_too_low"


    # ========================================================
    # Unknown
    # ========================================================

    else:

        switch_candidate = None
        switch_count = 0

        in_flight_streak = 0
        missing_streak = 0
        quality_streak = 0


        result.loc[
            i,
            "possession_v3"
        ] = "Unknown"


        result.loc[
            i,
            "possession_source"
        ] = "unknown"


    # Debug
    result.loc[
        i,
        "switch_candidate"
    ] = switch_candidate


    result.loc[
        i,
        "switch_candidate_count"
    ] = switch_count


    result.loc[
        i,
        "in_flight_streak"
    ] = in_flight_streak


# ============================================================
# Save frame CSV
# ============================================================

result.to_csv(
    OUTPUT_FRAME_CSV,
    index=False
)


# ============================================================
# Summary
# ============================================================

total_frames = len(
    result
)


A_frames = int(
    (
        result["possession_v3"]
        ==
        "A"
    ).sum()
)


B_frames = int(
    (
        result["possession_v3"]
        ==
        "B"
    ).sum()
)


contested_frames = int(
    (
        result["possession_v3"]
        ==
        "Contested"
    ).sum()
)


unknown_frames = int(
    (
        result["possession_v3"]
        ==
        "Unknown"
    ).sum()
)


known_team_frames = (
    A_frames
    +
    B_frames
)


if known_team_frames > 0:

    A_share = (
        A_frames
        /
        known_team_frames
        *
        100.0
    )

    B_share = (
        B_frames
        /
        known_team_frames
        *
        100.0
    )

else:

    A_share = 0.0
    B_share = 0.0


resolved_frames = (
    A_frames
    +
    B_frames
    +
    contested_frames
)


resolved_coverage = (
    resolved_frames
    /
    total_frames
    *
    100.0
)


# ============================================================
# Summary CSV
# ============================================================

summary = pd.DataFrame(
    [
        {
            "state": "A",
            "frames": A_frames,
            "seconds": A_frames / FPS,
            "share_known_team_pct":
                A_share,
        },

        {
            "state": "B",
            "frames": B_frames,
            "seconds": B_frames / FPS,
            "share_known_team_pct":
                B_share,
        },

        {
            "state": "Contested",
            "frames": contested_frames,
            "seconds":
                contested_frames
                /
                FPS,
            "share_known_team_pct":
                np.nan,
        },

        {
            "state": "Unknown",
            "frames": unknown_frames,
            "seconds":
                unknown_frames
                /
                FPS,
            "share_known_team_pct":
                np.nan,
        },
    ]
)


summary.to_csv(
    OUTPUT_SUMMARY_CSV,
    index=False
)


# ============================================================
# Diagnostics
# ============================================================

good_player_frames = int(
    result[
        "player_frame_good"
    ].sum()
)


projection_mismatch_frames = int(
    result[
        "projection_mismatch"
    ].sum()
)


# ============================================================
# Print
# ============================================================

print("")
print(
    "========================================"
)

print(
    "POSSESSION V3 SUMMARY"
)

print(
    "========================================"
)

print("")

print(
    "Frame range:",
    min_frame,
    "->",
    max_frame
)

print(
    "Time range:",
    f"{min_frame / FPS:.2f}s",
    "->",
    f"{max_frame / FPS:.2f}s"
)

print(
    "Total frames:",
    total_frames
)

print("")

print(
    "Stable tracks:",
    len(
        stable_track_ids
    )
)

print(
    "Removed team-inconsistent rows:",
    removed_inconsistent_rows
)

print(
    "Frames with >=",
    MIN_PLAYERS_PER_TEAM,
    "players/team:",
    good_player_frames,
    f"({good_player_frames / total_frames * 100:.1f}%)"
)


print("")
print(
    "DIRECT STATE"
)

print(
    result[
        "direct_state"
    ]
    .value_counts()
    .to_string()
)


print("")

print(
    "Pitch-close but foot-far "
    "(possible aerial/projection mismatch):",
    projection_mismatch_frames
)


print("")
print(
    "FINAL POSSESSION"
)

print(
    result[
        "possession_v3"
    ]
    .value_counts()
    .to_string()
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
    contested_frames
)

print(
    "Unknown:",
    unknown_frames
)


print("")
print(
    "Known team possession:"
)

print(
    "A:",
    f"{A_share:.1f}%"
)

print(
    "B:",
    f"{B_share:.1f}%"
)


print("")

print(
    "Resolved coverage:",
    f"{resolved_coverage:.1f}%"
)

print(
    "Confirmed turnovers:",
    confirmed_switches
)


print("")
print(
    "POSSESSION SOURCE"
)

print(
    result[
        "possession_source"
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
