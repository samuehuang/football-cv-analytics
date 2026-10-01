import os
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

INPUT_CSV = "outputs/ball_tracking_v3.csv"

OUTPUT_CSV = "outputs/ball_tracking_clean.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Parameters
# ============================================================

MAX_INTERPOLATION_GAP = 3

LOW_CONFIDENCE_THRESHOLD = 0.25

HIGH_SPEED_THRESHOLD = 40.0


# ============================================================
# Load
# ============================================================

df = pd.read_csv(INPUT_CSV)

df = (
    df
    .sort_values("frame")
    .reset_index(drop=True)
)


# ============================================================
# Initialize clean columns
# ============================================================

df["ball_valid"] = (
    df["detected"]
    &
    df["projected"]
)


df["interpolated"] = False


df["pitch_x_clean_m"] = (
    df["pitch_x_m"]
)

df["pitch_y_clean_m"] = (
    df["pitch_y_m"]
)


df["image_x_clean"] = (
    df["image_x"]
)

df["image_y_clean"] = (
    df["image_y"]
)


# ============================================================
# Observation quality flags
# ============================================================

df["low_confidence"] = (
    df["confidence"]
    .fillna(0)
    <
    LOW_CONFIDENCE_THRESHOLD
)


df["high_speed_observation"] = (
    df[
        "estimated_speed_from_last_mps"
    ]
    .fillna(0)
    >
    HIGH_SPEED_THRESHOLD
)


# ============================================================
# Short-gap interpolation
# ============================================================

n = len(df)

interpolated_frames = 0

i = 0


while i < n:

    # already has ball
    if bool(df.loc[i, "ball_valid"]):

        i += 1
        continue


    # start missing gap
    gap_start = i


    while (
        i < n
        and
        not bool(
            df.loc[i, "ball_valid"]
        )
    ):

        i += 1


    gap_end = i - 1

    gap_length = (
        gap_end
        -
        gap_start
        +
        1
    )


    previous_index = (
        gap_start - 1
    )

    next_index = i


    # ========================================================
    # Only interpolate:
    #
    # 1. gap <= 3 frames
    # 2. both endpoints exist
    # 3. endpoints belong to same trajectory segment
    # ========================================================

    if (
        gap_length
        <= MAX_INTERPOLATION_GAP
        and
        previous_index >= 0
        and
        next_index < n
        and
        bool(
            df.loc[
                previous_index,
                "ball_valid"
            ]
        )
        and
        bool(
            df.loc[
                next_index,
                "ball_valid"
            ]
        )
    ):

        previous_segment = (
            df.loc[
                previous_index,
                "segment_id"
            ]
        )

        next_segment = (
            df.loc[
                next_index,
                "segment_id"
            ]
        )


        # Do not connect two separate trajectories
        same_segment = (
            pd.notna(previous_segment)
            and
            pd.notna(next_segment)
            and
            previous_segment
            ==
            next_segment
        )


        if same_segment:

            # -----------------------------------------------
            # Endpoints
            # -----------------------------------------------

            x1 = float(
                df.loc[
                    previous_index,
                    "pitch_x_m"
                ]
            )

            y1 = float(
                df.loc[
                    previous_index,
                    "pitch_y_m"
                ]
            )

            x2 = float(
                df.loc[
                    next_index,
                    "pitch_x_m"
                ]
            )

            y2 = float(
                df.loc[
                    next_index,
                    "pitch_y_m"
                ]
            )


            ix1 = float(
                df.loc[
                    previous_index,
                    "image_x"
                ]
            )

            iy1 = float(
                df.loc[
                    previous_index,
                    "image_y"
                ]
            )

            ix2 = float(
                df.loc[
                    next_index,
                    "image_x"
                ]
            )

            iy2 = float(
                df.loc[
                    next_index,
                    "image_y"
                ]
            )


            total_steps = (
                gap_length + 1
            )


            for offset in range(
                1,
                gap_length + 1
            ):

                row_index = (
                    previous_index
                    +
                    offset
                )


                alpha = (
                    offset
                    /
                    total_steps
                )


                df.loc[
                    row_index,
                    "pitch_x_clean_m"
                ] = (
                    x1
                    +
                    alpha
                    *
                    (x2 - x1)
                )


                df.loc[
                    row_index,
                    "pitch_y_clean_m"
                ] = (
                    y1
                    +
                    alpha
                    *
                    (y2 - y1)
                )


                df.loc[
                    row_index,
                    "image_x_clean"
                ] = (
                    ix1
                    +
                    alpha
                    *
                    (ix2 - ix1)
                )


                df.loc[
                    row_index,
                    "image_y_clean"
                ] = (
                    iy1
                    +
                    alpha
                    *
                    (iy2 - iy1)
                )


                df.loc[
                    row_index,
                    "ball_valid"
                ] = True


                df.loc[
                    row_index,
                    "interpolated"
                ] = True


                interpolated_frames += 1


# ============================================================
# Quality label
# ============================================================

def get_quality(row):

    if row["interpolated"]:

        return "interpolated"


    if not row["ball_valid"]:

        return "missing"


    if (
        row["low_confidence"]
        or
        row["high_speed_observation"]
    ):

        return "low"


    return "high"


df["ball_quality"] = (
    df.apply(
        get_quality,
        axis=1
    )
)


# ============================================================
# Save
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# Statistics
# ============================================================

original_valid = int(
    (
        df["detected"]
        &
        df["projected"]
    ).sum()
)


final_valid = int(
    df["ball_valid"].sum()
)


coverage_original = (
    original_valid
    /
    len(df)
    *
    100
)


coverage_final = (
    final_valid
    /
    len(df)
    *
    100
)


# ============================================================
# Remaining missing gaps
# ============================================================

missing = (
    ~df["ball_valid"]
)


remaining_gaps = []

current = 0


for value in missing:

    if value:

        current += 1

    else:

        if current > 0:

            remaining_gaps.append(
                current
            )

            current = 0


if current > 0:

    remaining_gaps.append(
        current
    )


longest_remaining_gap = (
    max(remaining_gaps)
    if remaining_gaps
    else 0
)


# ============================================================
# Report
# ============================================================

print("")
print(
    "========================================"
)

print(
    "BALL TRAJECTORY CLEANING"
)

print(
    "========================================"
)

print("")

print(
    "Original valid observations:",
    original_valid
)

print(
    "Original coverage:",
    f"{coverage_original:.1f}%"
)

print("")

print(
    "Interpolated frames:",
    interpolated_frames
)

print("")

print(
    "Final valid frames:",
    final_valid
)

print(
    "Final coverage:",
    f"{coverage_final:.1f}%"
)

print("")

print(
    "High-quality observed:",
    int(
        (
            df["ball_quality"]
            ==
            "high"
        ).sum()
    )
)

print(
    "Low-quality observed:",
    int(
        (
            df["ball_quality"]
            ==
            "low"
        ).sum()
    )
)

print(
    "Interpolated:",
    int(
        (
            df["ball_quality"]
            ==
            "interpolated"
        ).sum()
    )
)

print(
    "Missing:",
    int(
        (
            df["ball_quality"]
            ==
            "missing"
        ).sum()
    )
)


print("")

print(
    "Longest remaining gap:",
    longest_remaining_gap,
    "frames"
)


print("")
print(
    "Saved:",
    OUTPUT_CSV
)
