import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.signal import savgol_filter


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Two-pass player trajectory cleaning: "
        "per-track smoothing followed by global "
        "homography anomaly correction."
    )
)


parser.add_argument(
    "--input",
    default="outputs/tracking_data_v4.csv",
    help="Input Radar V4 tracking CSV.",
)


parser.add_argument(
    "--output",
    default="outputs/tracking_data_clean_v2.csv",
    help="Final cleaned trajectory CSV.",
)


parser.add_argument(
    "--intermediate-output",
    default=None,
    help=(
        "Optional PASS-1 CSV for debugging/regression. "
        "Normally not required."
    ),
)


parser.add_argument(
    "--fps",
    type=float,
    default=25.0,
)


# ============================================================
# PASS 1 parameters
# ============================================================

parser.add_argument(
    "--min-track-frames",
    type=int,
    default=50,
)


parser.add_argument(
    "--median-window",
    type=int,
    default=5,
)


parser.add_argument(
    "--stage1-sg-window",
    type=int,
    default=11,
)


parser.add_argument(
    "--sg-polyorder",
    type=int,
    default=2,
)


parser.add_argument(
    "--stage1-velocity-half-window",
    type=int,
    default=2,
)


parser.add_argument(
    "--max-reasonable-speed",
    type=float,
    default=12.0,
)


# ============================================================
# PASS 2 parameters
# ============================================================

parser.add_argument(
    "--global-high-speed-threshold",
    type=float,
    default=12.0,
)


parser.add_argument(
    "--min-simultaneous-players",
    type=int,
    default=3,
)


parser.add_argument(
    "--expand-bad-frame",
    type=int,
    default=2,
)


parser.add_argument(
    "--stage2-sg-window",
    type=int,
    default=9,
)


parser.add_argument(
    "--stage2-velocity-half-window",
    type=int,
    default=4,
)


args = parser.parse_args()


INPUT_CSV = args.input

OUTPUT_CSV = args.output

INTERMEDIATE_OUTPUT = (
    args.intermediate_output
)


FPS = args.fps


MIN_TRACK_FRAMES = (
    args.min_track_frames
)


MEDIAN_WINDOW = (
    args.median_window
)


STAGE1_SG_WINDOW = (
    args.stage1_sg_window
)


SG_POLYORDER = (
    args.sg_polyorder
)


STAGE1_VELOCITY_HALF_WINDOW = (
    args.stage1_velocity_half_window
)


MAX_REASONABLE_SPEED = (
    args.max_reasonable_speed
)


HIGH_SPEED_THRESHOLD = (
    args.global_high_speed_threshold
)


MIN_SIMULTANEOUS_PLAYERS = (
    args.min_simultaneous_players
)


EXPAND_BAD_FRAME = (
    args.expand_bad_frame
)


STAGE2_SG_WINDOW = (
    args.stage2_sg_window
)


STAGE2_VELOCITY_HALF_WINDOW = (
    args.stage2_velocity_half_window
)


# ============================================================
# Validation
# ============================================================

if FPS <= 0:

    raise ValueError(
        "FPS must be > 0."
    )


for value, name in [

    (
        MEDIAN_WINDOW,
        "median-window",
    ),

    (
        STAGE1_SG_WINDOW,
        "stage1-sg-window",
    ),

    (
        STAGE2_SG_WINDOW,
        "stage2-sg-window",
    ),

]:

    if value < 1:

        raise ValueError(
            f"{name} must be >= 1"
        )


for value, name in [

    (
        STAGE1_SG_WINDOW,
        "stage1-sg-window",
    ),

    (
        STAGE2_SG_WINDOW,
        "stage2-sg-window",
    ),

]:

    if value % 2 == 0:

        raise ValueError(
            f"{name} must be odd."
        )


input_path = Path(
    INPUT_CSV
)


if not input_path.is_file():

    raise FileNotFoundError(
        f"Input CSV not found: "
        f"{INPUT_CSV}"
    )


Path(
    OUTPUT_CSV
).parent.mkdir(
    parents=True,
    exist_ok=True,
)


if (
    INTERMEDIATE_OUTPUT
    is not None
):

    Path(
        INTERMEDIATE_OUTPUT
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Helpers
# ============================================================

def calculate_speed(
    track,
    x,
    y,
    fps,
    half_window,
    max_gap_extra,
):

    speed = np.full(
        len(track),
        np.nan,
    )


    k = half_window


    for i in range(
        k,
        len(track) - k,
    ):

        frame_before = (
            track.loc[
                i - k,
                "frame",
            ]
        )


        frame_after = (
            track.loc[
                i + k,
                "frame",
            ]
        )


        frame_gap = (
            frame_after
            -
            frame_before
        )


        max_allowed_gap = (
            2 * k
            +
            max_gap_extra
        )


        if (
            frame_gap
            >
            max_allowed_gap
        ):

            continue


        dx = (
            x[i + k]
            -
            x[i - k]
        )


        dy = (
            y[i + k]
            -
            y[i - k]
        )


        distance = np.sqrt(
            dx ** 2
            +
            dy ** 2
        )


        dt = (
            frame_gap
            /
            fps
        )


        if dt > 0:

            speed[i] = (
                distance
                /
                dt
            )


    return speed


def contiguous_ranges(
    frames,
):

    frames = sorted(
        frames
    )


    if len(frames) == 0:

        return []


    groups = []


    start = frames[0]

    previous = frames[0]


    for frame in frames[1:]:

        if (
            frame
            ==
            previous + 1
        ):

            previous = frame

        else:

            groups.append(
                (
                    start,
                    previous,
                )
            )

            start = frame

            previous = frame


    groups.append(
        (
            start,
            previous,
        )
    )


    return groups


# ============================================================
# Load raw Radar V4 tracking
# ============================================================

print("")

print(
    "=" * 58
)

print(
    "TRAJECTORY CLEANING"
)

print(
    "=" * 58
)


print(
    f"Input:  {INPUT_CSV}"
)

print(
    f"Output: {OUTPUT_CSV}"
)

print(
    f"FPS:    {FPS:.2f}"
)


df = pd.read_csv(
    INPUT_CSV
)


required_columns = {

    "frame",

    "track_id",

    "team",

    "pitch_x_m",

    "pitch_y_m",

}


missing_columns = (
    required_columns
    -
    set(df.columns)
)


if missing_columns:

    raise ValueError(
        "Missing required columns: "
        +
        ", ".join(
            sorted(
                missing_columns
            )
        )
    )


# ============================================================
# Keep actual outfield teams
# ============================================================

df = df[
    df[
        "team"
    ].isin(
        [
            "A",
            "B",
        ]
    )
].copy()


df = df.sort_values(
    [
        "track_id",
        "frame",
    ]
)


print("")

print(
    "Raw rows:",
    len(df),
)


print(
    "Raw tracks:",
    df[
        "track_id"
    ].nunique(),
)


# ============================================================
# PASS 1
#
# Per-player trajectory cleaning
# ============================================================

print("")

print(
    "=" * 58
)

print(
    "PASS 1 — PER-TRACK CLEANING"
)

print(
    "=" * 58
)


# ============================================================
# Remove short tracks
# ============================================================

track_lengths = (

    df

    .groupby(
        "track_id"
    )

    .size()

)


valid_ids = (

    track_lengths[
        track_lengths
        >=
        MIN_TRACK_FRAMES
    ]

    .index

)


stage1_df = df[
    df[
        "track_id"
    ].isin(
        valid_ids
    )
].copy()


print(
    "Tracks after minimum-length filter:",
    stage1_df[
        "track_id"
    ].nunique(),
)


clean_tracks = []


# ============================================================
# Per-track processing
# ============================================================

for (
    track_id,
    track,
) in stage1_df.groupby(
    "track_id"
):

    track = (

        track

        .sort_values(
            "frame"
        )

        .copy()

        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # Raw coordinates
    # ========================================================

    x = (

        track[
            "pitch_x_m"
        ]

        .to_numpy(
            dtype=float
        )
    )


    y = (

        track[
            "pitch_y_m"
        ]

        .to_numpy(
            dtype=float
        )
    )


    # ========================================================
    # Rolling median
    # ========================================================

    x_median = (

        pd.Series(
            x
        )

        .rolling(
            window=(
                MEDIAN_WINDOW
            ),
            center=True,
            min_periods=1,
        )

        .median()

        .to_numpy()

    )


    y_median = (

        pd.Series(
            y
        )

        .rolling(
            window=(
                MEDIAN_WINDOW
            ),
            center=True,
            min_periods=1,
        )

        .median()

        .to_numpy()

    )


    # ========================================================
    # Savitzky-Golay smoothing
    # ========================================================

    if (
        len(track)
        >=
        STAGE1_SG_WINDOW
    ):

        x_smooth = savgol_filter(

            x_median,

            window_length=(
                STAGE1_SG_WINDOW
            ),

            polyorder=(
                SG_POLYORDER
            ),
        )


        y_smooth = savgol_filter(

            y_median,

            window_length=(
                STAGE1_SG_WINDOW
            ),

            polyorder=(
                SG_POLYORDER
            ),
        )


    else:

        x_smooth = (
            x_median
        )

        y_smooth = (
            y_median
        )


    track[
        "pitch_x_clean_m"
    ] = (
        x_smooth
    )


    track[
        "pitch_y_clean_m"
    ] = (
        y_smooth
    )


    # ========================================================
    # Initial velocity
    #
    # Original V1 behaviour:
    # max gap = 2*k + 2
    # ========================================================

    speed = calculate_speed(

        track=track,

        x=x_smooth,

        y=y_smooth,

        fps=FPS,

        half_window=(
            STAGE1_VELOCITY_HALF_WINDOW
        ),

        max_gap_extra=2,

    )


    track[
        "speed_clean_mps"
    ] = speed


    track[
        "speed_valid"
    ] = (

        track[
            "speed_clean_mps"
        ].isna()

        |

        (
            track[
                "speed_clean_mps"
            ]
            <=
            MAX_REASONABLE_SPEED
        )

    )


    clean_tracks.append(
        track
    )


if len(clean_tracks) == 0:

    raise RuntimeError(
        "No tracks survived PASS 1."
    )


stage1_df = pd.concat(

    clean_tracks,

    ignore_index=True,

)


print(
    "PASS 1 rows:",
    len(stage1_df),
)


print(
    "PASS 1 tracks:",
    stage1_df[
        "track_id"
    ].nunique(),
)


# ============================================================
# Optional intermediate output
# ============================================================

if (
    INTERMEDIATE_OUTPUT
    is not None
):

    stage1_df.to_csv(

        INTERMEDIATE_OUTPUT,

        index=False,

    )


    print(
        "PASS 1 debug CSV:",
        INTERMEDIATE_OUTPUT,
    )


# ============================================================
# PASS 1 Quality Summary
# ============================================================

pass1_speeds = (
    stage1_df[
        "speed_clean_mps"
    ]
    .dropna()
)


if len(pass1_speeds) > 0:

    print("")

    print(
        "PASS 1 speed:"
    )

    print(
        f"  median: "
        f"{pass1_speeds.median():.3f} m/s"
    )

    print(
        f"  p95:    "
        f"{pass1_speeds.quantile(0.95):.3f} m/s"
    )

    print(
        f"  max:    "
        f"{pass1_speeds.max():.3f} m/s"
    )


# ============================================================
# PASS 2
#
# Global homography anomaly removal
# ============================================================

print("")

print(
    "=" * 58
)

print(
    "PASS 2 — GLOBAL HOMOGRAPHY CORRECTION"
)

print(
    "=" * 58
)


df2 = (

    stage1_df[
        stage1_df[
            "team"
        ].isin(
            [
                "A",
                "B",
            ]
        )
    ]

    .copy()

    .sort_values(
        [
            "track_id",
            "frame",
        ]
    )

    .reset_index(
        drop=True
    )

)


# ============================================================
# Detect simultaneous high-speed anomalies
# ============================================================

high_speed_rows = df2[

    df2[
        "speed_clean_mps"
    ]

    >
    HIGH_SPEED_THRESHOLD

]


high_count = (

    high_speed_rows

    .groupby(
        "frame"
    )[
        "track_id"
    ]

    .nunique()

)


bad_frames = (

    high_count[
        high_count
        >=
        MIN_SIMULTANEOUS_PLAYERS
    ]

    .index

    .tolist()

)


print("")

print(
    "Detected global anomaly frames:"
)

print(
    bad_frames
)


# ============================================================
# Expand affected frames
# ============================================================

expanded_bad_frames = set()


for frame in bad_frames:

    for offset in range(

        -EXPAND_BAD_FRAME,

        EXPAND_BAD_FRAME + 1,

    ):

        expanded_bad_frames.add(
            frame + offset
        )


expanded_bad_frames = {

    frame

    for frame
    in expanded_bad_frames

    if frame >= 0

}


bad_ranges = contiguous_ranges(
    expanded_bad_frames
)


print(
    "Bad frames after expansion:",
    len(
        expanded_bad_frames
    ),
)


print(
    "Bad frame ranges:"
)


if len(
    bad_ranges
) == 0:

    print(
        "  None"
    )


else:

    for (
        start,
        end,
    ) in bad_ranges:

        print(

            f"  frame "
            f"{start:03d}-{end:03d} "

            f"("
            f"{start / FPS:.2f}s"
            f" - "
            f"{end / FPS:.2f}s"
            f")"

        )


# ============================================================
# Correct every track
# ============================================================

processed_tracks = []


for (
    track_id,
    track,
) in df2.groupby(
    "track_id"
):

    track = (

        track

        .sort_values(
            "frame"
        )

        .copy()

        .reset_index(
            drop=True
        )
    )


    frames = (

        track[
            "frame"
        ]

        .to_numpy(
            dtype=float
        )
    )


    x = (

        track[
            "pitch_x_clean_m"
        ]

        .to_numpy(
            dtype=float
        )
    )


    y = (

        track[
            "pitch_y_clean_m"
        ]

        .to_numpy(
            dtype=float
        )
    )


    is_bad = (

        track[
            "frame"
        ]

        .isin(
            expanded_bad_frames
        )

        .to_numpy()

    )


    good = (

        (~is_bad)

        &

        np.isfinite(
            x
        )

        &

        np.isfinite(
            y
        )

    )


    # ========================================================
    # Interpolate only bad frames
    # ========================================================

    if (
        good.sum()
        >=
        2
    ):

        x_interp = (
            x.copy()
        )

        y_interp = (
            y.copy()
        )


        x_estimated = np.interp(

            frames,

            frames[
                good
            ],

            x[
                good
            ],

        )


        y_estimated = np.interp(

            frames,

            frames[
                good
            ],

            y[
                good
            ],

        )


        x_interp[
            is_bad
        ] = (

            x_estimated[
                is_bad
            ]

        )


        y_interp[
            is_bad
        ] = (

            y_estimated[
                is_bad
            ]

        )


    else:

        x_interp = (
            x.copy()
        )

        y_interp = (
            y.copy()
        )


    # ========================================================
    # Light second smoothing
    # ========================================================

    if (
        len(track)
        >=
        STAGE2_SG_WINDOW
    ):

        x_final = savgol_filter(

            x_interp,

            window_length=(
                STAGE2_SG_WINDOW
            ),

            polyorder=(
                SG_POLYORDER
            ),

        )


        y_final = savgol_filter(

            y_interp,

            window_length=(
                STAGE2_SG_WINDOW
            ),

            polyorder=(
                SG_POLYORDER
            ),

        )


    else:

        x_final = (
            x_interp
        )

        y_final = (
            y_interp
        )


    track[
        "pitch_x_final_m"
    ] = (
        x_final
    )


    track[
        "pitch_y_final_m"
    ] = (
        y_final
    )


    track[
        "global_anomaly_frame"
    ] = (
        is_bad
    )


    # ========================================================
    # Final velocity
    #
    # Original V2 behaviour:
    # max gap = 2*k + 3
    # ========================================================

    final_speed = calculate_speed(

        track=track,

        x=x_final,

        y=y_final,

        fps=FPS,

        half_window=(
            STAGE2_VELOCITY_HALF_WINDOW
        ),

        max_gap_extra=3,

    )


    track[
        "speed_final_mps"
    ] = (
        final_speed
    )


    processed_tracks.append(
        track
    )


if len(
    processed_tracks
) == 0:

    raise RuntimeError(
        "No tracks survived PASS 2."
    )


final_df = pd.concat(

    processed_tracks,

    ignore_index=True,

)


# ============================================================
# Save final
# ============================================================

final_df.to_csv(

    OUTPUT_CSV,

    index=False,

)


# ============================================================
# Final Quality Report
# ============================================================

results = []


for (
    track_id,
    track,
) in final_df.groupby(
    "track_id"
):

    speeds = (

        track[
            "speed_final_mps"
        ]

        .dropna()

    )


    if len(speeds) == 0:

        continue


    results.append(
        {
            "track_id":
                track_id,

            "team":
                track[
                    "team"
                ].mode()[0],

            "frames":
                len(track),

            "median_speed":
                speeds.median(),

            "p95_speed":
                speeds.quantile(
                    0.95
                ),

            "max_speed":
                speeds.max(),

            "over_12":
                int(
                    (
                        speeds > 12
                    ).sum()
                ),

            "over_15":
                int(
                    (
                        speeds > 15
                    ).sum()
                ),
        }
    )


summary = pd.DataFrame(
    results
)


if len(summary) > 0:

    summary = (
        summary
        .sort_values(
            "max_speed",
            ascending=False,
        )
    )


print("")

print(
    "=" * 58
)

print(
    "FINAL TRAJECTORY QUALITY"
)

print(
    "=" * 58
)


if len(summary) > 0:

    print(
        summary.to_string(
            index=False
        )
    )


else:

    print(
        "No valid speed summary."
    )


# ============================================================
# Global anomaly test after correction
# ============================================================

high_final = final_df[

    final_df[
        "speed_final_mps"
    ]

    >
    HIGH_SPEED_THRESHOLD

]


counts_final = (

    high_final

    .groupby(
        "frame"
    )[
        "track_id"
    ]

    .nunique()

)


simultaneous_final = (

    counts_final[
        counts_final
        >=
        MIN_SIMULTANEOUS_PLAYERS
    ]

)


print("")

print(
    "=" * 58
)

print(
    "GLOBAL CHECK AFTER CORRECTION"
)

print(
    "=" * 58
)


print(

    "Frames with >="

    f"{MIN_SIMULTANEOUS_PLAYERS} "

    "players >"

    f"{HIGH_SPEED_THRESHOLD:g} m/s:",

    len(
        simultaneous_final
    ),

)


if (
    len(
        simultaneous_final
    )
    >
    0
):

    print("")

    print(

        simultaneous_final

        .sort_values(
            ascending=False
        )

        .head(
            20
        )

    )


print("")

print(
    "=" * 58
)

print(
    "DONE"
)

print(
    "=" * 58
)


print(
    f"Rows:   "
    f"{len(final_df)}"
)


print(
    f"Tracks: "
    f"{final_df['track_id'].nunique()}"
)


print(
    f"Saved:  "
    f"{OUTPUT_CSV}"
)
