import os
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

INPUT_CSV = "outputs/tracking_data_clean_v2.csv"
OUTPUT_CSV = "outputs/player_metrics.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Parameters
# ============================================================

# 每 5 frames 取一個 sample
# 25 FPS -> 約 0.2 秒
SAMPLE_EVERY_N_FRAMES = 5

# 超過這個速度的 segment 不拿來累積 distance
MAX_SEGMENT_SPEED = 12.0

# 至少追蹤兩秒
MIN_TRACK_FRAMES = 50


# ============================================================
# Load
# ============================================================

df = pd.read_csv(INPUT_CSV)

df = df[
    df["team"].isin(["A", "B"])
].copy()

df = df.sort_values(
    ["track_id", "frame"]
)


print("")
print("========================================")
print("PLAYER METRICS")
print("========================================")

print("")
print("Rows:", len(df))
print(
    "Tracks:",
    df["track_id"].nunique()
)


# ============================================================
# Process each player
# ============================================================

metrics = []


for track_id, track in df.groupby("track_id"):

    track = (
        track
        .sort_values("frame")
        .copy()
        .reset_index(drop=True)
    )


    if len(track) < MIN_TRACK_FRAMES:
        continue


    # ========================================================
    # Basic information
    # ========================================================

    team = track["team"].mode()[0]

    start_time = float(
        track["time_sec"].min()
    )

    end_time = float(
        track["time_sec"].max()
    )

    tracked_time = (
        end_time - start_time
    )


    # ========================================================
    # Average position
    # ========================================================

    avg_x = float(
        track[
            "pitch_x_final_m"
        ].median()
    )

    avg_y = float(
        track[
            "pitch_y_final_m"
        ].median()
    )


    # ========================================================
    # Downsample trajectory
    #
    # 不直接用 25 FPS 每 frame 加 distance
    # ========================================================

    sampled = (
        track.iloc[
            ::SAMPLE_EVERY_N_FRAMES
        ]
        .copy()
        .reset_index(drop=True)
    )


    # 確保最後一個位置有被加入
    if (
        sampled.iloc[-1]["frame"]
        != track.iloc[-1]["frame"]
    ):

        sampled = pd.concat(
            [
                sampled,
                track.iloc[[-1]]
            ],
            ignore_index=True
        )


    # ========================================================
    # Segment displacement
    # ========================================================

    sampled["dx"] = (
        sampled[
            "pitch_x_final_m"
        ].diff()
    )

    sampled["dy"] = (
        sampled[
            "pitch_y_final_m"
        ].diff()
    )


    sampled[
        "segment_distance_m"
    ] = np.sqrt(
        sampled["dx"] ** 2
        +
        sampled["dy"] ** 2
    )


    sampled["dt"] = (
        sampled["time_sec"].diff()
    )


    sampled[
        "segment_speed_mps"
    ] = (
        sampled[
            "segment_distance_m"
        ]
        /
        sampled["dt"]
    )


    # ========================================================
    # Reject bad segments
    # ========================================================

    valid_segment = (

        np.isfinite(
            sampled[
                "segment_speed_mps"
            ]
        )

        &

        (
            sampled[
                "segment_speed_mps"
            ]
            <= MAX_SEGMENT_SPEED
        )

        &

        (
            sampled["dt"] > 0
        )

        # 防止 track 中間消失太久，
        # 卻直接連成一條長距離
        &

        (
            sampled["dt"] <= 0.40
        )
    )


    valid_distances = sampled.loc[
        valid_segment,
        "segment_distance_m"
    ]


    estimated_distance = float(
        valid_distances.sum()
    )


    rejected_segments = int(
        (
            sampled[
                "segment_speed_mps"
            ].notna()
            &
            ~valid_segment
        ).sum()
    )


    total_segments = int(
        sampled[
            "segment_speed_mps"
        ].notna().sum()
    )


    # ========================================================
    # Speed Metrics
    #
    # 使用 trajectory_cleaning_v2 已經算好的 speed_final_mps
    # ========================================================

    speeds = (
        track[
            "speed_final_mps"
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )


    # 只留物理合理速度
    valid_speeds = speeds[
        speeds <= MAX_SEGMENT_SPEED
    ]


    if len(valid_speeds) > 0:

        median_speed = float(
            valid_speeds.median()
        )

        mean_speed = float(
            valid_speeds.mean()
        )

        p95_speed = float(
            valid_speeds.quantile(
                0.95
            )
        )

        max_filtered_speed = float(
            valid_speeds.max()
        )

    else:

        median_speed = np.nan
        mean_speed = np.nan
        p95_speed = np.nan
        max_filtered_speed = np.nan


    # ========================================================
    # Coverage / data quality
    # ========================================================

    if total_segments > 0:

        valid_segment_ratio = (
            (
                total_segments
                - rejected_segments
            )
            /
            total_segments
        )

    else:

        valid_segment_ratio = 0.0


    # ========================================================
    # Save
    # ========================================================

    metrics.append(
        {
            "track_id":
                int(track_id),

            "team":
                team,

            "frames":
                len(track),

            "tracked_time_sec":
                tracked_time,

            "estimated_distance_m":
                estimated_distance,

            "mean_speed_mps":
                mean_speed,

            "median_speed_mps":
                median_speed,

            "p95_speed_mps":
                p95_speed,

            "max_filtered_speed_mps":
                max_filtered_speed,

            "avg_x_m":
                avg_x,

            "avg_y_m":
                avg_y,

            "valid_segment_ratio":
                valid_segment_ratio,

            "rejected_segments":
                rejected_segments
        }
    )


# ============================================================
# DataFrame
# ============================================================

metrics_df = pd.DataFrame(
    metrics
)


metrics_df = metrics_df.sort_values(
    [
        "team",
        "estimated_distance_m"
    ],
    ascending=[
        True,
        False
    ]
)


# ============================================================
# Save
# ============================================================

metrics_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# Pretty output
# ============================================================

display_df = metrics_df.copy()


round_columns = [

    "tracked_time_sec",

    "estimated_distance_m",

    "mean_speed_mps",

    "median_speed_mps",

    "p95_speed_mps",

    "max_filtered_speed_mps",

    "avg_x_m",

    "avg_y_m",

    "valid_segment_ratio"
]


for col in round_columns:

    display_df[col] = (
        display_df[col]
        .round(2)
    )


print("")
print(
    display_df[
        [
            "track_id",
            "team",
            "frames",
            "tracked_time_sec",
            "estimated_distance_m",
            "mean_speed_mps",
            "p95_speed_mps",
            "max_filtered_speed_mps",
            "avg_x_m",
            "avg_y_m",
            "valid_segment_ratio",
            "rejected_segments"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Team summary
# ============================================================

print("")
print("========================================")
print("TEAM SUMMARY")
print("========================================")
print("")


for team_name in ["A", "B"]:

    team_df = metrics_df[
        metrics_df["team"] == team_name
    ]

    if len(team_df) == 0:
        continue


    print(
        f"Team {team_name}"
    )

    print(
        "Players:",
        len(team_df)
    )

    print(
        "Total estimated distance:",
        round(
            team_df[
                "estimated_distance_m"
            ].sum(),
            1
        ),
        "m"
    )

    print(
        "Mean player distance:",
        round(
            team_df[
                "estimated_distance_m"
            ].mean(),
            1
        ),
        "m"
    )

    print(
        "Mean P95 speed:",
        round(
            team_df[
                "p95_speed_mps"
            ].mean(),
            2
        ),
        "m/s"
    )

    print("")


print(
    "Saved:",
    OUTPUT_CSV
)
