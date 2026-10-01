import pandas as pd
import numpy as np


CSV_PATH = "outputs/tracking_data_clean.csv"

HIGH_SPEED = 12.0
EXTREME_SPEED = 15.0

# 同一 frame 有 >= 3 人超過 12 m/s
# 就值得檢查 camera/homography
MIN_SIMULTANEOUS_PLAYERS = 3


# ============================================================
# Load
# ============================================================

df = pd.read_csv(CSV_PATH)

df = df[
    df["team"].isin(["A", "B"])
].copy()


# 只留下有速度的 rows
df = df[
    np.isfinite(
        df["speed_clean_mps"]
    )
].copy()


# ============================================================
# 每 frame 統計
# ============================================================

frame_stats = (
    df.groupby("frame")
    .agg(
        time_sec=(
            "time_sec",
            "first"
        ),

        players=(
            "track_id",
            "nunique"
        ),

        median_speed=(
            "speed_clean_mps",
            "median"
        ),

        max_speed=(
            "speed_clean_mps",
            "max"
        ),

        mean_reprojection_error_px=(
            "mean_reprojection_error_px",
            "first"
        ),

        median_reprojection_error_px=(
            "median_reprojection_error_px",
            "first"
        ),

        homography_fallback=(
            "homography_fallback",
            "first"
        )
    )
    .reset_index()
)


# ============================================================
# 每 frame 有幾個人高速
# ============================================================

high_counts = (
    df[
        df["speed_clean_mps"] > HIGH_SPEED
    ]
    .groupby("frame")
    ["track_id"]
    .nunique()
)


extreme_counts = (
    df[
        df["speed_clean_mps"] > EXTREME_SPEED
    ]
    .groupby("frame")
    ["track_id"]
    .nunique()
)


frame_stats[
    "players_over_12"
] = (
    frame_stats["frame"]
    .map(high_counts)
    .fillna(0)
    .astype(int)
)


frame_stats[
    "players_over_15"
] = (
    frame_stats["frame"]
    .map(extreme_counts)
    .fillna(0)
    .astype(int)
)


# ============================================================
# 找 suspicious frames
# ============================================================

suspicious = frame_stats[
    frame_stats["players_over_12"]
    >= MIN_SIMULTANEOUS_PLAYERS
].copy()


suspicious = suspicious.sort_values(
    [
        "players_over_12",
        "max_speed"
    ],
    ascending=False
)


print("")
print("========================================")
print("GLOBAL TRAJECTORY ANOMALIES")
print("========================================")
print("")


if len(suspicious) == 0:

    print(
        "No frames with >= "
        f"{MIN_SIMULTANEOUS_PLAYERS} "
        f"players above {HIGH_SPEED} m/s."
    )

else:

    print(
        suspicious[
            [
                "frame",
                "time_sec",
                "players",
                "players_over_12",
                "players_over_15",
                "median_speed",
                "max_speed",
                "mean_reprojection_error_px",
                "median_reprojection_error_px",
                "homography_fallback"
            ]
        ]
        .head(50)
        .to_string(index=False)
    )


# ============================================================
# 找出高速 frame 裡是哪些球員
# ============================================================

print("")
print("========================================")
print("TOP HIGH-SPEED EVENTS")
print("========================================")
print("")


events = df[
    df["speed_clean_mps"] > HIGH_SPEED
][
    [
        "frame",
        "time_sec",
        "track_id",
        "team",
        "speed_clean_mps",
        "mean_reprojection_error_px",
        "homography_fallback"
    ]
].copy()


events = events.sort_values(
    "speed_clean_mps",
    ascending=False
)


print(
    events.head(50)
    .to_string(index=False)
)


# ============================================================
# Global summary
# ============================================================

print("")
print("========================================")
print("SUMMARY")
print("========================================")

print(
    "Frames analysed:",
    len(frame_stats)
)

print(
    "Frames with >=3 players >12 m/s:",
    int(
        (
            frame_stats["players_over_12"]
            >= 3
        ).sum()
    )
)

print(
    "Frames with >=3 players >15 m/s:",
    int(
        (
            frame_stats["players_over_15"]
            >= 3
        ).sum()
    )
)

print(
    "Maximum simultaneous players >12 m/s:",
    frame_stats[
        "players_over_12"
    ].max()
)

print(
    "Maximum simultaneous players >15 m/s:",
    frame_stats[
        "players_over_15"
    ].max()
)
