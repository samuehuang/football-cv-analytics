import os
import numpy as np
import pandas as pd

from scipy.signal import savgol_filter


INPUT_CSV = "outputs/tracking_data_v4.csv"
OUTPUT_CSV = "outputs/tracking_data_clean.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Parameters
# ============================================================

# 影片約 25 FPS
FPS = 25.0

# 至少存在 2 秒才拿來分析
MIN_TRACK_FRAMES = 50

# Savitzky-Golay
# 11 frames ≈ 0.44 sec
SG_WINDOW = 11
SG_POLYORDER = 2

# 不再用 adjacent-frame velocity
# 用前後各 2 frame：
#
# t-2 ---------------- t+2
#
# 時間跨度 = 4 / 25 = 0.16 sec
VELOCITY_HALF_WINDOW = 2

# 物理 sanity threshold
MAX_REASONABLE_SPEED = 12.0


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
print("Raw rows:", len(df))
print(
    "Raw tracks:",
    df["track_id"].nunique()
)


# ============================================================
# Remove short tracks
# ============================================================

track_lengths = (
    df.groupby("track_id")
    .size()
)

valid_ids = track_lengths[
    track_lengths >= MIN_TRACK_FRAMES
].index


df = df[
    df["track_id"].isin(valid_ids)
].copy()


print("")
print(
    "Tracks after minimum-length filter:",
    df["track_id"].nunique()
)


# ============================================================
# Process each track
# ============================================================

clean_tracks = []


for track_id, track in df.groupby("track_id"):

    track = (
        track
        .sort_values("frame")
        .copy()
        .reset_index(drop=True)
    )


    # ========================================================
    # Raw coordinates
    # ========================================================

    x = (
        track["pitch_x_m"]
        .to_numpy(dtype=float)
    )

    y = (
        track["pitch_y_m"]
        .to_numpy(dtype=float)
    )


    # ========================================================
    # Step 1:
    # Rolling median
    #
    # 移除單幀 spike
    # ========================================================

    x_median = (
        pd.Series(x)
        .rolling(
            window=5,
            center=True,
            min_periods=1
        )
        .median()
        .to_numpy()
    )


    y_median = (
        pd.Series(y)
        .rolling(
            window=5,
            center=True,
            min_periods=1
        )
        .median()
        .to_numpy()
    )


    # ========================================================
    # Step 2:
    # Savitzky-Golay smoothing
    # ========================================================

    if len(track) >= SG_WINDOW:

        x_smooth = savgol_filter(
            x_median,
            window_length=SG_WINDOW,
            polyorder=SG_POLYORDER
        )

        y_smooth = savgol_filter(
            y_median,
            window_length=SG_WINDOW,
            polyorder=SG_POLYORDER
        )

    else:

        x_smooth = x_median

        y_smooth = y_median


    track[
        "pitch_x_clean_m"
    ] = x_smooth

    track[
        "pitch_y_clean_m"
    ] = y_smooth


    # ========================================================
    # Step 3:
    # Velocity using larger temporal baseline
    #
    # central difference:
    #
    # position[t+2] - position[t-2]
    # -----------------------------
    #            Δt
    # ========================================================

    speed = np.full(
        len(track),
        np.nan
    )


    k = VELOCITY_HALF_WINDOW


    for i in range(
        k,
        len(track) - k
    ):

        frame_before = (
            track.loc[
                i - k,
                "frame"
            ]
        )

        frame_after = (
            track.loc[
                i + k,
                "frame"
            ]
        )


        frame_gap = (
            frame_after
            - frame_before
        )


        # Track 有中斷，不跨 gap 算速度
        if frame_gap > 2 * k + 2:
            continue


        dx = (
            x_smooth[i + k]
            - x_smooth[i - k]
        )

        dy = (
            y_smooth[i + k]
            - y_smooth[i - k]
        )


        distance = np.sqrt(
            dx ** 2
            + dy ** 2
        )


        dt = (
            frame_gap
            / FPS
        )


        if dt > 0:

            speed[i] = (
                distance / dt
            )


    track[
        "speed_clean_mps"
    ] = speed


    # ========================================================
    # Speed quality flag
    # ========================================================

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
            <= MAX_REASONABLE_SPEED
        )

    )


    clean_tracks.append(
        track
    )


# ============================================================
# Merge
# ============================================================

clean_df = pd.concat(
    clean_tracks,
    ignore_index=True
)


clean_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# Quality report
# ============================================================

print("")
print(
    "========================================"
)

print(
    "CLEANED TRAJECTORY QUALITY"
)

print(
    "========================================"
)


results = []


for track_id, track in clean_df.groupby(
    "track_id"
):

    speeds = (
        track[
            "speed_clean_mps"
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

            "over_12_mps":
                int(
                    (
                        speeds > 12
                    ).sum()
                ),

            "over_15_mps":
                int(
                    (
                        speeds > 15
                    ).sum()
                )

        }
    )


summary = pd.DataFrame(
    results
)


summary = summary.sort_values(
    "max_speed",
    ascending=False
)


print(
    summary.to_string(
        index=False
    )
)


print("")
print(
    "Saved:",
    OUTPUT_CSV
)
