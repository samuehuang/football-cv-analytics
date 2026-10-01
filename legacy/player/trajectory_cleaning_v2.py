import os
import numpy as np
import pandas as pd

from scipy.signal import savgol_filter


# ============================================================
# Paths
# ============================================================

INPUT_CSV = "outputs/tracking_data_clean.csv"
OUTPUT_CSV = "outputs/tracking_data_clean_v2.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Parameters
# ============================================================

FPS = 25.0

# Global anomaly definition
HIGH_SPEED_THRESHOLD = 12.0
MIN_SIMULTANEOUS_PLAYERS = 3

# 因為 velocity 是 central difference，
# 一個 Homography spike 會影響前後幾幀
EXPAND_BAD_FRAME = 2

# 第二次輕微 smoothing
SG_WINDOW = 9
SG_POLYORDER = 2

# 用比較長的 temporal baseline 算速度
# t-4 → t+4 = 8 frames = 0.32 sec
VELOCITY_HALF_WINDOW = 4


# ============================================================
# Load
# ============================================================

df = pd.read_csv(INPUT_CSV)

df = df[
    df["team"].isin(["A", "B"])
].copy()

df = df.sort_values(
    ["track_id", "frame"]
).reset_index(drop=True)


print("")
print("========================================")
print("GLOBAL HOMOGRAPHY ANOMALY REMOVAL")
print("========================================")


# ============================================================
# 1. 找出同一 frame 多人高速的 frame
# ============================================================

high_speed_rows = df[
    df["speed_clean_mps"] > HIGH_SPEED_THRESHOLD
]


high_count = (
    high_speed_rows
    .groupby("frame")
    ["track_id"]
    .nunique()
)


bad_frames = (
    high_count[
        high_count >= MIN_SIMULTANEOUS_PLAYERS
    ]
    .index
    .tolist()
)


print("")
print("Detected bad frames:")
print(bad_frames)


# ============================================================
# 2. Expand bad frames
#
# velocity central difference 會把影響擴散到鄰近 frame
# ============================================================

expanded_bad_frames = set()


for frame in bad_frames:

    for offset in range(
        -EXPAND_BAD_FRAME,
        EXPAND_BAD_FRAME + 1
    ):

        expanded_bad_frames.add(
            frame + offset
        )


expanded_bad_frames = {
    f
    for f in expanded_bad_frames
    if f >= 0
}


print("")
print(
    "Bad frames after expansion:",
    len(expanded_bad_frames)
)


# ============================================================
# 顯示 bad frame 區段
# ============================================================

sorted_bad = sorted(
    expanded_bad_frames
)


groups = []

if len(sorted_bad) > 0:

    start = sorted_bad[0]
    previous = sorted_bad[0]

    for frame in sorted_bad[1:]:

        if frame == previous + 1:

            previous = frame

        else:

            groups.append(
                (start, previous)
            )

            start = frame
            previous = frame

    groups.append(
        (start, previous)
    )


print("")
print("Bad frame ranges:")


for start, end in groups:

    print(
        f"frame {start:03d}-{end:03d} "
        f"({start / FPS:.2f}s - {end / FPS:.2f}s)"
    )


# ============================================================
# 3. 每個 track 對 global bad frames 做 interpolation
# ============================================================

processed_tracks = []


for track_id, track in df.groupby("track_id"):

    track = (
        track
        .sort_values("frame")
        .copy()
        .reset_index(drop=True)
    )


    frames = (
        track["frame"]
        .to_numpy(dtype=float)
    )


    x = (
        track["pitch_x_clean_m"]
        .to_numpy(dtype=float)
    )

    y = (
        track["pitch_y_clean_m"]
        .to_numpy(dtype=float)
    )


    is_bad = (
        track["frame"]
        .isin(expanded_bad_frames)
        .to_numpy()
    )


    # ================================================
    # good points
    # ================================================

    good = (
        (~is_bad)
        &
        np.isfinite(x)
        &
        np.isfinite(y)
    )


    # 至少需要兩個正常點才能 interpolation
    if good.sum() >= 2:

        x_interp = x.copy()
        y_interp = y.copy()


        # np.interp 使用實際 frame number
        x_estimated = np.interp(
            frames,
            frames[good],
            x[good]
        )

        y_estimated = np.interp(
            frames,
            frames[good],
            y[good]
        )


        # 只替換 global bad frames
        x_interp[is_bad] = (
            x_estimated[is_bad]
        )

        y_interp[is_bad] = (
            y_estimated[is_bad]
        )

    else:

        x_interp = x.copy()
        y_interp = y.copy()


    # ========================================================
    # 4. Light temporal smoothing
    # ========================================================

    if len(track) >= SG_WINDOW:

        x_final = savgol_filter(
            x_interp,
            window_length=SG_WINDOW,
            polyorder=SG_POLYORDER
        )

        y_final = savgol_filter(
            y_interp,
            window_length=SG_WINDOW,
            polyorder=SG_POLYORDER
        )

    else:

        x_final = x_interp
        y_final = y_interp


    track[
        "pitch_x_final_m"
    ] = x_final

    track[
        "pitch_y_final_m"
    ] = y_final


    track[
        "global_anomaly_frame"
    ] = is_bad


    # ========================================================
    # 5. Recalculate velocity
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


        before_frame = (
            track.loc[
                i - k,
                "frame"
            ]
        )

        after_frame = (
            track.loc[
                i + k,
                "frame"
            ]
        )


        frame_gap = (
            after_frame
            - before_frame
        )


        # 正常情況應該 8 frames
        # 如果 track 中間消失太久就不算
        if frame_gap > (
            2 * k + 3
        ):

            continue


        dx = (
            x_final[i + k]
            - x_final[i - k]
        )

        dy = (
            y_final[i + k]
            - y_final[i - k]
        )


        displacement = np.sqrt(
            dx ** 2
            + dy ** 2
        )


        dt = (
            frame_gap / FPS
        )


        if dt > 0:

            speed[i] = (
                displacement / dt
            )


    track[
        "speed_final_mps"
    ] = speed


    processed_tracks.append(
        track
    )


# ============================================================
# Merge
# ============================================================

final_df = pd.concat(
    processed_tracks,
    ignore_index=True
)


final_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# 6. Final quality report
# ============================================================

results = []


for track_id, track in final_df.groupby(
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
                track["team"].mode()[0],

            "frames":
                len(track),

            "median_speed":
                speeds.median(),

            "p95_speed":
                speeds.quantile(0.95),

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


print("")
print("========================================")
print("FINAL TRAJECTORY QUALITY")
print("========================================")
print("")


print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# Global anomaly test again
# ============================================================

high_final = final_df[
    final_df[
        "speed_final_mps"
    ] > 12
]


counts_final = (
    high_final
    .groupby("frame")
    ["track_id"]
    .nunique()
)


simultaneous_final = counts_final[
    counts_final >= 3
]


print("")
print("========================================")
print("GLOBAL CHECK AFTER CORRECTION")
print("========================================")

print(
    "Frames with >=3 players >12 m/s:",
    len(simultaneous_final)
)


if len(simultaneous_final) > 0:

    print("")
    print(
        simultaneous_final
        .sort_values(
            ascending=False
        )
        .head(20)
    )


print("")
print(
    "Saved:",
    OUTPUT_CSV
)
