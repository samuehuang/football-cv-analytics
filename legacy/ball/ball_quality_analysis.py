import numpy as np
import pandas as pd


CSV_PATH = "outputs/ball_tracking_v1.csv"

FPS = 25.0

# 只在短 detection gap 下評估 movement
MAX_FRAME_GAP_FOR_SPEED = 3

# 這不是絕對物理上限，只拿來找可疑 teleport
SUSPICIOUS_SPEED = 30.0
EXTREME_SPEED = 40.0


# ============================================================
# Load
# ============================================================

df = pd.read_csv(CSV_PATH)


print("")
print("========================================")
print("BALL TRACK QUALITY")
print("========================================")


total_frames = len(df)

detected = df[
    df["detected"] == True
].copy()

projected = df[
    df["projected"] == True
].copy()


print("")
print("Total frames:", total_frames)

print(
    "Detected frames:",
    len(detected),
    f"({len(detected) / total_frames * 100:.1f}%)"
)

print(
    "Projected frames:",
    len(projected),
    f"({len(projected) / total_frames * 100:.1f}%)"
)


# ============================================================
# Confidence
# ============================================================

confidence = (
    detected["confidence"]
    .dropna()
)


print("")
print("========================================")
print("CONFIDENCE")
print("========================================")

print(
    "Median:",
    round(
        confidence.median(),
        3
    )
)

print(
    "P25:",
    round(
        confidence.quantile(0.25),
        3
    )
)

print(
    "P75:",
    round(
        confidence.quantile(0.75),
        3
    )
)

print(
    "Minimum:",
    round(
        confidence.min(),
        3
    )
)


# ============================================================
# Detection gaps
# ============================================================

detected_frames = set(
    detected["frame"]
    .astype(int)
    .tolist()
)


gaps = []

current_gap = 0

gap_start = None


for frame in range(total_frames):

    if frame not in detected_frames:

        if current_gap == 0:
            gap_start = frame

        current_gap += 1

    else:

        if current_gap > 0:

            gaps.append(
                {
                    "start_frame":
                        gap_start,

                    "end_frame":
                        frame - 1,

                    "length_frames":
                        current_gap,

                    "duration_sec":
                        current_gap / FPS
                }
            )

            current_gap = 0

            gap_start = None


if current_gap > 0:

    gaps.append(
        {
            "start_frame":
                gap_start,

            "end_frame":
                total_frames - 1,

            "length_frames":
                current_gap,

            "duration_sec":
                current_gap / FPS
        }
    )


gap_df = pd.DataFrame(gaps)


print("")
print("========================================")
print("DETECTION GAPS")
print("========================================")


if len(gap_df) == 0:

    print(
        "No missing-ball gaps."
    )

else:

    print(
        "Number of gaps:",
        len(gap_df)
    )

    print(
        "Longest gap:",
        int(
            gap_df[
                "length_frames"
            ].max()
        ),
        "frames"
    )

    print(
        "Longest duration:",
        round(
            gap_df[
                "duration_sec"
            ].max(),
            2
        ),
        "sec"
    )

    print("")
    print("Longest gaps:")

    print(
        gap_df
        .sort_values(
            "length_frames",
            ascending=False
        )
        .head(15)
        .to_string(
            index=False
        )
    )


# ============================================================
# Pitch trajectory quality
# ============================================================

track = (
    projected
    .sort_values("frame")
    .copy()
    .reset_index(drop=True)
)


track["frame_gap"] = (
    track["frame"].diff()
)

track["dt"] = (
    track["time_sec"].diff()
)

track["dx_m"] = (
    track["pitch_x_m"].diff()
)

track["dy_m"] = (
    track["pitch_y_m"].diff()
)


track["distance_m"] = np.sqrt(

    track["dx_m"] ** 2

    +

    track["dy_m"] ** 2
)


track["speed_mps"] = (
    track["distance_m"]
    /
    track["dt"]
)


# Only compare detections that are temporally close
short_gap = track[
    (
        track["frame_gap"]
        <= MAX_FRAME_GAP_FOR_SPEED
    )
    &
    (
        track["frame_gap"] > 0
    )
].copy()


speeds = (
    short_gap["speed_mps"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .dropna()
)


print("")
print("========================================")
print("RAW BALL MOVEMENT")
print("========================================")


print(
    "Movement samples:",
    len(speeds)
)

print(
    "Median speed:",
    round(
        speeds.median(),
        2
    ),
    "m/s"
)

print(
    "P95 speed:",
    round(
        speeds.quantile(0.95),
        2
    ),
    "m/s"
)

print(
    "P99 speed:",
    round(
        speeds.quantile(0.99),
        2
    ),
    "m/s"
)

print(
    "Max speed:",
    round(
        speeds.max(),
        2
    ),
    "m/s"
)


print(
    "Samples >30 m/s:",
    int(
        (
            speeds
            > SUSPICIOUS_SPEED
        ).sum()
    )
)

print(
    "Samples >40 m/s:",
    int(
        (
            speeds
            > EXTREME_SPEED
        ).sum()
    )
)


# ============================================================
# Top suspicious events
# ============================================================

suspicious = short_gap[
    short_gap["speed_mps"]
    > SUSPICIOUS_SPEED
].copy()


suspicious = suspicious.sort_values(
    "speed_mps",
    ascending=False
)


print("")
print("========================================")
print("TOP SUSPICIOUS BALL JUMPS")
print("========================================")


if len(suspicious) == 0:

    print(
        "No >30 m/s short-gap events."
    )

else:

    print(
        suspicious[
            [
                "frame",
                "time_sec",
                "frame_gap",
                "confidence",
                "pitch_x_m",
                "pitch_y_m",
                "distance_m",
                "speed_mps"
            ]
        ]
        .head(30)
        .round(2)
        .to_string(
            index=False
        )
    )


# ============================================================
# Low-confidence detections
# ============================================================

print("")
print("========================================")
print("LOW CONFIDENCE DETECTIONS")
print("========================================")


low_conf = detected[
    detected["confidence"]
    < 0.25
].copy()


print(
    "conf < 0.25:",
    len(low_conf)
)

print(
    f"({len(low_conf) / len(detected) * 100:.1f}% of detections)"
)


# ============================================================
# Final
# ============================================================

print("")
print("========================================")
print("DONE")
print("========================================")
