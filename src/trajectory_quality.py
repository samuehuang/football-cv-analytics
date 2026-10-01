import pandas as pd
import numpy as np


CSV_PATH = "outputs/tracking_data_v4.csv"

df = pd.read_csv(CSV_PATH)

# 先只分析正常球員
df = df[
    df["team"].isin(["A", "B"])
].copy()

df = df.sort_values(
    ["track_id", "frame"]
)


results = []


for track_id, track in df.groupby("track_id"):

    track = track.sort_values("frame").copy()

    if len(track) < 2:
        continue

    # -------------------------------------------------
    # Frame difference
    # -------------------------------------------------

    track["dt"] = (
        track["time_sec"].diff()
    )

    # -------------------------------------------------
    # Pitch displacement
    # -------------------------------------------------

    dx = (
        track["pitch_x_m"].diff()
    )

    dy = (
        track["pitch_y_m"].diff()
    )

    track["displacement_m"] = np.sqrt(
        dx ** 2 + dy ** 2
    )

    # -------------------------------------------------
    # Instantaneous speed
    # -------------------------------------------------

    track["speed_mps"] = (
        track["displacement_m"]
        / track["dt"]
    )

    valid_speed = track[
        np.isfinite(track["speed_mps"])
    ]["speed_mps"]

    if len(valid_speed) == 0:
        continue

    results.append(
        {
            "track_id": track_id,
            "team": track["team"].mode()[0],
            "frames": len(track),

            "median_speed_mps":
                valid_speed.median(),

            "p95_speed_mps":
                valid_speed.quantile(0.95),

            "max_speed_mps":
                valid_speed.max(),

            "frames_over_12_mps":
                int(
                    (valid_speed > 12).sum()
                ),

            "frames_over_15_mps":
                int(
                    (valid_speed > 15).sum()
                )
        }
    )


summary = pd.DataFrame(results)

summary = summary.sort_values(
    "max_speed_mps",
    ascending=False
)


print("")
print("========================================")
print("TRAJECTORY QUALITY")
print("========================================")
print("")

print(
    summary.to_string(
        index=False
    )
)


print("")
print("========================================")
print("GLOBAL")
print("========================================")

print(
    "Tracks:",
    len(summary)
)

print(
    "Tracks with >15 m/s frame:",
    (
        summary["frames_over_15_mps"] > 0
    ).sum()
)
