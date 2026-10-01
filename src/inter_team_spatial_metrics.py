import os

import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist


# ============================================================
# Paths
# ============================================================

INPUT_CSV = "outputs/tracking_data_tactical.csv"

FRAME_OUTPUT = "outputs/inter_team_spatial_metrics_frame.csv"
SUMMARY_OUTPUT = "outputs/inter_team_spatial_metrics_summary.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Load
# ============================================================

df = pd.read_csv(INPUT_CSV)

df = df[
    df["team"].isin(["A", "B"])
].copy()


df = df[
    np.isfinite(df["pitch_x_final_m"])
    &
    np.isfinite(df["pitch_y_final_m"])
].copy()


print("")
print("========================================")
print("INTER-TEAM SPATIAL METRICS")
print("========================================")

print("")
print(
    "Frames:",
    df["frame"].nunique()
)


# ============================================================
# Helpers
# ============================================================

def convex_hull_area(points):

    if len(points) < 3:
        return np.nan

    try:

        hull = ConvexHull(points)

        # scipy 2D:
        # hull.volume = polygon area
        return float(
            hull.volume
        )

    except Exception:

        return np.nan


# ============================================================
# Frame-by-frame
# ============================================================

rows = []


for frame_id, frame_df in df.groupby("frame"):

    team_a = frame_df[
        frame_df["team"] == "A"
    ].copy()

    team_b = frame_df[
        frame_df["team"] == "B"
    ].copy()


    # Tactical dataset 理論上必須完整 10v10
    if (
        team_a["track_id"].nunique() != 10
        or
        team_b["track_id"].nunique() != 10
    ):
        continue


    time_sec = float(
        frame_df["time_sec"].iloc[0]
    )


    # ========================================================
    # Coordinates
    # ========================================================

    points_a = team_a[
        [
            "pitch_x_final_m",
            "pitch_y_final_m"
        ]
    ].to_numpy(dtype=float)


    points_b = team_b[
        [
            "pitch_x_final_m",
            "pitch_y_final_m"
        ]
    ].to_numpy(dtype=float)


    # ========================================================
    # 1. Centroids
    # ========================================================

    centroid_a = np.mean(
        points_a,
        axis=0
    )

    centroid_b = np.mean(
        points_b,
        axis=0
    )


    centroid_dx = float(
        centroid_a[0]
        -
        centroid_b[0]
    )

    centroid_dy = float(
        centroid_a[1]
        -
        centroid_b[1]
    )


    centroid_distance = float(
        np.linalg.norm(
            centroid_a
            -
            centroid_b
        )
    )


    # ========================================================
    # 2. Convex Hull Area
    # ========================================================

    hull_area_a = (
        convex_hull_area(
            points_a
        )
    )

    hull_area_b = (
        convex_hull_area(
            points_b
        )
    )


    # ========================================================
    # 3. Player-to-opponent distances
    # ========================================================

    distance_matrix = cdist(
        points_a,
        points_b
    )


    # 每個 A 球員距離最近的 B
    nearest_a = np.min(
        distance_matrix,
        axis=1
    )

    # 每個 B 球員距離最近的 A
    nearest_b = np.min(
        distance_matrix,
        axis=0
    )


    all_nearest = np.concatenate(
        [
            nearest_a,
            nearest_b
        ]
    )


    mean_nearest_opponent = float(
        np.mean(
            all_nearest
        )
    )

    median_nearest_opponent = float(
        np.median(
            all_nearest
        )
    )


    # 最接近的一對球員
    minimum_inter_team_distance = float(
        np.min(
            distance_matrix
        )
    )


    # ========================================================
    # 4. Team hull area difference
    # ========================================================

    hull_area_difference = float(
        hull_area_a
        -
        hull_area_b
    )


    if (
        np.isfinite(hull_area_a)
        and
        np.isfinite(hull_area_b)
        and
        hull_area_b > 0
    ):

        hull_area_ratio = float(
            hull_area_a
            /
            hull_area_b
        )

    else:

        hull_area_ratio = np.nan


    # ========================================================
    # Save
    # ========================================================

    rows.append(
        {
            "frame":
                int(frame_id),

            "time_sec":
                time_sec,

            "centroid_A_x_m":
                float(
                    centroid_a[0]
                ),

            "centroid_A_y_m":
                float(
                    centroid_a[1]
                ),

            "centroid_B_x_m":
                float(
                    centroid_b[0]
                ),

            "centroid_B_y_m":
                float(
                    centroid_b[1]
                ),

            "centroid_dx_m":
                centroid_dx,

            "centroid_dy_m":
                centroid_dy,

            "centroid_distance_m":
                centroid_distance,

            "convex_hull_A_m2":
                hull_area_a,

            "convex_hull_B_m2":
                hull_area_b,

            "convex_hull_difference_m2":
                hull_area_difference,

            "convex_hull_ratio_A_over_B":
                hull_area_ratio,

            "mean_nearest_opponent_m":
                mean_nearest_opponent,

            "median_nearest_opponent_m":
                median_nearest_opponent,

            "minimum_inter_team_distance_m":
                minimum_inter_team_distance
        }
    )


# ============================================================
# Save frame metrics
# ============================================================

frame_metrics = pd.DataFrame(
    rows
)


frame_metrics.to_csv(
    FRAME_OUTPUT,
    index=False
)


# ============================================================
# Summary
# ============================================================

summary_metrics = [

    "centroid_dx_m",
    "centroid_dy_m",
    "centroid_distance_m",

    "convex_hull_A_m2",
    "convex_hull_B_m2",

    "convex_hull_difference_m2",
    "convex_hull_ratio_A_over_B",

    "mean_nearest_opponent_m",
    "median_nearest_opponent_m",

    "minimum_inter_team_distance_m"
]


summary_rows = []


for metric in summary_metrics:

    values = (
        frame_metrics[
            metric
        ]
        .dropna()
    )


    summary_rows.append(
        {
            "metric":
                metric,

            "median":
                float(
                    values.median()
                ),

            "p25":
                float(
                    values.quantile(
                        0.25
                    )
                ),

            "p75":
                float(
                    values.quantile(
                        0.75
                    )
                )
        }
    )


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    SUMMARY_OUTPUT,
    index=False
)


# ============================================================
# Display
# ============================================================

print("")
print("========================================")
print("SPATIAL SUMMARY")
print("========================================")
print("")


for _, row in summary.iterrows():

    print(
        f"{row['metric']:<35}"
        f"{row['median']:8.2f} "
        f"[{row['p25']:.2f}, "
        f"{row['p75']:.2f}]"
    )


print("")
print("========================================")
print("INTERPRETATION HELP")
print("========================================")

print("")
print(
    "centroid_distance_m:"
)
print(
    "  distance between the two team centers"
)

print("")
print(
    "convex_hull_*:"
)
print(
    "  actual geometric footprint of the 10 players"
)

print("")
print(
    "mean_nearest_opponent_m:"
)
print(
    "  average distance to each player's nearest opponent"
)

print("")
print(
    "minimum_inter_team_distance_m:"
)
print(
    "  closest A-B player pair in each frame"
)


print("")
print("========================================")
print("DONE")
print("========================================")

print(
    "Frame metrics:",
    FRAME_OUTPUT
)

print(
    "Summary:",
    SUMMARY_OUTPUT
)
