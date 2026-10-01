import os
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

INPUT_CSV = "outputs/tracking_data_clean_v2.csv"

FRAME_OUTPUT = "outputs/team_tactical_metrics_frame.csv"
SUMMARY_OUTPUT = "outputs/team_tactical_metrics_summary.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Parameters
# ============================================================

# 每一幀至少偵測到多少名該隊球員才納入 tactical analysis
MIN_PLAYERS_PER_TEAM = 7

# 使用 percentile 而不是 min/max
# 避免單一定位異常把 width / length 拉爆
LOW_PERCENTILE = 10
HIGH_PERCENTILE = 90


# ============================================================
# Load
# ============================================================

df = pd.read_csv(INPUT_CSV)

df = df[
    df["team"].isin(["A", "B"])
].copy()


# 使用 final cleaned coordinates
df = df[
    np.isfinite(df["pitch_x_final_m"])
    &
    np.isfinite(df["pitch_y_final_m"])
].copy()


print("")
print("========================================")
print("TEAM TACTICAL METRICS")
print("========================================")

print("")
print("Rows:", len(df))
print(
    "Frames:",
    df["frame"].nunique()
)


# ============================================================
# Frame-by-frame analysis
# ============================================================

frame_results = []


for frame_id, frame_df in df.groupby("frame"):

    time_sec = frame_df[
        "time_sec"
    ].iloc[0]


    for team_name in ["A", "B"]:

        team = frame_df[
            frame_df["team"]
            == team_name
        ].copy()


        n_players = (
            team["track_id"]
            .nunique()
        )


        if n_players < MIN_PLAYERS_PER_TEAM:
            continue


        x = team[
            "pitch_x_final_m"
        ].to_numpy()

        y = team[
            "pitch_y_final_m"
        ].to_numpy()


        # ====================================================
        # 1. Team Centroid
        # ====================================================

        centroid_x = float(
            np.mean(x)
        )

        centroid_y = float(
            np.mean(y)
        )


        # ====================================================
        # 2. Team Length
        #
        # Pitch longitudinal direction = X
        #
        # 使用 P90 - P10，
        # 比 max-min 對單一 outlier 更 robust。
        # ====================================================

        x_low = np.percentile(
            x,
            LOW_PERCENTILE
        )

        x_high = np.percentile(
            x,
            HIGH_PERCENTILE
        )

        team_length = float(
            x_high - x_low
        )


        # ====================================================
        # 3. Team Width
        #
        # Pitch lateral direction = Y
        # ====================================================

        y_low = np.percentile(
            y,
            LOW_PERCENTILE
        )

        y_high = np.percentile(
            y,
            HIGH_PERCENTILE
        )

        team_width = float(
            y_high - y_low
        )


        # ====================================================
        # 4. Compactness
        #
        # 每名球員到 team centroid 的平均距離
        #
        # 越小 -> 越緊密
        # 越大 -> 越 stretched
        # ====================================================

        distances_to_centroid = np.sqrt(
            (x - centroid_x) ** 2
            +
            (y - centroid_y) ** 2
        )

        compactness = float(
            np.mean(
                distances_to_centroid
            )
        )


        # ====================================================
        # 5. Spread area proxy
        #
        # 先使用 length * width 當簡單 footprint proxy
        #
        # 後面可再做 convex hull area
        # ====================================================

        shape_area = float(
            team_length
            *
            team_width
        )


        frame_results.append(
            {
                "frame":
                    int(frame_id),

                "time_sec":
                    float(time_sec),

                "team":
                    team_name,

                "players":
                    int(n_players),

                "centroid_x_m":
                    centroid_x,

                "centroid_y_m":
                    centroid_y,

                "team_length_m":
                    team_length,

                "team_width_m":
                    team_width,

                "compactness_m":
                    compactness,

                "shape_area_m2":
                    shape_area
            }
        )


# ============================================================
# Frame DataFrame
# ============================================================

frame_metrics = pd.DataFrame(
    frame_results
)

frame_metrics.to_csv(
    FRAME_OUTPUT,
    index=False
)


# ============================================================
# Team Summary
# ============================================================

summary_rows = []


for team_name in ["A", "B"]:

    team = frame_metrics[
        frame_metrics["team"]
        == team_name
    ]


    if len(team) == 0:
        continue


    row = {
        "team":
            team_name,

        "valid_frames":
            len(team)
    }


    metric_columns = [

        "centroid_x_m",
        "centroid_y_m",

        "team_length_m",
        "team_width_m",

        "compactness_m",
        "shape_area_m2"
    ]


    for column in metric_columns:

        values = team[column]


        row[
            f"{column}_median"
        ] = float(
            values.median()
        )


        row[
            f"{column}_p25"
        ] = float(
            values.quantile(
                0.25
            )
        )


        row[
            f"{column}_p75"
        ] = float(
            values.quantile(
                0.75
            )
        )


    summary_rows.append(
        row
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
print("TACTICAL SUMMARY")
print("========================================")
print("")


for _, row in summary.iterrows():

    print(
        f"Team {row['team']}"
    )

    print(
        "Valid frames:",
        int(
            row["valid_frames"]
        )
    )

    print(
        "Centroid:",
        f"({row['centroid_x_m_median']:.2f}, "
        f"{row['centroid_y_m_median']:.2f}) m"
    )

    print(
        "Team Length:",
        f"{row['team_length_m_median']:.2f} m",
        f"[P25 {row['team_length_m_p25']:.2f}, "
        f"P75 {row['team_length_m_p75']:.2f}]"
    )

    print(
        "Team Width:",
        f"{row['team_width_m_median']:.2f} m",
        f"[P25 {row['team_width_m_p25']:.2f}, "
        f"P75 {row['team_width_m_p75']:.2f}]"
    )

    print(
        "Compactness:",
        f"{row['compactness_m_median']:.2f} m",
        f"[P25 {row['compactness_m_p25']:.2f}, "
        f"P75 {row['compactness_m_p75']:.2f}]"
    )

    print(
        "Shape Area:",
        f"{row['shape_area_m2_median']:.1f} m²"
    )

    print("")


print(
    "Frame metrics:",
    FRAME_OUTPUT
)

print(
    "Summary:",
    SUMMARY_OUTPUT
)
