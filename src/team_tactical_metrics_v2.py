import os
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

INPUT_CSV = "outputs/tracking_data_tactical.csv"

FRAME_OUTPUT = "outputs/team_tactical_metrics_frame_v2.csv"
SUMMARY_OUTPUT = "outputs/team_tactical_metrics_summary_v2.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Parameters
# ============================================================

LOW_PERCENTILE = 10
HIGH_PERCENTILE = 90


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
print("TACTICAL METRICS V2")
print("Complete 10v10 frames only")
print("========================================")
print("")

print(
    "Rows:",
    len(df)
)

print(
    "Frames:",
    df["frame"].nunique()
)


# ============================================================
# Frame-by-frame
# ============================================================

results = []


for frame_id, frame_df in df.groupby("frame"):

    time_sec = float(
        frame_df["time_sec"].iloc[0]
    )


    for team_name in ["A", "B"]:

        team = frame_df[
            frame_df["team"] == team_name
        ]


        # Tactical dataset should ALWAYS contain 10
        n_players = (
            team["track_id"]
            .nunique()
        )


        if n_players != 10:

            continue


        x = (
            team["pitch_x_final_m"]
            .to_numpy(dtype=float)
        )

        y = (
            team["pitch_y_final_m"]
            .to_numpy(dtype=float)
        )


        # ====================================================
        # Centroid
        # ====================================================

        centroid_x = float(
            np.mean(x)
        )

        centroid_y = float(
            np.mean(y)
        )


        # ====================================================
        # Length
        # ====================================================

        x_low = float(
            np.percentile(
                x,
                LOW_PERCENTILE
            )
        )

        x_high = float(
            np.percentile(
                x,
                HIGH_PERCENTILE
            )
        )

        team_length = (
            x_high - x_low
        )


        # ====================================================
        # Width
        # ====================================================

        y_low = float(
            np.percentile(
                y,
                LOW_PERCENTILE
            )
        )

        y_high = float(
            np.percentile(
                y,
                HIGH_PERCENTILE
            )
        )

        team_width = (
            y_high - y_low
        )


        # ====================================================
        # Compactness
        # ====================================================

        radial_distance = np.sqrt(
            (x - centroid_x) ** 2
            +
            (y - centroid_y) ** 2
        )

        compactness = float(
            np.mean(
                radial_distance
            )
        )


        # ====================================================
        # Shape area proxy
        # ====================================================

        shape_area = float(
            team_length
            *
            team_width
        )


        results.append(
            {
                "frame":
                    int(frame_id),

                "time_sec":
                    time_sec,

                "team":
                    team_name,

                "players":
                    n_players,

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
# Frame Metrics
# ============================================================

frame_metrics = pd.DataFrame(
    results
)


frame_metrics.to_csv(
    FRAME_OUTPUT,
    index=False
)


# ============================================================
# Summary
# ============================================================

summary_rows = []


columns = [

    "centroid_x_m",
    "centroid_y_m",

    "team_length_m",
    "team_width_m",

    "compactness_m",
    "shape_area_m2"
]


for team_name in ["A", "B"]:

    team = frame_metrics[
        frame_metrics["team"]
        == team_name
    ]


    row = {

        "team":
            team_name,

        "valid_frames":
            len(team)
    }


    for column in columns:

        values = team[column]


        row[
            f"{column}_median"
        ] = float(
            values.median()
        )

        row[
            f"{column}_p25"
        ] = float(
            values.quantile(0.25)
        )

        row[
            f"{column}_p75"
        ] = float(
            values.quantile(0.75)
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
# Print
# ============================================================

print("")
print("========================================")
print("CLEAN TACTICAL SUMMARY")
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
        "Length:",
        f"{row['team_length_m_median']:.2f} m",
        f"[{row['team_length_m_p25']:.2f}, "
        f"{row['team_length_m_p75']:.2f}]"
    )

    print(
        "Width:",
        f"{row['team_width_m_median']:.2f} m",
        f"[{row['team_width_m_p25']:.2f}, "
        f"{row['team_width_m_p75']:.2f}]"
    )

    print(
        "Compactness:",
        f"{row['compactness_m_median']:.2f} m",
        f"[{row['compactness_m_p25']:.2f}, "
        f"{row['compactness_m_p75']:.2f}]"
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
