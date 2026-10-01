import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

INPUT_CSV = "outputs/team_tactical_metrics_frame_v2.csv"

LENGTH_OUTPUT = "outputs/team_length_over_time_v2.png"
WIDTH_OUTPUT = "outputs/team_width_over_time_v2.png"
COMPACT_OUTPUT = "outputs/team_compactness_over_time_v2.png"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Parameters
# ============================================================

# 約 1 秒 smoothing
ROLLING_WINDOW = 25

# 如果兩個 valid observations 中間超過這麼久，
# 不要直接畫線連起來
MAX_TIME_GAP = 0.12


# ============================================================
# Load
# ============================================================

df = pd.read_csv(INPUT_CSV)

df = df.sort_values(
    ["team", "time_sec"]
).copy()


print("")
print("========================================")
print("CLEAN TEAM SHAPE OVER TIME")
print("========================================")

print("")
print(
    "Frames:",
    df["frame"].nunique()
)

print(
    "Time range:",
    f"{df['time_sec'].min():.2f}s",
    "-",
    f"{df['time_sec'].max():.2f}s"
)


# ============================================================
# Process
# ============================================================

processed = []


for team_name, team in df.groupby("team"):

    team = (
        team
        .sort_values("time_sec")
        .copy()
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # Rolling median
    # --------------------------------------------------------

    team["length_smooth"] = (
        team["team_length_m"]
        .rolling(
            ROLLING_WINDOW,
            center=True,
            min_periods=1
        )
        .median()
    )


    team["width_smooth"] = (
        team["team_width_m"]
        .rolling(
            ROLLING_WINDOW,
            center=True,
            min_periods=1
        )
        .median()
    )


    team["compactness_smooth"] = (
        team["compactness_m"]
        .rolling(
            ROLLING_WINDOW,
            center=True,
            min_periods=1
        )
        .median()
    )


    # --------------------------------------------------------
    # Detect gaps
    #
    # complete 10v10 dataset 中間可能有缺 frame。
    # 不應該讓 matplotlib 直接跨缺口連線。
    # --------------------------------------------------------

    team["time_gap"] = (
        team["time_sec"].diff()
    )


    team["segment_id"] = (
        (
            team["time_gap"]
            > MAX_TIME_GAP
        )
        .cumsum()
    )


    processed.append(team)


df = pd.concat(
    processed,
    ignore_index=True
)


# ============================================================
# Plot function
# ============================================================

def plot_metric(
    raw_column,
    smooth_column,
    ylabel,
    title,
    output_path
):

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )


    for team_name in ["A", "B"]:

        team = df[
            df["team"] == team_name
        ]


        first_segment = True


        for _, segment in team.groupby(
            "segment_id"
        ):

            # Raw observations
            ax.plot(
                segment["time_sec"],
                segment[raw_column],
                alpha=0.15,
                linewidth=1
            )


            # Smoothed
            ax.plot(
                segment["time_sec"],
                segment[smooth_column],
                linewidth=2.5,
                label=(
                    f"Team {team_name}"
                    if first_segment
                    else None
                )
            )


            first_segment = False


    ax.set_xlabel(
        "Time (seconds)"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.grid(
        alpha=0.20
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Length
# ============================================================

plot_metric(

    raw_column="team_length_m",

    smooth_column="length_smooth",

    ylabel="Team Length (m)",

    title="Team Length Over Time — Clean 10v10 Frames",

    output_path=LENGTH_OUTPUT
)


# ============================================================
# Width
# ============================================================

plot_metric(

    raw_column="team_width_m",

    smooth_column="width_smooth",

    ylabel="Team Width (m)",

    title="Team Width Over Time — Clean 10v10 Frames",

    output_path=WIDTH_OUTPUT
)


# ============================================================
# Compactness
# ============================================================

plot_metric(

    raw_column="compactness_m",

    smooth_column="compactness_smooth",

    ylabel="Mean Distance to Centroid (m)",

    title="Team Compactness Over Time — Clean 10v10 Frames",

    output_path=COMPACT_OUTPUT
)


# ============================================================
# Difference between teams
# ============================================================

pivot = df.pivot_table(

    index=[
        "frame",
        "time_sec"
    ],

    columns="team",

    values=[
        "team_length_m",
        "team_width_m",
        "compactness_m"
    ]
)


pivot = pivot.dropna()


length_difference = (
    pivot[
        "team_length_m"
    ]["A"]
    -
    pivot[
        "team_length_m"
    ]["B"]
)


width_difference = (
    pivot[
        "team_width_m"
    ]["A"]
    -
    pivot[
        "team_width_m"
    ]["B"]
)


compact_difference = (
    pivot[
        "compactness_m"
    ]["A"]
    -
    pivot[
        "compactness_m"
    ]["B"]
)


print("")
print("========================================")
print("TEAM DIFFERENCE")
print("A minus B")
print("========================================")

print("")
print(
    "Median Length Difference:",
    f"{length_difference.median():.2f} m"
)

print(
    "Median Width Difference:",
    f"{width_difference.median():.2f} m"
)

print(
    "Median Compactness Difference:",
    f"{compact_difference.median():.2f} m"
)


print("")
print("========================================")
print("DONE")
print("========================================")

print("")
print(
    "Length:",
    LENGTH_OUTPUT
)

print(
    "Width:",
    WIDTH_OUTPUT
)

print(
    "Compactness:",
    COMPACT_OUTPUT
)
