import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

INPUT_CSV = "outputs/team_tactical_metrics_frame.csv"

LENGTH_OUTPUT = "outputs/team_length_over_time.png"
WIDTH_OUTPUT = "outputs/team_width_over_time.png"
COMPACTNESS_OUTPUT = "outputs/team_compactness_over_time.png"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Parameters
# ============================================================

# Rolling smoothing
# 25 frames ≈ 1 second
ROLLING_WINDOW = 25


# ============================================================
# Load
# ============================================================

df = pd.read_csv(INPUT_CSV)


print("")
print("========================================")
print("TEAM SHAPE OVER TIME")
print("========================================")

print("")
print("Rows:", len(df))

print(
    "Frames:",
    df["frame"].nunique()
)


# ============================================================
# Smooth each team separately
# ============================================================

processed = []


for team_name, team in df.groupby("team"):

    team = (
        team
        .sort_values("frame")
        .copy()
    )


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


    processed.append(team)


df = pd.concat(
    processed,
    ignore_index=True
)


# ============================================================
# Helper
# ============================================================

def plot_metric(
    metric,
    smooth_metric,
    ylabel,
    title,
    output_path
):

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )


    for team_name in ["A", "B"]:

        team = df[
            df["team"]
            == team_name
        ]


        # Raw values
        ax.plot(
            team["time_sec"],
            team[metric],
            alpha=0.18,
            linewidth=1
        )


        # Smoothed curve
        ax.plot(
            team["time_sec"],
            team[smooth_metric],
            linewidth=2.5,
            label=f"Team {team_name}"
        )


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
# 1. Team Length
# ============================================================

plot_metric(

    metric="team_length_m",

    smooth_metric="length_smooth",

    ylabel="Team Length (m)",

    title="Team Length Over Time",

    output_path=LENGTH_OUTPUT
)


# ============================================================
# 2. Team Width
# ============================================================

plot_metric(

    metric="team_width_m",

    smooth_metric="width_smooth",

    ylabel="Team Width (m)",

    title="Team Width Over Time",

    output_path=WIDTH_OUTPUT
)


# ============================================================
# 3. Compactness
# ============================================================

plot_metric(

    metric="compactness_m",

    smooth_metric="compactness_smooth",

    ylabel="Mean Distance to Team Centroid (m)",

    title="Team Compactness Over Time",

    output_path=COMPACTNESS_OUTPUT
)


# ============================================================
# Find largest expansion moments
# ============================================================

print("")
print("========================================")
print("TOP EXPANSION MOMENTS")
print("========================================")


for team_name in ["A", "B"]:

    team = df[
        df["team"]
        == team_name
    ]


    top_length = (
        team
        .nlargest(
            5,
            "length_smooth"
        )
        [
            [
                "time_sec",
                "length_smooth",
                "width_smooth",
                "compactness_smooth"
            ]
        ]
    )


    print("")
    print(
        f"Team {team_name}"
    )

    print(
        top_length
        .round(2)
        .to_string(
            index=False
        )
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
    COMPACTNESS_OUTPUT
)
