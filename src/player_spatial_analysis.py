import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle, Circle
from scipy.ndimage import gaussian_filter


# ============================================================
# Paths
# ============================================================

TRACKING_CSV = "outputs/tracking_data_clean_v2.csv"
METRICS_CSV = "outputs/player_metrics.csv"

AVG_POSITION_OUTPUT = "outputs/average_positions.png"
TEAM_A_HEATMAP_OUTPUT = "outputs/team_A_heatmap.png"
TEAM_B_HEATMAP_OUTPUT = "outputs/team_B_heatmap.png"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Parameters
# ============================================================

PITCH_LENGTH = 120.0
PITCH_WIDTH = 70.0

# 太短的 Track 不拿來代表平均站位
MIN_TRACKED_TIME = 20.0

# Heatmap resolution
X_BINS = 60
Y_BINS = 35

# Gaussian smoothing
HEATMAP_SIGMA = 1.5


# ============================================================
# Draw Pitch
# ============================================================

def draw_pitch(ax):

    ax.set_xlim(0, PITCH_LENGTH)
    ax.set_ylim(0, PITCH_WIDTH)

    ax.set_aspect("equal")

    # Outer boundary
    ax.add_patch(
        Rectangle(
            (0, 0),
            PITCH_LENGTH,
            PITCH_WIDTH,
            fill=False,
            linewidth=2
        )
    )

    # Halfway line
    ax.plot(
        [PITCH_LENGTH / 2, PITCH_LENGTH / 2],
        [0, PITCH_WIDTH],
        linewidth=1.5
    )

    # Center circle
    ax.add_patch(
        Circle(
            (
                PITCH_LENGTH / 2,
                PITCH_WIDTH / 2
            ),
            9.15,
            fill=False,
            linewidth=1.5
        )
    )

    # Center point
    ax.scatter(
        [PITCH_LENGTH / 2],
        [PITCH_WIDTH / 2],
        s=15
    )

    # --------------------------------------------------------
    # Penalty areas
    # --------------------------------------------------------

    penalty_width = 40.32
    penalty_depth = 16.5

    penalty_y = (
        PITCH_WIDTH - penalty_width
    ) / 2

    # Left
    ax.add_patch(
        Rectangle(
            (0, penalty_y),
            penalty_depth,
            penalty_width,
            fill=False,
            linewidth=1.5
        )
    )

    # Right
    ax.add_patch(
        Rectangle(
            (
                PITCH_LENGTH - penalty_depth,
                penalty_y
            ),
            penalty_depth,
            penalty_width,
            fill=False,
            linewidth=1.5
        )
    )

    # --------------------------------------------------------
    # Goal areas
    # --------------------------------------------------------

    goal_area_width = 18.32
    goal_area_depth = 5.5

    goal_area_y = (
        PITCH_WIDTH - goal_area_width
    ) / 2

    # Left
    ax.add_patch(
        Rectangle(
            (0, goal_area_y),
            goal_area_depth,
            goal_area_width,
            fill=False,
            linewidth=1
        )
    )

    # Right
    ax.add_patch(
        Rectangle(
            (
                PITCH_LENGTH - goal_area_depth,
                goal_area_y
            ),
            goal_area_depth,
            goal_area_width,
            fill=False,
            linewidth=1
        )
    )

    ax.set_xlabel("Pitch X (m)")
    ax.set_ylabel("Pitch Y (m)")

    ax.set_xticks(
        np.arange(
            0,
            121,
            10
        )
    )

    ax.set_yticks(
        np.arange(
            0,
            71,
            10
        )
    )

    ax.grid(
        alpha=0.12
    )


# ============================================================
# Load
# ============================================================

tracking = pd.read_csv(
    TRACKING_CSV
)

metrics = pd.read_csv(
    METRICS_CSV
)


# Only A / B
tracking = tracking[
    tracking["team"].isin(["A", "B"])
].copy()

metrics = metrics[
    metrics["team"].isin(["A", "B"])
].copy()


# ============================================================
# Valid spatial-analysis players
# ============================================================

valid_metrics = metrics[
    metrics["tracked_time_sec"]
    >= MIN_TRACKED_TIME
].copy()


valid_ids = set(
    valid_metrics[
        "track_id"
    ].tolist()
)


tracking = tracking[
    tracking[
        "track_id"
    ].isin(valid_ids)
].copy()


print("")
print("========================================")
print("PLAYER SPATIAL ANALYSIS")
print("========================================")

print("")
print(
    "Players used:",
    len(valid_metrics)
)

print(
    "Team A:",
    (
        valid_metrics["team"] == "A"
    ).sum()
)

print(
    "Team B:",
    (
        valid_metrics["team"] == "B"
    ).sum()
)

print("")
print("Included Track IDs:")

print(
    valid_metrics[
        [
            "track_id",
            "team",
            "tracked_time_sec"
        ]
    ]
    .sort_values(
        ["team", "track_id"]
    )
    .to_string(index=False)
)


# ============================================================
# 1. Average Position Map
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 8)
)

draw_pitch(ax)


team_styles = {

    "A": {
        "color": "tab:blue",
        "marker": "o"
    },

    "B": {
        "color": "tab:red",
        "marker": "s"
    }

}


for team_name in ["A", "B"]:

    team_data = valid_metrics[
        valid_metrics["team"]
        == team_name
    ]


    style = team_styles[
        team_name
    ]


    ax.scatter(
        team_data["avg_x_m"],
        team_data["avg_y_m"],
        s=180,
        color=style["color"],
        marker=style["marker"],
        edgecolors="black",
        linewidths=1.2,
        alpha=0.85,
        label=f"Team {team_name}"
    )


    # Player ID labels
    for _, row in team_data.iterrows():

        ax.text(
            row["avg_x_m"],
            row["avg_y_m"],
            str(
                int(
                    row["track_id"]
                )
            ),
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold"
        )


    # --------------------------------------------------------
    # Team centroid
    # --------------------------------------------------------

    centroid_x = (
        team_data[
            "avg_x_m"
        ].mean()
    )

    centroid_y = (
        team_data[
            "avg_y_m"
        ].mean()
    )


    ax.scatter(
        centroid_x,
        centroid_y,
        marker="*",
        s=350,
        color=style["color"],
        edgecolors="black",
        linewidths=1.5
    )


    ax.text(
        centroid_x + 1.0,
        centroid_y + 1.0,
        f"{team_name} centroid",
        fontsize=9
    )


ax.set_title(
    "Average Player Positions"
)

ax.legend(
    loc="upper right"
)

plt.tight_layout()

plt.savefig(
    AVG_POSITION_OUTPUT,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Heatmap Function
# ============================================================

def build_team_heatmap(
    team_name,
    output_path
):

    team_tracking = tracking[
        tracking["team"]
        == team_name
    ].copy()


    team_ids = (
        team_tracking[
            "track_id"
        ]
        .unique()
    )


    # --------------------------------------------------------
    # 每個球員先各自做 histogram
    #
    # 再 normalize 後相加，
    # 避免追蹤比較久的球員支配整張 heatmap
    # --------------------------------------------------------

    combined_heatmap = np.zeros(
        (
            Y_BINS,
            X_BINS
        ),
        dtype=float
    )


    used_players = 0


    for track_id in team_ids:

        player = team_tracking[
            team_tracking["track_id"]
            == track_id
        ]


        x = (
            player[
                "pitch_x_final_m"
            ]
            .to_numpy()
        )

        y = (
            player[
                "pitch_y_final_m"
            ]
            .to_numpy()
        )


        valid = (

            np.isfinite(x)

            &

            np.isfinite(y)

            &

            (x >= 0)

            &

            (x <= PITCH_LENGTH)

            &

            (y >= 0)

            &

            (y <= PITCH_WIDTH)

        )


        x = x[valid]
        y = y[valid]


        if len(x) == 0:
            continue


        hist, _, _ = np.histogram2d(

            y,
            x,

            bins=[
                Y_BINS,
                X_BINS
            ],

            range=[
                [0, PITCH_WIDTH],
                [0, PITCH_LENGTH]
            ]

        )


        # Equal contribution per player
        if hist.sum() > 0:

            hist = (
                hist
                /
                hist.sum()
            )


            combined_heatmap += hist

            used_players += 1


    # --------------------------------------------------------
    # Average across players
    # --------------------------------------------------------

    if used_players > 0:

        combined_heatmap = (
            combined_heatmap
            /
            used_players
        )


    # Gaussian smoothing
    combined_heatmap = gaussian_filter(
        combined_heatmap,
        sigma=HEATMAP_SIGMA
    )


    # ========================================================
    # Draw
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(14, 8)
    )

    draw_pitch(ax)


    heatmap_img = ax.imshow(

        combined_heatmap,

        extent=[
            0,
            PITCH_LENGTH,
            0,
            PITCH_WIDTH
        ],

        origin="lower",

        aspect="auto",

        cmap="hot",

        alpha=0.72
    )


    # --------------------------------------------------------
    # Average positions overlay
    # --------------------------------------------------------

    team_metrics = valid_metrics[
        valid_metrics["team"]
        == team_name
    ]


    ax.scatter(
        team_metrics["avg_x_m"],
        team_metrics["avg_y_m"],
        s=80,
        edgecolors="black",
        linewidths=1,
        facecolors="white",
        alpha=0.9
    )


    for _, row in team_metrics.iterrows():

        ax.text(
            row["avg_x_m"],
            row["avg_y_m"],
            str(
                int(
                    row["track_id"]
                )
            ),
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            fontweight="bold"
        )


    ax.set_title(
        f"Team {team_name} Spatial Heatmap"
    )


    plt.colorbar(
        heatmap_img,
        ax=ax,
        fraction=0.03,
        pad=0.02,
        label="Normalized Occupancy Density"
    )


    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 2. Team Heatmaps
# ============================================================

build_team_heatmap(
    "A",
    TEAM_A_HEATMAP_OUTPUT
)

build_team_heatmap(
    "B",
    TEAM_B_HEATMAP_OUTPUT
)


# ============================================================
# Finish
# ============================================================

print("")
print("========================================")
print("DONE")
print("========================================")

print("")
print(
    "Average Positions:",
    AVG_POSITION_OUTPUT
)

print(
    "Team A Heatmap:",
    TEAM_A_HEATMAP_OUTPUT
)

print(
    "Team B Heatmap:",
    TEAM_B_HEATMAP_OUTPUT
)
