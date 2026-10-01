import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle, Circle, Polygon
from scipy.spatial import ConvexHull


# ============================================================
# Paths
# ============================================================

TRACKING_CSV = "outputs/tracking_data_tactical.csv"
METRICS_CSV = "outputs/team_tactical_metrics_frame_v2.csv"

OUTPUT_PATH = "outputs/team_shape_snapshot.png"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Pitch
# ============================================================

PITCH_LENGTH = 120.0
PITCH_WIDTH = 70.0


def draw_pitch(ax):

    ax.set_xlim(0, PITCH_LENGTH)
    ax.set_ylim(0, PITCH_WIDTH)

    ax.set_aspect("equal")

    # Boundary
    ax.add_patch(
        Rectangle(
            (0, 0),
            PITCH_LENGTH,
            PITCH_WIDTH,
            fill=False,
            linewidth=2
        )
    )

    # Halfway
    ax.plot(
        [60, 60],
        [0, 70],
        linewidth=1.5
    )

    # Center circle
    ax.add_patch(
        Circle(
            (60, 35),
            9.15,
            fill=False,
            linewidth=1.5
        )
    )

    ax.scatter(
        [60],
        [35],
        s=15
    )

    # Penalty areas
    penalty_width = 40.32
    penalty_depth = 16.5

    py = (
        PITCH_WIDTH
        - penalty_width
    ) / 2


    ax.add_patch(
        Rectangle(
            (0, py),
            penalty_depth,
            penalty_width,
            fill=False,
            linewidth=1.5
        )
    )


    ax.add_patch(
        Rectangle(
            (
                PITCH_LENGTH
                - penalty_depth,
                py
            ),
            penalty_depth,
            penalty_width,
            fill=False,
            linewidth=1.5
        )
    )


    # Goal areas
    goal_width = 18.32
    goal_depth = 5.5

    gy = (
        PITCH_WIDTH
        - goal_width
    ) / 2


    ax.add_patch(
        Rectangle(
            (0, gy),
            goal_depth,
            goal_width,
            fill=False,
            linewidth=1
        )
    )


    ax.add_patch(
        Rectangle(
            (
                PITCH_LENGTH
                - goal_depth,
                gy
            ),
            goal_depth,
            goal_width,
            fill=False,
            linewidth=1
        )
    )


    ax.set_xlabel(
        "Pitch X (m)"
    )

    ax.set_ylabel(
        "Pitch Y (m)"
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


# ============================================================
# Find representative frame
#
# 找一個兩隊 Length / Width / Compactness
# 都最接近整段 median 的 frame
# ============================================================

metric_names = [
    "team_length_m",
    "team_width_m",
    "compactness_m"
]


pivot_parts = []


for team_name in ["A", "B"]:

    team = metrics[
        metrics["team"]
        == team_name
    ].copy()


    team = team[
        [
            "frame",
            "time_sec"
        ]
        +
        metric_names
    ]


    rename = {
        col:
            f"{team_name}_{col}"
        for col in metric_names
    }


    team = team.rename(
        columns=rename
    )


    pivot_parts.append(
        team
    )


combined = pd.merge(

    pivot_parts[0],

    pivot_parts[1],

    on=[
        "frame",
        "time_sec"
    ],

    how="inner"
)


# ============================================================
# Median values
# ============================================================

feature_columns = [

    "A_team_length_m",
    "A_team_width_m",
    "A_compactness_m",

    "B_team_length_m",
    "B_team_width_m",
    "B_compactness_m"
]


medians = (
    combined[
        feature_columns
    ]
    .median()
)


# ============================================================
# Normalize by robust scale
# ============================================================

q25 = (
    combined[
        feature_columns
    ]
    .quantile(0.25)
)

q75 = (
    combined[
        feature_columns
    ]
    .quantile(0.75)
)

iqr = (
    q75 - q25
)

iqr = iqr.replace(
    0,
    1.0
)


# ============================================================
# Distance from typical state
# ============================================================

normalized = (

    combined[
        feature_columns
    ]
    - medians

) / iqr


combined[
    "typical_score"
] = np.sqrt(
    (
        normalized ** 2
    ).sum(axis=1)
)


representative = (
    combined
    .sort_values(
        "typical_score"
    )
    .iloc[0]
)


REP_FRAME = int(
    representative[
        "frame"
    ]
)

REP_TIME = float(
    representative[
        "time_sec"
    ]
)


print("")
print("========================================")
print("REPRESENTATIVE TACTICAL FRAME")
print("========================================")

print("")
print(
    "Frame:",
    REP_FRAME
)

print(
    "Time:",
    f"{REP_TIME:.2f}s"
)

print(
    "Typical score:",
    round(
        representative[
            "typical_score"
        ],
        3
    )
)


# ============================================================
# Get players
# ============================================================

frame_df = tracking[
    tracking["frame"]
    == REP_FRAME
].copy()


print("")
print(
    "Players:",
    len(frame_df)
)

print(
    "Team A:",
    (
        frame_df["team"]
        == "A"
    ).sum()
)

print(
    "Team B:",
    (
        frame_df["team"]
        == "B"
    ).sum()
)


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(15, 9)
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


# ============================================================
# Team drawing
# ============================================================

for team_name in ["A", "B"]:

    team = frame_df[
        frame_df["team"]
        == team_name
    ].copy()


    x = (
        team[
            "pitch_x_final_m"
        ]
        .to_numpy()
    )

    y = (
        team[
            "pitch_y_final_m"
        ]
        .to_numpy()
    )


    points = np.column_stack(
        [x, y]
    )


    style = team_styles[
        team_name
    ]


    # ========================================================
    # Convex Hull
    # ========================================================

    if len(points) >= 3:

        hull = ConvexHull(
            points
        )


        hull_points = (
            points[
                hull.vertices
            ]
        )


        polygon = Polygon(

            hull_points,

            closed=True,

            facecolor=style["color"],

            edgecolor=style["color"],

            alpha=0.12,

            linewidth=2
        )


        ax.add_patch(
            polygon
        )


    # ========================================================
    # Players
    # ========================================================

    ax.scatter(

        x,
        y,

        s=180,

        marker=style["marker"],

        color=style["color"],

        edgecolors="black",

        linewidths=1.2,

        zorder=5,

        label=f"Team {team_name}"
    )


    # Player ID
    for _, row in team.iterrows():

        ax.text(

            row[
                "pitch_x_final_m"
            ],

            row[
                "pitch_y_final_m"
            ],

            str(
                int(
                    row["track_id"]
                )
            ),

            ha="center",
            va="center",

            fontsize=9,

            color="white",

            fontweight="bold",

            zorder=6
        )


    # ========================================================
    # Centroid
    # ========================================================

    centroid_x = float(
        np.mean(x)
    )

    centroid_y = float(
        np.mean(y)
    )


    ax.scatter(

        centroid_x,
        centroid_y,

        marker="*",

        s=400,

        color=style["color"],

        edgecolors="black",

        linewidths=1.5,

        zorder=7
    )


    # ========================================================
    # Robust length / width boundaries
    # ========================================================

    x10 = np.percentile(
        x,
        10
    )

    x90 = np.percentile(
        x,
        90
    )

    y10 = np.percentile(
        y,
        10
    )

    y90 = np.percentile(
        y,
        90
    )


    # Length indicator
    ax.plot(

        [
            x10,
            x90
        ],

        [
            centroid_y,
            centroid_y
        ],

        linestyle="--",

        linewidth=2,

        color=style["color"],

        alpha=0.8
    )


    # Width indicator
    ax.plot(

        [
            centroid_x,
            centroid_x
        ],

        [
            y10,
            y90
        ],

        linestyle="--",

        linewidth=2,

        color=style["color"],

        alpha=0.8
    )


    length = (
        x90 - x10
    )

    width = (
        y90 - y10
    )


    compactness = np.mean(

        np.sqrt(

            (x - centroid_x) ** 2

            +

            (y - centroid_y) ** 2
        )
    )


    print("")
    print(
        f"Team {team_name}"
    )

    print(
        "Centroid:",
        f"({centroid_x:.2f}, "
        f"{centroid_y:.2f})"
    )

    print(
        "Length:",
        f"{length:.2f} m"
    )

    print(
        "Width:",
        f"{width:.2f} m"
    )

    print(
        "Compactness:",
        f"{compactness:.2f} m"
    )


# ============================================================
# Title
# ============================================================

ax.set_title(

    "Representative 10v10 Team Shape\n"
    f"Frame {REP_FRAME} | "
    f"{REP_TIME:.2f} seconds"
)


ax.legend(
    loc="upper right"
)


plt.tight_layout()


plt.savefig(
    OUTPUT_PATH,
    dpi=220,
    bbox_inches="tight"
)


plt.close()


print("")
print("========================================")
print("DONE")
print("========================================")

print("")
print(
    "Saved:",
    OUTPUT_PATH
)
