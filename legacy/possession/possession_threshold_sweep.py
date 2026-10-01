import pandas as pd
import numpy as np


INPUT_CSV = "outputs/possession_v1_frames.csv"

df = pd.read_csv(INPUT_CSV)


CONTROL_RADII = [
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
]

CONTESTED_RADIUS_EXTRA = 0.5
CONTESTED_MARGIN = 0.75


results = []


for control_radius in CONTROL_RADII:

    contested_radius = (
        control_radius
        +
        CONTESTED_RADIUS_EXTRA
    )

    states = []


    for _, row in df.iterrows():

        # --------------------------------------------
        # Ball unavailable
        # --------------------------------------------

        if not bool(row["ball_usable"]):

            states.append("Unknown")
            continue


        dA = row["nearest_A_distance_m"]
        dB = row["nearest_B_distance_m"]


        if pd.isna(dA) and pd.isna(dB):

            states.append("Unknown")
            continue


        # --------------------------------------------
        # Contested
        # --------------------------------------------

        if (
            pd.notna(dA)
            and
            pd.notna(dB)
            and
            dA <= contested_radius
            and
            dB <= contested_radius
            and
            abs(dA - dB)
            <= CONTESTED_MARGIN
        ):

            states.append("Contested")
            continue


        # --------------------------------------------
        # Team A
        # --------------------------------------------

        if (
            pd.notna(dA)
            and
            dA <= control_radius
            and
            (
                pd.isna(dB)
                or
                dA < dB
            )
        ):

            states.append("A")
            continue


        # --------------------------------------------
        # Team B
        # --------------------------------------------

        if (
            pd.notna(dB)
            and
            dB <= control_radius
            and
            (
                pd.isna(dA)
                or
                dB < dA
            )
        ):

            states.append("B")
            continue


        states.append("Unknown")


    states = pd.Series(states)


    A = int((states == "A").sum())
    B = int((states == "B").sum())
    contested = int(
        (states == "Contested").sum()
    )
    unknown = int(
        (states == "Unknown").sum()
    )


    total = len(states)

    known_team = A + B


    known_coverage = (
        known_team
        /
        total
        *
        100.0
    )


    resolved_coverage = (
        (
            A
            +
            B
            +
            contested
        )
        /
        total
        *
        100.0
    )


    if known_team > 0:

        A_share = (
            A
            /
            known_team
            *
            100.0
        )

        B_share = (
            B
            /
            known_team
            *
            100.0
        )

    else:

        A_share = 0.0
        B_share = 0.0


    results.append(
        {
            "control_radius_m":
                control_radius,

            "A_frames":
                A,

            "B_frames":
                B,

            "contested_frames":
                contested,

            "unknown_frames":
                unknown,

            "known_team_coverage_pct":
                known_coverage,

            "resolved_coverage_pct":
                resolved_coverage,

            "A_share_known_pct":
                A_share,

            "B_share_known_pct":
                B_share,
        }
    )


result = pd.DataFrame(
    results
)


print("")
print(
    "========================================"
)

print(
    "POSSESSION THRESHOLD SWEEP"
)

print(
    "========================================"
)

print("")

print(
    result.round(1).to_string(
        index=False
    )
)


result.to_csv(
    "outputs/possession_threshold_sweep.csv",
    index=False
)


# ============================================================
# Unknown diagnosis
# ============================================================

ball_missing = (
    ~df["ball_usable"]
    .astype(str)
    .str.lower()
    .isin(["true", "1"])
)


print("")
print(
    "========================================"
)

print(
    "UNKNOWN DIAGNOSIS"
)

print(
    "========================================"
)

print("")

print(
    "Tactical frames:",
    len(df)
)

print(
    "Ball unavailable:",
    int(ball_missing.sum())
)


nearest = df[
    "nearest_distance_m"
].dropna()


if len(nearest) > 0:

    print(
        "Ball/player distance median:",
        f"{nearest.median():.2f} m"
    )

    print(
        "P25:",
        f"{nearest.quantile(0.25):.2f} m"
    )

    print(
        "P75:",
        f"{nearest.quantile(0.75):.2f} m"
    )

    print(
        "P90:",
        f"{nearest.quantile(0.90):.2f} m"
    )

    print(
        "P95:",
        f"{nearest.quantile(0.95):.2f} m"
    )
