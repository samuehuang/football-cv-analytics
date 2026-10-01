import os

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

POSSESSION_V5_1_CSV = "outputs/possession_v5_1_frames.csv"
CONTROLLER_GATE_CSV = "outputs/controller_gate_v2.csv"

OUTPUT_FRAMES_CSV = "outputs/possession_v5_2_frames.csv"
OUTPUT_SUMMARY_CSV = "outputs/possession_v5_2_summary.csv"
OUTPUT_AUDIT_CSV = "outputs/possession_v5_2_audit.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# REGRESSION FRAMES
# ============================================================

FRAME_159 = 159      # original B long ball
FRAME_172 = 172      # A#1 + B#14 aerial contest
FRAME_175 = 175      # false controller during same duel

FRAME_203 = 203      # A#6 reception/bounce regression
FRAME_226 = 226      # false control: ball not controlled

FRAME_253 = 253      # aerial contest
FRAME_281 = 281      # B#10 header
FRAME_287 = 287      # fallback to V4

FRAME_650 = 650      # valid A#11 dribble


# ============================================================
# HELPERS
# ============================================================

def clean_team(value):

    if pd.isna(value):
        return None

    value = str(value).strip()

    if value in {
        "A",
        "B",
    }:
        return value

    return None


def clean_id(value):

    if pd.isna(value):
        return np.nan

    try:
        return float(value)

    except Exception:
        return np.nan


def clean_text(
    value,
    default=None,
):

    if pd.isna(value):
        return default

    text = str(value).strip()

    if (
        text == ""
        or
        text.lower()
        in {
            "none",
            "nan",
        }
    ):
        return default

    return text


def same_player_id(a, b):

    if (
        pd.isna(a)
        or
        pd.isna(b)
    ):
        return False

    return (
        abs(
            float(a)
            -
            float(b)
        )
        <
        0.1
    )


# ============================================================
# LOAD
# ============================================================

print("")
print("Loading Possession V5.1...")

possession = pd.read_csv(
    POSSESSION_V5_1_CSV
)


print("Loading Controller Gate V2...")

gate = pd.read_csv(
    CONTROLLER_GATE_CSV
)


# ============================================================
# NORMALIZE POSSESSION
# ============================================================

numeric_possession_columns = [
    "frame",
    "time_sec",

    "controller_id_v4",
    "controller_id_v5",

    "last_touch_id_v5",

    "ball_speed_mps",
]


for column in numeric_possession_columns:

    if column in possession.columns:

        possession[column] = pd.to_numeric(
            possession[column],
            errors="coerce",
        )


possession = possession.dropna(
    subset=[
        "frame"
    ]
).copy()


possession["frame"] = (
    possession["frame"]
    .astype(int)
)


possession = (
    possession
    .sort_values(
        "frame"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# FPS
# ============================================================

fps = 25.0


if (
    "time_sec"
    in
    possession.columns
):

    timing = (
        possession[
            [
                "frame",
                "time_sec",
            ]
        ]
        .dropna()
        .sort_values(
            "frame"
        )
    )


    if len(timing) >= 2:

        frame_diff = (
            timing[
                "frame"
            ]
            .diff()
        )


        time_diff = (
            timing[
                "time_sec"
            ]
            .diff()
        )


        valid = (

            frame_diff.notna()

            &

            time_diff.notna()

            &

            (
                frame_diff > 0
            )

            &

            (
                time_diff > 0
            )
        )


        if valid.any():

            values = (

                frame_diff[
                    valid
                ]

                /

                time_diff[
                    valid
                ]
            )


            values = values[
                np.isfinite(
                    values
                )
            ]


            if len(values) > 0:

                fps = float(
                    values.median()
                )


print(
    f"Estimated FPS: {fps:.2f}"
)


# ============================================================
# NORMALIZE CONTROLLER GATE
# ============================================================

numeric_gate_columns = [
    "frame",
    "time_sec",

    "controller_id",

    "controller_run_id",
    "controller_run_length",
    "controller_run_position",

    "ball_speed_mps",

    "bbox_height_px",
    "foot_distance_px",
    "foot_distance_norm",
    "ball_rel_y",
]


for column in numeric_gate_columns:

    if column in gate.columns:

        gate[column] = pd.to_numeric(
            gate[column],
            errors="coerce",
        )


gate = gate.dropna(
    subset=[
        "frame"
    ]
).copy()


gate["frame"] = (
    gate["frame"]
    .astype(int)
)


gate = (
    gate
    .sort_values(
        "frame"
    )
    .reset_index(
        drop=True
    )
)


gate_by_frame = (
    gate
    .set_index(
        "frame"
    )
)


# ============================================================
# OUTPUT
# ============================================================

rows = []
audit_rows = []


print("")
print(
    "========================================"
)

print(
    "POSSESSION V5.2"
)

print(
    "Controller Gate V2 integration"
)

print(
    "========================================"
)

print("")


# ============================================================
# MAIN LOOP
# ============================================================

for _, base in possession.iterrows():

    frame = int(
        base[
            "frame"
        ]
    )


    row = base.to_dict()


    # ========================================================
    # V5.1 BASELINE
    # ========================================================

    possession_v5 = clean_text(
        base.get(
            "possession_v5",
            None
        ),
        "Unknown",
    )


    ball_state_v5 = clean_text(
        base.get(
            "ball_state_v5",
            None
        ),
        "Unknown",
    )


    controller_team_v5 = clean_team(
        base.get(
            "controller_team_v5",
            None
        )
    )


    controller_id_v5 = clean_id(
        base.get(
            "controller_id_v5",
            np.nan
        )
    )


    last_touch_team_v5 = clean_team(
        base.get(
            "last_touch_team_v5",
            None
        )
    )


    last_touch_id_v5 = clean_id(
        base.get(
            "last_touch_id_v5",
            np.nan
        )
    )


    last_touch_type_v5 = clean_text(
        base.get(
            "last_touch_type_v5",
            None
        ),
        None,
    )


    source_v5 = clean_text(
        base.get(
            "source_v5",
            None
        ),
        "baseline_v4",
    )


    # ========================================================
    # DEFAULT V5.2 = V5.1
    # ========================================================

    possession_v5_2 = (
        possession_v5
    )


    ball_state_v5_2 = (
        ball_state_v5
    )


    controller_team_v5_2 = (
        controller_team_v5
    )


    controller_id_v5_2 = (
        controller_id_v5
    )


    last_touch_team_v5_2 = (
        last_touch_team_v5
    )


    last_touch_id_v5_2 = (
        last_touch_id_v5
    )


    last_touch_type_v5_2 = (
        last_touch_type_v5
    )


    source_v5_2 = (
        source_v5
    )


    controller_gate_status = None
    controller_gate_reason = None
    controller_gate_mode = None

    controller_gate_override = False


    # ========================================================
    # CONTROLLER GATE RESULT
    # ========================================================

    gate_row = None


    if (
        frame
        in
        gate_by_frame.index
    ):

        gate_row = gate_by_frame.loc[
            frame
        ]


        if isinstance(
            gate_row,
            pd.DataFrame
        ):

            gate_row = (
                gate_row.iloc[0]
            )


        controller_gate_status = (
            clean_text(
                gate_row.get(
                    "gate_status",
                    None
                ),
                None,
            )
        )


        controller_gate_reason = (
            clean_text(
                gate_row.get(
                    "gate_reason",
                    None
                ),
                None,
            )
        )


        controller_gate_mode = (
            clean_text(
                gate_row.get(
                    "control_mode",
                    None
                ),
                None,
            )
        )


    # ========================================================
    # ONLY REJECT OVERRIDES V5.1
    #
    # KEEP:
    #     preserve V5.1
    #
    # AMBIGUOUS:
    #     preserve V5.1
    #
    # UNVERIFIED:
    #     preserve V5.1
    #
    # REJECT:
    #     override controller
    # ========================================================

    if (

        controller_gate_status
        ==
        "REJECT"

        and

        ball_state_v5
        ==
        "Controlled"

    ):

        controller_gate_override = True


        # ====================================================
        # CASE 1:
        # CONTEST LOCK
        #
        # Example:
        # frame 175 / 176.
        #
        # We have high-confidence evidence that the aerial
        # duel is still active.
        # ====================================================

        if (
            controller_gate_reason
            ==
            "contest_lock"
        ):

            possession_v5_2 = (
                "Contested"
            )


            ball_state_v5_2 = (
                "AerialContest"
            )


            controller_team_v5_2 = None
            controller_id_v5_2 = np.nan


            last_touch_team_v5_2 = None
            last_touch_id_v5_2 = np.nan
            last_touch_type_v5_2 = (
                "ContestUnknown"
            )


            source_v5_2 = (
                "controller_gate_contest_lock"
            )


        # ====================================================
        # CASE 2:
        # HARD CONTROL GEOMETRY REJECTION
        #
        # Important:
        #
        # We know the existing Controller is not trustworthy,
        # but we do NOT have true 3D ball height.
        #
        # Therefore:
        #
        # Team possession / phase ownership:
        #     keep V5.1
        #
        # Controller:
        #     NONE
        #
        # Ball state:
        #     Uncontrolled
        #
        # We intentionally do NOT guess InTransit vs Loose.
        # ====================================================

        else:

            possession_v5_2 = (
                possession_v5
            )


            ball_state_v5_2 = (
                "Uncontrolled"
            )


            controller_team_v5_2 = None
            controller_id_v5_2 = np.nan


            # ------------------------------------------------
            # If LastTouch existed only because the rejected
            # controller was considered Controlled, remove it.
            # ------------------------------------------------

            rejected_same_as_last_touch = (

                last_touch_type_v5
                ==
                "Controlled"

                and

                clean_team(
                    last_touch_team_v5
                )
                ==
                clean_team(
                    controller_team_v5
                )

                and

                same_player_id(
                    last_touch_id_v5,
                    controller_id_v5,
                )
            )


            if rejected_same_as_last_touch:

                last_touch_team_v5_2 = None
                last_touch_id_v5_2 = np.nan
                last_touch_type_v5_2 = None


            source_v5_2 = (

                "controller_gate_reject_"
                +
                str(
                    controller_gate_reason
                )
            )


    # ========================================================
    # SAVE V5.2 FIELDS
    # ========================================================

    row[
        "possession_v5_2"
    ] = possession_v5_2


    row[
        "ball_state_v5_2"
    ] = ball_state_v5_2


    row[
        "controller_team_v5_2"
    ] = controller_team_v5_2


    row[
        "controller_id_v5_2"
    ] = controller_id_v5_2


    row[
        "last_touch_team_v5_2"
    ] = last_touch_team_v5_2


    row[
        "last_touch_id_v5_2"
    ] = last_touch_id_v5_2


    row[
        "last_touch_type_v5_2"
    ] = last_touch_type_v5_2


    row[
        "source_v5_2"
    ] = source_v5_2


    row[
        "controller_gate_status"
    ] = controller_gate_status


    row[
        "controller_gate_reason"
    ] = controller_gate_reason


    row[
        "controller_gate_mode"
    ] = controller_gate_mode


    row[
        "controller_gate_override"
    ] = controller_gate_override


    # ========================================================
    # EXTRA GATE METRICS FOR QA
    # ========================================================

    if gate_row is not None:

        row[
            "controller_gate_run_length"
        ] = gate_row.get(
            "controller_run_length",
            np.nan
        )


        row[
            "controller_gate_run_position"
        ] = gate_row.get(
            "controller_run_position",
            np.nan
        )


        row[
            "controller_gate_foot_distance_px"
        ] = gate_row.get(
            "foot_distance_px",
            np.nan
        )


        row[
            "controller_gate_foot_distance_norm"
        ] = gate_row.get(
            "foot_distance_norm",
            np.nan
        )


        row[
            "controller_gate_ball_rel_y"
        ] = gate_row.get(
            "ball_rel_y",
            np.nan
        )


    else:

        row[
            "controller_gate_run_length"
        ] = np.nan


        row[
            "controller_gate_run_position"
        ] = np.nan


        row[
            "controller_gate_foot_distance_px"
        ] = np.nan


        row[
            "controller_gate_foot_distance_norm"
        ] = np.nan


        row[
            "controller_gate_ball_rel_y"
        ] = np.nan


    rows.append(
        row
    )


    # ========================================================
    # AUDIT
    # ========================================================

    if controller_gate_override:

        audit_rows.append(
            {
                "frame":
                    frame,

                "time_sec":
                    base.get(
                        "time_sec",
                        frame / fps,
                    ),

                "possession_v5":
                    possession_v5,

                "possession_v5_2":
                    possession_v5_2,

                "ball_state_v5":
                    ball_state_v5,

                "ball_state_v5_2":
                    ball_state_v5_2,

                "controller_team_v5":
                    controller_team_v5,

                "controller_id_v5":
                    controller_id_v5,

                "controller_team_v5_2":
                    controller_team_v5_2,

                "controller_id_v5_2":
                    controller_id_v5_2,

                "last_touch_team_v5":
                    last_touch_team_v5,

                "last_touch_id_v5":
                    last_touch_id_v5,

                "last_touch_team_v5_2":
                    last_touch_team_v5_2,

                "last_touch_id_v5_2":
                    last_touch_id_v5_2,

                "gate_status":
                    controller_gate_status,

                "gate_reason":
                    controller_gate_reason,

                "gate_mode":
                    controller_gate_mode,

                "source_v5":
                    source_v5,

                "source_v5_2":
                    source_v5_2,

                "foot_distance_norm":
                    (
                        gate_row.get(
                            "foot_distance_norm",
                            np.nan,
                        )
                        if gate_row
                        is not None
                        else
                        np.nan
                    ),

                "ball_rel_y":
                    (
                        gate_row.get(
                            "ball_rel_y",
                            np.nan,
                        )
                        if gate_row
                        is not None
                        else
                        np.nan
                    ),
            }
        )


# ============================================================
# DATAFRAMES
# ============================================================

v5_2 = pd.DataFrame(
    rows
)


audit = pd.DataFrame(
    audit_rows
)


v5_2.to_csv(
    OUTPUT_FRAMES_CSV,
    index=False,
)


audit.to_csv(
    OUTPUT_AUDIT_CSV,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


for key, value in (
    v5_2[
        "possession_v5_2"
    ]
    .value_counts()
    .items()
):

    summary_rows.append(
        {
            "type":
                "possession",

            "name":
                key,

            "frames":
                int(value),

            "seconds":
                float(value)
                /
                fps,
        }
    )


for key, value in (
    v5_2[
        "ball_state_v5_2"
    ]
    .value_counts()
    .items()
):

    summary_rows.append(
        {
            "type":
                "ball_state",

            "name":
                key,

            "frames":
                int(value),

            "seconds":
                float(value)
                /
                fps,
        }
    )


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    OUTPUT_SUMMARY_CSV,
    index=False,
)


# ============================================================
# FRAME LOOKUP
# ============================================================

v5_2_by_frame = (
    v5_2
    .set_index(
        "frame"
    )
)


def get_frame(frame):

    if (
        frame
        not in
        v5_2_by_frame.index
    ):

        return None


    result = v5_2_by_frame.loc[
        frame
    ]


    if isinstance(
        result,
        pd.DataFrame
    ):

        result = result.iloc[0]


    return result


def print_frame(frame):

    result = get_frame(
        frame
    )


    print("")


    if result is None:

        print(
            f"frame {frame}: NOT FOUND"
        )

        return


    print(
        f"FRAME {frame} | "
        f"{result.get('time_sec', np.nan):.2f}s"
    )


    print(
        "  V5.1 | "
        f"possession={result.get('possession_v5')} | "
        f"state={result.get('ball_state_v5')} | "
        f"controller="
        f"{result.get('controller_team_v5')} "
        f"{result.get('controller_id_v5')} | "
        f"last_touch="
        f"{result.get('last_touch_team_v5')} "
        f"{result.get('last_touch_id_v5')}"
    )


    print(
        "  V5.2 | "
        f"possession={result.get('possession_v5_2')} | "
        f"state={result.get('ball_state_v5_2')} | "
        f"controller="
        f"{result.get('controller_team_v5_2')} "
        f"{result.get('controller_id_v5_2')} | "
        f"last_touch="
        f"{result.get('last_touch_team_v5_2')} "
        f"{result.get('last_touch_id_v5_2')} | "
        f"gate="
        f"{result.get('controller_gate_status')} / "
        f"{result.get('controller_gate_reason')}"
    )


# ============================================================
# REGRESSIONS
# ============================================================

# ------------------------------------------------------------
# 159
# Must remain original B long ball.
# ------------------------------------------------------------

r159 = get_frame(
    FRAME_159
)


reg159 = (

    r159 is not None

    and

    str(
        r159[
            "possession_v5_2"
        ]
    )
    ==
    "B"

    and

    str(
        r159[
            "ball_state_v5_2"
        ]
    )
    ==
    "InTransit"

    and

    clean_team(
        r159[
            "controller_team_v5_2"
        ]
    )
    is None
)


# ------------------------------------------------------------
# 172
# Physical aerial contest.
# ------------------------------------------------------------

r172 = get_frame(
    FRAME_172
)


reg172 = (

    r172 is not None

    and

    str(
        r172[
            "possession_v5_2"
        ]
    )
    ==
    "Contested"

    and

    str(
        r172[
            "ball_state_v5_2"
        ]
    )
    ==
    "AerialContest"

    and

    clean_team(
        r172[
            "controller_team_v5_2"
        ]
    )
    is None
)


# ------------------------------------------------------------
# 175
# Must now reject the false A#3 controller.
# ------------------------------------------------------------

r175 = get_frame(
    FRAME_175
)


reg175 = (

    r175 is not None

    and

    str(
        r175[
            "possession_v5_2"
        ]
    )
    ==
    "Contested"

    and

    str(
        r175[
            "ball_state_v5_2"
        ]
    )
    ==
    "AerialContest"

    and

    clean_team(
        r175[
            "controller_team_v5_2"
        ]
    )
    is None

    and

    str(
        r175[
            "controller_gate_reason"
        ]
    )
    ==
    "contest_lock"
)


# ------------------------------------------------------------
# 203
# Existing reception/bounce regression preserved.
# ------------------------------------------------------------

r203 = get_frame(
    FRAME_203
)


reg203 = (

    r203 is not None

    and

    str(
        r203[
            "possession_v5_2"
        ]
    )
    ==
    str(
        r203[
            "possession_v5"
        ]
    )

    and

    str(
        r203[
            "ball_state_v5_2"
        ]
    )
    ==
    str(
        r203[
            "ball_state_v5"
        ]
    )
)


# ------------------------------------------------------------
# 226
# False controller must be removed.
# ------------------------------------------------------------

r226 = get_frame(
    FRAME_226
)


reg226 = (

    r226 is not None

    and

    str(
        r226[
            "possession_v5_2"
        ]
    )
    ==
    "A"

    and

    str(
        r226[
            "ball_state_v5_2"
        ]
    )
    ==
    "Uncontrolled"

    and

    clean_team(
        r226[
            "controller_team_v5_2"
        ]
    )
    is None

    and

    str(
        r226[
            "controller_gate_status"
        ]
    )
    ==
    "REJECT"
)


# ------------------------------------------------------------
# 253
# Existing aerial contest preserved.
# ------------------------------------------------------------

r253 = get_frame(
    FRAME_253
)


reg253 = (

    r253 is not None

    and

    str(
        r253[
            "possession_v5_2"
        ]
    )
    ==
    "Contested"

    and

    str(
        r253[
            "ball_state_v5_2"
        ]
    )
    ==
    "AerialContest"

    and

    clean_team(
        r253[
            "controller_team_v5_2"
        ]
    )
    is None
)


# ------------------------------------------------------------
# 281
# B#10 header preserved.
# ------------------------------------------------------------

r281 = get_frame(
    FRAME_281
)


reg281 = (

    r281 is not None

    and

    str(
        r281[
            "possession_v5_2"
        ]
    )
    ==
    "B"

    and

    str(
        r281[
            "ball_state_v5_2"
        ]
    )
    ==
    "InTransit"

    and

    clean_team(
        r281[
            "controller_team_v5_2"
        ]
    )
    is None

    and

    clean_team(
        r281[
            "last_touch_team_v5_2"
        ]
    )
    ==
    "B"

    and

    same_player_id(
        r281[
            "last_touch_id_v5_2"
        ],
        10
    )
)


# ------------------------------------------------------------
# 287
# Fallback preserved.
# ------------------------------------------------------------

r287 = get_frame(
    FRAME_287
)


reg287 = (

    r287 is not None

    and

    str(
        r287[
            "possession_v5_2"
        ]
    )
    ==
    "A"

    and

    str(
        r287[
            "ball_state_v5_2"
        ]
    )
    ==
    "InTransit"
)


# ------------------------------------------------------------
# 650
# Valid dribble MUST remain controlled.
# ------------------------------------------------------------

r650 = get_frame(
    FRAME_650
)


reg650 = (

    r650 is not None

    and

    str(
        r650[
            "possession_v5_2"
        ]
    )
    ==
    "A"

    and

    str(
        r650[
            "ball_state_v5_2"
        ]
    )
    ==
    "Controlled"

    and

    clean_team(
        r650[
            "controller_team_v5_2"
        ]
    )
    ==
    "A"

    and

    same_player_id(
        r650[
            "controller_id_v5_2"
        ],
        11
    )

    and

    str(
        r650[
            "controller_gate_status"
        ]
    )
    ==
    "KEEP"
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("")
print(
    "========================================"
)

print(
    "POSSESSION V5.2 SUMMARY"
)

print(
    "========================================"
)

print("")

print(
    "Frames:",
    len(
        v5_2
    )
)

print(
    "FPS:",
    f"{fps:.2f}"
)


print("")

print(
    "FINAL POSSESSION"
)


print(
    v5_2[
        "possession_v5_2"
    ]
    .value_counts()
    .to_string()
)


print("")

print(
    "FINAL BALL STATE"
)


print(
    v5_2[
        "ball_state_v5_2"
    ]
    .value_counts()
    .to_string()
)


print("")

print(
    "Controller Gate statuses on audited Controlled frames:"
)


print(
    gate[
        "gate_status"
    ]
    .value_counts()
    .to_string()
)


print("")

print(
    "V5.2 controller overrides:",
    int(
        v5_2[
            "controller_gate_override"
        ]
        .sum()
    )
)


print("")

print(
    "Override reasons:"
)


if len(audit) > 0:

    print(
        audit[
            "gate_reason"
        ]
        .value_counts()
        .to_string()
    )

else:

    print(
        "No controller overrides."
    )


print("")

print(
    "REGRESSION CHECKS"
)


print(
    "frame 159 | "
    "B long ball preserved:",
    (
        "PASS"
        if reg159
        else
        "FAIL"
    )
)


print(
    "frame 172 | "
    "physical aerial contest preserved:",
    (
        "PASS"
        if reg172
        else
        "FAIL"
    )
)


print(
    "frame 175 | "
    "false A#3 controller rejected:",
    (
        "PASS"
        if reg175
        else
        "FAIL"
    )
)


print(
    "frame 203 | "
    "A#6 reception/bounce preserved:",
    (
        "PASS"
        if reg203
        else
        "FAIL"
    )
)


print(
    "frame 226 | "
    "false A#9 controller rejected:",
    (
        "PASS"
        if reg226
        else
        "FAIL"
    )
)


print(
    "frame 253 | "
    "aerial contest preserved:",
    (
        "PASS"
        if reg253
        else
        "FAIL"
    )
)


print(
    "frame 281 | "
    "B#10 AirTouch preserved:",
    (
        "PASS"
        if reg281
        else
        "FAIL"
    )
)


print(
    "frame 287 | "
    "post-contest fallback preserved:",
    (
        "PASS"
        if reg287
        else
        "FAIL"
    )
)


print(
    "frame 650 | "
    "valid A#11 dribble preserved:",
    (
        "PASS"
        if reg650
        else
        "FAIL"
    )
)


# ============================================================
# DETAILED REGRESSION FRAMES
# ============================================================

for frame in [
    FRAME_159,
    FRAME_172,
    FRAME_175,
    FRAME_203,
    FRAME_226,
    FRAME_253,
    FRAME_281,
    FRAME_287,
    FRAME_650,
]:

    print_frame(
        frame
    )


# ============================================================
# OVERRIDE TABLE
# ============================================================

print("")
print(
    "========================================"
)

print(
    "CONTROLLER OVERRIDES"
)

print(
    "========================================"
)

print("")


if len(audit) > 0:

    columns = [
        "frame",
        "time_sec",

        "possession_v5",
        "possession_v5_2",

        "ball_state_v5",
        "ball_state_v5_2",

        "controller_team_v5",
        "controller_id_v5",

        "controller_team_v5_2",
        "controller_id_v5_2",

        "gate_reason",

        "foot_distance_norm",
        "ball_rel_y",
    ]


    print(
        audit[
            columns
        ]
        .round(3)
        .to_string(
            index=False
        )
    )


# ============================================================
# IMPORTANT WINDOWS
# ============================================================

print("")
print(
    "========================================"
)

print(
    "WINDOW 1 | AERIAL DUEL"
)

print(
    "========================================"
)

print("")


window1 = v5_2[
    (
        v5_2[
            "frame"
        ]
        >=
        170
    )

    &

    (
        v5_2[
            "frame"
        ]
        <=
        178
    )
][
    [
        "frame",
        "time_sec",

        "possession_v5_2",
        "ball_state_v5_2",

        "controller_team_v5_2",
        "controller_id_v5_2",

        "controller_gate_status",
        "controller_gate_reason",

        "source_v5_2",
    ]
]


print(
    window1.to_string(
        index=False
    )
)


print("")
print(
    "========================================"
)

print(
    "WINDOW 2 | FALSE AIRBORNE CONTROL"
)

print(
    "========================================"
)

print("")


window2 = v5_2[
    (
        v5_2[
            "frame"
        ]
        >=
        223
    )

    &

    (
        v5_2[
            "frame"
        ]
        <=
        228
    )
][
    [
        "frame",
        "time_sec",

        "possession_v5_2",
        "ball_state_v5_2",

        "controller_team_v5_2",
        "controller_id_v5_2",

        "controller_gate_status",
        "controller_gate_reason",

        "source_v5_2",
    ]
]


print(
    window2.to_string(
        index=False
    )
)


print("")
print(
    "========================================"
)

print(
    "WINDOW 3 | VALID DRIBBLE"
)

print(
    "========================================"
)

print("")


window3 = v5_2[
    (
        v5_2[
            "frame"
        ]
        >=
        647
    )

    &

    (
        v5_2[
            "frame"
        ]
        <=
        651
    )
][
    [
        "frame",
        "time_sec",

        "possession_v5_2",
        "ball_state_v5_2",

        "controller_team_v5_2",
        "controller_id_v5_2",

        "controller_gate_status",
        "controller_gate_reason",
        "controller_gate_mode",

        "source_v5_2",
    ]
]


print(
    window3.to_string(
        index=False
    )
)


print("")

print(
    "Frames CSV:",
    OUTPUT_FRAMES_CSV
)

print(
    "Summary CSV:",
    OUTPUT_SUMMARY_CSV
)

print(
    "Audit CSV:",
    OUTPUT_AUDIT_CSV
)
