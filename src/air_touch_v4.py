import os
import cv2
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

VIDEO_PATH = "videos/match.mp4"

V3_INTERACTION_CSV = "outputs/air_touch_v3_interactions.csv"
V3_EVENT_CSV = "outputs/air_touch_v3_events.csv"

OUTPUT_EVENT_CSV = "outputs/air_touch_v4_events.csv"
OUTPUT_AIR_TOUCH_CSV = "outputs/air_touch_v4_confirmed_air_touch.csv"
OUTPUT_CONTEST_CSV = "outputs/air_touch_v4_aerial_contests.csv"
OUTPUT_VIDEO = "outputs/air_touch_v4_qa.mp4"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# CROSS-PLAYER ARBITRATION
# ============================================================

# Two AirTouch candidates whose temporal windows overlap
# or are only 1 frame apart may belong to the same physical
# interaction.
ARBITRATION_FRAME_GAP = 1


# ============================================================
# CLEAR PLAYER WINNER
#
# If two opposite-team candidates exist, we only choose a
# unique player if spatial evidence is VERY clearly better.
#
# Otherwise:
# -> AerialContest
# -> Last touch unknown
# ============================================================

CLEAR_WINNER_DISTANCE_RATIO = 0.55
CLEAR_WINNER_DISTANCE_MARGIN_NORM = 0.18


# ============================================================
# REGRESSION CASES
# ============================================================

# A #1 + B #14 aerial duel
CONTEST_FRAME = 172

# A #6 ground reception / bounce
RECEPTION_FRAME = 203
RECEPTION_TEAM = "A"
RECEPTION_TRACK_ID = 6

# B #10 real header
HEADER_FRAME = 281
HEADER_TEAM = "B"
HEADER_TRACK_ID = 10


# ============================================================
# HELPERS
# ============================================================

def as_bool(value):

    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def same_player_id(a, b):

    if (
        pd.isna(a)
        or
        pd.isna(b)
    ):
        return False

    return abs(
        float(a)
        -
        float(b)
    ) < 0.1


def participant_label(team, track_id):

    return (
        f"{team}"
        f"#{int(float(track_id))}"
    )


def parse_event_ids(value):

    if pd.isna(value):
        return []

    ids = []

    for part in str(value).split(";"):

        part = part.strip()

        if not part:
            continue

        ids.append(
            int(float(part))
        )

    return ids


# ============================================================
# LOAD V3
# ============================================================

print("")
print("Loading AirTouch V3 outputs...")


interactions = pd.read_csv(
    V3_INTERACTION_CSV
)

events = pd.read_csv(
    V3_EVENT_CSV
)


# ============================================================
# NORMALIZE
# ============================================================

interaction_numeric = [
    "frame",
    "event_id",
    "track_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "upper_distance_px",
    "ball_speed_mps",
    "direction_change_deg",
]


for column in interaction_numeric:

    if column in interactions.columns:

        interactions[column] = pd.to_numeric(
            interactions[column],
            errors="coerce"
        )


event_numeric = [
    "event_id",
    "track_id",
    "start_frame",
    "end_frame",
    "start_time_sec",
    "end_time_sec",
    "max_direction_change_deg",
    "max_speed_change_ratio",
    "max_ball_speed_mps",
    "min_upper_distance_px",
]


for column in event_numeric:

    if column in events.columns:

        events[column] = pd.to_numeric(
            events[column],
            errors="coerce"
        )


events["confirmed_air_touch"] = (
    events["confirmed_air_touch"]
    .apply(as_bool)
)


events["has_very_strong_frame"] = (
    events["has_very_strong_frame"]
    .apply(as_bool)
)


events["has_reception_evidence"] = (
    events["has_reception_evidence"]
    .apply(as_bool)
)


# ============================================================
# NORMALIZED UPPER-BODY DISTANCE
#
# Raw pixel distance changes with player scale.
# Normalize by player bbox height.
# ============================================================

interactions["bbox_height"] = (
    interactions["bbox_y2"]
    -
    interactions["bbox_y1"]
)


interactions["upper_distance_norm"] = (

    interactions["upper_distance_px"]

    /

    interactions["bbox_height"].clip(
        lower=1.0
    )
)


event_distance_norm = (

    interactions
    .groupby("event_id")[
        "upper_distance_norm"
    ]
    .min()
)


events["min_upper_distance_norm"] = (
    events["event_id"]
    .map(
        event_distance_norm
    )
)


# ============================================================
# SPLIT V3 EVENTS
# ============================================================

air_candidates = events[
    events["confirmed_air_touch"]
    ==
    True
].copy()


non_air_events = events[
    events["confirmed_air_touch"]
    ==
    False
].copy()


air_candidates = (
    air_candidates
    .sort_values(
        [
            "start_frame",
            "end_frame",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# GROUP TEMPORALLY OVERLAPPING AIR-TOUCH EVENTS
# ============================================================

candidate_groups = []

current_group = []
current_end = None


for _, row in air_candidates.iterrows():

    start_frame = int(
        row["start_frame"]
    )

    end_frame = int(
        row["end_frame"]
    )


    if not current_group:

        current_group = [
            row
        ]

        current_end = end_frame

        continue


    if (
        start_frame
        <=
        current_end
        +
        ARBITRATION_FRAME_GAP
    ):

        current_group.append(
            row
        )

        current_end = max(
            current_end,
            end_frame
        )


    else:

        candidate_groups.append(
            current_group
        )

        current_group = [
            row
        ]

        current_end = end_frame


if current_group:

    candidate_groups.append(
        current_group
    )


# ============================================================
# PHYSICAL EVENT OUTPUT
# ============================================================

physical_rows = []


# ============================================================
# FIRST:
# preserve V3 non-AirTouch events
# ============================================================

for _, row in non_air_events.iterrows():

    physical_rows.append(
        {
            "start_frame":
                int(
                    row[
                        "start_frame"
                    ]
                ),

            "end_frame":
                int(
                    row[
                        "end_frame"
                    ]
                ),

            "start_time_sec":
                float(
                    row[
                        "start_time_sec"
                    ]
                ),

            "end_time_sec":
                float(
                    row[
                        "end_time_sec"
                    ]
                ),

            "final_class":
                str(
                    row[
                        "event_class"
                    ]
                ),

            "last_touch_team":
                None,

            "last_touch_track_id":
                np.nan,

            "participants":
                participant_label(
                    row[
                        "team"
                    ],
                    row[
                        "track_id"
                    ]
                ),

            "source_v3_event_ids":
                str(
                    int(
                        row[
                            "event_id"
                        ]
                    )
                ),

            "candidate_count":
                1,

            "max_ball_speed_mps":
                row[
                    "max_ball_speed_mps"
                ],

            "max_direction_change_deg":
                row[
                    "max_direction_change_deg"
                ],

            "min_upper_distance_norm":
                row[
                    "min_upper_distance_norm"
                ],

            "decision_reason":
                str(
                    row[
                        "decision_reason"
                    ]
                ),
        }
    )


# ============================================================
# CROSS-PLAYER ARBITRATION
# ============================================================

for group in candidate_groups:

    group_df = pd.DataFrame(
        group
    ).copy()


    group_df = (
        group_df
        .sort_values(
            [
                "min_upper_distance_norm",
                "event_id",
            ],
            na_position="last"
        )
        .reset_index(
            drop=True
        )
    )


    start_frame = int(
        group_df[
            "start_frame"
        ].min()
    )

    end_frame = int(
        group_df[
            "end_frame"
        ].max()
    )


    start_time = float(
        group_df[
            "start_time_sec"
        ].min()
    )

    end_time = float(
        group_df[
            "end_time_sec"
        ].max()
    )


    participants = "|".join(
        [
            participant_label(
                row[
                    "team"
                ],
                row[
                    "track_id"
                ]
            )

            for _, row
            in group_df.iterrows()
        ]
    )


    source_event_ids = ";".join(
        [
            str(
                int(
                    event_id
                )
            )

            for event_id
            in group_df[
                "event_id"
            ].tolist()
        ]
    )


    teams = set(
        group_df[
            "team"
        ].astype(str)
    )


    max_ball_speed = (
        group_df[
            "max_ball_speed_mps"
        ].max()
    )


    max_direction = (
        group_df[
            "max_direction_change_deg"
        ].max()
    )


    candidate_count = len(
        group_df
    )


    # ========================================================
    # CASE 1:
    # only one candidate
    # ========================================================

    if candidate_count == 1:

        winner = group_df.iloc[0]


        final_class = (
            "AirTouch"
        )


        last_touch_team = (
            winner[
                "team"
            ]
        )


        last_touch_track_id = (
            winner[
                "track_id"
            ]
        )


        decision_reason = (
            "single_air_touch_candidate"
        )


        min_distance_norm = (
            winner[
                "min_upper_distance_norm"
            ]
        )


    # ========================================================
    # CASE 2:
    # multiple candidates but SAME team
    #
    # Team ownership is not ambiguous.
    # Pick nearest upper-body candidate as player identity.
    # ========================================================

    elif len(teams) == 1:

        winner = group_df.iloc[0]


        final_class = (
            "AirTouch"
        )


        last_touch_team = (
            winner[
                "team"
            ]
        )


        last_touch_track_id = (
            winner[
                "track_id"
            ]
        )


        decision_reason = (
            "same_team_overlap_choose_nearest"
        )


        min_distance_norm = (
            winner[
                "min_upper_distance_norm"
            ]
        )


    # ========================================================
    # CASE 3:
    # opposite-team candidates
    # ========================================================

    else:

        # ----------------------------------------------------
        # Rank by evidence class first:
        #
        # very strong = 2
        # strong      = 1
        # ----------------------------------------------------

        group_df[
            "_evidence_rank"
        ] = group_df[
            "has_very_strong_frame"
        ].apply(
            lambda value:
                2
                if as_bool(value)
                else 1
        )


        group_df = (
            group_df
            .sort_values(
                [
                    "_evidence_rank",
                    "min_upper_distance_norm",
                ],
                ascending=[
                    False,
                    True,
                ],
                na_position="last"
            )
            .reset_index(
                drop=True
            )
        )


        best = group_df.iloc[0]
        second = group_df.iloc[1]


        best_rank = int(
            best[
                "_evidence_rank"
            ]
        )


        second_rank = int(
            second[
                "_evidence_rank"
            ]
        )


        best_distance = (
            best[
                "min_upper_distance_norm"
            ]
        )


        second_distance = (
            second[
                "min_upper_distance_norm"
            ]
        )


        # ----------------------------------------------------
        # Different evidence strength:
        # VeryStrong beats ordinary Strong.
        # ----------------------------------------------------

        if (
            best_rank
            >
            second_rank
        ):

            final_class = (
                "AirTouch"
            )


            last_touch_team = (
                best[
                    "team"
                ]
            )


            last_touch_track_id = (
                best[
                    "track_id"
                ]
            )


            decision_reason = (
                "unique_stronger_cross_team_candidate"
            )


            min_distance_norm = (
                best_distance
            )


        # ----------------------------------------------------
        # Same evidence class.
        #
        # Only choose player if spatial evidence is VERY
        # clearly separated.
        # ----------------------------------------------------

        else:

            clear_spatial_winner = False


            if (
                pd.notna(
                    best_distance
                )

                and

                pd.notna(
                    second_distance
                )

                and

                second_distance
                >
                0
            ):

                ratio = (
                    best_distance
                    /
                    second_distance
                )


                margin = (
                    second_distance
                    -
                    best_distance
                )


                clear_spatial_winner = (

                    ratio
                    <=
                    CLEAR_WINNER_DISTANCE_RATIO

                    and

                    margin
                    >=
                    CLEAR_WINNER_DISTANCE_MARGIN_NORM
                )


            if clear_spatial_winner:

                final_class = (
                    "AirTouch"
                )


                last_touch_team = (
                    best[
                        "team"
                    ]
                )


                last_touch_track_id = (
                    best[
                        "track_id"
                    ]
                )


                decision_reason = (
                    "clear_spatial_winner_cross_team"
                )


                min_distance_norm = (
                    best_distance
                )


            else:

                final_class = (
                    "AerialContest"
                )


                last_touch_team = (
                    None
                )


                last_touch_track_id = (
                    np.nan
                )


                decision_reason = (
                    "cross_team_air_touch_ambiguity"
                )


                min_distance_norm = (
                    group_df[
                        "min_upper_distance_norm"
                    ].min()
                )


    physical_rows.append(
        {
            "start_frame":
                start_frame,

            "end_frame":
                end_frame,

            "start_time_sec":
                start_time,

            "end_time_sec":
                end_time,

            "final_class":
                final_class,

            "last_touch_team":
                last_touch_team,

            "last_touch_track_id":
                last_touch_track_id,

            "participants":
                participants,

            "source_v3_event_ids":
                source_event_ids,

            "candidate_count":
                candidate_count,

            "max_ball_speed_mps":
                max_ball_speed,

            "max_direction_change_deg":
                max_direction,

            "min_upper_distance_norm":
                min_distance_norm,

            "decision_reason":
                decision_reason,
        }
    )


# ============================================================
# FINAL EVENT TABLE
# ============================================================

final_events = pd.DataFrame(
    physical_rows
)


final_events = (
    final_events
    .sort_values(
        [
            "start_frame",
            "end_frame",
        ]
    )
    .reset_index(
        drop=True
    )
)


final_events[
    "physical_event_id"
] = np.arange(
    1,
    len(
        final_events
    )
    +
    1
)


# Put ID first.
column_order = [
    "physical_event_id",
    "start_frame",
    "end_frame",
    "start_time_sec",
    "end_time_sec",
    "final_class",
    "last_touch_team",
    "last_touch_track_id",
    "participants",
    "candidate_count",
    "max_ball_speed_mps",
    "max_direction_change_deg",
    "min_upper_distance_norm",
    "decision_reason",
    "source_v3_event_ids",
]


final_events = final_events[
    column_order
]


# ============================================================
# SAVE
# ============================================================

final_events.to_csv(
    OUTPUT_EVENT_CSV,
    index=False
)


confirmed_air_touch = final_events[
    final_events[
        "final_class"
    ]
    ==
    "AirTouch"
].copy()


aerial_contests = final_events[
    final_events[
        "final_class"
    ]
    ==
    "AerialContest"
].copy()


confirmed_air_touch.to_csv(
    OUTPUT_AIR_TOUCH_CSV,
    index=False
)


aerial_contests.to_csv(
    OUTPUT_CONTEST_CSV,
    index=False
)


# ============================================================
# REGRESSION LOOKUP
# ============================================================

def events_at_frame(frame):

    return final_events[

        (
            final_events[
                "start_frame"
            ]
            <=
            frame
        )

        &

        (
            final_events[
                "end_frame"
            ]
            >=
            frame
        )
    ]


# ============================================================
# REGRESSION 1:
# frame 172 must be AerialContest
# ============================================================

contest_rows = events_at_frame(
    CONTEST_FRAME
)


contest_pass = False


for _, row in contest_rows.iterrows():

    if (
        row[
            "final_class"
        ]
        ==
        "AerialContest"
    ):

        participants = str(
            row[
                "participants"
            ]
        )


        if (
            "A#1"
            in
            participants

            and

            "B#14"
            in
            participants
        ):

            contest_pass = True
            break


# ============================================================
# REGRESSION 2:
# frame 203 must remain ReceptionOrBounce
# ============================================================

reception_rows = events_at_frame(
    RECEPTION_FRAME
)


reception_pass = False


for _, row in reception_rows.iterrows():

    if (

        row[
            "final_class"
        ]
        ==
        "ReceptionOrBounce"

        and

        participant_label(
            RECEPTION_TEAM,
            RECEPTION_TRACK_ID
        )
        in
        str(
            row[
                "participants"
            ]
        )
    ):

        reception_pass = True
        break


# ============================================================
# REGRESSION 3:
# frame 281 must be AirTouch B #10
# ============================================================

header_rows = events_at_frame(
    HEADER_FRAME
)


header_pass = False


for _, row in header_rows.iterrows():

    if (

        row[
            "final_class"
        ]
        ==
        "AirTouch"

        and

        str(
            row[
                "last_touch_team"
            ]
        )
        ==
        HEADER_TEAM

        and

        same_player_id(
            row[
                "last_touch_track_id"
            ],
            HEADER_TRACK_ID
        )
    ):

        header_pass = True
        break


# ============================================================
# INTERACTION LOOKUP FOR QA VIDEO
# ============================================================

interaction_lookup = {}


for (
    frame,
    event_id
), group in interactions.groupby(
    [
        "frame",
        "event_id",
    ]
):

    interaction_lookup[
        (
            int(frame),
            int(event_id),
        )
    ] = group.copy()


final_events_by_frame = {}


for _, event in final_events.iterrows():

    start = int(
        event[
            "start_frame"
        ]
    )

    end = int(
        event[
            "end_frame"
        ]
    )


    for frame in range(
        start,
        end + 1
    ):

        final_events_by_frame.setdefault(
            frame,
            []
        ).append(
            event
        )


# ============================================================
# QA VIDEO
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)


if not cap.isOpened():

    raise RuntimeError(
        f"Cannot open video: "
        f"{VIDEO_PATH}"
    )


fps = float(
    cap.get(
        cv2.CAP_PROP_FPS
    )
)


width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)


height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)


writer = cv2.VideoWriter(

    OUTPUT_VIDEO,

    cv2.VideoWriter_fourcc(
        *"mp4v"
    ),

    fps,

    (
        width,
        height
    )
)


if not writer.isOpened():

    raise RuntimeError(
        f"Cannot create QA video: "
        f"{OUTPUT_VIDEO}"
    )


frame_idx = 0


print("")
print(
    "========================================"
)

print(
    "AIR TOUCH V4 QA VIDEO"
)

print(
    "========================================"
)

print("")


while True:

    ret, frame = cap.read()


    if not ret:

        break


    annotated = frame.copy()


    active_events = (
        final_events_by_frame.get(
            frame_idx,
            []
        )
    )


    y_overlay = 35


    for event in active_events:

        final_class = str(
            event[
                "final_class"
            ]
        )


        source_ids = parse_event_ids(
            event[
                "source_v3_event_ids"
            ]
        )


        # ====================================================
        # COLOR
        # ====================================================

        if (
            final_class
            ==
            "AirTouch"
        ):

            color = (
                0,
                255,
                0
            )


        elif (
            final_class
            ==
            "AerialContest"
        ):

            color = (
                255,
                0,
                255
            )


        elif (
            final_class
            ==
            "ReceptionOrBounce"
        ):

            color = (
                0,
                165,
                255
            )


        else:

            color = (
                180,
                180,
                180
            )


        # ====================================================
        # DRAW SOURCE PLAYER BOXES
        # ====================================================

        for source_event_id in source_ids:

            key = (
                frame_idx,
                source_event_id,
            )


            if (
                key
                not in
                interaction_lookup
            ):

                continue


            source_rows = (
                interaction_lookup[
                    key
                ]
            )


            for _, source_row in source_rows.iterrows():

                x1 = int(
                    source_row[
                        "bbox_x1"
                    ]
                )

                y1 = int(
                    source_row[
                        "bbox_y1"
                    ]
                )

                x2 = int(
                    source_row[
                        "bbox_x2"
                    ]
                )

                y2 = int(
                    source_row[
                        "bbox_y2"
                    ]
                )


                cv2.rectangle(
                    annotated,

                    (
                        x1,
                        y1
                    ),

                    (
                        x2,
                        y2
                    ),

                    color,

                    3
                )


                cv2.putText(
                    annotated,

                    participant_label(
                        source_row[
                            "team"
                        ],
                        source_row[
                            "track_id"
                        ]
                    ),

                    (
                        x1,
                        max(
                            22,
                            y1 - 8
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.48,

                    color,

                    2,

                    cv2.LINE_AA
                )


        # ====================================================
        # TOP EVENT LABEL
        # ====================================================

        if (
            final_class
            ==
            "AirTouch"
        ):

            event_text = (

                f"AIR TOUCH | "
                f"LAST TOUCH "
                f"{event['last_touch_team']} "
                f"#{int(event['last_touch_track_id'])}"
            )


        elif (
            final_class
            ==
            "AerialContest"
        ):

            event_text = (

                f"AERIAL CONTEST | "
                f"{event['participants']} | "
                f"LAST TOUCH UNKNOWN"
            )


        else:

            event_text = (

                f"{final_class} | "
                f"{event['participants']}"
            )


        cv2.putText(
            annotated,

            event_text,

            (
                20,
                y_overlay
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.62,

            color,

            2,

            cv2.LINE_AA
        )


        y_overlay += 30


    # ========================================================
    # FRAME/TIME
    # ========================================================

    cv2.putText(
        annotated,

        (
            f"AIR TOUCH V4 | "
            f"frame={frame_idx} | "
            f"time={frame_idx / fps:.2f}s"
        ),

        (
            20,
            height - 25
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (
            255,
            255,
            255
        ),

        2,

        cv2.LINE_AA
    )


    # ========================================================
    # REGRESSION BANNERS
    # ========================================================

    if (
        frame_idx
        ==
        CONTEST_FRAME
    ):

        text = (

            "PASS | A #1 + B #14 = AERIAL CONTEST"

            if contest_pass

            else

            "FAIL | FRAME 172 SHOULD BE AERIAL CONTEST"
        )


        banner_color = (

            (
                0,
                255,
                0
            )

            if contest_pass

            else

            (
                0,
                0,
                255
            )
        )


        cv2.rectangle(
            annotated,

            (
                20,
                90
            ),

            (
                850,
                140
            ),

            banner_color,

            -1
        )


        cv2.putText(
            annotated,

            text,

            (
                35,
                123
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.62,

            (
                0,
                0,
                0
            ),

            2,

            cv2.LINE_AA
        )


    if (
        frame_idx
        ==
        RECEPTION_FRAME
    ):

        text = (

            "PASS | A #6 = RECEPTION / BOUNCE"

            if reception_pass

            else

            "FAIL | A #6 SHOULD NOT BE AIR TOUCH"
        )


        banner_color = (

            (
                0,
                255,
                0
            )

            if reception_pass

            else

            (
                0,
                0,
                255
            )
        )


        cv2.rectangle(
            annotated,

            (
                20,
                90
            ),

            (
                820,
                140
            ),

            banner_color,

            -1
        )


        cv2.putText(
            annotated,

            text,

            (
                35,
                123
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.62,

            (
                0,
                0,
                0
            ),

            2,

            cv2.LINE_AA
        )


    if (
        frame_idx
        ==
        HEADER_FRAME
    ):

        text = (

            "PASS | B #10 = AIR TOUCH"

            if header_pass

            else

            "FAIL | B #10 SHOULD BE AIR TOUCH"
        )


        banner_color = (

            (
                0,
                255,
                0
            )

            if header_pass

            else

            (
                0,
                0,
                255
            )
        )


        cv2.rectangle(
            annotated,

            (
                20,
                90
            ),

            (
                720,
                140
            ),

            banner_color,

            -1
        )


        cv2.putText(
            annotated,

            text,

            (
                35,
                123
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.62,

            (
                0,
                0,
                0
            ),

            2,

            cv2.LINE_AA
        )


    writer.write(
        annotated
    )


    frame_idx += 1


cap.release()
writer.release()


# ============================================================
# SUMMARY
# ============================================================

print("")
print(
    "========================================"
)

print(
    "AIR TOUCH V4 SUMMARY"
)

print(
    "========================================"
)

print("")

print(
    "V3 events:",
    len(
        events
    )
)

print(
    "V4 physical events:",
    len(
        final_events
    )
)

print("")

print(
    "Final classes:"
)

print(
    final_events[
        "final_class"
    ]
    .value_counts()
    .to_string()
)


print("")

print(
    "REGRESSION 1 | "
    "A #1 + B #14 @ frame 172 -> AerialContest:",
    (
        "PASS"
        if contest_pass
        else
        "FAIL"
    )
)


print(
    "REGRESSION 2 | "
    "A #6 @ frame 203 -> ReceptionOrBounce:",
    (
        "PASS"
        if reception_pass
        else
        "FAIL"
    )
)


print(
    "REGRESSION 3 | "
    "B #10 @ frame 281 -> AirTouch:",
    (
        "PASS"
        if header_pass
        else
        "FAIL"
    )
)


print("")

display_columns = [
    "physical_event_id",
    "start_frame",
    "end_frame",
    "start_time_sec",
    "end_time_sec",
    "final_class",
    "last_touch_team",
    "last_touch_track_id",
    "participants",
    "candidate_count",
    "max_ball_speed_mps",
    "max_direction_change_deg",
    "min_upper_distance_norm",
    "decision_reason",
]


print(
    final_events[
        display_columns
    ]
    .round(2)
    .to_string(
        index=False
    )
)


print("")

print(
    "Final event CSV:",
    OUTPUT_EVENT_CSV
)

print(
    "Confirmed AirTouch CSV:",
    OUTPUT_AIR_TOUCH_CSV
)

print(
    "Aerial Contest CSV:",
    OUTPUT_CONTEST_CSV
)

print(
    "QA Video:",
    OUTPUT_VIDEO
)
