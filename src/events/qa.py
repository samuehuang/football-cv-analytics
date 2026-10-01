import argparse

import cv2
import numpy as np
import pandas as pd


def text(v, default="-"):
    if pd.isna(v):
        return default

    s = str(v).strip()

    if (
        s == ""
        or
        s.lower() in {"nan", "none"}
    ):
        return default

    return s


def track_name(
    team,
    track_id,
):
    team = text(
        team,
        None,
    )

    if (
        team is None
        or
        pd.isna(track_id)
    ):
        return "NONE"

    try:
        track_id = int(
            float(track_id)
        )

    except Exception:
        return "NONE"

    return (
        f"{team}#{track_id}"
    )


def draw_text(
    frame,
    message,
    x,
    y,
    color=(255, 255, 255),
    scale=0.58,
    thickness=1,
):
    cv2.putText(
        frame,
        message,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        message,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def event_label(row):

    event_type = text(
        row.get(
            "event_type"
        )
    )

    subtype = text(
        row.get(
            "subtype"
        )
    )

    team = text(
        row.get(
            "team"
        ),
        None,
    )

    player = row.get(
        "player_id"
    )

    target_team = text(
        row.get(
            "target_team"
        ),
        None,
    )

    target_player = row.get(
        "target_player_id"
    )


    if (
        event_type
        ==
        "Pass"
    ):

        source = track_name(
            team,
            player,
        )

        target = track_name(
            target_team,
            target_player,
        )

        return (
            f"PASS  {source} -> {target}"
        )


    if (
        event_type
        ==
        "Reception"
    ):

        receiver = track_name(
            team,
            player,
        )

        return (
            f"RECEPTION  {receiver}"
        )


    if (
        event_type
        ==
        "Dribble"
    ):

        player_name = track_name(
            team,
            player,
        )

        return (
            f"DRIBBLE GAP  {player_name}"
        )


    if (
        event_type
        ==
        "AirTouch"
    ):

        player_name = track_name(
            team,
            player,
        )

        return (
            f"AIR TOUCH  {player_name}"
        )


    if (
        event_type
        ==
        "AerialDuel"
    ):

        return (
            "AERIAL DUEL"
        )


    if (
        event_type
        ==
        "Turnover"
    ):

        player_name = track_name(
            team,
            player,
        )

        return (
            f"TURNOVER  {player_name}"
        )


    return (
        f"{event_type} | {subtype}"
    )


def event_color(
    event_type,
):

    if event_type == "Pass":
        return (
            80,
            255,
            80,
        )

    if event_type == "Reception":
        return (
            255,
            255,
            80,
        )

    if event_type == "AerialDuel":
        return (
            0,
            165,
            255,
        )

    if event_type == "AirTouch":
        return (
            255,
            100,
            255,
        )

    if event_type == "Dribble":
        return (
            255,
            255,
            0,
        )

    if event_type == "Turnover":
        return (
            0,
            0,
            255,
        )

    return (
        255,
        255,
        255,
    )


def main():

    parser = (
        argparse.ArgumentParser()
    )


    parser.add_argument(
        "--video",
        default=(
            "videos/match.mp4"
        ),
    )


    parser.add_argument(
        "--frames",
        default=(
            "outputs/"
            "possession_v6_frames.csv"
        ),
    )


    parser.add_argument(
        "--events",
        default=(
            "outputs/"
            "football_events_v1.csv"
        ),
    )


    parser.add_argument(
        "--output",
        default=(
            "outputs/"
            "football_event_qa_v1.mp4"
        ),
    )


    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
    )


    parser.add_argument(
        "--end-frame",
        type=int,
        default=-1,
    )


    args = (
        parser.parse_args()
    )


    # ========================================================
    # LOAD DATA
    # ========================================================

    frames = pd.read_csv(
        args.frames
    )


    events = pd.read_csv(
        args.events
    )


    frames[
        "frame"
    ] = pd.to_numeric(
        frames[
            "frame"
        ],
        errors="coerce",
    )


    frames = (
        frames
        .dropna(
            subset=[
                "frame"
            ]
        )
        .copy()
    )


    frames[
        "frame"
    ] = (
        frames[
            "frame"
        ]
        .astype(int)
    )


    frame_data = (
        frames
        .set_index(
            "frame"
        )
    )


    # ========================================================
    # EVENT LOOKUP
    # ========================================================

    important_types = {
        "Pass",
        "Reception",
        "Turnover",
        "AerialDuel",
        "AirTouch",
        "Dribble",
    }


    active_events = {}


    for _, event in (
        events.iterrows()
    ):

        event_type = text(
            event.get(
                "event_type"
            )
        )


        if (
            event_type
            not in
            important_types
        ):

            continue


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


        event_dict = (
            event.to_dict()
        )


        for frame_id in range(
            start,
            end + 1,
        ):

            active_events.setdefault(
                frame_id,
                [],
            )

            active_events[
                frame_id
            ].append(
                event_dict
            )


    # ========================================================
    # VIDEO
    # ========================================================

    cap = cv2.VideoCapture(
        args.video
    )


    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video: "
            f"{args.video}"
        )


    fps = cap.get(
        cv2.CAP_PROP_FPS
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


    total = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    start_frame = max(
        0,
        args.start_frame,
    )


    if (
        args.end_frame
        <
        0
    ):

        end_frame = (
            total - 1
        )

    else:

        end_frame = min(
            args.end_frame,
            total - 1,
        )


    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame,
    )


    writer = cv2.VideoWriter(
        args.output,

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        fps,

        (
            width,
            height,
        ),
    )


    if not writer.isOpened():

        raise RuntimeError(
            f"Cannot create video: "
            f"{args.output}"
        )


    print(
        f"Video FPS: {fps:.2f}"
    )


    print(
        f"Rendering frames "
        f"{start_frame} -> {end_frame}"
    )


    # ========================================================
    # FRAME LOOP
    # ========================================================

    frame_id = (
        start_frame
    )


    while (
        frame_id
        <=
        end_frame
    ):

        ok, image = (
            cap.read()
        )


        if not ok:

            break


        # ====================================================
        # FRAME STATE
        # ====================================================

        if (
            frame_id
            in
            frame_data.index
        ):

            row = (
                frame_data.loc[
                    frame_id
                ]
            )


            if isinstance(
                row,
                pd.DataFrame,
            ):

                row = (
                    row.iloc[0]
                )


            possession = text(
                row.get(
                    "possession_v6"
                )
            )


            ball_state = text(
                row.get(
                    "ball_state_v6"
                )
            )


            controller = track_name(
                row.get(
                    "controller_team_v6"
                ),
                row.get(
                    "controller_id_v6"
                ),
            )


            last_touch = track_name(
                row.get(
                    "last_touch_team_v6"
                ),
                row.get(
                    "last_touch_id_v6"
                ),
            )


            confidence = text(
                row.get(
                    "confidence_v6"
                )
            )


            reason = text(
                row.get(
                    "decision_reason_v6"
                )
            )


            motion = text(
                row.get(
                    "motion_state_v1"
                )
            )


        else:

            possession = "-"
            ball_state = "-"
            controller = "-"
            last_touch = "-"
            confidence = "-"
            reason = "-"
            motion = "-"


        # ====================================================
        # INFO PANEL
        # ====================================================

        overlay = (
            image.copy()
        )


        panel_width = min(
            670,
            width - 20,
        )


        cv2.rectangle(
            overlay,
            (
                10,
                10,
            ),
            (
                panel_width,
                255,
            ),
            (
                0,
                0,
                0,
            ),
            -1,
        )


        image = cv2.addWeighted(
            overlay,
            0.62,
            image,
            0.38,
            0,
        )


        current_time = (
            frame_id
            /
            fps
        )


        draw_text(
            image,
            (
                f"FRAME {frame_id} "
                f"| {current_time:.2f}s"
            ),
            25,
            38,
            scale=0.66,
            thickness=2,
        )


        # possession color

        if possession == "A":

            possession_color = (
                80,
                255,
                80,
            )


        elif possession == "B":

            possession_color = (
                255,
                180,
                80,
            )


        elif possession == "Contested":

            possession_color = (
                0,
                165,
                255,
            )


        else:

            possession_color = (
                190,
                190,
                190,
            )


        draw_text(
            image,
            (
                "POSSESSION: "
                f"{possession}"
            ),
            25,
            72,
            color=possession_color,
            scale=0.70,
            thickness=2,
        )


        draw_text(
            image,
            (
                "BALL STATE: "
                f"{ball_state}"
            ),
            25,
            105,
        )


        draw_text(
            image,
            (
                "MOTION: "
                f"{motion}"
            ),
            25,
            135,
        )


        draw_text(
            image,
            (
                "CONTROLLER: "
                f"{controller}"
            ),
            25,
            165,
        )


        draw_text(
            image,
            (
                "LAST TOUCH: "
                f"{last_touch}"
            ),
            25,
            195,
        )


        draw_text(
            image,
            (
                f"CONF: {confidence} "
                f"| {reason}"
            ),
            25,
            225,
            color=(
                200,
                200,
                200,
            ),
            scale=0.48,
        )


        # ====================================================
        # ACTIVE EVENTS
        # ====================================================

        events_now = (
            active_events.get(
                frame_id,
                [],
            )
        )


        event_y = (
            height - 35
        )


        for event in reversed(
            events_now[
                -3:
            ]
        ):

            event_type = text(
                event.get(
                    "event_type"
                )
            )


            label = event_label(
                event
            )


            draw_text(
                image,
                label,
                25,
                event_y,
                color=event_color(
                    event_type
                ),
                scale=0.72,
                thickness=2,
            )


            event_y -= 36


        # ====================================================
        # SPECIAL MARKER
        # ====================================================

        if events_now:

            cv2.circle(
                image,
                (
                    width - 45,
                    45,
                ),
                16,
                (
                    0,
                    0,
                    255,
                ),
                -1,
            )


            draw_text(
                image,
                "EVENT",
                width - 125,
                52,
                color=(
                    255,
                    255,
                    255,
                ),
                scale=0.55,
                thickness=2,
            )


        writer.write(
            image
        )


        frame_id += 1


    cap.release()

    writer.release()


    print("")

    print(
        "Event QA video saved:"
    )

    print(
        args.output
    )


if __name__ == "__main__":

    main()
