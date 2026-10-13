import csv
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from event_windows import analyze_event_windows


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


class EventWindowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.events = self.root / "data/football_events_v1.csv"
        self.spacing = (
            self.root / "analytics/outputs/"
            "inter_team_spatial_metrics_frame_identity_aware.csv"
        )

        self.event_rows = [
            (1, "Reception", 2, 2, "A", "", "", 9),
            (2, "Pass", 10, 14, "A", "A", 5, 4),
            (3, "Reception", 15, 15, "A", "", "", 5),
            (4, "Pass", 30, 34, "A", "A", 6, 5),
            (5, "Reception", 35, 35, "A", "", "", 6),
        ]
        self.write_events()

        write_csv(
            self.spacing,
            ("frame", "median_nearest_opponent_m"),
            [
                (frame, 4 if frame < 15 else 6)
                for frame in range(50)
                if frame != 38
            ],
        )
        self.phases = [
            {"start_frame": 0, "end_frame": 49, "possession": "A"}
        ]

    def write_events(self):
        write_csv(
            self.events,
            (
                "event_id", "event_type", "start_frame", "end_frame",
                "team", "target_team", "target_player_id", "player_id",
            ),
            self.event_rows,
        )

    def analyze(self, phases=None):
        return analyze_event_windows(
            self.root,
            "identity_aware",
            phases if phases is not None else self.phases,
            set(range(50)),
        )

    def test_supported_pair_and_explicit_omissions(self):
        result = self.analyze()
        self.assertEqual(len(result["supported"]), 1)
        pair = result["supported"][0]
        self.assertEqual(
            (pair["pass_event_id"], pair["reception_event_id"]),
            (2, 3),
        )
        self.assertEqual(pair["difference_after_minus_before"], 2)
        self.assertEqual(pair["before_frames"], [5, 9])
        self.assertEqual(pair["after_frames"], [16, 20])
        self.assertEqual(
            {item["reason"] for item in result["unavailable"]},
            {"missing_spacing_frames", "no_paired_pass"},
        )

    def test_phase_boundary_suppresses_observation(self):
        result = self.analyze([
            {"start_frame": 0, "end_frame": 17, "possession": "A"},
            {"start_frame": 18, "end_frame": 49, "possession": "B"},
        ])
        self.assertFalse(result["supported"])
        pair = next(
            item for item in result["unavailable"]
            if item.get("pass_event_id") == 2
        )
        self.assertEqual(
            pair["reason"],
            "phase_boundary_or_unknown_possession",
        )

    def test_target_track_mismatch_suppresses_pair(self):
        self.event_rows[2] = (
            3, "Reception", 15, 15, "A", "", "", 8
        )
        self.write_events()
        result = self.analyze()
        pair = next(
            item for item in result["unavailable"]
            if item.get("pass_event_id") == 2
        )
        self.assertEqual(pair["reason"], "target_track_mismatch")


if __name__ == "__main__":
    unittest.main()
