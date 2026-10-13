import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "coach_report_v1.py"


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


class CoachReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run_dir = Path(temporary.name) / "test_run"

        self.possession = (
            self.run_dir / "possession/data/possession_v6_frames.csv"
        )
        self.episodes = (
            self.run_dir / "possession/data/possession_v6_episodes.csv"
        )
        self.tactical = (
            self.run_dir / "analytics/outputs/"
            "team_tactical_metrics_frame_identity_aware.csv"
        )
        self.coverage = (
            self.run_dir / "analytics/outputs/tactical_coverage_comparison.csv"
        )

        write_csv(
            self.possession,
            ("frame", "possession_v6"),
            [(frame, "A" if frame < 12 else "B") for frame in range(24)],
        )
        write_csv(
            self.episodes,
            (
                "episode_id", "start_frame", "end_frame",
                "duration_frames", "possession",
            ),
            [(1, 0, 11, 12, "A"), (2, 12, 23, 12, "B")],
        )
        self.write_tactical(range(24))
        self.write_coverage(24)

    def write_tactical(self, frames):
        write_csv(
            self.tactical,
            (
                "frame", "team", "team_width_m",
                "team_length_m", "compactness_m",
            ),
            [
                (frame, team, 30 if frame < 12 else 20, 35, 15)
                for frame in frames
                for team in ("A", "B")
            ],
        )

    def write_coverage(self, valid_frames):
        write_csv(
            self.coverage,
            ("mode", "complete_frames", "total_frames"),
            [("identity_aware", valid_frames, 24)],
        )

    def run_report(self):
        return subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--run-dir", str(self.run_dir),
                "--branch", "identity_aware",
            ],
            capture_output=True,
            text=True,
        )

    def report(self):
        path = self.run_dir / "reports/coach_report_v1.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_complete_states_support_comparison(self):
        result = self.run_report()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = self.report()
        self.assertEqual(report["coverage"]["tactical_valid_frames"], 24)
        self.assertEqual(
            report["team_shape"]["valid_frames_by_possession"],
            {"A": 12, "B": 12},
        )
        comparison = report["team_shape"]["comparisons"][0]
        self.assertEqual(comparison["status"], "supported")
        self.assertEqual(comparison["difference_in_minus_out"], 10)
        self.assertEqual(
            len(report["team_shape"]["descriptive_medians"]), 6
        )

    def test_missing_b_state_suppresses_comparison(self):
        self.write_tactical(range(12))
        self.write_coverage(12)
        result = self.run_report()
        self.assertEqual(result.returncode, 0, result.stderr)
        shape = self.report()["team_shape"]
        self.assertEqual(shape["valid_frames_by_possession"]["B"], 0)
        self.assertEqual(len(shape["descriptive_medians"]), 4)
        self.assertTrue(
            all(item["status"] == "insufficient_data"
                for item in shape["comparisons"])
        )

    def test_episode_mismatch_is_rejected(self):
        write_csv(
            self.episodes,
            (
                "episode_id", "start_frame", "end_frame",
                "duration_frames", "possession",
            ),
            [(1, 0, 12, 13, "A"), (2, 13, 23, 11, "B")],
        )
        result = self.run_report()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("disagrees at frame 12", result.stderr)



    def test_custom_output_dir_keeps_default_report(self):
        first = self.run_report()
        self.assertEqual(first.returncode, 0, first.stderr)

        default_path = self.run_dir / "reports/coach_report_v1.json"
        original = default_path.read_bytes()
        separate_dir = self.run_dir / "alternative_reports"

        second = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--run-dir", str(self.run_dir),
                "--branch", "identity_aware",
                "--output-dir", str(separate_dir),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertTrue(
            (separate_dir / "coach_report_v1.json").is_file()
        )
        self.assertEqual(default_path.read_bytes(), original)

if __name__ == "__main__":
    unittest.main()
