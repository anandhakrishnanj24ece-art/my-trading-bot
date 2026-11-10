from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from ultimateorb_paper.session_tick import main

CT = ZoneInfo("America/Chicago")


def _session_bars() -> pd.DataFrame:
    index = pd.to_datetime(
        [
            "2026-11-09 08:30",
            "2026-11-09 08:45",
            "2026-11-09 09:00",
            "2026-11-09 09:15",
            "2026-11-09 09:30",
            "2026-11-09 09:45",
        ]
    ).tz_localize(CT)
    return pd.DataFrame(
        {
            "Open": [100, 100.5, 101, 101.5, 102, 102.5],
            "High": [101, 102, 102.2, 102.5, 102.8, 103.5],
            "Low": [99, 98, 100.8, 101.2, 101.8, 102.4],
            "Close": [100.5, 101, 101.5, 102, 102.5, 103],
        },
        index=index,
    )


class SessionTickTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_outside_session_stops(self) -> None:
        code = main(["--base", str(self.base), "--now", "2026-11-09T08:00:00-06:00"])
        self.assertEqual(code, 2)

    def test_tick_opens_and_session_close_stops(self) -> None:
        with patch("ultimateorb_paper.auto_paper._load_nq", return_value=_session_bars()):
            code = main(["--base", str(self.base), "--now", "2026-11-09T10:00:00-06:00"])
            self.assertEqual(code, 0)
            code = main(["--base", str(self.base), "--now", "2026-11-09T14:45:00-06:00"])
            self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
