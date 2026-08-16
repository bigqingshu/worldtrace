from __future__ import annotations

import json
import subprocess
import sys
import unittest


class ImportBoundaryTests(unittest.TestCase):
    def test_package_import_has_no_qt_or_native_listener_side_effect(self) -> None:
        code = (
            "import json, sys; "
            "import experiments.control_overlay_lab; "
            "print(json.dumps({"
            "'qt': any(name.startswith('PySide6') for name in sys.modules), "
            "'app': 'experiments.control_overlay_lab.app' in sys.modules, "
            "'hotkeys': 'experiments.control_overlay_lab.hotkeys' in sys.modules, "
            "'native': 'experiments.control_overlay_lab.native_overlay' in sys.modules"
            "}))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            json.loads(completed.stdout),
            {"qt": False, "app": False, "hotkeys": False, "native": False},
        )


if __name__ == "__main__":
    unittest.main()
