from __future__ import annotations

import json
import subprocess
import sys
import unittest


class InputExecutionImportBoundaryTests(unittest.TestCase):
    def test_package_import_does_not_load_qt_or_pynput_hooks(self) -> None:
        script = (
            "import json, sys; "
            "import experiments.input_execution_lab; "
            "print(json.dumps({"
            "'qt': any(name.startswith('PySide6') for name in sys.modules),"
            "'pynput': any(name.startswith('pynput') for name in sys.modules)"
            "}))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            json.loads(completed.stdout),
            {"qt": False, "pynput": False},
        )


if __name__ == "__main__":
    unittest.main()
