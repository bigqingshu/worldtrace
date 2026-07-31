from __future__ import annotations

import json
import subprocess
import sys
import unittest


class InputCaptureImportBoundaryTests(unittest.TestCase):
    def test_package_import_does_not_load_qt_or_native_hook_library(self) -> None:
        code = (
            "import json, sys; "
            "import experiments.input_capture_lab; "
            "print(json.dumps({"
            "'qt': any(name.startswith('PySide6') for name in sys.modules), "
            "'pynput': any(name.startswith('pynput') for name in sys.modules)"
            "}))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            json.loads(result.stdout),
            {"qt": False, "pynput": False},
        )


if __name__ == "__main__":
    unittest.main()
