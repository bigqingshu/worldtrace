from __future__ import annotations

import json
import subprocess
import sys
import unittest


class UnifiedTimelineImportBoundaryTests(unittest.TestCase):
    def test_core_package_import_stays_standard_library_only(self) -> None:
        code = (
            "import json, sys; "
            "import experiments.unified_timeline_lab; "
            "blocked = ('PySide6', 'cv2', 'numpy', 'pynput'); "
            "print(json.dumps({name: any(module == name or "
            "module.startswith(name + '.') for module in sys.modules) "
            "for name in blocked}))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            json.loads(result.stdout),
            {"PySide6": False, "cv2": False, "numpy": False, "pynput": False},
        )


if __name__ == "__main__":
    unittest.main()
