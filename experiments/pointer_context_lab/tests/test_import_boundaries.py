from __future__ import annotations

import subprocess
import sys
import unittest


class PointerContextImportBoundaryTests(unittest.TestCase):
    def test_package_import_does_not_load_qt_native_adapter_or_input_hooks(
        self,
    ) -> None:
        script = (
            "import sys; "
            "import experiments.pointer_context_lab; "
            "blocked = [name for name in sys.modules "
            "if name.startswith(('PySide6', 'pynput')) "
            "or name == 'experiments.pointer_context_lab.native_probe']; "
            "raise SystemExit(0 if not blocked else repr(blocked))"
        )

        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            completed.returncode,
            0,
            completed.stdout + completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
