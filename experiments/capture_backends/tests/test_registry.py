from __future__ import annotations

import unittest

from experiments.capture_backends.registry import (
    backend_names,
    create_backend,
    probe_backends,
)


class RegistryTests(unittest.TestCase):
    def test_registry_lists_expected_backends(self) -> None:
        self.assertEqual(
            backend_names(),
            ("mss", "printwindow", "wgc", "dxcam"),
        )

    def test_backend_construction_does_not_initialize_optional_dependencies(self) -> None:
        for name in backend_names():
            backend = create_backend(name)
            self.assertEqual(backend.backend_id, name)
            self.assertEqual(backend.get_health().state.value, "CLOSED")

    def test_probe_reports_every_backend(self) -> None:
        reports = probe_backends()
        self.assertEqual([report.backend_id for report in reports], list(backend_names()))
        for report in reports:
            self.assertTrue(report.availability.status.value)


if __name__ == "__main__":
    unittest.main()
