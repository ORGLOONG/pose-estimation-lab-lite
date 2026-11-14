"""Run the synthetic module tests."""
from pathlib import Path
import sys
import unittest


def main():
    root = Path(__file__).resolve().parents[1]
    required = [root / "src" / name for name in (
        "contour_keypoints.py", "camera_geometry.py", "visualization.py")]
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        print("Missing source files in src/: " + ", ".join(missing), file=sys.stderr)
        return 2
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern="test_public_contracts.py")
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

