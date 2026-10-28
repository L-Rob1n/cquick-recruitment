"""Run tasks 1-3 in order.

By default the existing images in ``data/images`` are reused.  Pass
``--regenerate-data`` to rebuild the synthetic test scenes and ground truth
before running the tasks.

Run from the project root:
    python src/run_all.py
    python src/run_all.py --regenerate-data
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_SCRIPTS = ["task1_basics.py", "task2_processing.py", "task3_color_detect.py"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate-data", action="store_true",
                        help="run make_test_images.py before the tasks")
    args = parser.parse_args()

    missing = [name for name in ["cv2", "numpy"]
               if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(f"missing dependencies: {', '.join(missing)}; "
                         "activate cv_practice and pip install -r requirements.txt")

    steps = (["make_test_images.py"] if args.regenerate_data else []) + TASK_SCRIPTS
    for name in steps:
        script = ROOT / "src" / name
        print(f"\n=== {name} ===", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)

    print("\nAll requested tasks finished. Results are under outputs/.")


if __name__ == "__main__":
    main()
