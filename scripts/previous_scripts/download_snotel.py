#!/usr/bin/env python3
"""Run the SNOTEL downloader from the reproducible analysis folder."""

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    target = repo_root / "scripts" / "snotel_downloader.py"
    if not target.exists():
        raise FileNotFoundError(f"Missing downloader script: {target}")
    python_exe = os.environ.get("SKAGIT_PYTHON", "/home/nksp2/skagit/skagit_2/skagit-met/.pixi/envs/data-download/bin/python")
    if not Path(python_exe).exists():
        python_exe = sys.executable
    subprocess.run([python_exe, str(target), *sys.argv[1:]], cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
