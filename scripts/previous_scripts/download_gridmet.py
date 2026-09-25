#!/usr/bin/env python3
"""Run the GridMET downloader from the reproducible analysis folder."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    target = repo_root / "scripts" / "gridmet_downloader.py"
    if not target.exists():
        raise FileNotFoundError(f"Missing downloader script: {target}")
    subprocess.run([sys.executable, str(target), *sys.argv[1:]], cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
