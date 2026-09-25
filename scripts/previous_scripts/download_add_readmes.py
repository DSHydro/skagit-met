#!/usr/bin/env python3
"""Run the README helper script from the reproducible analysis folder."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    target = repo_root / "scripts" / "add_readmes.py"
    if not target.exists():
        raise FileNotFoundError(f"Missing script: {target}")
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(f"This wrapper calls {target.name} with the arguments you provide.")
        return
    proc = subprocess.run([sys.executable, str(target), *sys.argv[1:]], cwd=repo_root)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
