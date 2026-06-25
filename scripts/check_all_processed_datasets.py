#!/usr/bin/env python3
"""Run validation checks for all processed datasets."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


CHECKS = [
    ("MoGCN BRCA", "scripts/check_processed_mogcn_brca.py"),
    ("CancerSD STAD", "scripts/check_processed_cancersd_stad.py"),
    ("MOGONET BRCA", "scripts/check_processed_mogonet_brca.py"),
]


def main() -> None:
    root = Path.cwd()
    failed = []

    print("=" * 80)
    print("Checking all processed datasets")
    print("=" * 80)

    for name, script in CHECKS:
        script_path = root / script
        print(f"\n[{name}] Running {script}...")

        if not script_path.exists():
            print(f"ERROR: missing script {script_path}")
            failed.append(name)
            continue

        result = subprocess.run(
            [sys.executable, str(script_path)],
            text=True,
            capture_output=True,
        )

        print(result.stdout)

        if result.returncode != 0:
            print(result.stderr)
            failed.append(name)
        else:
            print(f"[{name}] PASS")

    print("\n" + "=" * 80)

    if failed:
        print("Some datasets failed validation:")
        for name in failed:
            print(f"- {name}")
        raise SystemExit(1)

    print("All processed datasets passed validation.")
    print("=" * 80)


if __name__ == "__main__":
    main()