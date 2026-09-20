"""
Runs both test modules in one go and returns a single exit code.

    python tests/run_all.py

Each module is run in its own process because both of them pin
CRYPTO_BIAS_DB and CRYPTO_BIAS_OUT at import time, and they point at
different fixtures.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULES = ["test_preprocessing.py", "test_analysis.py"]


def main():
    failed = []
    for name in MODULES:
        rc = subprocess.call([sys.executable, str(HERE / name)])
        if rc:
            failed.append(name)
    print("\n" + "#" * 78)
    if failed:
        print(f" FAILURES in: {', '.join(failed)}")
    else:
        print(f" {len(MODULES)} test modules, all checks passed")
    print("#" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
