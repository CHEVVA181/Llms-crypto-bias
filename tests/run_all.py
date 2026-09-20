"""
Runs both test trees in one go and returns a single exit code.

    python tests/run_all.py

Each tree is a separate process with its own fixture database and its own
output folder, so neither can see or overwrite the other's files. They also
pin CRYPTO_BIAS_DB and CRYPTO_BIAS_OUT at import time, which is the other
reason they cannot share one interpreter.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREES = [
    HERE / "preprocessing" / "test_preprocessing.py",
    HERE / "analysis" / "test_analysis.py",
]


def main():
    failed = []
    for path in TREES:
        if subprocess.call([sys.executable, str(path)]):
            failed.append(path.parent.name)
    print("\n" + "#" * 78)
    if failed:
        print(f" FAILURES in: {', '.join(failed)}")
    else:
        print(f" {len(TREES)} test trees, all checks passed")
    print("#" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
