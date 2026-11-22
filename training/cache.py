"""Writes generated training samples to disk on every core.

    uv run --extra train-cuda python cache.py position --samples 80000
    FENRIR_CACHE=../build/cache uv run --extra train-cuda python main.py train position --game chess
"""

import argparse
import os
from pathlib import Path

from src.cache import TASKS, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--out", type=Path, default=Path("../build/cache"))
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--test", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    write(args.task, args.out / args.task / "test", args.test, args.workers, seed=1)
    write(args.task, args.out / args.task / "train", args.samples, args.workers, seed=2)
    (args.out / args.task / "DONE").touch()


if __name__ == "__main__":
    main()
