"""Scores diagram recognition on images named by their position.

An image's name is its piece placement with "-" for "/", for example
"rnbqkbnr-pppppppp-8-8-8-8-PPPPPPPP-RNBQKBNR.png". Runs of "1" also count.

    uv run python evaluate.py DIR                  # the Python reference
    uv run python evaluate.py DIR --tsv rust.tsv   # lines of "path<TAB>fen"
"""

import argparse
import re
from pathlib import Path

from PIL import Image


def expected(path: Path) -> str:
    stem = re.sub(r"\+$", "", path.stem)
    return normalise(stem.replace("-", "/"))


def normalise(placement: str) -> str:
    expanded = re.sub(r"\d", lambda m: "1" * int(m.group()), placement)
    return re.sub(r"1+", lambda m: str(len(m.group())), expanded)


def squares_right(a: str, b: str) -> int:
    def cells(p):
        return re.sub(r"\d", lambda m: "." * int(m.group()), p).replace("/", "")
    return sum(x == y for x, y in zip(cells(a), cells(b)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("--tsv", type=Path, help="predictions to score instead of running the reference")
    parser.add_argument("--write", type=Path, help="write the reference's predictions as TSV")
    args = parser.parse_args()

    images = sorted(p for p in args.dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if args.tsv:
        found = {}
        for line in args.tsv.read_text().splitlines():
            path, fen = line.split("\t")
            found[Path(path).name] = None if fen == "-" else fen.split()[0]
    else:
        from reference import get_fen
        found = {}
        for p in images:
            result = get_fen(Image.open(p), "chess")
            found[p.name] = result.fen.split()[0] if result and result.fen else None
        if args.write:
            args.write.write_text("".join(f"{p}\t{found[p.name] or '-'}\n" for p in images))

    boards = squares = 0
    for p in images:
        want = expected(p)
        got = found.get(p.name)
        if got and normalise(got) == want:
            boards += 1
            squares += 64
        else:
            squares += squares_right(want, normalise(got)) if got else 0
            print(f"wrong {p.name}: {got}")
    n = len(images)
    print(f"{boards}/{n} boards right ({boards / n:.1%}), {squares / (64 * n):.1%} of squares")


if __name__ == "__main__":
    main()
