"""Synthetic backgrounds for the board finder: book pages, web pages, tables
and photo-like noise. They replace a screenshot dataset with no known licence.
"""

import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.fen_recognition.generate_chessboards import _load_fonts

_PIECE_LETTERS = "KQRBN"
_FILES = "abcdefgh"


def _paper_colour() -> tuple[int, int, int]:
    if random.random() < 0.15:
        v = random.randint(10, 50)
        return (v, v, v + random.randint(0, 15))
    base = random.randint(200, 255)
    return (base, base - random.randint(0, 20), base - random.randint(0, 40))


def _ink_colour(paper: tuple[int, int, int]) -> tuple[int, int, int]:
    dark_paper = sum(paper) < 300
    if random.random() < 0.15:
        return tuple(random.randint(0, 255) for _ in range(3))
    v = random.randint(190, 255) if dark_paper else random.randint(0, 70)
    return (v, v, v)


def _move_text() -> str:
    words = []
    number = random.randint(1, 60)
    for _ in range(random.randint(2, 10)):
        piece = random.choice(_PIECE_LETTERS) if random.random() < 0.6 else ""
        capture = "x" if random.random() < 0.2 else ""
        move = f"{piece}{capture}{random.choice(_FILES)}{random.randint(1, 8)}"
        if random.random() < 0.5:
            words.append(f"{number}.{move}")
            number += 1
        else:
            words.append(move + random.choice(["", "", "+", "!", "?", "!?"]))
    return " ".join(words)


def _word_text(font_codepoints) -> str:
    letters = "abcdefghijklmnopqrstuvwxyz"
    words = ["".join(random.choice(letters) for _ in range(random.randint(1, 10))) for _ in range(random.randint(3, 14))]
    text = " ".join(words)
    if font_codepoints and random.random() < 0.1:
        text += " " + "".join(chr(random.choice(font_codepoints)) for _ in range(4))
    return text.capitalize()


def _font(size: int):
    fonts = _load_fonts()
    if not fonts:
        return ImageFont.load_default(), ()
    path, codepoints = random.choice(fonts)
    try:
        return ImageFont.truetype(path, size), codepoints
    except OSError:
        return ImageFont.load_default(), ()


def _page(width: int, height: int) -> Image.Image:
    paper = _paper_colour()
    img = Image.new("RGB", (width, height), paper)
    draw = ImageDraw.Draw(img)
    ink = _ink_colour(paper)
    columns = random.choice([1, 1, 2, 2, 3])
    margin = random.randint(5, max(6, width // 10))
    gutter = random.randint(8, 30)
    col_w = max(20, (width - 2 * margin - gutter * (columns - 1)) // columns)
    size = random.randint(8, max(9, height // 25))
    font, codepoints = _font(size)
    for c in range(columns):
        x = margin + c * (col_w + gutter)
        y = margin + random.randint(-size, size)
        while y < height:
            if random.random() < 0.05:
                y += random.randint(size, size * 6)
                continue
            text = _move_text() if random.random() < 0.4 else _word_text(codepoints)
            line = ""
            for word in text.split():
                trial = f"{line} {word}".strip()
                if draw.textlength(trial, font=font) > col_w:
                    break
                line = trial
            weight_ink = ink if random.random() > 0.05 else _ink_colour(paper)
            draw.text((x, y), line, fill=weight_ink, font=font)
            y += int(size * random.uniform(1.1, 1.6))
    return img


def _web(width: int, height: int) -> Image.Image:
    paper = _paper_colour()
    img = _page(width, height) if random.random() < 0.5 else Image.new("RGB", (width, height), paper)
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(3, 15)):
        w = random.randint(width // 10, width)
        h = random.randint(8, max(9, height // 4))
        x = random.randint(-w // 2, width)
        y = random.randint(-h // 2, height)
        colour = tuple(random.randint(0, 255) for _ in range(3))
        radius = random.choice([0, 0, 4, 8, 16])
        draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=colour)
        if random.random() < 0.5:
            font, codepoints = _font(max(8, h // 2))
            draw.text((x + 6, y + 2), _word_text(codepoints)[:30], fill=_ink_colour(colour), font=font)
    return img


def _table(width: int, height: int) -> Image.Image:
    """Grids that are not chess boards: tables, graph paper, tiles."""
    paper = _paper_colour()
    img = Image.new("RGB", (width, height), paper)
    draw = ImageDraw.Draw(img)
    ink = _ink_colour(paper)
    rows = random.randint(2, 30)
    cols = random.choice([random.randint(2, 7), random.randint(9, 30)])
    x0, y0 = random.randint(0, width // 4), random.randint(0, height // 4)
    cell_w = random.randint(8, max(9, (width - x0) // cols))
    cell_h = random.randint(8, max(9, (height - y0) // rows))
    filled = random.random() < 0.3
    for r in range(rows + 1):
        draw.line((x0, y0 + r * cell_h, x0 + cols * cell_w, y0 + r * cell_h), fill=ink, width=random.randint(1, 2))
    for c in range(cols + 1):
        draw.line((x0 + c * cell_w, y0, x0 + c * cell_w, y0 + rows * cell_h), fill=ink, width=random.randint(1, 2))
    if filled:
        colour = tuple(random.randint(0, 255) for _ in range(3))
        for r in range(rows):
            for c in range(cols):
                if random.random() < 0.3:
                    draw.rectangle((x0 + c * cell_w + 1, y0 + r * cell_h + 1, x0 + (c + 1) * cell_w - 1, y0 + (r + 1) * cell_h - 1), fill=colour)
    return img


def _photo(width: int, height: int) -> Image.Image:
    a = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
    b = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
    t = np.linspace(0, 1, width, dtype=np.float32)[None, :, None]
    if random.random() < 0.5:
        t = np.linspace(0, 1, height, dtype=np.float32)[:, None, None] * np.ones((1, width, 1), np.float32)
    else:
        t = t * np.ones((height, 1, 1), np.float32)
    arr = a * (1 - t) + b * t
    arr += np.random.normal(0, random.uniform(2, 25), arr.shape).astype(np.float32)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(0, 25)):
        x, y = random.randint(0, width), random.randint(0, height)
        r = random.randint(4, max(5, min(width, height) // 3))
        colour = tuple(random.randint(0, 255) for _ in range(3))
        if random.random() < 0.5:
            draw.ellipse((x - r, y - r, x + r, y + r), fill=colour)
        else:
            points = [(x + random.randint(-r, r), y + random.randint(-r, r)) for _ in range(random.randint(3, 6))]
            draw.polygon(points, fill=colour)
    return img.filter(ImageFilter.GaussianBlur(random.uniform(0, 3)))


def generate_background(width: int, height: int) -> Image.Image:
    kind = random.choices([_page, _web, _table, _photo], weights=[4, 3, 1, 2])[0]
    img = kind(width, height)
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.5)))
    return img
