"""Onnxruntime inference — replaces all torch model calls for the 5 chess models.

Preprocessing mirrors the original torchvision transforms exactly:
  - existence/bbox/image_rotation/position: PIL → float32 CHW → resize → normalize
  - orientation: piece-placement string → one-hot float32 array
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

import os

_MODELS_DIR = Path(os.environ.get("FENRIR_MODELS", Path(__file__).parent.parent.parent / "build" / "models"))
_SESSIONS: dict[str, ort.InferenceSession] = {}


def has_model(name: str) -> bool:
    return (_MODELS_DIR / f"{name}.onnx").exists()


def _session(game: str, name: str) -> ort.InferenceSession:
    key = name
    if key not in _SESSIONS:
        path = _MODELS_DIR / f"{name}.onnx"
        if not path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {path}\n"
                f"Run export_onnx.py once (requires torch) to generate it."
            )
        _SESSIONS[key] = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
    return _SESSIONS[key]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _pil_to_chw(img) -> np.ndarray:
    """PIL Image → float32 [3, H, W] in [0, 1]."""
    import numpy as np
    from PIL import Image
    return np.array(img.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0


def _resize_chw(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """[3, H, W] → [3, h, w] bicubic."""
    hwc = arr.transpose(1, 2, 0)
    hwc = cv2.resize(hwc, (w, h), interpolation=cv2.INTER_CUBIC)
    return hwc.transpose(2, 0, 1)


def _minmax_mean_norm(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize then subtract mean (matches MinMaxMeanNormalization)."""
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr)
    arr = arr - arr.mean()
    return arr


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize only (used for bbox model)."""
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        return (arr - mn) / (mx - mn)
    return np.zeros_like(arr)


def _prep_full_image(img, size: int, mean_sub: bool) -> np.ndarray:
    """PIL → [1, 3, size, size] ready for inference."""
    arr = _pil_to_chw(img)
    arr = _resize_chw(arr, size, size)
    arr = _minmax_mean_norm(arr) if mean_sub else _minmax_norm(arr)
    return arr[np.newaxis].astype(np.float32)


# ---------------------------------------------------------------------------
# Public inference functions
# ---------------------------------------------------------------------------

def check_existence(img, game: str) -> bool:
    """True if the image contains a chess board."""
    from src.consts import BBOX_IMAGE_SIZE
    x = _prep_full_image(img, BBOX_IMAGE_SIZE, mean_sub=True)
    logit = _session(game, "existence").run(None, {"image": x})[0]
    return float(logit.flat[0]) > 0.5


def get_bbox_mask(img_chw_minmax: np.ndarray, game: str) -> np.ndarray:
    """Segmentation mask inference. Input [3,512,512] min-max only. Returns [512,512] sigmoid."""
    x = img_chw_minmax[np.newaxis].astype(np.float32)
    logits = _session(game, "quad").run(None, {"image": x})[0]  # [1,1,512,512]
    return _sigmoid(logits[0, 0])


def classify_rotation(img, game: str) -> int:
    """Return rotation index (0-3 → 0°/90°/180°/270°)."""
    from src.consts import BOARD_PIXEL_WIDTH
    x = _prep_full_image(img, BOARD_PIXEL_WIDTH, mean_sub=True)
    logits = _session(game, "rotation").run(None, {"image": x})[0]  # [1,4]
    return int(np.argmax(logits[0]))


def recognize_position(img, game: str):
    """PIL board crop → common.Position, or None if empty/too small."""
    from src import consts
    from src.common import grid_to_piece_placement, Position
    from src.games import get_game

    spec = get_game(game)
    board_h, board_w = consts.board_pixel_size(spec)

    if img.width < 32 or img.height < 32:
        return None

    arr = _pil_to_chw(img)
    arr = _resize_chw(arr, board_h, board_w)
    arr = _minmax_mean_norm(arr)

    if np.isnan(arr).any():
        return None

    x = arr[np.newaxis].astype(np.float32)
    logits = _session(game, "position").run(None, {"board": x})[0]  # [1,64,ch]
    logits = np.clip(logits[0], 0, 1)  # [64, ch]

    indices = np.argmax(logits, axis=1)
    empty_idx = len(spec.piece_symbols)
    grid = [None if int(i) == empty_idx else spec.piece_symbols[int(i)] for i in indices]

    try:
        piece_placement = grid_to_piece_placement(grid, spec)
    except ValueError:
        return None

    pos = Position(game=spec.key, piece_placement=piece_placement)
    if pos.occupied == 0:
        return None
    return pos


def classify_orientation(position, game: str, no_rotate_bias: float | None = None) -> bool:
    """True if the board should be flipped (black's perspective)."""
    from src.common import parse_piece_placement
    from src.games import get_game

    if no_rotate_bias is None:
        no_rotate_bias = 0.45 if game == "xiangqi" else 0.2

    spec = get_game(game)
    piece_to_idx = {p: i for i, p in enumerate(spec.piece_symbols)}
    empty_idx = len(spec.piece_symbols)
    channels = len(spec.piece_symbols) + 1

    grid = parse_piece_placement(position.piece_placement, spec)
    arr = np.zeros((1, spec.num_squares, channels), dtype=np.float32)
    for sq, piece in enumerate(grid):
        idx = piece_to_idx.get(piece, empty_idx) if piece else empty_idx
        arr[0, sq, idx] = 1.0

    logit = _session(game, "orientation").run(None, {"squares": arr})[0]
    return float(logit.flat[0]) - no_rotate_bias > 0.5


def rotate_board_180(position, game: str):
    """Rotate position 180° and swap piece colors."""
    from src.common import grid_to_piece_placement, parse_piece_placement, Position
    from src.games import get_game

    spec = get_game(game)
    grid = parse_piece_placement(position.piece_placement, spec)
    rotated = list(reversed(grid))
    swapped = [
        spec.color_swap_map.get(p, p) if p is not None else None
        for p in rotated
    ]
    return Position(
        game=spec.key,
        piece_placement=grid_to_piece_placement(swapped, spec),
    )
