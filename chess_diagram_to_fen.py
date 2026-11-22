"""Extract FEN strings from images of chess board diagrams.

Uses onnxruntime for inference — no torch/torchvision required at runtime.
Run export_onnx.py once (requires torch) to generate the ONNX models from .pth files.
"""

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src import common, consts
from src.bounding_box.inference import get_all_quads, mask_to_corners
from src.games import GAMES, get_game
from src.ort_inference import (
    check_existence,
    classify_orientation,
    classify_rotation,
    get_bbox_mask,
    recognize_position,
    rotate_board_180,
    _pil_to_chw,
    _resize_chw,
    _minmax_norm,
)

_ROTATIONS = [0, 90, 180, 270]


# ---------------------------------------------------------------------------
# Perspective warp (PIL/skimage — no torch)
# ---------------------------------------------------------------------------

def _perspective_warp(img: Image.Image, corners_px: np.ndarray, output_size: int) -> Image.Image:
    """Warp img so the quad defined by corners_px maps to a square.

    corners_px: [4, 2] float32 array of (x, y) pixel coords, order TL/TR/BR/BL.
    """
    from skimage.transform import ProjectiveTransform

    dst = np.array(
        [(0, 0), (output_size, 0), (output_size, output_size), (0, output_size)],
        dtype=np.float64,
    )
    src = np.array(corners_px, dtype=np.float64)
    t = ProjectiveTransform.from_estimate(dst, src)
    coeffs = t.params.flatten()[:8]
    return img.transform(
        (output_size, output_size),
        Image.PERSPECTIVE,
        coeffs.tolist(),
        Image.BICUBIC,
    )


# ---------------------------------------------------------------------------
# Single-board warp (used by get_fen)
# ---------------------------------------------------------------------------

@dataclass
class WarpResult:
    image: Image.Image = None
    mask: np.ndarray = None    # [512, 512] sigmoid values
    corners: np.ndarray = None  # [4, 2] in BBOX_IMAGE_SIZE space


def warp_to_board(img: Image.Image, game: str, max_num_tries: int = 10) -> WarpResult:
    pad_factor = 0.05
    img = common.pad(img, img.width * pad_factor, img.height * pad_factor)

    for _ in range(max_num_tries):
        if img.width == 0 or img.height == 0:
            return WarpResult()

        arr = _pil_to_chw(img)
        arr = _resize_chw(arr, consts.BBOX_IMAGE_SIZE, consts.BBOX_IMAGE_SIZE)
        arr = _minmax_norm(arr)

        mask = get_bbox_mask(arr, game)
        corners = mask_to_corners(mask)
        if corners is None:
            return WarpResult(mask=mask, corners=None)

        x_factor = img.width / consts.BBOX_IMAGE_SIZE
        y_factor = img.height / consts.BBOX_IMAGE_SIZE
        corners_px = corners.copy()
        corners_px[:, 0] *= x_factor
        corners_px[:, 1] *= y_factor

        xs, ys = corners_px[:, 0], corners_px[:, 1]
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        quad_w, quad_h = x2 - x1, y2 - y1

        if quad_w / img.width > 0.7 and quad_h / img.height > 0.7:
            side_lengths = [
                np.linalg.norm(corners_px[j] - corners_px[(j + 1) % 4])
                for j in range(4)
            ]
            output_size = max(32, int(sum(side_lengths) / len(side_lengths)))
            warped = _perspective_warp(img, corners_px, output_size)
            return WarpResult(image=warped, mask=mask, corners=corners)

        pad = 0.1
        crop_x1 = int(max(x1 - quad_w * pad, 0))
        crop_y1 = int(max(y1 - quad_h * pad, 0))
        crop_x2 = int(min(x2 + quad_w * pad, img.width))
        crop_y2 = int(min(y2 + quad_h * pad, img.height))
        img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    return WarpResult()


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class FenResult:
    fen: str = None
    notation: str = None
    game: str = None
    cropped_image: Image.Image = None
    bbox_mask: np.ndarray = None    # [512, 512] sigmoid values
    bbox_corners: np.ndarray = None  # [4, 2] in BBOX_IMAGE_SIZE space
    image_rotation_angle: int = None
    board_is_flipped: bool = None


def get_supported_games() -> list[str]:
    return list(GAMES.keys())


# ---------------------------------------------------------------------------
# Core: single-board
# ---------------------------------------------------------------------------

def get_fen(
    img: Image.Image,
    game: str,
    auto_rotate_image: bool = True,
    mirror_when_180_rotation: bool = False,
    auto_rotate_board: bool = True,
) -> FenResult | None:
    """Extract FEN from an image containing one chess board.

    Returns a FenResult, or None if no board is detected.
    FenResult.fen is None if the board was found but the position couldn't be read.
    """
    spec = get_game(game)
    img = img.convert("RGB")

    if not check_existence(img, spec.key):
        return None

    result = FenResult()
    warp = warp_to_board(img, spec.key)
    result.bbox_mask = warp.mask
    result.bbox_corners = warp.corners

    if warp.image is None:
        return result

    result.image_rotation_angle = classify_rotation(warp.image, spec.key)

    if auto_rotate_image:
        cropped = warp.image.rotate(-_ROTATIONS[result.image_rotation_angle], expand=True)
        if mirror_when_180_rotation and _ROTATIONS[result.image_rotation_angle] == 180:
            cropped = ImageOps.mirror(cropped)
    else:
        cropped = warp.image

    result.cropped_image = cropped
    board = recognize_position(cropped, spec.key)

    if board is not None:
        result.board_is_flipped = classify_orientation(board, spec.key)
        if auto_rotate_board and result.board_is_flipped:
            board = rotate_board_180(board, spec.key)
        result.fen = board.fen()
        result.notation = result.fen
        result.game = spec.key

    return result


# ---------------------------------------------------------------------------
# Core: multi-board (our addition)
# ---------------------------------------------------------------------------

def get_all_fens(
    img: Image.Image,
    game: str,
    min_area_fraction: float = 0.02,
    mask_threshold: float = 0.2,
    auto_rotate_image: bool = True,
    auto_rotate_board: bool = True,
) -> list[FenResult]:
    """Like get_fen but returns one FenResult per detected board region.

    Runs a single bbox inference pass and finds all board regions above
    min_area_fraction of the mask area. Useful for pages with multiple diagrams.
    Results sorted top-to-bottom by position on the page.
    """
    spec = get_game(game)
    img = img.convert("RGB")

    if not check_existence(img, spec.key):
        return []

    pad_factor = 0.05
    padded = common.pad(img, img.width * pad_factor, img.height * pad_factor)

    arr = _pil_to_chw(padded)
    arr = _resize_chw(arr, consts.BBOX_IMAGE_SIZE, consts.BBOX_IMAGE_SIZE)
    arr = _minmax_norm(arr)

    mask = get_bbox_mask(arr, spec.key)
    quads = get_all_quads(mask, min_area_fraction=min_area_fraction, mask_threshold=mask_threshold)

    if not quads:
        return []

    x_factor = padded.width / consts.BBOX_IMAGE_SIZE
    y_factor = padded.height / consts.BBOX_IMAGE_SIZE

    results = []
    for corners in quads:
        corners_px = corners.copy()
        corners_px[:, 0] *= x_factor
        corners_px[:, 1] *= y_factor

        side_lengths = [
            np.linalg.norm(corners_px[j] - corners_px[(j + 1) % 4])
            for j in range(4)
        ]
        output_size = max(32, int(sum(side_lengths) / len(side_lengths)))
        warped = _perspective_warp(padded, corners_px, output_size)

        result = FenResult()
        result.bbox_corners = corners
        result.bbox_mask = mask

        result.image_rotation_angle = classify_rotation(warped, spec.key)
        cropped = warped.rotate(
            -_ROTATIONS[result.image_rotation_angle], expand=True
        ) if auto_rotate_image else warped
        result.cropped_image = cropped

        board = recognize_position(cropped, spec.key)
        if board is not None:
            result.board_is_flipped = classify_orientation(board, spec.key)
            if auto_rotate_board and result.board_is_flipped:
                board = rotate_board_180(board, spec.key)
            result.fen = board.fen()
            result.notation = result.fen
            result.game = spec.key

        results.append(result)

    # Sort top-to-bottom by TL corner y coordinate
    results.sort(
        key=lambda r: float(r.bbox_corners[1, 1]) if r.bbox_corners is not None else 0.0
    )
    return results


# ---------------------------------------------------------------------------
# Demo (requires matplotlib, kept for dev use)
# ---------------------------------------------------------------------------

def demo(root_dir: str, shuffle_files: bool, game: str, show_bbox: bool = False):
    import matplotlib.pyplot as plt
    spec = get_game(game)
    print("Using CPU (onnxruntime)")

    path = Path(root_dir)
    file_names = (
        common.glob_all_image_files_recursively(path) if path.is_dir() else [path]
    )
    if shuffle_files:
        random.shuffle(file_names)

    for file_name in file_names:
        print(file_name)
        img = Image.open(file_name).convert("RGB")
        img = img.rotate(random.choice(_ROTATIONS), expand=True)

        fen_result = get_fen(img, game=game)

        if fen_result is None:
            print("Couldn't detect board:", file_name)
        elif fen_result.fen is None:
            print("Couldn't detect FEN:", file_name)
        else:
            print(fen_result.fen)

        true_fen = common.normalize_position_notation(Path(file_name).stem, spec)
        if true_fen is None:
            print("WARNING: Couldn't find ground truth FEN")
        else:
            if (
                fen_result is not None
                and fen_result.fen is not None
                and fen_result.fen.split()[0] == true_fen.split()[0]
            ):
                print("Correct")
            else:
                print(true_fen)
                print("WRONG")

        if show_bbox:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 8))
        else:
            fig, (ax1, ax3, ax4) = plt.subplots(1, 3, figsize=(15, 8))
            ax2 = None

        ax1.imshow(img)
        if fen_result is not None:
            if show_bbox and ax2 is not None and fen_result.bbox_mask is not None:
                ax2.imshow(fen_result.bbox_mask, cmap="gray", vmin=0, vmax=1)
                if fen_result.bbox_corners is not None:
                    from matplotlib.patches import Polygon as MplPolygon
                    patch = MplPolygon(
                        fen_result.bbox_corners, closed=True,
                        fill=False, edgecolor="red", linewidth=2,
                    )
                    ax2.add_patch(patch)
            if fen_result.cropped_image is not None:
                ax3.imshow(fen_result.cropped_image)
            if fen_result.fen is not None:
                pos = common.position_from_notation(fen_result.fen, spec)
                if pos is not None:
                    fen_img = common.get_image(pos, width=512, height=512)
                    ax4.imshow(fen_img)

        for ax, title in [
            (ax1, "Original image"),
            (ax3, "Cropped to board"),
            (ax4, "Recognized board"),
        ]:
            ax.axis("off")
            ax.title.set_text(title)
        if ax2 is not None:
            ax2.axis("off")
            ax2.title.set_text("BBox mask")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--shuffle_files", action="store_true")
    parser.add_argument("--show_bbox", action="store_true")
    parser.add_argument("--game", type=str, required=True)
    args = parser.parse_args()
    demo(root_dir=args.dir, shuffle_files=args.shuffle_files, game=args.game, show_bbox=args.show_bbox)
