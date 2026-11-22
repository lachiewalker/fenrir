"""Export all trained .pth models to ONNX then convert to ORT format.

Run once with torch available:
    cd fenrir
    uv run --with torch --with torchvision python export_onnx.py

Outputs chess .onnx and .ort files alongside the .pth files in models/chess/.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from src.consts import BBOX_IMAGE_SIZE, DEFAULT_TILE_SIZE
from src.games import get_game

MODELS_DIR = Path(__file__).parent / "models" / "chess"
OPSET = 14


def _find_pth(pattern: str) -> Path:
    matches = list(MODELS_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No .pth matching {pattern} in {MODELS_DIR}")
    return sorted(matches)[-1]  # latest by name


def _load(model_cls, pattern: str) -> nn.Module:
    path = _find_pth(pattern)
    print(f"  loading {path.name}")
    m = model_cls()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m


def export_existence():
    from src.existence.model import BoardExistence
    m = _load(BoardExistence, "best_model_existence_*.pth")
    dummy = torch.randn(1, 3, BBOX_IMAGE_SIZE, BBOX_IMAGE_SIZE)
    out_path = MODELS_DIR / "existence.onnx"
    torch.onnx.export(m, dummy, str(out_path), opset_version=OPSET,
                      input_names=["image"], output_names=["logit"],
                      dynamic_axes={"image": {0: "batch"}})
    print(f"  → {out_path.name}")
    return out_path


def export_quad():
    from src.bounding_box.model import BoardQuad
    m = _load(BoardQuad, "best_model_quad_*.pth")
    dummy = torch.randn(1, 3, BBOX_IMAGE_SIZE, BBOX_IMAGE_SIZE)
    out_path = MODELS_DIR / "quad.onnx"
    torch.onnx.export(m, dummy, str(out_path), opset_version=OPSET,
                      input_names=["image"], output_names=["mask"],
                      dynamic_axes={"image": {0: "batch"}})
    print(f"  → {out_path.name}")
    return out_path


def export_image_rotation():
    from src.board_image_rotation.model import ImageRotation
    from src.consts import BOARD_PIXEL_WIDTH
    m = _load(ImageRotation, "best_model_image_rotation_*.pth")
    # model trained at BOARD_PIXEL_WIDTH (256), not BBOX_IMAGE_SIZE
    dummy = torch.randn(1, 3, BOARD_PIXEL_WIDTH, BOARD_PIXEL_WIDTH)
    out_path = MODELS_DIR / "image_rotation.onnx"
    torch.onnx.export(m, dummy, str(out_path), opset_version=OPSET,
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}})
    print(f"  → {out_path.name}")
    return out_path


def export_position():
    from src.fen_recognition.model import BoardRec
    game = get_game("chess")
    m = _load(lambda: BoardRec("chess"), "best_model_position_*.pth")
    board_h, board_w = game.board_rows * DEFAULT_TILE_SIZE, game.board_cols * DEFAULT_TILE_SIZE
    dummy = torch.randn(1, 3, board_h, board_w)
    out_path = MODELS_DIR / "position.onnx"
    torch.onnx.export(m, dummy, str(out_path), opset_version=OPSET,
                      input_names=["board_image"], output_names=["square_logits"],
                      dynamic_axes={"board_image": {0: "batch"}})
    print(f"  → {out_path.name}")
    return out_path


def export_orientation():
    from src.board_orientation.model import OrientationModel
    from src.games import get_game
    game = get_game("chess")
    m = _load(lambda: OrientationModel("chess"), "best_model_orientation_*.pth")
    num_squares = game.num_squares
    num_channels = len(game.piece_symbols) + 1
    dummy = torch.randn(1, num_squares, num_channels)
    out_path = MODELS_DIR / "orientation.onnx"
    torch.onnx.export(m, dummy, str(out_path), opset_version=OPSET,
                      input_names=["board_tensor"], output_names=["logit"],
                      dynamic_axes={"board_tensor": {0: "batch"}})
    print(f"  → {out_path.name}")
    return out_path


def convert_to_ort(onnx_paths: list[Path]):
    """Convert ONNX → ORT format for faster load and smaller size."""
    try:
        from onnxruntime.tools import convert_onnx_models_to_ort
        for p in onnx_paths:
            print(f"  converting {p.name} → .ort")
            convert_onnx_models_to_ort.convert_onnx_models_to_ort(
                str(p.parent),
                output_dir=str(p.parent),
                optimization_style="Fixed",
            )
        print("ORT conversion done.")
    except Exception as e:
        print(f"ORT conversion failed ({e}) — ONNX files still usable")


if __name__ == "__main__":
    print("Exporting models to ONNX...")
    onnx_paths = []
    for fn in [export_existence, export_quad, export_image_rotation, export_position, export_orientation]:
        name = fn.__name__.replace("export_", "")
        print(f"\n{name}:")
        try:
            onnx_paths.append(fn())
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\nExported {len(onnx_paths)}/5 models.")

    print("\nConverting to ORT format...")
    convert_to_ort(onnx_paths)

    print("\nDone. Files in", MODELS_DIR)
    for f in sorted(list(MODELS_DIR.glob("*.onnx")) + list(MODELS_DIR.glob("*.ort"))):
        print(f"  {f.name}  {f.stat().st_size // 1024}KB")
