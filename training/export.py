"""Exports trained checkpoints to ONNX files that tract can run.

    uv run --extra train-cuda python export.py --out ../build/models \\
        --quad Q.pth --rotation R.pth --position P.pth --orientation O.pth

Each model is exported at batch size 1 with fixed input shapes, by the
TorchScript exporter. The dynamo exporter writes a Loop for some reshapes,
and tract has no Loop. The script checks each file against PyTorch with
onnxruntime, and writes manifest.json with the size and SHA-256 of each file.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from src.board_image_rotation.model import ImageRotation
from src.board_orientation.model import OrientationModel
from src.bounding_box.model import BoardQuad
from src.consts import BBOX_IMAGE_SIZE, BOARD_PIXEL_WIDTH
from src.fen_recognition.model import BoardRec, upgrade_state_dict

OPSET = 17


def _load(model: torch.nn.Module, path: Path, upgrade=lambda s: s) -> torch.nn.Module:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(upgrade(state))
    return model.eval()


def _export(model, shape, input_name, output_name, path: Path) -> dict:
    dummy = torch.rand(*shape)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        opset_version=OPSET,
        input_names=[input_name],
        output_names=[output_name],
        dynamo=False,
    )
    with torch.no_grad():
        expected = model(dummy)
        if isinstance(expected, dict):
            expected = expected["out"]
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = session.run(None, {input_name: dummy.numpy()})[0]
    diff = float(np.abs(got - expected.numpy()).max())
    if diff > 1e-3:
        raise RuntimeError(f"{path.name}: onnxruntime differs from PyTorch by {diff}")
    data = path.read_bytes()
    print(f"{path.name}: {len(data) / 1e6:.1f} MB, max difference {diff:.1e}")
    return {
        "file": path.name,
        "input": input_name,
        "shape": list(shape),
        "output": output_name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quad", type=Path, required=True)
    parser.add_argument("--rotation", type=Path, required=True)
    parser.add_argument("--rotation-backbone", default="mobilenet_v3_large")
    parser.add_argument("--position", type=Path, required=True)
    parser.add_argument("--position-backbone", default="mobilenet_v3_large")
    parser.add_argument("--orientation", type=Path, required=True)
    parser.add_argument("--version", default="dev")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    side = BBOX_IMAGE_SIZE
    board = BOARD_PIXEL_WIDTH
    files = {
        "quad": _export(
            _load(BoardQuad(), args.quad),
            (1, 3, side, side), "image", "mask", args.out / "quad.onnx",
        ),
        "rotation": _export(
            _load(ImageRotation(args.rotation_backbone), args.rotation),
            (1, 3, board, board), "image", "logits", args.out / "rotation.onnx",
        ),
        "position": _export(
            _load(BoardRec("chess", backbone=args.position_backbone), args.position, upgrade_state_dict),
            (1, 3, board, board), "board", "squares", args.out / "position.onnx",
        ),
        "orientation": _export(
            _load(OrientationModel("chess"), args.orientation),
            (1, 64, 13), "squares", "logit", args.out / "orientation.onnx",
        ),
    }
    manifest = {"version": args.version, "models": files}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
