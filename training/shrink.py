"""Stores the large weights of exported models as float16, to halve the download.

    uv run --extra train-cuda python shrink.py ../build/models ../build/models-f16

Each large float32 initializer becomes a float16 initializer and a Cast back
to float32. tract folds the Cast when it loads the model, so inference still
runs in float32. The script rewrites manifest.json for the new files.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MIN_ELEMENTS = 1024


def shrink(src: Path, dst: Path) -> None:
    model = onnx.load(src)
    graph = model.graph
    casts = []
    for init in list(graph.initializer):
        if init.data_type != TensorProto.FLOAT:
            continue
        array = numpy_helper.to_array(init)
        if array.size < MIN_ELEMENTS:
            continue
        name = init.name
        half = numpy_helper.from_array(array.astype(np.float16), f"{name}__f16")
        graph.initializer.remove(init)
        graph.initializer.append(half)
        casts.append(helper.make_node("Cast", [half.name], [name], to=TensorProto.FLOAT, name=f"{name}__cast"))
    for i, node in enumerate(casts):
        graph.node.insert(i, node)
    onnx.checker.check_model(model)
    onnx.save(model, dst)


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src / "manifest.json").read_text())
    for entry in manifest["models"].values():
        shrink(src / entry["file"], dst / entry["file"])
        data = (dst / entry["file"]).read_bytes()
        print(f"{entry['file']}: {entry['bytes'] / 1e6:.1f} MB -> {len(data) / 1e6:.1f} MB")
        entry["bytes"] = len(data)
        entry["sha256"] = hashlib.sha256(data).hexdigest()
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
