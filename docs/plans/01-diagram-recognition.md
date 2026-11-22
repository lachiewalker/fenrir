# Plan: diagram recognition in Rust

Status: proposed, 2026-11-22.

## Goal

FENrir reads a chess diagram in an image and returns its FEN. A diagram is a
2D board: a book or magazine page, a PDF, a screenshot, or a phone photo of
a page or a screen. [Plan 02](02-physical-board-recognition.md) covers
photos of physical boards.

FENrir is a Rust library with a CLI. Training stays in Python. Trained
models are release assets, not part of the code. Druidschess is the only
consumer for now, so Rust comes first. Python bindings are a stretch goal.

## Where FENrir starts

- FENrir is a fork of
  [tsoj/Chess_diagram_to_FEN](https://github.com/tsoj/Chess_diagram_to_FEN)
  (MIT, © 2024 Jost Triller). The fork's `main` is upstream's `3d75112`
  (2026-03-11).
- Work exists only in the local working tree (2026-06-13), not in a commit:
  - Inference runs on onnxruntime without torch. Torch and the training
    tools are optional extras.
  - `get_all_fens` and `get_all_quads` find every diagram on a book page.
  - `export_onnx.py` exports the five chess models, and
    `src/ort_inference.py` runs them.
  - `models/chess/` holds the exported ONNX files (312 MB, untracked).
- Upstream also does xiangqi and shogi. FENrir does chess only.

## The pipeline

1. **quad**: a segmentation mask of the board (LRASPP, MobileNetV3), 512 px
   input. Contours and polygon fitting give the four corners.
2. A perspective warp makes the board square.
3. **image_rotation**: the board's rotation, 0, 90, 180 or 270 degrees
   (RegNetX-800MF, 256 px).
4. **position**: the piece on each square (two ConvNeXt-Tiny networks, one
   for each tile and one for the whole board, then attention), 256 px.
5. **orientation**: whether the board is shown from White's or Black's
   side (a small CNN on the 64 squares).

**existence** (RegNetX-800MF) answers "is there a board at all". The quad
mask can answer that too, so FENrir can drop existence. Test this before
the model is removed.

## Facts from the spike (2026-11-21)

The spike loaded the five exported models in `tract-onnx` 0.23.10 and
compared the outputs with onnxruntime on the same input.

| Model | Result in tract | Max difference | Desktop time |
|---|---|---|---|
| quad | loads as exported | 8e-5 | 110 ms |
| existence | loads as exported | 3e-6 | 160 ms |
| image_rotation | loads as exported | 0 | 35 ms |
| orientation | loads after a patch | 3e-8 | <1 ms |
| position | loads after a patch, optimiser off | 3e-7 | 1.9 s |

- The PyTorch 2.10 (dynamo) exporter writes a `Loop` for a `view` after a
  strided tensor. tract has no `Loop`. With a fixed input size the loop
  always gives the same indices, so the spike replaced it with a constant.
- position then fails in tract's declutter pass, at the EinSum that follows
  the `expand`, `cat` and `Linear` of the tile and board features. Without
  the optimiser, it runs and all 64 squares agree with onnxruntime.
- tract must ignore the exported intermediate shapes
  (`with_ignore_value_info(true)`), because they name a symbolic batch.
- The chess models total 312 MB, and position alone is 245 MB. That is too
  large for a phone download.

## Licences

- **Code:** MIT. Keep the upstream copyright notice next to FENrir's own.
- **Training images:** `resources/COPYING.md` lists piece sets and board
  themes under mixed licences. They include Chess.com sets (all rights
  reserved), Lichess sets (AGPL, GPL, CC BY-NC-SA), XBoard sets (GPL) and
  sets with no licence.
- **Backgrounds:** the Kaggle web screenshot dataset has no known licence.
- **Weights:** upstream's weights come from all of those images. The law is
  unsettled on whether weights inherit the licences of their training data.
  FENrir does not release weights trained on non-permissive images.
- **Permitted training assets:** CC0, CC BY, CC BY-SA, MIT, Apache-2.0 and
  BSD. No GPL, AGPL, NonCommercial, NoDerivatives, or "no licence".
- **Pretrained backbones:** torchvision code is BSD-3. Its ImageNet weights
  carry ImageNet's research terms. This is common practice, and it is a
  small risk. Record it in the model card.
- **Merida, cburnett and Alpha** are out, and books often use them. Find
  permissive chess fonts with a similar look, and measure accuracy on book
  diagrams.

## Shape of the repository

- `crates/fenrir`: the library. It decodes and resizes the image, runs the
  models with tract, finds the corners, warps the board, and returns a FEN
  with a confidence for each square.
- `crates/fenrir-cli`: `fenrir image.png` prints the FEN.
- `python/` (stretch goal): PyO3 bindings, built with maturin, for a
  `fenrir` wheel on PyPI (the name is free on 2026-11-21).
- `training/`: the Python training code (uv, PyTorch), the dataset
  generators, the export step, and the test-vector writer.
- `models/`: a manifest of the released model files with SHA-256 sums.
  The files themselves are release assets.
- `crates.io` has a `fenrir` crate already (2020, geolocation). The crate
  needs another name, for example `fenrir-chess`.

Image processing in Rust uses `image` and `imageproc` (MIT). `imageproc`
has contours, polygon fitting, the minimum-area rectangle and the
perspective warp. Check the licence of each crate before it is added.

## Order of work

1. **Save the fork's work.** Commit the local working tree to a branch, so
   the onnxruntime work and the multi-diagram code are not lost.
2. **Cut to chess and a library layout.** Remove xiangqi, shogi and the
   demo code. Move training into `training/`. Rename the Python package.
3. **Clean the assets.** Remove every non-permissive image from the
   repository. Keep a list of permitted sets with their sources and
   licences. Replace the screenshot backgrounds with permissive ones, or
   generate them.
4. **Fix the export.** Export at batch size 1 with the TorchScript
   exporter (`dynamo=False`), or use `reshape`/`flatten` in the models.
   Write the tile and board projection as two `Linear` layers that add,
   instead of `expand`, `cat` and one `Linear`. The maths is the same.
   Gate: all five models load in tract with the optimiser on, and match
   PyTorch to within 1e-4.
5. **Port inference to Rust.** Port the pre- and post-processing. Test
   each step against vectors that the Python pipeline writes: the resized
   input, the mask, the corners, the warped board and the FEN. Use
   upstream's weights for this step only. They are never released.
6. **Retrain on permitted assets.** Choose smaller backbones for position,
   for example MobileNetV3 or EfficientNet-B0 for the tiles. Gate: accuracy
   on the real test images is close to upstream's, and the model set is
   under about 40 MB with f16 weights.
7. **Release v1.** Publish the models as release assets with a model card:
   the training data, the licences, the accuracy and the export command.
   Publish the crate. A PyPI wheel waits for the stretch goal.

## Progress (2026-11-22)

- Steps 1 to 5 are done, on the `rust` branch.
- **Export:** the four models load in tract with the optimiser on. The
  orientation model is trained on game records only, with no images, so
  upstream's orientation weights carry no asset licence and stay in use.
  existence is dropped: the quad mask gives "no board".
- **Rust port:** on the 41 named test images, the Rust pipeline and the
  Python reference both read 36 boards right (97.2 % of squares), with
  upstream's weights. The Rust corner fit splits a closed contour at two
  far points, as OpenCV does. A fit from the contour's first point gave
  25 boards.
- **Python reference:** its corner order was BL, TL, TR, BR, not TL, TR,
  BR, BL. The rotation model hid the error. Both pipelines now use TL, TR,
  BR, BL.
- **Training speed:** one generated sample takes 0.2 to 0.9 s of CPU, so
  on-the-fly training left the GPU idle (1.4 steps a second on an RTX 2060).
  `cache.py` writes samples once on every core, and training reads them for
  several epochs.
- **Assets added:** plain and hatched two-tone boards drawn in code, and the
  Cburnett pieces from Wikimedia Commons under BSD.
- **Size:** `shrink.py` stores large weights as float16. Upstream's models
  score the same after it.
- **Retraining (on a two-GPU server, logged to MLflow):** on the 41 real
  test images, with the new quad and rotation models and upstream's
  orientation model:

  | Position backbone | Data | Boards right | Squares right |
  |---|---|---|---|
  | Upstream ConvNeXt-Tiny | upstream's | 36 | 97.2 % |
  | ConvNeXt-Tiny | round 1 | 29 | 96.0 % |
  | ConvNeXt-Tiny | round 2 (fine-tuned) | 29 | 96.2 % |
  | MobileNetV3 | round 1 | 26 | 95.7 % |
  | EfficientNet-B0 | round 1 | 25 | 95.8 % |
  | EfficientNet-B0 | round 2 (fine-tuned) | 25 | 95.8 % |

  The new quad model with upstream's position model scores 35, and the new
  rotation model scores 36, so both are ready. Round 2 adds dithered
  boards, low resolution and JPEG artefacts. It did not help. The misses
  are old book fonts and screen styles that no permitted piece set looks
  like: FENrir has 19 sets, and upstream trained on about 100.
- **Next:** more permitted piece designs, for example the chess glyphs of
  permissive fonts (DejaVu, Noto Sans Symbols 2) and more Wikimedia Commons
  sets. Then retrain position.

## Revisit when

- tract's declutter bug is fixed upstream. Then the `Linear` split is not
  needed, but it costs nothing to keep.
- Accuracy on book diagrams falls too far without the excluded fonts.
