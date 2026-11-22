# FENrir

FENrir reads a chess diagram in an image and returns its position as a FEN.
A diagram is a 2D board: a book or magazine page, a PDF, a screenshot, or a
photo of a page or a screen.

FENrir is a Rust library (`fenrir-chess`) and a CLI. The models are trained
in Python with PyTorch, exported to ONNX, and run in Rust with
[tract](https://github.com/sonos/tract). The trained models are release
assets, not part of the repository.

FENrir is a fork of
[Chess_diagram_to_FEN](https://github.com/tsoj/Chess_diagram_to_FEN) by
Jost Triller. It keeps the training method and cuts it down to chess.

## Use

```shell
cargo run --release -p fenrir-cli -- --models path/to/models diagram.png
```

```rust
let models = fenrir::Models::load("models".as_ref())?;
let image = fenrir::image::open("diagram.png")?;
if let Some(diagram) = models.read(&image)? {
    println!("{}", diagram.fen);
}
```

The models directory holds `quad.onnx`, `rotation.onnx`, `position.onnx`
and `orientation.onnx`.

## How it works

1. **quad** finds the board outline as a mask. The library fits four
   corners to it and warps the board square.
2. **rotation** finds how far the board image is turned.
3. **position** reads the piece on each square.
4. **orientation** decides whether the diagram shows Black at the bottom.

## Train

See `training/`. Training uses only permissively licensed assets, listed in
`training/resources/COPYING.md`.

```shell
cd training
uv sync --extra train-cuda
uv run --extra train-cuda python main.py train position --game chess
uv run --extra train-cuda python export.py --out ../build/models --quad ... --rotation ... --position ... --orientation ...
uv run python evaluate.py DIR --tsv predictions.tsv
```

## Plans

- [Diagram recognition](docs/plans/01-diagram-recognition.md)
- [Physical board recognition](docs/plans/02-physical-board-recognition.md)

## Licence

MIT. See `LICENSE`.
