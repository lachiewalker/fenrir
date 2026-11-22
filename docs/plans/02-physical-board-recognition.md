# Plan: physical board recognition

Status: proposed, 2026-11-22.

## Goal

FENrir reads a phone photo of a physical chess board and returns its FEN.
It uses the same crate, release process and licence rules as
[plan 01](01-diagram-recognition.md).

## Why a second pipeline

The diagram models do not carry over to physical boards:

- The camera views the board at an angle, not from above.
- Tall pieces cover the squares behind them.
- Each set has its own piece shapes, and pieces of one type look different
  from set to set.
- Light, shadows and reflections change more than on paper or a screen.

## Method

FENrir follows chesscog (Wölflein and Arandjelović, 2021):

1. **Corners.** Find the board's lines and corners with classical geometry,
   or with a small keypoint model. Then warp the board flat.
2. **Occupancy.** Classify each square as empty or occupied.
3. **Pieces.** Classify the piece on each occupied square. The crop for a
   square includes the space above it, because a piece stands up from its
   square.
4. **Orientation.** Choose White's side from the coordinates printed on the
   board, or from the position, or ask the user.

chesscog adapts to a new physical set with only a few photos of that set.
FENrir keeps that ability as a fine-tuning step.

## Datasets

| Dataset | Images | Type | Labels | Licence | Use |
|---|---|---|---|---|---|
| [chesscog](https://osf.io/xf3ka/) | about 4,900 | Blender renders | Position, piece boxes, corners | CC BY 4.0 | Train |
| [Fenit](https://data.mendeley.com/datasets/wpp9xnbcp6/1) | Not stated | Real photos | Piece boxes, 4 board corners | CC BY 4.0 | Fine-tune, after a check of its contents |
| [samryan18/chess-dataset](https://github.com/samryan18/chess-dataset) | 500 | Real photos | FEN | MIT (repository) | Fine-tune |
| [Roboflow Chess Pieces](https://public.roboflow.com/object-detection/chess-full) | about 290 | Real photos, one tripod angle | Piece boxes | Listed as Public Domain (confirm) | Fine-tune |
| [ChessReD](https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f) | 10,800 | Real phone photos | Position for all, boxes and corners for 2,078 | CC BY-NC-SA 4.0 | Measure only. Never train released weights on it |
| [LiveChess2FEN](https://github.com/davidmallasen/LiveChess2FEN) | Not stated | Piece crops | Piece classes | AGPL code, no dataset licence | Do not use |

Licences checked 2026-11-21. Check each one again before use, and record
the source and licence of every file in the training set.

## Rendered data

FENrir renders its own training images in Blender:

- 3D chess sets under CC0 or CC BY only, with the source of each set
  recorded.
- Positions sampled from real games, so piece patterns look natural.
- Random camera height, angle and distance, light, table textures,
  backgrounds, blur and noise.
- Labels come from the scene: the FEN, the board corners, and a box for
  each piece.

Rendering gives unlimited labelled data with a clean licence. The risk is
the gap between rendered and real images. The fine-tuning step and the
ChessReD measurement show how large that gap is.

## Order of work

1. **Data tools.** A loader for each dataset, in one format: image, FEN,
   corners, boxes. A check that refuses a file with no recorded licence.
2. **Measure first.** Run chesscog's published method on ChessReD, so
   FENrir has a number to beat.
3. **Render.** The Blender pipeline, with chesscog's renders as the first
   training set.
4. **Train.** The corner, occupancy and piece models on the renders.
   Fine-tune on Fenit, samryan18 and Roboflow. Measure on ChessReD.
   Gate: the share of fully correct boards on ChessReD is close to the
   published results.
5. **Export and port.** The same export rules and test vectors as plan 01.
   The Rust code goes in a `physical` module in the same crate.
6. **Route.** A small classifier, or the user, chooses between the diagram
   pipeline and the physical pipeline for each image.
7. **Release v2.** Models, model card, crate and wheel.
8. **Real data from users (later).** An app that uses FENrir can offer to
   send a photo with the position the user corrected. That needs the
   user's consent and a clear licence for the photo.

## Revisit when

- A larger permissively licensed set of real photos appears.
- ChessReD's licence changes.
