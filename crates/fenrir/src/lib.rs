//! Reads a chess diagram in an image and returns its position as a FEN.
//!
//! ```no_run
//! let models = fenrir::Models::load("models".as_ref())?;
//! let image = image::open("diagram.png")?;
//! if let Some(found) = models.read(&image)? {
//!     println!("{}", found.fen);
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod board;
mod geometry;

use std::path::Path;

use image::imageops::{self, FilterType};
use image::{DynamicImage, Rgb, RgbImage};
use imageproc::geometric_transformations::{Border, Interpolation, Projection, warp_into};
use tract_onnx::prelude::*;

pub use board::{Grid, PIECES, placement};
pub use geometry::Quad;
pub use image;

const MASK_SIZE: u32 = 512;
const BOARD_SIZE: u32 = 256;
const ATTEMPTS: usize = 10;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("model {name}: {source}")]
    Model {
        name: &'static str,
        source: tract_onnx::prelude::TractError,
    },
}

type Plan = std::sync::Arc<TypedSimplePlan>;

/// The four models of the diagram pipeline.
pub struct Models {
    quad: Plan,
    rotation: Plan,
    position: Plan,
    orientation: Plan,
}

/// A diagram found in an image.
#[derive(Debug, Clone)]
pub struct Diagram {
    /// The position, with White to move and no castling rights.
    pub fen: String,
    pub grid: Grid,
    /// The model's score for the chosen piece or empty square, from 0 to 1.
    pub confidence: [f32; 64],
    /// The board's corners in the input image's pixels.
    pub corners: Quad,
    /// How far the board was turned clockwise to stand upright: 0, 90, 180 or 270.
    pub rotation: u16,
    /// The diagram shows the board from Black's side.
    pub from_black: bool,
}

impl Models {
    /// Loads `quad.onnx`, `rotation.onnx`, `position.onnx` and `orientation.onnx` from `dir`.
    pub fn load(dir: &Path) -> Result<Self, Error> {
        Ok(Self {
            quad: plan(dir, "quad", &[1, 3, MASK_SIZE as usize, MASK_SIZE as usize])?,
            rotation: plan(
                dir,
                "rotation",
                &[1, 3, BOARD_SIZE as usize, BOARD_SIZE as usize],
            )?,
            position: plan(
                dir,
                "position",
                &[1, 3, BOARD_SIZE as usize, BOARD_SIZE as usize],
            )?,
            orientation: plan(dir, "orientation", &[1, 64, board::CHANNELS])?,
        })
    }

    /// Finds the most prominent diagram in `image` and reads its position.
    /// Returns `None` when no board is found or no piece is read.
    pub fn read(&self, image: &DynamicImage) -> Result<Option<Diagram>, Error> {
        let Some((board, corners)) = self.find_board(image.to_rgb8())? else {
            return Ok(None);
        };
        let rotation =
            argmax(&self.run(&self.rotation, "rotation", tensor(&board, BOARD_SIZE, true))?) as u16
                * 90;
        let board = match rotation {
            90 => imageops::rotate90(&board),
            180 => imageops::rotate180(&board),
            270 => imageops::rotate270(&board),
            _ => board,
        };
        if board.width() < 32 || board.height() < 32 {
            return Ok(None);
        }
        let scores = self.run(&self.position, "position", tensor(&board, BOARD_SIZE, true))?;
        let mut grid: Grid = [None; 64];
        let mut confidence = [0.0; 64];
        for (i, square) in scores.chunks(board::CHANNELS).enumerate() {
            let clipped: Vec<f32> = square.iter().map(|v| v.clamp(0.0, 1.0)).collect();
            let best = argmax(&clipped);
            grid[i] = PIECES.get(best).copied();
            confidence[i] = clipped[best];
        }
        if grid.iter().all(Option::is_none) {
            return Ok(None);
        }
        let input = Tensor::from_shape(&[1, 64, board::CHANNELS], &board::one_hot(&grid)).map_err(
            |source| Error::Model {
                name: "orientation",
                source,
            },
        )?;
        let logit = self.run(&self.orientation, "orientation", input)?[0];
        let from_black = logit - 0.2 > 0.5;
        if from_black {
            grid = board::turn(&grid);
            confidence.reverse();
        }
        Ok(Some(Diagram {
            fen: format!("{} w - - 0 1", placement(&grid)),
            grid,
            confidence,
            corners,
            rotation,
            from_black,
        }))
    }

    /// Pads the image, then narrows in on the board until it fills most of
    /// the view, and warps it square.
    fn find_board(&self, image: RgbImage) -> Result<Option<(RgbImage, Quad)>, Error> {
        let (pad_x, pad_y) = (
            (image.width() as f32 * 0.05) as u32,
            (image.height() as f32 * 0.05) as u32,
        );
        let mut view = RgbImage::from_pixel(
            image.width() + 2 * pad_x,
            image.height() + 2 * pad_y,
            Rgb([255; 3]),
        );
        imageops::replace(&mut view, &image, pad_x as i64, pad_y as i64);
        let mut origin = (-(pad_x as f32), -(pad_y as f32));
        for _ in 0..ATTEMPTS {
            if view.width() == 0 || view.height() == 0 {
                return Ok(None);
            }
            let logits = self.run(&self.quad, "quad", tensor(&view, MASK_SIZE, false))?;
            let mask: Vec<f32> = logits.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
            let Some(quad) = geometry::quad_from_mask(&mask, MASK_SIZE, 0.5) else {
                return Ok(None);
            };
            if std::env::var_os("FENRIR_TRACE").is_some() {
                eprintln!("view {}x{} quad {:?}", view.width(), view.height(), quad);
            }
            let (sx, sy) = (
                view.width() as f32 / MASK_SIZE as f32,
                view.height() as f32 / MASK_SIZE as f32,
            );
            let corners = quad.map(|(x, y)| (x * sx, y * sy));
            let xs = corners.map(|c| c.0);
            let ys = corners.map(|c| c.1);
            let (x1, x2) = (min(&xs), max(&xs));
            let (y1, y2) = (min(&ys), max(&ys));
            let (w, h) = (x2 - x1, y2 - y1);
            if w / view.width() as f32 > 0.7 && h / view.height() as f32 > 0.7 {
                let side = (0..4)
                    .map(|i| distance(corners[i], corners[(i + 1) % 4]))
                    .sum::<f32>()
                    / 4.0;
                let board = warp(&view, &corners, (side as u32).max(32));
                let found = corners.map(|(x, y)| (x + origin.0, y + origin.1));
                return Ok(board.map(|b| (b, found)));
            }
            let cx1 = (x1 - w * 0.1).max(0.0) as u32;
            let cy1 = (y1 - h * 0.1).max(0.0) as u32;
            let cx2 = ((x2 + w * 0.1).min(view.width() as f32) as u32).max(cx1);
            let cy2 = ((y2 + h * 0.1).min(view.height() as f32) as u32).max(cy1);
            view = imageops::crop_imm(&view, cx1, cy1, cx2 - cx1, cy2 - cy1).to_image();
            origin = (origin.0 + cx1 as f32, origin.1 + cy1 as f32);
        }
        Ok(None)
    }

    fn run(&self, plan: &Plan, name: &'static str, input: Tensor) -> Result<Vec<f32>, Error> {
        let model = |source| Error::Model { name, source };
        let out = plan.run(tvec!(input.into())).map_err(model)?;
        let view = out[0].to_plain_array_view::<f32>().map_err(model)?;
        Ok(view.iter().copied().collect())
    }
}

fn plan(dir: &Path, name: &'static str, shape: &[usize]) -> Result<Plan, Error> {
    tract_onnx::onnx()
        .with_ignore_value_info(true)
        .model_for_path(dir.join(format!("{name}.onnx")))
        .and_then(|m| m.with_input_fact(0, f32::fact(shape).into()))
        .and_then(|m| m.into_optimized())
        .and_then(|m| m.into_runnable())
        .map_err(|source| Error::Model { name, source })
}

/// The image resized to `size` x `size` as a 1x3xHxW tensor, scaled to 0..1
/// over all channels, then centred on its mean when `centre` is set.
fn tensor(image: &RgbImage, size: u32, centre: bool) -> Tensor {
    let resized = imageops::resize(image, size, size, FilterType::CatmullRom);
    let n = (size * size) as usize;
    let mut data = vec![0.0f32; 3 * n];
    for (i, pixel) in resized.pixels().enumerate() {
        for c in 0..3 {
            data[c * n + i] = pixel[c] as f32 / 255.0;
        }
    }
    let (lo, hi) = (min(&data), max(&data));
    if hi > lo {
        data.iter_mut().for_each(|v| *v = (*v - lo) / (hi - lo));
    } else {
        data.iter_mut().for_each(|v| *v = 0.0);
    }
    if centre {
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        data.iter_mut().for_each(|v| *v -= mean);
    }
    Tensor::from_shape(&[1, 3, size as usize, size as usize], &data).expect("shape matches data")
}

fn warp(image: &RgbImage, corners: &Quad, side: u32) -> Option<RgbImage> {
    let s = side as f32;
    let projection =
        Projection::from_control_points(*corners, [(0.0, 0.0), (s, 0.0), (s, s), (0.0, s)])?;
    let mut out = RgbImage::new(side, side);
    warp_into(
        image,
        projection,
        Interpolation::Bicubic,
        Border::Constant(Rgb([255; 3])),
        &mut out,
    );
    Some(out)
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best
}

fn min(values: &[f32]) -> f32 {
    values.iter().copied().fold(f32::INFINITY, f32::min)
}

fn max(values: &[f32]) -> f32 {
    values.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

fn distance(a: (f32, f32), b: (f32, f32)) -> f32 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}
