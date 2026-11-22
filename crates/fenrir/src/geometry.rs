use image::GrayImage;
use imageproc::contours::{BorderType, find_contours};
use imageproc::geometry::{arc_length, min_area_rect};
use imageproc::point::Point;

/// Four corners in the order top left, top right, bottom right, bottom left.
pub type Quad = [(f32, f32); 4];

/// The board outline in a mask of `size` x `size` probabilities: the largest
/// outer contour above `threshold`, fitted to four corners.
pub(crate) fn quad_from_mask(mask: &[f32], size: u32, threshold: f32) -> Option<Quad> {
    let binary = GrayImage::from_fn(size, size, |x, y| {
        image::Luma([if mask[(y * size + x) as usize] > threshold {
            255
        } else {
            0
        }])
    });
    let largest = find_contours::<i32>(&binary)
        .into_iter()
        .filter(|c| c.border_type == BorderType::Outer && c.parent.is_none())
        .max_by(|a, b| area(&a.points).total_cmp(&area(&b.points)))?;
    Some(order(&fit_quad(&largest.points)))
}

/// The shoelace area, as OpenCV's `contourArea` gives.
fn area(points: &[Point<i32>]) -> f64 {
    let n = points.len();
    let mut sum = 0.0;
    for i in 0..n {
        let (a, b) = (points[i], points[(i + 1) % n]);
        sum += a.x as f64 * b.y as f64 - b.x as f64 * a.y as f64;
    }
    (sum / 2.0).abs()
}

/// Searches for the polygon tolerance that leaves four corners. Falls back
/// to the smallest rotated rectangle around the contour.
fn fit_quad(points: &[Point<i32>]) -> [(f32, f32); 4] {
    let perimeter = arc_length(points, true);
    let (mut low, mut high) = (0.0, 0.2);
    for _ in 0..20 {
        let eps = (low + high) / 2.0;
        let corners = simplify_closed(points, eps * perimeter);
        if corners.len() == 4 {
            return [corners[0], corners[1], corners[2], corners[3]];
        }
        if corners.len() > 4 {
            low = eps;
        } else {
            high = eps;
        }
    }
    min_area_rect(points).map(|p| (p.x as f32, p.y as f32))
}

/// Douglas-Peucker on a closed contour. The contour is split at two points
/// far apart, as OpenCV's `approxPolyDP` does, so no corner depends on where
/// the contour starts.
fn simplify_closed(points: &[Point<i32>], eps: f64) -> Vec<(f32, f32)> {
    let n = points.len();
    if n < 3 {
        return points.iter().map(|p| (p.x as f32, p.y as f32)).collect();
    }
    let far = |from: usize| {
        (0..n)
            .max_by(|&a, &b| {
                dist2(points[from], points[a]).total_cmp(&dist2(points[from], points[b]))
            })
            .unwrap()
    };
    let mut a = 0;
    let mut b = far(a);
    for _ in 0..3 {
        a = far(b);
        b = far(a);
    }
    let (a, b) = (a.min(b), a.max(b));
    let first: Vec<Point<i32>> = points[a..=b].to_vec();
    let second: Vec<Point<i32>> = points[b..].iter().chain(&points[..=a]).copied().collect();
    let mut keep = Vec::new();
    for chain in [&first, &second] {
        let mut marks = vec![false; chain.len()];
        marks[0] = true;
        douglas_peucker(chain, 0, chain.len() - 1, eps, &mut marks);
        keep.extend(
            chain
                .iter()
                .zip(&marks)
                .filter(|(_, m)| **m)
                .map(|(p, _)| (p.x as f32, p.y as f32)),
        );
    }
    keep
}

fn douglas_peucker(chain: &[Point<i32>], start: usize, end: usize, eps: f64, marks: &mut [bool]) {
    if end <= start + 1 {
        return;
    }
    let (p, q) = (chain[start], chain[end]);
    let (dx, dy) = ((q.x - p.x) as f64, (q.y - p.y) as f64);
    let length = (dx * dx + dy * dy).sqrt();
    let mut best = (0.0, start);
    for (i, r) in chain.iter().enumerate().take(end).skip(start + 1) {
        let d = if length == 0.0 {
            dist2(p, *r).sqrt()
        } else {
            (dy * (r.x - p.x) as f64 - dx * (r.y - p.y) as f64).abs() / length
        };
        if d > best.0 {
            best = (d, i);
        }
    }
    if best.0 > eps {
        marks[best.1] = true;
        douglas_peucker(chain, start, best.1, eps, marks);
        douglas_peucker(chain, best.1, end, eps, marks);
    }
}

fn dist2(a: Point<i32>, b: Point<i32>) -> f64 {
    let (dx, dy) = ((a.x - b.x) as f64, (a.y - b.y) as f64);
    dx * dx + dy * dy
}

fn order(points: &[(f32, f32); 4]) -> Quad {
    let by = |key: fn(&(f32, f32)) -> f32, max: bool| {
        let it = points.iter().copied();
        if max {
            it.max_by(|a, b| key(a).total_cmp(&key(b))).unwrap()
        } else {
            it.min_by(|a, b| key(a).total_cmp(&key(b))).unwrap()
        }
    };
    let sum = |p: &(f32, f32)| p.0 + p.1;
    let diff = |p: &(f32, f32)| p.1 - p.0;
    [
        by(sum, false),
        by(diff, false),
        by(sum, true),
        by(diff, true),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_square_mask_gives_its_corners() {
        let size = 64;
        let mut mask = vec![0.0; (size * size) as usize];
        for y in 10..50 {
            for x in 12..40 {
                mask[(y * size + x) as usize] = 0.9;
            }
        }
        let quad = quad_from_mask(&mask, size, 0.5).unwrap();
        assert_eq!(
            quad,
            [(12.0, 10.0), (39.0, 10.0), (39.0, 49.0), (12.0, 49.0)]
        );
    }

    #[test]
    fn an_empty_mask_gives_none() {
        assert_eq!(quad_from_mask(&[0.1; 16], 4, 0.5), None);
    }
}
