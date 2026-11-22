/// The piece symbols in the order of the position model's output channels.
/// Channel 12 is an empty square.
pub const PIECES: [char; 12] = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k'];
pub(crate) const CHANNELS: usize = PIECES.len() + 1;

/// The 64 squares from a8 to h1, rank by rank, as in a FEN.
pub type Grid = [Option<char>; 64];

/// The piece placement field of a FEN.
pub fn placement(grid: &Grid) -> String {
    let mut out = String::new();
    for (rank, row) in grid.chunks(8).enumerate() {
        if rank > 0 {
            out.push('/');
        }
        let mut empty = 0;
        for square in row {
            match square {
                Some(piece) => {
                    if empty > 0 {
                        out.push(char::from_digit(empty, 10).unwrap());
                        empty = 0;
                    }
                    out.push(*piece);
                }
                None => empty += 1,
            }
        }
        if empty > 0 {
            out.push(char::from_digit(empty, 10).unwrap());
        }
    }
    out
}

/// The board seen from the other side: turned half a circle, with the
/// colours swapped.
pub(crate) fn turn(grid: &Grid) -> Grid {
    let mut out = [None; 64];
    for (i, square) in grid.iter().enumerate() {
        out[63 - i] = square.map(swap_colour);
    }
    out
}

fn swap_colour(piece: char) -> char {
    if piece.is_ascii_uppercase() {
        piece.to_ascii_lowercase()
    } else {
        piece.to_ascii_uppercase()
    }
}

/// The orientation model's input: one row of [`CHANNELS`] for each square.
pub(crate) fn one_hot(grid: &Grid) -> Vec<f32> {
    let mut out = vec![0.0; 64 * CHANNELS];
    for (i, square) in grid.iter().enumerate() {
        let channel = square
            .and_then(|p| PIECES.iter().position(|&q| q == p))
            .unwrap_or(PIECES.len());
        out[i * CHANNELS + channel] = 1.0;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(placement: &str) -> Grid {
        let mut out = [None; 64];
        let mut i = 0;
        for c in placement.chars() {
            match c {
                '/' => {}
                '1'..='8' => i += c.to_digit(10).unwrap() as usize,
                _ => {
                    out[i] = Some(c);
                    i += 1;
                }
            }
        }
        out
    }

    #[test]
    fn placement_round_trips() {
        let start = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR";
        assert_eq!(placement(&grid(start)), start);
    }

    #[test]
    fn turning_swaps_sides() {
        let g = grid("4k3/8/8/8/8/8/8/R3K3");
        assert_eq!(placement(&turn(&g)), "3k3r/8/8/8/8/8/8/3K4");
    }

    #[test]
    fn one_hot_marks_empty_squares() {
        let v = one_hot(&grid("K7/8/8/8/8/8/8/8"));
        assert_eq!(v[5], 1.0);
        assert_eq!(v[CHANNELS + 12], 1.0);
        assert_eq!(v.iter().sum::<f32>(), 64.0);
    }
}
