use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "usage: fenrir [--models DIR] [--verbose] IMAGE...";

fn main() -> ExitCode {
    let mut models_dir = PathBuf::from("models");
    let mut verbose = false;
    let mut images = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--models" => match args.next() {
                Some(dir) => models_dir = dir.into(),
                None => return fail(USAGE),
            },
            "--verbose" | "-v" => verbose = true,
            "--help" | "-h" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            _ => images.push(PathBuf::from(arg)),
        }
    }
    if images.is_empty() {
        return fail(USAGE);
    }
    let start = Instant::now();
    let models = match fenrir::Models::load(&models_dir) {
        Ok(m) => m,
        Err(e) => return fail(&e.to_string()),
    };
    if verbose {
        eprintln!("loaded models in {:.2} s", start.elapsed().as_secs_f32());
    }
    let mut status = ExitCode::SUCCESS;
    for path in &images {
        let image = match fenrir::image::open(path) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("{}: {e}", path.display());
                status = ExitCode::FAILURE;
                continue;
            }
        };
        let start = Instant::now();
        match models.read(&image) {
            Ok(Some(d)) => {
                if verbose {
                    let unsure = d.confidence.iter().filter(|c| **c < 0.5).count();
                    eprintln!(
                        "{}: {:.0} ms, turned {}, from black {}, unsure squares {unsure}",
                        path.display(),
                        start.elapsed().as_secs_f64() * 1e3,
                        d.rotation,
                        d.from_black
                    );
                }
                println!("{}\t{}", path.display(), d.fen);
            }
            Ok(None) => println!("{}\t-", path.display()),
            Err(e) => {
                eprintln!("{}: {e}", path.display());
                status = ExitCode::FAILURE;
            }
        }
    }
    status
}

fn fail(message: &str) -> ExitCode {
    eprintln!("{message}");
    ExitCode::FAILURE
}
