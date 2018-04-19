extern crate rustcv;
use rustcv::core::*;
use rustcv::imgproc::*;
use rustcv::imgcodecs::*;
use std::path::PathBuf;

fn asset_path(f: &str) -> PathBuf {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("assets");
    d.push(f);
    d
}

#[test]
fn test_resize() {
    let src = imread(asset_path("lenna.png"), ImageReadMode::Color).expect("failed to load lenna");
    assert!(!src.empty());
    let mut dst = Mat::new();

    let target_size = Size {
        width: 0,
        height: 0,
    };
    resize(
        &src,
        &mut dst,
        target_size,
        0.5,
        0.5,
        InterpolationFlag::Linear,
    );
    assert_eq!(dst.cols(), 256);
    assert_eq!(dst.rows(), 256);

    let target_size = Size {
        width: 440,
        height: 377,
    };
    resize(
        &src,
        &mut dst,
        target_size,
        0.0,
        0.0,
        InterpolationFlag::Cubic,
    );
    assert_eq!(dst.cols(), 440);
    assert_eq!(dst.rows(), 377);
}
