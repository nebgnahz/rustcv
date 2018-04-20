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

fn load(f: &str) -> Mat {
    let src = imread(asset_path(f), ImageReadMode::Color).expect("failed to load lenna");
    assert!(!src.empty());
    src
}

fn lenna() -> Mat {
    load("lenna.png")
}

fn messi() -> Mat {
    load("messi5.jpg")
}

fn messi_face() -> Mat {
    load("messi_face.jpg")
}

#[test]
fn test_canny() {
    let src = lenna();
    let mut dst = Mat::new();

    canny(&src, &mut dst, 50.0, 150.0);
    assert!(!dst.empty());
    assert_eq!(src.rows(), dst.rows());
    assert_eq!(src.cols(), dst.cols());
}

#[test]
fn test_gaussian_blur() {
    let src = lenna();
    let mut dst = Mat::new();

    let ksize = Size {
        width: 23,
        height: 23,
    };

    gaussian_blur(&src, &mut dst, ksize, 30.0, 50.0, BorderType::Reflect101);
    assert!(!dst.empty());
    assert_eq!(src.rows(), dst.rows());
    assert_eq!(src.cols(), dst.cols());
}

#[test]
fn test_match_template() {
    let messi = messi();
    let face = messi_face();
    let mut result = Mat::new();
    match_template(
        &messi,
        &face,
        &mut result,
        TemplateMatchMode::CcoeffNormed,
        &Mat::new(),
    );
    let (_, max_conf, _, _) = min_max_loc(&result);
    assert!(max_conf > 0.95);
}

#[test]
fn test_median_blur() {
    let src = lenna();
    let mut dst = Mat::new();
    median_blur(&src, &mut dst, 1);
    assert!(!dst.empty());
    assert_eq!(src.rows(), dst.rows());
    assert_eq!(src.cols(), dst.cols());
}

#[test]
fn test_pyr_down() {
    let src = lenna();
    let mut dst = Mat::new();
    let size = Size {
        width: dst.cols(),
        height: dst.rows(),
    };
    pyr_down(&src, &mut dst, size, BorderType::Reflect101);
    assert!(!dst.empty());
    assert!((src.cols() - 2 * dst.cols()).abs() < 2);
    assert!((src.rows() - 2 * dst.rows()).abs() < 2);
}

#[test]
fn test_pyr_up() {
    let src = lenna();
    let mut dst = Mat::new();
    let size = Size {
        width: dst.cols(),
        height: dst.rows(),
    };
    pyr_up(&src, &mut dst, size, BorderType::Reflect101);
    assert!(!dst.empty());
    assert!((2 * src.cols() - dst.cols()).abs() < 2);
    assert!((2 * src.rows() - dst.rows()).abs() < 2);
}

#[test]
fn test_resize() {
    let src = lenna();
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
