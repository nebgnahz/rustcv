extern crate rustcv;

#[cfg(feature = "cuda")]
use rustcv::cuda::*;
#[cfg(feature = "cuda")]
use rustcv::imgcodecs::*;
use std::path::PathBuf;

#[allow(dead_code)]
fn asset_path(f: &str) -> PathBuf {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("assets");
    d.push(f);
    d
}

#[cfg(feature = "cuda")]
#[test]
fn test_gpu_mat() {
    let image = imread(asset_path("lenna.png"), ImageReadMode::Color).unwrap();
    assert!(!image.empty());
    let mut gpu_image = GpuMat::new();
    gpu_image.upload(&image);
    assert!(!gpu_image.empty());
    assert_eq!(gpu_image.rows(), 512);
    assert_eq!(gpu_image.cols(), 512);
}

#[cfg(feature = "cuda")]
#[test]
fn test_cascade_classifier() {
    let image = imread(asset_path("lenna.png"), ImageReadMode::Grayscale).unwrap();
    assert!(!image.empty());

    let model = asset_path("cuda_haarcascade_frontalface_default.xml");
    let classifier = GpuCascade::new(model).expect("failed to load cascade model");

    let mut gpu_image = GpuMat::new();
    gpu_image.upload(&image);

    let rects = classifier.detect_multiscale(&gpu_image);
    assert_eq!(rects.len(), 2);
    assert_eq!(rects[0].x, 33);
    assert_eq!(rects[0].y, 99);
    assert_eq!(rects[0].width, 24);
    assert_eq!(rects[0].height, 24);
    assert_eq!(rects[1].x, 219);
    assert_eq!(rects[1].y, 204);
    assert_eq!(rects[1].width, 167);
    assert_eq!(rects[1].height, 167);
}
