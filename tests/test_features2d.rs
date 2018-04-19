extern crate rustcv;
use rustcv::imgcodecs::*;
use rustcv::features2d::*;
use std::path::PathBuf;

fn asset_path(f: &str) -> PathBuf {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("assets");
    d.push(f);
    d
}

#[test]
fn test_mser() {
    let image = imread(asset_path("lenna.png"), ImageReadMode::Color).unwrap();
    assert!(!image.empty());
    let mser = MSER::new();
    let keypoints = mser.detect(&image);
    assert_eq!(keypoints.len(), 228);
}
