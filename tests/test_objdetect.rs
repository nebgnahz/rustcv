extern crate rustcv;
use rustcv::objdetect::*;
use rustcv::imgcodecs::*;
use std::path::PathBuf;

fn asset_path(f: &str) -> PathBuf {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("assets");
    d.push(f);
    d
}

#[test]
fn test_cascade_classifier() {
    let image =
        imread(asset_path("lenna.png"), ImageReadMode::Color).expect("failed to load lenna.png");
    assert!(!image.empty());

    let mut classifier = CascadeClassifier::new();
    classifier
        .load(asset_path("haarcascade_frontalface_default.xml"))
        .expect("failed to load cascade model");

    let rects = classifier.detect_multiscale(&image);
    assert_eq!(rects.len(), 1);
    assert_eq!(rects[0].x, 217);
    assert_eq!(rects[0].y, 201);
    assert_eq!(rects[0].width, 173);
    assert_eq!(rects[0].height, 173);
}
