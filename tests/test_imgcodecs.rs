extern crate tempdir;
extern crate rustcv;
use rustcv::imgcodecs::*;
use std::path::PathBuf;

fn asset_path(f: &str) -> PathBuf {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("assets");
    d.push(f);
    d
}

#[test]
fn test_imread() {
    let src = imread(asset_path("lenna.png"), ImageReadMode::Color).expect("failed to load lenna");
    assert!(!src.empty());
}

#[test]
fn test_write() {
    let src = imread(asset_path("lenna.png"), ImageReadMode::Color).expect("failed to load lenna");
    assert!(!src.empty());

    let temp_dir = tempdir::TempDir::new("out").unwrap();
    let temp_file = temp_dir.path().join("lenna.png");
    imwrite(&temp_file, &src).expect("failed to write lenna to tempfile");

    let src2 = imread(temp_file, ImageReadMode::Color).expect("failed to load stored file");
    assert_eq!(src2.cols(), 512);
    assert_eq!(src2.rows(), 512);
}
