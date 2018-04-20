extern crate rustcv;
use rustcv::core::*;
use rustcv::imgcodecs::*;
use rustcv::dnn::*;
use std::path::PathBuf;
use std::env;

fn asset_path(f: &str) -> PathBuf {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("assets");
    d.push(f);
    d
}

fn caffe_path(f: &str) -> String {
    let mut path = env::var("RUSTCV_CAFFE_TEST_FILES").unwrap();
    path += "/";
    path += f;
    path
}

#[test]
#[ignore]
fn test_caffe() {
    let proto = caffe_path("bvlc_googlenet.prototxt");
    let model = caffe_path("bvlc_googlenet.caffemodel");

    let mut net = Net::from_caffe(&proto, &model).unwrap();
    let image = imread(asset_path("space_shuttle.jpg"), ImageReadMode::Color)
        .expect("failed to load space.png");
    let blob = blob_from_image(
        &image,
        1.0,
        Size {
            width: 224,
            height: 224,
        },
        Scalar {
            val1: 104.0,
            val2: 117.0,
            val3: 123.0,
            val4: 0.0,
        },
        false,
        false,
    );

    assert!(!blob.empty());
    net.set_input(&blob, "data").unwrap();
    let prob = net.forward("prob").unwrap();
    assert!(!prob.empty());
    let prob_mat = prob.reshape(1, 1);
    let (_, max, min_loc, max_loc) = min_max_loc(&prob_mat);
    assert_eq!((max * 10000.0).round() / 10000.0, 0.9999);
    assert_eq!(min_loc.x, 793);
    assert_eq!(min_loc.y, 0);
    assert_eq!(max_loc.x, 812);
    assert_eq!(max_loc.y, 0);
}
