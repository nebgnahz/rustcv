extern crate rustcv;
use rustcv::core::*;

#[test]
fn mat_new() {
    let mat = Mat::new();
    assert!(mat.empty());
}

#[test]
fn mat_with_size() {
    let mat = Mat::new_with_size(101, 102, CvType::Cv8UC1);
    assert_eq!(mat.rows(), 101);
    assert_eq!(mat.cols(), 102);
    assert_eq!(mat.channels(), 1);
    assert_eq!(mat.cv_type() as i32, 0);
}

#[test]
fn mat_clone() {
    let mat = Mat::new_with_size(101, 102, CvType::Cv8UC1);
    let clone = mat.clone();
    assert_eq!(clone.rows(), 101);
    assert_eq!(clone.cols(), 102);
}

#[test]
fn mat_copy_to() {
    let mat = Mat::new_with_size(101, 102, CvType::Cv8UC1);
    let mut dst = Mat::new();
    mat.copy_to(&mut dst);
    assert_eq!(dst.rows(), 101);
    assert_eq!(dst.cols(), 102);
}

#[test]
fn mat_copy_to_with_mask() {
    let mut mat = Mat::new_with_size(101, 102, CvType::Cv8UC1);
    let mut mask = Mat::new_with_size(101, 102, CvType::Cv8UC1);
    let mut diff = Mat::new();

    mat.set_uchar_at(0, 0, 255);
    mat.set_uchar_at(0, 1, 255);
    mask.set_uchar_at(0, 0, 255);

    let mut copy = Mat::new();
    mat.copy_to_with_mask(&mut copy, &mask);
    assert_eq!(copy.rows(), 101);
    assert_eq!(copy.cols(), 102);
    assert_eq!(copy.uchar_at(0, 0), 255);
    assert_eq!(copy.uchar_at(0, 1), 0);

    compare(&mat, &copy, &mut diff, CompareType::Eq);
    assert_ne!(count_non_zero(&diff), 0);
}
