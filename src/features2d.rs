//! [2D Features Framework][opencv-features2d].
//!
//! [opencv-features2d]: https://docs.opencv.org/master/da/d9b/group__features2d.html

use opencv_sys as ffi;
use core::*;

/// Maximally stable extremal region extractor.
#[derive(Debug)]
pub struct MSER {
    inner: ffi::MSER,
}

impl Drop for MSER {
    fn drop(&mut self) {
        unsafe { ffi::MSER_Close(self.inner) }
    }
}

impl MSER {
    /// Creates a new maximally stable extremal region (MSER) extractor criteria.
    pub fn new() -> Self {
        MSER {
            inner: unsafe { ffi::MSER_Create() },
        }
    }

    /// Detects keypoints in an image using MSER.
    pub fn detect(&self, src: &Mat) -> Vec<KeyPoint> {
        let keypoints = unsafe { ffi::MSER_Detect(self.inner, src.inner) };
        get_keypoints(keypoints)
    }
}

/// Class for extracting blobs from an image.
///
/// The class implements a simple algorithm for extracting blobs from an image:
///
/// 1. Convert the source image to binary images by applying thresholding with
/// several thresholds from minThreshold (inclusive) to maxThreshold (exclusive)
/// with distance thresholdStep between neighboring thresholds.
///
/// 2. Extract connected components from every binary image by findContours and
/// calculate their centers.
///
/// 3. Group centers from several binary images by their coordinates. Close
/// centers form one group that corresponds to one blob, which is controlled by
/// the minDistBetweenBlobs parameter.
///
/// 4. From the groups, estimate final centers of blobs and their radiuses and
/// return as locations and sizes of keypoints.
#[derive(Debug)]
pub struct SimpleBlobDetector {
    inner: ffi::SimpleBlobDetector,
}

impl Drop for SimpleBlobDetector {
    fn drop(&mut self) {
        unsafe { ffi::SimpleBlobDetector_Close(self.inner) }
    }
}

impl SimpleBlobDetector {
    /// Returns a new `SimpleBlobDetector`.
    pub fn new() -> Self {
        SimpleBlobDetector {
            inner: unsafe { ffi::SimpleBlobDetector_Create() },
        }
    }

    /// Detect keypoints in an image using SimpleBlobDetector.
    pub fn detect(&self, src: &Mat) -> Vec<KeyPoint> {
        let keypoints = unsafe { ffi::SimpleBlobDetector_Detect(self.inner, src.inner) };
        get_keypoints(keypoints)
    }
}

fn get_keypoints(keypoints: ffi::KeyPoints) -> Vec<KeyPoint> {
    (0..(keypoints.length as isize))
        .map(|i| unsafe { *keypoints.keypoints.offset(i) })
        .collect()
}
