//! [Object Detection](https://docs.opencv.org/master/d5/d54/group__objdetect.html).

use opencv_sys as ffi;
use std::ffi::CString;
use Error;
use CvError;
use core::{Mat, Rect, Size};
use std::path::Path;

/// Cascade classifier class for object detection.
#[derive(Debug)]
pub struct CascadeClassifier {
    inner: ffi::CascadeClassifier,
}

impl CascadeClassifier {
    /// Creates a cascade classifier, uninitialized. Before use, call load.
    pub fn new() -> Self {
        CascadeClassifier {
            inner: unsafe { ffi::CascadeClassifier_New() },
        }
    }

    /// Loads the classifier model from a path.
    pub fn load<P: AsRef<Path>>(&mut self, path: P) -> Result<(), Error> {
        if let Some(p) = path.as_ref().to_str() {
            let s = CString::new(p)?;
            let r = unsafe { ffi::CascadeClassifier_Load(self.inner, (&s).as_ptr()) };
            if r != 0 {
                Ok(())
            } else {
                Err(CvError::InvalidCascadeModel(path.as_ref().to_path_buf()).into())
            }
        } else {
            Err(CvError::InvalidPath(path.as_ref().to_path_buf()).into())
        }
    }

    /// Detects objects of different sizes in the input Mat image. The detected
    /// objects are returned as a vector of rectangles.
    pub fn detect_multiscale(&self, mat: &Mat) -> Vec<Rect> {
        let rects = unsafe { ffi::CascadeClassifier_DetectMultiScale(self.inner, mat.inner) };
        (0..(rects.length as isize))
            .map(|i| unsafe { *(rects.rects.offset(i)) })
            .collect()
    }

    /// Detects the object using parameters specified.
    ///
    /// * `mat` - Matrix of the type CV_8U containing an image where objects are
    ///   detected.
    /// * `scale_factor` - Parameter specifying how much the image size is
    ///   reduced at each image scale.
    /// * `min_neighbors` - Parameter specifying how many neighbors each
    ///   candidate rectangle should have to retain it.
    /// * `min_size` - Minimum possible object size. Objects smaller than that
    ///   are ignored.
    /// * `max_size` - Maximum possible object size. Objects larger than that
    ///   are ignored
    ///
    /// OpenCV has a parameter (`flags`) that's not used at all.
    pub fn detect_multiscale_with_params(
        &self,
        mat: &Mat,
        scale: f64,
        min_neighbors: i32,
        min_size: Size,
        max_size: Size,
    ) -> Vec<Rect> {
        let rects = unsafe {
            ffi::CascadeClassifier_DetectMultiScaleWithParams(
                self.inner,
                mat.inner,
                scale,
                min_neighbors,
                0,
                min_size,
                max_size,
            )
        };
        (0..(rects.length as isize))
            .map(|i| unsafe { *(rects.rects.offset(i)) })
            .collect()
    }
}

impl Drop for CascadeClassifier {
    fn drop(&mut self) {
        unsafe { ffi::CascadeClassifier_Close(self.inner) }
    }
}
