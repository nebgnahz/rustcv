//! [CUDA][opencv-cuda].
//!
//! [opencv-cuda]: https://docs.opencv.org/master/d1/d1a/namespacecv_1_1cuda.html

use opencv_sys as ffi;
use core::*;
use {CvError, Error};
use std::ffi::CString;
use std::path::Path;

/// Base storage class for GPU memory with reference counting.
///
/// Its interface matches the Mat interface with the following limitations: (1)
/// no arbitrary dimensions support (only 2D), (2) no functions that return
/// references to their data (because references on GPU are not valid for CPU),
/// (3) no expression templates technique support/
///
/// Beware that the latter limitation may lead to overloaded matrix operators
/// that cause memory allocations. The GpuMat class is convertible to
/// cuda::PtrStepSz and cuda::PtrStep so it can be passed directly to the
/// kernel.
#[derive(Debug)]
pub struct GpuMat {
    pub(crate) inner: ffi::GpuMat,
}

impl Drop for GpuMat {
    fn drop(&mut self) {
        unsafe { ffi::GpuMat_Close(self.inner) }
    }
}

impl GpuMat {
    /// Default constructor.
    pub fn new() -> GpuMat {
        GpuMat {
            inner: unsafe { ffi::GpuMat_New() },
        }
    }

    /// Number of rows.
    pub fn rows(&self) -> i32 {
        unsafe { ffi::GpuMat_Rows(self.inner) }
    }

    /// Number of cols.
    pub fn cols(&self) -> i32 {
        unsafe { ffi::GpuMat_Cols(self.inner) }
    }

    /// Whether or not the GpuMat is empty.
    pub fn empty(&self) -> bool {
        unsafe { ffi::GpuMat_Empty(self.inner) != 0 }
    }

    /// Uploads a normal `Mat`
    pub fn upload(&mut self, mat: &Mat) {
        unsafe { ffi::GpuMat_Upload(self.inner, mat.inner) }
    }
}

/// Data structure that performs object detection with a cascade classifier.
#[derive(Debug)]
pub struct GpuCascade {
    pub(crate) inner: ffi::GpuCascade,
}

impl Drop for GpuCascade {
    fn drop(&mut self) {
        unsafe { ffi::GpuCascade_Close(self.inner) }
    }
}

impl GpuCascade {
    /// Creates the classifier from a file.
    ///
    /// Name of the file from which the classifier is loaded. Only the old haar
    /// classifier (trained by the haar training application) and NVIDIA's nvbin
    /// are supported for HAAR and only new type of OpenCV XML cascade supported
    /// for LBP. The working haar models can be found at
    /// opencv_folder/data/haarcascades_cuda/.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        if let Some(p) = path.as_ref().to_str() {
            let s = CString::new(p)?;
            let c = unsafe { ffi::GpuCascade_Create((&s).as_ptr()) };
            Ok(GpuCascade { inner: c })
        } else {
            Err(CvError::InvalidPath(path.as_ref().to_path_buf()).into())
        }
    }

    /// Detects objects of different sizes in the input image.
    pub fn detect_multiscale(&self, mat: &GpuMat) -> Vec<Rect> {
        let rects = unsafe { ffi::GpuCascade_DetectMultiScale(self.inner, mat.inner) };
        (0..(rects.length as isize))
            .map(|i| unsafe { *(rects.rects.offset(i)) })
            .collect()
    }
}
