//! [Core data structures in OpenCV][opencv-core].
//!
//! [opencv-core]: https://docs.opencv.org/master/d0/de1/group__core.html
use opencv_sys as ffi;

/// The class `Mat` represents an n-dimensional dense numerical single-channel or multi-channel array.
/// It can be used to store real or complex-valued vectors and matrices, grayscale or color images,
/// voxel volumes, vector fields, point clouds, tensors, histograms
#[derive(Debug)]
pub struct Mat {
    pub(crate) inner: ffi::Mat,
}

pub use opencv_sys::Scalar;

/// Here is the `CvType` in an easy-to-read table.
///
/// |        | C1 | C2 | C3 | C4 | C(5) | C(6) | C(7) | C(8) |
/// |--------|----|----|----|----|------|------|------|------|
/// | CV_8U  |  0 |  8 | 16 | 24 |   32 |   40 |   48 |   56 |
/// | CV_8S  |  1 |  9 | 17 | 25 |   33 |   41 |   49 |   57 |
/// | CV_16U |  2 | 10 | 18 | 26 |   34 |   42 |   50 |   58 |
/// | CV_16S |  3 | 11 | 19 | 27 |   35 |   43 |   51 |   59 |
/// | CV_32S |  4 | 12 | 20 | 28 |   36 |   44 |   52 |   60 |
/// | CV_32F |  5 | 13 | 21 | 29 |   37 |   45 |   53 |   61 |
/// | CV_64F |  6 | 14 | 22 | 30 |   38 |   46 |   54 |   62 |
#[repr(C)]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum CvType {
    /// 8 bit unsigned, single channel (grey image)
    Cv8UC1 = 0,
    /// 8 bit signed, single channel (grey image)
    Cv8SC1 = 1,
    /// 16 bit unsigned, single channel (grey image)
    Cv16UC1 = 2,
    /// 16 bit signed, single channel (grey image)
    Cv16SC1 = 3,
    /// 32 bit signed, single channel (grey image)
    Cv32SC1 = 4,
    /// 32 bit float, single channel (grey image)
    Cv32FC1 = 5,
    /// 32 bit float, single channel (grey image)
    Cv64FC1 = 6,
    /// 8 bit, two channel (rarelly seen)
    Cv8UC2 = 8,
    /// 8 bit unsigned, three channels (RGB image)
    Cv8UC3 = 16,
    /// 8 bit signed, three channels (RGB image)
    Cv8SC3 = 17,
    /// 16 bit unsigned, three channels (RGB image)
    Cv16UC3 = 18,
    /// 16 bit signed, three channels (RGB image)
    Cv16SC3 = 19,
    /// 32 bit signed, three channels (RGB image)
    Cv32SC3 = 20,
    /// 32 bit float, three channels (RGB image)
    Cv32FC3 = 21,
    /// 32 bit float, three channels (RGB image)
    Cv64FC3 = 22,
}

impl From<ffi::Mat> for Mat {
    fn from(inner: ffi::Mat) -> Mat {
        Mat { inner: inner }
    }
}

impl Mat {
    /// Creates an empty `Mat` struct.
    pub fn new() -> Mat {
        Mat {
            inner: unsafe { ffi::Mat_New() },
        }
    }

    /// Creates an empty `Mat` struct with size.
    pub fn new_with_size(rows: i32, cols: i32, t: CvType) -> Mat {
        Mat {
            inner: unsafe { ffi::Mat_NewWithSize(rows, cols, t as i32) },
        }
    }

    /// Creates an empty `Mat` struct with a constant scalar.
    pub fn new_from_scalar(s: Scalar, t: CvType) -> Mat {
        Mat {
            inner: unsafe { ffi::Mat_NewFromScalar(s, t as i32) },
        }
    }

    /// Creates an empty `Mat` struct from buffer.
    pub fn new_from_bytes(rows: i32, cols: i32, t: CvType, buf: &mut [i8]) -> Mat {
        Mat {
            inner: unsafe { ffi::Mat_NewFromBytes(rows, cols, t as i32, to_byte_array(buf)) },
        }
    }

    /// Determines if the Mat is empty or not.
    pub fn empty(&self) -> bool {
        unsafe { ffi::Mat_Empty(self.inner) != 0 }
    }

    /// Copies Mat into destination Mat.
    ///
    /// For further details, please see [OpenCV
    /// documentation](https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#a33fd5d125b4c302b0c9aa86980791a77)
    pub fn copy_to(&self, dst: &mut Mat) {
        unsafe { ffi::Mat_CopyTo(self.inner, dst.inner) }
    }

    /// Copies Mat into destination Mat, with a mask.
    pub fn copy_to_with_mask(&self, dst: &mut Mat, mask: &Mat) {
        unsafe { ffi::Mat_CopyToWithMask(self.inner, dst.inner, mask.inner) }
    }

    /// Converts Mat into destination Mat.
    ///
    /// For further details, please see [OpenCV
    /// documentation](https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#adf88c60c5b4980e05bb556080916978b)
    pub fn convert_to(&self, dst: &mut Mat, t: CvType) {
        unsafe { ffi::Mat_ConvertTo(self.inner, dst.inner, t as i32) }
    }

    /// Copies the underlying Mat data to a byte array.
    ///
    /// For further details, please see [OpenCV
    /// documentation](https://docs.opencv.org/3.3.1/d3/d63/classcv_1_1Mat.html#a4d33bed1c850265370d2af0ff02e1564)
    pub fn to_bytes(&self) -> Vec<u8> {
        let array = unsafe { ffi::Mat_ToBytes(self.inner) };
        from_byte_array(&array)
    }
}

impl Clone for Mat {
    fn clone(&self) -> Self {
        Mat {
            inner: unsafe { ffi::Mat_Clone(self.inner) },
        }
    }
}

fn to_byte_array(buf: &mut [i8]) -> ffi::ByteArray {
    ffi::ByteArray {
        data: buf.as_mut_ptr(),
        length: buf.len() as i32,
    }
}

fn from_byte_array(arr: &ffi::ByteArray) -> Vec<u8> {
    unsafe {
        Vec::from_raw_parts(
            ::std::mem::transmute(arr.data),
            arr.length as usize,
            arr.length as usize,
        )
    }
}

impl Drop for Mat {
    fn drop(&mut self) {
        unsafe { ffi::Mat_Close(self.inner) }
    }
}
