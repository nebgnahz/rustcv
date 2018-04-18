//! [Core data structures in OpenCV][opencv-core].
//!
//! [opencv-core]: https://docs.opencv.org/master/d0/de1/group__core.html
use opencv_sys as ffi;
use num_traits::FromPrimitive;

/// The class `Mat` represents an n-dimensional dense numerical single-channel or multi-channel array.
/// It can be used to store real or complex-valued vectors and matrices, grayscale or color images,
/// voxel volumes, vector fields, point clouds, tensors, histograms
#[derive(Debug)]
pub struct Mat {
    pub(crate) inner: ffi::Mat,
}

pub use opencv_sys::Scalar;
pub use opencv_sys::Rect;

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
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, FromPrimitive, ToPrimitive)]
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

    /// Returns a new Mat that points to a region of this Mat. Changes made to
    /// the region Mat will affect the original Mat, since they are pointers to
    /// the underlying OpenCV Mat object.
    pub fn region(&self, roi: Rect) -> Mat {
        Mat::from(unsafe { ffi::Mat_Region(self.inner, roi) })
    }

    /// Changes the shape and/or the number of channels of a 2D matrix without
    /// copying the data. The method makes a new matrix header for the internal
    /// data.
    pub fn reshape(&self, channel: i32, rows: i32) -> Mat {
        Mat::from(unsafe { ffi::Mat_Reshape(self.inner, channel, rows) })
    }

    /// Converts a Mat to half-precision floating point.
    ///
    /// This function converts FP32 (single precision floating point) from/to
    /// FP16 (half precision floating point). The input array has to have type
    /// of CV_32F or CV_16S to represent the bit depth. If the input array is
    /// neither of them, the function will raise an error. The format of half
    /// precision floating point is defined in IEEE 754-2008.
    pub fn convert_fp16(&self) -> Mat {
        Mat::from(unsafe { ffi::Mat_ConvertFp16(self.inner) })
    }

    /// Calculates the mean value M of array elements, independently for each
    /// channel, and return it as Scalar.
    /// TODO: pass second paramter with mask
    pub fn mean(&self) -> Scalar {
        unsafe { ffi::Mat_Mean(self.inner) }
    }

    /// Calculates the per-channel pixel sum of an image.
    pub fn sum(&self) -> Scalar {
        unsafe { ffi::Mat_Sum(self.inner) }
    }

    /// Performs a look-up table transform of an array.
    ///
    /// The function LUT fills the output array with values from the look-up table.
    /// Indices of the entries are taken from the input array.
    pub fn lookup_table_transform(&self, table: &Mat, dst: &mut Mat) {
        unsafe { ffi::LUT(self.inner, table.inner, dst.inner) }
    }

    /// Returns the number of rows for this Mat.
    pub fn rows(&self) -> i32 {
        unsafe { ffi::Mat_Rows(self.inner) }
    }

    /// Returns the number of cols for this Mat.
    pub fn cols(&self) -> i32 {
        unsafe { ffi::Mat_Cols(self.inner) }
    }

    /// Returns the number of channels for this Mat.
    pub fn channels(&self) -> i32 {
        unsafe { ffi::Mat_Channels(self.inner) }
    }

    /// Returns the type for this Mat.
    pub fn cv_type(&self) -> CvType {
        let t = unsafe { ffi::Mat_Type(self.inner) };
        CvType::from_i32(t).expect("Unknown CvType")
    }

    /// Returns the number of bytes each matrix row occupies.
    pub fn cv_step(&self) -> i32 {
        unsafe { ffi::Mat_Step(self.inner) }
    }

    /// Returns a value from a specific row/col in this Mat (must be CV_8U).
    pub fn uchar_at(&self, row: i32, col: i32) -> u8 {
        unsafe { ffi::Mat_GetUChar(self.inner, row, col) }
    }

    /// Returns a value from a specific x, y, z in this Mat (must be CV_8U).
    pub fn uchar_at3(&self, x: i32, y: i32, z: i32) -> u8 {
        unsafe { ffi::Mat_GetUChar3(self.inner, x, y, z) }
    }

    /// Returns a value from a specific row/col in this Mat (must be CV_8S).
    pub fn schar_at(&self, row: i32, col: i32) -> i8 {
        unsafe { ffi::Mat_GetSChar(self.inner, row, col) }
    }

    /// Returns a value from a specific x, y, z in this Mat (must be CV_8S).
    pub fn schar_at3(&self, x: i32, y: i32, z: i32) -> i8 {
        unsafe { ffi::Mat_GetSChar3(self.inner, x, y, z) }
    }

    /// Returns a value from a specific row/col in this Mat (must be CV_16S).
    pub fn short_at(&self, row: i32, col: i32) -> i16 {
        unsafe { ffi::Mat_GetShort(self.inner, row, col) }
    }

    /// Returns a value from a specific x, y, z in this Mat (must be CV_16S).
    pub fn short_at3(&self, x: i32, y: i32, z: i32) -> i16 {
        unsafe { ffi::Mat_GetShort3(self.inner, x, y, z) }
    }

    /// Returns a value from a specific row/col in this Mat (must be CV_32S).
    pub fn int_at(&self, row: i32, col: i32) -> i32 {
        unsafe { ffi::Mat_GetInt(self.inner, row, col) }
    }

    /// Returns a value from a specific x, y, z in this Mat (must be CV_32S).
    pub fn int_at3(&self, x: i32, y: i32, z: i32) -> i32 {
        unsafe { ffi::Mat_GetInt3(self.inner, x, y, z) }
    }

    /// Returns a value from a specific row/col in this Mat (must be CV_32F).
    pub fn float_at(&self, row: i32, col: i32) -> f32 {
        unsafe { ffi::Mat_GetFloat(self.inner, row, col) }
    }

    /// Returns a value from a specific x, y, z in this Mat (must be CV_32F).
    pub fn float_at3(&self, x: i32, y: i32, z: i32) -> f32 {
        unsafe { ffi::Mat_GetFloat3(self.inner, x, y, z) }
    }

    /// Returns a value from a specific row/col in this Mat (must be CV_64F).
    pub fn double_at(&self, row: i32, col: i32) -> f64 {
        unsafe { ffi::Mat_GetDouble(self.inner, row, col) }
    }

    /// Returns a value from a specific x, y, z in this Mat (must be CV_64F).
    pub fn double_at3(&self, x: i32, y: i32, z: i32) -> f64 {
        unsafe { ffi::Mat_GetDouble3(self.inner, x, y, z) }
    }

    /// Returns a value from a specific row/col in this Mat (must be CV_8U).
    pub fn set_uchar_at(&mut self, row: i32, col: i32) {
        unsafe { ffi::Mat_SetUChar(self.inner, row, col) }
    }

    /// Returns a value from a specific x, y, z in this Mat (must be CV_8U).
    pub fn set_uchar_at3(&mut self, x: i32, y: i32, z: i32) {
        unsafe { ffi::Mat_SetUChar3(self.inner, x, y, z) }
    }

    /// Returns a value from a specific row/col in this Mat (must be CV_8S).
    pub fn set_schar_at(&mut self, row: i32, col: i32) {
        unsafe { ffi::Mat_SetSChar(self.inner, row, col) }
    }

    /// Returns a value from a specific x, y, z in this Mat (must be CV_8S).
    pub fn set_schar_at3(&mut self, x: i32, y: i32, z: i32) {
        unsafe { ffi::Mat_SetSChar3(self.inner, x, y, z) }
    }

    /// Returns a value from a specific row/col in this Mat (must be CV_16S).
    pub fn set_short_at(&mut self, row: i32, col: i32) {
        unsafe { ffi::Mat_SetShort(self.inner, row, col) }
    }

    /// Returns a value from a specific x, y, z in this Mat (must be CV_16S).
    pub fn set_short_at3(&mut self, x: i32, y: i32, z: i32) {
        unsafe { ffi::Mat_SetShort3(self.inner, x, y, z) }
    }

    /// Returns a value from a specific row/col in this Mat (must be CV_32S).
    pub fn set_int_at(&mut self, row: i32, col: i32) {
        unsafe { ffi::Mat_SetInt(self.inner, row, col) }
    }

    /// Returns a value from a specific x, y, z in this Mat (must be CV_32S).
    pub fn set_int_at3(&mut self, x: i32, y: i32, z: i32) {
        unsafe { ffi::Mat_SetInt3(self.inner, x, y, z) }
    }

    /// Returns a value from a specific row/col in this Mat (must be CV_32F).
    pub fn set_float_at(&mut self, row: i32, col: i32) {
        unsafe { ffi::Mat_SetFloat(self.inner, row, col) }
    }

    /// Returns a value from a specific x, y, z in this Mat (must be CV_32F).
    pub fn set_float_at3(&mut self, x: i32, y: i32, z: i32) {
        unsafe { ffi::Mat_SetFloat3(self.inner, x, y, z) }
    }

    /// Returns a value from a specific row/col in this Mat (must be CV_64F).
    pub fn set_double_at(&mut self, row: i32, col: i32) {
        unsafe { ffi::Mat_SetDouble(self.inner, row, col) }
    }

    /// Returns a value from a specific x, y, z in this Mat (must be CV_64F).
    pub fn set_double_at3(&mut self, x: i32, y: i32, z: i32) {
        unsafe { ffi::Mat_SetDouble3(self.inner, x, y, z) }
    }

    /// Calculates the per-element absolute difference between two arrays or
    /// between an array and a scalar.
    fn abs_diff(&self, other: &Mat, dst: &mut Mat) {
        unsafe { Mat_AbsDiff(self.inner, other.inner, dst.inner) }
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
