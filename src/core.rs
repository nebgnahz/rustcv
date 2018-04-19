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

impl Clone for Mat {
    fn clone(&self) -> Self {
        Mat {
            inner: unsafe { ffi::Mat_Clone(self.inner) },
        }
    }
}

impl Drop for Mat {
    fn drop(&mut self) {
        unsafe { ffi::Mat_Close(self.inner) }
    }
}

pub use opencv_sys::Scalar;
pub use opencv_sys::Rect;
pub use opencv_sys::Size;
pub use opencv_sys::Point;
pub use opencv_sys::KeyPoint;

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

/// Various border types, image boundaries are denoted with `|`.
#[derive(Debug, Copy, Clone)]
pub enum BorderType {
    /// `iiiiii|abcdefgh|iiiiiii`  with some specified `i`
    Constant = 0,
    /// `aaaaaa|abcdefgh|hhhhhhh`
    Replicate = 1,
    /// `fedcba|abcdefgh|hgfedcb`
    Reflect = 2,
    /// `cdefgh|abcdefgh|abcdefg`
    Wrap = 3,
    /// `gfedcb|abcdefgh|gfedcba`
    Reflect101 = 4,
    /// `uvwxyz|abcdefgh|ijklmno`
    Transparent = 5,
    /// Do not look outside of ROI.
    Isolated = 16,
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

    /// Sets a value to a specific row/col in this Mat (must be CV_8U).
    pub fn set_uchar_at(&mut self, row: i32, col: i32, val: u8) {
        unsafe { ffi::Mat_SetUChar(self.inner, row, col, val) }
    }

    /// Sets a value to a specific x, y, z in this Mat (must be CV_8U).
    pub fn set_uchar_at3(&mut self, x: i32, y: i32, z: i32, val: u8) {
        unsafe { ffi::Mat_SetUChar3(self.inner, x, y, z, val) }
    }

    /// Sets a value to a specific row/col in this Mat (must be CV_8S).
    pub fn set_schar_at(&mut self, row: i32, col: i32, val: i8) {
        unsafe { ffi::Mat_SetSChar(self.inner, row, col, val) }
    }

    /// Sets a value to a specific x, y, z in this Mat (must be CV_8S).
    pub fn set_schar_at3(&mut self, x: i32, y: i32, z: i32, val: i8) {
        unsafe { ffi::Mat_SetSChar3(self.inner, x, y, z, val) }
    }

    /// Sets a value to a specific row/col in this Mat (must be CV_16S).
    pub fn set_short_at(&mut self, row: i32, col: i32, val: i16) {
        unsafe { ffi::Mat_SetShort(self.inner, row, col, val) }
    }

    /// Sets a value to a specific x, y, z in this Mat (must be CV_16S).
    pub fn set_short_at3(&mut self, x: i32, y: i32, z: i32, val: i16) {
        unsafe { ffi::Mat_SetShort3(self.inner, x, y, z, val) }
    }

    /// Sets a value to a specific row/col in this Mat (must be CV_32S).
    pub fn set_int_at(&mut self, row: i32, col: i32, val: i32) {
        unsafe { ffi::Mat_SetInt(self.inner, row, col, val) }
    }

    /// Sets a value to a specific x, y, z in this Mat (must be CV_32S).
    pub fn set_int_at3(&mut self, x: i32, y: i32, z: i32, val: i32) {
        unsafe { ffi::Mat_SetInt3(self.inner, x, y, z, val) }
    }

    /// Sets a value to a specific row/col in this Mat (must be CV_32F).
    pub fn set_float_at(&mut self, row: i32, col: i32, val: f32) {
        unsafe { ffi::Mat_SetFloat(self.inner, row, col, val) }
    }

    /// Sets a value to a specific x, y, z in this Mat (must be CV_32F).
    pub fn set_float_at3(&mut self, x: i32, y: i32, z: i32, val: f32) {
        unsafe { ffi::Mat_SetFloat3(self.inner, x, y, z, val) }
    }

    /// Sets a value to a specific row/col in this Mat (must be CV_64F).
    pub fn set_double_at(&mut self, row: i32, col: i32, val: f64) {
        unsafe { ffi::Mat_SetDouble(self.inner, row, col, val) }
    }

    /// Sets a value to a specific x, y, z in this Mat (must be CV_64F).
    pub fn set_double_at3(&mut self, x: i32, y: i32, z: i32, val: f64) {
        unsafe { ffi::Mat_SetDouble3(self.inner, x, y, z, val) }
    }
}

/// Calculates the per-element absolute difference between two arrays or
/// between an array and a scalar.
pub fn abs_diff(this: &Mat, other: &Mat, dst: &mut Mat) {
    unsafe { ffi::Mat_AbsDiff(this.inner, other.inner, dst.inner) }
}

/// Calculates the per-element sum of two arrays or an array and a scalar.
pub fn add(this: &Mat, other: &Mat, dst: &mut Mat) {
    unsafe { ffi::Mat_AbsDiff(this.inner, other.inner, dst.inner) }
}

/// Calculates the weighted sum of two arrays (dst = src1\*alpha + src2\*beta +
/// gamma).
pub fn add_weighted(src1: &Mat, alpha: f64, src2: &Mat, beta: f64, gamma: f64, dst: &mut Mat) {
    unsafe { ffi::Mat_AddWeighted(src1.inner, alpha, src2.inner, beta, gamma, dst.inner) }
}

/// Computes bitwise conjunction of the two arrays (dst = src1 & src2).
pub fn bitwise_and(src1: &Mat, src2: &Mat, dst: &mut Mat) {
    unsafe { ffi::Mat_BitwiseAnd(src1.inner, src2.inner, dst.inner) }
}

/// Inverts every bit of an array (dst = !src).
pub fn bitwise_not(src: &Mat, dst: &mut Mat) {
    unsafe { ffi::Mat_BitwiseNot(src.inner, dst.inner) }
}

/// Computes bitwise disjunction of the two arrays (dst = src1 | src2).
pub fn bitwise_or(src1: &Mat, src2: &Mat, dst: &mut Mat) {
    unsafe { ffi::Mat_BitwiseOr(src1.inner, src2.inner, dst.inner) }
}

/// Computes bitwise "exclusive or" of the two arrays (dst = src1 ^ src2).
pub fn bitwise_xor(src1: &Mat, src2: &Mat, dst: &mut Mat) {
    unsafe { ffi::Mat_BitwiseXor(src1.inner, src2.inner, dst.inner) }
}

/// A naive nearest neighbor finder.
pub fn batch_distance(
    src1: &Mat,
    src2: &Mat,
    dist: &Mat,
    dtype: i32,
    nidx: &Mat,
    norm_type: i32,
    k: i32,
    mask: &Mat,
    update: i32,
    crosscheck: bool,
) {
    unsafe {
        ffi::Mat_BatchDistance(
            src1.inner,
            src2.inner,
            dist.inner,
            dtype,
            nidx.inner,
            norm_type,
            k,
            mask.inner,
            update,
            crosscheck,
        )
    }
}

/// Computes the source location of an extrapolated pixel.
pub fn border_interpolate(p: i32, len: i32, t: BorderType) -> i32 {
    unsafe { ffi::Mat_BorderInterpolate(p, len, t as i32) }
}

/// [Covariation flags](https://docs.opencv.org/master/d0/de1/group__core.html#ga719ebd4a73f30f4fab258ab7616d0f0f).
#[derive(Copy, Clone, Debug)]
pub enum CovarFlag {
    /// The covariance matrix will be nsamples x nsamples. Such an unusual
    /// covariance matrix is used for fast PCA of a set of very large vectors
    /// (see, for example, the EigenFaces technique for face
    /// recognition). Eigenvalues of this "scrambled" matrix match the
    /// eigenvalues of the true covariance matrix. The "true" eigenvectors can
    /// be easily calculated from the eigenvectors of the "scrambled" covariance
    /// matrix.
    Scrambled = 0,

    /// covar will be a square matrix of the same size as the total number of
    /// elements in each input vector. One and only one of COVAR_SCRAMBLED and
    /// COVAR_NORMAL must be specified.
    Normal = 1,

    /// If the flag is specified, the function does not calculate mean from the
    /// input vectors but, instead, uses the passed mean vector. This is useful
    /// if mean has been pre-calculated or known in advance, or if the
    /// covariance matrix is calculated by parts. In this case, mean is not a
    /// mean vector of the input sub-set of vectors but rather the mean vector
    /// of the whole set.
    UseAvg = 2,

    /// If the flag is specified, the covariance matrix is scaled. In the
    /// "normal" mode, scale is 1./nsamples . In the "scrambled" mode, scale is
    /// the reciprocal of the total number of elements in each input vector. By
    /// default (if the flag is not specified), the covariance matrix is not
    /// scaled ( scale=1 ).
    Scale = 4,

    /// If the flag is specified, all the input vectors are stored as rows of
    /// the samples matrix. mean should be a single-row vector in this case.
    Rows = 8,

    /// If the flag is specified, all the input vectors are stored as columns of
    /// the samples matrix. mean should be a single-column vector in this case.
    Cols = 16,
}

/// Calculates the covariance matrix of a set of vectors.
pub fn calc_covar_matrix(
    samples: &Mat,
    covar: &mut Mat,
    mean: &mut Mat,
    flags: CovarFlag,
    ctype: i32,
) {
    unsafe { ffi::Mat_CalcCovarMatrix(samples.inner, covar.inner, mean.inner, flags as i32, ctype) }
}

/// Calculates the magnitude and angle of 2D vectors.
pub fn cart_to_polar(x: &Mat, y: &Mat, magnitude: &mut Mat, angle: &mut Mat, use_degree: bool) {
    unsafe { ffi::Mat_CartToPolar(x.inner, y.inner, magnitude.inner, angle.inner, use_degree) }
}

/// Comparison type.
#[repr(C)]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, FromPrimitive, ToPrimitive)]
pub enum CompareType {
    /// src1 is equal to src2.
    Eq = 0,
    /// src1 is greater to src2.
    Gt = 1,
    /// src1 is greater than or equal to src2.
    Ge = 2,
    /// src1 is less than to src2.
    Lt = 3,
    /// src1 is less than or equal to src2.
    Le = 4,
    /// src1 is not equal to src2.
    Ne = 5,
}

/// Performs the per-element comparison of two arrays or an array and scalar
/// value.
pub fn compare(src1: &Mat, src2: &Mat, dst: &mut Mat, ct: CompareType) {
    unsafe { ffi::Mat_Compare(src1.inner, src2.inner, dst.inner, ct as i32) }
}

/// Counts non-zero array elements.
pub fn count_non_zero(src: &Mat) -> i32 {
    unsafe { ffi::Mat_CountNonZero(src.inner) }
}

/// Copies the lower or the upper half of a square matrix to its another half.
pub fn complete_symm(m: &mut Mat, lower_to_upper: bool) {
    unsafe { ffi::Mat_CompleteSymm(m.inner, lower_to_upper) }
}

/// Scales, calculates absolute values, and converts the result to 8-bit.
pub fn convert_scale_abs(src: &Mat, dst: &mut Mat, alpha: f64, beta: f64) {
    unsafe { ffi::Mat_ConvertScaleAbs(src.inner, dst.inner, alpha, beta) }
}

/// Forms a border around an image.
///
/// The function copies the source image into the middle of the destination
/// image. The areas to the left, to the right, above and below the copied
/// source image will be filled with extrapolated pixels. This is not what
/// filtering functions based on it do (they extrapolate pixels on-fly), but
/// what other more complex functions, including your own, may do to simplify
/// image boundary handling.
pub fn copy_make_border(
    src: &Mat,
    dst: &mut Mat,
    top: i32,
    bottom: i32,
    left: i32,
    right: i32,
    type_: BorderType,
    value: Scalar,
) {
    unsafe {
        ffi::Mat_CopyMakeBorder(
            src.inner,
            dst.inner,
            top,
            bottom,
            left,
            right,
            type_ as i32,
            value,
        )
    }
}

/// Finds the global minimum and maximum in an array.
pub fn min_max_loc(input: &Mat) -> (f64, f64, Point, Point) {
    let mut min = 0.0;
    let mut max = 0.0;
    let mut min_loc = Point { x: 0, y: 0 };
    let mut max_loc = Point { x: 0, y: 0 };
    unsafe { ffi::Mat_MinMaxLoc(input.inner, &mut min, &mut max, &mut min_loc, &mut max_loc) }
    (min, max, min_loc, max_loc)
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
