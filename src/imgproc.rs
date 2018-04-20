//! [Image Processing](https://docs.opencv.org/master/d7/dbd/group__imgproc.html)

use opencv_sys as ffi;
use core::{BorderType, Mat, Point, Rect, Scalar, Size};

fn to_points(curve: &mut [Point]) -> ffi::Points {
    ffi::Points {
        points: curve.as_mut_ptr(),
        length: curve.len() as i32,
    }
}

/// Calculates a contour perimeter or a curve length.
pub fn arc_length(curve: &mut Vec<Point>, is_closed: bool) -> f64 {
    unsafe { ffi::ArcLength(to_points(curve), is_closed) }
}

/// Color conversion code used in `cvt_color`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
#[allow(non_camel_case_types, missing_docs)]
pub enum ColorConversion {
    BGR2BGRA = 0,
    BGRA2BGR = 1,
    BGR2RGBA = 2,
    RGBA2BGR = 3,
    BGR2RGB = 4,
    BGRA2RGBA = 5,
    BGR2GRAY = 6,
    RGB2GRAY = 7,
    GRAY2BGR = 8,
    GRAY2BGRA = 9,
    BGRA2GRAY = 10,
    RGBA2GRAY = 11,
    BGR2BGR565 = 12,
    RGB2BGR565 = 13,
    BGR5652BGR = 14,
    BGR5652RGB = 15,
    BGRA2BGR565 = 16,
    RGBA2BGR565 = 17,
    BGR5652BGRA = 18,
    BGR5652RGBA = 19,
    GRAY2BGR565 = 20,
    BGR5652GRAY = 21,
    BGR2BGR555 = 22,
    RGB2BGR555 = 23,
    BGR5552BGR = 24,
    BGR5552RGB = 25,
    BGRA2BGR555 = 26,
    RGBA2BGR555 = 27,
    BGR5552BGRA = 28,
    BGR5552RGBA = 29,
    GRAY2BGR555 = 30,
    BGR5552GRAY = 31,
    BGR2XYZ = 32,
    RGB2XYZ = 33,
    XYZ2BGR = 34,
    XYZ2RGB = 35,
    BGR2YCrCb = 36,
    RGB2YCrCb = 37,
    YCrCb2BGR = 38,
    YCrCb2RGB = 39,
    BGR2HSV = 40,
    RGB2HSV = 41,
    BGR2Lab = 44,
    RGB2Lab = 45,
    BGR2Luv = 50,
    RGB2Luv = 51,
    BGR2HLS = 52,
    RGB2HLS = 53,
    HSV2BGR = 54,
    HSV2RGB = 55,
    Lab2BGR = 56,
    Lab2RGB = 57,
    Luv2BGR = 58,
    Luv2RGB = 59,
    HLS2BGR = 60,
    HLS2RGB = 61,
    BGR2HSV_FULL = 66,
    RGB2HSV_FULL = 67,
    BGR2HLS_FULL = 68,
    RGB2HLS_FULL = 69,
    HSV2BGR_FULL = 70,
    HSV2RGB_FULL = 71,
    HLS2BGR_FULL = 72,
    HLS2RGB_FULL = 73,
    LBGR2Lab = 74,
    LRGB2Lab = 75,
    LBGR2Luv = 76,
    LRGB2Luv = 77,
    Lab2LBGR = 78,
    Lab2LRGB = 79,
    Luv2LBGR = 80,
    Luv2LRGB = 81,
    BGR2YUV = 82,
    RGB2YUV = 83,
    YUV2BGR = 84,
    YUV2RGB = 85,
    YUV2RGB_NV12 = 90,
    YUV2BGR_NV12 = 91,
    YUV2RGB_NV21 = 92,
    YUV2BGR_NV21 = 93,
    YUV2RGBA_NV12 = 94,
    YUV2BGRA_NV12 = 95,
    YUV2RGBA_NV21 = 96,
    YUV2BGRA_NV21 = 97,
    YUV2RGB_YV12 = 98,
    YUV2BGR_YV12 = 99,
    YUV2RGB_IYUV = 100,
    YUV2BGR_IYUV = 101,
    YUV2RGBA_YV12 = 102,
    YUV2BGRA_YV12 = 103,
    YUV2RGBA_IYUV = 104,
    YUV2BGRA_IYUV = 105,
    YUV2GRAY_420 = 106,
    YUV2RGB_UYVY = 107,
    YUV2BGR_UYVY = 108,
    YUV2RGBA_UYVY = 111,
    YUV2BGRA_UYVY = 112,
    YUV2RGB_YUY2 = 115,
    YUV2BGR_YUY2 = 116,
    YUV2RGB_YVYU = 117,
    YUV2BGR_YVYU = 118,
    YUV2RGBA_YUY2 = 119,
    YUV2BGRA_YUY2 = 120,
    YUV2RGBA_YVYU = 121,
    YUV2BGRA_YVYU = 122,
    YUV2GRAY_UYVY = 123,
    YUV2GRAY_YUY2 = 124,
    RGBA2mRGBA = 125,
    mRGBA2RGBA = 126,
    RGB2YUV_I420 = 127,
    BGR2YUV_I420 = 128,
    RGBA2YUV_I420 = 129,
    BGRA2YUV_I420 = 130,
    RGB2YUV_YV12 = 131,
    BGR2YUV_YV12 = 132,
    RGBA2YUV_YV12 = 133,
    BGRA2YUV_YV12 = 134,
    BayerBG2BGR = 46,
    BayerGB2BGR = 47,
    BayerRG2BGR = 48,
    BayerGR2BGR = 49,
    BayerBG2GRAY = 86,
    BayerGB2GRAY = 87,
    BayerRG2GRAY = 88,
    BayerGR2GRAY = 89,
    BayerBG2BGR_VNG = 62,
    BayerGB2BGR_VNG = 63,
    BayerRG2BGR_VNG = 64,
    BayerGR2BGR_VNG = 65,
    BayerBG2BGR_EA = 135,
    BayerGB2BGR_EA = 136,
    BayerRG2BGR_EA = 137,
    BayerGR2BGR_EA = 138,
    COLORCVT_MAX = 139,
}

/// Convert an image from one color space to another.
pub fn cvt_color(src: &Mat, dst: &mut Mat, code: ColorConversion) {
    unsafe { ffi::CvtColor(src.inner, dst.inner, code as i32) }
}

/// TemplateMatchMode is the type of the template matching operation.
#[repr(C)]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
#[allow(non_camel_case_types, missing_docs)]
pub enum TemplateMatchMode {
    SqDiff = 0,
    SqDiffNormed = 1,
    Ccorr = 2,
    CcorrNormed = 3,
    Ccoeff = 4,
    CcoeffNormed = 5,
}

/// Compares a template against overlapped image regions.
pub fn match_template(
    image: &Mat,
    templ: &Mat,
    dst: &mut Mat,
    mode: TemplateMatchMode,
    mask: &Mat,
) {
    unsafe { ffi::MatchTemplate(image.inner, templ.inner, dst.inner, mode as i32, mask.inner) }
}

/// Blurs an image and downsamples it. This function performs the
/// downsampling step of the Gaussian pyramid construction.
pub fn pyr_down(src: &Mat, dst: &mut Mat, size: Size, border: BorderType) {
    unsafe { ffi::PyrDown(src.inner, dst.inner, size, border as i32) }
}

/// Upsamples an image and then blurs it. This function performs the upsampling
/// step of the Gaussian pyramid construction.
pub fn pyr_up(src: &Mat, dst: &mut Mat, size: Size, border: BorderType) {
    unsafe { ffi::PyrUp(src.inner, dst.inner, size, border as i32) }
}

/// GaussianBlur blurs an image Mat using a Gaussian filter.
///
/// The function convolves the `src` Mat image into the `dst` Mat using the
/// specified Gaussian kernel params.
///
/// * `src`: input image; the image can have any number of channels, which are
/// processed independently, but the depth should be `CV_8U`, `CV_16U`, `CV_16S`,
/// `CV_32F` or `CV_64F`.
/// * `dst`: output image of the same size and type as `src`.
/// * `ksize`: Gaussian kernel size. ksize.width and ksize.height can differ but
/// they both must be positive and odd. Or, they can be zero's and then they are
/// computed from sigma.
/// * `sigmaX`: Gaussian kernel standard deviation in X direction.
/// * `sigmaY`: Gaussian kernel standard deviation in Y direction. if sigmaY is
/// zero, it is set to be equal to sigmaX, if both sigmas are zeros, they are
/// computed from ksize.width and ksize.height, respectively (see
/// cv::getGaussianKernel for details). To fully control the result regardless
/// of possible future modifications of all this semantics, it is recommended to
/// specify all of ksize, sigmaX, and sigmaY.
/// * `borderType`: pixel extrapolation method, see
/// [BorderType](../core/enum.BorderType.html).
pub fn gaussian_blur(
    src: &Mat,
    dst: &mut Mat,
    ksize: Size,
    sigma_x: f64,
    sigma_y: f64,
    border: BorderType,
) {
    unsafe { ffi::GaussianBlur(src.inner, dst.inner, ksize, sigma_x, sigma_y, border as i32) }
}

/// Calculates the Laplacian of an image.
pub fn laplacian(
    src: &Mat,
    dst: &mut Mat,
    depth: i32,
    size: i32,
    scale: f64,
    delta: f64,
    border: BorderType,
) {
    unsafe {
        ffi::Laplacian(
            src.inner,
            dst.inner,
            depth,
            size,
            scale,
            delta,
            border as i32,
        )
    }
}

/// Calculates the first x- or y- image derivative using Scharr operator.
pub fn scharr(
    src: &Mat,
    dst: &mut Mat,
    depth: i32,
    dx: i32,
    dy: i32,
    scale: f64,
    delta: f64,
    border: BorderType,
) {
    unsafe {
        ffi::Scharr(
            src.inner,
            dst.inner,
            depth,
            dx,
            dy,
            scale,
            delta,
            border as i32,
        )
    }
}

/// Blurs an image using the median filter.
pub fn median_blur(src: &Mat, dst: &mut Mat, size: i32) {
    unsafe { ffi::MedianBlur(src.inner, dst.inner, size) }
}

/// Finds edges in an image using the Canny algorithm.
///
/// The function finds edges in the input image image and marks them in the
/// output map edges using the Canny algorithm. The smallest value between
/// threshold1 and threshold2 is used for edge linking. The largest value is
/// used to find initial segments of strong edges. See
/// [wikipedia](http://en.wikipedia.org/wiki/Canny_edge_detector).
///
/// * `image`: 8-bit input image.
/// * `edges`: output edge map; single channels 8-bit image, which has the same
/// size as image .
/// * `threshold1`: first threshold for the hysteresis procedure.
/// * `threshold2`: second threshold for the hysteresis procedure.
pub fn canny(src: &Mat, edges: &mut Mat, threshold1: f64, threshold2: f64) {
    unsafe { ffi::Canny(src.inner, edges.inner, threshold1, threshold2) }
}

/// Determines strong corners on an image. The function finds the most prominent
/// corners in the image or in the specified image region.
pub fn good_features_to_track(
    img: &Mat,
    corners: &mut Mat,
    max_corners: i32,
    quality: f64,
    min_dist: f64,
) {
    unsafe { ffi::GoodFeaturesToTrack(img.inner, corners.inner, max_corners, quality, min_dist) }
}

/// Type of threshold operation.
#[repr(C)]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum ThresholdType {
    /// ThresholdBinary
    Binary = 0,
    /// ThresholdBinaryInv
    BinaryInv = 1,
    /// ThresholdTrunc
    Trunc = 2,
    /// ThresholdToZero
    ToZero = 3,
    /// ThresholdToZeroInv
    ToZeroInv = 4,
    /// ThresholdMask
    Mask = 7,
    /// ThresholdOtsu
    Otsu = 8,
    /// ThresholdTriangle
    Triangle = 16,
}

/// Applies a fixed-level threshold to each array element.
pub fn threshold(src: &Mat, dst: &mut Mat, thresh: f64, max: f64, type_: ThresholdType) {
    unsafe { ffi::Threshold(src.inner, dst.inner, thresh, max, type_ as i32) }
}

/// Draws a circle.
pub fn circle(img: &mut Mat, center: Point, radius: i32, color: Scalar, thickness: i32) {
    unsafe { ffi::Circle(img.inner, center, radius, color, thickness) }
}

/// Draws a simple or thick elliptic arc or fills an ellipse sector.
pub fn ellipse(
    img: &mut Mat,
    center: Point,
    axes: Point,
    angle: f64,
    start_angle: f64,
    end_angle: f64,
    color: Scalar,
    thickness: i32,
) {
    unsafe {
        ffi::Ellipse(
            img.inner,
            center,
            axes,
            angle,
            start_angle,
            end_angle,
            color,
            thickness,
        )
    }
}

/// Draws a line segment connecting two points.
pub fn line(img: &mut Mat, pt1: Point, pt2: Point, color: Scalar, thickness: i32) {
    unsafe { ffi::Line(img.inner, pt1, pt2, color, thickness) }
}

/// Rectangle draws a simple, thick, or filled up-right rectangle.  It renders a
/// rectangle with the desired characteristics to the target Mat image.
pub fn rectangle(img: &mut Mat, r: Rect, c: Scalar, thickness: i32) {
    unsafe { ffi::Rectangle(img.inner, r, c, thickness) }
}

/// Interpolation algorithm
#[repr(C)]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum InterpolationFlag {
    /// nearest neighbor interpolation
    Nearst = 0,
    /// bilinear interpolation
    Linear = 1,
    /// bicubic interpolation
    Cubic = 2,
    /// resampling using pixel area relation. It may be a preferred method for
    /// image decimation, as it gives moire'-free results. But when the image is
    /// zoomed, it is similar to the INTER_NEAREST method.
    Area = 3,
    /// Lanczos interpolation over 8x8 neighborhood
    Lanczos4 = 4,
    /// Bit exact bilinear interpolation
    LinearExact = 5,
    /// mask for interpolation codes
    Max = 7,
    /// flag, fills all of the destination image pixels. If some of them
    /// correspond to outliers in the source image, they are set to zero
    WarpFillOutliers = 8,
    /// flag, inverse transformation
    WarpInverseMap = 16,
}

/// Resize resizes an image.  It resizes the image src down to or up to the
/// specified size, storing the result in dst. Note that src and dst may be the
/// same image. If you wish to scale by factor, an empty sz may be passed and
/// non-zero fx and fy. Likewise, if you wish to scale to an explicit size, a
/// non-empty sz may be passed with zero for both fx and fy.
pub fn resize(src: &Mat, dst: &mut Mat, sz: Size, fx: f64, fy: f64, interp: InterpolationFlag) {
    unsafe { ffi::Resize(src.inner, dst.inner, sz, fx, fy, interp as i32) }
}
