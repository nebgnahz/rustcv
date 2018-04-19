//! [Image file reading and writing][opencv-imgcodecs].
//!
//! [opencv-imgcodecs]: http://docs.opencv.org/master/d4/da8/group__imgcodecs.html

use core::Mat;
use opencv_sys as ffi;
use std::path::Path;
use failure::Error;

/// [ImageReadMode][opencv-imread].
///
/// [opencv-imread]: https://docs.opencv.org/master/d4/da8/group__imgcodecs.html#ga61d9b0126a3e57d9277ac48327799c80
#[repr(C)]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum ImageReadMode {
    /// If set, return the loaded image as is (with alpha channel, otherwise it
    /// gets cropped
    Unchanged = -1,
    /// If set, always convert image to the single channel grayscale image.
    Grayscale = 0,
    /// If set, always convert image to the 3 channel BGR color image.
    Color = 1,
    /// If set, return 16-bit/32-bit image when the input has the corresponding
    /// depth, otherwise convert it to 8-bit.
    AnyDepth = 2,
    /// If set, the image is read in any possible color format.
    AnyColor = 4,
    /// If set, use the gdal driver for loading the image.
    LoadGdal = 8,
    /// If set, always convert image to the single channel grayscale image and
    /// the image size reduced 1/2.
    ReducedGrayscale2 = 16,
    /// If set, always convert image to the 3 channel BGR color image and the
    /// image size reduced 1/2.
    ReducedColor2 = 17,
    /// If set, always convert image to the single channel grayscale image and
    /// the image size reduced 1/4.
    ReducedGrayscale4 = 32,
    /// If set, always convert image to the 3 channel BGR color image and the
    /// image size reduced 1/4.
    ReducedColor4 = 33,
    /// If set, always convert image to the single channel grayscale image and
    /// the image size reduced 1/8.
    ReducedGrayscale8 = 64,
    /// If set, always convert image to the 3 channel BGR color image and the
    /// image size reduced 1/8.
    ReducedColor8 = 65,
}

/// Loads an image from a file.
///
/// The function imread loads an image from the specified file and returns
/// it. If the image cannot be read (because of missing file, improper
/// permissions, unsupported or invalid format), the function returns an empty
/// matrix ( Mat::data==NULL ).
///
/// Currently, the following file formats are supported:
///
/// - Windows bitmaps - *.bmp, *.dib (always supported)
/// - JPEG files - *.jpeg, *.jpg, *.jpe (see the Notes section)
/// - JPEG 2000 files - *.jp2 (see the Notes section)
/// - Portable Network Graphics - *.png (see the Notes section)
/// - WebP - *.webp (see the Notes section)
/// - Portable image format - *.pbm, *.pgm, *.ppm *.pxm, *.pnm (always supported)
/// - Sun rasters - *.sr, *.ras (always supported)
/// - TIFF files - *.tiff, *.tif (see the Notes section)
/// - OpenEXR Image files - *.exr (see the Notes section)
/// - Radiance HDR - *.hdr, *.pic (always supported)
/// - Raster and Vector geospatial data supported by Gdal (see the Notes section)
pub fn imread<P: AsRef<Path>>(path: P, flags: ImageReadMode) -> Result<Mat, Error> {
    let path = ::path_to_cstring(path)?;
    let path = path.as_ptr();
    let mat = unsafe { ffi::Image_IMRead(path, flags as i32) };
    Ok(mat.into())
}

/// Image write mode. [See documentation](https://docs.opencv.org/master/d4/da8/group__imgcodecs.html#ga292d81be8d76901bff7988d18d2b42ac) for details.
#[repr(C)]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum ImageWriteMode {
    /// For JPEG, it can be a quality from 0 to 100 (the higher is the
    /// better). Default value is 95.
    JpegQuality = 1,
    /// Enable JPEG features, 0 or 1, default is False.
    JpegProgressive = 2,
    /// Enable JPEG features, 0 or 1, default is False.
    JpegOptimize = 3,
    /// JPEG restart interval, 0 - 65535, default is 0 - no restart.
    JpegRstInterval = 4,
    /// Separate luma quality level, 0 - 100, default is 0 - don't use.
    JpegLumaQuality = 5,
    /// Separate chroma quality level, 0 - 100, default is 0 - don't use.
    JpegChromaQuality = 6,
    /// For PNG, it can be the compression level from 0 to 9. A higher value
    /// means a smaller size and longer compression time. Default value is 3.
    /// Also strategy is changed to IMWRITE_PNG_STRATEGY_DEFAULT
    /// (Z_DEFAULT_STRATEGY).
    PngCompression = 16,
    /// One of cv::ImwritePNGFlags, default is IMWRITE_PNG_STRATEGY_DEFAULT.
    PngStrategy = 17,
    /// Binary level PNG, 0 or 1, default is 0.
    PngBilevel = 18,
    /// For PPM, PGM, or PBM, it can be a binary format flag, 0 or 1. Default
    /// value is 1.
    PxmBinary = 32,
    /// For WEBP, it can be a quality from 1 to 100 (the higher is the
    /// better). By default (without any parameter) and for quality above 100
    /// the lossless compression is used.
    WebpQuality = 64,
    /// For PAM, sets the TUPLETYPE field to the corresponding string value that
    /// is defined for the format
    PamTupletype = 128,
}

/// Image write PNG flag. [See
/// documentation](https://docs.opencv.org/master/d4/da8/group__imgcodecs.html#gaa60044d347ffd187161b5ec9ea2ef2f9)
/// for details.
#[repr(C)]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum ImageWritePngFlag {
    /// Use this value for normal data.
    Default = 0,
    ///  Use this value for data produced by a filter (or predictor).Filtered
    ///  data consists mostly of small values with a somewhat random
    ///  distribution. In this case, the compression algorithm is tuned to
    ///  compress them better.
    Filtered = 1,
    /// Use this value to force Huffman encoding only (no string match).
    HuffmanOnly = 2,
    /// Use this value to limit match distances to one (run-length encoding).
    RLE = 3,
    /// Using this value prevents the use of dynamic Huffman codes, allowing for
    /// a simpler decoder for special applications.
    Fixed = 4,
}

/// Writes an image to a file.
pub fn imwrite<P: AsRef<Path>>(path: P, mat: &Mat) -> Result<bool, Error> {
    let path = ::path_to_cstring(path)?;
    let path = path.as_ptr();
    let ret = unsafe { ffi::Image_IMWrite(path, mat.inner) };
    Ok(ret)
}

/// Writes an image to a file.
pub fn imwrite_with_params<P: AsRef<Path>>(
    path: P,
    mat: &Mat,
    mut flags: Vec<i32>,
) -> Result<bool, Error> {
    let int_vector = ffi::IntVector {
        val: flags.as_mut_ptr(),
        length: flags.len() as i32,
    };
    let path = ::path_to_cstring(path)?;
    let path = path.as_ptr();
    let ret = unsafe { ffi::Image_IMWrite_WithParams(path, mat.inner, int_vector) };
    Ok(ret)
}
