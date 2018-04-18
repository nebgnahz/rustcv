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
