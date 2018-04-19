//! [Deep Neural Network](https://docs.opencv.org/master/d6/d0f/group__dnn.html).

use opencv_sys as ffi;
use core::{Mat, Scalar, Size};
use str_to_cstring;

/// Cascade classifier class for object detection.
#[derive(Debug)]
pub struct Net {
    inner: ffi::Net,
}

impl Net {
    /// Reads a network model stored in Caffe framework's format.
    pub fn from_caffe(prototxt: &str, model: &str) -> Self {
        Net {
            inner: unsafe {
                ffi::Net_ReadNetFromCaffe(str_to_cstring(prototxt), str_to_cstring(model))
            },
        }
    }

    /// Reads a network model stored in Caffe framework's format.
    pub fn from_tensorflow(model: &str) -> Self {
        Net {
            inner: unsafe { ffi::Net_ReadNetFromTensorflow(str_to_cstring(model)) },
        }
    }

    /// Returns true if there are no layers in the network.
    pub fn empty(&self) -> bool {
        unsafe { ffi::Net_Empty(self.inner) }
    }

    /// Sets the new value for the layer output blob.
    pub fn set_input(&mut self, blob: &Mat, name: &str) {
        unsafe { ffi::Net_SetInput(self.inner, blob.inner, str_to_cstring(name)) }
    }

    /// Runs forward pass to compute output of layer with name outputName.
    pub fn forward(&self, output_name: &str) -> Mat {
        Mat::from(unsafe { ffi::Net_Forward(self.inner, str_to_cstring(output_name)) })
    }
}

/// Creates 4-dimensional blob from image. Optionally resizes and crops image
/// from center, subtract mean values, scales values by scalefactor, swap Blue
/// and Red channels.
pub fn blob_from_image(
    img: &Mat,
    scale: f64,
    size: Size,
    mean: Scalar,
    swap_rb: bool,
    crop: bool,
) -> Mat {
    Mat::from(unsafe { ffi::Net_BlobFromImage(img.inner, scale, size, mean, swap_rb, crop) })
}

impl Drop for Net {
    fn drop(&mut self) {
        unsafe { ffi::Net_Close(self.inner) }
    }
}

/// Extracts a single (2d)channel from a 4 dimensional blob structure (this
///  might e.g. contain the results of a SSD or YOLO detection, a bones
///  structure from pose detection, or a color plane from Colorization)
pub fn get_blob_channel(blob: &Mat, image_index: i32, channel_index: i32) -> Mat {
    Mat::from(unsafe { ffi::Net_GetBlobChannel(blob.inner, image_index, channel_index) })
}

/// Retrieves the 4 dimensional size information in (N,C,H,W) order
pub fn get_blob_size(blob: &Mat) -> Scalar {
    unsafe { ffi::Net_GetBlobSize(blob.inner) }
}
