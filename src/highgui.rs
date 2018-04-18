//! [High-level GUI](https://docs.opencv.org/master/d7/dfc/group__highgui.html).

use opencv_sys as ffi;
use std::ffi::CString;
use Error;
use core::Mat;

/// Window is a wrapper around OpenCV's "HighGUI" named windows. While OpenCV
/// was designed for use in full-scale applications and can be used within
/// functionally rich UI frameworks (such as Qt*, WinForms*, or Cocoa*) or
/// without any UI at all, sometimes there it is required to try functionality
/// quickly and visualize the results. This is what the HighGUI module has been
/// designed for.
#[derive(Debug)]
pub struct Window {
    name: CString,
    open: bool,
}

/// Flags for [highgui_named_window](fn.highgui_named_window.html). This only
/// supports a subset of all cv::WindowFlags because C/C++ allows enum with the
/// same value but Rust is stricter.
#[repr(C)]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum WindowFlag {
    /// the window can be resized (no constraint) or switched to fullscreen.
    Normal = 0x00000000,
    /// the window is constrained by the image displayed.
    Autosize = 0x00000001,
    /// the window is with opengl support.
    Opengl = 0x00001000,
    /// the window can be resized arbitrarily (no ratio constraint).
    FreeRatio = 0x00000100,
}

impl Window {
    /// Creates a new named window.
    pub fn new(name: &str, flag: WindowFlag) -> Result<Self, Error> {
        let s = CString::new(name)?;
        unsafe {
            ffi::Window_New(s.as_ptr(), flag as i32);
        }
        Ok(Window {
            name: s,
            open: false,
        })
    }

    /// Checks if the Window is open.
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Displays an image in the specified window. This function should be
    /// followed by the `WaitKey` function which displays the image for
    /// specified milliseconds. Otherwise, it won't display the image.
    pub fn show(&self, image: &Mat) {
        unsafe { ffi::Window_IMShow(self.name.as_ptr(), image.inner) }
    }

    /// Waits for a pressed key. This function is the only method in OpenCV's
    /// HighGUI that can fetch and handle events, so it needs to be called
    /// periodically for normal event processing
    pub fn wait_key(&self, delay: i32) -> i32 {
        unsafe { ffi::Window_WaitKey(delay) }
    }
}
