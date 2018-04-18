//! This library primarily provides a binding and API for OpenCV 3.x.
//!
//! The wrapper is a direct steal from
//! [GoCV](https://github.com/hybridgroup/gocv).

#![deny(missing_docs)]
#![deny(missing_debug_implementations)]
#![deny(missing_copy_implementations)]
#![deny(trivial_casts)]
#![deny(trivial_numeric_casts)]
#![deny(unused_import_braces)]
#![deny(unused_qualifications)]

#[macro_use]
extern crate failure;
#[macro_use]
extern crate num_derive;
extern crate num_traits;
extern crate opencv_sys;

use failure::Error;
use std::path::{Path, PathBuf};
use std::ffi::CString;

pub mod core;
pub mod imgcodecs;
pub mod highgui;

#[derive(Debug, Fail)]
/// Custom errors.
pub enum CvError {
    #[fail(display = "invalid path: {:?}", _0)]
    /// Indicates that path was invalid
    InvalidPath(PathBuf),
    #[fail(display = "EntryNotFound: {:?}", _0)]
    /// Indicates that there is no entry on specified path
    EntryNotFound(PathBuf),
    #[fail(display = "failed to convert from primitive: {}", value)]
    /// Indicates that conversion from primitive to enum type is failed
    EnumFromPrimitiveConversionError {
        /// Value that caused an error
        value: i32,
    },
    #[fail(display = "Unknown error: {:?}", _0)]
    /// Indicates that error occurred in C++ code
    UnknownError(String),
    #[fail(display = "Non ascii characters found in string: {:?}", _0)]
    /// Indicates that string contains non ascii characters
    UnicodeChars(String),
}

fn path_to_cstring<P: AsRef<Path>>(path: P) -> Result<CString, Error> {
    let path = path.as_ref();
    let x = path.to_str().ok_or(CvError::InvalidPath(path.into()))?;
    let result = CString::new(x)?;
    Ok(result)
}
