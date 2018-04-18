extern crate bindgen;
extern crate cc;

use std::env;
use std::path::PathBuf;

#[cfg(unix)]
fn opencv_link() {
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    println!("cargo:rustc-link-lib=opencv_core");
    println!("cargo:rustc-link-lib=opencv_features2d");
    println!("cargo:rustc-link-lib=opencv_xfeatures2d");
    println!("cargo:rustc-link-lib=opencv_highgui");
    println!("cargo:rustc-link-lib=opencv_imgcodecs");
    println!("cargo:rustc-link-lib=opencv_imgproc");
    println!("cargo:rustc-link-lib=opencv_objdetect");
    println!("cargo:rustc-link-lib=opencv_text");
    println!("cargo:rustc-link-lib=opencv_videoio");
    println!("cargo:rustc-link-lib=opencv_video");
    if cfg!(feature = "cuda") {
        println!("cargo:rustc-link-lib=opencv_cudaobjdetect");
    }
}

fn source(module: &str) -> PathBuf {
    let mut path = PathBuf::from("gocv");
    path.push(module);
    path.set_extension("cpp");
    path
}

fn generate_binding() {
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    bindgen::builder()
        .header("wrapper.h")
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn main() {
    generate_binding();

    let modules = vec![
        "core",
        "dnn",
        "features2d",
        "highgui",
        "imgcodecs",
        "imgproc",
        "objdetect",
        "version",
        "video",
        "videoio",
    ];

    cc::Build::new()
        .flag("-std=c++11")
        .warnings(false)
        .cpp(true)
        .files(modules.iter().map(|m| source(m)))
        .compile("cv");

    opencv_link();
}
