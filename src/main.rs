// This resembles the OpenCV read image example code:
// http://docs.opencv.org/3.1.0/db/deb/tutorial_display_image.html
extern crate rustcv;
use rustcv::imgcodecs::*;
use rustcv::highgui::*;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 {
        println!("Usage: display_image ImageToLoadAndDisplay");
        std::process::exit(-1);
    }

    let image = imread(&args[1], ImageReadMode::Color).expect("Failed to read from path");
    let window = Window::new("Display", WindowFlag::Normal).expect("Failed to open window");
    window.show(&image);
    window.wait_key(0);
}
