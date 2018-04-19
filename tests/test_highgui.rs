extern crate rustcv;
use rustcv::highgui::*;

fn main() {
    let mut window = Window::new("test", WindowFlag::Autosize).expect("failed to create window");
    assert_eq!(window.name(), "test");
    let val = window.wait_key(1);
    assert_eq!(val, -1);
    assert!(window.is_open());

    window.set_property(WindowProperty::Fullscreen, WindowFlag::Normal);
    let prop = window.get_property(WindowProperty::Fullscreen);
    assert_eq!(prop as i32, WindowFlag::Normal as i32);
    window
        .set_title("My new title")
        .expect("failed to set title");
    window.move_window(100, 100);
    window.resize(100, 100);
    window.close();
    assert!(!window.is_open());
    ::std::mem::forget(window);
}
