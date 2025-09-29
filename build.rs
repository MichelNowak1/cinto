
fn main() {

    let dst = cmake::build("cinto_root_geometry");
    println!("cargo:rustc-link-search=native={}", std::env::var("OUT_DIR").unwrap());
    println!("cargo:rustc-link-lib=dylib=cinto_root_geometry");
    println!("cargo:rustc-link-lib=dylib=stdc++"); 
}
