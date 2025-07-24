use std::env;
use std::path::PathBuf;

fn main() {
    let libtorch_path = env::var("LIBTORCH_PATH")
        .unwrap_or_else(|_| "../libtorch".to_string());
    
    println!("cargo:rustc-link-search=native={}/lib", libtorch_path);
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=c10");
    
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rustc-link-lib=dylib=torch_cuda");
        println!("cargo:rustc-link-lib=dylib=c10_cuda");
    }
    
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}/include", libtorch_path))
        .clang_arg(format!("-I{}/include/torch/csrc/api/include", libtorch_path))
        .clang_arg("-std=c++17")
        .clang_arg("-xc++")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");
    
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
    
    cc::Build::new()
        .cpp(true)
        .file("src/cpp/torch_wrapper.cpp")
        .include(format!("{}/include", libtorch_path))
        .include(format!("{}/include/torch/csrc/api/include", libtorch_path))
        .flag("-std=c++17")
        .flag("-D_GLIBCXX_USE_CXX11_ABI=0")
        .compile("torch_wrapper");
}