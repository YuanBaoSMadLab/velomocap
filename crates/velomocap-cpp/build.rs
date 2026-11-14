use std::path::{Path, PathBuf};
use std::process::Command;

/// Locate a LibTorch distribution: the `LIBTORCH` environment variable, or the
/// torch wheel's C++ payload (headers + import libraries ship with it).
fn find_libtorch() -> Option<PathBuf> {
    if let Ok(value) = std::env::var("LIBTORCH") {
        let path = PathBuf::from(value);
        if is_libtorch(&path) {
            return Some(path);
        }
    }
    for candidate in [
        r"C:\Users\38820\miniconda3\Lib\site-packages\torch",
        r"C:\Python311\Lib\site-packages\torch",
    ] {
        let path = PathBuf::from(candidate);
        if is_libtorch(&path) {
            return Some(path);
        }
    }
    // Ask Python, which is how the wheel was installed in the first place.
    for python in ["python", "python3"] {
        let output = match Command::new(python)
            .args(["-c", "import torch,os;print(os.path.dirname(torch.__file__))"])
            .output()
        {
            Ok(output) => output,
            Err(_) => continue,
        };
        if output.status.success() {
            let path = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
            if is_libtorch(&path) {
                return Some(path);
            }
        }
    }
    None
}

fn is_libtorch(path: &Path) -> bool {
    path.join("include/torch/script.h").is_file()
        && path.join("lib/torch_cpu.lib").is_file()
        && path.join("lib/torch.lib").is_file()
}

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest.join("../..");
    let cpp = root.join("cpp");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file(cpp.join("src/kernels.cpp"))
        .include(cpp.join("include"))
        .warnings(false)
        .std("c++17");

    let msvc = std::env::var("CARGO_CFG_TARGET_ENV")
        .map(|v| v == "msvc")
        .unwrap_or(false);
    if msvc {
        build
            .flag_if_supported("/O2")
            .flag_if_supported("/fp:precise")
            .flag_if_supported("/EHsc")
            .flag_if_supported("/arch:AVX2");
    } else {
        build
            .flag_if_supported("-O3")
            .flag_if_supported("-fno-math-errno")
            .flag_if_supported("-mavx2")
            .flag_if_supported("-mfma");
    }

    // LibTorch is opt-in: `VELOMOCAP_LIBTORCH=1` links it, everything else
    // builds the stub. Linking it unconditionally would make every binary in the
    // workspace need `torch_cpu.dll` and the CUDA DLLs on `PATH` just to start,
    // which breaks `cargo test` (STATUS_DLL_NOT_FOUND) on any machine.
    //
    // Torch 2.13 headers need C++20, and the wheel's libraries use the release
    // iterator level, which this translation unit has to match.
    let wanted = std::env::var("VELOMOCAP_LIBTORCH").map(|v| v == "1").unwrap_or(false);
    match find_libtorch().filter(|_| wanted) {
        Some(torch) => {
            println!("cargo:rustc-cfg=libtorch");
            println!("cargo:rustc-env=VELOMOCAP_LIBTORCH={}", torch.display());
            let lib = torch.join("lib");
            println!("cargo:rustc-link-search=native={}", lib.display());
            println!("cargo:rustc-link-lib=torch_cpu");
            println!("cargo:rustc-link-lib=torch");
            println!("cargo:rustc-link-lib=c10");
            // Without these two, the process only registers the CPU backend and
            // `torch::jit::load(path, kCUDA)` fails with
            // "Could not run 'aten::empty_strided' ... from the 'CUDA' backend".
            // They exist in the CUDA wheel and are absent from the CPU one.
            if lib.join("torch_cuda.lib").is_file() {
                println!("cargo:rustc-link-lib=torch_cuda");
                if lib.join("c10_cuda.lib").is_file() {
                    println!("cargo:rustc-link-lib=c10_cuda");
                }
                println!("cargo:rustc-cfg=torch_cuda");
            }
            build
                .file(cpp.join("src/torch_sam3d.cpp"))
                .include(torch.join("include"))
                .include(torch.join("include/torch/csrc/api/include"))
                .std("c++20")
                .flag_if_supported("/D_ITERATOR_DEBUG_LEVEL=0")
                .flag_if_supported("/bigobj");
        }
        None => {
            if wanted {
                println!(
                    "cargo:warning=VELOMOCAP_LIBTORCH=1 but no LibTorch was found (set LIBTORCH)"
                );
            }
            build.file(cpp.join("src/torch_sam3d_stub.cpp"));
        }
    }

    build.compile("velomocap_kernels");

    println!("cargo:rerun-if-changed={}", cpp.join("src/kernels.cpp").display());
    println!(
        "cargo:rerun-if-changed={}",
        cpp.join("src/torch_sam3d.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cpp.join("include/velomocap/kernels.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cpp.join("include/velomocap/torch_sam3d.h").display()
    );
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=VELOMOCAP_LIBTORCH");
}
