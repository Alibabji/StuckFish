use std::{env, path::Path};

fn libtorch_lib_dir() -> Option<String> {
    if let Ok(path) = env::var("DEP_TCH_LIBTORCH_LIB") {
        return Some(path);
    }
    if let Ok(path) = env::var("LIBTORCH_LIB") {
        return Some(path);
    }
    if let Ok(path) = env::var("LIBTORCH") {
        return Some(format!("{path}/lib"));
    }
    None
}

fn main() {
    let lib_dir = match libtorch_lib_dir() {
        Some(path) => path,
        None => {
            println!(
                "cargo:warning=Could not locate libtorch lib directory; skipping CUDA link hints"
            );
            return;
        }
    };
    let link = |name: &str| println!("cargo:rustc-link-lib={name}");

    let push_link_if_exists = |name: &str| {
        let file = format!("{}/lib{}.so", lib_dir, name);
        if Path::new(&file).exists() {
            println!("cargo:warning=linking {name} from {file}");
            link(name);
        }
    };

    push_link_if_exists("torch_cuda");
    push_link_if_exists("torch_cuda_cpp");
    push_link_if_exists("torch_cuda_cu");
    push_link_if_exists("torch_cuda_linalg");
    push_link_if_exists("c10_cuda");
}
