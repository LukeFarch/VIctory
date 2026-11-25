//! The translator's fingerprint for the shader cache
//! (`src/shader_cache.rs`): a hash of the sources that decide what a
//! translated shader is (victory-shader, victory-pipeline's cache, the
//! cache's own encoding), the lock file that pins their dependencies, and
//! the features this build has. A new executable whose translator did not
//! change then finds the shaders the last one translated, instead of
//! translating all of them again (2-3 minutes on every new build).

use std::path::{Path, PathBuf};

fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs" || e == "toml" || e == "hlsl" || e == "bin") {
            out.push(path);
        }
    }
}

/// FNV-1a, 64 bits, at two seeds: no build dependency needed, and a
/// changed source is all it has to tell.
fn fnv(seed: u64, bytes: &[u8], mut h: u64) -> u64 {
    h ^= seed;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn main() {
    let here = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let crates = here.parent().unwrap().to_path_buf();
    let root = crates.parent().unwrap().to_path_buf();
    let mut inputs = Vec::new();
    for dir in ["victory-shader/src", "victory-pipeline/src"] {
        files(&crates.join(dir), &mut inputs);
        println!("cargo:rerun-if-changed={}", crates.join(dir).display());
    }
    for file in [
        crates.join("victory-shader/Cargo.toml"),
        crates.join("victory-pipeline/Cargo.toml"),
        here.join("src/shader_cache.rs"),
        root.join("Cargo.lock"),
    ] {
        println!("cargo:rerun-if-changed={}", file.display());
        inputs.push(file);
    }
    inputs.sort();
    let (mut a, mut b) = (0xcbf2_9ce4_8422_2325u64, 0x8422_2325_cbf2_9ce4u64);
    for path in &inputs {
        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };
        // The path relative to the workspace, so a clone elsewhere keys the same.
        let name = path.strip_prefix(&root).unwrap_or(path).to_string_lossy().replace('\\', "/");
        a = fnv(1, name.as_bytes(), a);
        b = fnv(2, name.as_bytes(), b);
        // Line endings as git may check them out on Windows or not.
        let bytes: Vec<u8> = bytes.into_iter().filter(|&c| c != b'\r').collect();
        a = fnv(3, &bytes, a);
        b = fnv(4, &bytes, b);
    }
    let mut features: Vec<String> = std::env::vars().map(|(k, _)| k).filter(|k| k.starts_with("CARGO_FEATURE_")).collect();
    features.sort();
    for f in &features {
        a = fnv(5, f.as_bytes(), a);
        b = fnv(6, f.as_bytes(), b);
    }
    println!("cargo:rustc-env=VICTORY_TRANSLATOR_HASH={a:016x}{b:016x}");
}
