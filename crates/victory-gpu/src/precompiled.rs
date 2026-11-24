//! The shader containers behind the programs of the device's precompiled
//! pipeline methods (0x1c8 graphics, 0x218 compute).
//!
//! Those methods hand over console machine code, not containers. Its first
//! 24 bytes carry the program's hash (the console compiler starts every
//! program with it), and every such program is also in a container of the
//! title's shader pack. `victory translate --warm` writes, under the cache
//! folder `precompiled`, an index from those 24 bytes to the container's
//! hash (`index.txt`, one `<head hex> <stage> <container hash>` line each)
//! and each container once (`<container hash>.cdxil`); a run looks the
//! code up there and makes the pipeline from the container as usual.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// The bytes of a program's code that name it.
pub const HEAD: usize = 24;

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Looks programs up in (or, when warming, writes them to) `dir`.
pub fn use_precompiled(dir: PathBuf) {
    let _ = DIR.set(dir);
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// When warming: records that the program whose code starts with `code`
/// (at least [`HEAD`] bytes) is in `container`.
pub fn record(code: &[u8], stage: &str, container: &[u8]) {
    let Some(dir) = DIR.get() else { return };
    if code.len() < HEAD {
        return;
    }
    static INDEX: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
    let Some(index) = INDEX.get_or_init(|| {
        std::fs::create_dir_all(dir).ok()?;
        std::fs::File::create(dir.join("index.txt")).ok().map(Mutex::new)
    }) else {
        return;
    };
    let hash = blake3::hash(container).to_hex();
    let file = dir.join(format!("{hash}.cdxil"));
    if !file.exists() {
        let _ = std::fs::write(&file, container);
    }
    let line = format!("{} {stage} {hash}\n", hex(&code[..HEAD]));
    let _ = index.lock().unwrap_or_else(|e| e.into_inner()).write_all(line.as_bytes());
}

/// The index, read on first use: code head to (stage, container hash).
fn index() -> &'static HashMap<[u8; HEAD], (String, String)> {
    static INDEX: OnceLock<HashMap<[u8; HEAD], (String, String)>> = OnceLock::new();
    INDEX.get_or_init(|| {
        let text = DIR
            .get()
            .and_then(|d| std::fs::read_to_string(d.join("index.txt")).ok())
            .unwrap_or_default();
        let mut map = HashMap::new();
        for line in text.lines() {
            let mut words = line.split_whitespace();
            let (Some(head), Some(stage), Some(hash)) = (words.next(), words.next(), words.next())
            else {
                continue;
            };
            let Some(key) = unhex(head) else { continue };
            map.entry(key).or_insert((stage.to_owned(), hash.to_owned()));
        }
        tracing::info!(
            programs = map.len(),
            "precompiled pipeline index ({})",
            if map.is_empty() { "none: run `victory translate --warm`" } else { "read" }
        );
        map
    })
}

fn unhex(text: &str) -> Option<[u8; HEAD]> {
    let mut out = [0u8; HEAD];
    if text.len() != 2 * HEAD {
        return None;
    }
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(text.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(out)
}

/// The container of the program whose code starts with `head`, and its
/// stage as the index names it (`Vs`, `Ps`, `Cs`, ...).
pub(crate) fn container(head: &[u8; HEAD]) -> Option<(String, Vec<u8>)> {
    let (stage, hash) = index().get(head)?;
    let bytes = std::fs::read(DIR.get()?.join(format!("{hash}.cdxil"))).ok()?;
    Some((stage.clone(), bytes))
}
