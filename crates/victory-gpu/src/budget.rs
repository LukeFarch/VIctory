//! The memory line: once every five seconds, in every build, where the
//! card's memory and the PC's go, by holder, in MB, beside the budgets.
//!
//! The diagnostics build has longer memory lines (`copystat`); this one is
//! cheap enough for the fast build: counters kept anyway, one look at the
//! table of windows, and two system queries.
//!
//! What the numbers are for: the console gives the title about 13.5 GB in
//! all, and the host's copies of it must fit beside it on a 31 GB PC with
//! a 16 GB card. A holder that keeps growing while the scene stays is the
//! one to look at; "evicted" and "waiting for room" say whether the
//! texture budget ([`crate::resource::limit`]) is being pressed.

use std::sync::atomic::Ordering::Relaxed;

/// The process's memory in bytes: in physical memory, and committed
/// (private bytes).
fn process_memory() -> (u64, u64) {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    (c.WorkingSetSize as u64, c.PagefileUsage as u64)
}

/// The PC's memory in bytes: physical memory in all, free, and what is
/// left to commit.
fn system_memory() -> (u64, u64, u64) {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: plain data the call fills in.
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: a sized structure.
    unsafe { GlobalMemoryStatusEx(&mut m) };
    (m.ullTotalPhys, m.ullAvailPhys, m.ullAvailPageFile)
}

fn mb(bytes: u64) -> u64 {
    bytes >> 20
}

/// Logs the memory line (the caller says when: every five seconds).
/// `VICTORY_MEMORY_LINE=0`: none.
pub(crate) fn line() {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_MEMORY_LINE").is_none_or(|v| v != "0")) {
        return;
    }
    let card = crate::host::host().and_then(|h| h.video_memory()).unwrap_or_default();
    let (working, private) = process_memory();
    let (total, free, commit_left) = system_memory();
    let textures = [0, 1].map(|i| crate::resource::HOST_TEXTURES[i].load(Relaxed));
    let filled = [0, 1].map(|i| crate::resource::KINDS[i].load(Relaxed));
    let let_go = [2, 3].map(|i| crate::resource::KINDS[i].load(Relaxed));
    let evicted = [0, 1].map(|i| crate::resource::EVICTED[i].load(Relaxed));
    let trimmed = [0, 1].map(|i| crate::resource::TRIMMED[i].load(Relaxed));
    let twins_and_targets = textures[1].saturating_sub(filled[1]);
    let copies = crate::shadow::holders();
    let staged = crate::texture::staging_alive();
    let alive: Vec<u64> = crate::host::ALIVE.iter().map(|a| a.load(Relaxed)).collect();
    let lists_never_reset = crate::host::LISTS[3].load(Relaxed);
    let kept = crate::pipeline::kept();
    tracing::info!(
        "memory budget: card {} of {} MB in use (the PC's memory shared with it {} MB); \
         title textures' host copies {} MB in {} of a limit of {} MB (evicted to make room: {} copies {} MB since the start, \
         {} slots made null, {} slots waiting for room, {} passes that found no room); \
         render targets, depth and twins {} MB in {}; mips left out of host copies: {} copies, {} MB not taken; \
         let go as idle since the start: {} copies {} MB. \
         PC: {} MB private ({} MB in memory) of {} MB, {} MB free, {} MB left to commit; \
         host copies of title memory {} MB in {} (of them windows outside buffers {} MB in {}), \
         in the card's memory {} MB and their \"before\" twins {} MB; texture staging {} MB in {}; \
         host lists {} ({} never reset), their scratch {} MB and other upload buffers {} MB; \
         shader containers {} MB, translations {} MB, the pipeline library's file {} MB",
        mb(card[1]),
        mb(card[0]),
        mb(card[3]),
        mb(filled[1]),
        filled[0],
        mb(crate::resource::limit()),
        evicted[0],
        mb(evicted[1]),
        crate::draw::EVICTED_SLOTS.load(Relaxed),
        crate::draw::HEAP_TEXTURES_EVICTED.load(Relaxed),
        crate::draw::HEAP_TEXTURES_FULL.load(Relaxed),
        mb(twins_and_targets),
        textures[0].saturating_sub(filled[0]),
        trimmed[0],
        mb(trimmed[1]),
        let_go[0],
        mb(let_go[1]),
        mb(private),
        mb(working),
        mb(total),
        mb(free),
        mb(commit_left),
        mb(copies[1]),
        copies[0],
        mb(copies[3]),
        copies[2],
        mb(copies[4]),
        mb(copies[5]),
        mb(staged[1]),
        staged[0],
        alive[0],
        lists_never_reset,
        mb(alive[2]),
        mb(alive[7]),
        mb(kept[1]),
        mb(kept[3]),
        mb(kept[4]),
    );
}
