//! `VICTORY_REBAR` (opt-in, off by default): host copies of title memory in
//! the card's own memory that the CPU maps (Resizable BAR, D3D12's GPU
//! upload heap, `D3D12_HEAP_TYPE_GPU_UPLOAD`).
//!
//! Adapted from the `rebar` module of the VIctory forks vitutu1/victory-perf
//! and ZigZagBoi1/ScarlettPort (GPL-3.0, as this project): the placement
//! rule and its tests are theirs. Added here: shader-written memory is kept
//! out (and moved out when found), CPU reads of placed copies are counted,
//! and level 3 for the big buffers the card twin path held.
//!
//! The console's memory is one: what the CPU writes, the GPU reads where it
//! lies. Here a host copy of title memory is either in the PC's memory
//! (`Host::shared_buffer`: the CPU writes it at full speed, every shader
//! load crosses the bus) or, for big title buffers, in the card's memory
//! with a second side in the PC's memory and a copy between the two before
//! each submission (`crate::shadow::cards`). With a large CPU-visible
//! window on the card's memory the CPU writes the card's side itself: no
//! PC-side twin, no copy before the submission, shader loads at the card's
//! speed.
//!
//! The title's own addresses cannot be the card's memory: D3D12 maps card
//! memory where the driver chooses, and the title (and its allocator, which
//! places graphics memory itself) picks its addresses. So the title's
//! writes are still found (write watch) and copied, but straight into the
//! card's memory, and nothing then moves over the bus at draw time.
//!
//! Why the in-place try (`VICTORY_INPLACE=3`) failed and what this avoids:
//! - shaders read PC memory over the bus there (latency-bound): here they
//!   read the card's own memory;
//! - reads of CPU-visible card memory by the CPU are uncached (tens of MB/s
//!   against GB/s): what VIctory reads back from a copy (snapshots before
//!   shader writes, copy-back compares, quarantine merges, settles) would
//!   crawl. So only memory shaders do not write is placed; a placed copy
//!   that shaders turn out to write has its pages noted
//!   ([`shader_wrote`]), is let go as soon as nothing holds it, and the
//!   copy made next for that memory stays in the PC's memory. Every CPU
//!   read of a placed copy is counted ([`cpu_reads`]) and the count is in
//!   the window caption and the log.
//! - its other copies of the same bytes saw the card's writes late: here
//!   a placed copy is a copy like any other, written only by refreshes,
//!   with the same flights and waits as before.
//!
//! Levels:
//! - `VICTORY_REBAR=1`: windows of title memory outside buffers (the
//!   title's own allocator: interface vertices, mesh streams, constants).
//! - `VICTORY_REBAR=2`: also title buffers below the card-buffer size.
//! - `VICTORY_REBAR=3`: also the big title buffers and windows the card
//!   twin path (`VICTORY_CARD_BUFFERS`, window cards) would have held:
//!   no staging side and no card send for them.
//!
//! Without the heap (Resizable BAR off: a 256 MB BAR; an older driver or
//! system; the software adapter) nothing changes: every copy is made as
//! before, and the start line says why.
//!
//! `VICTORY_REBAR_MB` sets the most bytes placed (default: a quarter of
//! the card's budget, at most 4 GiB); a copy goes there only while a fifth
//! of the card's budget stays free.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// What a copy is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// A window of title memory outside buffers.
    Window,
    /// A title buffer's copy (one too small for the card-buffer path).
    Buffer,
    /// A copy the card twin path would hold (a big title buffer, or a
    /// window with room on the card).
    Big,
}

/// Where a copy goes, and why not to the card's mapped memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Decision {
    Rebar,
    Off,
    Unsupported,
    /// The level does not place this kind.
    NotThisKind,
    /// The title reads it on the CPU, or shaders wrote it before.
    CpuRead,
    OverBudget,
    /// The card's memory is nearly full.
    CardFull,
}

/// What a decision is made from.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Ask {
    pub level: u8,
    pub supported: bool,
    pub kind: Kind,
    pub size: u64,
    pub cpu_read: bool,
    /// Bytes already placed, and the most that may be.
    pub placed: u64,
    pub most: u64,
    /// The card's budget and use, if known.
    pub card: Option<(u64, u64)>,
}

/// The placement rule (pure: tested below).
pub(crate) fn decide(ask: &Ask) -> Decision {
    if ask.level == 0 {
        return Decision::Off;
    }
    if !ask.supported {
        return Decision::Unsupported;
    }
    if ask.cpu_read {
        return Decision::CpuRead;
    }
    match ask.kind {
        Kind::Window => {}
        Kind::Buffer if ask.level >= 2 => {}
        Kind::Buffer => return Decision::NotThisKind,
        Kind::Big if ask.level >= 3 => {}
        Kind::Big => return Decision::NotThisKind,
    }
    if ask.placed.saturating_add(ask.size) > ask.most {
        return Decision::OverBudget;
    }
    if let Some((budget, used)) = ask.card
        && used.saturating_add(ask.size) > budget / 10 * 8
    {
        return Decision::CardFull;
    }
    Decision::Rebar
}

/// `VICTORY_REBAR`: 0 (unset, or anything else), 1, 2 or 3.
pub(crate) fn level_of(value: Option<&str>) -> u8 {
    match value.map(str::trim) {
        Some("1") => 1,
        Some("2") => 2,
        Some("3") => 3,
        _ => 0,
    }
}

/// The default most bytes: a quarter of the card's budget, at most 4 GiB
/// (256 MiB when the budget is unknown).
pub(crate) fn default_most(card_budget: Option<u64>) -> u64 {
    match card_budget {
        Some(b) => (b / 4).min(4 << 30),
        None => 256 << 20,
    }
}

pub(crate) fn level() -> u8 {
    static LEVEL: OnceLock<u8> = OnceLock::new();
    *LEVEL.get_or_init(|| level_of(std::env::var("VICTORY_REBAR").ok().as_deref()))
}

/// Whether the host's device has GPU upload heaps: set once at the host's
/// start ([`probe`]).
static SUPPORTED: OnceLock<bool> = OnceLock::new();

pub(crate) fn supported() -> bool {
    SUPPORTED.get().copied().unwrap_or(false)
}

fn most() -> u64 {
    static MOST: OnceLock<u64> = OnceLock::new();
    *MOST.get_or_init(|| {
        let env = std::env::var("VICTORY_REBAR_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok());
        match env {
            Some(mb) => mb << 20,
            None => default_most(
                crate::host::host()
                    .and_then(|h| h.video_memory())
                    .map(|[b, ..]| b),
            ),
        }
    })
}

/// Bytes placed alive, copies placed alive, and copies that wanted to be
/// placed but were not (over the budget, card full, creation failed).
static PLACED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// The next multiple of 64 MiB of placed bytes at which a line is written.
static NEXT_LINE: AtomicU64 = AtomicU64::new(64 << 20);

/// At the host's start: whether the device has GPU upload heaps, and the
/// line that says so.
pub(crate) fn probe(
    device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
    card_budget: Option<u64>,
) {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_FEATURE_D3D12_OPTIONS16, D3D12_FEATURE_DATA_D3D12_OPTIONS16,
    };
    let mut o16 = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
    // SAFETY: a plain query that fills our structure.
    let asked = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS16,
            (&raw mut o16).cast(),
            size_of_val(&o16) as u32,
        )
    };
    let supported = asked.is_ok() && o16.GPUUploadHeapSupported.as_bool();
    let _ = SUPPORTED.set(supported);
    let level = level();
    let most_mb = match std::env::var("VICTORY_REBAR_MB")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
    {
        Some(mb) => mb,
        None => default_most(card_budget) >> 20,
    };
    tracing::info!(
        gpu_upload_heaps = supported,
        options16_known = asked.is_ok(),
        card_budget_mb = card_budget.map(|b| b >> 20),
        level,
        most_mb,
        "Resizable BAR (D3D12 GPU upload heaps): {}",
        match (supported, level) {
            (true, 0) => "available, not used (VICTORY_REBAR=1, 2 or 3 places host copies there)",
            (true, 1) => "used for windows of title memory",
            (true, 2) => "used for windows of title memory and small title buffers",
            (true, _) =>
                "used for windows, small title buffers and the big copies the card twins held",
            (false, 0) => "not available",
            (false, _) =>
                "NOT available: VICTORY_REBAR is ignored, copies are made as before \
                           (Resizable BAR off in the firmware/driver, or no OS/driver support)",
        }
    );
}

/// Whether a copy of `size` bytes of `kind` goes to the card's mapped
/// memory; if it does, its bytes are counted as placed (the caller gives
/// them back with [`released`] if the buffer cannot be made, or when it
/// goes).
pub(crate) fn reserve(kind: Kind, size: u64, cpu_read: bool) -> bool {
    let level = level();
    if level == 0 || !supported() {
        return false;
    }
    let host = crate::host::host();
    let card = host.and_then(|h| h.video_memory()).map(|[b, u, ..]| (b, u));
    let ask = Ask {
        level,
        supported: true,
        kind,
        size,
        cpu_read,
        placed: PLACED[0].load(Ordering::Relaxed),
        most: most(),
        card,
    };
    match decide(&ask) {
        Decision::Rebar => {}
        Decision::OverBudget | Decision::CardFull => {
            if PLACED[2].fetch_add(1, Ordering::Relaxed) == 0 {
                tracing::warn!(
                    ?kind,
                    size,
                    "Resizable BAR: a copy stays in the PC's memory (budget or card full)"
                );
            }
            return false;
        }
        _ => return false,
    }
    let now = PLACED[0].fetch_add(size, Ordering::Relaxed) + size;
    PLACED[1].fetch_add(1, Ordering::Relaxed);
    let line = NEXT_LINE.load(Ordering::Relaxed);
    if now >= line
        && NEXT_LINE
            .compare_exchange(
                line,
                now / (64 << 20) * (64 << 20) + (64 << 20),
                Ordering::Relaxed,
                Ordering::Relaxed,
            )
            .is_ok()
    {
        tracing::info!(
            placed_mb = now >> 20,
            copies = PLACED[1].load(Ordering::Relaxed),
            "Resizable BAR: host copies placed in the card's mapped memory"
        );
    }
    true
}

/// Gives back what [`reserve`] counted.
pub(crate) fn released(size: u64) {
    PLACED[0].fetch_sub(size, Ordering::Relaxed);
    PLACED[1].fetch_sub(1, Ordering::Relaxed);
}

/// 64 KiB pages of title memory (title address / 64 KiB) that shaders
/// wrote through a placed copy, or a quarantined pass owned: copies made
/// later that hold any of them stay in the PC's memory.
static SHADER_PAGES: std::sync::Mutex<std::collections::BTreeSet<u64>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());
const PIECE: u64 = 1 << 16;

/// Shaders write `[start, start + len)` of a placed copy: noted, so that
/// no later copy of it is placed. Returns whether it was new.
pub(crate) fn shader_wrote(start: u64, len: u64) -> bool {
    if len == 0 {
        return false;
    }
    let mut pages = SHADER_PAGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut new = false;
    for page in start / PIECE..=(start + len - 1) / PIECE {
        new |= pages.insert(page);
    }
    if new {
        let n = STATS[2].fetch_add(1, Ordering::Relaxed);
        if n < 8 || n.is_power_of_two() {
            tracing::info!(
                start = format_args!("{start:#x}"),
                len,
                times = n + 1,
                "Resizable BAR: shaders write a placed copy; that memory's next copy stays in the PC's memory"
            );
        }
    }
    new
}

/// Whether shaders wrote any of `[start, start + len)` through a placed
/// copy before ([`shader_wrote`]).
pub(crate) fn shader_written(start: u64, len: u64) -> bool {
    if len == 0 {
        return false;
    }
    let pages = SHADER_PAGES.lock().unwrap_or_else(|e| e.into_inner());
    pages
        .range(start / PIECE..=(start + len - 1) / PIECE)
        .next()
        .is_some()
}

/// Bytes the CPU read from placed copies (uncached: each one is slow),
/// times it did, placed copies let go to move out, and shader-written
/// ranges found.
static STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// The CPU reads `n` bytes of a placed copy.
pub(crate) fn cpu_reads(n: usize) {
    if n == 0 {
        return;
    }
    let before = STATS[0].fetch_add(n as u64, Ordering::Relaxed);
    let times = STATS[1].fetch_add(1, Ordering::Relaxed);
    if times == 0 || (before >> 26) != ((before + n as u64) >> 26) {
        tracing::warn!(
            read_mb = (before + n as u64) >> 20,
            times = times + 1,
            "Resizable BAR: the CPU reads placed copies (uncached card memory: slow)"
        );
    }
}

/// A placed copy shaders wrote is let go to move out.
pub(crate) fn moved_out() {
    STATS[3].fetch_add(1, Ordering::Relaxed);
}

/// For the caption: MB placed, copies placed, MB the CPU read back from
/// them, and copies moved out.
pub(crate) fn caption() -> Option<String> {
    if level() == 0 {
        return None;
    }
    if !supported() {
        return Some("rebar OFF (no heap)".to_owned());
    }
    Some(format!(
        "rebar{} {} MB/{} rd {} MB out {}",
        level(),
        PLACED[0].load(Ordering::Relaxed) >> 20,
        PLACED[1].load(Ordering::Relaxed),
        STATS[0].load(Ordering::Relaxed) >> 20,
        STATS[3].load(Ordering::Relaxed),
    ))
}

/// Makes the CPU's writes into placed copies (write-combined) visible
/// before the card is told to run: called before a submission.
pub(crate) fn flush_writes() {
    if level() != 0 && PLACED[1].load(Ordering::Relaxed) != 0 {
        // SAFETY: a store fence has no preconditions.
        unsafe { std::arch::x86_64::_mm_sfence() };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;

    fn ask(level: u8, kind: Kind) -> Ask {
        Ask {
            level,
            supported: true,
            kind,
            size: 4 * MIB,
            cpu_read: false,
            placed: 0,
            most: GIB,
            card: Some((11 * GIB, 2 * GIB)),
        }
    }

    #[test]
    fn the_switch_reads_only_one_and_two() {
        assert_eq!(level_of(None), 0);
        assert_eq!(level_of(Some("0")), 0);
        assert_eq!(level_of(Some(" 1 ")), 1);
        assert_eq!(level_of(Some("2")), 2);
        assert_eq!(level_of(Some("3")), 3);
        assert_eq!(level_of(Some("yes")), 0);
        assert_eq!(level_of(Some("4")), 0);
    }

    #[test]
    fn off_or_unsupported_places_nothing() {
        assert_eq!(decide(&ask(0, Kind::Window)), Decision::Off);
        let mut a = ask(2, Kind::Window);
        a.supported = false;
        assert_eq!(decide(&a), Decision::Unsupported);
    }

    #[test]
    fn level_one_places_windows_only_level_two_buffers_too() {
        assert_eq!(decide(&ask(1, Kind::Window)), Decision::Rebar);
        assert_eq!(decide(&ask(1, Kind::Buffer)), Decision::NotThisKind);
        assert_eq!(decide(&ask(2, Kind::Buffer)), Decision::Rebar);
        assert_eq!(decide(&ask(2, Kind::Big)), Decision::NotThisKind);
        assert_eq!(decide(&ask(3, Kind::Big)), Decision::Rebar);
    }

    #[test]
    fn cpu_read_buffers_never_go_there() {
        for level in [1, 2, 3] {
            for kind in [Kind::Window, Kind::Buffer, Kind::Big] {
                let mut a = ask(level, kind);
                a.cpu_read = true;
                assert_eq!(decide(&a), Decision::CpuRead);
            }
        }
    }

    #[test]
    fn the_budget_holds_to_the_byte() {
        let mut a = ask(1, Kind::Window);
        a.placed = GIB - 4 * MIB;
        assert_eq!(decide(&a), Decision::Rebar);
        a.placed += 1;
        assert_eq!(decide(&a), Decision::OverBudget);
        a.placed = u64::MAX;
        assert_eq!(decide(&a), Decision::OverBudget);
    }

    #[test]
    fn a_fifth_of_the_card_stays_free() {
        let mut a = ask(1, Kind::Window);
        a.card = Some((10 * GIB, 8 * GIB - 4 * MIB));
        assert_eq!(decide(&a), Decision::Rebar);
        a.card = Some((10 * GIB, 8 * GIB));
        assert_eq!(decide(&a), Decision::CardFull);
        a.card = None;
        assert_eq!(decide(&a), Decision::Rebar);
    }

    #[test]
    fn default_budget_is_a_quarter_of_the_card_at_most_4_gib() {
        assert_eq!(default_most(None), 256 * MIB);
        assert_eq!(default_most(Some(11 * GIB)), 11 * GIB / 4);
        assert_eq!(default_most(Some(30 * GIB)), 4 * GIB);
    }
}

#[cfg(test)]
mod written_tests {
    use super::*;

    #[test]
    fn shader_written_pages_are_found_by_overlap() {
        let base = 0x7777_0000_0000u64;
        assert!(!shader_written(base, 0x10000));
        assert!(shader_wrote(base + 0x2_0010, 4));
        assert!(!shader_wrote(base + 0x2_0020, 4));
        assert!(shader_written(base, 0x3_0000));
        assert!(!shader_written(base, 0x2_0000));
        assert!(shader_written(base + 0x2_ffff, 1));
        assert!(!shader_written(base + 0x3_0000, 0x1_0000));
        assert!(!shader_written(base, 0));
    }
}
