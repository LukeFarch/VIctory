//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection. Blocks found
//! wholly uncommitted are skipped while they remain so; one query can
//! confirm a whole region, including ordinary reservations between views.
//!
//! The title shows one physical page at several addresses. A block is
//! noted clean only while every other address of its bytes is read-only
//! too ([`guard_twins`]), and a write taken at one address makes the blocks
//! at the others not clean ([`stale_twins`]); so does a view mapped anew
//! ([`view_mapped`]). Before, a copy was guarded at its own address alone,
//! and a write through another left it as it was for the rest of the run:
//! the first room's walls were missing in every run without this and
//! there in every run with it (runs G0 and G1). `VICTORY_VIEW_TWINS=0`:
//! as before.
//!
//! shortcut: writes into physical pages through another view (xmem's
//! `write_physical`, DirectStorage's page destinations) are not seen at
//! the guarded addresses; they go to the texture pool, whose textures are
//! hashed, not guarded. A system call writing into a guarded block fails
//! (none is known to).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
    /// Page ranges asked ahead for a submission's refreshes ([`ask_ahead`])
    /// and the asking that did: `writes` answers for a range inside one
    /// from the journal, without another kernel call.
    ahead: Vec<(usize, usize)>,
    ahead_asked: u64,
    /// Of pages the kernel found written: which of their 4 KiB pieces, by
    /// asking, the latest few ([`merge_4k`]).
    pieces: BTreeMap<usize, [(u64, u16); PIECE_HISTORY]>,
    /// The ahead asking stands for a whole submission ([`ask_once`]): no
    /// other ahead asking is made, and `ahead` stays, until
    /// [`ask_once_end`].
    kept: bool,
}

static JOURNAL: std::sync::RwLock<Journal> = std::sync::RwLock::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    found: Vec::new(),
    ahead: Vec::new(),
    ahead_asked: 0,
    pieces: BTreeMap::new(),
    kept: false,
});

/// Askings of a page's 4 KiB pieces kept ([`Journal::pieces`]).
const PIECE_HISTORY: usize = 4;

/// Whether quarantine merges read only the 4 KiB pieces of a page the
/// kernel found written (`VICTORY_MERGE_4K=1`; off by default). The
/// journal is kept by 64 KiB page, and a title write of a few bytes made a
/// merge read and compare all 64 KiB: 2,240 merges a frame in run 15's
/// prison profile, 618 ms summed. The kernel answers by 4 KiB page; those
/// answers are kept too, and a merge takes the pieces written since its
/// last (the whole page where the journal knows less: a write the guards
/// or we noted, or an older asking than it keeps).
pub(crate) fn merge_4k() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MERGE_4K").is_some_and(|v| v == "1"))
}

/// Notes the kernel's answer `at` (a written 4 KiB page) under `now`.
fn note_piece(j: &mut Journal, at: usize, now: u64) {
    note_bits(j, at / PAGE, 1u16 << ((at % PAGE) / KERNEL_PAGE), now);
}

/// A write to page `page` that no kernel answer pins to 4 KiB pieces (the
/// views' answers, the guards', ours): every piece, so a later or same
/// asking's narrower answer cannot hide it from `pieces_since`.
fn note_whole(j: &mut Journal, page: usize, now: u64) {
    note_bits(j, page, u16::MAX, now);
}

fn note_bits(j: &mut Journal, page: usize, bit: u16, now: u64) {
    if !merge_4k() {
        return;
    }
    let hist = j.pieces.entry(page).or_insert([(0, 0); PIECE_HISTORY]);
    if hist[0].0 == now {
        hist[0].1 |= bit;
        return;
    }
    hist.rotate_right(1);
    hist[0] = (now, bit);
}

/// The 4 KiB pieces of page `page` written after asking `since`, as a
/// mask (bit `k`: bytes `k * 4096..(k + 1) * 4096` of the page); all of
/// them where the journal cannot tell.
pub(crate) fn pieces_since(page: usize, since: u64) -> u16 {
    let j = journal_shared();
    let last = j.written.get(&page).copied().unwrap_or(0);
    if last <= since {
        return 0;
    }
    let Some(hist) = j.pieces.get(&page) else {
        return u16::MAX;
    };
    // A later write than the kernel's answers (the guards', ours): all.
    if hist[0].0 < last {
        return u16::MAX;
    }
    let mut mask = 0;
    for &(asked, bits) in hist {
        if asked == 0 {
            // Fewer askings than kept: every one since is here.
            return mask;
        }
        if asked <= since {
            return mask;
        }
        mask |= bits;
    }
    // Older askings than kept may be after `since`.
    u16::MAX
}

/// The journal, alone.
fn journal() -> std::sync::RwLockWriteGuard<'static, Journal> {
    crate::trace::rw("lock wait: write-watch journal", &JOURNAL)
}

/// The journal, shared with other readers: the refresh threads' answers
/// from an asking made ahead ([`ask_ahead`], [`ask_once`]) only read it.
fn journal_shared() -> std::sync::RwLockReadGuard<'static, Journal> {
    crate::trace::shared("lock wait: write-watch journal (shared)", &JOURNAL)
}

/// Whether [`writes`] answers from an asking made ahead with the journal
/// shared (`VICTORY_JOURNAL_SHARED=1`; off by default). In run 5's prison
/// profile the refresh threads waited 459 ms a frame, summed, for the
/// journal; most of their askings are answered from the journal alone,
/// which needs no one else kept out. What is answered is the same.
fn shared_answers() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_JOURNAL_SHARED").is_some_and(|v| v == "1"))
}

/// Ranges of one submission closer than this are asked about together:
/// walking the pages between them costs less than another kernel call (a
/// call that resets written pages costs 30-70 microseconds whatever the
/// range).
const AHEAD_GAP: usize = (8 << 20) / PAGE;
/// The most one asking covers (its answer needs 8 bytes per 4 KiB page).
const AHEAD_MAX: usize = (1 << 30) / PAGE;

/// Asks the kernel about many ranges `(start, end)` at once, before the
/// [`writes`] of each: ranges close together are one asking, and `writes`
/// then answers from the journal until [`ask_ahead_done`]. The world's
/// submissions refresh thousands of host copies (6,664 askings a frame,
/// 456 ms, in run 14). Ranges that are not write-watched, or lie alone, are
/// left to `writes`.
pub(crate) fn ask_ahead(ranges: &[(usize, usize)]) {
    let _held = all_stripes();
    ask_ahead_held(ranges);
}

/// [`ask_ahead`], with the stripes held ([`all_stripes`]).
fn ask_ahead_held(ranges: &[(usize, usize)]) {
    let mut pieces: Vec<(usize, usize)> = ranges
        .iter()
        .filter(|r| r.0 < r.1)
        .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
        .filter(|&(first, _)| !known_view(first))
        .collect();
    if pieces.len() < 2 {
        return;
    }
    pieces.sort_unstable();
    let mut j = journal();
    let j = &mut *j;
    // The submission's one asking stands ([`ask_once`]): what it does not
    // hold is left to `writes`.
    if j.kept {
        return;
    }
    j.asked += 1;
    j.ahead_asked = j.asked;
    j.ahead.clear();
    let mut i = 0;
    while i < pieces.len() {
        let first = pieces[i].0;
        let mut last = pieces[i].1;
        let mut k = i + 1;
        while k < pieces.len()
            && pieces[k].0 <= last + AHEAD_GAP
            && last.max(pieces[k].1) - first <= AHEAD_MAX
        {
            last = last.max(pieces[k].1);
            k += 1;
        }
        ask_run(j, &pieces[i..k]);
        i = k;
    }
}

/// Since the last frame line: the kernel askings that took a millisecond
/// or more, their microseconds, and the longest asking's. An asking walks
/// page tables, some ten microseconds for most ranges; in run 75 the 2,644
/// askings of a frame took 219 ms in the lines where the pass over the
/// heap's views ran long, 25-50 ms in the others. These numbers say
/// whether a few askings wait for something or all of them are slow.
#[cfg(feature = "diagnostics")]
static SLOW: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

#[cfg(feature = "diagnostics")]
fn slow(us: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if us >= 1000 {
        SLOW[0].fetch_add(1, Relaxed);
        SLOW[1].fetch_add(us, Relaxed);
    }
    SLOW[2].fetch_max(us, Relaxed);
}

/// [`SLOW`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn slow_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| SLOW[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// The submission's refreshes are done: `writes` asks the kernel again.
pub(crate) fn ask_ahead_done() {
    let mut j = journal();
    if !j.kept {
        j.ahead.clear();
    }
}

/// Whether a submission asks the kernel once, before its first refresh,
/// about every range its lists refresh (`VICTORY_ASK_ONCE=1`; off by
/// default).
///
/// A submission's refreshes are done in groups, between the title's own
/// buffer copies (each copy flushes the refreshes recorded before it), and
/// every group asked the kernel again: 2,644 askings a frame, 70 a
/// submission. With the switch the journal answers all of them from one
/// asking made when the submission starts.
///
/// What changes: memory the title's other threads write while the
/// submission is being prepared is seen by the next submission's asking,
/// not by this one's later groups. What this submission's lists read was
/// written before they were submitted, so before the asking. Our own
/// writes into title memory while preparing are told to the journal
/// ([`wrote`]) or end the asking ([`ask_once_end`]).
pub(crate) fn once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default since run 81 (with card buffers: 375-450 ms a frame
    // against 573-647, the picture unchanged over four minutes);
    // `VICTORY_ASK_ONCE=0` asks per range, as before.
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_ONCE").is_none_or(|v| v != "0"))
}

/// With [`once`]: asks about `ranges` (everything a submission's lists
/// refresh) as [`ask_ahead`] does, and lets that asking stand until
/// [`ask_once_end`].
pub(crate) fn ask_once(ranges: &[(usize, usize)]) {
    if !once() {
        return;
    }
    ask_once_end();
    // View memory first, without the journal held: the guards answer for
    // the submission's view ranges joined (overlapping and touching ones
    // once), and the journal takes the answers below. Asked under the
    // journal, every refresh thread waited for the guards behind it (run
    // 15: 2,660 askings a frame, 436 ms of guard waits, 213 of journal).
    let views: Vec<((usize, usize), Option<Vec<usize>>)> = if views_once() {
        let mut v: Vec<(usize, usize)> = ranges
            .iter()
            .filter(|r| r.0 < r.1)
            .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
            .filter(|&(first, _)| known_view(first))
            .collect();
        v.sort_unstable();
        let mut joined: Vec<(usize, usize)> = Vec::with_capacity(v.len());
        for (a, b) in v {
            match joined.last_mut() {
                Some(last) if a <= last.1 => last.1 = last.1.max(b),
                _ => joined.push((a, b)),
            }
        }
        let _z = crate::trace::zone("guard: a submission's view ranges, once");
        joined.into_iter().map(|(a, b)| ((a, b), guard(a * PAGE, b * PAGE))).collect()
    } else {
        Vec::new()
    };
    let _held = all_stripes();
    ask_ahead_held(ranges);
    let mut j = journal();
    let j = &mut *j;
    if j.ahead.is_empty() {
        // `ask_ahead` asked nothing (fewer than two ranges).
        j.asked += 1;
        j.ahead_asked = j.asked;
    }
    // `ask_ahead` leaves a range that lies alone to `writes`, which asks
    // once per flush: with one asking for the submission that was every
    // such range asked again at each of its 70 flushes (run 76: 2,300
    // askings a frame with the switch on, as without it). They are asked
    // here, once; what is not write-watched stays with `writes`.
    for &(start, end) in ranges {
        if start >= end {
            continue;
        }
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        if j.ahead.iter().any(|&(a, b)| a <= first && last <= b) {
            continue;
        }
        if !known_view(first) && ask_kernel(j, first, last) {
            j.ahead.push((first, last));
        }
    }
    // The views' answers, under this asking.
    let now = j.ahead_asked;
    for ((first, last), pages) in views {
        if let Some(pages) = pages {
            for page in pages {
                j.written.insert(page, now);
                note_whole(j, page, now);
            }
            j.ahead.push((first, last));
        }
    }
    j.kept = !j.ahead.is_empty();
}

/// The submission's one asking stands no longer: its refreshes are done,
/// or title memory was written in a way the journal was not told.
pub(crate) fn ask_once_end() {
    let mut j = journal();
    if j.kept {
        j.kept = false;
        j.ahead.clear();
    }
}

/// We wrote title memory `[at, at + len)` while a submission's one asking
/// stands (the title's buffer copies, done at submission): its pages count
/// as written after that asking, so the refreshes that follow copy them.
pub(crate) fn wrote(at: usize, len: usize) {
    let mut j = journal();
    if !j.kept || len == 0 {
        return;
    }
    j.asked += 1;
    let now = j.asked;
    for page in at / PAGE..at.saturating_add(len).div_ceil(PAGE) {
        j.written.insert(page, now);
        note_whole(&mut j, page, now);
    }
}

/// One kernel asking for `pieces` (page ranges in order); by halves if it
/// fails, as a run across two reservations or into memory that is not
/// write-watched does. A piece alone is left to [`writes`].
fn ask_run(j: &mut Journal, pieces: &[(usize, usize)]) {
    if pieces.len() < 2 {
        return;
    }
    let first = pieces[0].0;
    let last = pieces.iter().map(|p| p.1).max().unwrap_or(first);
    if ask_kernel(j, first, last) {
        j.ahead.push((first, last));
        return;
    }
    let (a, b) = pieces.split_at(pieces.len() / 2);
    ask_run(j, a);
    ask_run(j, b);
}

/// Asks the kernel for the written pages of pages `[first, last)` and notes
/// them in the journal under the ahead asking; false (and nothing changed)
/// if the range is not write-watched memory of one reservation.
fn ask_kernel(j: &mut Journal, first: usize, last: usize) -> bool {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let mut _z = crate::trace::zone("GetWriteWatch: a submission's ranges together");
    _z.arg(((last - first) * PAGE) as u64);
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    if failed {
        drop(_z);
        let _f = crate::trace::zone("GetWriteWatch failed (not watched: cut in halves)");
        return false;
    }
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    let now = j.ahead_asked;
    for k in 0..count {
        let at = j.found[k];
        j.written.insert(at / PAGE, now);
        note_piece(j, at, now);
    }
    true
}

/// Whether the kernel is not asked about view memory the guards hold
/// (`VICTORY_WATCH_SKIP_VIEWS=1`; off by default). Views (xmem's page
/// pool) cannot be write-watched: every asking about one failed, and then
/// the guards answered. A submission's ranges asked together
/// ([`ask_ahead`]) were cut in halves after each failure, down to single
/// ranges: about two failed calls per view range, most of the 15,500
/// write-watch calls a frame of the S2b timeline. A range whose first
/// block the guards hold read-only is a view and goes to the guards at
/// once (only `protected`: `absent` blocks can lie in write-watched
/// reservations too).
fn skip_views() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WATCH_SKIP_VIEWS").is_some_and(|v| v == "1"))
}

/// Whether a submission's one asking ([`ask_once`]) also asks the guards
/// about the view memory its lists refresh (`VICTORY_VIEWS_ASK_ONCE=1`;
/// off by default; with `VICTORY_WATCH_SKIP_VIEWS=1`). Views were left to
/// [`writes`], which asked the guards at every refresh: 4,300 askings a
/// frame in run 14's prison profile, 438 ms summed with the kernel's, and
/// most of the 242 ms the threads waited for the guards' lock. Answered as
/// the kernel's askings are: what the title writes after the submission's
/// asking is seen by the next submission's.
fn views_once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| skip_views() && std::env::var_os("VICTORY_VIEWS_ASK_ONCE").is_some_and(|v| v == "1"))
}

/// Whether pages `[first, last)` begin in a block the guards hold
/// read-only: view memory, which the kernel cannot watch.
fn known_view(first: usize) -> bool {
    skip_views() && GUARDS.lock().unwrap_or_else(|e| e.into_inner()).protected.contains(&(first * PAGE / BLOCK))
}

/// The number of the latest asking: a page the journal notes written
/// under a higher one was written after this call.
pub(crate) fn asking() -> u64 {
    journal_shared().asked
}

/// What [`writes`] would say of `[start, end)` from the journal alone, if
/// the range lies in one asked ahead ([`ask_ahead`], [`ask_once`]): the
/// pages written since tracking began and the asking that last found each
/// written, with the number of the asking that stands (as [`writes`] gives
/// it). `None` if the kernel would have to be asked.
pub(crate) fn asked(start: usize, end: usize) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let j = journal_shared();
    j.ahead
        .iter()
        .any(|&(a, b)| a <= first && last <= b)
        .then(|| (j.ahead_asked, j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()))
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    if unlocked_ask() {
        return writes_unlocked(start, end, guarded);
    }
    writes_locked(start, end, guarded)
}

/// Whether [`writes`] asks the kernel without holding the journal
/// (`VICTORY_JOURNAL_UNLOCKED_ASK=1`; off by default).
///
/// The journal's lock was held across the kernel call and the view guards:
/// in run 5's prison profile the refresh threads waited 459 ms a frame,
/// summed, on it ("lock wait: write-watch journal"), behind each other's
/// kernel calls. With the switch an asking holds only the stripes of the
/// address space its range covers ([`STRIPES`]) across the kernel call,
/// and the journal just to number the asking and note what it found.
///
/// Why this is safe: two askings whose ranges share a page share a stripe,
/// so they are made one after the other, as before; an asking that resets
/// a page notes it before another asking about the page can be made.
/// Askings about different memory no longer wait for each other. The
/// asking's number is taken after the kernel answered, so it is higher
/// than any asking that noted pages meanwhile; the caller copies after
/// [`writes`] returns, so a write the number puts before the copy is in
/// the copy.
fn unlocked_ask() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_JOURNAL_UNLOCKED_ASK").is_some_and(|v| v == "1");
        if on {
            tracing::info!("VICTORY_JOURNAL_UNLOCKED_ASK=1: write-watch askings hold address stripes, not the journal, across the kernel call");
        }
        on
    })
}

/// Address-space stripes for [`unlocked_ask`]: stripe `k` holds every
/// `STRIPE`-sized piece whose number is `k` modulo the count.
const STRIPE: usize = 16 << 20;
static STRIPES: [Mutex<()>; 64] = [const { Mutex::new(()) }; 64];

/// The stripes of pages `[first, last)`, taken in order (so two askings
/// never wait on each other's stripes in a cycle). The ahead askings
/// ([`ask_ahead`], [`ask_once`]) take every stripe before the journal.
fn stripes(first: usize, last: usize) -> Vec<std::sync::MutexGuard<'static, ()>> {
    let (a, b) = (first * PAGE / STRIPE, (last * PAGE).div_ceil(STRIPE));
    let mut held = [false; 64];
    for s in a..b.max(a + 1) {
        held[s % 64] = true;
        if b - a >= 64 {
            held = [true; 64];
            break;
        }
    }
    let mut _z = crate::trace::zone("lock wait: write-watch stripes");
    held.iter()
        .enumerate()
        .filter(|&(_, &h)| h)
        .map(|(k, _)| STRIPES[k].lock().unwrap_or_else(|e| e.into_inner()))
        .collect()
}

/// Every stripe, for an asking that covers memory [`stripes`] cannot name
/// in advance; none (and no cost) with the switch off.
fn all_stripes() -> Vec<std::sync::MutexGuard<'static, ()>> {
    if !unlocked_ask() {
        return Vec::new();
    }
    let _z = crate::trace::zone("lock wait: write-watch stripes");
    STRIPES.iter().map(|s| s.lock().unwrap_or_else(|e| e.into_inner())).collect()
}

fn full_refresh() -> bool {
    // `VICTORY_FULL_REFRESH=1`: every page asked about counts as written,
    // so every host copy is brought up to date whole each time it is used
    // (slow; to tell a missed write from another cause of a stale copy).
    static FULL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FULL.get_or_init(|| {
        let on = std::env::var_os("VICTORY_FULL_REFRESH").is_some_and(|v| v == "1");
        if on {
            tracing::warn!("VICTORY_FULL_REFRESH=1: every page asked about counts as written; copies are refreshed whole (slow)");
        }
        on
    })
}

/// [`writes`] with [`unlocked_ask`].
fn writes_unlocked(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    thread_local! {
        static FOUND: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let full = full_refresh();
    let from_journal = |j: &Journal, now: u64| {
        Some((now, j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()))
    };
    if !full {
        let j = journal_shared();
        if j.ahead.iter().any(|&(a, b)| a <= first && last <= b) {
            return from_journal(&j, j.ahead_asked);
        }
    }
    let _held = stripes(first, last);
    FOUND.with_borrow_mut(|found| {
        let mut count = (last - first) * PAGE / KERNEL_PAGE;
        found.resize(count.max(found.len()), 0);
        let mut granularity = 0;
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let mut _z = crate::trace::zone("GetWriteWatch: one range");
        _z.arg(((last - first) * PAGE) as u64);
        let view = guarded && known_view(first);
        // SAFETY: the answer buffer holds `count` entries.
        let failed = view || unsafe {
            GetWriteWatch(
                WRITE_WATCH_FLAG_RESET,
                (first * PAGE) as *const _,
                (last - first) * PAGE,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        } != 0;
        let guarded = if failed {
            count = 0;
            if !guarded {
                return None;
            }
            #[cfg(feature = "diagnostics")]
            let guarding = std::time::Instant::now();
            let written = guard(first * PAGE, last * PAGE);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
            written?
        } else {
            Vec::new()
        };
        drop(_z);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::Watches, 1);
            add(C::WatchUs, started.elapsed().as_micros() as u64);
            slow(started.elapsed().as_micros() as u64);
        }
        let mut j = journal();
        let j = &mut *j;
        j.asked += 1;
        let now = j.asked;
        for page in found[..count].iter().map(|&at| at / PAGE) {
            j.written.insert(page, now);
        }
        for &at in &found[..count] {
            note_piece(j, at, now);
        }
        // The guards' pages and an answer that overflowed: no 4 KiB pieces known.
        for page in guarded.into_iter().chain(if full { first..last } else { 0..0 }) {
            j.written.insert(page, now);
            note_whole(j, page, now);
        }
        from_journal(j, now)
    })
}

/// [`writes`] holding the journal across the kernel call, as before
/// [`unlocked_ask`].
fn writes_locked(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let full = full_refresh();
    if shared_answers() && !full {
        let j = journal_shared();
        if j.ahead.iter().any(|&(a, b)| a <= first && last <= b) {
            return Some((j.ahead_asked, j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()));
        }
    }
    let mut j = journal();
    let j = &mut *j;
    // Asked ahead with the submission's other ranges: the journal has it.
    if !full && j.ahead.iter().any(|&(a, b)| a <= first && last <= b) {
        return Some((
            j.ahead_asked,
            j.written
                .range(first..last)
                .map(|(&p, &w)| (p, w))
                .collect(),
        ));
    }
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let mut _z = crate::trace::zone("GetWriteWatch: one range");
    _z.arg(((last - first) * PAGE) as u64);
    let view = guarded && known_view(first);
    // SAFETY: the answer buffer holds `count` entries.
    let failed = view || unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    let guarded = if failed {
        count = 0;
        if !guarded {
            return None;
        }
        #[cfg(feature = "diagnostics")]
        let guarding = std::time::Instant::now();
        let written = guard(first * PAGE, last * PAGE);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
        written?
    } else {
        Vec::new()
    };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    j.asked += 1;
    let now = j.asked;
    let pages = j.found[..count].iter().map(|&at| at / PAGE);
    for page in pages.collect::<Vec<_>>() {
        j.written.insert(page, now);
    }
    for k in 0..count {
        let at = j.found[k];
        note_piece(j, at, now);
    }
    // The guards' pages and an answer that overflowed: no 4 KiB pieces known.
    for page in guarded.into_iter().chain(if full { first..last } else { 0..0 }) {
        j.written.insert(page, now);
        note_whole(j, page, now);
    }
    Some((
        now,
        j.written
            .range(first..last)
            .map(|(&p, &w)| (p, w))
            .collect(),
    ))
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

struct Guards {
    /// Blocks we made read-only.
    protected: BTreeSet<usize>,
    /// Of those, the ones not written since a copy took them.
    clean: BTreeSet<usize>,
    /// Blocks already copied as zeros. While they remain wholly uncommitted,
    /// the copy can skip hundreds of MB of failing kernel reads per world frame.
    absent: BTreeSet<usize>,
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BTreeSet::new(),
    clean: BTreeSet::new(),
    absent: BTreeSet::new(),
});

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    crate::trace::lock("lock wait: guards", &GUARDS)
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    let mut _z = crate::trace::zone("guard: view memory check");
    _z.arg((end - start) as u64);
    // A range whose first block the guards hold already (read-only by our
    // hand, or noted absent) is a view: no question to Windows. The
    // question walks the page tables to the end of the pages that share the
    // first one's state, and a guarded view is one long run of read-only
    // pages: 0.6 ms for the title's 174 MB heap, 16 times a frame, and most
    // of the 68 ms a frame of "guards" (run 58).
    // `VICTORY_GUARD_ASK=1` asks every time, as before.
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let known = !*ASK.get_or_init(|| std::env::var_os("VICTORY_GUARD_ASK").is_some_and(|v| v == "1")) && {
        let g = guards();
        let first = start / BLOCK;
        g.protected.contains(&first) || g.absent.contains(&first)
    };
    if !known && !is_view(start, end) {
        return None;
    }
    install();
    let mut g = guards();
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // The range's clean blocks, in order: read off the set in one pass,
    // not looked up one block at a time. A frame asks about some 10 GB of
    // view memory, 150,000 blocks, nearly all clean: the lookups were most
    // of the 60-90 ms a frame of "guards" (run 54). What the loop adds to
    // the set meanwhile is the block it stands at or ones before it.
    let clean: Vec<usize> = g.clean.range(first..last).copied().collect();
    if clean.len() == last - first {
        // Nothing written, mapped or new since the range was last taken.
        return Some(written);
    }
    let mut next_clean = 0;
    let mut uncommitted_until = first;
    for block in first..=last {
        while next_clean < clean.len() && clean[next_clean] < block {
            next_clean += 1;
        }
        let is_clean = next_clean < clean.len() && clean[next_clean] == block;
        let mut open = block < last && !is_clean;
        if open && g.absent.contains(&block) {
            // Ordinary commits need no runtime mapping notification. Reuse
            // this asking's query for the following blocks in the same region.
            if block >= uncommitted_until {
                uncommitted_until = uncommitted_end(block);
            }
            open = block >= uncommitted_until;
            if open {
                g.absent.remove(&block);
            }
        }
        if open {
            written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            if g.protected.contains(&block) {
                // Written through the runtime or at another address,
                // still read-only.
                g.clean.insert(block);
                guard_twins(&mut g, block, block + 1);
            } else {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() {
            protect(&mut g, from, block);
        }
    }
    Some(written)
}

/// Whether `[start, end)` is view memory. The title's views lie in
/// placeholders, mapped and unmapped block by block as it streams, so a
/// range can begin in a block that is not mapped right now: the first
/// pages of the range that are there decide. A range with nothing there
/// counts as a view (its blocks are noted absent, and looked at again when
/// the runtime maps them).
///
/// Looking at the first address only took such a range for "not a view":
/// its copy was marked not watched for good and copied whole at every
/// refresh, around its unmapped pages (run 54: 300 such ranges, 570 MB
/// copied a frame for them, 255 copies a frame reading around holes).
/// `VICTORY_GUARD_HOLES=0` looks at the first address only, as before.
fn is_view(start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    static HOLES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let holes = *HOLES.get_or_init(|| std::env::var_os("VICTORY_GUARD_HOLES").is_none_or(|v| v != "0"));
    let mut at = start;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return false;
        }
        if info.Type == MEM_MAPPED {
            return true;
        }
        // Committed memory that is no view (the title's heaps, ours), or
        // the answer as it was before.
        if info.State == MEM_COMMIT || !holes {
            return false;
        }
        // A hole (a placeholder, or free): on to what follows it.
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= at || next >= end {
            return true;
        }
        at = next;
    }
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail, and
/// noting those not committed (unmapped) absent.
fn protect(g: &mut Guards, from: usize, to: usize) {
    let mut _z = crate::trace::zone("VirtualProtect: guard (read-only)");
    _z.arg((to - from) as u64);
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        }
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.extend(from..to);
        g.clean.extend(from..to);
    };
    if set(from, to) {
        note(g, from, to);
        guard_twins(g, from, to);
        return;
    }
    for block in from..to {
        if set(block, block + 1) {
            note(g, block, block + 1);
            guard_twins(g, block, block + 1);
        } else if block < uncommitted_end(block) {
            g.absent.insert(block);
        }
    }
}

/// Whether the other addresses of a block's bytes are guarded with it
/// (`VICTORY_VIEW_TWINS=0`: no, as before).
fn view_twins() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_TWINS").is_none_or(|v| v != "0"))
}

/// Twin blocks made read-only, and twin blocks made not clean by a write
/// at another address; since the start.
pub(crate) static TWINS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Blocks `[from, to)` were just noted clean: every other address that
/// shows their bytes is made read-only too, so that a write through it
/// faults ([`stale_twins`]). Such a block is not noted clean: the copies
/// taken through it have not taken the bytes, and a block that is
/// read-only and not clean is reported written at their next check.
fn guard_twins(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    if !view_twins() {
        return;
    }
    crate::inplace::twins(from * BLOCK, to * BLOCK, |_, twin| {
        let twin = twin / BLOCK;
        if g.protected.contains(&twin) {
            return;
        }
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        if unsafe { VirtualProtect((twin * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) } != 0 {
            g.protected.insert(twin);
            let n = TWINS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n == 1 {
                tracing::info!("a block of title memory is shown at another address too: guarded there as well (VICTORY_VIEW_TWINS=0: not)");
            }
        }
    });
}

/// Blocks `[from, to)` are written, or can be without a fault: the blocks
/// at the other addresses of their bytes are no longer clean. They stay
/// read-only, so no call to the system is made for them.
fn stale_twins(g: &mut Guards, from: usize, to: usize) {
    let on = view_twins();
    crate::inplace::twins(from * BLOCK, to * BLOCK, |_, twin| {
        // Switched off, the write that would have been missed is still
        // counted, so that a run without the guard says how many were.
        if if on { g.clean.remove(&(twin / BLOCK)) } else { g.clean.contains(&(twin / BLOCK)) } {
            TWINS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    });
}

/// The runtime mapped a view at `[at, at + len)` (`crate::inplace`): it is
/// writable and not guarded, so what a copy took through another address
/// of the same bytes can change with no fault. Those blocks are no longer
/// clean; their next check guards this view with them.
pub(crate) fn view_mapped(at: usize, len: usize) {
    stale_twins(&mut guards(), at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
}

/// The end (exclusive) of the whole uncommitted blocks starting at `block`.
fn uncommitted_end(block: usize) -> usize {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    if n != 0 && matches!(info.State, MEM_RESERVE | MEM_FREE) {
        (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK
    } else {
        block
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded.
fn open(g: &mut Guards, start: usize, end: usize, why: Seen) -> bool {
    let blocks = open_only(g, start, end, why);
    // The same bytes at their other addresses are written with them.
    let mut i = 0;
    while i < blocks.len() {
        let mut j = i + 1;
        while j < blocks.len() && blocks[j] == blocks[j - 1] + 1 {
            j += 1;
        }
        stale_twins(g, blocks[i], blocks[j - 1] + 1);
        i = j;
    }
    !blocks.is_empty()
}

/// [`open`] without a word to the other addresses of the blocks' bytes;
/// the blocks it opened.
fn open_only(g: &mut Guards, start: usize, end: usize, why: Seen) -> Vec<usize> {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    // Kept blocks get the card's bytes before anything writes them.
    resident_write(start / BLOCK, end.div_ceil(BLOCK), 4);
    let blocks: Vec<usize> = g
        .protected
        .range(start / BLOCK..end.div_ceil(BLOCK))
        .copied()
        .collect();
    for &block in &blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
    }
    let open = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: blocks we made read-only.
        unsafe { VirtualProtect((from * BLOCK) as *const _, (to - from) * BLOCK, PAGE_READWRITE, &mut old) != 0 }
    };
    // Blocks next to each other in one call (run 16: 4,900 waits a frame for
    // the guards, their longest 20-30 ms, behind calls made block by
    // block); block by block where one call fails (the run spans views).
    let mut i = 0;
    while i < blocks.len() {
        let mut j = i + 1;
        while j < blocks.len() && blocks[j] == blocks[j - 1] + 1 {
            j += 1;
        }
        if j - i == 1 || !open(blocks[i], blocks[j - 1] + 1) {
            for &block in &blocks[i..j] {
                open(block, block + 1);
            }
        }
        i = j;
    }
    census_forget(&blocks, why);
    blocks
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    // The length is the title's (`fread`'s size times count, saturated).
    open(&mut guards(), at, at.saturating_add(len), Seen::KernelWrite);
    // A placed texture there changes with no descriptor written.
    crate::descriptor::host::remapped(at, len);
}

/// Our own write into title memory (a copy-back of what shaders wrote),
/// begun with [`own_write_begin`]: the guarded blocks it opened.
pub(crate) struct OwnWrite {
    blocks: Vec<usize>,
    at: usize,
    len: usize,
}

/// [`writing`] for a write of ours that the host copy doing it already
/// holds: opens the guarded blocks of `[at, at + len)` as [`writing`] does,
/// and remembers which were guarded, so that [`own_write_end`] can guard
/// them again at once instead of leaving them to show as title writes.
///
/// Before this, every block a copy-back wrote showed as written at the
/// copy's next refresh, and the refresh copied all 64 KiB of it back into
/// the copy (and, for a copy in the card's memory, sent it up again):
/// bytes the copy held already. In run 23's prison window that was some
/// 1,000 blocks a frame, about 70 of the 160 MB the frame line calls
/// "copies".
pub(crate) fn own_write_begin(at: usize, len: usize) -> OwnWrite {
    let _z = crate::trace::zone("copy-back: open guarded blocks");
    let end = at.saturating_add(len);
    let blocks: Vec<usize> = {
        // Not `open`: the other addresses of these bytes are told at the
        // end, in the journal, with no fault and no call to the system.
        open_only(&mut guards(), at, end, Seen::CardAgain)
    };
    crate::descriptor::host::remapped(at, len);
    OwnWrite { blocks, at, len }
}

/// After the write [`own_write_begin`] began: the blocks it opened are
/// guarded again and their pages noted written under a new asking (so the
/// other host copies of the same memory take the new bytes); for memory
/// that is write-watched, its pages are asked about, which clears the
/// kernel's bits our write set. Returns that asking's number, or `None`
/// if the memory is neither (then nothing changed, and the write shows at
/// the next asking as before).
///
/// A title write between the two calls is not lost: its block is among
/// those noted written, and the caller marks its copy current for a page
/// only where the copy's bytes equal title memory after this call
/// ([`crate::shadow::Shadow`]'s settling). A title write after this call
/// faults into the guard, or sets the kernel's bit, as any other.
pub(crate) fn own_write_end(w: OwnWrite) -> Option<u64> {
    let _z = crate::trace::zone("copy-back: guard again, note written");
    if w.len == 0 {
        return None;
    }
    if w.blocks.is_empty() {
        if census() {
            seen(Seen::Unwatched, (w.at.saturating_add(w.len).div_ceil(BLOCK) - w.at / BLOCK) as u64);
        }
        return writes(w.at, w.at.saturating_add(w.len), false).map(|(now, _)| now);
    }
    // The same bytes at their other addresses: written too, for the
    // copies taken there. Said in the journal (they stay read-only and
    // clean, so a title write through them still faults).
    let mut twins: Vec<usize> = Vec::new();
    {
        let mut g = guards();
        let mut i = 0;
        while i < w.blocks.len() {
            let mut j = i + 1;
            while j < w.blocks.len() && w.blocks[j] == w.blocks[j - 1] + 1 {
                j += 1;
            }
            // Only blocks no one guarded again or remapped meanwhile.
            let from = w.blocks[i];
            let to = w.blocks[j - 1] + 1;
            if !g.protected.contains(&from) {
                protect(&mut g, from, to);
            }
            if view_twins() {
                crate::inplace::twins(from * BLOCK, to * BLOCK, |_, twin| twins.push(twin / BLOCK));
            }
            i = j;
        }
        if census() {
            census_mark(&g, &w.blocks, w.at, w.len);
        }
    }
    let mut j = journal();
    j.asked += 1;
    let now = j.asked;
    for &block in w.blocks.iter().chain(&twins) {
        for page in block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE {
            j.written.insert(page, now);
            note_whole(&mut j, page, now);
        }
    }
    Some(now)
}

/// Held shared by a read of title memory in place, and alone by the runtime
/// while it changes what title memory shows.
///
/// The runtime does not change a mapping in one step. To unmap one page of
/// a view it unmaps the whole view and maps what stays again
/// (`xmem.rs` `unmap`), and a view it maps is a reservation until the next
/// call commits it (`map`, then `commit`). For that moment pages the title
/// never gave up are "reserved, never committed", and a thread reading
/// them in place takes an access violation: run 71 in `Layout::detile`,
/// run 74 in the hash before it, both under `Staging::refresh_now`, which
/// had asked whether the texture was all mapped and been told yes. Asking
/// cannot help; not changing the mapping while such a read is under way
/// does.
static MAPPING: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// What the runtime holds while it changes what title memory shows
/// ([`title_memory_remapping`]).
pub struct Remapping(#[allow(dead_code)] std::sync::RwLockWriteGuard<'static, ()>);

/// The runtime is about to map, unmap or move physical pages of title
/// memory: until what it returns is dropped, no read of title memory in
/// place is under way and none begins ([`title_memory_reading`]). Taken
/// around the whole change, the unmapping and the mapping again.
pub fn title_memory_remapping() -> Remapping {
    Remapping(MAPPING.write().unwrap_or_else(|e| e.into_inner()))
}

/// What a read of title memory in place holds ([`title_memory_reading`]).
pub struct Reading(#[allow(dead_code)] std::sync::RwLockReadGuard<'static, ()>);

/// For reading title memory in place: until what it returns is dropped,
/// the runtime changes no mapping, so memory found mapped stays mapped.
/// Not to be taken twice by one thread, nor held across anything that
/// maps title memory.
pub fn title_memory_reading() -> Reading {
    Reading(MAPPING.read().unwrap_or_else(|e| e.into_inner()))
}

/// [`title_memory_reading`] if it can be had at once: `None` while the
/// runtime changes a mapping (or waits to), for a caller that has another
/// way to read and would rather not wait.
pub fn title_memory_reading_now() -> Option<Reading> {
    match MAPPING.try_read() {
        Ok(guard) => Some(Reading(guard)),
        Err(std::sync::TryLockError::Poisoned(e)) => Some(Reading(e.into_inner())),
        Err(std::sync::TryLockError::WouldBlock) => None,
    }
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded (nor known
/// unmapped).
pub fn title_memory_remapped(at: usize, len: usize) {
    // A placed texture there arrives, goes or changes with no descriptor
    // written.
    crate::descriptor::host::remapped(at, len);
    // The views the runtime told us of there are no longer what it shows.
    crate::inplace::views_gone(at, len);
    // And nothing stays in place over it.
    crate::inplace::title_memory_freeing(at, len);
    let mut g = guards();
    let range = at / BLOCK..at.saturating_add(len).div_ceil(BLOCK);
    // Kept blocks: written in while they still show what was kept (before
    // an unmapping), dropped if they show something new already.
    resident_write(range.start, range.end, 5);
    let absent: Vec<usize> = g.absent.range(range.clone()).copied().collect();
    for block in absent {
        g.absent.remove(&block);
    }
    let blocks: Vec<usize> = g.protected.range(range).copied().collect();
    for &block in &blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
    }
    census_forget(&blocks, Seen::Lost);
}

/// Whether what shaders wrote into guarded view memory stays the card's
/// until something touches it (`VICTORY_CARD_RESIDENT=1`; off by default;
/// it turns the read census off, which uses the same protection).
///
/// A landing's copy-back read each piece of title memory the card wrote,
/// merged the card's bytes in and wrote it back, and only then told the
/// title the card was done: 150 ms a frame in run 5's prison profile, for
/// memory the read census found the title's CPU code never reads (0 title
/// reads of copied-back blocks). With the switch the merged bytes are kept
/// here, block by block ([`Resident`]), the copy that sent them is noted
/// current, and the block is left with no access. The first touch of it,
/// whoever's, writes the card's bytes in before it goes on:
///
/// - a fault of the title's code or ours, read or write ([`install`]);
/// - our kernel reads of title memory (refreshes, `ReadProcessMemory`:
///   [`resident_open`], as [`census_open`]);
/// - opening a guarded block for a write, the title's (a fault), the
///   kernel's on the title's behalf ([`writing`]) or a copy-back's
///   ([`own_write_begin`]): [`open_only`];
/// - the runtime unmapping or remapping it ([`title_memory_remapped`]).
///
/// Writes the runtime made through its own view of the same physical pages
/// (DirectStorage decoding in place) are newer than the card's bytes: those
/// are dropped ([`kernel_wrote`]). A block shown at another title address
/// too (a view twin) is never kept: a read there would not fault.
/// The bytes a block gets are exactly the ones the copy-back would have
/// written, at the first moment anything could see them.
pub(crate) fn resident() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_CARD_RESIDENT").is_some_and(|v| v == "1");
        if on {
            tracing::info!(
                "VICTORY_CARD_RESIDENT=1: what shaders wrote into view memory is written into title memory at its first touch, not at landing (cap {} MB)",
                resident_cap() >> 20
            );
        }
        on
    })
}

/// The most bytes of blocks kept ([`resident`]), `VICTORY_CARD_RESIDENT_MB`
/// (512 by default); past it a landing writes as before.
fn resident_cap() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("VICTORY_CARD_RESIDENT_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(512)
            << 20
    })
}

/// A block kept for [`resident`]: what title memory is to hold (its bytes
/// when kept, with every merge since), and which of them the card wrote.
struct Resident {
    image: Box<[u8]>,
    card: Box<[bool]>,
    /// The block's bytes when it was kept: at write-in, a byte of title
    /// memory that differs from it was written by someone who never
    /// faulted (a reader or writer we do not see), and is newer than the
    /// card's: it is left as it is, and the block is reported.
    was: Box<[u8]>,
    /// The section that holds the block's physical pages, and the block's
    /// offset in it ([`write_kept`] writes through it).
    section: (usize, u64),
}

static RESIDENT: Mutex<BTreeMap<usize, Resident>> = Mutex::new(BTreeMap::new());
/// Blocks in [`RESIDENT`], for the fault handler's look without a lock.
static RESIDENT_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Since the start: blocks kept; merges into blocks already kept; blocks
/// written in by a fault, by our kernel reads, by an opening for a write,
/// by a remapping; blocks dropped (newer bytes, or the protection changed
/// behind us); landings written as before (not view memory, twins, the
/// cap, the copy not current); bytes merged without a write; blocks found
/// changed behind us at write-in, and their bytes changed.
pub(crate) static RESIDENT_STATS: [std::sync::atomic::AtomicU64; 12] = [const { std::sync::atomic::AtomicU64::new(0) }; 12];

fn resident_stat(i: usize, n: u64) {
    RESIDENT_STATS[i].fetch_add(n, std::sync::atomic::Ordering::Relaxed);
}

fn resident_lock() -> std::sync::MutexGuard<'static, BTreeMap<usize, Resident>> {
    RESIDENT.lock().unwrap_or_else(|e| e.into_inner())
}

/// What [`resident_merge`] did.
pub(crate) enum Kept {
    /// The checks failed before anything changed: write as before.
    No,
    /// The merge changed nothing in title memory.
    Same,
    /// Kept; the pages are noted written under this asking (for the other
    /// copies of the memory), which the caller's copy is current for.
    At(u64),
}

/// The pages of `[first, last)` the journal notes written, with the asking
/// that last did.
pub(crate) fn journal_written(first: usize, last: usize) -> Vec<(usize, u64)> {
    let j = journal_shared();
    j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()
}

/// [`resident`]: title memory `[at, at + n)` is to get a copy-back's
/// merge. If every block of it is guarded view memory with no twin, and
/// room is left: `merge` gets what title memory holds there (the kept
/// bytes where a block is kept already) and merges the card's bytes into
/// it (true if it changed any); the result is kept, the new blocks left
/// with no access, and the pages noted written. Lock order: the caller's
/// copy, then the guards, then the kept blocks.
pub(crate) fn resident_merge(at: usize, n: usize, merge: impl FnOnce(&mut [u8]) -> bool) -> Kept {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, VirtualProtect};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let _z = crate::trace::zone("card resident: keep a landing's merge");
    if n == 0 {
        return Kept::Same;
    }
    let (first, last) = (at / BLOCK, (at + n).div_ceil(BLOCK));
    let g = guards();
    if (first..last).any(|b| !g.protected.contains(&b)) {
        resident_stat(7, 1);
        return Kept::No;
    }
    // Each block one run of a section, so that it can be written in with
    // the title's address kept closed.
    let mut sections = Vec::with_capacity(last - first);
    for b in first..last {
        match crate::inplace::cover(b * BLOCK, BLOCK) {
            crate::inplace::Cover::Run(section, offset) if offset.is_multiple_of(BLOCK as u64) => sections.push((section, offset)),
            _ => {
                resident_stat(7, 1);
                return Kept::No;
            }
        }
    }
    if crate::inplace::shown_elsewhere(first * BLOCK, last * BLOCK) {
        resident_stat(7, 1);
        return Kept::No;
    }
    let mut r = resident_lock();
    let new = (first..last).filter(|b| !r.contains_key(b)).count();
    if (r.len() + new) * BLOCK > resident_cap() {
        resident_stat(7, 1);
        return Kept::No;
    }
    // What title memory holds there, block by block.
    let mut blocks: Vec<(usize, Box<[u8]>, bool)> = Vec::with_capacity(last - first);
    for b in first..last {
        if let Some(k) = r.get(&b) {
            blocks.push((b, k.image.clone(), false));
            continue;
        }
        let mut image = vec![0u8; BLOCK].into_boxed_slice();
        let mut done = 0;
        // SAFETY: a local buffer of BLOCK bytes; the kernel checks the
        // source (read-only by our hand, so readable if committed). Not
        // `host::read`: its retry opens kept blocks, under the guards.
        if unsafe { ReadProcessMemory(GetCurrentProcess(), (b * BLOCK) as *const _, image.as_mut_ptr().cast(), BLOCK, &mut done) } == 0 {
            resident_stat(7, 1);
            return Kept::No;
        }
        blocks.push((b, image, true));
    }
    let mut title = vec![0u8; n];
    for (b, image, _) in &blocks {
        let (lo, hi) = ((b * BLOCK).max(at), ((b + 1) * BLOCK).min(at + n));
        title[lo - at..hi - at].copy_from_slice(&image[lo - b * BLOCK..hi - b * BLOCK]);
    }
    if !merge(&mut title) {
        return Kept::Same;
    }
    for (b, image, fresh) in blocks {
        let (lo, hi) = ((b * BLOCK).max(at), ((b + 1) * BLOCK).min(at + n));
        let mut kept = match r.remove(&b) {
            Some(k) => k,
            None => Resident {
                was: image.clone(),
                image,
                card: vec![false; BLOCK].into_boxed_slice(),
                section: sections[b - first],
            },
        };
        for x in lo..hi {
            let (i, t) = (x - b * BLOCK, title[x - at]);
            if kept.image[i] != t {
                kept.image[i] = t;
                kept.card[i] = true;
            }
        }
        if fresh {
            let mut old = 0;
            // SAFETY: a guarded block of title view memory: read-only by
            // our hand; only its protection changes.
            if unsafe { VirtualProtect((b * BLOCK) as *const _, BLOCK, PAGE_NOACCESS, &mut old) } == 0 {
                // Not kept after all: written now, as before.
                write_kept(b, &kept);
                continue;
            }
            resident_stat(0, 1);
        } else {
            resident_stat(1, 1);
        }
        r.insert(b, kept);
    }
    RESIDENT_N.store(r.len(), std::sync::atomic::Ordering::Release);
    resident_stat(8, n as u64);
    drop(r);
    drop(g);
    let mut j = journal();
    j.asked += 1;
    let now = j.asked;
    for page in at / PAGE..(at + n).div_ceil(PAGE) {
        j.written.insert(page, now);
        note_whole(&mut j, page, now);
    }
    Kept::At(now)
}

/// Writes the card's bytes of kept block `b` into title memory and leaves
/// it read-only (guarded). The block has no access (it is kept).
///
/// Through a view of our own of the block's physical pages (its section,
/// [`Resident::section`]): the title's address stays with no access until
/// every byte is in. Run 7 made the title's address writable, then copied:
/// a title thread that read the block in that moment, without a fault,
/// took the old bytes (a pointer the card had not written yet) and
/// crashed on it.
fn write_kept(b: usize, kept: &Resident) {
    use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile, PAGE_READONLY, UnmapViewOfFile, VirtualProtect};
    let (section, offset) = kept.section;
    // SAFETY: the runtime's section of the block's physical pages, which
    // outlives the title's views of it; a view of BLOCK bytes from an
    // offset on the allocation granularity (64 KiB), unmapped below.
    let view = unsafe { MapViewOfFile(section as _, FILE_MAP_WRITE, (offset >> 32) as u32, offset as u32, BLOCK) };
    let base = view.Value.cast::<u8>();
    if base.is_null() {
        resident_stat(6, 1);
        return;
    }
    // SAFETY: our view of BLOCK bytes; written only where the card wrote
    // and nobody else did since the block was kept.
    unsafe {
        let now = std::slice::from_raw_parts_mut(base, BLOCK);
        let (mut foreign, mut kept_out, mut first) = (0usize, 0usize, None);
        for i in 0..BLOCK {
            if now[i] != kept.was[i] {
                foreign += 1;
                first.get_or_insert(i);
                if kept.card[i] {
                    kept_out += 1;
                }
            } else if kept.card[i] {
                now[i] = kept.image[i];
            }
        }
        UnmapViewOfFile(view);
        if foreign > 0 {
            resident_stat(9, 1);
            resident_stat(10, foreign as u64);
            let n = RESIDENT_STATS[9].load(std::sync::atomic::Ordering::Relaxed);
            if n <= 64 {
                tracing::warn!(
                    block = format_args!("{:#x}", b * BLOCK),
                    first = format_args!("{:#x}", b * BLOCK + first.unwrap_or(0)),
                    foreign,
                    card_bytes_left = kept_out,
                    card_bytes = kept.card.iter().filter(|&&c| c).count(),
                    section = format_args!("{:#x}+{:#x}", section, offset),
                    twins_now = {
                        let mut t = 0;
                        crate::inplace::twins(b * BLOCK, (b + 1) * BLOCK, |_, _| t += 1);
                        t
                    },
                    "card resident: a kept block was changed while kept by someone who never faulted; their bytes are kept, the card's left out where they overlap"
                );
            }
        }
    }
    let mut old = 0;
    // SAFETY: a kept block of title view memory, which we left with no
    // access; back to read-only (guarded), its bytes all in.
    unsafe { VirtualProtect((b * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) };
}

/// Whether the address `at` can be read now (committed, with some
/// access, no guard page).
fn readable_at(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect & 0xff != PAGE_NOACCESS && info.Protect & PAGE_GUARD == 0 && info.Protect != 0
}

/// Whether block `b` can be read now (committed, with some access).
#[allow(dead_code)]
fn readable(b: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((b * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect != PAGE_NOACCESS
}

/// Whether block `b` still has no access, as we left it when kept: if not,
/// what it shows changed behind us (remapped) and the kept bytes are old.
fn still_kept(b: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((b * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_NOACCESS
}

/// Kept blocks of `[from, to)` (block numbers) are written in, for `why`
/// (a stat: 2 fault, 3 our read, 4 opened for a write, 5 remapped), and
/// read-only again; the guards are held by the caller. The blocks written.
fn resident_write(from: usize, to: usize, why: usize) -> usize {
    use std::sync::atomic::Ordering;
    if RESIDENT_N.load(Ordering::Acquire) == 0 || from >= to {
        return 0;
    }
    // No timeline zone: this runs in the fault handler too.
    let mut r = resident_lock();
    let blocks: Vec<usize> = r.range(from..to).map(|(&b, _)| b).collect();
    let mut written = 0;
    for b in blocks {
        let Some(kept) = r.remove(&b) else { continue };
        if still_kept(b) {
            write_kept(b, &kept);
            written += 1;
            resident_stat(why, 1);
        } else {
            resident_stat(6, 1);
        }
    }
    RESIDENT_N.store(r.len(), Ordering::Release);
    written
}

/// [`resident`]: the runtime now shows bytes `[offset, offset + len)` of
/// `section` at another address too (a new view). A kept block of those
/// bytes would have a twin the title can write through without a fault,
/// and its kept bytes would then be written in over the title's newer
/// ones at the first touch of the old address (runs 7 and 8, at the
/// diner's teardown, as the title moved its pages): written in now,
/// before the runtime returns the new view to the title.
pub(crate) fn resident_shown(section: usize, offset: u64, len: usize) {
    use std::sync::atomic::Ordering;
    if !resident() || RESIDENT_N.load(Ordering::Acquire) == 0 {
        return;
    }
    let _g = guards();
    let mut r = resident_lock();
    let end = offset.saturating_add(len as u64);
    let blocks: Vec<usize> = r.iter().filter(|(_, k)| k.section.0 == section && k.section.1 < end && k.section.1 + BLOCK as u64 > offset).map(|(&b, _)| b).collect();
    for b in blocks {
        let Some(kept) = r.remove(&b) else { continue };
        if still_kept(b) {
            write_kept(b, &kept);
            resident_stat(5, 1);
        } else {
            resident_stat(6, 1);
        }
    }
    RESIDENT_N.store(r.len(), Ordering::Release);
}

/// [`resident`]: whether any block of `[at, at + len)` is kept (no lock
/// on the guards).
pub(crate) fn resident_holds(at: usize, len: usize) -> bool {
    resident()
        && RESIDENT_N.load(std::sync::atomic::Ordering::Acquire) > 0
        && len > 0
        && resident_lock().range(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK)).next().is_some()
}

/// [`resident`]: our read of `[at, at + len)` into `dst` failed; if kept
/// blocks are why, the read is made with their kept bytes (what title
/// memory is to hold), the rest from title memory, and the blocks stay
/// kept: a refresh of another copy of the memory no longer writes them in
/// (run 12: 1,500 write-ins a frame, 104 ms, and the guards' lock waited
/// 556 ms a frame behind them). False if a part not kept cannot be read.
pub(crate) fn resident_read(at: usize, dst: *mut u8, len: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    if !resident() || RESIDENT_N.load(std::sync::atomic::Ordering::Acquire) == 0 || len == 0 {
        return false;
    }
    let _z = crate::trace::zone("card resident: our read takes kept bytes");
    let r = resident_lock();
    let end = at + len;
    let mut kept_any = false;
    let mut x = at;
    while x < end {
        let b = x / BLOCK;
        let to = ((b + 1) * BLOCK).min(end);
        // SAFETY: inside the caller's destination of `len` bytes.
        let out = unsafe { dst.add(x - at) };
        if let Some(k) = r.get(&b) {
            // SAFETY: as above; the image holds the whole block.
            unsafe { std::ptr::copy_nonoverlapping(k.image.as_ptr().add(x - b * BLOCK), out, to - x) };
            kept_any = true;
        } else {
            let mut done = 0;
            // SAFETY: the destination is the caller's; the kernel checks
            // the source.
            if unsafe { ReadProcessMemory(GetCurrentProcess(), x as *const _, out.cast(), to - x, &mut done) } == 0 {
                return false;
            }
        }
        x = to;
    }
    if kept_any {
        resident_stat(11, 1);
    }
    kept_any
}

/// [`resident`]: our kernel read of `[at, at + len)` failed; kept blocks
/// there are written in (true if any was: the read is worth trying again).
pub(crate) fn resident_open(at: usize, len: usize) -> bool {
    // Asked of every texture's memory before it is read: the guards' lock
    // only when a kept block is there (run 13: 2,250 asks a frame took it,
    // and the guards' lock waited 425 ms a frame behind them).
    if !resident_holds(at, len) {
        return false;
    }
    let _z = crate::trace::zone("card resident: our read writes kept blocks in");
    let _g = guards();
    resident_write(at / BLOCK, at.saturating_add(len).div_ceil(BLOCK), 3) > 0
}

/// The runtime wrote `[at, at + len)` through a view of its own of the
/// same physical pages (DirectStorage decoding in place): the bytes are
/// newer than any the card left kept there, which are dropped; then as
/// [`writing`].
pub fn kernel_wrote(at: usize, len: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    if resident() && RESIDENT_N.load(std::sync::atomic::Ordering::Acquire) > 0 {
        let _g = guards();
        let mut r = resident_lock();
        let blocks: Vec<usize> = r.range(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK)).map(|(&b, _)| b).collect();
        for b in blocks {
            r.remove(&b);
            let mut old = 0;
            // SAFETY: a block we left with no access; back to read-only
            // (guarded), which `writing` then opens.
            unsafe { VirtualProtect((b * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) };
            resident_stat(6, 1);
        }
        RESIDENT_N.store(r.len(), std::sync::atomic::Ordering::Release);
    }
    writing(at, len);
}

/// For the memory line: what [`resident`] did since the line before.
pub(crate) fn resident_line() {
    use std::sync::atomic::Ordering::Relaxed;
    if !resident() {
        return;
    }
    static LAST: Mutex<[u64; 12]> = Mutex::new([0; 12]);
    let now: [u64; 12] = std::array::from_fn(|i| RESIDENT_STATS[i].load(Relaxed));
    let d: [u64; 12] = {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        let d = std::array::from_fn(|i| now[i] - last[i]);
        *last = now;
        d
    };
    let held = RESIDENT_N.load(Relaxed);
    tracing::info!(
        "card resident (VICTORY_CARD_RESIDENT=1): {} blocks kept, {} merged again, {:.1} MB merged with no write; written in at first touch: {} by a fault, {} by our reads, {} opened for a write, {} remapped; {} dropped (newer bytes); {} landings written as before; {} blocks ({} MB) kept now; {} found changed behind us at write-in ({} bytes, theirs kept); {} of our reads took kept bytes (no write-in)",
        d[0],
        d[1],
        d[8] as f64 / 1e6,
        d[2],
        d[3],
        d[4],
        d[5],
        d[6],
        d[7],
        held,
        (held * BLOCK) >> 20,
        d[9],
        d[10],
        d[11]
    );
}

/// The read census (`VICTORY_READ_CENSUS=1`, diagnostics only, off by
/// default): of the title memory our copy-backs write (what shaders made,
/// put where the title's CPU code would see it), how much the title ever
/// reads before the card writes it again. Each guarded block a copy-back
/// writes is left with no access at all (it was read-only for the guards),
/// and the first touch of it faults into our handler, which counts it, says
/// whose code it was (the title's or ours, by the module of the faulting
/// instruction) and how many frames after the copy-back, and puts the
/// block back to read-only: what the block holds is the same either way,
/// the bytes were copied back already. A copy-back into a block still
/// waiting counts it as written again by the card with no read between.
///
/// What it is for: copying back only what the title reads (the copy-back's
/// wait on the card was the biggest cost of the frame). The bytes nobody
/// reads could stay the card's, as on the console, where the CPU and the
/// card share one memory and nothing is copied at all.
///
/// Our own kernel reads of such a block (refreshes, `ReadProcessMemory`)
/// open it and try again ([`census_open`]); the title's kernel I/O into it
/// opens it first ([`writing`]). A kernel read of the title's (a file
/// write from GPU memory) would fail while a block waits: none is known.
/// Only blocks the guards hold are counted (the view memory); a copy-back
/// elsewhere is only counted as not watched. The other addresses of the
/// same bytes stay as they were, so a read through one of them is missed.
fn census() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let asked = std::env::var_os("VICTORY_READ_CENSUS").is_some_and(|v| v == "1") && crate::inplace::level() == 0;
        // The card-resident blocks use the same protection: they win.
        let resident = std::env::var_os("VICTORY_CARD_RESIDENT").is_some_and(|v| v == "1");
        if asked && resident {
            tracing::warn!("read census off: VICTORY_CARD_RESIDENT=1 uses the same protection (it wins over VICTORY_READ_CENSUS=1)");
        }
        let on = asked && !resident;
        if on {
            tracing::info!("read census on (VICTORY_READ_CENSUS=1): copied-back blocks are watched for the title's reads");
        }
        on
    })
}

/// What the census counts, since the start (the memory line shows what
/// changed since the line before).
#[derive(Clone, Copy)]
#[repr(usize)]
enum Seen {
    /// Guarded blocks a copy-back wrote, and the bytes it wrote in them.
    Copied,
    CopiedBytes,
    /// Copy-backs into memory the guards do not hold, in 64 KiB blocks.
    Unwatched,
    /// First touch by the title's code (any module but ours), reading;
    /// and the bytes copied back into those blocks.
    TitleRead,
    TitleReadBytes,
    /// First touch by our own code in place (texture detiling and the like).
    OurRead,
    /// First touch by the title's code, writing.
    TitleWrite,
    /// Copied back again before anything touched it: the card's bytes were
    /// never read; and those bytes.
    CardAgain,
    CardAgainBytes,
    /// Our kernel reads (refreshes of other copies of the same memory).
    OurKernelRead,
    /// The title's kernel I/O into it.
    KernelWrite,
    /// Its memory was mapped anew, or a guard changed it.
    Lost,
    /// Title reads by frames after the copy-back: same, 1, 2-4, 5-30, more.
    Delay0,
    Delay1,
    Delay2,
    Delay5,
    Delay31,
    /// Our kernel reads, by who read: the copy that copied the block back,
    /// another copy that is a window, a title buffer's copy in the card's
    /// memory, another title buffer's copy, a texture, something else.
    ReadSame,
    ReadWindow,
    ReadCard,
    ReadBuffer,
    ReadTexture,
    ReadOther,
}
const SEEN: usize = Seen::ReadOther as usize + 1;

thread_local! {
    /// Who is reading or writing title memory on this thread now, for the
    /// census: a copy's number and kind (1 window, 2 card buffer, 3 other
    /// buffer, 4 texture), or (0, 0).
    static WHO: std::cell::Cell<(u64, u8)> = const { std::cell::Cell::new((0, 0)) };
}

/// Until what it returns is dropped, title memory read or written on this
/// thread is copy `id`'s (of `kind`, as [`WHO`]), for the census.
pub(crate) fn census_who(id: u64, kind: u8) -> Who {
    if !census() {
        return Who(None);
    }
    Who(Some(WHO.with(|w| w.replace((id, kind)))))
}

/// [`census_who`]'s guard.
pub(crate) struct Who(Option<(u64, u8)>);
impl Drop for Who {
    fn drop(&mut self) {
        if let Some(was) = self.0 {
            WHO.with(|w| w.set(was));
        }
    }
}
static SEEN_N: [std::sync::atomic::AtomicU64; SEEN] = [const { std::sync::atomic::AtomicU64::new(0) }; SEEN];
fn seen(what: Seen, n: u64) {
    SEEN_N[what as usize].fetch_add(n, std::sync::atomic::Ordering::Relaxed);
}

/// Blocks waiting for their first touch (frame of the copy-back, bytes it
/// wrote), and by 16 MiB region: blocks copied back, read by the title,
/// read by us, copied back again unread.
struct Census {
    waiting: BTreeMap<usize, (u64, u32, u64)>,
    regions: BTreeMap<usize, [u64; 4]>,
}
static CENSUS: Mutex<Census> = Mutex::new(Census { waiting: BTreeMap::new(), regions: BTreeMap::new() });
const REGION_SHIFT: u32 = 8; // 256 blocks of 64 KiB
/// Modules of the title reads' instructions (base, count); 0 is free.
static MODULES: [[std::sync::atomic::AtomicUsize; 2]; 8] = [const { [const { std::sync::atomic::AtomicUsize::new(0) }; 2] }; 8];

#[link(name = "ntdll", kind = "raw-dylib")]
unsafe extern "system" {
    /// The base of the module holding `pc`, or null; what the exception
    /// dispatcher itself uses, so safe in our handler.
    fn RtlPcToFileHeader(pc: *const std::ffi::c_void, base: *mut *mut std::ffi::c_void) -> *mut std::ffi::c_void;
}

fn module_of(pc: usize) -> usize {
    let mut base = std::ptr::null_mut();
    // SAFETY: a lookup in the loader's tables; writes only `base`.
    unsafe { RtlPcToFileHeader(pc as *const _, &mut base) };
    base as usize
}

fn our_module() -> usize {
    static OURS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *OURS.get_or_init(|| module_of(census as fn() -> bool as usize))
}

/// After a copy-back wrote `[at, at + len)` and the guards took `blocks`
/// again: those still guarded wait for their first touch.
fn census_mark(g: &Guards, blocks: &[usize], at: usize, len: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, VirtualProtect};
    let _ = our_module();
    let frame = crate::frame::current();
    let end = at + len;
    let mut c = CENSUS.lock().unwrap_or_else(|e| e.into_inner());
    for &block in blocks {
        if !g.protected.contains(&block) {
            continue;
        }
        let bytes = (end.min((block + 1) * BLOCK) - at.max(block * BLOCK)) as u32;
        let mut old = 0;
        // SAFETY: a guarded block of title view memory: read-only by our
        // hand; only its protection changes.
        if unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_NOACCESS, &mut old) } == 0 {
            continue;
        }
        c.waiting.insert(block, (frame, bytes, WHO.with(|w| w.get().0)));
        c.regions.entry(block >> REGION_SHIFT).or_default()[0] += 1;
        seen(Seen::Copied, 1);
        seen(Seen::CopiedBytes, u64::from(bytes));
    }
}

/// Blocks of `blocks` that waited are done waiting, for `why` (their
/// protection was changed by the caller).
fn census_forget(blocks: &[usize], why: Seen) {
    if !census() || blocks.is_empty() {
        return;
    }
    let mut c = CENSUS.lock().unwrap_or_else(|e| e.into_inner());
    if c.waiting.is_empty() {
        return;
    }
    for &block in blocks {
        if let Some((_, bytes, _)) = c.waiting.remove(&block) {
            seen(why, 1);
            if let Seen::CardAgain = why {
                seen(Seen::CardAgainBytes, u64::from(bytes));
                c.regions.entry(block >> REGION_SHIFT).or_default()[3] += 1;
            }
        }
    }
}

/// The census's part of a fault at `at` (`write`: a write; `pc`: the
/// faulting instruction): true if the block was waiting, now read-only
/// again (the access is tried again, and a write then faults as a guard's).
fn census_fault(g: &Guards, at: usize, write: bool, pc: usize) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE, VirtualProtect};
    let block = at / BLOCK;
    let mut c = CENSUS.lock().unwrap_or_else(|e| e.into_inner());
    let Some((frame, bytes, _)) = c.waiting.remove(&block) else {
        return false;
    };
    let mut old = 0;
    let back = if g.protected.contains(&block) { PAGE_READONLY } else { PAGE_READWRITE };
    // SAFETY: a block we left with no access; back to what the guards hold.
    unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, back, &mut old) };
    let module = module_of(pc);
    let region = c.regions.entry(block >> REGION_SHIFT).or_default();
    if module != 0 && module == our_module() {
        region[2] += 1;
        seen(Seen::OurRead, 1);
        return true;
    }
    if write {
        seen(Seen::TitleWrite, 1);
        return true;
    }
    region[1] += 1;
    seen(Seen::TitleRead, 1);
    seen(Seen::TitleReadBytes, u64::from(bytes));
    let late = crate::frame::current().saturating_sub(frame);
    seen(
        match late {
            0 => Seen::Delay0,
            1 => Seen::Delay1,
            2..=4 => Seen::Delay2,
            5..=30 => Seen::Delay5,
            _ => Seen::Delay31,
        },
        1,
    );
    for slot in &MODULES {
        let b = slot[0].load(Relaxed);
        if b == module || (b == 0 && slot[0].compare_exchange(0, module, Relaxed, Relaxed).is_ok_and(|_| true)) {
            slot[1].fetch_add(1, Relaxed);
            break;
        }
    }
    true
}

/// Our kernel read of title memory `[at, at + len)` failed: blocks there
/// that wait for the census are read-only again, counted as read by us;
/// true if any was (then the read is worth trying again).
pub(crate) fn census_open(at: usize, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE, VirtualProtect};
    if !census() || len == 0 {
        return false;
    }
    let g = guards();
    let mut c = CENSUS.lock().unwrap_or_else(|e| e.into_inner());
    let blocks: Vec<usize> = c.waiting.range(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK)).map(|(&b, _)| b).collect();
    let (reader, kind) = WHO.with(|w| w.get());
    for &block in &blocks {
        let writer = c.waiting.remove(&block).map_or(0, |w| w.2);
        seen(
            match kind {
                _ if reader != 0 && reader == writer => Seen::ReadSame,
                1 => Seen::ReadWindow,
                2 => Seen::ReadCard,
                3 => Seen::ReadBuffer,
                4 => Seen::ReadTexture,
                _ => Seen::ReadOther,
            },
            1,
        );
        let back = if g.protected.contains(&block) { PAGE_READONLY } else { PAGE_READWRITE };
        let mut old = 0;
        // SAFETY: as in `census_fault`.
        unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, back, &mut old) };
        seen(Seen::OurKernelRead, 1);
    }
    !blocks.is_empty()
}

/// For the memory line: what the census saw since the line before.
pub(crate) fn census_line() {
    use std::sync::atomic::Ordering::Relaxed;
    if !census() {
        return;
    }
    static LAST: Mutex<[u64; SEEN]> = Mutex::new([0; SEEN]);
    let now: [u64; SEEN] = std::array::from_fn(|i| SEEN_N[i].load(Relaxed));
    let d: [u64; SEEN] = {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        let d = std::array::from_fn(|i| now[i] - last[i]);
        *last = now;
        d
    };
    let s = |w: Seen| d[w as usize];
    let (waiting, regions) = {
        let c = CENSUS.lock().unwrap_or_else(|e| e.into_inner());
        let mut r: Vec<(usize, [u64; 4])> = c.regions.iter().map(|(&k, &v)| (k, v)).collect();
        r.sort_by_key(|(_, v)| std::cmp::Reverse(v[0]));
        r.truncate(8);
        (c.waiting.len(), r)
    };
    let regions: Vec<String> = regions
        .iter()
        .map(|(k, v)| format!("{:#x}: {} copied back, {} title-read, {} ours, {} unread when copied again", k << REGION_SHIFT << 16, v[0], v[1], v[2], v[3]))
        .collect();
    let modules: Vec<String> = MODULES
        .iter()
        .filter(|m| m[0].load(Relaxed) != 0)
        .map(|m| format!("{} {}", module_name(m[0].load(Relaxed)), m[1].load(Relaxed)))
        .collect();
    tracing::info!(
        "read census since the last such line: {} guarded blocks copied back ({} MB written), {} blocks copied back unwatched; first touches: title read {} ({} MB of copied-back bytes; frames after: same {}, 1 {}, 2-4 {}, 5-30 {}, more {}), our code in place {}, our kernel reads {}, title wrote first {}, title I/O {}; copied back again unread {} ({} MB); {} blocks waiting now",
        s(Seen::Copied),
        s(Seen::CopiedBytes) >> 20,
        s(Seen::Unwatched),
        s(Seen::TitleRead),
        s(Seen::TitleReadBytes) >> 20,
        s(Seen::Delay0),
        s(Seen::Delay1),
        s(Seen::Delay2),
        s(Seen::Delay5),
        s(Seen::Delay31),
        s(Seen::OurRead),
        s(Seen::OurKernelRead),
        s(Seen::TitleWrite),
        s(Seen::KernelWrite),
        s(Seen::CardAgain),
        s(Seen::CardAgainBytes) >> 20,
        waiting,
    );
    tracing::info!(
        "read census, our kernel reads by reader since the last such line: the copy that copied it back {}, a window {}, a buffer copy in the card's memory {}, another buffer copy {}, a texture {}, other {}",
        s(Seen::ReadSame),
        s(Seen::ReadWindow),
        s(Seen::ReadCard),
        s(Seen::ReadBuffer),
        s(Seen::ReadTexture),
        s(Seen::ReadOther),
    );
    tracing::info!("read census since the start, by 16 MiB region (most copied back first): {}; title reads by module: {}", regions.join("; "), modules.join(", "));
}

/// The census since the start, for a timeline file's metadata.
pub(crate) fn census_json() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    if !census() {
        return "null".into();
    }
    let names = [
        "copied", "copied_bytes", "unwatched", "title_read", "title_read_bytes", "our_read", "title_write", "card_again", "card_again_bytes",
        "our_kernel_read", "kernel_write", "lost", "delay_0", "delay_1", "delay_2_4", "delay_5_30", "delay_31",
        "read_same", "read_window", "read_card", "read_buffer", "read_texture", "read_other",
    ];
    let parts: Vec<String> = names.iter().enumerate().map(|(i, n)| format!("\"{n}\":{}", SEEN_N[i].load(Relaxed))).collect();
    format!("{{{}}}", parts.join(","))
}

fn module_name(base: usize) -> String {
    use windows_sys::Win32::System::LibraryLoader::GetModuleFileNameW;
    let mut buf = [0u16; 260];
    // SAFETY: a module base from the loader's tables; the buffer's length.
    let n = unsafe { GetModuleFileNameW(base as _, buf.as_mut_ptr(), buf.len() as u32) } as usize;
    let path = String::from_utf16_lossy(&buf[..n]);
    let name = path.rsplit(['\\', '/']).next().unwrap_or("").to_string();
    if name.is_empty() { format!("{base:#x}") } else { name }
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        // A block kept for the card ([`resident`]): its bytes written in,
        // read-only again; the access is tried again (a write then faults
        // once more, as a guard's). A read of a block another thread wrote
        // in since this one faulted is tried again too (a write goes on to
        // the guard below).
        if RESIDENT_N.load(std::sync::atomic::Ordering::Acquire) > 0
            && r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            && r.NumberParameters >= 2
            && resident()
        {
            let block = r.ExceptionInformation[1] / BLOCK;
            let _g = GUARDS.lock().unwrap_or_else(|e| e.into_inner());
            // A read is tried again whenever the address can be read now:
            // the block may have been written in and opened for a write
            // (no longer guarded) since this thread faulted (run 9's
            // crash: a read of a kept block, read-write by the report).
            if resident_write(block, block + 1, 2) > 0 || (r.ExceptionInformation[0] != WRITE && readable_at(r.ExceptionInformation[1])) {
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }
        // A block the read census left with no access: counted, and
        // read-only again (a write then faults once more, as a guard's).
        if census()
            && r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            && r.NumberParameters >= 2
            && census_fault(
                &GUARDS.lock().unwrap_or_else(|e| e.into_inner()),
                r.ExceptionInformation[1],
                r.ExceptionInformation[0] == WRITE,
                r.ExceptionAddress as usize,
            )
        {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        // Not `guards()`: nothing recorded in the exception handler.
        let mut g = GUARDS.lock().unwrap_or_else(|e| e.into_inner());
        if open(&mut g, at, at + 1, Seen::TitleWrite) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        // The journal is shared by the whole process and keeps pages it
        // was told of after their memory goes: a test running beside this
        // one (or before it, at an address reused here) may have left
        // entries for these pages. Only what is newer than this first
        // asking is this test's.
        let (first, _) = writes(start, end, false).expect("watched");
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert!(now > first);
        let after: Vec<_> = after.into_iter().filter(|&(_, w)| w > first).collect();
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        let landed = unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) };
        assert_eq!(landed, b"from the disk");
        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some((view / PAGE..view / PAGE + 1).collect())
        );
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
        title_memory_remapped(view, 2 * BLOCK);
    }

    /// A write of ours into guarded view memory leaves its blocks guarded
    /// again and noted written once, under its own asking; a title write
    /// after it is still taken by the guard.
    #[test]
    fn our_own_write_is_guarded_again_and_noted_once() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        const SIZE: usize = 3 * BLOCK;
        // SAFETY: a new pagefile section of three blocks, mapped once.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            view
        };
        let start = view.Value as usize;
        // First asking: every block is new, then all are guarded.
        assert_eq!(guard(start, start + SIZE).map(|w| w.len()), Some(3));
        assert_eq!(guard(start, start + SIZE), Some(vec![]));
        // Our write into the middle block, through the kernel.
        let own = own_write_begin(start + BLOCK + 8, 4);
        crate::host::write_mapped((start + BLOCK + 8) as *mut u8, b"ours".as_ptr(), 4);
        let now = own_write_end(own).expect("guarded memory");
        {
            let j = JOURNAL.read().unwrap();
            assert_eq!(j.written.get(&((start + BLOCK) / PAGE)), Some(&now));
        }
        // Guarded again: the next asking finds nothing written.
        assert_eq!(guard(start, start + SIZE), Some(vec![]));
        // A title write after it faults into the guard and is reported.
        install();
        // SAFETY: inside our view; the handler gives write access back.
        unsafe { ((start + BLOCK + 100) as *mut u8).write_volatile(7) };
        assert_eq!(guard(start, start + SIZE), Some(vec![(start + BLOCK) / PAGE]));
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(view) };
        title_memory_remapped(start, SIZE);
    }

    #[test]
    fn writes_in_partly_committed_view_blocks_are_reported() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };

        const SIZE: usize = 3 * BLOCK;
        struct View(MEMORY_MAPPED_VIEW_ADDRESS);
        impl Drop for View {
            fn drop(&mut self) {
                title_memory_remapped(self.0.Value as usize, SIZE);
                // SAFETY: our view is still mapped.
                unsafe { UnmapViewOfFile(self.0) };
            }
        }
        // SAFETY: a new reserved pagefile section. The view holds its own
        // reference after we close the handle, including on assertion failure.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            View(view)
        };
        let start = view.0.Value as usize;
        let commit = |at: usize, len| {
            // SAFETY: these pages belong to our reserved view.
            let committed = unsafe { VirtualAlloc(at as _, len, MEM_COMMIT, PAGE_READWRITE) };
            assert_eq!(committed as usize, at);
        };
        let early = start + BLOCK + 3 * KERNEL_PAGE;
        let late = start + 2 * BLOCK + 3 * KERNEL_PAGE;
        commit(start, BLOCK);
        commit(early, KERNEL_PAGE);
        let refresh = || guard(start, start + SIZE).unwrap();
        assert_eq!(refresh(), [start / PAGE, early / PAGE, late / PAGE]);
        // The last block is still entirely reserved.
        assert!(!refresh().contains(&(late / PAGE)));

        commit(late, KERNEL_PAGE); // Ordinary commit, without a runtime mapping notification.
        for at in [early, late] {
            assert!(writable(at));
            // SAFETY: each address is inside its committed read-write page.
            unsafe { (at as *mut u8).write_volatile(1) };
        }
        assert_eq!(refresh(), [early / PAGE, late / PAGE]);
    }

    /// A range of view memory whose first block is not mapped is a view
    /// all the same: its blocks are reported once, then the ones written
    /// or mapped since.
    #[test]
    fn a_range_that_begins_in_an_unmapped_block_is_guarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of three; the middle block is mapped first.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (3 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        // Nothing mapped at all: a view with every block absent.
        assert!(is_view(p, p + 3 * BLOCK));
        let middle = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        // The range begins in the placeholder, its middle block is a view.
        assert!(is_view(p, p + 3 * BLOCK));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()), "nothing written since");
        // SAFETY: inside the mapped middle block.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        let first = map(0);
        title_memory_remapped(p, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(0).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // Memory of ours that is no view stays none.
        let mut local = [0u8; 16];
        let at = local.as_mut_ptr() as usize;
        assert!(!is_view(at, at + local.len()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), middle, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// Blocks of a placeholder between views are reported once, then not
    /// again (they cannot be guarded) until the runtime maps them.
    #[test]
    fn unmapped_blocks_are_reported_once_until_mapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of two; block 0 shows the section's first block, as the
        // runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let first = map(0);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some(Vec::new()),
            "nothing written since"
        );
        let second = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }
}

#[cfg(test)]
mod mapping_tests {
    use super::{title_memory_reading, title_memory_remapping};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    /// A change of mapping waits for a read in place that is under way, and
    /// a read in place waits for a change of mapping.
    #[test]
    fn reads_in_place_and_remappings_wait_for_each_other() {
        // A reader holds; the change does not begin until it lets go.
        let changed = Arc::new(AtomicBool::new(false));
        let reading = title_memory_reading();
        let (began, wait) = mpsc::channel();
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _changing = title_memory_remapping();
                changed.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!changed.load(Ordering::SeqCst), "the mapping changed under a read in place");
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
        // A change holds; the read does not begin until it is done.
        let read = Arc::new(AtomicBool::new(false));
        let changing = title_memory_remapping();
        let (began, wait) = mpsc::channel();
        let reader = {
            let read = read.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _reading = title_memory_reading();
                read.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!read.load(Ordering::SeqCst), "a read in place began during a change of mapping");
        drop(changing);
        reader.join().unwrap();
        assert!(read.load(Ordering::SeqCst));
    }
}
