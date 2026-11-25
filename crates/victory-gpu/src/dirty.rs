//! The dirty-page mask shaders write (`VICTORY_DIRTY_MASK`; translator side:
//! `victory_shader::emit::SLOT_DIRTY`). A translated store also sets the bit
//! of each 4 KiB page of title memory it writes, in one buffer in the card's
//! memory; the lane that finds a bit clear lists the page. After each
//! submission that ran such a shader, a list of ours (its own
//! `ExecuteCommandLists`, so the submission's writes are done) copies the
//! count and the page list into a read-back buffer and copies zeros over
//! the whole mask. When the submission has run, its pages are known
//! without a snapshot or a compare.
//!
//! Step 1 (`VICTORY_DIRTY_MASK=count`, this module now): nothing relies on
//! the marks. Each landed submission's pages are kept for a while, and each
//! byte run the copy-back finds the card changed and writes into title
//! memory is checked against them: a changed page no recent submission
//! marked is a store the translator does not mark (a miss), counted and
//! logged. Zero misses over a run is the gate for step 2 (the marks
//! replace the compare).

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::emit::{DIRTY_DWORDS, DIRTY_LIST, DIRTY_LIST_AT};
use windows::Win32::Graphics::Direct3D12::*;

use crate::host::Host;

/// The key a list notes (`HostList::first_use`) when a draw of it binds the
/// mask.
pub(crate) const USED: usize = usize::MAX - 0x70;

/// Bytes of the head and page list copied back per submission.
const BACK_BYTES: u64 = (DIRTY_LIST_AT as u64 + DIRTY_LIST as u64 + 1) * 4;

/// Submissions a page's mark is kept for, for the check: a range the
/// copy-back writes was declared by a submission at most this far back.
const KEEP: u64 = 256;

/// Whether shaders mark (0 off, 1 count only, 2 relied on).
pub(crate) fn mode() -> u8 {
    victory_shader::emit::dirty_mask()
}

struct Dirty {
    /// The mask (card memory), and zeros of its size to clear it with.
    mask: ID3D12Resource,
    zeros: ID3D12Resource,
    /// Read-back buffers free for a submission, and lists to record in.
    free: Vec<ID3D12Resource>,
    lists: Vec<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>,
    /// Per submission sent and not landed: its read-back buffer and list.
    out: HashMap<
        u64,
        (
            ID3D12Resource,
            (ID3D12CommandAllocator, ID3D12GraphicsCommandList),
        ),
    >,
    /// Page → the latest landed submission that marked it.
    marked: BTreeMap<u32, u64>,
    /// The latest submission landed.
    landed: u64,
    /// Submissions whose marks are not all there (a store the translator
    /// could not place ran, or the list overflowed): the check skips
    /// changed pages up to this one.
    blind_until: u64,
    /// Since the last line: submissions not all marked, by the container
    /// hash (low 32 bits) of a shader whose store was not placed (the last
    /// such store of the submission).
    unplaced: HashMap<u32, u64>,
}

// SAFETY: COM objects used under the mutex only.
unsafe impl Send for Dirty {}

static DIRTY: Mutex<Option<Dirty>> = Mutex::new(None);

/// Since the last line: submissions with marks read; pages listed; of
/// them submissions not all marked (stores not placed), list overflows;
/// pages' bytes the card changed, checked; of them in pages marked, in
/// pages not marked (misses), not marked while a recent submission had
/// unplaced stores; miss runs. Step 2: copy-backs that looked everywhere
/// (marks not trusted), that looked at marked pages only, the bytes they
/// were declared over and the bytes they looked at.
static STATS: [AtomicU64; 13] = [const { AtomicU64::new(0) }; 13];

fn lock() -> std::sync::MutexGuard<'static, Option<Dirty>> {
    DIRTY.lock().unwrap_or_else(|e| e.into_inner())
}

/// The mask's GPU address for a draw whose shader marks; `None` if it
/// could not be made (the draw then runs with the root view unset, which
/// D3D12 refuses: the caller drops it).
pub(crate) fn mask(host: &Host) -> Option<u64> {
    let mut d = lock();
    if d.is_none() {
        let bytes = u64::from(DIRTY_DWORDS) * 4;
        let mask = host.uav_buffer(bytes)?;
        crate::host::name(&mask, || "dirty pages: mask and list".into());
        // A committed buffer in the card's memory starts zeroed.
        let zeros = host.uav_buffer(bytes)?;
        crate::host::name(&zeros, || "dirty pages: zeros".into());
        tracing::info!(
            mode = mode(),
            "dirty pages (VICTORY_DIRTY_MASK): shaders mark the 4 KiB pages their buffer stores write ({} MB of card memory); count only: {}",
            (bytes * 2) >> 20,
            mode() == 1
        );
        *d = Some(Dirty {
            mask,
            zeros,
            free: Vec::new(),
            lists: Vec::new(),
            out: HashMap::new(),
            marked: BTreeMap::new(),
            landed: 0,
            blind_until: 0,
            unplaced: HashMap::new(),
        });
    }
    // SAFETY: a live buffer.
    d.as_ref().map(|d| unsafe { d.mask.GetGPUVirtualAddress() })
}

/// After submission `number`'s lists (of which some bound the mask): a list
/// of ours that takes the marks back and clears them, to be executed on its
/// own right after them. `None` if there is none.
pub(crate) fn after(host: &Host, number: u64) -> Option<ID3D12CommandList> {
    let mut g = lock();
    let d = g.as_mut()?;
    let back = match d.free.pop() {
        Some(b) => b,
        None => {
            let b = host.read_back_buffer(BACK_BYTES)?;
            crate::host::name(&b, || "dirty pages: a submission's list".into());
            b
        }
    };
    let (allocator, list) = match d.lists.pop() {
        Some(l) => l,
        None => host.make_prelude()?,
    };
    // SAFETY: the list's last run finished (its submission landed before
    // it came back); live buffers. The mask decayed to the common state at
    // the end of the submission's ExecuteCommandLists: the copy out
    // promotes it to a copy source, then it is moved to a copy destination
    // for the zeros and back.
    let recorded = unsafe {
        (|| -> windows::core::Result<ID3D12CommandList> {
            allocator.Reset()?;
            list.Reset(&allocator, None)?;
            list.CopyBufferRegion(&back, 0, &d.mask, 0, BACK_BYTES);
            list.ResourceBarrier(&[crate::host::transition(
                &d.mask,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            list.CopyBufferRegion(&d.mask, 0, &d.zeros, 0, u64::from(DIRTY_DWORDS) * 4);
            list.ResourceBarrier(&[crate::host::transition(
                &d.mask,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
            list.Close()?;
            windows::core::Interface::cast(&list)
        })()
    };
    if d.out.len() == 1024 {
        tracing::warn!(
            "dirty pages: 1024 submissions' marks were never read (a landing path that does not call dirty::landed)"
        );
    }
    match recorded {
        Ok(l) => {
            d.out.insert(number, (back, (allocator, list)));
            Some(l)
        }
        Err(e) => {
            tracing::warn!(error = %e, "dirty pages: the list that takes the marks back could not be recorded");
            d.free.push(back);
            d.lists.push((allocator, list));
            None
        }
    }
}

/// Submission `number` has run (before its copy-back): its marks are read.
pub(crate) fn landed(number: u64) {
    let mut g = lock();
    let Some(d) = g.as_mut() else {
        return;
    };
    d.landed = d.landed.max(number);
    let Some((back, list)) = d.out.remove(&number) else {
        return;
    };
    let mut data = std::ptr::null_mut();
    let range = D3D12_RANGE {
        Begin: 0,
        End: BACK_BYTES as usize,
    };
    // SAFETY: a read-back buffer the card has finished writing.
    if unsafe { back.Map(0, Some(&range), Some(&mut data)) }.is_ok() {
        let words = data.cast::<u32>();
        // SAFETY: inside the mapping.
        let (count, unplaced) = unsafe { (words.read(), words.add(1).read()) };
        let n = count.min(DIRTY_LIST) as usize;
        STATS[0].fetch_add(1, Ordering::Relaxed);
        STATS[1].fetch_add(n as u64, Ordering::Relaxed);
        if unplaced != 0 {
            STATS[2].fetch_add(1, Ordering::Relaxed);
            *d.unplaced.entry(unplaced).or_default() += 1;
        }
        if count > DIRTY_LIST {
            STATS[3].fetch_add(1, Ordering::Relaxed);
        }
        if unplaced != 0 || count > DIRTY_LIST {
            d.blind_until = d.blind_until.max(number);
        }
        for i in 0..n {
            // SAFETY: inside the mapping.
            let page = unsafe { words.add(DIRTY_LIST_AT as usize + i).read() };
            d.marked.insert(page, number);
        }
        // SAFETY: mapped above; nothing written.
        unsafe { back.Unmap(0, Some(&D3D12_RANGE::default())) };
    }
    d.free.push(back);
    d.lists.push(list);
    // Old marks go now and then.
    if d.marked.len() > 1 << 18 {
        let keep_from = d.landed.saturating_sub(KEEP);
        d.marked.retain(|_, &mut n| n >= keep_from);
    }
}

/// Step 2 (`VICTORY_DIRTY_MASK=1`): the runs of 4 KiB pages of title
/// memory `[from, to)` that submissions landed lately marked, as title
/// address ranges cut to `[from, to)`; the copy-back looks only there for
/// what the card changed. `None` (look everywhere, as before) when the
/// marks are not relied on, or a submission landed lately ran a store the
/// translator could not place or overflowed its list.
pub(crate) fn marked_in(from: usize, to: usize) -> Option<Vec<(usize, usize)>> {
    if mode() != 2 || from >= to {
        return None;
    }
    let g = lock();
    let d = g.as_ref()?;
    let since = d.landed.saturating_sub(KEEP);
    if d.blind_until > 0 && d.blind_until >= since {
        STATS[9].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let last = victory_shader::emit::DIRTY_PAGES - 1;
    let (first_page, last_page) = (((from >> 12) as u32).min(last), (((to - 1) >> 12) as u32).min(last));
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (&page, &k) in d.marked.range(first_page..=last_page) {
        if k < since {
            continue;
        }
        let (a, b) = (((page as usize) << 12).max(from), (((page as usize) + 1) << 12).min(to));
        match runs.last_mut() {
            Some(r) if r.1 == a => r.1 = b,
            _ => runs.push((a, b)),
        }
    }
    STATS[10].fetch_add(1, Ordering::Relaxed);
    STATS[11].fetch_add((to - from) as u64, Ordering::Relaxed);
    STATS[12].fetch_add(runs.iter().map(|r| (r.1 - r.0) as u64).sum(), Ordering::Relaxed);
    Some(runs)
}

/// Count only: at title address `at` the card's bytes are `now`, the
/// copy's bytes before the submission `was` (the copy-back is about to put
/// those that differ into title memory). Each 4 KiB page with a byte that
/// differs should have been marked by a recent submission.
pub(crate) fn check_changed(at: usize, now: &[u8], was: &[u8]) {
    if mode() != 1 || now.is_empty() {
        return;
    }
    let n = now.len().min(was.len());
    let mut changed: Vec<(usize, u64)> = Vec::new();
    let mut i = 0;
    while i < n {
        let page_end = ((at + i) & !0xFFF) + 4096;
        let j = (page_end - at).min(n);
        if now[i..j] != was[i..j] {
            changed.push(((at + i) >> 12, (j - i) as u64));
        }
        i = j;
    }
    if changed.is_empty() {
        return;
    }
    let g = lock();
    let Some(d) = g.as_ref() else {
        return;
    };
    let since = d.landed.saturating_sub(KEEP);
    let blind = d.blind_until > 0 && d.blind_until >= since;
    let (mut missed, mut first_miss) = (0u64, None);
    for (page, bytes) in changed {
        STATS[4].fetch_add(bytes, Ordering::Relaxed);
        let key = page.min(victory_shader::emit::DIRTY_PAGES as usize - 1) as u32;
        match d.marked.get(&key) {
            Some(&k) if k >= since => {
                STATS[5].fetch_add(bytes, Ordering::Relaxed);
            }
            _ if blind => {
                STATS[7].fetch_add(bytes, Ordering::Relaxed);
            }
            _ => {
                missed += bytes;
                first_miss.get_or_insert(page << 12);
            }
        }
    }
    if missed > 0 {
        STATS[6].fetch_add(missed, Ordering::Relaxed);
        if STATS[8].fetch_add(1, Ordering::Relaxed) < 40 {
            tracing::warn!(
                at = format_args!("{at:#x}"),
                n,
                page = format_args!("{:#x}", first_miss.unwrap_or(0)),
                "dirty pages: the card changed bytes in a page no recent submission marked (a store the translator does not mark)"
            );
        }
    }
}

/// The line on the memory log, since the last.
pub(crate) fn line() {
    if mode() == 0 {
        return;
    }
    let s: Vec<u64> = STATS.iter().map(|a| a.swap(0, Ordering::Relaxed)).collect();
    let mut unplaced: Vec<(u32, u64)> = lock().as_mut().map(|d| std::mem::take(&mut d.unplaced).into_iter().collect()).unwrap_or_default();
    unplaced.sort_by_key(|u| std::cmp::Reverse(u.1));
    if !unplaced.is_empty() {
        let top: Vec<String> = unplaced.iter().take(12).map(|(h, n)| format!("{h:08x} {n}")).collect();
        tracing::info!(
            "dirty pages: submissions not all marked since the last such line, by a shader whose store was not placed (container hash, last 8 digits, then submissions): {}",
            top.join(", ")
        );
    }
    let sites = &victory_shader::emit::DIRTY_SITES;
    tracing::info!(
        "dirty pages since the last such line: {} submissions read, {} pages marked ({:.1} MB); {} not all marked (a store not placed ran), {} list overflows; pages the card changed, checked: {:.2} MB, {:.2} MB of them in marked pages, {:.2} MB in pages not marked (misses, {} runs), {:.2} MB not marked while stores not placed ran; store sites translated: {} placed, {} not; copy-back by marks (VICTORY_DIRTY_MASK=1): {} ranges looked at marked pages only ({:.1} MB declared, {:.1} MB looked at), {} looked everywhere (marks not all there)",
        s[0],
        s[1],
        s[1] as f64 * 4096.0 / 1e6,
        s[2],
        s[3],
        s[4] as f64 / 1e6,
        s[5] as f64 / 1e6,
        s[6] as f64 / 1e6,
        s[8],
        s[7] as f64 / 1e6,
        sites[0].load(Ordering::Relaxed),
        sites[1].load(Ordering::Relaxed),
        s[10],
        s[11] as f64 / 1e6,
        s[12] as f64 / 1e6,
        s[9],
    );
}
