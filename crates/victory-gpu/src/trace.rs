//! A timeline of where every thread's time goes, for a few seconds of
//! play: `VICTORY_TRACE=<file.json>` writes a Chrome trace (open it at
//! ui.perfetto.dev or chrome://tracing, or read it with a script) of
//! named spans on every thread of ours and the title's that runs our
//! code: submissions, refreshes, write-watch askings, guards and their
//! protection changes, lock waits, waits for the card, copy-back landings,
//! presents. Nothing is recorded, and each span costs one relaxed load,
//! unless the variable is set.
//!
//! `VICTORY_TRACE_FROM` (presents after the world's first frame, the
//! first with 500 draws or more; 600 by default) is where recording starts and `VICTORY_TRACE_FRAMES` (30) how many
//! presents it covers. The file is written on a thread of its own when
//! the window ends. Spans past [`MOST`] in the window are dropped (and
//! counted in the file's metadata).
//!
//! What game developers use for the same question is a profiler that
//! records zones (Tracy, PIX timing captures, Superluminal): this is that,
//! with no tool to install, so a run leaves a file we can read.

use std::cell::Cell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

/// Spans kept in a window: some 3 million a second at 4 fps is far more
/// than a frame makes; room is reserved when the window opens, so a span
/// never allocates.
const MOST: usize = 3 << 20;

/// Whether the window is open now.
static ON: AtomicBool = AtomicBool::new(false);
/// Spans dropped for lack of room.
static DROPPED: AtomicU64 = AtomicU64::new(0);
/// Presents seen so far.
static PRESENTS: AtomicU64 = AtomicU64::new(0);
/// Draws counted at the last present, and the present of the world's
/// first frame (0: not yet).
static DRAWN: AtomicU64 = AtomicU64::new(0);
static WORLD: AtomicU64 = AtomicU64::new(0);
/// A small number per thread for the file (thread ids are long).
static NEXT_THREAD: AtomicUsize = AtomicUsize::new(1);

struct Span {
    name: &'static str,
    thread: u32,
    /// Microseconds since [`epoch`], with fractions.
    start: f64,
    len: f64,
    /// A number to show with the span (bytes, blocks, pieces), or 0.
    arg: u64,
}

static SPANS: Mutex<Vec<Span>> = Mutex::new(Vec::new());
/// Per span name, every span in the window: count and microseconds, also
/// of those too short to be kept one by one.
static TOTALS: Mutex<Vec<(&'static str, u64, f64)>> = Mutex::new(Vec::new());
/// Spans shorter than this many microseconds are only counted in
/// [`TOTALS`] (`VICTORY_TRACE_MIN_US`, 5): a frame makes tens of
/// thousands of copy refreshes of a few microseconds.
fn min_us() -> f64 {
    static MIN: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| number("VICTORY_TRACE_MIN_US", 5) as f64)
}

/// Thread numbers and names, for the file.
static THREADS: Mutex<Vec<(u32, String)>> = Mutex::new(Vec::new());

thread_local! {
    static THREAD: Cell<u32> = const { Cell::new(0) };
}

fn epoch() -> Instant {
    static EPOCH: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

fn path() -> Option<&'static std::path::PathBuf> {
    static PATH: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    PATH.get_or_init(|| std::env::var_os("VICTORY_TRACE").map(std::path::PathBuf::from)).as_ref()
}

fn number(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

fn thread() -> u32 {
    THREAD.with(|t| {
        if t.get() == 0 {
            let n = NEXT_THREAD.fetch_add(1, Ordering::Relaxed) as u32;
            t.set(n);
            let name = std::thread::current().name().map_or_else(
                // SAFETY: plain call.
                || format!("thread {}", unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }),
                str::to_owned,
            );
            THREADS.lock().unwrap_or_else(|e| e.into_inner()).push((n, name));
        }
        t.get()
    })
}

/// Whether spans are being recorded now.
#[inline]
pub(crate) fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// A span from now until it is dropped.
pub(crate) struct Zone {
    name: &'static str,
    start: Option<Instant>,
    arg: u64,
}

impl Zone {
    /// A number to show with the span.
    pub(crate) fn arg(&mut self, arg: u64) {
        self.arg = arg;
    }
}

impl Drop for Zone {
    fn drop(&mut self) {
        let Some(start) = self.start else {
            return;
        };
        if !on() {
            return;
        }
        let end = Instant::now();
        let base = epoch();
        let len = end.saturating_duration_since(start).as_secs_f64() * 1e6;
        {
            let mut totals = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
            match totals.iter_mut().find(|t| std::ptr::eq(t.0, self.name)) {
                Some(t) => {
                    t.1 += 1;
                    t.2 += len;
                }
                None => totals.push((self.name, 1, len)),
            }
        }
        if len < min_us() {
            return;
        }
        let span = Span {
            name: self.name,
            thread: thread(),
            start: start.saturating_duration_since(base).as_secs_f64() * 1e6,
            len,
            arg: self.arg,
        };
        let mut spans = SPANS.lock().unwrap_or_else(|e| e.into_inner());
        if spans.len() < spans.capacity() {
            spans.push(span);
        } else {
            DROPPED.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// A span named `name` on this thread until what it returns is dropped.
#[inline]
pub(crate) fn zone(name: &'static str) -> Zone {
    Zone { name, start: on().then(Instant::now), arg: 0 }
}

/// Takes `lock`, recording the wait for it as a span named `name` when
/// the wait was 20 microseconds or more.
#[inline]
pub(crate) fn lock<'a, T>(name: &'static str, lock: &'a Mutex<T>) -> std::sync::MutexGuard<'a, T> {
    if !on() {
        return lock.lock().unwrap_or_else(|e| e.into_inner());
    }
    if let Ok(guard) = lock.try_lock() {
        return guard;
    }
    let mut zone = zone(name);
    let guard = lock.lock().unwrap_or_else(|e| e.into_inner());
    if zone.start.is_some_and(|s| s.elapsed().as_micros() < 20) {
        zone.start = None;
    }
    guard
}

/// `WaitForSingleObject(event, INFINITE)`, recorded as a span.
///
/// # Safety
/// As the call it makes: `event` is a live event handle.
pub(crate) unsafe fn wait(event: windows::Win32::Foundation::HANDLE) -> windows::Win32::Foundation::WAIT_EVENT {
    let _z = zone("wait: the card (fence event)");
    // SAFETY: the caller's.
    unsafe { windows::Win32::System::Threading::WaitForSingleObject(event, windows::Win32::System::Threading::INFINITE) }
}

/// A present: the frame boundary. Opens and closes the window.
pub(crate) fn present() {
    let Some(path) = path() else {
        return;
    };
    let n = PRESENTS.fetch_add(1, Ordering::Relaxed) + 1;
    // The world's first frame: the first with 500 draws or more since the
    // present before it (menus and loading screens draw a few dozen).
    let drawn = crate::draw::drawn();
    let before = DRAWN.swap(drawn, Ordering::Relaxed);
    if WORLD.load(Ordering::Relaxed) == 0 && drawn.saturating_sub(before) >= 500 {
        WORLD.store(n, Ordering::Relaxed);
        tracing::info!(present = n, "trace: the world's first frame");
    }
    let world = WORLD.load(Ordering::Relaxed);
    if world == 0 {
        return;
    }
    let from = world + number("VICTORY_TRACE_FROM", 600);
    let frames = number("VICTORY_TRACE_FRAMES", 30).max(1);
    if n == from {
        let _ = epoch();
        SPANS.lock().unwrap_or_else(|e| e.into_inner()).reserve_exact(MOST);
        ON.store(true, Ordering::Relaxed);
        tracing::info!(path = %path.display(), frames, "trace: recording");
    }
    if on() {
        // The present itself, as a mark on the presenting thread.
        let now = epoch().elapsed().as_secs_f64() * 1e6;
        let thread = thread();
        let mut spans = SPANS.lock().unwrap_or_else(|e| e.into_inner());
        if spans.len() < spans.capacity() {
            spans.push(Span { name: "present", thread, start: now, len: 0.0, arg: n });
        }
    }
    if n == from + frames {
        ON.store(false, Ordering::Relaxed);
        let spans = std::mem::take(&mut *SPANS.lock().unwrap_or_else(|e| e.into_inner()));
        let threads = THREADS.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let totals = std::mem::take(&mut *TOTALS.lock().unwrap_or_else(|e| e.into_inner()));
        let path = path.clone();
        let _ = std::thread::Builder::new()
            .name("victory-trace-write".into())
            .spawn(move || write(&path, &spans, &threads, &totals, frames));
    }
}

fn write(path: &std::path::Path, spans: &[Span], threads: &[(u32, String)], totals: &[(&'static str, u64, f64)], frames: u64) {
    use std::io::Write;
    let file = match std::fs::File::create(path) {
        Ok(f) => f,
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "trace: the file could not be made");
            return;
        }
    };
    let mut out = std::io::BufWriter::new(file);
    let mut result = (|| -> std::io::Result<()> {
        let totals: Vec<String> = totals
            .iter()
            .map(|(name, n, us)| format!("\"{name}\":{{\"count\":{n},\"ms_per_frame\":{:.3}}}", us / 1e3 / frames as f64))
            .collect();
        writeln!(
            out,
            "{{\"displayTimeUnit\":\"ms\",\"otherData\":{{\"dropped\":{},\"frames\":{frames},\"min_us\":{},\"totals\":{{{}}}}},\"traceEvents\":[",
            DROPPED.load(Ordering::Relaxed),
            min_us(),
            totals.join(",")
        )?;
        for (n, name) in threads {
            let name: String = name.chars().filter(|c| *c != '"' && *c != '\\').collect();
            writeln!(out, "{{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":1,\"tid\":{n},\"args\":{{\"name\":\"{name}\"}}}},")?;
        }
        for s in spans {
            if s.len == 0.0 {
                writeln!(out, "{{\"ph\":\"i\",\"s\":\"g\",\"name\":\"{}\",\"pid\":1,\"tid\":{},\"ts\":{:.3},\"args\":{{\"n\":{}}}}},", s.name, s.thread, s.start, s.arg)?;
            } else {
                writeln!(out, "{{\"ph\":\"X\",\"name\":\"{}\",\"pid\":1,\"tid\":{},\"ts\":{:.3},\"dur\":{:.3},\"args\":{{\"n\":{}}}}},", s.name, s.thread, s.start, s.len, s.arg)?;
            }
        }
        writeln!(out, "{{\"ph\":\"M\",\"name\":\"process_name\",\"pid\":1,\"args\":{{\"name\":\"victory\"}}}}]}}")?;
        out.flush()
    })();
    if let Err(e) = result.as_mut() {
        tracing::warn!(path = %path.display(), error = %e, "trace: writing failed");
    } else {
        tracing::info!(path = %path.display(), spans = spans.len(), dropped = DROPPED.load(Ordering::Relaxed), "trace: written");
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_zone_off_records_nothing() {
        let z = super::zone("x");
        assert!(z.start.is_none());
    }
}
