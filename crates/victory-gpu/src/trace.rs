//! A timeline of where every thread's time goes, for a few seconds of
//! play: `VICTORY_TRACE=<file.json>` writes a Chrome trace (open it at
//! ui.perfetto.dev or chrome://tracing, or read it with a script) of
//! named spans on every thread of ours and the title's that runs our
//! code: submissions, refreshes, write-watch askings, guards and their
//! protection changes, lock waits, waits for the card, copy-back landings,
//! presents. Nothing is recorded, and each span costs one relaxed load,
//! unless the variable is set.
//!
//! Recording starts at the present after F9 is pressed (press it where
//! the scene of interest is; F9 again later for another file,
//! `name-2.json` and on), or by itself `VICTORY_TRACE_FROM` presents after
//! the world's first frame (the first with 500 draws or more) if that is
//! set. `VICTORY_TRACE_FRAMES` (30) is how many presents a window covers. The file is written on a thread of its own when
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

#[derive(Clone)]
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
    static MIN: std::sync::OnceLock<(f64, f64)> = std::sync::OnceLock::new();
    let (frame, load) = *MIN.get_or_init(|| (number("VICTORY_TRACE_MIN_US", 5) as f64, number("VICTORY_TRACE_LOAD_MIN_US", 20) as f64));
    if LOAD.load(Ordering::Relaxed) { load } else { frame }
}

/// Whether the window open now (or last) is a load profile (F10): longer
/// (`VICTORY_TRACE_LOAD_FRAMES`, 60 presents), its own floor for spans
/// kept one by one (`VICTORY_TRACE_LOAD_MIN_US`, 20), a file named
/// `name-load-N.json`. Pressed while the world streams in (or anywhere it
/// stutters), it says which loading work costs the time: file reads,
/// shaders translated or taken from the cache, pipelines loaded or made
/// by the driver, textures and host copies brought up to date, with the
/// window's counters (bytes, requests) in the file's metadata.
static LOAD: AtomicBool = AtomicBool::new(false);

/// The window's counters by name ([`count`]).
static COUNTS: Mutex<Vec<(&'static str, u64)>> = Mutex::new(Vec::new());

/// Adds `n` to the window's counter `name`, while a window records.
pub(crate) fn count(name: &'static str, n: u64) {
    if !on() {
        return;
    }
    let mut counts = COUNTS.lock().unwrap_or_else(|e| e.into_inner());
    match counts.iter_mut().find(|c| std::ptr::eq(c.0, name)) {
        Some(c) => c.1 += n,
        None => counts.push((name, n)),
    }
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
    let window = WINDOW.load(Ordering::Relaxed);
    let frames = if window == 0 { 0 } else { window_frames() };
    if window == 0 {
        // A window opens at F9 (held at a present; the game's window need
        // not have the keyboard's focus), or by itself once
        // VICTORY_TRACE_FROM presents after the world's first frame if that
        // is set (`VICTORY_TRACE_FROM=0`: only by key).
        let world = WORLD.load(Ordering::Relaxed);
        let from = number("VICTORY_TRACE_FROM", 0);
        let by_count = from > 0 && world > 0 && n == world + from && TAKEN.load(Ordering::Relaxed) == 0;
        let load = !by_count && key_pressed(VK_F10);
        if !(by_count || load || key_pressed(VK_F9)) {
            return;
        }
        LOAD.store(load, Ordering::Relaxed);
        FLUSHED.store(0, Ordering::Relaxed);
        *PARTIAL.lock().unwrap_or_else(|e| e.into_inner()) = load.then(|| {
            let full = numbered(path);
            let stem = full.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            (full.with_file_name(format!("{stem}.partial.json")), false)
        });
        COUNTS.lock().unwrap_or_else(|e| e.into_inner()).clear();
        let shaders = crate::shader_cache::counts();
        *SHADERS_AT.lock().unwrap_or_else(|e| e.into_inner()) = shaders;
        let frames = window_frames();
        let _ = epoch();
        SPANS.lock().unwrap_or_else(|e| e.into_inner()).reserve_exact(MOST);
        WINDOW.store(n, Ordering::Relaxed);
        ON.store(true, Ordering::Relaxed);
        tracing::info!(path = %numbered(path).display(), frames, present = n, "trace: recording");
        return;
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
    // A load profile is also written as it goes, every second present, to
    // `name-load.partial.json` (a trace array the viewers read unfinished):
    // a run stopped by its memory guard while the world loads keeps what
    // was recorded until then.
    if on() && LOAD.load(Ordering::Relaxed) && n > window && n < window + frames && (n - window) % 2 == 0 {
        flush_partial();
    }
    if n == window + frames {
        ON.store(false, Ordering::Relaxed);
        let spans = std::mem::take(&mut *SPANS.lock().unwrap_or_else(|e| e.into_inner()));
        let threads = THREADS.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let totals = std::mem::take(&mut *TOTALS.lock().unwrap_or_else(|e| e.into_inner()));
        let mut counts = std::mem::take(&mut *COUNTS.lock().unwrap_or_else(|e| e.into_inner()));
        {
            let (hits, misses) = crate::shader_cache::counts();
            let (h0, m0) = *SHADERS_AT.lock().unwrap_or_else(|e| e.into_inner());
            counts.push(("shaders from the cache", hits - h0));
            counts.push(("shaders translated", misses - m0));
        }
        DROPPED.store(0, Ordering::Relaxed);
        let path = numbered(path);
        TAKEN.fetch_add(1, Ordering::Relaxed);
        let _ = std::thread::Builder::new()
            .name("victory-trace-write".into())
            .spawn(move || write(&path, &spans, &threads, &totals, &counts, frames));
    }
    // F9 or F10 again, after a window and once the keys were let go:
    // another window.
    if n >= window + frames && !key_down(VK_F9) && !key_down(VK_F10) {
        WINDOW.store(0, Ordering::Relaxed);
    }
}

/// The present that opened the window now recording or last recorded (0:
/// none, or the key may open another), and the windows written so far.
static WINDOW: AtomicU64 = AtomicU64::new(0);
static TAKEN: AtomicU64 = AtomicU64::new(0);

/// The load profile's partial file and whether it was begun; spans
/// written to it so far.
static PARTIAL: Mutex<Option<(std::path::PathBuf, bool)>> = Mutex::new(None);
static FLUSHED: AtomicUsize = AtomicUsize::new(0);

/// Appends the spans recorded since the last flush, and the totals and
/// counters so far, to the load profile's partial file (on a thread of its
/// own; one at a time).
fn flush_partial() {
    let spans: Vec<Span> = {
        let spans = SPANS.lock().unwrap_or_else(|e| e.into_inner());
        let from = FLUSHED.swap(spans.len(), Ordering::Relaxed).min(spans.len());
        spans[from..].to_vec()
    };
    let totals = TOTALS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let counts = COUNTS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let threads = THREADS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let _ = std::thread::Builder::new().name("victory-trace-partial".into()).spawn(move || {
        use std::io::Write;
        let mut partial = PARTIAL.lock().unwrap_or_else(|e| e.into_inner());
        let Some((path, begun)) = partial.as_mut() else {
            return;
        };
        let file = std::fs::OpenOptions::new().create(true).append(true).open(&*path);
        let Ok(file) = file else {
            return;
        };
        let mut out = std::io::BufWriter::new(file);
        let _ = (|| -> std::io::Result<()> {
            if !*begun {
                writeln!(out, "[")?;
                *begun = true;
            }
            for (n, name) in &threads {
                let name: String = name.chars().filter(|c| *c != '"' && *c != '\\').collect();
                writeln!(out, "{{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":1,\"tid\":{n},\"args\":{{\"name\":\"{name}\"}}}},")?;
            }
            for s in &spans {
                if s.len == 0.0 {
                    writeln!(out, "{{\"ph\":\"i\",\"s\":\"g\",\"name\":\"{}\",\"pid\":1,\"tid\":{},\"ts\":{:.3},\"args\":{{\"n\":{}}}}},", s.name, s.thread, s.start, s.arg)?;
                } else {
                    writeln!(out, "{{\"ph\":\"X\",\"name\":\"{}\",\"pid\":1,\"tid\":{},\"ts\":{:.3},\"dur\":{:.3},\"args\":{{\"n\":{}}}}},", s.name, s.thread, s.start, s.len, s.arg)?;
                }
            }
            // The totals and counters so far, as a mark whose arguments
            // hold them (the last such mark is the latest).
            let totals: Vec<String> = totals.iter().map(|(name, n, us)| format!("\"{name}\":[{n},{:.0}]", us)).collect();
            let counts: Vec<String> = counts.iter().map(|(name, v)| format!("\"{name}\":{v}")).collect();
            let now = epoch().elapsed().as_secs_f64() * 1e6;
            writeln!(
                out,
                "{{\"ph\":\"i\",\"s\":\"g\",\"name\":\"load profile so far\",\"pid\":1,\"tid\":0,\"ts\":{now:.3},\"args\":{{\"census\":{},\"counts\":{{{}}},\"totals_count_us\":{{{}}}}}}},",
                crate::watch::census_json(),
                counts.join(","),
                totals.join(",")
            )?;
            out.flush()
        })();
    });
}

/// Shader counts when the window opened.
static SHADERS_AT: Mutex<(u64, u64)> = Mutex::new((0, 0));

/// Presents the window open now covers.
fn window_frames() -> u64 {
    if LOAD.load(Ordering::Relaxed) {
        number("VICTORY_TRACE_LOAD_FRAMES", 60).max(1)
    } else {
        number("VICTORY_TRACE_FRAMES", 30).max(1)
    }
}

/// The file for the next window: the path given, then `name-2.json`, ...
/// (`name-load...` for a load profile).
fn numbered(path: &std::path::Path) -> std::path::PathBuf {
    let k = TAKEN.load(Ordering::Relaxed);
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let ext = path.extension().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "json".into());
    let stem = if LOAD.load(Ordering::Relaxed) { format!("{stem}-load") } else { stem };
    if k == 0 {
        return path.with_file_name(format!("{stem}.{ext}"));
    }
    path.with_file_name(format!("{stem}-{}.{ext}", k + 1))
}

use windows_sys::Win32::UI::Input::KeyboardAndMouse::{VK_F9, VK_F10};

/// Whether `key` was pressed since the last asking, or is held now: at 3
/// fps a quick press falls between two presents.
fn key_pressed(key: windows_sys::Win32::UI::Input::KeyboardAndMouse::VIRTUAL_KEY) -> bool {
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;
    // SAFETY: plain call.
    (unsafe { GetAsyncKeyState(i32::from(key)) } as u16) & 0x8001 != 0
}

/// Whether `key` is held now.
fn key_down(key: windows_sys::Win32::UI::Input::KeyboardAndMouse::VIRTUAL_KEY) -> bool {
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;
    // SAFETY: plain call.
    (unsafe { GetAsyncKeyState(i32::from(key)) } as u16) & 0x8000 != 0
}

fn write(path: &std::path::Path, spans: &[Span], threads: &[(u32, String)], totals: &[(&'static str, u64, f64)], counts: &[(&'static str, u64)], frames: u64) {
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
            "{{\"displayTimeUnit\":\"ms\",\"otherData\":{{\"dropped\":{},\"frames\":{frames},\"min_us\":{},\"census\":{},\"load\":{},\"counts\":{{{}}},\"totals\":{{{}}}}},\"traceEvents\":[",
            DROPPED.load(Ordering::Relaxed),
            min_us(),
            crate::watch::census_json(),
            LOAD.load(Ordering::Relaxed),
            counts.iter().map(|(n, v)| format!("\"{n}\":{v}")).collect::<Vec<_>>().join(","),
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
