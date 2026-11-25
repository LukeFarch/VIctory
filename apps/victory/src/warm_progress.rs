//! What `translate --warm` shows while it fills the shader cache: a small
//! window (Windows) and a console progress line, so a warm that takes
//! minutes is not a blank wait before the game window.
//!
//! The walk only bumps atomics and, once per shader, swaps the name of the
//! one it is on (a lock no other worker waits on: they skip it when it is
//! taken); a timer reads them a few times a second. Nothing leaves the
//! machine. `VICTORY_WARM_WINDOW=0` keeps to the console line.

use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::shader::MAX_FILE;

/// What the walk has done so far.
struct Progress {
    started: OnceLock<Instant>,
    finished: AtomicBool,
    threads: AtomicU64,
    /// From a metadata walk beside the real one (0 until it ends).
    files_total: AtomicU64,
    bytes_total: AtomicU64,
    files_done: AtomicU64,
    bytes_done: AtomicU64,
    /// Containers found in the files read, duplicates included.
    found: AtomicU64,
    duplicates: AtomicU64,
    /// Distinct containers through the translator, by outcome.
    done: AtomicU64,
    translated: AtomicU64,
    /// The translator stopped on code it does not handle yet ("is not
    /// supported"): a known gap.
    refused: AtomicU64,
    /// Anything else: malformed or undecodable code, a cache error.
    failed: AtomicU64,
    current: Mutex<String>,
    last_failure: Mutex<String>,
    /// The previous complete warm: distinct containers and seconds.
    last_done: AtomicU64,
    last_secs: AtomicU64,
}

static P: Progress = Progress {
    started: OnceLock::new(),
    finished: AtomicBool::new(false),
    threads: AtomicU64::new(0),
    files_total: AtomicU64::new(0),
    bytes_total: AtomicU64::new(0),
    files_done: AtomicU64::new(0),
    bytes_done: AtomicU64::new(0),
    found: AtomicU64::new(0),
    duplicates: AtomicU64::new(0),
    done: AtomicU64::new(0),
    translated: AtomicU64::new(0),
    refused: AtomicU64::new(0),
    failed: AtomicU64::new(0),
    current: Mutex::new(String::new()),
    last_failure: Mutex::new(String::new()),
    last_done: AtomicU64::new(0),
    last_secs: AtomicU64::new(0),
};

/// Shows the progress until [`Shown::finish`].
pub(crate) struct Shown {
    console: Option<std::thread::JoinHandle<()>>,
    terminal: bool,
    #[cfg(windows)]
    window: Option<std::thread::JoinHandle<()>>,
    stats: Option<PathBuf>,
}

/// Starts the display for a warm of `paths` on `threads` workers.
pub(crate) fn start(paths: &[PathBuf], threads: usize) -> Shown {
    let _ = P.started.set(Instant::now());
    P.threads.store(threads as u64, Ordering::Relaxed);
    let stats = stats_file();
    if let Some((done, secs)) = stats.as_deref().and_then(read_stats) {
        P.last_done.store(done, Ordering::Relaxed);
        P.last_secs.store(secs, Ordering::Relaxed);
    }
    // Sizes only, for the ETA of a first warm; it ends long before the walk.
    let roots = paths.to_vec();
    let _ = std::thread::Builder::new()
        .name("warm-size".into())
        .spawn(move || {
            let (mut files, mut bytes) = (0, 0);
            for root in &roots {
                for entry in walkdir::WalkDir::new(root)
                    .into_iter()
                    .filter_map(Result::ok)
                {
                    let len = entry.metadata().map_or(0, |m| m.len());
                    if entry.file_type().is_file() && len <= MAX_FILE {
                        files += 1;
                        bytes += len;
                    }
                }
            }
            P.files_total.store(files, Ordering::Relaxed);
            P.bytes_total.store(bytes, Ordering::Relaxed);
        });
    // On a terminal, a line redrawn in place; into a file (a script's log),
    // a plain line every half minute, so a warm with no window still tells.
    let terminal = std::io::stderr().is_terminal();
    let console = std::thread::Builder::new()
        .name("warm-console".into())
        .spawn(move || {
            let mut ticks = 0u32;
            while !P.finished.load(Ordering::Relaxed) {
                ticks += 1;
                if terminal {
                    let mut err = std::io::stderr().lock();
                    let _ = write!(err, "\r{}\x1b[K", console_line());
                    let _ = err.flush();
                } else if ticks.is_multiple_of(60) {
                    eprintln!("{}", console_line());
                }
                std::thread::sleep(Duration::from_millis(500));
            }
        })
        .ok();
    #[cfg(windows)]
    let window = (std::env::var("VICTORY_WARM_WINDOW").as_deref() != Ok("0"))
        .then(window::spawn)
        .flatten();
    Shown {
        console,
        terminal,
        #[cfg(windows)]
        window,
        stats,
    }
}

impl Shown {
    /// Closes the display and keeps this warm's size and time for the
    /// next one's ETA. Returns the one-line summary.
    pub(crate) fn finish(self) -> String {
        P.finished.store(true, Ordering::Relaxed);
        #[cfg(windows)]
        if let Some(w) = self.window {
            window::close();
            let _ = w.join();
        }
        if let Some(c) = self.console {
            let _ = c.join();
            if self.terminal {
                let mut err = std::io::stderr().lock();
                let _ = writeln!(err, "\r{}\x1b[K", console_line());
            }
        }
        let secs = elapsed().as_secs();
        let done = P.done.load(Ordering::Relaxed);
        // A filtered warm is not the size of the next one.
        if let Some(f) = &self.stats
            && done > 0
            && std::env::var_os("VICTORY_TRANSLATE_ONLY").is_none()
        {
            let _ = std::fs::write(f, format!("done={done}\nsecs={secs}\n"));
        }
        let (hits, new) = cache_counts();
        format!(
            "warm         {done} shaders in {}: {} translated ({hits} from the cache, {new} new), \
             {} refused, {} failed; {} duplicates skipped",
            clock(secs),
            P.translated.load(Ordering::Relaxed),
            P.refused.load(Ordering::Relaxed),
            P.failed.load(Ordering::Relaxed),
            P.duplicates.load(Ordering::Relaxed),
        )
    }
}

/// A file was read, holding `containers` shader containers.
pub(crate) fn file(bytes: usize, containers: usize) {
    P.files_done.fetch_add(1, Ordering::Relaxed);
    P.bytes_done.fetch_add(bytes as u64, Ordering::Relaxed);
    P.found.fetch_add(containers as u64, Ordering::Relaxed);
}

pub(crate) fn duplicate() {
    P.duplicates.fetch_add(1, Ordering::Relaxed);
}

/// The translator starts on `name`.
pub(crate) fn begin(name: impl FnOnce() -> String) {
    if let Ok(mut c) = P.current.try_lock() {
        *c = name();
    }
}

/// How the last translation `name` started ended.
pub(crate) fn outcome(result: &Result<(), String>, name: impl FnOnce() -> String) {
    P.done.fetch_add(1, Ordering::Relaxed);
    match result {
        Ok(()) => {
            P.translated.fetch_add(1, Ordering::Relaxed);
            return;
        }
        Err(e) if e.contains("is not supported") => P.refused.fetch_add(1, Ordering::Relaxed),
        Err(_) => P.failed.fetch_add(1, Ordering::Relaxed),
    };
    if let (Err(e), Ok(mut last)) = (result, P.last_failure.try_lock()) {
        *last = format!("{}: {e}", name());
    }
}

fn stats_file() -> Option<PathBuf> {
    let cfg = victory_core::Config::load().ok()?;
    Some(cfg.cache_subdir("warm").ok()?.join("last-warm.txt"))
}

fn read_stats(f: &Path) -> Option<(u64, u64)> {
    let text = std::fs::read_to_string(f).ok()?;
    let value = |key: &str| {
        text.lines()
            .find_map(|l| l.strip_prefix(key)?.strip_prefix('=')?.trim().parse().ok())
    };
    Some((value("done")?, value("secs")?))
}

fn elapsed() -> Duration {
    P.started.get().map_or(Duration::ZERO, Instant::elapsed)
}

fn cache_counts() -> (u64, u64) {
    victory_gpu::shader_cache::counts()
}

/// A snapshot for the display.
struct View {
    fraction: f64,
    /// `None` while there is nothing to go on.
    eta: Option<u64>,
    elapsed: u64,
    /// The previous warm's count: what `done` is heading for.
    expected: Option<u64>,
}

fn view() -> View {
    let elapsed = elapsed();
    let done = P.done.load(Ordering::Relaxed);
    let last = P.last_done.load(Ordering::Relaxed);
    let bytes_total = P.bytes_total.load(Ordering::Relaxed);
    let finished = P.finished.load(Ordering::Relaxed);
    // The previous warm's count says best how far this one is (the same
    // game, the same shaders); a first warm goes by the bytes read.
    let fraction = if finished {
        1.0
    } else if last > 0 {
        (done as f64 / last as f64).min(0.99)
    } else if bytes_total > 0 {
        (P.bytes_done.load(Ordering::Relaxed) as f64 / bytes_total as f64).min(0.99)
    } else {
        0.0
    };
    let secs = elapsed.as_secs_f64();
    let eta = (fraction > 0.01 && secs > 2.0 && !finished)
        .then(|| (secs * (1.0 - fraction) / fraction).round() as u64);
    View {
        fraction,
        eta,
        elapsed: elapsed.as_secs(),
        expected: (last > 0).then_some(last),
    }
}

fn clock(secs: u64) -> String {
    format!("{}:{:02}", secs / 60, secs % 60)
}

fn gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1e9)
}

fn console_line() -> String {
    let v = view();
    let width = 24;
    let filled = (v.fraction * width as f64).round() as usize;
    let (hits, new) = cache_counts();
    let total = v.expected.map_or(String::new(), |n| format!("/~{n}"));
    format!(
        "warm [{}{}] {:>3.0}%  {}{total} shaders  ok {} (cache {hits}, new {new})  refused {}  failed {}  {}  ETA {}",
        "#".repeat(filled),
        "-".repeat(width - filled),
        v.fraction * 100.0,
        P.done.load(Ordering::Relaxed),
        P.translated.load(Ordering::Relaxed),
        P.refused.load(Ordering::Relaxed),
        P.failed.load(Ordering::Relaxed),
        clock(v.elapsed),
        v.eta.map_or("--:--".into(), clock),
    )
}

/// The lines under the bar: label, text.
fn window_lines() -> Vec<(&'static str, String)> {
    let v = view();
    let n = |a: &AtomicU64| a.load(Ordering::Relaxed);
    let (hits, new) = cache_counts();
    let files_total = n(&P.files_total);
    let bytes_total = n(&P.bytes_total);
    let of = |done: String, total: u64, show: fn(u64) -> String| {
        if total > 0 {
            format!("{done} of {}", show(total))
        } else {
            done
        }
    };
    let current = P.current.lock().map(|c| c.clone()).unwrap_or_default();
    let last_failure = P.last_failure.lock().map(|c| c.clone()).unwrap_or_default();
    let last_run = match (n(&P.last_done), n(&P.last_secs)) {
        (0, _) => "first warm on this cache".to_owned(),
        (d, s) => format!("last warm: {d} shaders in {}", clock(s)),
    };
    vec![
        (
            "Shaders",
            format!(
                "{} done{}   ({} found, {} duplicates skipped)",
                n(&P.done),
                v.expected.map_or(String::new(), |e| format!(" of ~{e}")),
                n(&P.found),
                n(&P.duplicates)
            ),
        ),
        (
            "Result",
            format!(
                "{} translated   {} refused (not supported yet)   {} failed",
                n(&P.translated),
                n(&P.refused),
                n(&P.failed)
            ),
        ),
        (
            "Cache",
            format!("{hits} already cached   {new} newly translated"),
        ),
        (
            "Files",
            format!(
                "{} read   ({})",
                of(n(&P.files_done).to_string(), files_total, |t| t.to_string()),
                of(gb(n(&P.bytes_done)), bytes_total, gb)
            ),
        ),
        (
            "Time",
            format!(
                "{} elapsed   ETA {}   ({last_run})",
                clock(v.elapsed),
                v.eta.map_or("--:--".into(), clock)
            ),
        ),
        (
            "Host",
            format!(
                "wave {} lanes   {} worker threads",
                victory_shader::emit::host_max_wave(),
                n(&P.threads)
            ),
        ),
        ("Current", current),
        (
            "Last fail",
            if last_failure.is_empty() {
                "none".into()
            } else {
                last_failure
            },
        ),
    ]
}

#[cfg(windows)]
mod window {
    //! A plain Win32 window drawn with GDI on its own thread.

    use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering};

    use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM};
    use windows_sys::Win32::Graphics::Gdi::*;
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows_sys::Win32::UI::WindowsAndMessaging::*;

    /// The window, for [`close`] from the walk's thread (0: none).
    static HWND_: AtomicIsize = AtomicIsize::new(0);
    const DONE: u32 = WM_APP + 1;
    const WIDTH: i32 = 760;
    const HEIGHT: i32 = 360;

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain([0]).collect()
    }

    const fn rgb(r: u8, g: u8, b: u8) -> u32 {
        r as u32 | (g as u32) << 8 | (b as u32) << 16
    }

    pub(super) fn spawn() -> Option<std::thread::JoinHandle<()>> {
        let (tx, rx) = std::sync::mpsc::channel();
        let ui = std::thread::Builder::new()
            .name("warm-window".into())
            .spawn(move || {
                // SAFETY: plain Win32 calls on this thread's own window.
                unsafe {
                    let ok = create();
                    let _ = tx.send(ok);
                    if ok {
                        let mut msg = std::mem::zeroed::<MSG>();
                        while GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) > 0 {
                            TranslateMessage(&msg);
                            DispatchMessageW(&msg);
                        }
                    }
                }
            })
            .ok()?;
        // Without a window (a session with no desktop) the console line is all.
        if rx.recv().unwrap_or(false) {
            Some(ui)
        } else {
            let _ = ui.join();
            None
        }
    }

    pub(super) fn close() {
        let hwnd = HWND_.load(Ordering::Acquire);
        if hwnd != 0 {
            // SAFETY: posting to a window that may already be gone is harmless.
            unsafe { PostMessageW(hwnd as HWND, DONE, 0, 0) };
        }
    }

    unsafe fn create() -> bool {
        // SAFETY: the caller's; every pointer is to a live local.
        unsafe {
            let instance = GetModuleHandleW(std::ptr::null());
            let class = wide("VictoryWarm");
            let wc = WNDCLASSW {
                lpfnWndProc: Some(proc_),
                hInstance: instance,
                hCursor: LoadCursorW(std::ptr::null_mut(), IDC_ARROW),
                lpszClassName: class.as_ptr(),
                ..std::mem::zeroed()
            };
            RegisterClassW(&wc);
            let style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_VISIBLE;
            let mut r = RECT {
                left: 0,
                top: 0,
                right: WIDTH,
                bottom: HEIGHT,
            };
            AdjustWindowRect(&mut r, style, 0);
            let (w, h) = (r.right - r.left, r.bottom - r.top);
            let x = (GetSystemMetrics(SM_CXSCREEN) - w) / 2;
            let y = (GetSystemMetrics(SM_CYSCREEN) - h) / 2;
            let title = wide("VIctory: warming the shader cache");
            let hwnd = CreateWindowExW(
                0,
                class.as_ptr(),
                title.as_ptr(),
                style,
                x,
                y,
                w,
                h,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                instance,
                std::ptr::null(),
            );
            if hwnd.is_null() {
                return false;
            }
            HWND_.store(hwnd as isize, Ordering::Release);
            SetTimer(hwnd, 1, 250, None);
            SetForegroundWindow(hwnd);
            true
        }
    }

    /// The window's code panicked once: it is gone, and nothing of ours
    /// runs for its last messages.
    static BROKEN: AtomicBool = AtomicBool::new(false);

    unsafe extern "system" fn proc_(hwnd: HWND, msg: u32, w: WPARAM, l: LPARAM) -> LRESULT {
        // A panic must not leave this function: unwinding out of a system
        // callback ends the process (0xC000041D), and the warm with it.
        if !BROKEN.load(Ordering::Relaxed) {
            // SAFETY: called by the system with this thread's window.
            match std::panic::catch_unwind(|| unsafe { handle(hwnd, msg, w, l) }) {
                Ok(r) => return r,
                Err(_) => {
                    BROKEN.store(true, Ordering::Relaxed);
                    eprintln!("warm: the progress window failed and was closed; the warm goes on");
                    HWND_.store(0, Ordering::Release);
                    // SAFETY: this thread's window; its messages now go to
                    // the system's handler below.
                    unsafe {
                        DestroyWindow(hwnd);
                        PostQuitMessage(0);
                    }
                    return 0;
                }
            }
        }
        // SAFETY: the system's own handler, with the system's arguments.
        unsafe { DefWindowProcW(hwnd, msg, w, l) }
    }

    unsafe fn handle(hwnd: HWND, msg: u32, w: WPARAM, l: LPARAM) -> LRESULT {
        // SAFETY: the caller's: the system's call, with this thread's window.
        unsafe {
            match msg {
                WM_TIMER => {
                    InvalidateRect(hwnd, std::ptr::null(), 0);
                    0
                }
                WM_ERASEBKGND => 1,
                WM_PAINT => {
                    paint(hwnd);
                    0
                }
                // The user closing it only hides the progress; the warm goes on.
                DONE | WM_CLOSE => {
                    DestroyWindow(hwnd);
                    0
                }
                WM_DESTROY => {
                    HWND_.store(0, Ordering::Release);
                    PostQuitMessage(0);
                    0
                }
                _ => DefWindowProcW(hwnd, msg, w, l),
            }
        }
    }

    unsafe fn font(height: i32, bold: bool, face: &str) -> HFONT {
        let mono = face == "Consolas";
        let face = wide(face);
        // SAFETY: `face` lives across the call.
        unsafe {
            CreateFontW(
                height,
                0,
                0,
                0,
                if bold {
                    FW_BOLD as i32
                } else {
                    FW_NORMAL as i32
                },
                0,
                0,
                0,
                DEFAULT_CHARSET as u32,
                OUT_DEFAULT_PRECIS as u32,
                CLIP_DEFAULT_PRECIS as u32,
                CLEARTYPE_QUALITY as u32,
                if mono {
                    (FIXED_PITCH | FF_MODERN) as u32
                } else {
                    0
                },
                face.as_ptr(),
            )
        }
    }

    unsafe fn text(dc: HDC, s: &str, mut r: RECT, color: u32, format: u32) {
        // Nothing to draw: an empty `Vec`'s pointer is not one to hand the
        // system (the "Current" line is empty until the first shader starts).
        if s.is_empty() {
            return;
        }
        let s = wide(s);
        // SAFETY: `s` (terminated) and `r` live across the call.
        unsafe {
            SetTextColor(dc, color);
            DrawTextW(
                dc,
                s.as_ptr(),
                s.len() as i32 - 1,
                &mut r,
                format | DT_SINGLELINE | DT_NOPREFIX | DT_VCENTER | DT_END_ELLIPSIS,
            );
        }
    }

    unsafe fn fill(dc: HDC, r: RECT, color: u32) {
        // SAFETY: the brush is made, used and deleted here.
        unsafe {
            let b = CreateSolidBrush(color);
            FillRect(dc, &r, b);
            DeleteObject(b);
        }
    }

    unsafe fn paint(hwnd: HWND) {
        // SAFETY: the paint cycle of this thread's window; every GDI object
        // made here is deleted here.
        unsafe {
            let mut ps = std::mem::zeroed::<PAINTSTRUCT>();
            let screen = BeginPaint(hwnd, &mut ps);
            let mut client = std::mem::zeroed::<RECT>();
            GetClientRect(hwnd, &mut client);
            let (cw, ch) = (client.right, client.bottom);
            // Drawn off screen, then copied: no flicker at 4 frames a second.
            let dc = CreateCompatibleDC(screen);
            let bmp = CreateCompatibleBitmap(screen, cw, ch);
            let old_bmp = SelectObject(dc, bmp);
            SetBkMode(dc, TRANSPARENT as i32);
            fill(dc, client, rgb(0x16, 0x18, 0x1d));

            let big = font(26, true, "Segoe UI");
            let small = font(16, false, "Segoe UI");
            let mono = font(16, false, "Consolas");
            let mono_bold = font(16, true, "Consolas");
            let old_font = SelectObject(dc, big);
            let pad = 24;
            let row = |top: i32, h: i32| RECT {
                left: pad,
                top,
                right: cw - pad,
                bottom: top + h,
            };

            let v = super::view();
            let finished = super::P.finished.load(Ordering::Relaxed);
            text(
                dc,
                if finished {
                    "Shader cache ready"
                } else {
                    "Warming the shader cache"
                },
                row(16, 34),
                rgb(0xf2, 0xf2, 0xf2),
                DT_LEFT,
            );
            SelectObject(dc, small);
            text(
                dc,
                "The game starts when this finishes; the shaders it needs are then already translated.",
                row(50, 20),
                rgb(0x9a, 0xa0, 0xaa),
                DT_LEFT,
            );

            // The bar.
            let bar = row(86, 26);
            fill(dc, bar, rgb(0x2a, 0x2e, 0x36));
            let width = ((bar.right - bar.left) as f64 * v.fraction) as i32;
            let failing = super::P.failed.load(Ordering::Relaxed) > 0;
            let color = if failing {
                rgb(0xe0, 0xa0, 0x30)
            } else {
                rgb(0x4c, 0xaf, 0x50)
            };
            fill(
                dc,
                RECT {
                    right: bar.left + width,
                    ..bar
                },
                color,
            );
            SelectObject(dc, mono_bold);
            let pct = format!("{:.0}%", v.fraction * 100.0);
            text(
                dc,
                &pct,
                RECT {
                    left: bar.left + 8,
                    right: bar.right - 8,
                    ..bar
                },
                rgb(0xff, 0xff, 0xff),
                DT_RIGHT,
            );

            let mut top = 126;
            for (label, value) in super::window_lines() {
                SelectObject(dc, mono_bold);
                text(
                    dc,
                    label,
                    RECT {
                        right: pad + 100,
                        ..row(top, 24)
                    },
                    rgb(0x9a, 0xa0, 0xaa),
                    DT_LEFT,
                );
                SelectObject(dc, mono);
                text(
                    dc,
                    &value,
                    RECT {
                        left: pad + 100,
                        ..row(top, 24)
                    },
                    rgb(0xe6, 0xe6, 0xe6),
                    DT_LEFT,
                );
                top += 26;
            }

            BitBlt(screen, 0, 0, cw, ch, dc, 0, 0, SRCCOPY);
            SelectObject(dc, old_font);
            SelectObject(dc, old_bmp);
            for f in [big, small, mono, mono_bold] {
                DeleteObject(f);
            }
            DeleteObject(bmp);
            DeleteDC(dc);
            EndPaint(hwnd, &ps);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_last_warm_comes_back() {
        let f = std::env::temp_dir().join(format!("victory-warm-{}.txt", std::process::id()));
        std::fs::write(&f, "done=2900\nsecs=171\n").unwrap();
        assert_eq!(read_stats(&f), Some((2900, 171)));
        std::fs::write(&f, "done=2900\n").unwrap();
        assert_eq!(read_stats(&f), None);
        let _ = std::fs::remove_file(&f);
    }

    #[test]
    fn times_read_as_minutes_and_seconds() {
        assert_eq!(clock(0), "0:00");
        assert_eq!(clock(171), "2:51");
    }
}
