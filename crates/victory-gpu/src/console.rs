//! Which console the title runs on. Titles pick settings (resolution,
//! memory budgets) by it, so the user chooses; Series X unless told.

use std::sync::atomic::{AtomicBool, Ordering};

const GIB: usize = 1 << 30;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Console {
    #[default]
    SeriesX,
    SeriesS,
}

impl Console {
    /// Title memory in bytes: (GPU-optimal, standard). Series X in the
    /// advanced configuration; Series S has one range, counted as standard
    /// (7,776 MiB, the published console-mode figure).
    // shortcut: later system updates gave Series S titles more (amount not
    // published here); the title sizes nothing from this figure
    // (it goes by the device type), so update it if a title does.
    pub const fn title_memory(self) -> (usize, usize) {
        match self {
            Console::SeriesX => (10 * GIB, 7 * GIB / 2),
            Console::SeriesS => (0, 7_776 << 20),
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Console::SeriesX => "Xbox Series X",
            Console::SeriesS => "Xbox Series S",
        }
    }
}

static SERIES_S: AtomicBool = AtomicBool::new(false);

/// Set before the title starts.
pub fn set(console: Console) {
    SERIES_S.store(console == Console::SeriesS, Ordering::Relaxed);
}

pub fn current() -> Console {
    if SERIES_S.load(Ordering::Relaxed) {
        Console::SeriesS
    } else {
        Console::SeriesX
    }
}

/// The size the title renders at when the user asks for one:
/// `VICTORY_RENDER_SIZE=WxH`, or `1080p` / `1440p` / `2160p`. The model
/// stays as chosen (a Series X can run a title at 1080p: it is what the
/// title does on a 1080p television). The title takes its size from its
/// own saved video settings (`resolutionFullscreen` in its storage's
/// `settings/settings.xml`), read before it creates its window or asks
/// for the display mode, so the size is written there and also reported
/// as the display mode.
pub fn render_size() -> Option<(u32, u32)> {
    static SIZE: std::sync::OnceLock<Option<(u32, u32)>> = std::sync::OnceLock::new();
    *SIZE.get_or_init(|| {
        let asked = std::env::var("VICTORY_RENDER_SIZE").ok()?;
        let size = parse_size(&asked);
        match size {
            Some((width, height)) => tracing::info!(width, height, "render size asked for"),
            None => tracing::warn!(
                asked,
                "VICTORY_RENDER_SIZE not understood (WxH, 1080p, 1440p or 2160p)"
            ),
        }
        size
    })
}

/// `WxH` within the sizes a console outputs, or a height with `p` for the
/// 16:9 size of that height.
pub fn parse_size(text: &str) -> Option<(u32, u32)> {
    let text = text.trim();
    let size = match text.to_ascii_lowercase().as_str() {
        "720p" => (1280, 720),
        "1080p" => (1920, 1080),
        "1440p" => (2560, 1440),
        "2160p" | "4k" => (3840, 2160),
        _ => {
            let (w, h) = text.split_once(['x', 'X'])?;
            (w.trim().parse().ok()?, h.trim().parse().ok()?)
        }
    };
    ((640..=7680).contains(&size.0) && (360..=4320).contains(&size.1)).then_some(size)
}

#[cfg(test)]
mod size_tests {
    use super::parse_size;

    #[test]
    fn sizes_by_name_or_by_width_and_height() {
        assert_eq!(parse_size("1080p"), Some((1920, 1080)));
        assert_eq!(parse_size(" 1440P "), Some((2560, 1440)));
        assert_eq!(parse_size("1920x1080"), Some((1920, 1080)));
        assert_eq!(parse_size("2560 X 1440"), Some((2560, 1440)));
        assert_eq!(parse_size("1080"), None);
        assert_eq!(parse_size("10x10"), None);
    }
}
