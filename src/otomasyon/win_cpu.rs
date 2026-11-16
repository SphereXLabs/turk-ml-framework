//! Windows `GetSystemTimes` — Task Manager ile aynı yöntem (kernel+user tick farkı).

use std::sync::Mutex;
use std::time::{Duration, Instant};
use windows_sys::Win32::Foundation::FILETIME;
use windows_sys::Win32::System::Threading::GetSystemTimes;

const MIN_ORNEK_ARALIGI: Duration = Duration::from_millis(500);

struct TickOrnek {
    idle: u64,
    kernel: u64,
    user: u64,
    zaman: Instant,
}

fn filetime_u64(ft: &FILETIME) -> u64 {
    ((ft.dwHighDateTime as u64) << 32) | ft.dwLowDateTime as u64
}

static ONCEKI: Mutex<Option<TickOrnek>> = Mutex::new(None);

/// Tick farkından CPU yüzdesi; örnek aralığı dolmadıysa `None`.
pub fn cpu_yuzde_ornek() -> Option<f32> {
    let mut idle = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    let mut kernel = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    let mut user = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    if unsafe { GetSystemTimes(&mut idle, &mut kernel, &mut user) } == 0 {
        return None;
    }

    let now = Instant::now();
    let cur = TickOrnek {
        idle: filetime_u64(&idle),
        kernel: filetime_u64(&kernel),
        user: filetime_u64(&user),
        zaman: now,
    };

    let mut onceki = ONCEKI.lock().ok()?;
    let Some(prev) = onceki.as_ref() else {
        *onceki = Some(cur);
        return None;
    };
    if now.duration_since(prev.zaman) < MIN_ORNEK_ARALIGI {
        return None;
    }

    let idle_d = cur.idle.saturating_sub(prev.idle);
    let total_d = cur
        .kernel
        .saturating_sub(prev.kernel)
        .saturating_add(cur.user.saturating_sub(prev.user));

    *onceki = Some(cur);

    if total_d == 0 {
        return None;
    }

    let yuzde = 100.0 * (1.0 - idle_d as f64 / total_d as f64);
    Some(yuzde.clamp(0.0, 100.0) as f32)
}

/// İlk örnek noktasını kaydet (ısınma).
pub fn isin() {
    let mut idle = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    let mut kernel = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    let mut user = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    if unsafe { GetSystemTimes(&mut idle, &mut kernel, &mut user) } == 0 {
        return;
    }
    if let Ok(mut onceki) = ONCEKI.lock() {
        if onceki.is_none() {
            *onceki = Some(TickOrnek {
                idle: filetime_u64(&idle),
                kernel: filetime_u64(&kernel),
                user: filetime_u64(&user),
                zaman: Instant::now(),
            });
        }
    }
}
