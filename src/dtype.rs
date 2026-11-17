//! Kasa genelinde sayısal hassasiyet.
//!
//! Varsayılan: `f64`. `--features f32` ile bellek yarıya iner ve SIMD yolu hızlanır.

/// Model tensörleri ve optimizer durumları için sayı tipi.
#[cfg(feature = "f32")]
pub type Float = f32;

/// Model tensörleri ve optimizer durumları için sayı tipi.
#[cfg(not(feature = "f32"))]
pub type Float = f64;

/// `Float` için sıfır sabiti.
pub const ZERO: Float = 0.0;
/// `Float` için bir sabiti.
pub const ONE: Float = 1.0;

/// DSL / JSON sınırında `Float` → `f64`.
#[inline]
pub fn to_f64(v: Float) -> f64 {
    #[cfg(feature = "f32")]
    {
        f64::from(v)
    }
    #[cfg(not(feature = "f32"))]
    {
        v
    }
}

/// DSL parametrelerinden `f64` → `Float`.
#[inline]
pub fn from_f64(v: f64) -> Float {
    #[cfg(feature = "f32")]
    {
        v as Float
    }
    #[cfg(not(feature = "f32"))]
    {
        v
    }
}
