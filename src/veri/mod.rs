//! Veri yükleme ve akış modülleri — RAM dostu CSV okuma.

mod csv_akisi;

pub use csv_akisi::{CsvEpochOkuyucu, CsvSemasi, CsvVeriAkisi, csv_semasi_oku};
