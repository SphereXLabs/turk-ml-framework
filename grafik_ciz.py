#!/usr/bin/env python3
"""
TÜRK — Siberpunk Eğitim Grafiği
Kullanım: python grafik_ciz.py [--dosya egitim_gecmisi.json] [--cikti egitim_grafigi.png]
"""

import json
import argparse
import sys
import io
from pathlib import Path

# Windows terminali UTF-8'e zorla (box-drawing karakterleri için)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ─── Bağımlılık kontrolü ───────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")          # başsız sunucular için; pencereli ise "TkAgg" yap
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
except ImportError as e:
    print(f"\n[HATA] Eksik bağımlılık: {e}")
    print("Lütfen şu komutu çalıştırın:\n  pip install matplotlib numpy\n")
    sys.exit(1)

# ─── Renk paleti (siberpunk) ───────────────────────────────────────────────────
BG_DARK   = "#0d0f1a"   # gece mavisi arkaplan
BG_AXES   = "#111320"   # panel arkaplanı
NEON_YSL  = "#39ff14"   # neon yeşil  → train loss
NEON_PNK  = "#ff2d78"   # neon pembe  → val loss
NEON_BLU  = "#00e5ff"   # neon mavi   → vurgular
GRID_CLR  = "#1e2540"   # ızgara rengi
TICK_CLR  = "#5a6080"   # eksen tik rengi
TEXT_CLR  = "#c8d0f0"   # genel metin
TITLE_CLR = "#e040fb"   # başlık (mor neon)
GLOW      = 0.35        # parıltı alpha'sı

def json_oku(dosya: str) -> dict:
    p = Path(dosya)
    if not p.exists():
        print(f"[HATA] '{dosya}' bulunamadı. Önce `cargo run --example dsl_ornek` çalıştırın.")
        sys.exit(1)
    with p.open(encoding="utf-8") as f:
        return json.load(f)

def grafik_ciz(veri: dict, cikti: str, goster: bool) -> None:
    epochlar   = veri.get("epochlar", [])
    ozet       = veri.get("ozet", {})

    if not epochlar:
        print("[HATA] JSON içinde epoch verisi bulunamadı.")
        sys.exit(1)

    x          = [e["epoch"]                       for e in epochlar]
    train_loss = [e.get("train_loss") or 0.0       for e in epochlar]
    val_loss   = [e.get("val_loss")                for e in epochlar]
    has_val    = any(v is not None for v in val_loss)
    val_loss   = [v if v is not None else float("nan") for v in val_loss]

    toplam_epoch = ozet.get("toplam_epoch", x[-1])
    son_train    = ozet.get("son_train_loss", train_loss[-1])

    # ─── Figür & Eksen ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_AXES)

    # Parıltı efekti: aynı çizgiyi yarı şeffaf ve kalın çizerek
    def neon_cizgi(xs, ys, renk, etiket, lw=2.2, glow_lw=10, alpha=GLOW):
        ax.plot(xs, ys, color=renk, lw=glow_lw, alpha=alpha, zorder=2)
        ax.plot(xs, ys, color=renk, lw=lw, label=etiket, zorder=3, marker="o",
                markersize=3, markerfacecolor=renk, markeredgewidth=0)

    neon_cizgi(x, train_loss, NEON_YSL, "Train Loss")
    if has_val:
        neon_cizgi(x, val_loss,   NEON_PNK, "Validation Loss")

    # ─── Alan dolgusu ──────────────────────────────────────────────────────────
    ax.fill_between(x, train_loss, alpha=0.07, color=NEON_YSL, zorder=1)
    if has_val:
        ax.fill_between(x, val_loss, alpha=0.07, color=NEON_PNK, zorder=1)

    # ─── Son nokta işaretçisi ──────────────────────────────────────────────────
    ax.annotate(
        f"  {son_train:.2e}",
        xy=(x[-1], train_loss[-1]),
        fontsize=9, color=NEON_YSL,
        fontfamily="monospace",
    )

    # ─── Eksen biçimlendirme ───────────────────────────────────────────────────
    ax.set_xlabel("Epoch",       color=TEXT_CLR, fontsize=12, labelpad=8)
    ax.set_ylabel("Kayıp (MSE)", color=TEXT_CLR, fontsize=12, labelpad=8)

    ax.tick_params(colors=TICK_CLR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)

    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2e"))

    ax.grid(True, color=GRID_CLR, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_xlim(left=0)

    # ─── Başlık & Altyazı ──────────────────────────────────────────────────────
    ax.set_title(
        "◈  TÜRK — XOR Eğitim Kayıp Eğrisi  ◈",
        color=TITLE_CLR, fontsize=15, fontweight="bold", pad=18,
        fontfamily="monospace",
    )
    ax.text(
        0.5, -0.13,
        f"[ Toplam Epoch: {toplam_epoch} │ Son Train Loss: {son_train:.2e} ]",
        transform=ax.transAxes, ha="center",
        fontsize=9, color=NEON_BLU, fontfamily="monospace",
    )

    # ─── Legend ────────────────────────────────────────────────────────────────
    legend = ax.legend(
        frameon=True, facecolor=BG_DARK, edgecolor=GRID_CLR,
        labelcolor=TEXT_CLR, fontsize=10, loc="upper right",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # ─── Kaydet ────────────────────────────────────────────────────────────────
    fig.savefig(cikti, dpi=180, bbox_inches="tight", facecolor=BG_DARK)
    print(f"\n╔══════════════════════════════════════════════════════╗")
    print(f"║  ▸ Grafik '{cikti}' olarak kaydedildi!              ║")
    print(f"║    Epoch: {toplam_epoch:>4}  │  Son Train Loss: {son_train:.2e}        ║")
    print(f"╚══════════════════════════════════════════════════════╝\n")

    if goster:
        matplotlib.use("TkAgg")
        plt.show()

    plt.close(fig)

# ─── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="TÜRK eğitim geçmişini siberpunk grafik olarak çizer."
    )
    parser.add_argument(
        "--dosya",  "-d",
        default="egitim_gecmisi.json",
        help="JSON giriş dosyası (varsayılan: egitim_gecmisi.json)",
    )
    parser.add_argument(
        "--cikti",  "-o",
        default="egitim_grafigi.png",
        help="Çıktı PNG dosyası (varsayılan: egitim_grafigi.png)",
    )
    parser.add_argument(
        "--goster", "-g",
        action="store_true",
        help="Grafiği ayrıca pencerede göster (TkAgg backend gerekir)",
    )
    args = parser.parse_args()

    print(f"[TÜRK] '{args.dosya}' okunuyor...")
    veri = json_oku(args.dosya)

    print(f"[TÜRK] {veri.get('ozet', {}).get('toplam_epoch', '?')} epoch verisi bulundu.")
    print(f"[TÜRK] Grafik çiziliyor...")

    grafik_ciz(veri, args.cikti, args.goster)

if __name__ == "__main__":
    main()
