"""
TÜRK — Türkçe Yapay Zeka DSL Python bağlantısı.

Rust `turk` binary'sini sarmalayarak script çalıştırma, statik analiz ve ONNX export sağlar.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

__version__ = "0.1.0"
__author__ = "SphereX"


def _is_turk_binary(path: Path) -> bool:
    """Yalnızca turk CLI binary'sini kabul eder (cargo/rustup değil)."""
    name = path.name.lower()
    blocked = {"cargo.exe", "cargo", "rustup.exe", "rustup", "rustc.exe", "rustc"}
    if name in blocked:
        return False
    return name in ("turk.exe", "turk") or name.startswith("turk.")


def _validate_binary(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"TURK binary bulunamadı: {path}")
    if not _is_turk_binary(path):
        raise FileNotFoundError(
            f"Geçersiz TURK_BIN: '{path}' — cargo/rustup değil, "
            f"target\\debug\\turk.exe yolunu verin."
        )
    return str(path.resolve())


def _search_turk_in_parents(start: Path) -> Optional[Path]:
    """Proje kökündeki target/debug/turk.exe dosyasını arar."""
    exe = "turk.exe" if sys.platform == "win32" else "turk"
    for base in [start, *start.parents]:
        candidate = base / "target" / "debug" / exe
        if candidate.exists() and _is_turk_binary(candidate):
            return candidate
        candidate = base / "target" / "release" / exe
        if candidate.exists() and _is_turk_binary(candidate):
            return candidate
    return None


def _default_project_binary() -> Optional[Path]:
    """python/ paketinin bir üst dizinindeki target/debug/turk.exe"""
    root = Path(__file__).resolve().parent.parent
    exe = "turk.exe" if sys.platform == "win32" else "turk"
    for sub in ("debug", "release"):
        candidate = root / "target" / sub / exe
        if candidate.exists() and _is_turk_binary(candidate):
            return candidate
    return None


def _smoke_test_binary(path: str) -> bool:
    """Binary'nin gerçekten TÜRK CLI olduğunu doğrular."""
    try:
        r = subprocess.run(
            [path, "--yardim"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        out = (r.stdout or "") + (r.stderr or "")
        return "TÜRK" in out or "TURK" in out.upper()
    except (OSError, subprocess.TimeoutExpired):
        return False


def _find_turk_binary(explicit: Optional[str] = None) -> str:
    """turk executable yolunu bulur."""
    if explicit:
        path = _validate_binary(Path(explicit))
        if not _smoke_test_binary(path):
            raise FileNotFoundError(f"Binary TÜRK CLI değil: {path}")
        return path

    env = os.environ.get("TURK_BIN")
    if env:
        p = Path(env)
        if p.exists() and _is_turk_binary(p):
            path = str(p.resolve())
            if _smoke_test_binary(path):
                return path
        # Yanlış TURK_BIN (cargo/rustup vb.) — yoksay, aşağıda ara
        print(
            f"[turk-py] UYARI: TURK_BIN geçersiz veya TÜRK değil, yoksayılıyor: {env}\n"
            f"          Doğrusu: ..\\target\\debug\\turk.exe",
            file=sys.stderr,
        )

    # Geliştirme: paket konumuna göre proje binary'si (en güvenilir)
    project = _default_project_binary()
    if project:
        return str(project.resolve())

    found = _search_turk_in_parents(Path.cwd())
    if found:
        path = str(found.resolve())
        if _smoke_test_binary(path):
            return path

    which = shutil.which("turk")
    if which and _is_turk_binary(Path(which)) and _smoke_test_binary(which):
        return which

    home_turk = Path.home() / ".cargo" / "bin" / ("turk.exe" if sys.platform == "win32" else "turk")
    if home_turk.exists() and _is_turk_binary(home_turk) and _smoke_test_binary(str(home_turk)):
        return str(home_turk.resolve())

    raise FileNotFoundError(
        "turk binary bulunamadı.\n"
        "  1) Proje kökünde: .\\cargo.ps1 build\n"
        "  2) Sonra tekrar deneyin (TURK_BIN gerekmez)\n"
        "  Not: TURK_BIN=cargo.exe veya rustup.exe YANLIŞ."
    )


class TurkRunner:
    """TÜRK CLI sarmalayıcısı."""

    def __init__(self, binary: Optional[str] = None):
        self.binary = _find_turk_binary(binary)

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        cmd = [self.binary, *args]
        cwd: Optional[str] = None
        for arg in args:
            p = Path(arg)
            if p.suffix in (".turk", ".json") and p.exists():
                cwd = str(p.resolve().parent)
                break
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=check,
            cwd=cwd,
        )

    def version(self) -> str:
        result = self._run("--yardim", check=False)
        for line in (result.stdout + result.stderr).splitlines():
            if "Sürüm:" in line or "Surum:" in line:
                return line.split(":", 1)[-1].strip()
        return __version__

    def run(self, script_path: Union[str, Path]) -> str:
        """`.turk` dosyasını çalıştırır."""
        path = str(Path(script_path).resolve())
        result = self._run("run", path)
        return result.stdout

    def check(self, script_path: Union[str, Path]) -> str:
        """Statik analiz (`turk check`)."""
        path = str(Path(script_path).resolve())
        result = self._run("check", path)
        return result.stdout

    def export_onnx(
        self,
        model_json: Union[str, Path],
        output: Union[str, Path] = "model.onnx.json",
    ) -> str:
        """JSON modeli ONNX JSON formatına aktarır."""
        result = self._run(
            "export",
            str(Path(model_json).resolve()),
            str(Path(output).resolve()),
        )
        return result.stdout

    def repl(self) -> None:
        """Etkileşimli REPL başlatır (stdout/stderr doğrudan terminale)."""
        subprocess.run([self.binary, "repl"], check=False)


def run(script_path: Union[str, Path], binary: Optional[str] = None) -> str:
    return TurkRunner(binary).run(script_path)


def check(script_path: Union[str, Path], binary: Optional[str] = None) -> str:
    return TurkRunner(binary).check(script_path)


def export_onnx(
    model_json: Union[str, Path],
    output: Union[str, Path] = "model.onnx.json",
    binary: Optional[str] = None,
) -> str:
    return TurkRunner(binary).export_onnx(model_json, output)


def load_model_summary(model_json: Union[str, Path]) -> Dict[str, Any]:
    """Kaydedilmiş TURK JSON modelinin özet bilgisini döndürür."""
    with open(model_json, encoding="utf-8") as f:
        data = json.load(f)
    layers: List[Dict[str, Any]] = data.get("layers", [])
    return {
        "layer_count": len(layers),
        "layers": [l.get("layer_type", "?") for l in layers],
        "compiled": data.get("compiled", False),
        "optimizer": data.get("optimizer"),
    }


# Geriye dönük uyumluluk — basit NumPy katmanları (offline demo)
try:
    import numpy as np

    class Dense:  # noqa: D101 — legacy API
        def __init__(self, input_size: int, output_size: int, activation: str = "relu"):
            self.input_size = input_size
            self.output_size = output_size
            self.activation = activation
            self.weights = np.random.randn(input_size, output_size) * 0.01
            self.biases = np.zeros((1, output_size))

        def forward(self, inputs: np.ndarray) -> np.ndarray:
            z = np.dot(inputs, self.weights) + self.biases
            if self.activation == "relu":
                return np.maximum(0, z)
            if self.activation == "sigmoid":
                return 1 / (1 + np.exp(-z))
            return z

    class Sequential:  # noqa: D101 — legacy API
        def __init__(self):
            self.layers: List[Dense] = []

        def add(self, layer: Dense) -> None:
            self.layers.append(layer)

        def predict(self, x: np.ndarray) -> np.ndarray:
            out = x
            for layer in self.layers:
                out = layer.forward(out)
            return out

except ImportError:
    Dense = None  # type: ignore
    Sequential = None  # type: ignore
