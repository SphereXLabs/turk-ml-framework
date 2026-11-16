"""TÜRK Python CLI — turk-py komutu."""

from __future__ import annotations

import argparse
import subprocess
import sys

from turk import TurkRunner, __version__


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="turk-py",
        description="TÜRK DSL Python sarmalayıcısı (Rust turk binary gerekir)",
    )
    parser.add_argument("--binary", help="turk executable yolu (veya TURK_BIN)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help=".turk dosyası çalıştır")
    p_run.add_argument("script")

    p_check = sub.add_parser("check", help="statik analiz")
    p_check.add_argument("script")

    p_export = sub.add_parser("export", help="JSON → ONNX JSON")
    p_export.add_argument("model_json")
    p_export.add_argument("output", nargs="?", default="model.onnx.json")

    sub.add_parser("version", help="sürüm bilgisi")

    args = parser.parse_args()
    runner = TurkRunner(args.binary)

    try:
        if args.command == "run":
            print(runner.run(args.script), end="")
        elif args.command == "check":
            print(runner.check(args.script), end="")
        elif args.command == "export":
            print(runner.export_onnx(args.model_json, args.output), end="")
        elif args.command == "version":
            print(f"turk-py {__version__} | binary: {runner.binary}")
        return 0
    except subprocess.CalledProcessError as e:
        print(e.stderr or e.stdout or str(e), file=sys.stderr)
        return e.returncode
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
