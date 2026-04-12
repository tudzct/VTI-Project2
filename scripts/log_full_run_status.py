#!/usr/bin/env python3
"""Log status for long-running full reproduction jobs."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import psutil


TARGETS = {
    "base": {
        "pid": 89900,
        "output_dir": "artifacts/base_full",
    },
    "word2vec": {
        "pid": 90288,
        "output_dir": "artifacts/word2vec_full",
    },
    "codebert": {
        "pid": 90372,
        "output_dir": "artifacts/codebert_full",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--max-iterations", type=int, default=0)
    parser.add_argument("--log-file", default="artifacts/full_run_status.log")
    return parser.parse_args()


def collect_snapshot() -> dict:
    snapshot = {"timestamp": datetime.now().isoformat(timespec="seconds"), "targets": {}}
    for name, spec in TARGETS.items():
        target = {"pid": spec["pid"]}
        try:
            proc = psutil.Process(spec["pid"])
            target["status"] = proc.status()
            target["cpu_percent"] = proc.cpu_percent(interval=0.2)
            target["memory_percent"] = round(proc.memory_percent(), 3)
            target["create_time"] = proc.create_time()
        except Exception as exc:
            target["error"] = str(exc)

        output_dir = Path(spec["output_dir"])
        if output_dir.exists():
            children = sorted(p.name for p in output_dir.iterdir())
            target["output_file_count"] = len(children)
            target["output_preview"] = children[:12]
        else:
            target["output_missing"] = True
        snapshot["targets"][name] = target
    return snapshot


def main() -> int:
    args = parse_args()
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    iteration = 0
    while True:
        snapshot = collect_snapshot()
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, ensure_ascii=True) + "\n")
        print(json.dumps(snapshot, ensure_ascii=True), flush=True)

        iteration += 1
        if args.max_iterations and iteration >= args.max_iterations:
            break
        time.sleep(args.interval_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
