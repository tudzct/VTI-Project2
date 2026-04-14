#!/usr/bin/env python3
"""Build Joern-aligned train/test views and run the Enhanced notebook-compatible stage."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vti_repro.constants import LABEL_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-name", required=True)
    parser.add_argument("--train-source", required=True)
    parser.add_argument("--test-source", required=True)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--raw-preds")
    parser.add_argument("--labels")
    parser.add_argument("--test-predictions")
    parser.add_argument("--prediction-threshold", type=float, default=0.3)
    parser.add_argument("--freq-threshold", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extension", default=".cpp")
    parser.add_argument("--output-root", default="artifacts/joern_runs")
    parser.add_argument("--enhanced-output")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--joern-home",
        default="tools/joern/install/joern-cli",
        help="Joern CLI directory containing joern and joern-parse executables.",
    )
    return parser.parse_args()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _subset_frame(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=seed).sort_values("sample_id").reset_index(drop=True)


def _write_aligned_view(src: Path, dst: Path, limit: int, seed: int, force: bool) -> pd.DataFrame:
    if dst.exists() and not force:
        return pd.read_csv(dst)
    df = pd.read_csv(src, compression="infer")
    out = _subset_frame(df, limit, seed)
    _ensure_parent(dst)
    out.to_csv(dst, index=False)
    return out


def _run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _prepare_predictions(args: argparse.Namespace, bundle_dir: Path, cwd: Path) -> tuple[Path, Path]:
    prediction_dir = bundle_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    raw_preds_path = prediction_dir / "raw_preds.csv"
    labels_path = prediction_dir / "labels.csv"
    if raw_preds_path.exists() and labels_path.exists() and not args.force:
        return raw_preds_path, labels_path

    if args.test_predictions:
        cmd = [
            sys.executable,
            "scripts/split_prediction_table.py",
            "--input",
            args.test_predictions,
            "--output-dir",
            str(prediction_dir),
        ]
        _run(cmd, cwd=cwd)
        return raw_preds_path, labels_path

    if not args.raw_preds or not args.labels:
        raise ValueError("Provide either --test-predictions or both --raw-preds and --labels.")

    shutil.copy2(args.raw_preds, raw_preds_path)
    shutil.copy2(args.labels, labels_path)
    return raw_preds_path, labels_path


def _validate_alignment(test_df: pd.DataFrame, labels_path: Path) -> dict[str, object]:
    labels_df = pd.read_csv(labels_path)
    expected = test_df[LABEL_COLUMNS].astype(int).reset_index(drop=True)
    actual = labels_df[LABEL_COLUMNS].astype(int).reset_index(drop=True)
    labels_match = expected.equals(actual)
    return {
        "test_rows": int(len(test_df)),
        "label_rows": int(len(labels_df)),
        "labels_match": labels_match,
    }


def _materialize_sources(input_csv: Path, output_dir: Path, extension: str, cwd: Path, force: bool) -> Path:
    manifest_path = output_dir / "manifest.csv"
    if manifest_path.exists() and not force:
        return manifest_path
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    cmd = [
        sys.executable,
        "scripts/materialize_c_sources.py",
        "--input",
        str(input_csv),
        "--output-dir",
        str(output_dir),
        "--extension",
        extension,
    ]
    _run(cmd, cwd=cwd)
    return manifest_path


def _joern_env(cwd: Path) -> dict[str, str]:
    env = os.environ.copy()
    shim_dir = cwd / "tools" / "joern" / "shims"
    env["PATH"] = f"{shim_dir}:{env.get('PATH', '')}"
    return env


def _run_joern_parse(joern_home: Path, source_dir: Path, output_cpg: Path, cwd: Path, env: dict[str, str], force: bool) -> None:
    if output_cpg.exists() and not force:
        return
    if output_cpg.exists() and force:
        output_cpg.unlink()
    cmd = [
        str(joern_home / "joern-parse"),
        str(source_dir),
        "--language",
        "newc",
        "-o",
        str(output_cpg),
    ]
    _run(cmd, env=env, cwd=cwd)


def _dump_joern_nodes(joern_home: Path, cpg_path: Path, dump_path: Path, cwd: Path, env: dict[str, str], force: bool) -> None:
    if dump_path.exists() and not force:
        return
    if dump_path.exists() and force:
        dump_path.unlink()
    proc = subprocess.run(
        [
            str(joern_home / "joern"),
            str(cpg_path),
            "--script",
            "scripts/joern_dump_nodes.sc",
        ],
        cwd=str(cwd),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    dump_path.write_text(proc.stdout, encoding="utf-8")


def _build_cpg_csv(dump_path: Path, manifest_path: Path, source_csv: Path, output_csv: Path, cwd: Path, force: bool) -> None:
    if output_csv.exists() and not force:
        return
    if output_csv.exists() and force:
        output_csv.unlink()
    cmd = [
        sys.executable,
        "scripts/build_cpg_column_from_joern_dump.py",
        "--dump-tsv",
        str(dump_path),
        "--manifest",
        str(manifest_path),
        "--source-csv",
        str(source_csv),
        "--output-csv",
        str(output_csv),
    ]
    _run(cmd, cwd=cwd)


def _run_enhanced(
    train_cpg_csv: Path,
    test_cpg_csv: Path,
    raw_preds_path: Path,
    labels_path: Path,
    output_dir: Path,
    prediction_threshold: float,
    freq_threshold: float,
    cwd: Path,
    force: bool,
) -> None:
    if (output_dir / "metrics.json").exists() and not force:
        return
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    cmd = [
        sys.executable,
        "scripts/run_enhanced_notebook_compat.py",
        "--train",
        str(train_cpg_csv),
        "--test",
        str(test_cpg_csv),
        "--raw-preds",
        str(raw_preds_path),
        "--labels",
        str(labels_path),
        "--output-dir",
        str(output_dir),
        "--prediction-threshold",
        str(prediction_threshold),
        "--freq-threshold",
        str(freq_threshold),
    ]
    _run(cmd, cwd=cwd)


def main() -> int:
    args = parse_args()
    cwd = Path.cwd()
    bundle_dir = Path(args.output_root) / args.bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    views_dir = bundle_dir / "aligned_views"
    train_view_path = views_dir / "train.csv"
    test_view_path = views_dir / "test.csv"
    train_df = _write_aligned_view(Path(args.train_source), train_view_path, args.train_limit, args.seed, args.force)
    test_df = _write_aligned_view(Path(args.test_source), test_view_path, args.test_limit, args.seed, args.force)

    raw_preds_path, labels_path = _prepare_predictions(args, bundle_dir, cwd)
    alignment = _validate_alignment(test_df, labels_path)
    if alignment["test_rows"] != alignment["label_rows"]:
        raise ValueError(f"Label row count mismatch: {alignment}")
    if not alignment["labels_match"]:
        raise ValueError(f"Label values do not align with reconstructed test view: {alignment}")

    joern_dir = bundle_dir / "joern"
    train_sources_dir = joern_dir / "train_sources"
    test_sources_dir = joern_dir / "test_sources"
    train_manifest = _materialize_sources(train_view_path, train_sources_dir, args.extension, cwd, args.force)
    test_manifest = _materialize_sources(test_view_path, test_sources_dir, args.extension, cwd, args.force)

    env = _joern_env(cwd)
    joern_home = Path(args.joern_home)
    train_cpg_path = joern_dir / "train.cpg.bin"
    test_cpg_path = joern_dir / "test.cpg.bin"
    _run_joern_parse(joern_home, train_sources_dir, train_cpg_path, cwd, env, args.force)
    _run_joern_parse(joern_home, test_sources_dir, test_cpg_path, cwd, env, args.force)

    train_dump_path = joern_dir / "train_dump.tsv"
    test_dump_path = joern_dir / "test_dump.tsv"
    _dump_joern_nodes(joern_home, train_cpg_path, train_dump_path, cwd, env, args.force)
    _dump_joern_nodes(joern_home, test_cpg_path, test_dump_path, cwd, env, args.force)

    train_cpg_csv = joern_dir / "train_with_cpg.csv"
    test_cpg_csv = joern_dir / "test_with_cpg.csv"
    _build_cpg_csv(train_dump_path, train_manifest, train_view_path, train_cpg_csv, cwd, args.force)
    _build_cpg_csv(test_dump_path, test_manifest, test_view_path, test_cpg_csv, cwd, args.force)

    enhanced_output = Path(args.enhanced_output or f"artifacts/enhanced_{args.bundle_name}_joern_aligned")
    _run_enhanced(
        train_cpg_csv,
        test_cpg_csv,
        raw_preds_path,
        labels_path,
        enhanced_output,
        args.prediction_threshold,
        args.freq_threshold,
        cwd,
        args.force,
    )

    summary = {
        "bundle_name": args.bundle_name,
        "bundle_dir": str(bundle_dir),
        "enhanced_output": str(enhanced_output),
        "alignment": alignment,
        "train_view_rows": int(len(train_df)),
        "test_view_rows": int(len(test_df)),
        "raw_preds_path": str(raw_preds_path),
        "labels_path": str(labels_path),
        "train_cpg_csv": str(train_cpg_csv),
        "test_cpg_csv": str(test_cpg_csv),
    }
    (bundle_dir / "bundle_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
