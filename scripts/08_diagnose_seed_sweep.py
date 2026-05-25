"""Diagnose fold-level DL instability by sweeping random seeds.

This script has two modes:

1. Run mode: launch `scripts.04_train_dl` repeatedly for one fold and multiple
   experiment seeds, then summarize the created experiment folders.

   .venv\\Scripts\\python.exe -m scripts.08_diagnose_seed_sweep ^
       --config configs/exp_dl_multimodal_5fold.yaml --fold 4 ^
       --seeds 42,43,44,45,46 --run

2. Summarize-only mode: aggregate already-created experiment folders.

   .venv\\Scripts\\python.exe -m scripts.08_diagnose_seed_sweep ^
       --experiments outputs/experiments/2026-05-25_23-27_dl-multimodal-5fold

Outputs are written under a new `outputs/experiments/<ts>_seed-sweep-diagnosis/`
folder so the diagnosis itself follows the project experiment layout.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import yaml

from src.utils import make_experiment_dir


def _parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise ValueError("no seeds parsed")
    return seeds


def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _extract_experiment_dir(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("Experiment dir:"):
            rel = line.split(":", 1)[1].strip()
            return _resolve_path(rel)
    raise RuntimeError("could not find 'Experiment dir:' in training stdout")


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or float(a.std(ddof=0)) == 0.0 or float(b.std(ddof=0)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a.to_numpy(), b.to_numpy())[0, 1])


def _safe_slope(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2 or float(x.std(ddof=0)) == 0.0:
        return float("nan")
    return float(np.polyfit(x.to_numpy(), y.to_numpy(), 1)[0])


def _read_fold_metric(exp_dir: Path, target: str, fold: int) -> dict[str, Any]:
    metrics = json.loads((exp_dir / "metrics.json").read_text(encoding="utf-8"))
    rows = metrics[target]["per_fold"]
    for row in rows:
        if int(row["fold"]) == fold:
            return row
    raise KeyError(f"{exp_dir}: target={target!r}, fold={fold} not found in metrics.json")


def _summarize_experiment(exp_dir: Path, target: str, fold: int) -> tuple[dict[str, Any], pd.DataFrame]:
    metric = _read_fold_metric(exp_dir, target=target, fold=fold)
    pred_path = exp_dir / "logs" / "sample_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"missing sample predictions: {pred_path}")

    pred = pd.read_csv(pred_path)
    pred = pred[(pred["target"] == target) & (pred["fold"] == fold)].copy()
    if pred.empty:
        raise ValueError(f"{exp_dir}: no sample predictions for target={target}, fold={fold}")

    err = pred["y_pred"] - pred["y_true"]
    wafer = pred.groupby("experiment_key").agg(
        y_mean=("y_true", "mean"),
        pred_mean=("y_pred", "mean"),
    ).reset_index()

    row: dict[str, Any] = {
        "experiment_dir": str(exp_dir.relative_to(PROJECT_ROOT)),
        "target": target,
        "fold": fold,
        "rmse": float(metric["rmse"]),
        "mae": float(metric["mae"]),
        "r2": float(metric["r2"]),
        "mape_pct": float(metric["mape_pct"]),
        "best_epoch": int(metric.get("best_epoch", -1)),
        "fit_seconds": float(metric.get("fit_seconds", float("nan"))),
        "bias_pred_minus_true": float(err.mean()),
        "true_std": float(pred["y_true"].std(ddof=0)),
        "pred_std": float(pred["y_pred"].std(ddof=0)),
        "point_corr": _safe_corr(pred["y_true"], pred["y_pred"]),
        "wafer_mean_slope": _safe_slope(wafer["y_mean"], wafer["pred_mean"]),
        "wafer_mean_corr": _safe_corr(wafer["y_mean"], wafer["pred_mean"]),
        "true_mean_min": float(wafer["y_mean"].min()),
        "true_mean_max": float(wafer["y_mean"].max()),
        "pred_mean_min": float(wafer["pred_mean"].min()),
        "pred_mean_max": float(wafer["pred_mean"].max()),
        "train_target_mean": float(metric.get("train_target_mean", float("nan"))),
        "val_target_mean": float(metric.get("val_target_mean", float("nan"))),
        "target_mean_shift": float(metric.get("target_mean_shift", float("nan"))),
    }

    epoch_path = exp_dir / "logs" / "epoch_log.csv"
    if epoch_path.exists():
        ep = pd.read_csv(epoch_path)
        ep = ep[(ep["target"] == target) & (ep["fold"] == fold)].copy()
        if not ep.empty:
            best_idx = ep["val_rmse"].idxmin()
            row["train_rmse_at_best"] = float(ep.loc[best_idx, "train_rmse"])
            row["min_train_rmse"] = float(ep["train_rmse"].min())
            row["last_train_rmse"] = float(ep["train_rmse"].iloc[-1])

    worst = pred.groupby("experiment_key").apply(
        lambda g: pd.Series({
            "rmse": float(np.sqrt(np.mean((g["y_pred"] - g["y_true"]) ** 2))),
            "mae": float(np.mean(np.abs(g["y_pred"] - g["y_true"]))),
            "bias_pred_minus_true": float(np.mean(g["y_pred"] - g["y_true"])),
            "y_mean": float(g["y_true"].mean()),
            "pred_mean": float(g["y_pred"].mean()),
            "y_std": float(g["y_true"].std(ddof=0)),
            "pred_std": float(g["y_pred"].std(ddof=0)),
            "corr": _safe_corr(g["y_true"], g["y_pred"]),
        })
    ).reset_index()
    worst.insert(0, "experiment_dir", str(exp_dir.relative_to(PROJECT_ROOT)))
    worst.insert(1, "target", target)
    worst.insert(2, "fold", fold)
    worst = worst.sort_values("rmse", ascending=False)
    return row, worst


def _write_temp_config(base_config: Path, out_dir: Path, seed: int, fold: int) -> Path:
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    cfg["experiment"]["seed"] = int(seed)
    cfg["experiment"]["limit_folds"] = 1
    base_title = str(cfg["experiment"]["title"])
    cfg["experiment"]["title"] = f"{base_title} seed{seed} fold{fold}"

    temp_dir = out_dir / "logs" / "temp_configs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / f"seed_{seed}_fold_{fold}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _run_training(temp_config: Path, fold: int, log_path: Path) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "scripts.04_train_dl",
        "--config",
        str(temp_config.relative_to(PROJECT_ROOT)),
        "--folds",
        str(fold),
    ]
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        returncode = proc.wait()

    stdout = log_path.read_text(encoding="utf-8", errors="replace")
    if returncode != 0:
        raise RuntimeError(f"training failed with code {returncode}; see {log_path}")
    return _extract_experiment_dir(stdout)


def _write_outputs(
    *,
    exp_dir: Path,
    summary_rows: list[dict[str, Any]],
    worst_rows: list[pd.DataFrame],
    config_payload: dict[str, Any],
) -> None:
    summary = pd.DataFrame(summary_rows)
    summary_path = exp_dir / "logs" / "seed_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)

    worst = pd.concat(worst_rows, ignore_index=True) if worst_rows else pd.DataFrame()
    worst_path = exp_dir / "logs" / "worst_wafers.csv"
    worst.to_csv(worst_path, index=False)

    metrics = {
        "summary": summary_rows,
        "aggregate": {
            "rmse_mean": float(summary["rmse"].mean()) if not summary.empty else float("nan"),
            "rmse_std": float(summary["rmse"].std(ddof=0)) if not summary.empty else float("nan"),
            "r2_mean": float(summary["r2"].mean()) if not summary.empty else float("nan"),
            "r2_std": float(summary["r2"].std(ddof=0)) if not summary.empty else float("nan"),
            "pred_std_mean": float(summary["pred_std"].mean()) if not summary.empty else float("nan"),
            "wafer_mean_slope_mean": (
                float(summary["wafer_mean_slope"].mean()) if not summary.empty else float("nan")
            ),
        },
    }
    (exp_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (exp_dir / "config.yaml").write_text(
        yaml.safe_dump(config_payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    notes = [
        "\n## 자동 기록된 결과 (seed sweep diagnosis)\n",
        f"- Summary CSV: `{summary_path.relative_to(exp_dir)}`",
        f"- Worst wafer CSV: `{worst_path.relative_to(exp_dir)}`",
    ]
    if not summary.empty:
        notes.extend([
            f"- RMSE mean/std: {metrics['aggregate']['rmse_mean']:.4f} ± "
            f"{metrics['aggregate']['rmse_std']:.4f}",
            f"- R² mean/std: {metrics['aggregate']['r2_mean']:.4f} ± "
            f"{metrics['aggregate']['r2_std']:.4f}",
        ])
    with (exp_dir / "NOTES.md").open("a", encoding="utf-8") as f:
        f.write("\n".join(notes) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/exp_dl_multimodal_5fold.yaml"))
    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument("--target", default="oxide_etch")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--run", action="store_true", help="actually launch training jobs")
    parser.add_argument("--experiments", nargs="*", default=[],
                        help="existing experiment directories to summarize")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    out_dir = make_experiment_dir("seed sweep diagnosis")
    base_config = _resolve_path(args.config)
    shutil.copy(base_config, out_dir / "base_config.yaml")

    config_payload = {
        "diagnosis": {
            "base_config": str(base_config.relative_to(PROJECT_ROOT)),
            "target": args.target,
            "fold": args.fold,
            "seeds": seeds,
            "run_training": bool(args.run),
            "input_experiments": args.experiments,
        }
    }

    experiment_dirs = [_resolve_path(p) for p in args.experiments]
    summary_rows: list[dict[str, Any]] = []
    worst_rows: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []

    if args.run:
        for seed in seeds:
            print(f"[run] seed={seed} fold={args.fold}")
            temp_config = _write_temp_config(base_config, out_dir, seed=seed, fold=args.fold)
            log_path = out_dir / "logs" / f"train_seed_{seed}_fold_{args.fold}.log"
            try:
                train_exp = _run_training(temp_config, fold=args.fold, log_path=log_path)
                experiment_dirs.append(train_exp)
                print(f"  -> {train_exp.relative_to(PROJECT_ROOT)}")
            except Exception as exc:
                print(f"  !! failed: {exc}")
                failures.append({"seed": seed, "error": str(exc), "log": str(log_path)})
                if args.fail_fast:
                    raise

    if not experiment_dirs:
        raise SystemExit("No experiments to summarize. Pass --run or --experiments.")

    seen: set[Path] = set()
    for exp in experiment_dirs:
        exp = exp.resolve()
        if exp in seen:
            continue
        seen.add(exp)
        try:
            row, worst = _summarize_experiment(exp, target=args.target, fold=args.fold)
            summary_rows.append(row)
            worst_rows.append(worst)
            print(
                f"[summary] {exp.relative_to(PROJECT_ROOT)} "
                f"rmse={row['rmse']:.4f} r2={row['r2']:.4f} "
                f"pred_std={row['pred_std']:.4f} slope={row['wafer_mean_slope']:.3f}"
            )
        except Exception as exc:
            print(f"[summary] failed for {exp}: {exc}")
            failures.append({"experiment_dir": str(exp), "error": str(exc)})
            if args.fail_fast:
                raise

    config_payload["failures"] = failures
    _write_outputs(
        exp_dir=out_dir,
        summary_rows=summary_rows,
        worst_rows=worst_rows,
        config_payload=config_payload,
    )
    print(f"\nSaved diagnosis: {out_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
