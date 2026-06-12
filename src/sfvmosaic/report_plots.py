from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_report_plots(
    experiment_dir: str = "outputs/experiment",
    eval_dir: str = "outputs/eval_suite",
    output: str = "outputs/report_figures",
) -> dict[str, object]:
    experiment_path = Path(experiment_dir)
    eval_path = Path(eval_dir)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    trends = _read_csv(experiment_path / "trends.csv")
    quality = _read_csv(eval_path / "quality.csv")
    noise = _read_csv(eval_path / "noise.csv")
    embedded_noise = _read_csv(eval_path / "embedded_noise.csv")

    generated = []
    if trends:
        generated.append(_plot_block_trends(trends, output_path / "fig1_block_trends.png"))
    if quality:
        generated.append(_plot_method_comparison(quality, output_path / "fig2_method_comparison.png"))
        generated.append(_plot_ablation(quality, output_path / "fig3_ablation.png"))
    if embedded_noise:
        generated.append(
            _plot_embedded_noise_summary(
                embedded_noise, output_path / "fig4_embedded_noise_summary.png"
            )
        )
    if noise:
        generated.append(_plot_noise_recovery(noise, output_path / "fig5_noise_recovery.png"))

    notes = _summarize(trends, quality, noise, embedded_noise)
    notes["figures"] = [str(path) for path in generated]
    (output_path / "figure_notes.json").write_text(
        json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return notes


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in {"", "inf"}:
        return float("inf") if value == "inf" else float("nan")
    return float(value)


def _ok(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("status", "ok") == "ok"]


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_block_trends(rows: list[dict[str, str]], path: Path) -> Path:
    rows = sorted(rows, key=lambda row: _number(row, "block"))
    blocks = [_number(row, "block") for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))

    _line(
        axes[0, 0],
        blocks,
        [_number(row, "core_mosaic_rmse") for row in rows],
        "Core mosaic",
        "Mosaic RMSE vs target",
        "RMSE",
    )
    if "embedded_mosaic_rmse" in rows[0]:
        _line(
            axes[0, 0],
            blocks,
            [_number(row, "embedded_mosaic_rmse") for row in rows],
            "Embedded mosaic",
            "Mosaic RMSE vs target",
            "RMSE",
            marker="s",
        )
        axes[0, 0].legend()

    _line(
        axes[0, 1],
        blocks,
        [_number(row, "core_mssim") for row in rows],
        "Core mosaic",
        "Mosaic MSSIM vs target",
        "MSSIM",
    )
    if "embedded_mssim" in rows[0]:
        _line(
            axes[0, 1],
            blocks,
            [_number(row, "embedded_mssim") for row in rows],
            "Embedded mosaic",
            "Mosaic MSSIM vs target",
            "MSSIM",
            marker="s",
        )
        axes[0, 1].legend()

    _line(
        axes[1, 0],
        blocks,
        [_number(row, "recovered_rmse") for row in rows],
        "Recovered secret",
        "Recovery RMSE vs secret",
        "RMSE",
    )
    _line(
        axes[1, 1],
        blocks,
        [_number(row, "paper_total_recovery_bits") / 1000 for row in rows],
        "Recovery payload",
        "Recovery payload size",
        "Kilobits",
    )
    for axis in axes.flat:
        axis.set_xscale("log", base=2)
        axis.set_xticks(blocks)
        axis.set_xticklabels([str(int(block)) for block in blocks])
        axis.set_xlabel("Block size")
        axis.grid(True, alpha=0.3)
    return _save(fig, path)


def _line(
    axis: plt.Axes,
    x: list[float],
    y: list[float],
    label: str,
    title: str,
    ylabel: str,
    marker: str = "o",
) -> None:
    axis.plot(x, y, marker=marker, linewidth=2, label=label)
    axis.set_title(title)
    axis.set_ylabel(ylabel)


def _plot_method_comparison(rows: list[dict[str, str]], path: Path) -> Path:
    rows = _ok(rows)
    rows = [row for row in rows if not row["variant"].startswith("ablate_")]
    labels = [row["variant"].replace("_", "\n") for row in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    mosaic_values = [_number(row, "mosaic_target_rmse") for row in rows]
    axes[0].bar(x, mosaic_values, color="#4C78A8")
    axes[0].set_title("Mosaic visual distortion")
    axes[0].set_ylabel("RMSE")
    _zoom_bar_axis(axes[0], mosaic_values, floor=30.0)

    axes[1].bar(x, [_number(row, "recovered_secret_rmse") for row in rows], color="#F58518")
    axes[1].set_title("Secret recovery error")
    axes[1].set_ylabel("RMSE")

    payload_values = [_number(row, "paper_total_recovery_bits") / 1000 for row in rows]
    axes[2].bar(x, [0 if np.isnan(value) else value for value in payload_values], color="#54A24B")
    for index, value in enumerate(payload_values):
        if np.isnan(value):
            axes[2].text(index, 0, "N/A", ha="center", va="bottom", fontsize=9)
    axes[2].set_title("Paper-mode recovery payload")
    axes[2].set_ylabel("Kilobits")

    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.grid(True, axis="y", alpha=0.3)
    return _save(fig, path)


def _zoom_bar_axis(axis: plt.Axes, values: list[float], floor: float) -> None:
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return
    lower = min(finite) - max((max(finite) - min(finite)) * 0.35, 0.5)
    lower = max(floor, lower)
    upper = max(finite) + max((max(finite) - min(finite)) * 0.25, 0.5)
    if lower < upper:
        axis.set_ylim(lower, upper)
        axis.text(
            0.01,
            0.02,
            f"y-axis starts at {lower:.1f}",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="#555555",
        )


def _plot_ablation(rows: list[dict[str, str]], path: Path) -> Path:
    rows_by_variant = {row["variant"]: row for row in _ok(rows)}
    names = ["paper_baseline", "ablate_rotation", "ablate_boundary_residual"]
    rows = [rows_by_variant[name] for name in names if name in rows_by_variant]
    labels = ["Baseline", "No rotation", "No boundary\nresidual"]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    metrics = [
        ("mosaic_target_rmse", "Mosaic RMSE"),
        ("recovered_secret_rmse", "Recovery RMSE"),
        ("paper_total_recovery_bits", "Payload bits"),
    ]
    for axis, (key, title) in zip(axes, metrics):
        values = [_number(row, key) for row in rows]
        if key == "paper_total_recovery_bits":
            values = [value / 1000 for value in values]
            axis.set_ylabel("Kilobits")
        else:
            axis.set_ylabel("RMSE")
        axis.bar(x, values)
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=15, ha="right")
        axis.grid(True, axis="y", alpha=0.3)
    return _save(fig, path)


def _plot_embedded_noise_summary(rows: list[dict[str, str]], path: Path) -> Path:
    variants = sorted({row["variant"] for row in rows})
    ok_counts = []
    fail_counts = []
    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        ok_counts.append(sum(row["status"] == "ok" for row in variant_rows))
        fail_counts.append(sum(row["status"] != "ok" for row in variant_rows))

    x = np.arange(len(variants))
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.bar(x, ok_counts, label="Success", color="#54A24B")
    axis.bar(x, fail_counts, bottom=ok_counts, label="Failed", color="#E45756")
    axis.set_title("Self-contained extraction robustness")
    axis.set_ylabel("Number of noise settings")
    axis.set_xticks(x)
    axis.set_xticklabels([name.replace("_", "\n") for name in variants])
    axis.legend()
    axis.grid(True, axis="y", alpha=0.3)
    for index, (ok_count, fail_count) in enumerate(zip(ok_counts, fail_counts)):
        total = ok_count + fail_count
        axis.text(index, total + 0.1, f"{ok_count}/{total}", ha="center", fontsize=9)
    return _save(fig, path)


def _plot_noise_recovery(rows: list[dict[str, str]], path: Path) -> Path:
    variants = sorted({row["variant"] for row in rows})
    ok_counts = []
    fail_counts = []
    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        ok_counts.append(sum(row["status"] == "ok" for row in variant_rows))
        fail_counts.append(sum(row["status"] != "ok" for row in variant_rows))

    ok_rows = _ok(rows)
    noise_labels = list(dict.fromkeys(row["noise"] for row in rows))
    plotted_variants = []
    for variant in variants:
        variant_ok = [row for row in ok_rows if row["variant"] == variant]
        if any(row["noise"] != "none" for row in variant_ok):
            plotted_variants.append(variant)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(variants))
    axes[0].bar(x, ok_counts, label="Recovered", color="#54A24B")
    axes[0].bar(x, fail_counts, bottom=ok_counts, label="Failed", color="#E45756")
    axes[0].set_title("External-metadata recovery outcomes")
    axes[0].set_ylabel("Number of noise settings")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([name.replace("_", "\n") for name in variants], rotation=25, ha="right")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)

    for variant in plotted_variants:
        values = []
        labels = []
        for noise in noise_labels:
            matches = [
                row
                for row in ok_rows
                if row["variant"] == variant and row["noise"] == noise
            ]
            if matches:
                labels.append(noise)
                values.append(_number(matches[0], "recovered_secret_rmse"))
        axes[1].plot(labels, values, marker="o", linewidth=2, label=variant)
    axes[1].set_title("Recovery error when decoding still succeeds")
    axes[1].set_ylabel("Recovered secret RMSE")
    axes[1].set_xlabel("Noise setting")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, alpha=0.3)
    if plotted_variants:
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "No noisy case recovered", ha="center", va="center")
    return _save(fig, path)


def _summarize(
    trends: list[dict[str, str]],
    quality: list[dict[str, str]],
    noise: list[dict[str, str]],
    embedded_noise: list[dict[str, str]],
) -> dict[str, object]:
    notes: dict[str, object] = {}
    if trends:
        sorted_trends = sorted(trends, key=lambda row: _number(row, "block"))
        notes["block_range"] = [int(_number(sorted_trends[0], "block")), int(_number(sorted_trends[-1], "block"))]
        notes["smallest_block_best_visual"] = min(
            sorted_trends, key=lambda row: _number(row, "core_mosaic_rmse")
        )
        notes["lowest_recovery_bits"] = min(
            sorted_trends, key=lambda row: _number(row, "paper_total_recovery_bits")
        )
    ok_quality = _ok(quality)
    if ok_quality:
        notes["best_visual_variant"] = min(
            ok_quality, key=lambda row: _number(row, "mosaic_target_rmse")
        )
        notes["best_recovery_variant"] = min(
            ok_quality, key=lambda row: _number(row, "recovered_secret_rmse")
        )
    if embedded_noise:
        notes["embedded_noise_success_rate"] = sum(
            row["status"] == "ok" for row in embedded_noise
        ) / len(embedded_noise)
    if noise:
        notes["external_metadata_noise_success_rate"] = sum(
            row.get("status") == "ok" for row in noise
        ) / len(noise)
    return notes
