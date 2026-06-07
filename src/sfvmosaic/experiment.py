from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .api import core_roundtrip, hide, reveal
from .io import load_rgb, resize_canonical, resize_to_shape, save_rgb
from .metrics import mssim, rmse
from .rcm import CapacityError


def demo(
    secret_path: str, target_path: str, output: str, block: int, key: str, mode: str = "robust"
) -> dict:
    output_path = Path(output)
    secret = load_rgb(secret_path)
    target = resize_to_shape(load_rgb(target_path), secret.shape[:2])
    core, core_recovered = core_roundtrip(secret, target, block, mode=mode)
    result = {
        "block": block,
        "mode": mode,
        "shape": list(secret.shape),
        "core_mosaic_target_rmse": rmse(core.image, target),
        "core_mosaic_target_mssim": mssim(core.image, target, block),
        "core_recovered_secret_rmse": rmse(core_recovered, secret),
        **{f"core_{k}": v for k, v in core.stats.items()},
    }
    save_rgb(output_path / "core_mosaic.png", core.image, rgba=True)
    save_rgb(output_path / "core_recovered.png", core_recovered, rgba=True)
    try:
        mosaic, hide_stats = hide(secret, target, block, key, mode=mode)
        recovered, extract_stats = reveal(mosaic, key)
        save_rgb(output_path / "mosaic.png", mosaic, rgba=True)
        save_rgb(output_path / "recovered.png", recovered, rgba=True)
        result.update(
            {
                "self_contained": True,
                "embedded_mosaic_target_rmse": rmse(mosaic, target),
                "embedded_mosaic_target_mssim": mssim(mosaic, target, block),
                "recovered_secret_rmse": rmse(recovered, secret),
                "wrong_key_rejected": _wrong_key_rejected(mosaic),
                "hide_stats": hide_stats,
                "extract_stats": extract_stats,
            }
        )
    except CapacityError as exc:
        result.update({"self_contained": False, "capacity_error": str(exc)})
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _wrong_key_rejected(mosaic: np.ndarray) -> bool:
    try:
        reveal(mosaic, "definitely-wrong-key")
        return False
    except Exception:
        return True


def run_trends(
    secret_path: str,
    target_path: str,
    output: str,
    blocks: list[int],
    mode: str = "paper",
    include_embedded: bool = True,
) -> list[dict]:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    secret = resize_canonical(load_rgb(secret_path))
    target = resize_to_shape(load_rgb(target_path), secret.shape[:2])
    rows = []
    for block in sorted(blocks):
        if secret.shape[0] % block or secret.shape[1] % block:
            raise ValueError(f"block {block} must divide fixed experiment dimensions")
        row, core = _evaluate(secret, target, block, mode, include_embedded)
        rows.append(row)
        save_rgb(output_path / f"mosaic_b{block}.png", core.image)
    _write_csv(output_path / "trends.csv", rows)
    _plot_trends(output_path / "fig8_trends.png", rows)
    return rows


def run_pairs(
    dataset: str,
    output: str,
    blocks: list[int] | None = None,
    limit: int = 12,
    mode: str = "paper",
    include_embedded: bool = False,
) -> list[dict]:
    blocks = sorted(blocks or [8, 16, 32])
    paths = sorted(Path(dataset).glob("*.png"))[:limit]
    if len(paths) < 2:
        raise ValueError("pair experiment requires at least two PNG images")
    images = [resize_canonical(load_rgb(path)) for path in paths]
    rows = []
    for block in blocks:
        for i, secret in enumerate(images):
            for j, source_target in enumerate(images):
                if i == j:
                    continue
                target = resize_to_shape(source_target, secret.shape[:2])
                row, _ = _evaluate(secret, target, block, mode, include_embedded)
                rows.append({"secret": paths[i].name, "target": paths[j].name, **row})
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_csv(output_path / "pairs.csv", rows)
    summary = {
        "mode": mode,
        "pairs_per_block": len(paths) * (len(paths) - 1),
        "blocks": {
            str(block): _summarize([row for row in rows if row["block"] == block])
            for block in blocks
        },
    }
    (output_path / "pairs_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_rows = [
        {
            "block": block,
            **{
                key: summary["blocks"][str(block)][key]["mean"]
                for key in (
                    "core_mosaic_rmse",
                    "recovered_rmse",
                    "core_mssim",
                    "paper_total_recovery_bits",
                    "paper_protocol_bits",
                )
            },
            "recovered_rmse_ge_1": summary["blocks"][str(block)]["recovered_rmse_ge_1"],
        }
        for block in blocks
    ]
    _write_csv(output_path / "pairs_trends.csv", summary_rows)
    _plot_trends(output_path / "fig8_pairs_trends.png", summary_rows)
    return rows


def _evaluate(
    secret: np.ndarray,
    target: np.ndarray,
    block: int,
    mode: str,
    include_embedded: bool,
) -> tuple[dict, object]:
    core, recovered = core_roundtrip(secret, target, block, mode=mode)
    row = {
        "block": block,
        "width": secret.shape[1],
        "height": secret.shape[0],
        "mode": mode,
        "core_mosaic_rmse": rmse(core.image, target),
        "core_mssim": mssim(core.image, target, block),
        "recovered_rmse": rmse(recovered, secret),
        "paper_protocol_bits": core.stats.get(
            "paper_total_recovery_bits", len(core.metadata) * 8
        ),
        "paper_mt_bits": core.stats["paper_mt_bits"],
        "paper_i_bits": core.stats.get("paper_i_bits", 0),
        "paper_total_recovery_bits": core.stats.get(
            "paper_total_recovery_bits", len(core.metadata) * 8
        ),
        "residual_values": core.stats["residual_values"],
    }
    if include_embedded:
        try:
            embedded, _ = hide(secret, target, block, "experiment-key", mode=mode)
            row.update(
                {
                    "embedded_available": True,
                    "embedded_mosaic_rmse": rmse(embedded, target),
                    "embedded_mssim": mssim(embedded, target, block),
                }
            )
        except CapacityError:
            row.update(
                {
                    "embedded_available": False,
                    "embedded_mosaic_rmse": "",
                    "embedded_mssim": "",
                }
            )
    return row, core


def _summarize(rows: list[dict]) -> dict:
    keys = [
        "core_mosaic_rmse",
        "recovered_rmse",
        "core_mssim",
        "paper_mt_bits",
        "paper_i_bits",
        "paper_total_recovery_bits",
        "paper_protocol_bits",
    ]
    result: dict[str, object] = {"pairs": len(rows)}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        result[key] = {
            "min": float(values.min()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
        }
    result["recovered_rmse_ge_1"] = sum(float(row["recovered_rmse"]) >= 1 for row in rows)
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_trends(path: Path, rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, key, title in zip(
        axes.flat,
        ("core_mosaic_rmse", "paper_total_recovery_bits", "recovered_rmse", "core_mssim"),
        ("Core mosaic RMSE", "Paper recovery bits (Mt + I)", "Recovered RMSE", "RGB block MSSIM"),
    ):
        axis.plot([r["block"] for r in rows], [r[key] for r in rows], marker="o")
        embedded_key = {
            "core_mosaic_rmse": "embedded_mosaic_rmse",
            "core_mssim": "embedded_mssim",
        }.get(key)
        if embedded_key and embedded_key in rows[0]:
            axis.lines[0].set_label("core")
            axis.plot(
                [r["block"] for r in rows],
                [r[embedded_key] for r in rows],
                marker="s",
                label="embedded",
            )
            axis.legend()
        axis.set_title(title)
        axis.set_xlabel("Block size")
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
