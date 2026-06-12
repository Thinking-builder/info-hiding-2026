from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .api import core_roundtrip, hide, reveal
from .core import recover_secret
from .io import load_rgb, resize_to_shape, save_rgb
from .metrics import mssim, rmse


DEFAULT_VARIANTS = [
    {
        "name": "paper_baseline",
        "mode": "paper",
        "match_strategy": "avg_std",
        "q_strategy": "paper",
        "use_rotation": True,
        "residual_strategy": "default",
    },
    {
        "name": "robust_baseline",
        "mode": "robust",
        "match_strategy": "avg_std",
        "q_strategy": "paper",
        "use_rotation": True,
        "residual_strategy": "default",
    },
    {
        "name": "match_luminance",
        "mode": "paper",
        "match_strategy": "luminance_std",
        "q_strategy": "paper",
        "use_rotation": True,
        "residual_strategy": "default",
    },
    {
        "name": "match_entropy",
        "mode": "paper",
        "match_strategy": "entropy",
        "q_strategy": "paper",
        "use_rotation": True,
        "residual_strategy": "default",
    },
    {
        "name": "match_max_channel",
        "mode": "paper",
        "match_strategy": "max_channel_std",
        "q_strategy": "paper",
        "use_rotation": True,
        "residual_strategy": "default",
    },
    {
        "name": "q_global",
        "mode": "paper",
        "match_strategy": "avg_std",
        "q_strategy": "global",
        "use_rotation": True,
        "residual_strategy": "default",
    },
    {
        "name": "q_clamped",
        "mode": "paper",
        "match_strategy": "avg_std",
        "q_strategy": "clamped",
        "use_rotation": True,
        "residual_strategy": "default",
    },
    {
        "name": "ablate_rotation",
        "mode": "paper",
        "match_strategy": "avg_std",
        "q_strategy": "paper",
        "use_rotation": False,
        "residual_strategy": "default",
    },
    {
        "name": "ablate_boundary_residual",
        "mode": "paper",
        "match_strategy": "avg_std",
        "q_strategy": "paper",
        "use_rotation": True,
        "residual_strategy": "none",
    },
]


DEFAULT_NOISES = [
    {"name": "none", "kind": "none", "level": 0.0},
    {"name": "gaussian_2", "kind": "gaussian", "level": 2.0},
    {"name": "gaussian_5", "kind": "gaussian", "level": 5.0},
    {"name": "salt_pepper_0.001", "kind": "salt_pepper", "level": 0.001},
    {"name": "salt_pepper_0.005", "kind": "salt_pepper", "level": 0.005},
    {"name": "jpeg_95", "kind": "jpeg", "level": 95.0},
    {"name": "jpeg_85", "kind": "jpeg", "level": 85.0},
]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    value = rmse(a, b)
    if value == 0:
        return float("inf")
    return float(20 * np.log10(255.0 / value))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def pixel_error_rate(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.any(a != b, axis=2)))


def image_metrics(
    image: np.ndarray, reference: np.ndarray, block: int, prefix: str
) -> dict[str, float]:
    return {
        f"{prefix}_rmse": rmse(image, reference),
        f"{prefix}_psnr": psnr(image, reference),
        f"{prefix}_mae": mae(image, reference),
        f"{prefix}_mssim": mssim(image, reference, block),
    }


def add_noise(image: np.ndarray, kind: str, level: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "none":
        return image.copy()
    if kind == "gaussian":
        noisy = image.astype(np.float64) + rng.normal(0.0, level, image.shape)
        return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)
    if kind == "salt_pepper":
        noisy = image.copy()
        mask = rng.random(image.shape[:2])
        salt = mask < level / 2
        pepper = (mask >= level / 2) & (mask < level)
        noisy[salt] = 255
        noisy[pepper] = 0
        return noisy
    if kind == "jpeg":
        buffer = io.BytesIO()
        Image.fromarray(image, "RGB").save(buffer, format="JPEG", quality=int(level))
        buffer.seek(0)
        return np.asarray(Image.open(buffer).convert("RGB"), dtype=np.uint8)
    raise ValueError(f"unsupported noise kind: {kind}")


def run_eval_suite(
    secret_path: str,
    target_path: str,
    output: str,
    blocks: list[int],
    variants: list[dict] | None = None,
    noises: list[dict] | None = None,
    embedded: bool = True,
    key: str = "eval-suite-key",
    seed: int = 0,
) -> dict[str, object]:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    secret = load_rgb(secret_path)
    target = resize_to_shape(load_rgb(target_path), secret.shape[:2])
    variants = variants or DEFAULT_VARIANTS
    noises = noises or DEFAULT_NOISES

    quality_rows: list[dict[str, object]] = []
    noise_rows: list[dict[str, object]] = []
    embedded_rows: list[dict[str, object]] = []

    for block in blocks:
        if secret.shape[0] % block or secret.shape[1] % block:
            raise ValueError(f"block {block} must divide secret image dimensions")
        for variant in variants:
            row_base = _variant_row(block, variant)
            try:
                core, recovered = core_roundtrip(
                    secret,
                    target,
                    block,
                    mode=str(variant["mode"]),
                    match_strategy=str(variant["match_strategy"]),
                    q_strategy=str(variant["q_strategy"]),
                    use_rotation=bool(variant["use_rotation"]),
                    residual_strategy=str(variant["residual_strategy"]),
                )
                quality = {
                    **row_base,
                    "status": "ok",
                    **image_metrics(core.image, target, block, "mosaic_target"),
                    **image_metrics(recovered, secret, block, "recovered_secret"),
                    "recovered_pixel_error_rate": pixel_error_rate(recovered, secret),
                    **_selected_stats(core.stats),
                }
                quality_rows.append(quality)

                for noise_index, noise in enumerate(noises):
                    noisy = add_noise(
                        core.image,
                        str(noise["kind"]),
                        float(noise["level"]),
                        seed + noise_index,
                    )
                    try:
                        noisy_recovered = recover_secret(noisy, core.metadata, core.info)
                        noise_rows.append(
                            {
                                **row_base,
                                "noise": noise["name"],
                                "noise_kind": noise["kind"],
                                "noise_level": noise["level"],
                                "status": "ok",
                                **image_metrics(noisy, core.image, block, "carrier"),
                                **image_metrics(noisy_recovered, secret, block, "recovered_secret"),
                                "recovered_pixel_error_rate": pixel_error_rate(
                                    noisy_recovered, secret
                                ),
                            }
                        )
                    except Exception as exc:
                        noise_rows.append(
                            {
                                **row_base,
                                "noise": noise["name"],
                                "noise_kind": noise["kind"],
                                "noise_level": noise["level"],
                                "status": "failed",
                                "error": str(exc),
                            }
                        )

                if embedded and variant["name"] in {"paper_baseline", "robust_baseline"}:
                    embedded_rows.extend(
                        _embedded_noise_rows(secret, target, block, variant, noises, key, seed)
                    )
            except Exception as exc:
                quality_rows.append({**row_base, "status": "failed", "error": str(exc)})

    _write_csv(output_path / "quality.csv", quality_rows)
    _write_csv(output_path / "noise.csv", noise_rows)
    if embedded_rows:
        _write_csv(output_path / "embedded_noise.csv", embedded_rows)
    summary = {
        "secret": secret_path,
        "target": target_path,
        "shape": list(secret.shape),
        "blocks": blocks,
        "quality_rows": len(quality_rows),
        "noise_rows": len(noise_rows),
        "embedded_noise_rows": len(embedded_rows),
        "best_by_mosaic_rmse": _best_rows(quality_rows, "mosaic_target_rmse"),
        "best_by_recovery_rmse": _best_rows(quality_rows, "recovered_secret_rmse"),
    }
    (output_path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if quality_rows:
        _save_baseline_images(output_path, secret, target, int(blocks[0]), variants[0], key)
    return summary


def _embedded_noise_rows(
    secret: np.ndarray,
    target: np.ndarray,
    block: int,
    variant: dict,
    noises: list[dict],
    key: str,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    row_base = _variant_row(block, variant)
    try:
        mosaic, hide_stats = hide(secret, target, block, key, mode=str(variant["mode"]))
    except Exception as exc:
        return [{**row_base, "status": "embed_failed", "error": str(exc)}]
    for noise_index, noise in enumerate(noises):
        noisy = add_noise(mosaic, str(noise["kind"]), float(noise["level"]), seed + noise_index)
        try:
            recovered, _ = reveal(noisy, key)
            rows.append(
                {
                    **row_base,
                    "noise": noise["name"],
                    "noise_kind": noise["kind"],
                    "noise_level": noise["level"],
                    "status": "ok",
                    **image_metrics(noisy, mosaic, block, "carrier"),
                    **image_metrics(recovered, secret, block, "recovered_secret"),
                    "recovered_pixel_error_rate": pixel_error_rate(recovered, secret),
                    "rcm_payload_bits": _payload_bits(hide_stats),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    **row_base,
                    "noise": noise["name"],
                    "noise_kind": noise["kind"],
                    "noise_level": noise["level"],
                    "status": "extract_failed",
                    "error": str(exc),
                    "rcm_payload_bits": _payload_bits(hide_stats),
                }
            )
    return rows


def _payload_bits(stats: dict) -> int:
    rcm = stats.get("rcm", {})
    if "payload_bits" in rcm:
        return int(rcm["payload_bits"])
    return int(sum(layer.get("payload_bits", 0) for layer in rcm.values() if isinstance(layer, dict)))


def _variant_row(block: int, variant: dict) -> dict[str, object]:
    return {
        "block": block,
        "variant": variant["name"],
        "mode": variant["mode"],
        "match_strategy": variant["match_strategy"],
        "q_strategy": variant["q_strategy"],
        "use_rotation": variant["use_rotation"],
        "residual_strategy": variant["residual_strategy"],
    }


def _selected_stats(stats: dict[str, object]) -> dict[str, object]:
    keys = (
        "blocks",
        "residual_values",
        "metadata_bytes",
        "paper_mt_bits",
        "paper_i_bits",
        "paper_total_recovery_bits",
        "paper_protocol",
    )
    return {key: stats[key] for key in keys if key in stats}


def _best_rows(rows: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    ok_rows = [row for row in rows if row.get("status") == "ok" and key in row]
    return sorted(ok_rows, key=lambda row: float(row[key]))[:3]


def _save_baseline_images(
    output_path: Path,
    secret: np.ndarray,
    target: np.ndarray,
    block: int,
    variant: dict,
    key: str,
) -> None:
    try:
        core, core_recovered = core_roundtrip(
            secret,
            target,
            block,
            mode=str(variant["mode"]),
            match_strategy=str(variant["match_strategy"]),
            q_strategy=str(variant["q_strategy"]),
            use_rotation=bool(variant["use_rotation"]),
            residual_strategy=str(variant["residual_strategy"]),
        )
        mosaic, _ = hide(secret, target, block, key, mode=str(variant["mode"]))
        recovered, _ = reveal(mosaic, key)
    except Exception:
        return
    save_rgb(output_path / "baseline_secret.png", secret, rgba=True)
    save_rgb(output_path / "baseline_target.png", target, rgba=True)
    save_rgb(output_path / "baseline_core_mosaic.png", core.image, rgba=True)
    save_rgb(output_path / "baseline_core_recovered.png", core_recovered, rgba=True)
    save_rgb(output_path / "baseline_embedded_mosaic.png", mosaic, rgba=True)
    save_rgb(output_path / "baseline_mosaic.png", mosaic, rgba=True)
    save_rgb(output_path / "baseline_recovered.png", recovered, rgba=True)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
