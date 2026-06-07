from __future__ import annotations

import numpy as np

from .core import (
    MosaicResult,
    create_mosaic,
    is_paper_info,
    recover_secret,
    update_paper_info,
)
from .crypto import decrypt, encrypt
from .rcm import embed, embed_multi, extract, extract_multi
from .io import resize_to_shape


def hide(
    secret: np.ndarray, target: np.ndarray, block: int, key: str, mode: str = "robust"
) -> tuple[np.ndarray, dict]:
    target = resize_to_shape(target, secret.shape[:2])
    core = create_mosaic(secret, target, block, mode=mode)
    encrypted = encrypt(core.metadata, key)
    if mode == "paper":
        mt_embedded, mt_stats = embed(core.image, encrypted)
        info = update_paper_info(
            core.info,
            encrypted_mt_bytes=len(encrypted),
            ni=1,
            npair=int(mt_stats["used_pair_count"]),
        )
        embedded, i_stats = embed(mt_embedded, info)
        stats = {
            **core.stats,
            "paper_i_bits": len(info) * 8,
            "paper_total_recovery_bits": core.stats["paper_mt_bits"] + len(info) * 8,
            "rcm_mt_used_pairs": mt_stats["used_pair_count"],
            "rcm_i_used_pairs": i_stats["used_pair_count"],
        }
        return embedded, {"core": stats, "rcm": {"mt": mt_stats, "i": i_stats}}
    embedded, rcm_stats = embed_multi(core.image, encrypted)
    return embedded, {"core": core.stats, "rcm": rcm_stats}


def reveal(mosaic: np.ndarray, key: str) -> tuple[np.ndarray, dict]:
    try:
        mt_mosaic, info, i_stats = extract(mosaic)
    except Exception:
        info = b""
    if info and is_paper_info(info):
        restored_mosaic, encrypted, mt_stats = extract(mt_mosaic)
        metadata = decrypt(encrypted, key)
        return recover_secret(restored_mosaic, metadata, info), {
            "rcm": {"mt": mt_stats, "i": i_stats}
        }
    restored_mosaic, encrypted, rcm_stats = extract_multi(mosaic)
    metadata = decrypt(encrypted, key)
    return recover_secret(restored_mosaic, metadata), {"rcm": rcm_stats}


def core_roundtrip(
    secret: np.ndarray, target: np.ndarray, block: int, mode: str = "robust"
) -> tuple[MosaicResult, np.ndarray]:
    target = resize_to_shape(target, secret.shape[:2])
    result = create_mosaic(secret, target, block, mode=mode)
    return result, recover_secret(result.image, result.metadata, result.info)
