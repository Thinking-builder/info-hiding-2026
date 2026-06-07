import numpy as np

from sfvmosaic.core import create_mosaic, parse_paper_info, recover_secret
from sfvmosaic.metrics import rmse


def test_core_roundtrip_is_nearly_reversible():
    rng = np.random.default_rng(4)
    secret = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    target = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    result = create_mosaic(secret, target, 8)
    recovered = recover_secret(result.image, result.metadata)
    assert recovered.shape == secret.shape
    assert rmse(recovered, secret) < 1.0


def test_invalid_block_rejected():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    try:
        create_mosaic(image, image, 8)
    except ValueError as exc:
        assert "must divide" in str(exc)
    else:
        raise AssertionError("invalid block was accepted")


def test_paper_mode_reports_paper_bitstream_and_roundtrips():
    rng = np.random.default_rng(12)
    secret = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    target = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    result = create_mosaic(secret, target, 8, mode="paper")
    recovered = recover_secret(result.image, result.metadata, result.info)
    assert result.stats["paper_mt_bits"] > result.stats["paper_fixed_recovery_bits"]
    info = parse_paper_info(result.info)
    assert result.stats["paper_mt_bits"] == info["mt_bit_length"]
    assert 0 <= len(result.metadata) * 8 - result.stats["paper_mt_bits"] < 8
    assert result.stats["paper_total_recovery_bits"] == (
        result.stats["paper_mt_bits"] + result.stats["paper_i_bits"]
    )
    # Paper mode is nearly (not exactly) lossless: boundary pixels are corrected
    # via cs/cL residuals; non-boundary pixels accept small truncation/rounding error.
    assert rmse(recovered, secret) < 2.0


def test_zero_standard_deviation_and_saturated_boundaries():
    secret = np.zeros((16, 16, 3), dtype=np.uint8)
    secret[:, :, 0] = 128
    checker = (np.indices((16, 16)).sum(axis=0) % 2 * 255).astype(np.uint8)
    target = np.repeat(checker[:, :, None], 3, axis=2)
    result = create_mosaic(secret, target, 8)
    recovered = recover_secret(result.image, result.metadata)
    assert rmse(recovered, secret) < 1.0


def test_paper_mode_nearly_recovers_zero_std_and_boundaries():
    # Paper mode: boundary (overflow/underflow) pixels are exactly corrected via
    # cs/cL residuals; non-boundary pixels accept the small truncation/rounding
    # error inherent to the paper's "nearly lossless" design.
    secret = np.zeros((16, 16, 3), dtype=np.uint8)
    secret[:, :, 0] = 128
    secret[::2, :, 1] = 255
    checker = (np.indices((16, 16)).sum(axis=0) % 2 * 255).astype(np.uint8)
    target = np.repeat(checker[:, :, None], 3, axis=2)
    result = create_mosaic(secret, target, 8, mode="paper")
    recovered = recover_secret(result.image, result.metadata, result.info)
    assert rmse(recovered, secret) < 2.0
