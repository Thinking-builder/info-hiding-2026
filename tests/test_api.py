import numpy as np

from sfvmosaic.api import hide, reveal
from sfvmosaic.metrics import rmse
from sfvmosaic.rcm import CapacityError


def test_self_contained_hide_and_reveal_and_wrong_key():
    rng = np.random.default_rng(7)
    secret = rng.integers(70, 180, (64, 64, 3), dtype=np.uint8)
    target = rng.integers(70, 180, (64, 64, 3), dtype=np.uint8)
    mosaic, _ = hide(secret, target, 32, "correct")
    recovered, _ = reveal(mosaic, "correct")
    assert rmse(recovered, secret) < 1.0
    try:
        reveal(mosaic, "wrong")
    except Exception:
        pass
    else:
        raise AssertionError("wrong key was accepted")


def test_hide_resizes_target_to_secret_dimensions():
    secret = np.full((32, 32, 3), 100, dtype=np.uint8)
    target = np.full((16, 24, 3), 120, dtype=np.uint8)
    mosaic, _ = hide(secret, target, 16, "correct")
    recovered, _ = reveal(mosaic, "correct")
    assert mosaic.shape == secret.shape
    assert rmse(recovered, secret) < 1.0


def test_paper_hide_and_reveal_are_exact_for_embeddable_case():
    secret = np.full((64, 64, 3), 120, dtype=np.uint8)
    mosaic, stats = hide(secret, secret.copy(), 32, "correct", mode="paper")
    recovered, _ = reveal(mosaic, "correct")
    assert np.array_equal(recovered, secret)
    assert stats["core"]["paper_protocol"] is True
    assert stats["core"]["rcm_mt_used_pairs"] > 0
    try:
        reveal(mosaic, "wrong")
    except Exception:
        pass
    else:
        raise AssertionError("wrong key was accepted")


def test_paper_hide_uses_single_layer_and_reports_capacity_error():
    rng = np.random.default_rng(23)
    secret = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    target = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    try:
        hide(secret, target, 8, "correct", mode="paper")
    except CapacityError:
        pass
    else:
        raise AssertionError("paper mode unexpectedly fell back to multi-layer RCM")
