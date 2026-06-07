import numpy as np

from sfvmosaic.io import resize_canonical, resize_to_shape


def test_experiment_resize_preserves_orientation_and_fixed_divisibility():
    landscape = np.zeros((40, 60, 3), dtype=np.uint8)
    portrait = np.zeros((60, 40, 3), dtype=np.uint8)
    assert resize_canonical(landscape).shape == (768, 1024, 3)
    assert resize_canonical(portrait).shape == (1024, 768, 3)
    for block in (8, 16, 32):
        assert 768 % block == 0
        assert 1024 % block == 0


def test_target_resize_matches_secret_shape():
    target = np.zeros((20, 30, 3), dtype=np.uint8)
    assert resize_to_shape(target, (32, 64)).shape == (32, 64, 3)
