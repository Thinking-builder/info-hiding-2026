import numpy as np
from skimage.metrics import structural_similarity

from sfvmosaic.metrics import mssim


def test_mssim_identity_is_one():
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    assert mssim(image, image, 8) == 1.0


def test_mssim_averages_channels_independently():
    a = np.zeros((8, 8, 3), dtype=np.uint8)
    b = np.zeros_like(a)
    a[:, :, 0] = 255
    b[:, :, 1] = 255
    assert -1.0 <= mssim(a, b, 8) <= 1.0


def test_mssim_matches_skimage_for_one_block_per_channel():
    rng = np.random.default_rng(9)
    a = rng.integers(0, 256, (7, 7, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (7, 7, 3), dtype=np.uint8)
    reference = np.mean(
        [
            structural_similarity(
                a[:, :, channel],
                b[:, :, channel],
                data_range=255,
                win_size=7,
                use_sample_covariance=False,
            )
            for channel in range(3)
        ]
    )
    assert np.isclose(mssim(a, b, 7), reference)
