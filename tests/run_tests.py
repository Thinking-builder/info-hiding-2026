"""Minimal test runner for environments where the local pytest executable crashes."""

from test_api import (
    test_hide_resizes_target_to_secret_dimensions,
    test_self_contained_hide_and_reveal_and_wrong_key,
)
from test_core import (
    test_core_roundtrip_is_nearly_reversible,
    test_invalid_block_rejected,
    test_paper_mode_reports_paper_bitstream_and_roundtrips,
    test_zero_standard_deviation_and_saturated_boundaries,
)
from test_huffman import test_canonical_huffman_roundtrip
from test_io import (
    test_experiment_resize_preserves_orientation_and_fixed_divisibility,
    test_target_resize_matches_secret_shape,
)
from test_metrics import (
    test_mssim_averages_channels_independently,
    test_mssim_identity_is_one,
    test_mssim_matches_skimage_for_one_block_per_channel,
)
from test_rcm import (
    test_multi_layer_rcm_roundtrip,
    test_rcm_capacity_error,
    test_rcm_restores_carrier_and_payload,
)


TESTS = [
    test_canonical_huffman_roundtrip,
    test_rcm_restores_carrier_and_payload,
    test_rcm_capacity_error,
    test_multi_layer_rcm_roundtrip,
    test_core_roundtrip_is_nearly_reversible,
    test_paper_mode_reports_paper_bitstream_and_roundtrips,
    test_zero_standard_deviation_and_saturated_boundaries,
    test_invalid_block_rejected,
    test_self_contained_hide_and_reveal_and_wrong_key,
    test_hide_resizes_target_to_secret_dimensions,
    test_mssim_identity_is_one,
    test_mssim_averages_channels_independently,
    test_mssim_matches_skimage_for_one_block_per_channel,
    test_experiment_resize_preserves_orientation_and_fixed_divisibility,
    test_target_resize_matches_secret_shape,
]


if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"PASS {test.__name__}")
