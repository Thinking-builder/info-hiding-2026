import numpy as np

from sfvmosaic.rcm import CapacityError, embed, embed_multi, extract, extract_multi


def test_rcm_restores_carrier_and_payload():
    image = np.full((32, 32, 3), 120, dtype=np.uint8)
    image[::2] = 121
    marked, _ = embed(image, b"metadata")
    restored, payload, _ = extract(marked)
    assert payload == b"metadata"
    assert np.array_equal(restored, image)


def test_rcm_capacity_error():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    try:
        embed(image, b"x" * 1000)
    except CapacityError:
        pass
    else:
        raise AssertionError("capacity failure was not reported")


def test_multi_layer_rcm_roundtrip():
    image = np.full((64, 64, 3), 120, dtype=np.uint8)
    payload = bytes(range(256)) * 4
    marked, stats = embed_multi(image, payload)
    restored, extracted, recovered_stats = extract_multi(marked)
    assert stats["layer_count"] > 1
    assert recovered_stats["layer_count"] == stats["layer_count"]
    assert extracted == payload
    assert np.array_equal(restored, image)
