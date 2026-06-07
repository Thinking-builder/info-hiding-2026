from sfvmosaic.huffman import decode, encode


def test_canonical_huffman_roundtrip():
    values = [0, 0, 0, 1, 2, 2, 5, 8, 8, 8, 8]
    lengths, bits = encode(values)
    assert decode(lengths, bits, len(values)) == values

