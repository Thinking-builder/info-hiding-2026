from __future__ import annotations

import heapq
from collections import Counter


def code_lengths(values: list[int]) -> dict[int, int]:
    counts = Counter(values)
    if not counts:
        return {}
    if len(counts) == 1:
        return {next(iter(counts)): 1}
    heap = [(count, value, {value: 0}) for value, count in counts.items()]
    heapq.heapify(heap)
    serial = 256
    while len(heap) > 1:
        ca, _, la = heapq.heappop(heap)
        cb, _, lb = heapq.heappop(heap)
        merged = {k: v + 1 for k, v in (la | lb).items()}
        heapq.heappush(heap, (ca + cb, serial, merged))
        serial += 1
    return heap[0][2]


def canonical_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    code = 0
    previous = 0
    for symbol, length in sorted(lengths.items(), key=lambda item: (item[1], item[0])):
        code <<= length - previous
        result[symbol] = (code, length)
        code += 1
        previous = length
    return result


def encode(values: list[int]) -> tuple[dict[int, int], list[int]]:
    lengths = code_lengths(values)
    codes = canonical_codes(lengths)
    bits: list[int] = []
    for value in values:
        code, length = codes[value]
        bits.extend((code >> shift) & 1 for shift in range(length - 1, -1, -1))
    return lengths, bits


def decode(lengths: dict[int, int], bits: list[int], count: int) -> list[int]:
    reverse = {(length, code): symbol for symbol, (code, length) in canonical_codes(lengths).items()}
    values: list[int] = []
    code = length = 0
    for bit in bits:
        code = (code << 1) | bit
        length += 1
        key = (length, code)
        if key in reverse:
            values.append(reverse[key])
            if len(values) == count:
                return values
            code = length = 0
    if len(values) != count:
        raise ValueError("truncated Huffman stream")
    return values

