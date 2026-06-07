from __future__ import annotations

import struct
import zlib


def bytes_to_bits(data: bytes) -> list[int]:
    return [((byte >> shift) & 1) for byte in data for shift in range(7, -1, -1)]


def bits_to_bytes(bits: list[int]) -> bytes:
    if len(bits) % 8:
        raise ValueError("bit length must be a multiple of 8")
    return bytes(
        sum(bits[i + j] << (7 - j) for j in range(8))
        for i in range(0, len(bits), 8)
    )


def pack_frame(payload: bytes) -> bytes:
    return b"SFV1" + struct.pack(">II", len(payload), zlib.crc32(payload)) + payload


def unpack_frame(data: bytes) -> bytes:
    if len(data) < 12 or data[:4] != b"SFV1":
        raise ValueError("invalid embedded frame")
    length, checksum = struct.unpack(">II", data[4:12])
    payload = data[12 : 12 + length]
    if len(payload) != length or zlib.crc32(payload) != checksum:
        raise ValueError("embedded frame is truncated or corrupt")
    return payload

