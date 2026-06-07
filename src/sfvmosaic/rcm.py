from __future__ import annotations

import numpy as np

from .bits import bits_to_bytes, bytes_to_bits, pack_frame, unpack_frame


class CapacityError(ValueError):
    pass


def _in_domain(x: int, y: int) -> bool:
    return 0 <= 2 * x - y <= 255 and 0 <= 2 * y - x <= 255


def _safe_odd(x: int, y: int) -> bool:
    """Return whether an odd pair is unambiguous after its LSBs are overwritten."""
    return x % 2 == 1 and y % 2 == 1 and all(
        _in_domain(x - dx, y - dy) for dx in (0, 1) for dy in (0, 1)
    )


def _embeddable(x: int, y: int) -> bool:
    return _in_domain(x, y) and not (x % 2 == 1 and y % 2 == 1 and not _safe_odd(x, y))


def inspect_capacity(image: np.ndarray) -> dict[str, int]:
    flat = image.transpose(2, 0, 1).reshape(-1).astype(np.uint8)
    pairs = len(flat) // 2
    slots = 0
    saved = 0
    for i in range(pairs):
        x, y = int(flat[2 * i]), int(flat[2 * i + 1])
        if _embeddable(x, y):
            slots += 1
        else:
            saved += 1
    return {"pairs": pairs, "slots": slots, "saved_lsb_bits": saved, "net_bits": slots - saved}


def embed_multi(image: np.ndarray, payload: bytes) -> tuple[np.ndarray, dict]:
    current = image
    remaining = payload
    layers = []
    index = 0
    while remaining:
        capacity = inspect_capacity(current)
        chunk_bytes = capacity["net_bits"] // 8 - 12 - 7
        if chunk_bytes <= 0:
            raise CapacityError(f"RCM layer {index} has no net payload capacity")
        chunk, remaining = remaining[:chunk_bytes], remaining[chunk_bytes:]
        body = b"MR1" + index.to_bytes(4, "big") + chunk
        current, stats = embed(current, body)
        layers.append(stats)
        index += 1
    return current, {
        "layers": layers,
        "layer_count": len(layers),
        "payload_bits": sum(layer["payload_bits"] for layer in layers),
    }


def embed(image: np.ndarray, payload: bytes) -> tuple[np.ndarray, dict[str, int]]:
    framed_bits = bytes_to_bits(pack_frame(payload))
    flat = image.transpose(2, 0, 1).reshape(-1).astype(np.uint8).copy()
    pairs = len(flat) // 2
    slots: list[int] = []
    saved: list[int] = []
    for i in range(pairs):
        j = 2 * i
        x, y = int(flat[j]), int(flat[j + 1])
        if _embeddable(x, y):
            if x % 2 and y % 2:
                flat[j] = x & 254
                slots.append(j + 1)
            else:
                xp, yp = 2 * x - y, 2 * y - x
                flat[j], flat[j + 1] = xp | 1, yp
                slots.append(j + 1)
        else:
            saved.append(x & 1)
            flat[j] = x & 254
    stream = framed_bits + saved
    if len(stream) > len(slots):
        raise CapacityError(
            f"RCM capacity insufficient: need {len(stream)} bits, have {len(slots)} "
            f"(payload={len(framed_bits)}, saved_lsb={len(saved)})"
        )
    for index, bit in zip(slots, stream):
        flat[index] = (flat[index] & 254) | bit
    stats = {
        "pairs": pairs,
        "slots": len(slots),
        "saved_lsb_bits": len(saved),
        "payload_bits": len(framed_bits),
        "used_pair_count": len(stream),
        "unused_slots": len(slots) - len(stream),
    }
    return flat.reshape(3, image.shape[0], image.shape[1]).transpose(1, 2, 0), stats


def extract(image: np.ndarray) -> tuple[np.ndarray, bytes, dict[str, int]]:
    flat = image.transpose(2, 0, 1).reshape(-1).astype(np.uint8).copy()
    extracted: list[int] = []
    nontransform: list[int] = []
    for i in range(len(flat) // 2):
        j = 2 * i
        xp, yp = int(flat[j]), int(flat[j + 1])
        if xp & 1:
            extracted.append(yp & 1)
            xp &= 254
            yp &= 254
            x = (2 * xp + yp + 2) // 3
            y = (xp + 2 * yp + 2) // 3
            flat[j], flat[j + 1] = x, y
        else:
            ox, oy = xp | 1, yp | 1
            if _safe_odd(ox, oy):
                extracted.append(yp & 1)
                flat[j], flat[j + 1] = ox, oy
            else:
                nontransform.append(j)
    if len(extracted) < 96:
        raise ValueError("not enough embedded RCM data")
    header = bits_to_bytes(extracted[:96])
    if header[:4] != b"SFV1":
        raise ValueError("invalid embedded RCM frame")
    length = int.from_bytes(header[4:8], "big")
    frame_bits = (12 + length) * 8
    if len(extracted) < frame_bits + len(nontransform):
        raise ValueError("truncated embedded RCM data")
    payload = unpack_frame(bits_to_bytes(extracted[:frame_bits]))
    saved = extracted[frame_bits : frame_bits + len(nontransform)]
    for index, bit in zip(nontransform, saved):
        flat[index] = (flat[index] & 254) | bit
    restored = flat.reshape(3, image.shape[0], image.shape[1]).transpose(1, 2, 0)
    return restored, payload, {
        "slots": len(extracted),
        "saved_lsb_bits": len(nontransform),
        "payload_bits": frame_bits,
        "used_pair_count": frame_bits + len(nontransform),
    }


def extract_multi(image: np.ndarray) -> tuple[np.ndarray, bytes, dict]:
    current = image
    chunks: list[bytes] = []
    layers = []
    expected_index = None
    while True:
        current, body, stats = extract(current)
        if len(body) < 7 or body[:3] != b"MR1":
            raise ValueError("invalid multi-layer RCM payload")
        index = int.from_bytes(body[3:7], "big")
        if expected_index is None:
            expected_index = index
        if index != expected_index:
            raise ValueError("RCM layer order is corrupt")
        chunks.append(body[7:])
        layers.append(stats)
        if index == 0:
            break
        expected_index -= 1
    return current, b"".join(reversed(chunks)), {
        "layers": layers,
        "layer_count": len(layers),
        "payload_bits": sum(layer["payload_bits"] for layer in layers),
    }
