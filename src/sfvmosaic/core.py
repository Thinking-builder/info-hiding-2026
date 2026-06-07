from __future__ import annotations

import base64
import json
import math
import struct
import zlib
from dataclasses import dataclass

import numpy as np

from . import huffman
from .bits import bits_to_bytes, bytes_to_bits


@dataclass
class MosaicResult:
    image: np.ndarray
    metadata: bytes
    stats: dict[str, int | float]
    info: bytes = b""


MODES = {"robust", "paper"}
PAPER_INFO_MAGIC = b"SFI1"
PAPER_INFO_HEADER = struct.Struct(">4sBIIHIBIIIIH")


def _validate(secret: np.ndarray, target: np.ndarray, block: int) -> None:
    if secret.shape != target.shape or secret.ndim != 3 or secret.shape[2] != 3:
        raise ValueError("secret and target must be equal-sized RGB images")
    if block <= 0 or secret.shape[0] % block or secret.shape[1] % block:
        raise ValueError(f"block size {block} must divide image dimensions")


def _blocks(image: np.ndarray, block: int) -> list[np.ndarray]:
    return [
        image[y : y + block, x : x + block]
        for y in range(0, image.shape[0], block)
        for x in range(0, image.shape[1], block)
    ]


class _BitWriter:
    def __init__(self) -> None:
        self.bits: list[int] = []

    def write(self, value: int, width: int) -> None:
        if width < 0:
            raise ValueError("bit width must be non-negative")
        if width and not 0 <= value < (1 << width):
            raise ValueError(f"value {value} does not fit in {width} bits")
        self.bits.extend((value >> shift) & 1 for shift in range(width - 1, -1, -1))

    def write_bits(self, bits: list[int]) -> None:
        self.bits.extend(bits)

    def bytes(self) -> bytes:
        padded = self.bits.copy()
        if len(padded) % 8:
            padded.extend([0] * (8 - len(padded) % 8))
        return bits_to_bytes(padded)


class _BitReader:
    def __init__(self, data: bytes, bit_length: int) -> None:
        self.bits = bytes_to_bits(data)[:bit_length]
        self.offset = 0

    def read(self, width: int) -> int:
        if self.offset + width > len(self.bits):
            raise ValueError("truncated paper metadata bitstream")
        value = 0
        for bit in self.bits[self.offset : self.offset + width]:
            value = (value << 1) | bit
        self.offset += width
        return value


def _pack_legacy_metadata(document: dict, residuals: list[int]) -> bytes:
    lengths, bits = huffman.encode(residuals)
    document["huffman_lengths"] = {str(k): v for k, v in lengths.items()}
    document["residual_count"] = len(residuals)
    document["residual_bit_length"] = len(bits)
    if len(bits) % 8:
        bits += [0] * (8 - len(bits) % 8)
    document["residual_data"] = base64.b64encode(bits_to_bytes(bits)).decode("ascii")
    return zlib.compress(json.dumps(document, separators=(",", ":")).encode("utf-8"), 9)


def _unpack_legacy_metadata(data: bytes) -> tuple[dict, list[int]]:
    document = json.loads(zlib.decompress(data))
    lengths = {int(k): int(v) for k, v in document.pop("huffman_lengths").items()}
    bit_length = int(document.pop("residual_bit_length"))
    encoded = base64.b64decode(document.pop("residual_data"))
    residuals = huffman.decode(
        lengths, bytes_to_bits(encoded)[:bit_length], int(document.pop("residual_count"))
    )
    return document, residuals


def _paper_residual_symbols(
    source: np.ndarray,
    source_mean: np.ndarray,
    target_mean: np.ndarray,
    q: np.ndarray,
    converted: np.ndarray,
) -> list[int]:
    """Paper Section III.C / Eq. (5): record |ci - cs| for overflow pixels and
    |cL - ci| for underflow pixels only.  Non-boundary pixels incur the small
    truncation/rounding loss described in the paper ("nearly lossless").
    Per-channel order: overflow residuals first, then underflow residuals."""
    residuals: list[int] = []
    for channel in range(3):
        values = source[:, :, channel].reshape(-1).astype(int)
        converted_ch = converted[:, :, channel].reshape(-1).astype(int)
        q_ch = float(q[channel])
        cs = math.ceil((255.0 - float(target_mean[channel])) / q_ch + float(source_mean[channel]))
        cl = math.floor((0.0 - float(target_mean[channel])) / q_ch + float(source_mean[channel]))
        over = converted_ch == 255
        under = converted_ch == 0
        residuals.extend(int(abs(int(v) - cs)) for v in values[over])
        residuals.extend(int(abs(cl - int(v))) for v in values[under])
    return residuals


def _decode_huffman_values(
    reader: _BitReader, lengths: dict[int, int], count: int
) -> list[int]:
    if count == 0:
        return []
    if not lengths:
        return [0] * count
    reverse = {
        (length, code): symbol
        for symbol, (code, length) in huffman.canonical_codes(lengths).items()
    }
    values: list[int] = []
    code = length = 0
    while len(values) < count:
        code = (code << 1) | reader.read(1)
        length += 1
        key = (length, code)
        if key in reverse:
            values.append(reverse[key])
            code = length = 0
    return values


def _pack_paper_info(
    *,
    height: int,
    width: int,
    block: int,
    block_count: int,
    index_bits: int,
    mt_bit_length: int,
    huffman_lengths: dict[int, int],
    encrypted_mt_bytes: int = 0,
    ni: int = 0,
    npair: int = 0,
) -> bytes:
    entries = sorted(huffman_lengths.items())
    if any(
        not 0 <= symbol <= 65535 or not 0 <= length <= 255
        for symbol, length in entries
    ):
        raise ValueError("Huffman table entry out of paper info range")
    header = PAPER_INFO_HEADER.pack(
        PAPER_INFO_MAGIC,
        1,
        height,
        width,
        block,
        block_count,
        index_bits,
        mt_bit_length,
        encrypted_mt_bytes,
        ni,
        npair,
        len(entries),
    )
    return header + b"".join(
        struct.pack(">HB", symbol, length) for symbol, length in entries
    )


def parse_paper_info(data: bytes) -> dict:
    if len(data) < PAPER_INFO_HEADER.size:
        raise ValueError("paper info is truncated")
    (
        magic,
        version,
        height,
        width,
        block,
        block_count,
        index_bits,
        mt_bit_length,
        encrypted_mt_bytes,
        ni,
        npair,
        entry_count,
    ) = PAPER_INFO_HEADER.unpack(data[: PAPER_INFO_HEADER.size])
    if magic != PAPER_INFO_MAGIC or version != 1:
        raise ValueError("invalid paper info")
    offset = PAPER_INFO_HEADER.size
    lengths: dict[int, int] = {}
    for _ in range(entry_count):
        if offset + 3 > len(data):
            raise ValueError("paper Huffman table is truncated")
        symbol, length = struct.unpack(">HB", data[offset : offset + 3])
        lengths[symbol] = length
        offset += 3
    if offset != len(data):
        raise ValueError("paper info has trailing data")
    return {
        "version": version,
        "height": height,
        "width": width,
        "block": block,
        "block_count": block_count,
        "index_bits": index_bits,
        "mt_bit_length": mt_bit_length,
        "encrypted_mt_bytes": encrypted_mt_bytes,
        "ni": ni,
        "npair": npair,
        "huffman_lengths": lengths,
    }


def update_paper_info(
    data: bytes, *, encrypted_mt_bytes: int, ni: int, npair: int
) -> bytes:
    info = parse_paper_info(data)
    return _pack_paper_info(
        height=info["height"],
        width=info["width"],
        block=info["block"],
        block_count=info["block_count"],
        index_bits=info["index_bits"],
        mt_bit_length=info["mt_bit_length"],
        huffman_lengths=info["huffman_lengths"],
        encrypted_mt_bytes=encrypted_mt_bytes,
        ni=ni,
        npair=npair,
    )


def is_paper_info(data: bytes) -> bool:
    return data.startswith(PAPER_INFO_MAGIC)


def _pack_paper_metadata(
    document: dict, records: list[dict], residuals_by_record: list[list[int]]
) -> tuple[bytes, bytes, int, int]:
    block_count = len(records)
    index_bits = math.ceil(math.log2(block_count)) if block_count > 1 else 0
    residuals = [value for tile in residuals_by_record for value in tile]
    lengths = (
        {}
        if residuals and all(value == 0 for value in residuals)
        else huffman.code_lengths(residuals)
    )
    codes = huffman.canonical_codes(lengths)
    writer = _BitWriter()
    residual_bit_length = 0
    for record, tile_residuals in zip(records, residuals_by_record):
        writer.write(int(record["target"]), index_bits)
        writer.write(int(record["rotation"]), 2)
        for value in record["source_mean"]:
            writer.write(int(value), 8)
        for value in record["target_mean"]:
            writer.write(int(value), 8)
        for value in record["qcode"]:
            writer.write(int(value) - 1, 7)
        for value in tile_residuals:
            if not lengths:
                continue
            code, length = codes[value]
            bits = [(code >> shift) & 1 for shift in range(length - 1, -1, -1)]
            writer.write_bits(bits)
            residual_bit_length += length
    info = _pack_paper_info(
        height=int(document["height"]),
        width=int(document["width"]),
        block=int(document["block"]),
        block_count=block_count,
        index_bits=index_bits,
        mt_bit_length=len(writer.bits),
        huffman_lengths=lengths,
    )
    return writer.bytes(), info, len(writer.bits), residual_bit_length


def _paper_bit_stats(
    records: list[dict], mt_bits: int, residual_bit_length: int, info: bytes
) -> dict[str, int | bool]:
    block_count = len(records)
    index_bits = math.ceil(math.log2(block_count)) if block_count > 1 else 0
    fixed_bits = block_count * (index_bits + 2 + 48 + 21)
    return {
        "paper_protocol": True,
        "paper_index_bits_per_block": index_bits,
        "paper_fixed_recovery_bits": fixed_bits,
        "paper_residual_huffman_bits": residual_bit_length,
        "paper_mt_bits": mt_bits,
        "paper_i_bits": len(info) * 8,
        "paper_total_recovery_bits": mt_bits + len(info) * 8,
        "paper_total_recovery_bits_estimate": mt_bits + len(info) * 8,
    }


def create_mosaic(
    secret: np.ndarray, target: np.ndarray, block: int, mode: str = "robust"
) -> MosaicResult:
    _validate(secret, target, block)
    if mode not in MODES:
        raise ValueError(f"unsupported mode: {mode}")
    secret = secret.astype(np.uint8)
    target = target.astype(np.uint8)
    secret_blocks, target_blocks = _blocks(secret, block), _blocks(target, block)
    secret_order = sorted(
        range(len(secret_blocks)),
        key=lambda i: (float(secret_blocks[i].std(axis=(0, 1)).mean()), i),
    )
    target_order = sorted(
        range(len(target_blocks)),
        key=lambda i: (float(target_blocks[i].std(axis=(0, 1)).mean()), i),
    )
    mapping = dict(zip(secret_order, target_order))
    output_blocks: list[np.ndarray | None] = [None] * len(target_blocks)
    records: list[dict] = []
    all_residuals: list[int] = []
    residuals_by_record: list[list[int]] = []

    for secret_index, source_u8 in enumerate(secret_blocks):
        target_index = mapping[secret_index]
        destination = target_blocks[target_index].astype(np.float64)
        source = source_u8.astype(np.float64)
        source_mean = np.rint(source.mean(axis=(0, 1))).astype(int)
        target_mean = np.rint(destination.mean(axis=(0, 1))).astype(int)
        source_std = source.std(axis=(0, 1))
        target_std = destination.std(axis=(0, 1))
        quotient = np.divide(
            target_std,
            source_std,
            out=np.full(3, 0.1),
            where=source_std > 1e-12,
        )
        qcode = np.clip(np.rint(quotient * 10), 1, 128).astype(int)
        q = qcode / 10.0
        transformed_real = q * (source - source_mean) + target_mean
        raw = np.trunc(transformed_real)
        converted = np.clip(raw, 0, 255).astype(np.uint8)

        paper_residuals = _paper_residual_symbols(
            source_u8, source_mean, target_mean, q, converted
        )
        if mode == "paper":
            tile_residuals = paper_residuals
        else:
            tile_residuals = []
            for channel in range(3):
                raw_channel = raw[:, :, channel].reshape(-1)
                source_channel = source_u8[:, :, channel].reshape(-1).astype(int)
                boundary = (raw_channel >= 255) | (raw_channel <= 0)
                tile_residuals.extend(source_channel[boundary].tolist())

        best_rmse, best_rotation, best_tile = None, 0, converted
        for rotation in range(4):
            candidate = np.rot90(converted, rotation)
            error = float(np.sqrt(np.mean((candidate.astype(float) - destination) ** 2)))
            if best_rmse is None or error < best_rmse:
                best_rmse, best_rotation, best_tile = error, rotation, candidate
        output_blocks[target_index] = best_tile
        record = {
            "target": target_index,
            "rotation": best_rotation,
            "source_mean": source_mean.tolist(),
            "target_mean": target_mean.tolist(),
            "qcode": qcode.tolist(),
        }
        if mode != "paper":
            record["residual_count"] = len(tile_residuals)
        records.append(record)
        all_residuals.extend(tile_residuals)
        residuals_by_record.append(tile_residuals)

    mosaic = np.empty_like(secret)
    columns = secret.shape[1] // block
    for index, tile in enumerate(output_blocks):
        y, x = divmod(index, columns)
        mosaic[y * block : (y + 1) * block, x * block : (x + 1) * block] = tile
    document = {
        "version": 1,
        "height": secret.shape[0],
        "width": secret.shape[1],
        "block": block,
        "mode": mode,
        "records": records,
    }
    if mode == "paper":
        metadata, info, mt_bits, residual_bit_length = _pack_paper_metadata(
            document, records, residuals_by_record
        )
        paper_stats = _paper_bit_stats(records, mt_bits, residual_bit_length, info)
    else:
        metadata = _pack_legacy_metadata(document, all_residuals)
        info = b""
        paper_stats = {"paper_protocol": False}
    return MosaicResult(
        mosaic,
        metadata,
        {
            "blocks": len(records),
            "residual_values": len(all_residuals),
            "metadata_bytes": len(metadata),
            **paper_stats,
        },
        info,
    )


def _recover_paper_secret(mosaic: np.ndarray, metadata: bytes, info_data: bytes) -> np.ndarray:
    info = parse_paper_info(info_data)
    block = int(info["block"])
    height, width = int(info["height"]), int(info["width"])
    if mosaic.shape != (height, width, 3):
        raise ValueError("mosaic dimensions do not match paper info")
    reader = _BitReader(metadata, int(info["mt_bit_length"]))
    mosaic_blocks = _blocks(mosaic.astype(np.uint8), block)
    output_blocks: list[np.ndarray] = []
    for _ in range(int(info["block_count"])):
        target = reader.read(int(info["index_bits"]))
        rotation = reader.read(2)
        source_mean = np.asarray([reader.read(8) for _ in range(3)], dtype=float)
        target_mean = np.asarray([reader.read(8) for _ in range(3)], dtype=float)
        q = np.asarray([reader.read(7) + 1 for _ in range(3)], dtype=float) / 10.0
        transformed = np.rot90(mosaic_blocks[target], -rotation)
        recovered = np.rint((transformed.astype(float) - target_mean) / q + source_mean)
        for channel in range(3):
            recovered_channel = recovered[:, :, channel].reshape(-1).copy()
            transformed_ch = transformed[:, :, channel].reshape(-1).astype(int)
            q_ch = float(q[channel])
            cs = math.ceil(
                (255.0 - float(target_mean[channel])) / q_ch + float(source_mean[channel])
            )
            cl = math.floor(
                (0.0 - float(target_mean[channel])) / q_ch + float(source_mean[channel])
            )
            over = transformed_ch == 255
            under = transformed_ch == 0
            over_count = int(over.sum())
            under_count = int(under.sum())
            if over_count:
                over_res = _decode_huffman_values(reader, info["huffman_lengths"], over_count)
                recovered_channel[over] = cs + np.asarray(over_res, dtype=float)
            if under_count:
                under_res = _decode_huffman_values(reader, info["huffman_lengths"], under_count)
                recovered_channel[under] = cl - np.asarray(under_res, dtype=float)
            recovered[:, :, channel] = recovered_channel.reshape(block, block)
        output_blocks.append(np.clip(recovered, 0, 255).astype(np.uint8))
    if reader.offset != int(info["mt_bit_length"]):
        raise ValueError("paper metadata bitstream was not fully consumed")
    secret = np.empty((height, width, 3), dtype=np.uint8)
    columns = width // block
    for index, tile in enumerate(output_blocks):
        y, x = divmod(index, columns)
        secret[y * block : (y + 1) * block, x * block : (x + 1) * block] = tile
    return secret


def recover_secret(mosaic: np.ndarray, metadata: bytes, info: bytes | None = None) -> np.ndarray:
    if info:
        return _recover_paper_secret(mosaic, metadata, info)
    document, residuals = _unpack_legacy_metadata(metadata)
    block = int(document["block"])
    mode = document.get("mode", "robust")
    height, width = int(document["height"]), int(document["width"])
    if mosaic.shape != (height, width, 3):
        raise ValueError("mosaic dimensions do not match metadata")
    mosaic_blocks = _blocks(mosaic.astype(np.uint8), block)
    output_blocks: list[np.ndarray] = []
    residual_offset = 0
    for record in document["records"]:
        transformed = np.rot90(
            mosaic_blocks[int(record["target"])], -int(record["rotation"])
        )
        source_mean = np.asarray(record["source_mean"], dtype=float)
        target_mean = np.asarray(record["target_mean"], dtype=float)
        q = np.asarray(record["qcode"], dtype=float) / 10.0
        recovered = np.rint((transformed.astype(float) - target_mean) / q + source_mean)
        count = int(record["residual_count"])
        tile_residuals = residuals[residual_offset : residual_offset + count]
        residual_offset += count
        ri = 0
        for channel in range(3):
            transformed_channel = transformed[:, :, channel].reshape(-1)
            recovered_channel = recovered[:, :, channel].reshape(-1).copy()
            over = transformed_channel == 255
            under = transformed_channel == 0
            boundary = over | under
            boundary_count = int(boundary.sum())
            values = np.asarray(tile_residuals[ri : ri + boundary_count], dtype=float)
            if mode == "paper":
                q_channel = q[channel]
                cs = math.ceil(
                    (255 - target_mean[channel]) / q_channel + source_mean[channel]
                )
                cl = math.floor(
                    (0 - target_mean[channel]) / q_channel + source_mean[channel]
                )
                recovered_channel[over] = cs + values[: int(over.sum())]
                recovered_channel[under] = cl - values[int(over.sum()) :]
            else:
                recovered_channel[boundary] = values
            recovered[:, :, channel] = recovered_channel.reshape(block, block)
            ri += boundary_count
        if ri != count:
            raise ValueError("residual stream does not match boundary pixels")
        output_blocks.append(np.clip(recovered, 0, 255).astype(np.uint8))
    secret = np.empty((height, width, 3), dtype=np.uint8)
    columns = width // block
    for index, tile in enumerate(output_blocks):
        y, x = divmod(index, columns)
        secret[y * block : (y + 1) * block, x * block : (x + 1) * block] = tile
    return secret
