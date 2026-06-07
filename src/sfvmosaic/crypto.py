from __future__ import annotations

import hashlib
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def _derive_key(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32
    )


def encrypt(data: bytes, password: str) -> bytes:
    salt, nonce = os.urandom(16), os.urandom(12)
    return salt + nonce + AESGCM(_derive_key(password, salt)).encrypt(nonce, data, b"SFV1")


def decrypt(data: bytes, password: str) -> bytes:
    if len(data) < 44:
        raise ValueError("encrypted metadata is truncated")
    salt, nonce, ciphertext = data[:16], data[16:28], data[28:]
    return AESGCM(_derive_key(password, salt)).decrypt(nonce, ciphertext, b"SFV1")

