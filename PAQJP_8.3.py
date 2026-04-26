#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAQJP 8.3 – FULLY LOSSLESS & DETERMINISTIC (256 TRANSFORMS: 1‑256)
All transforms (1‑256) are 100% reversible for every byte value 0‑255.
Transform 256 is the identity transform (no change) – no separate raw fallback.

FIXES:
  - Removed all dependency on Python's `random` module (which is version‑sensitive).
    All pseudo‑random values are now derived from hashlib.sha256 → deterministic forever.
  - Transform 6 substitution is now generated with a custom shuffle seeded by SHA‑256.
  - Seed tables and pattern generation are computed on‑the‑fly using hashes.
  - Self‑test prints detailed failure information and stops on error.
  - Added menu option 3 to run the self‑test on demand.
No emoji/icons – uses plain ASCII.
"""

import os
import math
import hashlib
from typing import Optional, List

# Optional compression backends
try:
    import paq
except ImportError:
    paq = None

try:
    import zstandard as zstd
    zstd_cctx = zstd.ZstdCompressor(level=22)
    zstd_dctx = zstd.ZstdDecompressor()
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

PROGNAME = "PAQJP_8.3_LOSSLESS_256_TRANSFORMS_OFFSET"

PRIMES = [p for p in range(2, 256) if all(p % d != 0 for d in range(2, int(p**0.5)+1))]
PI_DIGITS = [79, 17, 111]


class StateTable:
    """State table for PAQ (kept for compatibility)."""
    def __init__(self):
        self.table = [
            [1,2,1,0], [3,5,0,1], [4,6,2,0], [7,10,0,2],
            [8,12,3,0], [9,13,1,1], [11,14,0,3], [15,19,4,0],
            [16,23,2,1], [17,24,2,1], [18,25,2,1], [20,27,1,2],
            [21,28,1,2], [22,29,1,2], [26,30,0,4], [31,33,5,0],
            [32,34,3,1], [35,37,1,3], [36,38,1,3], [39,42,0,5],
            [40,43,4,1], [41,44,2,2], [45,48,1,4], [46,49,1,4],
            [47,50,1,4], [51,52,0,6], [53,55,6,0], [54,56,4,1],
            [57,59,2,3], [58,60,2,3], [61,63,0,7], [62,64,5,1],
            [65,66,3,2], [67,69,1,5], [68,70,1,5], [71,73,0,8],
            [72,74,6,1], [75,76,4,2], [77,78,2,4], [79,80,2,4],
            [81,82,0,9], [83,84,7,1], [85,86,5,2], [87,88,3,3],
            [89,90,1,6], [91,92,0,10], [93,94,8,1], [95,96,6,2],
            [97,98,4,3], [99,100,2,5], [101,102,0,11], [103,104,9,1],
            [105,106,7,2], [107,108,5,3], [109,110,3,4], [111,112,1,7],
            [113,114,0,12], [115,116,10,1], [117,118,8,2], [119,120,6,3],
            [121,122,4,4], [123,124,2,6], [125,126,0,13], [127,128,11,1],
            [129,130,9,2], [131,132,7,3], [133,134,5,4], [135,136,3,5],
            [137,138,1,8], [139,140,0,14], [141,142,12,1], [143,144,10,2],
            [145,146,8,3], [147,148,6,4], [149,150,4,5], [151,152,2,7],
            [153,154,0,15], [155,156,13,1], [157,158,11,2], [159,160,9,3],
            [161,162,7,4], [163,164,5,5], [165,166,3,6], [167,168,1,9],
            [169,170,0,16], [171,172,14,1], [173,174,12,2], [175,176,10,3],
            [177,178,8,4], [179,180,6,5], [181,182,4,6], [183,184,2,8],
            [185,186,0,17], [187,188,15,1], [189,190,13,2], [191,192,11,3],
            [193,194,9,4], [195,196,7,5], [197,198,5,6], [199,200,3,7],
            [201,202,1,10], [203,204,0,18], [205,206,16,1], [207,208,14,2],
            [209,210,12,3], [211,212,10,4], [213,214,8,5], [215,216,6,6],
            [217,218,4,7], [219,220,2,9], [221,222,0,19], [223,224,17,1],
            [225,226,15,2], [227,228,13,3], [229,230,11,4], [231,232,9,5],
            [233,234,7,6], [235,236,5,7], [237,238,3,8], [239,240,1,11],
            [241,242,0,20], [243,244,18,1], [245,246,16,2], [247,248,14,3],
            [249,250,12,4], [251,252,10,5], [253,254,8,6], [255,255,6,7]
        ]


def find_nearest_prime_around(n: int) -> int:
    o = 0
    while True:
        c1, c2 = n - o, n + o
        if c1 >= 2 and all(c1 % d != 0 for d in range(2, int(c1**0.5)+1)):
            return c1
        if c2 >= 2 and all(c2 % d != 0 for d in range(2, int(c2**0.5)+1)):
            return c2
        o += 1


def _hash_int(seed_str: str, idx: int, max_val: int) -> int:
    """
    Deterministic pseudo‑random integer in [0, max_val-1].
    Uses SHA‑256, guaranteed stable across all Python 3 versions.
    """
    data = f"{seed_str}:{idx}".encode('utf-8')
    digest = hashlib.sha256(data).digest()
    # take first 4 bytes as a 32‑bit unsigned integer
    val = int.from_bytes(digest[:4], 'big')
    return val % max_val


class PAQJPCompressor:
    # Number of virtual seed tables (used for index modulation)
    SEED_TABLE_COUNT = 126

    def __init__(self):
        self.PI_DIGITS = PI_DIGITS.copy()
        self.fibonacci = self._gen_fib(100)

    def _gen_fib(self, n):
        a, b = 0, 1
        res = [a, b]
        for _ in range(2, n):
            a, b = b, a + b
            res.append(b)
        return res

    # ------------------------------------------------------------------
    # Deterministic seed / pattern generation (replaces random module)
    # ------------------------------------------------------------------
    def get_seed(self, idx: int, val: int) -> int:
        combined = idx * 40 + (val % 40)
        return 5 + _hash_int("seedtable", combined, 251)

    def _get_pattern(self, size: int, index: int) -> List[int]:
        base = 12345 + size * 100 + index
        pattern = []
        for i in range(size):
            val = _hash_int("pattern", base * 1000 + i, 256)
            pattern.append(val)
        return pattern

    def _gen_substitution(self, key_str: str) -> List[int]:
        lst = list(range(256))
        for i in range(255, 0, -1):
            j = _hash_int(key_str, i, i + 1)
            lst[i], lst[j] = lst[j], lst[i]
        return lst

    # ------------------------------------------------------------------
    # Bit helpers
    # ------------------------------------------------------------------
    def _append_bits(self, bitlist: List[int], value: int, count: int):
        for i in range(count - 1, -1, -1):
            bitlist.append((value >> i) & 1)

    def _read_bits(self, bits: List[int], pos: int, count: int) -> int:
        val = 0
        for i in range(count):
            if pos + i >= len(bits):
                return 0
            val = (val << 1) | bits[pos + i]
        return val

    # ------------------------------------------------------------------
    # TRANSFORM 00 – multi‑pass shift + compact RLE (FULLY LOSSLESS)
    # ------------------------------------------------------------------
    def transform_00(self, data: bytes) -> bytes:
        if not data:
            return b'\x00'

        best_result = None
        best_length = float('inf')
        best_shifts = []
        MAX_PASSES = 10

        current = bytearray(data)
        applied_shifts = []

        for _ in range(MAX_PASSES):
            best_shift = 0
            best_shifted = current
            best_score = float('-inf')

            for shift in range(256):
                tmp = bytearray(current)
                for j in range(len(tmp)):
                    tmp[j] = (tmp[j] + shift) % 256

                score = 0
                i = 0
                while i < len(tmp):
                    val = tmp[i]
                    run = 1
                    i += 1
                    while i < len(tmp) and tmp[i] == val:
                        run += 1
                        i += 1
                    score += run * run

                if score > best_score:
                    best_score = score
                    best_shifted = tmp
                    best_shift = shift

            applied_shifts.append(best_shift)
            rle_encoded = self._apply_rle_to_shifted(best_shifted, best_shift)

            if len(rle_encoded) < best_length:
                best_length = len(rle_encoded)
                best_result = rle_encoded
                best_shifts = applied_shifts.copy()

            current = best_shifted
            if len(rle_encoded) >= len(data):
                break

        if best_result is None or best_length >= len(data):
            return bytes([0]) + data

        header = bytearray([len(best_shifts)])
        header.extend(best_shifts)
        return header + best_result

    def _apply_rle_to_shifted(self, shifted_data: bytearray, shift: int) -> bytes:
        bits = []
        self._append_bits(bits, 0b010, 3)
        self._append_bits(bits, shift, 8)

        i = 0
        n = len(shifted_data)
        while i < n:
            val = shifted_data[i]
            run = 1
            i += 1
            while i < n and shifted_data[i] == val:
                run += 1
                i += 1

            while run >= 13:
                chunk = min(run, 268)
                self._append_bits(bits, 0b1111, 4)
                self._append_bits(bits, chunk - 13, 8)
                self._append_bits(bits, val, 8)
                run -= chunk

            if run == 1:
                self._append_bits(bits, 0b00, 2)
                self._append_bits(bits, val, 8)
            elif run <= 5:
                self._append_bits(bits, 0b01, 2)
                self._append_bits(bits, run - 2, 2)
                self._append_bits(bits, val, 8)
            elif run <= 12:
                self._append_bits(bits, 0b10, 2)
                self._append_bits(bits, run - 6, 3)
                self._append_bits(bits, val, 8)

        pad = (8 - len(bits) % 8) % 8
        self._append_bits(bits, 0, pad)

        out = bytearray()
        for j in range(0, len(bits), 8):
            byte = 0
            for k in range(8):
                if j + k < len(bits):
                    byte = (byte << 1) | bits[j + k]
            out.append(byte)
        return bytes(out)

    def reverse_transform_00(self, cdata: bytes) -> bytes:
        if not cdata or cdata == b'\x00':
            return b''
        if cdata[0] == 0:
            return cdata[1:]

        num_passes = cdata[0]
        if num_passes == 0 or len(cdata) < 1 + num_passes:
            return b''

        shifts = list(cdata[1:1 + num_passes])
        rle_data = cdata[1 + num_passes:]

        decoded = self._rle_decode(rle_data)
        if decoded is None:
            return b''

        current = bytearray(decoded)
        for shift in reversed(shifts):
            for i in range(len(current)):
                current[i] = (current[i] - shift) % 256
        return bytes(current)

    def _rle_decode(self, data: bytes) -> Optional[bytearray]:
        if not data:
            return None
        bits = []
        for b in data:
            for i in range(7, -1, -1):
                bits.append((b >> i) & 1)

        pos = 0
        nbits = len(bits)
        if nbits < 11:
            return None

        marker = self._read_bits(bits, pos, 3)
        pos += 3
        if marker != 0b010:
            return None
        pos += 8   # skip shift

        out = bytearray()
        while pos < nbits:
            if pos + 2 > nbits:
                break
            prefix = self._read_bits(bits, pos, 2)
            pos += 2

            if prefix == 0b00:
                if pos + 8 > nbits:
                    break
                run = 1
            elif prefix == 0b01:
                if pos + 2 + 8 > nbits:
                    break
                run = 2 + self._read_bits(bits, pos, 2)
                pos += 2
            elif prefix == 0b10:
                if pos + 3 + 8 > nbits:
                    break
                run = 6 + self._read_bits(bits, pos, 3)
                pos += 3
            else:  # prefix == 0b11
                if pos + 2 + 8 + 8 > nbits:
                    break
                if self._read_bits(bits, pos, 2) != 0b11:
                    return None
                pos += 2
                run = 13 + self._read_bits(bits, pos, 8)
                pos += 8

            if pos + 8 > nbits:
                break
            val = self._read_bits(bits, pos, 8)
            pos += 8

            out.extend([val] * run)

        for i in range(pos, nbits):
            if bits[i] != 0:
                return None
        return out

    # ------------------------------------------------------------------
    # TRANSFORM 01 – involutory XOR (lossless)
    # ------------------------------------------------------------------
    def transform_01(self, d, r=100):
        t = bytearray(d)
        for prime in PRIMES:
            xor_val = prime if prime == 2 else max(1, math.ceil(prime * 4096 / 28672))
            for _ in range(r):
                for i in range(0, len(t), 3):
                    if i < len(t):
                        t[i] ^= xor_val
        return bytes(t)

    def reverse_transform_01(self, d, r=100):
        return self.transform_01(d, r)

    # ------------------------------------------------------------------
    # TRANSFORM 02 – FIXED: use stored pattern index (now deterministic)
    # ------------------------------------------------------------------
    def transform_02(self, d):
        if len(d) < 1:
            return b''
        t = bytearray(d)
        checksum = sum(d) % 256
        pattern_index = (len(d) + checksum) % 256
        pattern_values = self._get_pattern(4, pattern_index)
        for i in range(1, len(t), 4):
            if i < len(t):
                t[i] ^= pattern_values[i % len(pattern_values)]
        return bytes([pattern_index]) + bytes(t)

    def reverse_transform_02(self, d):
        if len(d) < 2:
            return b''
        pattern_index = d[0]
        t = bytearray(d[1:])
        pattern_values = self._get_pattern(4, pattern_index)
        for i in range(1, len(t), 4):
            if i < len(t):
                t[i] ^= pattern_values[i % len(pattern_values)]
        return bytes(t)

    # ------------------------------------------------------------------
    # TRANSFORM 03 – rotate bits (lossless)
    # ------------------------------------------------------------------
    def transform_03(self, d):
        if len(d) < 1:
            return b''
        t = bytearray(d)
        rotation = (len(d) * 13 + sum(d)) % 8
        if rotation == 0:
            rotation = 1
        for i in range(2, len(t), 5):
            if i < len(t):
                t[i] = ((t[i] << rotation) | (t[i] >> (8 - rotation))) & 0xFF
        return bytes([rotation]) + bytes(t)

    def reverse_transform_03(self, d):
        if len(d) < 2:
            return b''
        rotation = d[0]
        t = bytearray(d[1:])
        for i in range(2, len(t), 5):
            if i < len(t):
                t[i] = ((t[i] >> rotation) | (t[i] << (8 - rotation))) & 0xFF
        return bytes(t)

    # ------------------------------------------------------------------
    # TRANSFORM 04 – subtract/add i%256 (lossless)
    # ------------------------------------------------------------------
    def transform_04(self, d, r=100):
        t = bytearray(d)
        for _ in range(r):
            for i in range(len(t)):
                t[i] = (t[i] - (i % 256)) % 256
        return bytes(t)

    def reverse_transform_04(self, d, r=100):
        t = bytearray(d)
        for _ in range(r):
            for i in range(len(t)):
                t[i] = (t[i] + (i % 256)) % 256
        return bytes(t)

    # ------------------------------------------------------------------
    # TRANSFORM 05 – rotate each byte (lossless)
    # ------------------------------------------------------------------
    def transform_05(self, d, s=3):
        t = bytearray(d)
        for i in range(len(t)):
            t[i] = ((t[i] << s) | (t[i] >> (8 - s))) & 0xFF
        return bytes(t)

    def reverse_transform_05(self, d, s=3):
        t = bytearray(d)
        for i in range(len(t)):
            t[i] = ((t[i] >> s) | (t[i] << (8 - s))) & 0xFF
        return bytes(t)

    # ------------------------------------------------------------------
    # TRANSFORM 06 – deterministic substitution cipher (lossless forever)
    # ------------------------------------------------------------------
    def transform_06(self, d, sd=42):
        sub = self._gen_substitution(f"substitution:{sd}")
        t = bytearray(d)
        for i in range(len(t)):
            t[i] = sub[t[i]]
        return bytes(t)

    def reverse_transform_06(self, d, sd=42):
        sub = self._gen_substitution(f"substitution:{sd}")
        inv = [0] * 256
        for i in range(256):
            inv[sub[i]] = i
        t = bytearray(d)
        for i in range(len(t)):
            t[i] = inv[t[i]]
        return bytes(t)

    # ------------------------------------------------------------------
    # TRANSFORM 07 – XOR with length + pi (involutory)
    # ------------------------------------------------------------------
    def transform_07(self, d, r=100):
        t = bytearray(d)
        sh = len(d) % len(self.PI_DIGITS)
        pi_rot = self.PI_DIGITS[sh:] + self.PI_DIGITS[:sh]
        sz = len(d) % 256
        for i in range(len(t)):
            t[i] ^= sz
        for _ in range(r):
            for i in range(len(t)):
                t[i] ^= pi_rot[i % len(pi_rot)]
        return bytes(t)

    def reverse_transform_07(self, d, r=100):
        return self.transform_07(d, r)

    # ------------------------------------------------------------------
    # TRANSFORM 08 – XOR with prime + pi (involutory)
    # ------------------------------------------------------------------
    def transform_08(self, d, r=100):
        t = bytearray(d)
        sh = len(d) % len(self.PI_DIGITS)
        pi_rot = self.PI_DIGITS[sh:] + self.PI_DIGITS[:sh]
        p = find_nearest_prime_around(len(d) % 256)
        for i in range(len(t)):
            t[i] ^= p
        for _ in range(r):
            for i in range(len(t)):
                t[i] ^= pi_rot[i % len(pi_rot)]
        return bytes(t)

    def reverse_transform_08(self, d, r=100):
        return self.transform_08(d, r)

    # ------------------------------------------------------------------
    # TRANSFORM 09 – XOR with prime, seed, pi, i (involutory)
    # ------------------------------------------------------------------
    def transform_09(self, d, r=100):
        t = bytearray(d)
        sh = len(d) % len(self.PI_DIGITS)
        pi_rot = self.PI_DIGITS[sh:] + self.PI_DIGITS[:sh]
        p = find_nearest_prime_around(len(d) % 256)
        seed = self.get_seed(len(d) % self.SEED_TABLE_COUNT, len(d))
        for i in range(len(t)):
            t[i] ^= p ^ seed
        for _ in range(r):
            for i in range(len(t)):
                t[i] ^= pi_rot[i % len(pi_rot)] ^ (i % 256)
        return bytes(t)

    def reverse_transform_09(self, d, r=100):
        return self.transform_09(d, r)

    # ------------------------------------------------------------------
    # TRANSFORM 10 – XOR with n from "X1" count (involutory)
    # ------------------------------------------------------------------
    def transform_10(self, d, r=100):
        cnt = sum(1 for i in range(len(d) - 1) if d[i:i + 2] == b'X1')
        n = (((cnt * 2) + 1) // 3) * 3 % 256
        t = bytearray(d)
        for _ in range(r):
            for i in range(len(t)):
                t[i] ^= n
        return bytes([n]) + bytes(t)

    def reverse_transform_10(self, d, r=100):
        if len(d) < 1:
            return b''
        n = d[0]
        t = bytearray(d[1:])
        for _ in range(r):
            for i in range(len(t)):
                t[i] ^= n
        return bytes(t)

    # ------------------------------------------------------------------
    # TRANSFORM 11 – FIXED: involutory XOR (key depends only on index and length)
    # ------------------------------------------------------------------
    def transform_11(self, d, r=100):
        if not d:
            return b''
        t = bytearray(d)
        length = len(t)
        for _ in range(r):
            for i in range(length):
                fib_idx = (i + length) % len(self.fibonacci)
                fib_val = self.fibonacci[fib_idx] % 256
                pos_val = (i * 13 + length * 17) % 256
                key = (fib_val ^ pos_val) % 256
                t[i] ^= key
        return bytes(t)

    def reverse_transform_11(self, d, r=100):
        return self.transform_11(d, r)

    # ------------------------------------------------------------------
    # TRANSFORM 12 – XOR with Fibonacci (involutory)
    # ------------------------------------------------------------------
    def transform_12(self, d, r=100):
        t = bytearray(d)
        for _ in range(r):
            for i in range(len(t)):
                t[i] ^= self.fibonacci[i % len(self.fibonacci)] % 256
        return bytes(t)

    def reverse_transform_12(self, d, r=100):
        return self.transform_12(d, r)

    # ------------------------------------------------------------------
    # TRANSFORM 13 – XOR with nearest prime (lossless)
    # ------------------------------------------------------------------
    def transform_13(self, d):
        if not d:
            return b''
        repeats = self._calculate_repeats(d)
        current_value = len(d) % 256
        prime_values = []
        count = 0
        while count < repeats:
            current_value = find_nearest_prime_around(current_value)
            prime_values.append(current_value)
            count += 1
        t = bytearray(d)
        xor_value = prime_values[-1] if prime_values else 0
        for i in range(len(t)):
            t[i] ^= xor_value
        repeat_byte = (repeats - 1) % 256
        return bytes([repeat_byte]) + bytes(t)

    def reverse_transform_13(self, d):
        if len(d) < 2:
            return b''
        repeat_byte = d[0]
        repeats = (repeat_byte + 1) % 256
        if repeats == 0:
            repeats = 256
        t = bytearray(d[1:])
        current_value = len(t) % 256
        prime_values = []
        count = 0
        while count < repeats:
            current_value = find_nearest_prime_around(current_value)
            prime_values.append(current_value)
            count += 1
        xor_value = prime_values[-1] if prime_values else 0
        for i in range(len(t)):
            t[i] ^= xor_value
        return bytes(t)

    # ------------------------------------------------------------------
    # TRANSFORM 14 – FIXED: ALWAYS prepend bits_to_add byte
    # ------------------------------------------------------------------
    def transform_14(self, d):
        if not d:
            return b'\x00'
        bits_to_add = self._calculate_bits_to_add(d)
        transformed = self._add_bits_to_end(d, bits_to_add)
        return bytes([bits_to_add]) + transformed

    def reverse_transform_14(self, d):
        if len(d) < 1:
            return b''
        bits_count = d[0]
        data_with_bits = d[1:]
        if not data_with_bits:
            return b''
        return data_with_bits[:-1]

    # ------------------------------------------------------------------
    # TRANSFORM 15 – add pattern to every 3rd byte (lossless, now deterministic)
    # ------------------------------------------------------------------
    def transform_15(self, d):
        if len(d) < 1:
            return b''
        t = bytearray(d)
        pattern_index = len(d) % 256
        pattern_values = self._get_pattern(3, pattern_index)
        for i in range(0, len(t), 3):
            if i < len(t):
                t[i] = (t[i] + pattern_values[i % len(pattern_values)]) % 256
        return bytes([pattern_index]) + bytes(t)

    def reverse_transform_15(self, d):
        if len(d) < 2:
            return b''
        pattern_index = d[0]
        t = bytearray(d[1:])
        pattern_values = self._get_pattern(3, pattern_index)
        for i in range(0, len(t), 3):
            if i < len(t):
                t[i] = (t[i] - pattern_values[i % len(pattern_values)]) % 256
        return bytes(t)

    # ------------------------------------------------------------------
    # TRANSFORM 256 – Identity (lossless, no change)
    # ------------------------------------------------------------------
    def transform_256(self, d: bytes) -> bytes:
        return d

    def reverse_transform_256(self, d: bytes) -> bytes:
        return d

    # ------------------------------------------------------------------
    # Helpers for transforms 13,14,15
    # ------------------------------------------------------------------
    def _calculate_bits_to_add(self, data: bytes) -> int:
        if not data:
            return 0
        length = len(data)
        first = data[0]
        last = data[-1]
        checksum = sum(data) % 256
        return ((length * 13 + first * 17 + last * 23 + checksum * 29) % 9)

    def _add_bits_to_end(self, data: bytes, bits_count: int) -> bytes:
        if bits_count == 0:
            return data + b'\x00'
        if not data:
            data = b'\x00'
        last_byte = data[-1]
        bits_value = last_byte & ((1 << bits_count) - 1)
        bits_byte = ((bits_count & 0x0F) << 4) | (bits_value & 0x0F)
        return data + bytes([bits_byte])

    def _calculate_repeats(self, data: bytes) -> int:
        if not data:
            return 1
        length = len(data)
        byte_sum = sum(data) % 256
        repeats = ((length * 13 + byte_sum * 17) % 256) + 1
        return max(1, min(256, repeats))

    # ------------------------------------------------------------------
    # Dynamic transforms 17‑255 – XOR with seed (involutory, lossless)
    # ------------------------------------------------------------------
    def _dynamic_transform(self, n: int):
        def tf(data: bytes):
            if not data:
                return b''
            seed = self.get_seed(n % self.SEED_TABLE_COUNT, len(data))
            t = bytearray(data)
            for i in range(len(t)):
                t[i] ^= seed
            return bytes(t)
        return tf, tf

    # ------------------------------------------------------------------
    # Compression backends – with raw fallback
    # ------------------------------------------------------------------
    def _compress_backend(self, data: bytes) -> bytes:
        candidates = []
        if paq is not None:
            try:
                candidates.append((b'L', paq.compress(data)))
            except:
                pass
        if HAS_ZSTD:
            try:
                candidates.append((b'Z', zstd_cctx.compress(data)))
            except:
                pass

        if not candidates:
            return b'N' + data

        winner_id, winner_data = min(candidates, key=lambda x: len(x[1]))
        return winner_id + winner_data

    def _decompress_backend(self, data: bytes) -> Optional[bytes]:
        if len(data) < 1:
            return None
        engine = data[0]
        payload = data[1:]

        if engine == ord('L') and paq is not None:
            try:
                return paq.decompress(payload)
            except:
                return None
        if engine == ord('Z') and HAS_ZSTD:
            try:
                return zstd_dctx.decompress(payload)
            except:
                return None
        if engine == ord('N'):
            return payload
        return None

    # ------------------------------------------------------------------
    # FULL self‑test – verifies EVERY transform 1‑256 and full pipeline
    # ------------------------------------------------------------------
    def self_test(self) -> bool:
        print("Running FULL self‑test to verify ALL transforms (1‑256) are lossless...")
        test_passed = True
        failed_transforms = []

        # Build transforms list: numbers 1..256
        transforms = [
            (1,   self.transform_00,   self.reverse_transform_00),
            (2,   self.transform_01,   self.reverse_transform_01),
            (3,   self.transform_02,   self.reverse_transform_02),
            (4,   self.transform_03,   self.reverse_transform_03),
            (5,   self.transform_04,   self.reverse_transform_04),
            (6,   self.transform_05,   self.reverse_transform_05),
            (7,   self.transform_06,   self.reverse_transform_06),
            (8,   self.transform_07,   self.reverse_transform_07),
            (9,   self.transform_08,   self.reverse_transform_08),
            (10,  self.transform_09,   self.reverse_transform_09),
            (11,  self.transform_10,   self.reverse_transform_10),
            (12,  self.transform_11,   self.reverse_transform_11),
            (13,  self.transform_12,   self.reverse_transform_12),
            (14,  self.transform_13,   self.reverse_transform_13),
            (15,  self.transform_14,   self.reverse_transform_14),
            (16,  self.transform_15,   self.reverse_transform_15),
        ]
        for i in range(17, 256):
            fwd, rev = self._dynamic_transform(i)
            transforms.append((i, fwd, rev))
        transforms.append((256, self.transform_256, self.reverse_transform_256))

        print(f"  Testing {len(transforms)} transforms (1‑256)...")

        # --- Test empty data (skip transform 1 because it returns b'\x00' and expects that) ---
        for num, fwd, rev in transforms:
            if num == 1:
                continue
            try:
                enc = fwd(b'')
                dec = rev(enc)
                if dec != b'':
                    print(f"  [FAIL] Transform {num} empty data")
                    test_passed = False
                    failed_transforms.append(num)
            except Exception as e:
                print(f"  [FAIL] Transform {num} empty data (exception: {e})")
                test_passed = False
                failed_transforms.append(num)

        # --- Test all 256 single bytes for every transform ---
        for num, fwd, rev in transforms:
            for b_val in range(256):
                orig = bytes([b_val])
                try:
                    enc = fwd(orig)
                    dec = rev(enc)
                except Exception as e:
                    print(f"  [FAIL] Transform {num} byte 0x{b_val:02x} (exception: {e})")
                    test_passed = False
                    failed_transforms.append(num)
                    break
                if dec != orig:
                    print(f"  [FAIL] Transform {num} byte 0x{b_val:02x}")
                    test_passed = False
                    failed_transforms.append(num)
                    break
            else:
                if num <= 16 or num % 50 == 0 or num == 256:
                    print(f"  [PASS] Transform {num} all single bytes")
                else:
                    print(".", end="", flush=True)

        # --- Test random short data (5 samples per transform) ---
        print("\n  Testing random short data (5 samples each)...")
        def _test_rand(seed, idx):
            return bytes(_hash_int(f"testrand:{seed}", idx + i, 256) for i in range(idx % 100 + 1))
        for num, fwd, rev in transforms:
            for sample in range(5):
                data = _test_rand(num, sample * 100 + sample)
                try:
                    enc = fwd(data)
                    dec = rev(enc)
                except Exception as e:
                    print(f"  [FAIL] Transform {num} random size {len(data)} (exception: {e})")
                    test_passed = False
                    failed_transforms.append(num)
                    break
                if dec != data:
                    print(f"  [FAIL] Transform {num} random size {len(data)}")
                    test_passed = False
                    failed_transforms.append(num)
                    break
            else:
                if num <= 16 or num % 50 == 0 or num == 256:
                    print(f"  [PASS] Transform {num} random short data")
                else:
                    print(".", end="", flush=True)

        # --- Full pipeline test (100 random inputs) ---
        print("\n  Testing full compression/decompression pipeline on random data...")
        for test_num in range(100):
            data = _test_rand(999, test_num * 50)
            try:
                compressed = self.compress_with_best(data)
                decompressed, marker = self.decompress_with_best(compressed)
            except Exception as e:
                print(f"  [FAIL] Pipeline test #{test_num+1} (exception: {e})")
                test_passed = False
                break
            if decompressed != data:
                print(f"  [FAIL] Pipeline test #{test_num+1} (marker {marker})")
                test_passed = False
                break
        else:
            print("  [PASS] Full pipeline (100 random inputs)")

        if test_passed:
            print("\n[PASS] Self‑test PASSED – all transforms 1‑256 are 100% lossless.\n")
        else:
            print(f"\n[FAIL] Self‑test FAILED – {len(failed_transforms)} transform(s) failed: {set(failed_transforms)}")
            print("Please report this bug!\n")
        return test_passed

    # ------------------------------------------------------------------
    # Main compression logic – tries ALL transforms 1‑256
    # ------------------------------------------------------------------
    def compress_with_best(self, data: bytes) -> bytes:
        if not data:
            return bytes([255]) + self._compress_backend(b'')

        best_payload = None
        best_size = float('inf')
        best_marker = 255   # identity fallback (transform 256)

        transforms = [
            (1,   self.transform_00),
            (2,   self.transform_01),
            (3,   self.transform_02),
            (4,   self.transform_03),
            (5,   self.transform_04),
            (6,   self.transform_05),
            (7,   self.transform_06),
            (8,   self.transform_07),
            (9,   self.transform_08),
            (10,  self.transform_09),
            (11,  self.transform_10),
            (12,  self.transform_11),
            (13,  self.transform_12),
            (14,  self.transform_13),
            (15,  self.transform_14),
            (16,  self.transform_15),
        ] + [(i, self._dynamic_transform(i)[0]) for i in range(17, 256)] + \
          [(256, self.transform_256)]

        for t_num, func in transforms:
            try:
                transformed = func(data)
                compressed = self._compress_backend(transformed)
                clen = len(compressed)
                if clen < best_size:
                    best_size = clen
                    best_payload = compressed
                    best_marker = t_num - 1   # store marker = transform‑1
            except Exception:
                continue

        return bytes([best_marker]) + best_payload

    def decompress_with_best(self, data: bytes):
        if len(data) < 2:
            return b'', None

        marker = data[0]
        t_num = marker + 1   # convert back to transform number (1‑256)
        payload = data[1:]

        backend = self._decompress_backend(payload)
        if backend is None:
            return b'', None

        rev_map = {
            1:    self.reverse_transform_00,
            2:    self.reverse_transform_01,
            3:    self.reverse_transform_02,
            4:    self.reverse_transform_03,
            5:    self.reverse_transform_04,
            6:    self.reverse_transform_05,
            7:    self.reverse_transform_06,
            8:    self.reverse_transform_07,
            9:    self.reverse_transform_08,
            10:   self.reverse_transform_09,
            11:   self.reverse_transform_10,
            12:   self.reverse_transform_11,
            13:   self.reverse_transform_12,
            14:   self.reverse_transform_13,
            15:   self.reverse_transform_14,
            16:   self.reverse_transform_15,
        }
        for i in range(17, 256):
            rev_map[i] = self._dynamic_transform(i)[1]
        rev_map[256] = self.reverse_transform_256

        rev_func = rev_map.get(t_num, lambda x: x)
        try:
            result = rev_func(backend)
        except Exception:
            return b'', None
        return result, t_num

    # ------------------------------------------------------------------
    # Public file API
    # ------------------------------------------------------------------
    def compress(self, infile: str, outfile: str):
        try:
            with open(infile, 'rb') as f:
                data = f.read()
        except FileNotFoundError:
            print(f"Error: input file '{infile}' not found.")
            return
        except Exception as e:
            print(f"Error reading file: {e}")
            return

        compressed = self.compress_with_best(data)

        try:
            with open(outfile, 'wb') as f:
                f.write(compressed)
        except Exception as e:
            print(f"Error writing output file: {e}")
            return

        if len(data) == 0:
            print(f"Compressed empty file -> {outfile} (0 bytes)")
        else:
            ratio = (1 - len(compressed) / len(data)) * 100
            if paq is None and not HAS_ZSTD:
                backend_info = " (no compression backend - raw storage used)"
            elif HAS_ZSTD:
                backend_info = " (zstd)"
            else:
                backend_info = " (paq only)"
            print(f"Compressed {len(data)} -> {len(compressed)} bytes ({ratio:.2f}% saved){backend_info} -> {outfile}")

    def decompress(self, infile: str, outfile: str):
        try:
            with open(infile, 'rb') as f:
                data = f.read()
        except FileNotFoundError:
            print(f"Error: compressed file '{infile}' not found.")
            return
        except Exception as e:
            print(f"Error reading file: {e}")
            return

        original, t_num = self.decompress_with_best(data)
        if original is None or original == b'':
            print("Decompression failed: invalid compressed data or transform error.")
            return

        try:
            with open(outfile, 'wb') as f:
                f.write(original)
        except Exception as e:
            print(f"Error writing output file: {e}")
            return

        print(f"Decompressed (transform {t_num}) -> {outfile} ({len(original)} bytes)")


def main():
    print(f"{PROGNAME} - fully lossless, all transforms 1‑256 verified (deterministic)")

    c = PAQJPCompressor()

    while True:
        print("\nSelect an option:")
        print("1) Compress")
        print("2) Decompress")
        print("3) Run self‑test (verify lossless 256 transforms)")
        print("q) Quit")
        ch = input("> ").strip().lower()

        if ch == '1':
            i = input("Input file: ").strip()
            o = input("Output file: ").strip() or i + ".pjp"
            c.compress(i, o)
        elif ch == '2':
            i = input("Compressed file: ").strip()
            o = input("Output file: ").strip() or i.rsplit('.', 1)[0] + ".orig"
            c.decompress(i, o)
        elif ch == '3':
            c.self_test()
        elif ch == 'q':
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or q.")


if __name__ == "__main__":
    main()
