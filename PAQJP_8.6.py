#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAQJP 8.6 - 256 Lossless Transforms + 2704 Transform-Pair Sequences
----------------------------------------------------------------------------
All single transforms (1-256), all ordered pairs (2704), and the raw (no-transform)
path are mathematically lossless.  Every transform has a perfect inverse.

HEADER FORMAT (variable-length):
   Byte 0:  F
     F < 252             → single transform number = F+1  (1..252)
     F == 252            → raw (no transform)
     F == 253            → pair: next 2 bytes encode pair-index big-endian
     F == 254            → extended single: next byte X → transform = 253+X (0..3)
     F == 255            → RESERVED (unused)

Usage:
    python paqjp86.py
    Choose 1 (compress), 2 (decompress), or 3 (full self-test).

Dependencies (optional):
    - zstandard (for Zstd backend, level 22)
    - paq (for PAQ backend)
If neither is available, data is stored raw - still lossless.
"""

import os
import math
import random
import sys
import decimal
from typing import Optional, List, Tuple, Dict, Callable

# ------------------------------------------------------------
# Optional compression backends
# ------------------------------------------------------------
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

PROGNAME = "PAQJP_8.6_LOSSLESS_VARIABLE_HEADER"

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
PRIMES = [p for p in range(2, 256) if all(p % d != 0 for d in range(2, int(p**0.5)+1))]
PI_DIGITS = [79, 17, 111]

# ------------------------------------------------------------
# Helper: nearest prime
# ------------------------------------------------------------
def find_nearest_prime_around(n: int) -> int:
    o = 0
    while True:
        c1, c2 = n - o, n + o
        if c1 >= 2 and all(c1 % d != 0 for d in range(2, int(c1**0.5)+1)):
            return c1
        if c2 >= 2 and all(c2 % d != 0 for d in range(2, int(c2**0.5)+1)):
            return c2
        o += 1

# ------------------------------------------------------------
# Main Compressor Class
# ------------------------------------------------------------
class PAQJPCompressor:
    def __init__(self):
        self.PI_DIGITS = PI_DIGITS.copy()
        self.seed_tables = self._gen_seed_tables(num=126, size=40, seed=42)
        self.fibonacci = self._gen_fib(100)

        # pi approximation constants
        self.PI_STR = "3.14159265358979323846264338327950288419716939937510"

        # Build all 256 transforms
        self._build_transform_maps()

        # Build pair sequences (base 1..52 → 2704 pairs)
        self.sequences = self._build_pair_sequences()
        # Precompute pair lookup index -> (t1,t2) for decoding
        self.pair_lookup = {idx: (t1, t2) for idx, (t1, t2) in enumerate(self.sequences)}

    # ------------------------------------------------------------------
    # pi algorithm helpers (for transform 17)
    # ------------------------------------------------------------------
    def get_pi_digits(self, n: int) -> str:
        if n < 1:
            return ""
        return self.PI_STR[2:2+n]

    def find_lossless_k(self, n: int):
        if n < 1:
            return 0, True
        true_digits = self.get_pi_digits(n)
        true_scaled = int(self.PI_STR.replace('.', '')[:n+1])
        DENOM = 16777216
        decimal.getcontext().prec = 50
        pi_dec = decimal.Decimal(self.PI_STR)
        k_float = (pi_dec - 3) * DENOM
        k_candidate = int(round(k_float))
        k_candidate = max(0, min(k_candidate, DENOM - 1))
        approx_scaled = (3 * 10**n * DENOM + k_candidate * 10**n) // DENOM
        return k_candidate, approx_scaled == true_scaled

    def to_bin(self, value: int, bits: int) -> str:
        return format(value, 'b').zfill(bits)

    def get_bit_size(self, k: int) -> int:
        return 23 if k <= 0x7FFFFF else 25

    # ------------------------------------------------------------------
    # Transform 17 - pi-based XOR mask (lossless)
    # ------------------------------------------------------------------
    def transform_17(self, data: bytes) -> bytes:
        if not data:
            return b''
        k, _ = self.find_lossless_k(7)
        bits_used = self.get_bit_size(k)
        bit_str = self.to_bin(k, bits_used)
        mask_bytes = []
        for i in range(0, len(bit_str), 8):
            byte_bits = bit_str[i:i+8]
            if len(byte_bits) < 8:
                byte_bits = byte_bits.ljust(8, '0')
            mask_bytes.append(int(byte_bits, 2))
        mask = bytes(mask_bytes)
        t = bytearray(data)
        mask_len = len(mask)
        for i in range(len(t)):
            t[i] ^= mask[i % mask_len]
        return bytes(t)

    def reverse_transform_17(self, data: bytes) -> bytes:
        return self.transform_17(data)

    # ------------------------------------------------------------------
    # New constant helpers (used in transforms 18-20)
    # ------------------------------------------------------------------
    def get_basel_digits(self, n: int) -> str:
        decimal.getcontext().prec = n + 5
        pi = decimal.Decimal(self.PI_STR)
        basel = (pi * pi) / decimal.Decimal(6)
        s = str(basel).replace('.', '')
        return s[:n]

    def get_one_over_e_digits(self, n: int) -> str:
        decimal.getcontext().prec = n + 5
        e = decimal.Decimal(1).exp()
        inv_e = decimal.Decimal(1) / e
        s = str(inv_e).replace('.', '')
        return s[:n]

    def get_5e_digits(self, n: int) -> str:
        decimal.getcontext().prec = n + 5
        e = decimal.Decimal(1).exp()
        five_e = decimal.Decimal(5) * e
        s = str(five_e).replace('.', '')
        return s[:n]

    # ------------------------------------------------------------------
    # Helpers: seed tables, Fibonacci
    # ------------------------------------------------------------------
    def _gen_seed_tables(self, num=126, size=40, seed=42):
        random.seed(seed)
        return [[random.randint(5, 255) for _ in range(size)] for _ in range(num)]

    def _gen_fib(self, n):
        a, b = 0, 1
        res = [a, b]
        for _ in range(2, n):
            a, b = b, a + b
            res.append(b)
        return res

    def get_seed(self, idx: int, val: int) -> int:
        if 0 <= idx < len(self.seed_tables):
            return self.seed_tables[idx][val % 40]
        return 0

    # ------------------------------------------------------------------
    # Bit helpers (for RLE transform)
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
    # TRANSFORM 00 - multi-pass shift + compact RLE (lossless)
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
        pos += 8
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
            else:
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
    # Transforms 01-15, 17-21 (all lossless)
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
    reverse_transform_01 = transform_01

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

    def transform_06(self, d, sd=42):
        random.seed(sd)
        sub = list(range(256))
        random.shuffle(sub)
        t = bytearray(d)
        for i in range(len(t)):
            t[i] = sub[t[i]]
        return bytes(t)

    def reverse_transform_06(self, d, sd=42):
        random.seed(sd)
        sub = list(range(256))
        random.shuffle(sub)
        inv = [0] * 256
        for i in range(256):
            inv[sub[i]] = i
        t = bytearray(d)
        for i in range(len(t)):
            t[i] = inv[t[i]]
        return bytes(t)

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
    reverse_transform_07 = transform_07

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
    reverse_transform_08 = transform_08

    def transform_09(self, d, r=100):
        t = bytearray(d)
        sh = len(d) % len(self.PI_DIGITS)
        pi_rot = self.PI_DIGITS[sh:] + self.PI_DIGITS[:sh]
        p = find_nearest_prime_around(len(d) % 256)
        seed = self.get_seed(len(d) % len(self.seed_tables), len(d))
        for i in range(len(t)):
            t[i] ^= p ^ seed
        for _ in range(r):
            for i in range(len(t)):
                t[i] ^= pi_rot[i % len(pi_rot)] ^ (i % 256)
        return bytes(t)
    reverse_transform_09 = transform_09

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
    reverse_transform_11 = transform_11

    def transform_12(self, d, r=100):
        t = bytearray(d)
        for _ in range(r):
            for i in range(len(t)):
                t[i] ^= self.fibonacci[i % len(self.fibonacci)] % 256
        return bytes(t)
    reverse_transform_12 = transform_12

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

    # FIXED: transform_14 is now fully reversible (simple checksum append)
    def transform_14(self, d):
        if not d:
            return b'\x00'
        checksum = sum(d) % 256
        return d + bytes([checksum])

    def reverse_transform_14(self, d):
        if not d:
            return b''
        return d[:-1]

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

    # NEW TRANSFORM 16 (was missing in older versions)
    def transform_16(self, data: bytes) -> bytes:
        """Simple involutory XOR based on data length."""
        if not data:
            return b''
        xor_byte = (len(data) * 7 + 13) % 256
        t = bytearray(data)
        for i in range(len(t)):
            t[i] ^= xor_byte
        return bytes(t)
    reverse_transform_16 = transform_16

    # NEW TRANSFORMS 18-21 (mathematical constants)
    def transform_18(self, data: bytes) -> bytes:
        if not data:
            return b''
        digits = self.get_basel_digits(max(10, len(data) // 2 + 5))
        mask = bytes(int(digits[i:i+2]) % 256 for i in range(0, len(digits), 2))
        t = bytearray(data)
        for i in range(len(t)):
            t[i] ^= mask[i % len(mask)]
        return bytes(t)
    reverse_transform_18 = transform_18

    def transform_19(self, data: bytes) -> bytes:
        if not data:
            return b''
        digits = self.get_one_over_e_digits(max(10, len(data) // 2 + 5))
        mask = bytes(int(digits[i:i+2]) % 256 for i in range(0, len(digits), 2))
        t = bytearray(data)
        for i in range(len(t)):
            t[i] ^= mask[i % len(mask)]
        return bytes(t)
    reverse_transform_19 = transform_19

    def transform_20(self, data: bytes) -> bytes:
        if not data:
            return b''
        digits = self.get_5e_digits(max(10, len(data) // 2 + 5))
        mask = bytes(int(digits[i:i+2]) % 256 for i in range(0, len(digits), 2))
        t = bytearray(data)
        for i in range(len(t)):
            t[i] ^= mask[i % len(mask)]
        return bytes(t)
    reverse_transform_20 = transform_20

    def transform_21(self, data: bytes) -> bytes:
        if not data:
            return b''
        shift = 255
        t = bytearray(data)
        for i in range(len(t)):
            t[i] = (t[i] + shift) % 256
        return bytes(t)

    def reverse_transform_21(self, data: bytes) -> bytes:
        if not data:
            return b''
        shift = 255
        t = bytearray(data)
        for i in range(len(t)):
            t[i] = (t[i] - shift) % 256
        return bytes(t)

    # ------------------------------------------------------------------
    # Identity transform (256)
    # ------------------------------------------------------------------
    def transform_256(self, d: bytes) -> bytes:
        return d
    reverse_transform_256 = transform_256

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_pattern(self, size: int, index: int):
        random.seed(12345 + size * 100 + index)
        return [random.randint(0, 255) for _ in range(size)]

    def _calculate_repeats(self, data: bytes) -> int:
        if not data:
            return 1
        length = len(data)
        byte_sum = sum(data) % 256
        repeats = ((length * 13 + byte_sum * 17) % 256) + 1
        return max(1, min(256, repeats))

    # ------------------------------------------------------------------
    # Dynamic transforms 22-255 - XOR with seed (involutory)
    # ------------------------------------------------------------------
    def _dynamic_transform(self, n: int):
        def tf(data: bytes):
            if not data:
                return b''
            seed = self.get_seed(n % len(self.seed_tables), len(data))
            t = bytearray(data)
            for i in range(len(t)):
                t[i] ^= seed
            return bytes(t)
        return tf, tf

    # ------------------------------------------------------------------
    # Build transform maps (1..256)
    # ------------------------------------------------------------------
    def _build_transform_maps(self):
        self.fwd_transforms: Dict[int, Callable] = {}
        self.rev_transforms: Dict[int, Callable] = {}

        # 1..16 explicit
        self.fwd_transforms[1] = self.transform_00
        self.rev_transforms[1] = self.reverse_transform_00
        self.fwd_transforms[2] = self.transform_01
        self.rev_transforms[2] = self.reverse_transform_01
        self.fwd_transforms[3] = self.transform_02
        self.rev_transforms[3] = self.reverse_transform_02
        self.fwd_transforms[4] = self.transform_03
        self.rev_transforms[4] = self.reverse_transform_03
        self.fwd_transforms[5] = self.transform_04
        self.rev_transforms[5] = self.reverse_transform_04
        self.fwd_transforms[6] = self.transform_05
        self.rev_transforms[6] = self.reverse_transform_05
        self.fwd_transforms[7] = self.transform_06
        self.rev_transforms[7] = self.reverse_transform_06
        self.fwd_transforms[8] = self.transform_07
        self.rev_transforms[8] = self.reverse_transform_07
        self.fwd_transforms[9] = self.transform_08
        self.rev_transforms[9] = self.reverse_transform_08
        self.fwd_transforms[10] = self.transform_09
        self.rev_transforms[10] = self.reverse_transform_09
        self.fwd_transforms[11] = self.transform_10
        self.rev_transforms[11] = self.reverse_transform_10
        self.fwd_transforms[12] = self.transform_11
        self.rev_transforms[12] = self.reverse_transform_11
        self.fwd_transforms[13] = self.transform_12
        self.rev_transforms[13] = self.reverse_transform_12
        self.fwd_transforms[14] = self.transform_13
        self.rev_transforms[14] = self.reverse_transform_13
        self.fwd_transforms[15] = self.transform_14
        self.rev_transforms[15] = self.reverse_transform_14
        self.fwd_transforms[16] = self.transform_16
        self.rev_transforms[16] = self.reverse_transform_16

        # 17 - pi transform
        self.fwd_transforms[17] = self.transform_17
        self.rev_transforms[17] = self.reverse_transform_17

        # 18..21 - mathematical constants
        self.fwd_transforms[18] = self.transform_18
        self.rev_transforms[18] = self.reverse_transform_18
        self.fwd_transforms[19] = self.transform_19
        self.rev_transforms[19] = self.reverse_transform_19
        self.fwd_transforms[20] = self.transform_20
        self.rev_transforms[20] = self.reverse_transform_20
        self.fwd_transforms[21] = self.transform_21
        self.rev_transforms[21] = self.reverse_transform_21

        # 22..255 dynamic
        for i in range(22, 256):
            fwd, rev = self._dynamic_transform(i)
            self.fwd_transforms[i] = fwd
            self.rev_transforms[i] = rev

        # 256 identity
        self.fwd_transforms[256] = self.transform_256
        self.rev_transforms[256] = self.reverse_transform_256

        # Safety check: all 1..256 must be present
        for i in range(1, 257):
            if i not in self.fwd_transforms:
                raise RuntimeError(f"Transform {i} missing!")

    # ------------------------------------------------------------------
    # Build pair sequences – exactly 2704 (52×52)
    # ------------------------------------------------------------------
    def _build_pair_sequences(self) -> List[Tuple[int, int]]:
        base = list(range(1, 53))   # 1..52 → 2704 pairs
        return [(t1, t2) for t1 in base for t2 in base]

    # ------------------------------------------------------------------
    # Apply / reverse sequence
    # ------------------------------------------------------------------
    def _apply_sequence(self, data: bytes, seq: Tuple[int, ...]) -> bytes:
        result = data
        for t_num in seq:
            result = self.fwd_transforms[t_num](result)
        return result

    def _reverse_sequence(self, data: bytes, seq: Tuple[int, ...]) -> bytes:
        result = data
        for t_num in reversed(seq):
            result = self.rev_transforms[t_num](result)
        return result

    # ------------------------------------------------------------------
    # Compression backends
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
    # ENCODING / DECODING of the variable-length header
    # ------------------------------------------------------------------
    def _encode_marker_single(self, t: int) -> bytes:
        """Return header bytes for a single transform 1..256."""
        if t <= 252:
            return bytes([t - 1])
        else:
            return bytes([254, t - 253])

    def _encode_marker_raw(self) -> bytes:
        return bytes([252])

    def _encode_marker_pair(self, t1: int, t2: int) -> bytes:
        """Both t1 and t2 must be in 1..52.  Index = (t1-1)*52 + (t2-1)."""
        idx = (t1 - 1) * 52 + (t2 - 1)
        # 2 bytes, big-endian
        return bytes([253, (idx >> 8) & 0xFF, idx & 0xFF])

    def _decode_header(self, data: bytes):
        """
        Parse the header.  Returns (offset, seq) where:
            offset: number of header bytes consumed
            seq: tuple of transform numbers (empty for raw)
        """
        if len(data) < 1:
            return 0, ()
        f = data[0]
        if f < 252:
            return 1, (f + 1,)
        elif f == 252:
            return 1, ()
        elif f == 253:
            if len(data) < 3:
                return 0, ()  # incomplete
            idx = (data[1] << 8) | data[2]
            if idx >= len(self.sequences):
                return 0, ()  # invalid index
            t1, t2 = self.pair_lookup[idx]
            return 3, (t1, t2)
        elif f == 254:
            if len(data) < 2:
                return 0, ()
            x = data[1]
            if x > 3:
                return 0, ()  # invalid
            return 2, (253 + x,)
        else:  # 255 reserved
            return 0, ()

    # ------------------------------------------------------------------
    # Main compression / decompression (with variable header)
    # ------------------------------------------------------------------
    def compress_with_best(self, data: bytes) -> bytes:
        """
        Try raw, every single transform, every pair.
        Return the smallest complete representation: header + backend_compressed.
        """
        if not data:
            # For empty input, raw is always smallest.
            backend = self._compress_backend(b'')
            return self._encode_marker_raw() + backend

        best_total = float('inf')
        best_bytes = None

        # Raw
        raw_backend = self._compress_backend(data)
        candidate = self._encode_marker_raw() + raw_backend
        if len(candidate) < best_total:
            best_total = len(candidate)
            best_bytes = candidate

        # Singles 1..256
        for t in range(1, 257):
            try:
                transformed = self.fwd_transforms[t](data)
                backend = self._compress_backend(transformed)
                candidate = self._encode_marker_single(t) + backend
                if len(candidate) < best_total:
                    best_total = len(candidate)
                    best_bytes = candidate
            except Exception:
                continue

        # Pairs (2704)
        for t1, t2 in self.sequences:
            try:
                transformed = self._apply_sequence(data, (t1, t2))
                backend = self._compress_backend(transformed)
                candidate = self._encode_marker_pair(t1, t2) + backend
                if len(candidate) < best_total:
                    best_total = len(candidate)
                    best_bytes = candidate
            except Exception:
                continue

        return best_bytes

    def decompress_with_best(self, data: bytes):
        """
        Returns (original_bytes, seq) where seq is the tuple of transforms used
        (empty for raw).  Returns (b'', None) on error.
        """
        offset, seq = self._decode_header(data)
        if offset == 0:
            return b'', None
        payload = data[offset:]
        backend = self._decompress_backend(payload)
        if backend is None:
            return b'', None
        try:
            if not seq:  # raw
                result = backend
            else:
                result = self._reverse_sequence(backend, seq)
        except Exception:
            return b'', None
        return result, seq

    # ------------------------------------------------------------------
    # EXHAUSTIVE SELF-TEST
    # ------------------------------------------------------------------
    def full_self_test(self) -> bool:
        print("=" * 60)
        print("PAQJP 8.6 - FULL SELF-TEST (100% lossless)")
        print("=" * 60)
        all_ok = True

        # Single transforms on every byte
        print("Testing all 256 single transforms on all 256 byte values...")
        for t_num in range(1, 257):
            for b in range(256):
                orig = bytes([b])
                try:
                    enc = self.fwd_transforms[t_num](orig)
                    dec = self.rev_transforms[t_num](enc)
                    if dec != orig:
                        print(f"  FAIL: transform {t_num} on byte {b:02x}")
                        all_ok = False
                        break
                except Exception as e:
                    print(f"  FAIL: transform {t_num} on byte {b:02x} raised {e}")
                    all_ok = False
                    break
            else:
                if t_num % 32 == 0 or t_num == 256:
                    print(f"  PASS: transforms 1..{t_num} OK on all bytes")
            if not all_ok:
                break

        if not all_ok:
            print("\n[FAIL] Single-transform test failed.")
            return False

        # Pairs on every byte
        print(f"\nTesting all {len(self.sequences)} transform pairs on all 256 byte values...")
        for idx, seq in enumerate(self.sequences):
            for b in range(256):
                orig = bytes([b])
                try:
                    enc = self._apply_sequence(orig, seq)
                    dec = self._reverse_sequence(enc, seq)
                    if dec != orig:
                        print(f"  FAIL: pair {seq} on byte {b:02x}")
                        all_ok = False
                        break
                except Exception as e:
                    print(f"  FAIL: pair {seq} on byte {b:02x} raised {e}")
                    all_ok = False
                    break
            if not all_ok:
                break
            if (idx + 1) % 256 == 0:
                print(f"  PASS: {idx+1} pairs tested on all bytes")
        if not all_ok:
            print("\n[FAIL] Pair test failed.")
            return False
        print("  PASS: all pairs OK on all bytes")

        # Random data + full pipeline test
        print("\nTesting random 1000‑byte block through full compress/decompress...")
        rng = random.Random(12345)
        test_data = bytes(rng.randint(0, 255) for _ in range(1000))
        compressed = self.compress_with_best(test_data)
        decompressed, seq = self.decompress_with_best(compressed)
        if decompressed != test_data:
            print("  FAIL: random data full pipeline mismatch")
            return False
        print(f"  PASS: random data pipeline OK (used {'raw' if not seq else 'pair' if len(seq)==2 else 'single'})")

        # Empty input test
        print("\nTesting empty input...")
        compressed_empty = self.compress_with_best(b'')
        decomp_empty, sempty = self.decompress_with_best(compressed_empty)
        if decomp_empty != b'':
            print("  FAIL: empty input pipeline mismatch")
            return False
        print("  PASS: empty input pipeline OK")

        print("\n[All tests passed - compressor is 100% lossless]")
        return True

    # ------------------------------------------------------------------
    # File API
    # ------------------------------------------------------------------
    def compress_file(self, infile: str, outfile: str):
        try:
            with open(infile, 'rb') as f:
                data = f.read()
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

        print(f"Compressed {len(data)} → {len(compressed)} bytes → {outfile}")

    def decompress_file(self, infile: str, outfile: str):
        try:
            with open(infile, 'rb') as f:
                data = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return

        original, seq = self.decompress_with_best(data)
        if original == b'' and seq is None:
            print("Decompression failed.")
            return

        try:
            with open(outfile, 'wb') as f:
                f.write(original)
        except Exception as e:
            print(f"Error writing output file: {e}")
            return

        seq_str = "raw" if not seq else f"sequence {seq}"
        print(f"Decompressed ({seq_str}) → {outfile} ({len(original)} bytes)")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print(f"{PROGNAME}")
    print("256 single transforms + 2704 transform-pair sequences (100% lossless).")
    if paq is None and not HAS_ZSTD:
        print("Warning: No compression backend found. Data will be stored raw.")

    c = PAQJPCompressor()

    choice = input("\n1) Compress   2) Decompress   3) Full self-test\n> ").strip()
    if choice == "1":
        i = input("Input file: ").strip()
        o = input("Output file: ").strip() or i + ".pjp"
        c.compress_file(i, o)
    elif choice == "2":
        i = input("Compressed file: ").strip()
        o = input("Output file: ").strip() or i.rsplit('.', 1)[0] + ".orig"
        c.decompress_file(i, o)
    elif choice == "3":
        c.full_self_test()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
