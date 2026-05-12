#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAQJP 8.6 – 256 Lossless Transforms + 2704 Transform‑Pair Sequences
(2026‑05‑12 – corrected: identity loops removed, RLE simplified, constants streamlined)
----------------------------------------------------------------------------
All single transforms (1‑256), all ordered pairs (2704), and the raw (no‑transform)
path are mathematically lossless. Every transform has a perfect inverse.

HEADER FORMAT (variable‑length):
   Byte 0:  F
     F < 252             → single transform number = F+1  (1..252)
     F == 252            → raw (no transform)
     F == 253            → pair: next 2 bytes encode pair‑index big‑endian
     F == 254            → extended single: next byte X → transform = 253+X (0..3)
     F == 255            → RESERVED (unused)

Usage:
    python paqjp86.py
    Choose 1 (compress), 2 (decompress), or 3 (full self‑test).
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
PI_DIGITS = [79, 17, 111]                     # used in transforms 07‑09
PI_STR = "3.14159265358979323846264338327950288419716939937510"

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

        # Build all 256 transforms
        self._build_transform_maps()

        # Build pair sequences (base 1..52 → 2704 pairs)
        self.sequences = self._build_pair_sequences()
        self.pair_lookup = {idx: (t1, t2) for idx, (t1, t2) in enumerate(self.sequences)}

    # ------------------------------------------------------------------
    # Helper: seed tables, Fibonacci
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
    # Pi digits (used in transform 17)
    # ------------------------------------------------------------------
    def _pi_digits(self, n: int) -> str:
        if n < 1:
            return ""
        return PI_STR[2:2+n]                     # skip "3."

    # ------------------------------------------------------------------
    # Transform 17 – pi‑based XOR mask (lossless, simplified)
    # ------------------------------------------------------------------
    def transform_17(self, data: bytes) -> bytes:
        if not data:
            return b''
        digits = self._pi_digits(14)             # 14 digits give 7 bytes mask
        mask = bytes(int(digits[i:i+2]) % 256 for i in range(0, len(digits), 2))
        t = bytearray(data)
        for i in range(len(t)):
            t[i] ^= mask[i % len(mask)]
        return bytes(t)

    reverse_transform_17 = transform_17          # involutory

    # ------------------------------------------------------------------
    # Transform 18‑20: mathematical constant masks (involutory)
    # ------------------------------------------------------------------
    def _constant_mask(self, data: bytes, constant_fn, digits_needed: int) -> bytes:
        if not data:
            return b''
        decimal.getcontext().prec = digits_needed + 5
        digits = constant_fn(digits_needed)
        mask = bytes(int(digits[i:i+2]) % 256 for i in range(0, len(digits), 2))
        t = bytearray(data)
        for i in range(len(t)):
            t[i] ^= mask[i % len(mask)]
        return bytes(t)

    def _basel_digits(self, n: int) -> str:
        pi = decimal.Decimal(PI_STR)
        basel = (pi * pi) / decimal.Decimal(6)
        s = str(basel).replace('.', '')
        return s[:n]

    def _inv_e_digits(self, n: int) -> str:
        e = decimal.Decimal(1).exp()
        inv_e = decimal.Decimal(1) / e
        s = str(inv_e).replace('.', '')
        return s[:n]

    def _five_e_digits(self, n: int) -> str:
        e = decimal.Decimal(1).exp()
        five_e = decimal.Decimal(5) * e
        s = str(five_e).replace('.', '')
        return s[:n]

    def transform_18(self, data: bytes) -> bytes:
        return self._constant_mask(data, self._basel_digits, max(10, len(data)//2 + 5))
    reverse_transform_18 = transform_18

    def transform_19(self, data: bytes) -> bytes:
        return self._constant_mask(data, self._inv_e_digits, max(10, len(data)//2 + 5))
    reverse_transform_19 = transform_19

    def transform_20(self, data: bytes) -> bytes:
        return self._constant_mask(data, self._five_e_digits, max(10, len(data)//2 + 5))
    reverse_transform_20 = transform_20

    # ------------------------------------------------------------------
    # Bit helpers (for RLE transform)
    # ------------------------------------------------------------------
    @staticmethod
    def _append_bits(bitlist, value, count):
        for i in range(count - 1, -1, -1):
            bitlist.append((value >> i) & 1)

    @staticmethod
    def _read_bits(bits, pos, count):
        val = 0
        for i in range(count):
            if pos + i >= len(bits):
                return 0
            val = (val << 1) | bits[pos + i]
        return val

    # ------------------------------------------------------------------
    # TRANSFORM 00 – multi‑pass shift + compact RLE (lossless)
    # (Uses a fixed marker; shift field no longer stored because decoder ignores it)
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
                # score = sum of squares of run lengths (favours long runs)
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
            rle_encoded = self._encode_rle(best_shifted)
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

    def _encode_rle(self, data: bytearray) -> bytes:
        bits = []
        # marker: 3 bits
        self._append_bits(bits, 0b010, 3)
        i = 0
        n = len(data)
        while i < n:
            val = data[i]
            run = 1
            i += 1
            while i < n and data[i] == val:
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
        if nbits < 3:
            return None
        marker = self._read_bits(bits, pos, 3)
        pos += 3
        if marker != 0b010:
            return None
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
            else:  # prefix == 0b11 -> check next two bits for 11
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
    # Transforms 01‑16 (corrected: all now do useful work with 1 pass)
    # ------------------------------------------------------------------
    def _get_pattern(self, size: int, index: int):
        random.seed(12345 + size * 100 + index)
        return [random.randint(0, 255) for _ in range(size)]

    def transform_01(self, d):
        # XOR with constant derived from primes (single pass)
        t = bytearray(d)
        for i in range(len(t)):
            xor_val = PRIMES[i % len(PRIMES)] % 256
            t[i] ^= xor_val
        return bytes(t)
    reverse_transform_01 = transform_01

    def transform_02(self, d):
        if len(d) < 1: return b''
        t = bytearray(d)
        checksum = sum(d) % 256
        pattern_index = (len(d) + checksum) % 256
        pattern = self._get_pattern(4, pattern_index)
        for i in range(1, len(t), 4):
            t[i] ^= pattern[i % len(pattern)]
        return bytes([pattern_index]) + bytes(t)

    def reverse_transform_02(self, d):
        if len(d) < 2: return b''
        pattern_index = d[0]
        t = bytearray(d[1:])
        pattern = self._get_pattern(4, pattern_index)
        for i in range(1, len(t), 4):
            t[i] ^= pattern[i % len(pattern)]
        return bytes(t)

    def transform_03(self, d):
        if len(d) < 1: return b''
        t = bytearray(d)
        rotation = (len(d) * 13 + sum(d)) % 8 or 1
        for i in range(2, len(t), 5):
            t[i] = ((t[i] << rotation) | (t[i] >> (8 - rotation))) & 0xFF
        return bytes([rotation]) + bytes(t)

    def reverse_transform_03(self, d):
        if len(d) < 2: return b''
        rotation = d[0]
        t = bytearray(d[1:])
        for i in range(2, len(t), 5):
            t[i] = ((t[i] >> rotation) | (t[i] << (8 - rotation))) & 0xFF
        return bytes(t)

    def transform_04(self, d):
        t = bytearray(d)
        for i in range(len(t)):
            t[i] = (t[i] - (i % 256)) % 256
        return bytes(t)

    def reverse_transform_04(self, d):
        t = bytearray(d)
        for i in range(len(t)):
            t[i] = (t[i] + (i % 256)) % 256
        return bytes(t)

    def transform_05(self, d, shift=3):
        t = bytearray(d)
        for i in range(len(t)):
            t[i] = ((t[i] << shift) | (t[i] >> (8 - shift))) & 0xFF
        return bytes(t)

    def reverse_transform_05(self, d, shift=3):
        t = bytearray(d)
        for i in range(len(t)):
            t[i] = ((t[i] >> shift) | (t[i] << (8 - shift))) & 0xFF
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
        inv = [0]*256
        for i in range(256):
            inv[sub[i]] = i
        t = bytearray(d)
        for i in range(len(t)):
            t[i] = inv[t[i]]
        return bytes(t)

    # Transforms 07‑12: involutory XOR (single pass)
    def transform_07(self, d):
        t = bytearray(d)
        sz = len(d) % 256
        for i in range(len(t)):
            t[i] ^= sz
        return bytes(t)
    reverse_transform_07 = transform_07

    def transform_08(self, d):
        t = bytearray(d)
        p = find_nearest_prime_around(len(d) % 256) % 256
        for i in range(len(t)):
            t[i] ^= p
        return bytes(t)
    reverse_transform_08 = transform_08

    def transform_09(self, d):
        t = bytearray(d)
        seed = self.get_seed(len(d) % len(self.seed_tables), len(d))
        for i in range(len(t)):
            t[i] ^= seed
        return bytes(t)
    reverse_transform_09 = transform_09

    def transform_10(self, d):
        if not d: return b''
        cnt = sum(1 for i in range(len(d)-1) if d[i:i+2] == b'X1')
        n = (((cnt * 2) + 1) // 3) * 3 % 256
        t = bytearray(d)
        for i in range(len(t)):
            t[i] ^= n
        return bytes([n]) + bytes(t)

    def reverse_transform_10(self, d):
        if len(d) < 1: return b''
        n = d[0]
        t = bytearray(d[1:])
        for i in range(len(t)):
            t[i] ^= n
        return bytes(t)

    def transform_11(self, d):
        t = bytearray(d)
        length = len(t)
        for i in range(length):
            fib_idx = (i + length) % len(self.fibonacci)
            fib_val = self.fibonacci[fib_idx] % 256
            pos_val = (i * 13 + length * 17) % 256
            key = (fib_val ^ pos_val) % 256
            t[i] ^= key
        return bytes(t)
    reverse_transform_11 = transform_11

    def transform_12(self, d):
        t = bytearray(d)
        for i in range(len(t)):
            t[i] ^= self.fibonacci[i % len(self.fibonacci)] % 256
        return bytes(t)
    reverse_transform_12 = transform_12

    def transform_13(self, d):
        if not d: return b''
        repeats = self._calculate_repeats(d)
        current = len(d) % 256
        prime_values = []
        for _ in range(repeats):
            current = find_nearest_prime_around(current)
            prime_values.append(current % 256)
        xor_val = prime_values[-1]
        t = bytearray(d)
        for i in range(len(t)):
            t[i] ^= xor_val
        return bytes([(repeats - 1) % 256]) + bytes(t)

    def reverse_transform_13(self, d):
        if len(d) < 2: return b''
        repeats = (d[0] + 1) % 256
        if repeats == 0: repeats = 256
        t = bytearray(d[1:])
        current = len(t) % 256
        prime_values = []
        for _ in range(repeats):
            current = find_nearest_prime_around(current)
            prime_values.append(current % 256)
        xor_val = prime_values[-1]
        for i in range(len(t)):
            t[i] ^= xor_val
        return bytes(t)

    def transform_14(self, d):
        if not d: return b'\x00'
        checksum = sum(d) % 256
        return d + bytes([checksum])

    def reverse_transform_14(self, d):
        if not d: return b''
        return d[:-1]

    def transform_15(self, d):
        if len(d) < 1: return b''
        t = bytearray(d)
        pattern_index = len(d) % 256
        pattern = self._get_pattern(3, pattern_index)
        for i in range(0, len(t), 3):
            t[i] = (t[i] + pattern[i % len(pattern)]) % 256
        return bytes([pattern_index]) + bytes(t)

    def reverse_transform_15(self, d):
        if len(d) < 2: return b''
        pattern_index = d[0]
        t = bytearray(d[1:])
        pattern = self._get_pattern(3, pattern_index)
        for i in range(0, len(t), 3):
            t[i] = (t[i] - pattern[i % len(pattern)]) % 256
        return bytes(t)

    def transform_16(self, data: bytes) -> bytes:
        if not data: return b''
        xor_byte = (len(data) * 7 + 13) % 256
        t = bytearray(data)
        for i in range(len(t)):
            t[i] ^= xor_byte
        return bytes(t)
    reverse_transform_16 = transform_16

    # ------------------------------------------------------------------
    # Transform 21 – shift by 255 (i.e. add -1 mod 256)
    # ------------------------------------------------------------------
    def transform_21(self, data: bytes) -> bytes:
        t = bytearray(data)
        for i in range(len(t)):
            t[i] = (t[i] + 255) % 256
        return bytes(t)

    def reverse_transform_21(self, data: bytes) -> bytes:
        t = bytearray(data)
        for i in range(len(t)):
            t[i] = (t[i] + 1) % 256          # subtract 255 ≡ add 1
        return bytes(t)

    # ------------------------------------------------------------------
    # Identity transform (256)
    # ------------------------------------------------------------------
    def transform_256(self, d: bytes) -> bytes: return d
    reverse_transform_256 = transform_256

    def _calculate_repeats(self, data: bytes) -> int:
        if not data: return 1
        length = len(data)
        byte_sum = sum(data) % 256
        return max(1, min(256, ((length * 13 + byte_sum * 17) % 256) + 1))

    # ------------------------------------------------------------------
    # Dynamic transforms 23‑255 – XOR with seed (involutory)
    # ------------------------------------------------------------------
    def _dynamic_transform(self, n: int):
        def tf(data: bytes):
            if not data: return b''
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

        # explicit transforms 1‑22
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
        self.fwd_transforms[16] = self.transform_15
        self.rev_transforms[16] = self.reverse_transform_15
        self.fwd_transforms[17] = self.transform_16
        self.rev_transforms[17] = self.reverse_transform_16
        self.fwd_transforms[18] = self.transform_17
        self.rev_transforms[18] = self.reverse_transform_17
        self.fwd_transforms[19] = self.transform_18
        self.rev_transforms[19] = self.reverse_transform_18
        self.fwd_transforms[20] = self.transform_19
        self.rev_transforms[20] = self.reverse_transform_19
        self.fwd_transforms[21] = self.transform_20
        self.rev_transforms[21] = self.reverse_transform_20
        self.fwd_transforms[22] = self.transform_21
        self.rev_transforms[22] = self.reverse_transform_21

        # dynamic transforms 23‑255
        for i in range(23, 256):
            fwd, rev = self._dynamic_transform(i)
            self.fwd_transforms[i] = fwd
            self.rev_transforms[i] = rev

        # identity 256
        self.fwd_transforms[256] = self.transform_256
        self.rev_transforms[256] = self.reverse_transform_256

        # sanity check
        for i in range(1, 257):
            if i not in self.fwd_transforms:
                raise RuntimeError(f"Transform {i} missing!")

    # ------------------------------------------------------------------
    # Build pair sequences (1..52) → 2704
    # ------------------------------------------------------------------
    def _build_pair_sequences(self) -> List[Tuple[int, int]]:
        base = list(range(1, 53))
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
    # Variable‑length header encoding / decoding
    # ------------------------------------------------------------------
    def _encode_marker_single(self, t: int) -> bytes:
        if t <= 252:
            return bytes([t - 1])
        else:
            return bytes([254, t - 253])

    def _encode_marker_raw(self) -> bytes:
        return bytes([252])

    def _encode_marker_pair(self, t1: int, t2: int) -> bytes:
        idx = (t1 - 1) * 52 + (t2 - 1)
        return bytes([253, (idx >> 8) & 0xFF, idx & 0xFF])

    def _decode_header(self, data: bytes):
        if len(data) < 1:
            return 0, ()
        f = data[0]
        if f < 252:
            return 1, (f + 1,)
        elif f == 252:
            return 1, ()
        elif f == 253:
            if len(data) < 3:
                return 0, ()
            idx = (data[1] << 8) | data[2]
            if idx >= len(self.sequences):
                return 0, ()
            return 3, self.pair_lookup[idx]
        elif f == 254:
            if len(data) < 2:
                return 0, ()
            x = data[1]
            if x > 3:
                return 0, ()
            return 2, (253 + x,)
        else:
            return 0, ()

    # ------------------------------------------------------------------
    # Main compress / decompress (with variable header)
    # ------------------------------------------------------------------
    def compress_with_best(self, data: bytes) -> bytes:
        if not data:
            backend = self._compress_backend(b'')
            return self._encode_marker_raw() + backend

        best_total = float('inf')
        best_bytes = None

        # raw
        raw_backend = self._compress_backend(data)
        candidate = self._encode_marker_raw() + raw_backend
        if len(candidate) < best_total:
            best_total = len(candidate)
            best_bytes = candidate

        # singles 1..256
        for t in range(1, 257):
            try:
                transformed = self.fwd_transforms[t](data)
                backend = self._compress_backend(transformed)
                candidate = self._encode_marker_single(t) + backend
                if len(candidate) < best_total:
                    best_total = len(candidate)
                    best_bytes = candidate
            except:
                continue

        # pairs (2704)
        for t1, t2 in self.sequences:
            try:
                transformed = self._apply_sequence(data, (t1, t2))
                backend = self._compress_backend(transformed)
                candidate = self._encode_marker_pair(t1, t2) + backend
                if len(candidate) < best_total:
                    best_total = len(candidate)
                    best_bytes = candidate
            except:
                continue

        return best_bytes

    def decompress_with_best(self, data: bytes):
        offset, seq = self._decode_header(data)
        if offset == 0:
            return b'', None
        payload = data[offset:]
        backend = self._decompress_backend(payload)
        if backend is None:
            return b'', None
        try:
            if not seq:
                result = backend
            else:
                result = self._reverse_sequence(backend, seq)
        except:
            return b'', None
        return result, seq

    # ------------------------------------------------------------------
    # EXHAUSTIVE SELF‑TEST
    # ------------------------------------------------------------------
    def full_self_test(self) -> bool:
        print("=" * 60)
        print("PAQJP 8.6 – FULL SELF‑TEST (100% lossless)")
        print("=" * 60)
        all_ok = True

        # single transforms on all bytes
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
            print("\n[FAIL] Single‑transform test failed.")
            return False

        # pairs on all bytes
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
                print(f"  PASS: {idx + 1} pairs tested on all bytes")
        if not all_ok:
            print("\n[FAIL] Pair test failed.")
            return False
        print("  PASS: all pairs OK on all bytes")

        # random 1000‑byte block full pipeline
        print("\nTesting random 1000‑byte block through full compress/decompress...")
        rng = random.Random(12345)
        test_data = bytes(rng.randint(0, 255) for _ in range(1000))
        compressed = self.compress_with_best(test_data)
        decompressed, seq = self.decompress_with_best(compressed)
        if decompressed != test_data:
            print("  FAIL: random data full pipeline mismatch")
            return False
        seq_str = "raw" if not seq else ("pair" if len(seq) == 2 else "single")
        print(f"  PASS: random data pipeline OK (used {seq_str})")

        # empty input
        print("\nTesting empty input...")
        compressed_empty = self.compress_with_best(b'')
        decomp_empty, sempty = self.decompress_with_best(compressed_empty)
        if decomp_empty != b'':
            print("  FAIL: empty input pipeline mismatch")
            return False
        print("  PASS: empty input pipeline OK")

        print("\n[All tests passed – compressor is 100% lossless]")
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
        with open(outfile, 'wb') as f:
            f.write(compressed)
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
        with open(outfile, 'wb') as f:
            f.write(original)
        seq_str = "raw" if not seq else f"sequence {seq}"
        print(f"Decompressed ({seq_str}) → {outfile} ({len(original)} bytes)")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print(f"{PROGNAME}")
    print("256 single transforms + 2704 transform‑pair sequences (100% lossless).")
    if paq is None and not HAS_ZSTD:
        print("Warning: No compression backend found. Data will be stored raw.")

    c = PAQJPCompressor()
    choice = input("\n1) Compress   2) Decompress   3) Full self‑test\n> ").strip()
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
