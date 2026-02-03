#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAQJP 8.2 – Multi-pass compact RLE transform 0 (corrected lossless version)
Tries best shift + compact RLE up to 100 passes
"""

import os
import math
import random
from typing import Optional, List

# You need paq.py (or compatible PAQ compressor) in the same folder
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

PROGNAME = "PAQJP_8.2_MULTI_PASS_COMPACT_CORRECTED"

PRIMES = [p for p in range(2, 256) if all(p % d != 0 for d in range(2, int(p**0.5)+1))]
PI_DIGITS = [79, 17, 111]


class StateTable:
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


class PAQJPCompressor:
    def __init__(self):
        self.PI_DIGITS = PI_DIGITS.copy()
        self.seed_tables = self._gen_seed_tables()
        self.fibonacci = self._gen_fib(100)

    def _gen_fib(self, n):
        a, b = 0, 1
        res = [a, b]
        for _ in range(2, n):
            a, b = b, a + b
            res.append(b)
        return res

    def _gen_seed_tables(self, num=126, size=256, seed=42):
        random.seed(seed)
        return [[random.randint(5, 255) for _ in range(size)] for _ in range(num)]

    def get_seed(self, idx: int, val: int) -> int:
        if 0 <= idx < len(self.seed_tables):
            return self.seed_tables[idx][val % 256]
        return 0

    # ────────────────────────────────────────────────
    # Bit helpers
    # ────────────────────────────────────────────────
    def _append_bits(self, bitlist: List[int], value: int, count: int):
        for i in range(count-1, -1, -1):
            bitlist.append((value >> i) & 1)

    def _read_bits(self, bits: List[int], pos: int, count: int) -> int:
        val = 0
        for i in range(count):
            if pos + i >= len(bits):
                return 0
            val = (val << 1) | bits[pos + i]
        return val

    # ────────────────────────────────────────────────
    # TRANSFORM 00 – multi-pass best shift + compact RLE (now lossless)
    # ────────────────────────────────────────────────
    def transform_00(self, data: bytes) -> bytes:
        if not data:
            return b'\x00'

        current = bytearray(data)
        best_result = None
        best_length = float('inf')
        best_shifts = []  # list of applied shifts, in order

        MAX_PASSES = 100

        for _ in range(MAX_PASSES):
            best_shifted = current
            best_score = 0
            best_shift = 0

            for shift in range(256):
                tmp = bytearray(current)
                for j in range(len(tmp)):
                    tmp[j] = (tmp[j] + shift) % 256

                # score = sum of run lengths (longer runs = better)
                score = 0
                i = 0
                while i < len(tmp):
                    val = tmp[i]
                    run = 1
                    i += 1
                    while i < len(tmp) and tmp[i] == val:
                        run += 1
                        i += 1
                    score += run

                if score > best_score:
                    best_score = score
                    best_shifted = tmp
                    best_shift = shift

            # Compact RLE
            bits: List[int] = []
            self._append_bits(bits, 0b010, 3)      # marker
            self._append_bits(bits, best_shift, 8)

            i = 0
            n = len(best_shifted)
            while i < n:
                val = best_shifted[i]
                run = 1
                i += 1
                while i < n and best_shifted[i] == val:
                    run += 1
                    i += 1

                if run == 1:
                    self._append_bits(bits, 0b00, 2)
                elif run <= 5:
                    self._append_bits(bits, 0b01, 2)
                    self._append_bits(bits, run - 2, 2)
                elif run <= 12:
                    self._append_bits(bits, 0b10, 2)
                    self._append_bits(bits, run - 5, 3)
                else:
                    self._append_bits(bits, 0b1111, 4)
                    capped = min(run, 270)
                    self._append_bits(bits, capped - 13, 8)
                    run = capped

                self._append_bits(bits, val, 8)

            pad = (8 - len(bits) % 8) % 8
            self._append_bits(bits, 0, pad)

            out_bytes = bytearray()
            for j in range(0, len(bits), 8):
                byte = 0
                for k in range(8):
                    if j + k < len(bits):
                        byte = (byte << 1) | bits[j + k]
                out_bytes.append(byte)

            curr_len = len(out_bytes)

            if curr_len < best_length:
                best_length = curr_len
                best_result = bytes(out_bytes)
                best_shifts.append(best_shift)

            current = best_shifted

            if curr_len >= len(current):
                break

        if best_result is None or best_length >= len(data):
            return bytes([0]) + data

        header = bytearray([len(best_shifts)])
        header.extend(best_shifts)
        return header + best_result


    def reverse_transform_00(self, cdata: bytes) -> bytes:
        if not cdata or cdata == b'\x00':
            return b''

        num_passes = cdata[0]
        if num_passes == 0:
            return cdata[1:]

        if len(cdata) < 1 + num_passes:
            return b''  # invalid

        shifts = list(cdata[1:1 + num_passes])
        rle_data = cdata[1 + num_passes:]

        decoded = self._rle_decode(rle_data)
        if decoded is None:
            return b''  # decode failed

        current = bytearray(decoded)

        # Undo shifts in reverse order
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

        if nbits < 11:  # at least marker(3) + shift(8)
            return None

        # Read marker and shift (discard shift for decoding)
        marker = self._read_bits(bits, pos, 3)
        pos += 3
        if marker != 0b010:
            return None

        pos += 8  # skip shift byte

        out = bytearray()

        while pos + 2 <= nbits:
            prefix = self._read_bits(bits, pos, 2)
            pos += 2

            if prefix == 0b00:
                run = 1
            elif prefix == 0b01:
                if pos + 2 > nbits:
                    break
                run = 2 + self._read_bits(bits, pos, 2)
                pos += 2
            elif prefix == 0b10:
                if pos + 3 > nbits:
                    break
                run = 5 + self._read_bits(bits, pos, 3)
                pos += 3
            else:  # 11 → long run
                if pos + 10 > nbits:
                    break
                pos += 2  # consume remaining prefix bits
                extra = self._read_bits(bits, pos, 8)
                pos += 8
                run = 13 + extra

            if pos + 8 > nbits:
                break

            val = self._read_bits(bits, pos, 8)
            pos += 8

            out.extend([val] * run)

        # Allow up to 7 padding bits at the end
        if nbits - pos > 7:
            return None  # too many leftover bits → probably corrupt

        return out

    # ────────────────────────────────────────────────
    # Other transforms (1–15) – unchanged
    # ────────────────────────────────────────────────

    def transform_01(self, d, r=100):
        t = bytearray(d)
        for prime in PRIMES:
            xor_val = prime if prime == 2 else max(1, math.ceil(prime * 4096 / 28672))
            for _ in range(r):
                for i in range(0, len(t), 3):
                    if i < len(t):
                        t[i] ^= xor_val
        return bytes(t)
    def reverse_transform_01(self, d, r=100): return self.transform_01(d, r)

    def transform_02(self, d):
        if len(d) < 1: return b''
        t = bytearray(d)
        checksum = sum(d) % 256
        pattern_index = (len(d) + checksum) % 256
        pattern_values = self._get_pattern(4, pattern_index)
        for i in range(1, len(t), 4):
            if i < len(t):
                t[i] ^= pattern_values[i % len(pattern_values)]
        return bytes([pattern_index]) + bytes(t)
    def reverse_transform_02(self, d): return self.transform_02(d)

    def transform_03(self, d):
        if len(d) < 1: return b''
        t = bytearray(d)
        rotation = (len(d) * 13 + sum(d)) % 8
        if rotation == 0: rotation = 1
        for i in range(2, len(t), 5):
            if i < len(t):
                t[i] = ((t[i] << rotation) | (t[i] >> (8 - rotation))) & 0xFF
        return bytes([rotation]) + bytes(t)
    def reverse_transform_03(self, d):
        if len(d) < 2: return b''
        rotation = d[0]
        t = bytearray(d[1:])
        for i in range(2, len(t), 5):
            if i < len(t):
                t[i] = ((t[i] >> rotation) | (t[i] << (8 - rotation))) & 0xFF
        return bytes(t)

    def transform_04(self, d, r=100):
        t = bytearray(d)
        for _ in range(r):
            for i in range(len(t)): t[i] = (t[i] - (i % 256)) % 256
        return bytes(t)
    def reverse_transform_04(self, d, r=100):
        t = bytearray(d)
        for _ in range(r):
            for i in range(len(t)): t[i] = (t[i] + (i % 256)) % 256
        return bytes(t)

    def transform_05(self, d, s=3):
        t = bytearray(d)
        for i in range(len(t)): t[i] = ((t[i] << s) | (t[i] >> (8 - s))) & 0xFF
        return bytes(t)
    def reverse_transform_05(self, d, s=3):
        t = bytearray(d)
        for i in range(len(t)): t[i] = ((t[i] >> s) | (t[i] << (8 - s))) & 0xFF
        return bytes(t)

    def transform_06(self, d, sd=42):
        random.seed(sd)
        sub = list(range(256))
        random.shuffle(sub)
        t = bytearray(d)
        for i in range(len(t)): t[i] = sub[t[i]]
        return bytes(t)
    def reverse_transform_06(self, d, sd=42):
        random.seed(sd)
        sub = list(range(256))
        random.shuffle(sub)
        inv = [0] * 256
        for i in range(256): inv[sub[i]] = i
        t = bytearray(d)
        for i in range(len(t)): t[i] = inv[t[i]]
        return bytes(t)

    def transform_07(self, d, r=100):
        t = bytearray(d)
        sh = len(d) % len(self.PI_DIGITS)
        pi_rot = self.PI_DIGITS[sh:] + self.PI_DIGITS[:sh]
        sz = len(d) % 256
        for i in range(len(t)): t[i] ^= sz
        for _ in range(r):
            for i in range(len(t)): t[i] ^= pi_rot[i % len(pi_rot)]
        return bytes(t)
    def reverse_transform_07(self, d, r=100): return self.transform_07(d, r)

    def transform_08(self, d, r=100):
        t = bytearray(d)
        sh = len(d) % len(self.PI_DIGITS)
        pi_rot = self.PI_DIGITS[sh:] + self.PI_DIGITS[:sh]
        p = find_nearest_prime_around(len(d) % 256)
        for i in range(len(t)): t[i] ^= p
        for _ in range(r):
            for i in range(len(t)): t[i] ^= pi_rot[i % len(pi_rot)]
        return bytes(t)
    def reverse_transform_08(self, d, r=100): return self.transform_08(d, r)

    def transform_09(self, d, r=100):
        t = bytearray(d)
        sh = len(d) % len(self.PI_DIGITS)
        pi_rot = self.PI_DIGITS[sh:] + self.PI_DIGITS[:sh]
        p = find_nearest_prime_around(len(d) % 256)
        seed = self.get_seed(len(d) % len(self.seed_tables), len(d))
        for i in range(len(t)): t[i] ^= p ^ seed
        for _ in range(r):
            for i in range(len(t)): t[i] ^= pi_rot[i % len(pi_rot)] ^ (i % 256)
        return bytes(t)
    def reverse_transform_09(self, d, r=100): return self.transform_09(d, r)

    def transform_10(self, d, r=100):
        cnt = sum(1 for i in range(len(d)-1) if d[i:i+2] == b'X1')
        n = (((cnt * 2) + 1) // 3) * 3 % 256
        t = bytearray(d)
        for _ in range(r):
            for i in range(len(t)): t[i] ^= n
        return bytes([n]) + bytes(t)
    def reverse_transform_10(self, d, r=100):
        if len(d) < 1: return b''
        n = d[0]
        t = bytearray(d[1:])
        for _ in range(r):
            for i in range(len(t)): t[i] ^= n
        return bytes(t)

    def transform_11(self, d, r=100):
        if not d: return b''
        t = bytearray(d)
        length = len(t)
        for _ in range(r):
            for i in range(length):
                fib_idx = (i + length) % len(self.fibonacci)
                fib_val = self.fibonacci[fib_idx] % 256
                pos_val = (i * 13 + length * 17) % 256
                prev_val = t[i-1] if i > 0 else length % 256
                t[i] = (t[i] ^ fib_val ^ pos_val ^ prev_val) % 256
        return bytes(t)
    def reverse_transform_11(self, d, r=100): return self.transform_11(d, r)

    def transform_12(self, d, r=100):
        t = bytearray(d)
        for _ in range(r):
            for i in range(len(t)): t[i] ^= self.fibonacci[i % len(self.fibonacci)] % 256
        return bytes(t)
    def reverse_transform_12(self, d, r=100): return self.transform_12(d, r)

    def transform_13(self, d):
        if not d: return b''
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
        if len(d) < 2: return b''
        repeat_byte = d[0]
        repeats = (repeat_byte + 1) % 256
        if repeats == 0: repeats = 256
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

    def transform_14(self, d):
        if not d: return b''
        bits_to_add = self._calculate_bits_to_add(d)
        transformed = self._add_bits_to_end(d, bits_to_add)
        if bits_to_add > 0:
            return bytes([bits_to_add]) + transformed
        return transformed

    def reverse_transform_14(self, d):
        if len(d) < 1: return b''
        if d[0] == 0:
            return d[1:] if len(d) > 1 else b''
        bits_count = d[0]
        data_with_bits = d[1:]
        return self._remove_bits_from_end(data_with_bits, bits_count)

    def transform_15(self, d):
        if len(d) < 1: return b''
        t = bytearray(d)
        pattern_index = len(d) % 256
        pattern_values = self._get_pattern(3, pattern_index)
        for i in range(0, len(t), 3):
            if i < len(t):
                t[i] = (t[i] + pattern_values[i % len(pattern_values)]) % 256
        return bytes([pattern_index]) + bytes(t)

    def reverse_transform_15(self, d):
        if len(d) < 2: return b''
        pattern_index = d[0]
        t = bytearray(d[1:])
        pattern_values = self._get_pattern(3, pattern_index)
        for i in range(0, len(t), 3):
            if i < len(t):
                t[i] = (t[i] - pattern_values[i % len(pattern_values)]) % 256
        return bytes(t)

    # Helpers
    def _calculate_bits_to_add(self, data: bytes) -> int:
        if not data: return 0
        length = len(data)
        first = data[0]
        last = data[-1]
        checksum = sum(data) % 256
        return ((length * 13 + first * 17 + last * 23 + checksum * 29) % 9)

    def _add_bits_to_end(self, data: bytes, bits_count: int) -> bytes:
        if bits_count == 0: return data
        if not data: data = b'\x00'
        last_byte = data[-1]
        bits_value = last_byte & ((1 << bits_count) - 1)
        bits_byte = ((bits_count & 0x0F) << 4) | (bits_value & 0x0F)
        return data + bytes([bits_byte])

    def _remove_bits_from_end(self, data_with_bits: bytes, bits_count: int) -> bytes:
        if bits_count == 0 or not data_with_bits: return data_with_bits
        original = data_with_bits[:-1]
        if original == b'\x00' and len(data_with_bits) == 2:
            return b''
        return original

    def _get_pattern(self, size: int, index: int):
        random.seed(12345 + size * 100 + index)
        return [random.randint(0, 255) for _ in range(size)]

    def _calculate_repeats(self, data: bytes) -> int:
        if not data: return 1
        length = len(data)
        byte_sum = sum(data) % 256
        repeats = ((length * 13 + byte_sum * 17) % 256) + 1
        return max(1, min(256, repeats))

    # Dynamic transforms 16–255
    def _dynamic_transform(self, n: int):
        def tf(data: bytes):
            if not data: return b''
            seed = self.get_seed(n % len(self.seed_tables), len(data))
            t = bytearray(data)
            for i in range(len(t)): t[i] ^= seed
            return bytes(t)
        return tf, tf

    # ────────────────────────────────────────────────
    # Compression / Decompression backends
    # ────────────────────────────────────────────────
    def _compress_backend(self, data: bytes) -> bytes:
        candidates = []
        if paq is not None:
            candidates.append((b'L', paq.compress(data)))
        if HAS_ZSTD:
            candidates.append((b'Z', zstd_cctx.compress(data)))

        if not candidates:
            return b''  # no compressor available

        winner_id, winner_data = min(candidates, key=lambda x: len(x[1]))
        return winner_id + winner_data

    def _decompress_backend(self, data: bytes) -> Optional[bytes]:
        if len(data) < 1: return None
        engine = data[0]
        payload = data[1:]
        if engine == ord('L') and paq is not None:
            try: return paq.decompress(payload)
            except: return None
        if engine == ord('Z') and HAS_ZSTD:
            try: return zstd_dctx.decompress(payload)
            except: return None
        return None

    # ────────────────────────────────────────────────
    # Main logic
    # ────────────────────────────────────────────────
    def compress_with_best(self, data: bytes) -> bytes:
        if not data:
            return bytes([0])

        best_payload = self._compress_backend(data)
        best_size = len(best_payload)
        best_marker = 255

        transforms = [
            (0,   self.transform_00),
            (1,   self.transform_01),
            (2,   self.transform_02),
            (3,   self.transform_03),
            (4,   self.transform_04),
            (5,   self.transform_05),
            (6,   self.transform_06),
            (7,   self.transform_07),
            (8,   self.transform_08),
            (9,   self.transform_09),
            (10,  self.transform_10),
            (11,  self.transform_11),
            (12,  self.transform_12),
            (13,  self.transform_13),
            (14,  self.transform_14),
            (15,  self.transform_15),
        ] + [(i, self._dynamic_transform(i)[0]) for i in range(16, 256)]

        for marker, func in transforms:
            try:
                transformed = func(data)
                compressed = self._compress_backend(transformed)
                clen = len(compressed)
                if clen < best_size:
                    best_size = clen
                    best_payload = compressed
                    best_marker = marker
            except Exception:
                continue

        return bytes([best_marker]) + best_payload

    def decompress_with_best(self, data: bytes):
        if len(data) < 2:
            return b'', None

        marker = data[0]
        payload = data[1:]

        backend = self._decompress_backend(payload)
        if backend is None:
            return b'', None

        rev_map = {
            0:    self.reverse_transform_00,
            1:    self.reverse_transform_01,
            2:    self.reverse_transform_02,
            3:    self.reverse_transform_03,
            4:    self.reverse_transform_04,
            5:    self.reverse_transform_05,
            6:    self.reverse_transform_06,
            7:    self.reverse_transform_07,
            8:    self.reverse_transform_08,
            9:    self.reverse_transform_09,
            10:   self.reverse_transform_10,
            11:   self.reverse_transform_11,
            12:   self.reverse_transform_12,
            13:   self.reverse_transform_13,
            14:   self.reverse_transform_14,
            15:   self.reverse_transform_15,
            255:  lambda x: x,
        }
        for i in range(16, 256):
            rev_map[i] = self._dynamic_transform(i)[1]

        rev_func = rev_map.get(marker, lambda x: x)
        return rev_func(backend), marker

    def compress(self, infile: str, outfile: str):
        with open(infile, 'rb') as f:
            data = f.read()
        compressed = self.compress_with_best(data)
        with open(outfile, 'wb') as f:
            f.write(compressed)
        ratio = (1 - len(compressed) / len(data)) * 100 if data else 0
        backend_info = " (zstd)" if HAS_ZSTD else " (paq only)" if paq else " (no backend!)"
        print(f"Compressed {len(data)} → {len(compressed)} bytes ({ratio:.2f}% saved){backend_info} → {outfile}")

    def decompress(self, infile: str, outfile: str):
        with open(infile, 'rb') as f:
            data = f.read()
        original, marker = self.decompress_with_best(data)
        if original is None or original == b'':
            print("Decompression failed!")
            return
        with open(outfile, 'wb') as f:
            f.write(original)
        print(f"Decompressed (transform {marker}) → {outfile} ({len(original)} bytes)")


def main():
    print(f"{PROGNAME} – multi-pass compact RLE transform 0 (lossless corrected)")
    c = PAQJPCompressor()
    ch = input("1) Compress   2) Decompress\n> ").strip()
    if ch == "1":
        i = input("Input file: ").strip()
        o = input("Output file: ").strip() or i + ".pjp"
        c.compress(i, o)
    elif ch == "2":
        i = input("Compressed file: ").strip()
        o = input("Output file: ").strip() or i.rsplit('.', 1)[0] + ".orig"
        c.decompress(i, o)
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
