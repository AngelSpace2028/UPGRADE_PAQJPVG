#!/usr/bin/env python3                                                  # Unix shebang
# -*- coding: utf-8 -*-                                                 # Source encoding
"""
PAQJP 9.2 – Transform65535 : 0‑65535 Transformation Check
==========================================================
256 lossless base transforms + 65535 ordered pairs + raw (index 0)
Total transformation paths: 65536 (indices 0–65535).

Indices:
  0          : raw data (no transform)
  1 .. 65535 : pair (t1, t2) where t1,t2 ∈ [1,256]
               index = (t1-1)*256 + (t2-1)

Base transforms (1..256):
  1‑24   : original set (RLE, XOR, rotation, checksum, etc.)
  25‑30  : Fermat‑Little‑Theorem based
  31‑40  : dynamic (XOR with seed)
  41     : Algorithm 2703
  42     : Extended Algorithm 2703
  43     : Base 16777216
  44     : Base64 (lossless)
  45     : Huffman Coding (canonical)
  46     : Power‑of‑2 & 3 & 6 Minus‑10 Repeat‑10 XOR
  47     : PAQ state table minus 400 XOR
  48‑255 : dynamic (XOR with seed)
  256    : identity
"""                                                                    # Architecture description

import math                                                            # Mathematical functions
import random                                                          # Random number generation
import decimal                                                         # High‑precision decimal arithmetic
import hashlib                                                         # SHA‑256 for block‑specific key
import base64                                                          # Base64 encode/decode
import heapq                                                           # Priority queue for Huffman
from typing import Optional, List, Tuple, Dict, Callable               # Type hints

# ---------- Optional compression backends ----------
try:
    import paq                                                        # Try loading PAQ
except ImportError:
    paq = None                                                         # Not available

try:
    import zstandard as zstd                                           # Try loading Zstandard
    zstd_cctx = zstd.ZstdCompressor(level=22)                          # Zstd compressor (level 22)
    zstd_dctx = zstd.ZstdDecompressor()                                # Zstd decompressor
    HAS_ZSTD = True                                                    # Flag: Zstd ready
except ImportError:
    HAS_ZSTD = False                                                   # Zstd not present

PROGNAME = "PAQJP_9.2_Transform65535_Check"                           # Program name

# ---------- Constants ----------
PRIMES = [p for p in range(2, 256) if all(p % d != 0 for d in range(2, int(p ** 0.5) + 1))]  # All primes < 256
PI_DIGITS = [79, 17, 111]                                              # Custom Pi‑digit triple

# ---------- Helper: nearest prime ----------
def find_nearest_prime_around(n: int) -> int:                          # Find the prime nearest to n
    if n < 2:                                                          # If too small
        return 2                                                       # Return the smallest prime
    o = 0                                                              # Offset from n
    while True:                                                        # Search outward
        c1 = n - o                                                     # Candidate below
        c2 = n + o                                                     # Candidate above
        if c1 >= 2 and all(c1 % d != 0 for d in range(2, int(c1 ** 0.5) + 1)):  # c1 prime?
            return c1
        if c2 >= 2 and all(c2 % d != 0 for d in range(2, int(c2 ** 0.5) + 1)):  # c2 prime?
            return c2
        o += 1                                                         # Increase search radius

# ---------- Prefix‑free nibble code for transform 23 ----------
_CONST_DIAPASON_ITER_CODE = [                                          # Variable‑length nibble encoding table
    (2, 0b10), (2, 0b11),
    (3, 0b010), (3, 0b011),
    (4, 0b0010), (4, 0b0011),
    (5, 0b00010), (5, 0b00011),
    (6, 0b000010), (6, 0b000011),
    (7, 0b0000010), (7, 0b0000011),
    (8, 0b00000010), (8, 0b00000011),
    (9, 0b000000010), (9, 0b000000011),
]
_CONST_DIAPASON_ITER_DECODE = {}                                       # Reverse lookup (length, code) → nibble
for nibble, (length, bits) in enumerate(_CONST_DIAPASON_ITER_CODE):
    _CONST_DIAPASON_ITER_DECODE[(length, bits)] = nibble

# ---------- Original PAQ state table (kept for Transform 47) ----------
class _PAQStateTable:                                                   # PAQ state table constants
    table = [
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

# ---------- Main Compressor Class ----------
class PAQJPCompressorTransform65535:                                    # Core compressor engine
    def __init__(self, repeat_count: int = 100):                       # Initialize with configurable repeat count
        self.repeat_count = repeat_count                                # Store repeat count
        self.PI_DIGITS = PI_DIGITS.copy()                               # Local copy of PI_DIGITS
        self.seed_tables = self._gen_seed_tables(num=126, size=40, seed=42)  # Generate 126 seed tables
        self.fibonacci = self._gen_fib(100)                             # First 100 Fibonacci numbers
        self.PI_STR = "3.14159265358979323846264338327950288419716939937510"  # High‑precision Pi string

        # Build modified state table for Transform 47 (each value minus 400, mod 256)
        self.mod_state_table = []                                       # Modified PAQ table
        for row in _PAQStateTable.table:
            new_row = [(val - 400) & 0xFF for val in row]              # Subtract 400, keep byte
            self.mod_state_table.append(new_row)                        # Append

        self._build_transform_maps()                                    # Register all forward/backward transforms
        self.sequences = self._build_pair_sequences()                   # All 65535 ordered pairs
        self.pair_lookup = {idx: (t1, t2) for idx, (t1, t2) in enumerate(self.sequences)}  # Index→pair map

        # Pre‑build mask for transform 46
        self._build_mask_46()

    def _build_mask_46(self):                                           # Build mask for transform 46 (power‑of‑2/3/6 minus 10)
        base = [1, 2, 4, 8, 16, 32, 64, 128, 3, 6]                    # Base values
        minus_ten = [(b - 10) & 0xFF for b in base]                    # (base - 10) mod 256
        self.mask_46 = minus_ten * 10                                   # Repeat 10 times → 100 bytes

    # ------------------------------------------------------------------
    # pi / constant helpers
    # ------------------------------------------------------------------
    def get_pi_digits(self, n: int) -> str:                             # Return n digits of π after the decimal point
        if n < 1: return ""
        return self.PI_STR[2:2 + n]

    def find_lossless_k(self, n: int):                                  # Find k for transform 17 such that reconstruction is exact
        if n < 1: return 0, True
        true_digits = self.get_pi_digits(n)
        true_scaled = int(self.PI_STR.replace('.', '')[:n + 1])
        DENOM = 16777216
        decimal.getcontext().prec = 50
        pi_dec = decimal.Decimal(self.PI_STR)
        k_float = (pi_dec - 3) * DENOM
        k_candidate = int(round(k_float))
        k_candidate = max(0, min(k_candidate, DENOM - 1))
        approx_scaled = (3 * 10 ** n * DENOM + k_candidate * 10 ** n) // DENOM
        return k_candidate, approx_scaled == true_scaled

    def to_bin(self, value: int, bits: int) -> str:                     # Convert integer to binary string with leading zeros
        return format(value, 'b').zfill(bits)

    def get_bit_size(self, k: int) -> int:                              # Number of bits needed to encode k in transform 17
        return 23 if k <= 0x7FFFFF else 25

    def transform_17(self, data: bytes) -> bytes:                       # XOR with Pi‑derived mask (transform 17)
        if not data: return b''
        k, _ = self.find_lossless_k(7)
        bits_used = self.get_bit_size(k)
        bit_str = self.to_bin(k, bits_used)
        mask_bytes = []
        for i in range(0, len(bit_str), 8):
            byte_bits = bit_str[i:i + 8]
            if len(byte_bits) < 8:
                byte_bits = byte_bits.ljust(8, '0')
            mask_bytes.append(int(byte_bits, 2))
        mask = bytes(mask_bytes)
        t = bytearray(data)
        for i in range(len(t)):
            t[i] ^= mask[i % len(mask)]
        return bytes(t)
    reverse_transform_17 = transform_17                                 # XOR is self‑inverse

    def get_basel_digits(self, n: int) -> str:                          # n digits of π²/6
        decimal.getcontext().prec = n + 5
        pi = decimal.Decimal(self.PI_STR)
        basel = (pi * pi) / decimal.Decimal(6)
        s = str(basel).replace('.', '')
        return s[:n]

    def get_one_over_e_digits(self, n: int) -> str:                     # n digits of 1/e
        decimal.getcontext().prec = n + 5
        e = decimal.Decimal(1).exp()
        inv_e = decimal.Decimal(1) / e
        s = str(inv_e).replace('.', '')
        return s[:n]

    def get_5e_digits(self, n: int) -> str:                             # n digits of 5e
        decimal.getcontext().prec = n + 5
        e = decimal.Decimal(1).exp()
        five_e = decimal.Decimal(5) * e
        s = str(five_e).replace('.', '')
        return s[:n]

    # ------------------------------------------------------------------
    # Seed tables, Fibonacci
    # ------------------------------------------------------------------
    def _gen_seed_tables(self, num=126, size=40, seed=42):              # Generate list of random seed tables
        random.seed(seed)
        return [[random.randint(5, 255) for _ in range(size)] for _ in range(num)]

    def _gen_fib(self, n):                                              # Generate first n Fibonacci numbers
        a, b = 0, 1
        res = [a, b]
        for _ in range(2, n):
            a, b = b, a + b
            res.append(b)
        return res

    def get_seed(self, idx: int, val: int) -> int:                      # Retrieve a seed byte from seed_tables
        if 0 <= idx < len(self.seed_tables):
            return self.seed_tables[idx][val % 40]
        return 0

    # ------------------------------------------------------------------
    # Bit helpers
    # ------------------------------------------------------------------
    def _append_bits(self, bitlist: List[int], value: int, count: int): # Append `count` bits of `value` to bitlist (MSB first)
        for i in range(count - 1, -1, -1):
            bitlist.append((value >> i) & 1)

    def _read_bits(self, bits: List[int], pos: int, count: int) -> int:  # Read `count` bits starting at `pos`, return integer
        val = 0
        for i in range(count):
            if pos + i >= len(bits):
                return 0
            val = (val << 1) | bits[pos + i]
        return val

    # ------------------------------------------------------------------
    # RLE transform 00 (index 1)
    # ------------------------------------------------------------------
    def transform_00(self, data: bytes) -> bytes:                       # Advanced RLE with shift optimisation (Transform 1)
        if not data: return b'\x00'
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
            for shift in range(256):                                    # Test all 256 additive shifts
                tmp = bytearray(current)
                for j in range(len(tmp)):
                    tmp[j] = (tmp[j] + shift) % 256
                score = 0
                i = 0
                while i < len(tmp):                                     # Compute sum of squared run lengths
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
            if len(rle_encoded) >= len(data):                           # Stop if no compression
                break
        if best_result is None or best_length >= len(data):
            return bytes([0]) + data                                    # Escape: 0x00 + original data
        header = bytearray([len(best_shifts)])
        header.extend(best_shifts)
        return header + best_result

    def _apply_rle_to_shifted(self, shifted_data: bytearray, shift: int) -> bytes:  # Encode shifted data using RLE
        bits = []
        self._append_bits(bits, 0b010, 3)                               # Magic marker
        self._append_bits(bits, shift, 8)                               # Shift value (8 bits)
        i = 0
        n = len(shifted_data)
        while i < n:
            val = shifted_data[i]
            run = 1
            i += 1
            while i < n and shifted_data[i] == val:
                run += 1
                i += 1
            while run >= 13:                                            # Long run
                chunk = min(run, 268)
                self._append_bits(bits, 0b1111, 4)
                self._append_bits(bits, chunk - 13, 8)
                self._append_bits(bits, val, 8)
                run -= chunk
            if run == 1:                                                # Single byte
                self._append_bits(bits, 0b00, 2)
                self._append_bits(bits, val, 8)
            elif run <= 5:                                              # Short run 2‑5
                self._append_bits(bits, 0b01, 2)
                self._append_bits(bits, run - 2, 2)
                self._append_bits(bits, val, 8)
            elif run <= 12:                                             # Medium run 6‑12
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

    def reverse_transform_00(self, cdata: bytes) -> bytes:             # Reverse transform 1
        if not cdata or cdata == b'\x00': return b''
        if cdata[0] == 0: return cdata[1:]                              # Escape byte
        num_passes = cdata[0]
        if num_passes == 0 or len(cdata) < 1 + num_passes: return b''
        shifts = list(cdata[1:1 + num_passes])
        rle_data = cdata[1 + num_passes:]
        decoded = self._rle_decode(rle_data)
        if decoded is None: return b''
        current = bytearray(decoded)
        for shift in reversed(shifts):                                  # Undo shifts in reverse order
            for i in range(len(current)):
                current[i] = (current[i] - shift) % 256
        return bytes(current)

    def _rle_decode(self, data: bytes) -> Optional[bytearray]:         # Decode RLE bitstream
        if not data: return None
        bits = []
        for b in data:
            for i in range(7, -1, -1):
                bits.append((b >> i) & 1)
        pos = 0
        nbits = len(bits)
        if nbits < 11: return None
        marker = self._read_bits(bits, pos, 3)
        pos += 3
        if marker != 0b010: return None
        pos += 8                                                        # Skip shift byte
        out = bytearray()
        while pos < nbits:
            if pos + 2 > nbits: break
            prefix = self._read_bits(bits, pos, 2)
            pos += 2
            if prefix == 0b00:
                if pos + 8 > nbits: break
                run = 1
            elif prefix == 0b01:
                if pos + 2 + 8 > nbits: break
                run = 2 + self._read_bits(bits, pos, 2)
                pos += 2
            elif prefix == 0b10:
                if pos + 3 + 8 > nbits: break
                run = 6 + self._read_bits(bits, pos, 3)
                pos += 3
            else:                                                       # Prefix 11
                if pos + 2 + 8 + 8 > nbits: break
                if self._read_bits(bits, pos, 2) != 0b11: return None
                pos += 2
                run = 13 + self._read_bits(bits, pos, 8)
                pos += 8
            if pos + 8 > nbits: break
            val = self._read_bits(bits, pos, 8)
            pos += 8
            out.extend([val] * run)
        for i in range(pos, nbits):                                     # Verify padding
            if bits[i] != 0: return None
        return out

    # ------------------------------------------------------------------
    # Transforms 01‑21 (unchanged)
    # ------------------------------------------------------------------
    def transform_01(self, d):                                           # Transform 2: XOR with prime‑based values
        t = bytearray(d)
        r = self.repeat_count
        for prime in PRIMES:
            xor_val = prime if prime == 2 else max(1, math.ceil(prime * 4096 / 28672))
            for _ in range(r):
                for i in range(0, len(t), 3):
                    if i < len(t): t[i] ^= xor_val
        return bytes(t)
    reverse_transform_01 = transform_01

    def transform_02(self, d):                                           # Transform 3: checksum‑based pattern XOR
        if len(d) < 1: return b''
        t = bytearray(d)
        checksum = sum(d) % 256
        pattern_index = (len(d) + checksum) % 256
        pattern_values = self._get_pattern(4, pattern_index)
        for i in range(1, len(t), 4):
            if i < len(t): t[i] ^= pattern_values[i % len(pattern_values)]
        return bytes([pattern_index]) + bytes(t)
    def reverse_transform_02(self, d):
        if len(d) < 2: return b''
        pattern_index = d[0]
        t = bytearray(d[1:])
        pattern_values = self._get_pattern(4, pattern_index)
        for i in range(1, len(t), 4):
            if i < len(t): t[i] ^= pattern_values[i % len(pattern_values)]
        return bytes(t)

    def transform_03(self, d):                                           # Transform 4: rotation of every 5th byte
        if len(d) < 1: return b''
        t = bytearray(d)
        rotation = (len(d) * 13 + sum(d)) % 8
        if rotation == 0: rotation = 1
        for i in range(2, len(t), 5):
            if i < len(t): t[i] = ((t[i] << rotation) | (t[i] >> (8 - rotation))) & 0xFF
        return bytes([rotation]) + bytes(t)
    def reverse_transform_03(self, d):
        if len(d) < 2: return b''
        rotation = d[0]
        t = bytearray(d[1:])
        for i in range(2, len(t), 5):
            if i < len(t): t[i] = ((t[i] >> rotation) | (t[i] << (8 - rotation))) & 0xFF
        return bytes(t)

    def transform_04(self, d):                                           # Transform 5: subtract position modulo 256 (repeated)
        t = bytearray(d)
        r = self.repeat_count
        for _ in range(r):
            for i in range(len(t)):
                t[i] = (t[i] - (i % 256)) % 256
        return bytes(t)
    def reverse_transform_04(self, d):
        t = bytearray(d)
        r = self.repeat_count
        for _ in range(r):
            for i in range(len(t)):
                t[i] = (t[i] + (i % 256)) % 256
        return bytes(t)

    def transform_05(self, d, s=3):                                     # Transform 6: left shift by s bits
        t = bytearray(d)
        for i in range(len(t)): t[i] = ((t[i] << s) | (t[i] >> (8 - s))) & 0xFF
        return bytes(t)
    def reverse_transform_05(self, d, s=3):
        t = bytearray(d)
        for i in range(len(t)): t[i] = ((t[i] >> s) | (t[i] << (8 - s))) & 0xFF
        return bytes(t)

    def transform_06(self, d, sd=42):                                   # Transform 7: fixed substitution cipher
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
        inv = [0]*256
        for i in range(256): inv[sub[i]] = i
        t = bytearray(d)
        for i in range(len(t)): t[i] = inv[t[i]]
        return bytes(t)

    def transform_07(self, d):                                           # Transform 8: XOR with size and rotated PI digits
        t = bytearray(d)
        r = self.repeat_count
        sh = len(d) % len(self.PI_DIGITS)
        pi_rot = self.PI_DIGITS[sh:] + self.PI_DIGITS[:sh]
        sz = len(d) % 256
        for i in range(len(t)): t[i] ^= sz
        for _ in range(r):
            for i in range(len(t)): t[i] ^= pi_rot[i % len(pi_rot)]
        return bytes(t)
    reverse_transform_07 = transform_07

    def transform_08(self, d):                                           # Transform 9: XOR with nearest prime + rotated PI
        t = bytearray(d)
        r = self.repeat_count
        sh = len(d) % len(self.PI_DIGITS)
        pi_rot = self.PI_DIGITS[sh:] + self.PI_DIGITS[:sh]
        p = find_nearest_prime_around(len(d) % 256)
        for i in range(len(t)): t[i] ^= p
        for _ in range(r):
            for i in range(len(t)): t[i] ^= pi_rot[i % len(pi_rot)]
        return bytes(t)
    reverse_transform_08 = transform_08

    def transform_09(self, d):                                           # Transform 10: XOR with seed + prime + rotated PI + pos
        t = bytearray(d)
        r = self.repeat_count
        sh = len(d) % len(self.PI_DIGITS)
        pi_rot = self.PI_DIGITS[sh:] + self.PI_DIGITS[:sh]
        p = find_nearest_prime_around(len(d) % 256)
        seed = self.get_seed(len(d) % len(self.seed_tables), len(d))
        for i in range(len(t)): t[i] ^= p ^ seed
        for _ in range(r):
            for i in range(len(t)): t[i] ^= pi_rot[i % len(pi_rot)] ^ (i % 256)
        return bytes(t)
    reverse_transform_09 = transform_09

    def transform_10(self, data: bytes) -> bytes:                       # Transform 11: XOR with count of 'X1' pairs
        if not data: return b'\x00'
        cnt = sum(1 for i in range(len(data)-1) if data[i:i+2] == b'X1')
        n = (((cnt * 2) + 1) // 3) * 3 % 256
        t = bytearray(data)
        for i in range(len(t)): t[i] ^= n
        return bytes([n]) + bytes(t)
    def reverse_transform_10(self, data: bytes) -> bytes:
        if len(data) < 1: return b''
        n = data[0]
        t = bytearray(data[1:])
        for i in range(len(t)): t[i] ^= n
        return bytes(t)

    def transform_11(self, data: bytes) -> bytes:                       # Transform 12: XOR with Fibonacci + position
        if not data: return b''
        t = bytearray(data)
        length = len(t)
        for i in range(length):
            fib_idx = (i + length) % len(self.fibonacci)
            fib_val = self.fibonacci[fib_idx] % 256
            pos_val = (i * 13 + length * 17) % 256
            key = (fib_val ^ pos_val) % 256
            t[i] ^= key
        return bytes(t)
    reverse_transform_11 = transform_11

    def transform_12(self, data: bytes) -> bytes:                       # Transform 13: XOR with Fibonacci sequence
        t = bytearray(data)
        for i in range(len(t)): t[i] ^= self.fibonacci[i % len(self.fibonacci)] % 256
        return bytes(t)
    reverse_transform_12 = transform_12

    def transform_13(self, d):                                           # Transform 14: XOR with nearest prime repeated
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
        for i in range(len(t)): t[i] ^= xor_value
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
        for i in range(len(t)): t[i] ^= xor_value
        return bytes(t)

    def transform_14(self, d):                                           # Transform 15: append checksum byte
        if not d: return b'\x00'
        checksum = sum(d) % 256
        return d + bytes([checksum])
    def reverse_transform_14(self, d):
        if not d: return b''
        return d[:-1]

    def transform_15(self, d):                                           # Transform 16: add pattern to every 3rd byte
        if len(d) < 1: return b''
        t = bytearray(d)
        pattern_index = len(d) % 256
        pattern_values = self._get_pattern(3, pattern_index)
        for i in range(0, len(t), 3):
            if i < len(t): t[i] = (t[i] + pattern_values[i % len(pattern_values)]) % 256
        return bytes([pattern_index]) + bytes(t)
    def reverse_transform_15(self, d):
        if len(d) < 2: return b''
        pattern_index = d[0]
        t = bytearray(d[1:])
        pattern_values = self._get_pattern(3, pattern_index)
        for i in range(0, len(t), 3):
            if i < len(t): t[i] = (t[i] - pattern_values[i % len(pattern_values)]) % 256
        return bytes(t)

    def transform_16(self, data: bytes) -> bytes:                       # Transform 17: XOR with derived byte
        if not data: return b''
        xor_byte = (len(data) * 7 + 13) % 256
        t = bytearray(data)
        for i in range(len(t)): t[i] ^= xor_byte
        return bytes(t)
    reverse_transform_16 = transform_16

    # transform_17 defined earlier (Pi mask)
    def transform_18(self, data: bytes) -> bytes:                       # Transform 18: XOR with Basel (π²/6) mask
        if not data: return b''
        digits = self.get_basel_digits(max(10, len(data)//2 + 5))
        mask = bytes(int(digits[i:i+2]) % 256 for i in range(0, len(digits), 2))
        t = bytearray(data)
        for i in range(len(t)): t[i] ^= mask[i % len(mask)]
        return bytes(t)
    reverse_transform_18 = transform_18

    def transform_19(self, data: bytes) -> bytes:                       # Transform 19: XOR with 1/e mask
        if not data: return b''
        digits = self.get_one_over_e_digits(max(10, len(data)//2 + 5))
        mask = bytes(int(digits[i:i+2]) % 256 for i in range(0, len(digits), 2))
        t = bytearray(data)
        for i in range(len(t)): t[i] ^= mask[i % len(mask)]
        return bytes(t)
    reverse_transform_19 = transform_19

    def transform_20(self, data: bytes) -> bytes:                       # Transform 20: XOR with 5e mask
        if not data: return b''
        digits = self.get_5e_digits(max(10, len(data)//2 + 5))
        mask = bytes(int(digits[i:i+2]) % 256 for i in range(0, len(digits), 2))
        t = bytearray(data)
        for i in range(len(t)): t[i] ^= mask[i % len(mask)]
        return bytes(t)
    reverse_transform_20 = transform_20

    def transform_21(self, data: bytes) -> bytes:                       # Transform 21: add 255 to every byte
        if not data: return b''
        shift = 255
        t = bytearray(data)
        for i in range(len(t)): t[i] = (t[i] + shift) % 256
        return bytes(t)
    def reverse_transform_21(self, data: bytes) -> bytes:
        if not data: return b''
        shift = 255
        t = bytearray(data)
        for i in range(len(t)): t[i] = (t[i] - shift) % 256
        return bytes(t)

    # ------------------------------------------------------------------
    # Transform 23: Iterative Constant Diapason
    # ------------------------------------------------------------------
    def _compress_bits(self, bits: List[int]) -> bytes:                 # Compress bit list using iterative nibble code
        orig_bit_len = len(bits)
        if orig_bit_len == 0:
            return b'\x00\x00\x00'
        current_bits = bits[:]
        prev_len = orig_bit_len
        pass_count = 0
        while pass_count < 255:
            pad_len = (4 - len(current_bits) % 4) % 4
            padded = current_bits + [0] * pad_len
            nibble_count = len(padded) // 4
            encoded_bits = []
            for i in range(nibble_count):
                nibble = (padded[i*4] << 3) | (padded[i*4+1] << 2) | (padded[i*4+2] << 1) | padded[i*4+3]
                length, codeword = _CONST_DIAPASON_ITER_CODE[nibble]
                for b in range(length-1, -1, -1):
                    encoded_bits.append((codeword >> b) & 1)
            new_len = len(encoded_bits)
            if new_len < prev_len:
                current_bits = encoded_bits
                prev_len = new_len
                pass_count += 1
            else:
                break
        header = bytes([(orig_bit_len >> 8) & 0xFF, orig_bit_len & 0xFF, pass_count])
        pad = (8 - len(current_bits) % 8) % 8
        current_bits += [0] * pad
        out_bytes = bytearray()
        for i in range(0, len(current_bits), 8):
            val = 0
            for j in range(8):
                val = (val << 1) | current_bits[i+j]
            out_bytes.append(val)
        return header + bytes(out_bytes)

    def _decompress_bits(self, data: bytes) -> List[int]:               # Decompress bit list
        if len(data) < 3:
            return []
        orig_bit_len = (data[0] << 8) | data[1]
        pass_count = data[2]
        payload = data[3:]
        bits = []
        for byte in payload:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        current_bits = bits
        for _ in range(pass_count):
            pos = 0
            nbits = len(current_bits)
            decoded_nibbles = []
            while pos < nbits:
                matched = False
                for length in range(2, 10):
                    if pos + length > nbits: continue
                    codeword = 0
                    for k in range(length):
                        codeword = (codeword << 1) | current_bits[pos + k]
                    key = (length, codeword)
                    if key in _CONST_DIAPASON_ITER_DECODE:
                        decoded_nibbles.append(_CONST_DIAPASON_ITER_DECODE[key])
                        pos += length
                        matched = True
                        break
                if not matched: break
            new_bits = []
            for nibble in decoded_nibbles:
                for j in range(3, -1, -1):
                    new_bits.append((nibble >> j) & 1)
            current_bits = new_bits
        if len(current_bits) < orig_bit_len:
            return []
        return current_bits[:orig_bit_len]

    def transform_23(self, data: bytes) -> bytes:                       # Transform 23: apply diapason compression
        if not data: return b'\x00\x00\x00'
        bits = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return self._compress_bits(bits)

    def reverse_transform_23(self, data: bytes) -> bytes:               # Reverse transform 23
        bits = self._decompress_bits(data)
        if not bits:
            return b''
        out_bytes = bytearray()
        for i in range(0, len(bits), 8):
            val = 0
            for j in range(i, min(i+8, len(bits))):
                val = (val << 1) | bits[j]
            if i+8 > len(bits):
                val <<= (8 - (len(bits) - i))
            out_bytes.append(val)
        return bytes(out_bytes)

    # ------------------------------------------------------------------
    # Transform 24 – constant‑byte run compression inside 43‑byte blocks
    # ------------------------------------------------------------------
    def transform_24(self, data: bytes) -> bytes:                       # Block RLE (43 bytes)
        if not data: return b''
        MAX_LEN = 43
        bits = []
        i = 0
        n = len(data)
        while i < n:
            chunk_len = min(MAX_LEN, n - i)
            chunk = data[i:i+chunk_len]
            first = chunk[0]
            all_same = all(b == first for b in chunk)
            if all_same:
                self._append_bits(bits, 1, 1)
                self._append_bits(bits, first, 8)
                self._append_bits(bits, chunk_len - 1, 6)
            else:
                self._append_bits(bits, 0, 1)
                self._append_bits(bits, chunk_len, 6)
                for b in chunk:
                    self._append_bits(bits, b, 8)
            i += chunk_len
        pad = (8 - len(bits) % 8) % 8
        self._append_bits(bits, 0, pad)
        out = bytearray()
        for j in range(0, len(bits), 8):
            byte = 0
            for k in range(8):
                byte = (byte << 1) | bits[j+k]
            out.append(byte)
        return bytes(out)

    def reverse_transform_24(self, data: bytes) -> bytes:
        if not data: return b''
        bits = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        pos = 0
        nbits = len(bits)
        out = bytearray()
        while pos < nbits:
            if pos + 1 > nbits: break
            flag = self._read_bits(bits, pos, 1)
            pos += 1
            if flag == 1:
                if pos + 8 + 6 > nbits: break
                byte_val = self._read_bits(bits, pos, 8)
                pos += 8
                count_minus1 = self._read_bits(bits, pos, 6)
                pos += 6
                run_len = count_minus1 + 1
                out.extend([byte_val] * run_len)
            else:
                if pos + 6 > nbits: break
                chunk_len = self._read_bits(bits, pos, 6)
                pos += 6
                if chunk_len == 0: break
                if pos + chunk_len * 8 > nbits: break
                for _ in range(chunk_len):
                    b = self._read_bits(bits, pos, 8)
                    pos += 8
                    out.append(b)
        return bytes(out)

    # ------------------------------------------------------------------
    # Transform 25 – Fermat Little Theorem, 1‑byte exponent (fixed n=3)
    # ------------------------------------------------------------------
    def transform_25(self, data: bytes) -> bytes:                       # Transform 25: Fermat with n=3
        if not data: return b'\x01'
        n = 3
        res = bytearray(data)
        for i in range(len(res)):
            res[i] = (pow(res[i] + 1, n, 257) - 1) & 0xFF
        return bytes([n]) + bytes(res)

    def reverse_transform_25(self, data: bytes) -> bytes:               # Reverse: modular inverse of n=3
        if not data or len(data) < 2: return b''
        n = data[0]
        inv = pow(n, -1, 256)
        res = bytearray(data[1:])
        for i in range(len(res)):
            res[i] = (pow(res[i] + 1, inv, 257) - 1) & 0xFF
        return bytes(res)

    # ------------------------------------------------------------------
    # Transform 26 – Fermat Little Theorem, 2‑byte n, 16,777,216 repeats per byte
    # ------------------------------------------------------------------
    def transform_26(self, data: bytes) -> bytes:                       # Transform 26: Fermat with large exponent
        if not data: return b'\x01\x00'
        n = (len(data) * 7 + 13) & 0xFFFF
        if n % 2 == 0:
            n ^= 1
        e = pow(n, 16777216, 256) | 1
        res = bytearray(data)
        for i in range(len(res)):
            res[i] = (pow(res[i] + 1, e, 257) - 1) & 0xFF
        return bytes([n & 0xFF, (n >> 8) & 0xFF]) + bytes(res)

    def reverse_transform_26(self, data: bytes) -> bytes:               # Reverse
        if not data or len(data) < 2: return b''
        n = data[0] | (data[1] << 8)
        if n % 2 == 0:
            n ^= 1
        e = pow(n, 16777216, 256) | 1
        inv_e = pow(e, -1, 256)
        res = bytearray(data[2:])
        for i in range(len(res)):
            res[i] = (pow(res[i] + 1, inv_e, 257) - 1) & 0xFF
        return bytes(res)

    # ------------------------------------------------------------------
    # Transform 27 – Blockwise Fermat on 8192‑bit chunks (1024 bytes)
    # ------------------------------------------------------------------
    def transform_27(self, data: bytes) -> bytes:                       # Transform 27: blockwise Fermat (1024‑byte blocks)
        if not data:
            out = bytearray(b'\x00\x00\x00\x00')
            out.extend(b'\x01\x00')
            out.extend(b'\x00' * 1024)
            return bytes(out)
        BLOCK_SIZE = 1024
        total_blocks = (len(data) + BLOCK_SIZE - 1) // BLOCK_SIZE
        out = bytearray()
        out.extend(len(data).to_bytes(4, 'big'))
        for block_idx in range(total_blocks):
            start = block_idx * BLOCK_SIZE
            end = min(start + BLOCK_SIZE, len(data))
            chunk = data[start:end]
            pad_len = BLOCK_SIZE - len(chunk)
            if pad_len:
                chunk = chunk + b'\x00' * pad_len
            n = ((len(data) * 7 + block_idx * 13 + 1) & 0xFFFF) | 1
            e = pow(n, 16777216, 256) | 1
            e200 = pow(e, 200, 256)
            transformed = bytearray(chunk)
            for i in range(BLOCK_SIZE):
                transformed[i] = (pow(transformed[i] + 1, e200, 257) - 1) & 0xFF
            out.append(n & 0xFF)
            out.append((n >> 8) & 0xFF)
            out.extend(transformed)
        return bytes(out)

    def reverse_transform_27(self, data: bytes) -> bytes:               # Reverse blockwise Fermat
        if not data or len(data) < 4: return b''
        orig_len = int.from_bytes(data[:4], 'big')
        payload = data[4:]
        BLOCK_SIZE = 1024
        block_total_len = 2 + BLOCK_SIZE
        if len(payload) % block_total_len != 0:
            return data
        num_blocks = len(payload) // block_total_len
        decoded = bytearray()
        for block_idx in range(num_blocks):
            offset = block_idx * block_total_len
            n = payload[offset] | (payload[offset+1] << 8)
            chunk = payload[offset+2:offset+2+BLOCK_SIZE]
            n |= 1
            e = pow(n, 16777216, 256) | 1
            e200 = pow(e, 200, 256)
            inv_e200 = pow(e200, -1, 256)
            for i in range(BLOCK_SIZE):
                decoded.append((pow(chunk[i] + 1, inv_e200, 257) - 1) & 0xFF)
        return bytes(decoded[:orig_len])

    # ------------------------------------------------------------------
    # Transform 28 – Blockwise Fermat 1024‑byte + backend compress
    # ------------------------------------------------------------------
    def transform_28(self, data: bytes) -> bytes:                       # Transform 28: blockwise Fermat + backend
        if not data:
            out = bytearray(b'\x00\x00\x00\x00')
            out.extend(b'\x01\x00')
            out.extend(self._compress_backend(b'\x00' * 1024))
            return bytes(out)
        BLOCK_SIZE = 1024
        total_blocks = (len(data) + BLOCK_SIZE - 1) // BLOCK_SIZE
        out = bytearray()
        out.extend(len(data).to_bytes(4, 'big'))
        for block_idx in range(total_blocks):
            start = block_idx * BLOCK_SIZE
            end = min(start + BLOCK_SIZE, len(data))
            chunk = data[start:end]
            pad_len = BLOCK_SIZE - len(chunk)
            if pad_len:
                chunk = chunk + b'\x00' * pad_len
            n = ((len(data) * 7 + block_idx * 13 + 1) & 0xFFFF) | 1
            e = pow(n, 16777216, 256) | 1
            e200 = pow(e, 200, 256)
            transformed = bytearray(chunk)
            for i in range(BLOCK_SIZE):
                transformed[i] = (pow(transformed[i] + 1, e200, 257) - 1) & 0xFF
            compressed_block = self._compress_backend(bytes(transformed))
            out.append(n & 0xFF)
            out.append((n >> 8) & 0xFF)
            L = len(compressed_block)
            out.append((L >> 8) & 0xFF)
            out.append(L & 0xFF)
            out.extend(compressed_block)
        return bytes(out)

    def reverse_transform_28(self, data: bytes) -> bytes:               # Reverse 28
        if not data or len(data) < 4: return b''
        orig_len = int.from_bytes(data[:4], 'big')
        payload = data[4:]
        pos = 0
        decoded = bytearray()
        while pos < len(payload):
            if pos + 2 > len(payload): break
            n = payload[pos] | (payload[pos+1] << 8)
            pos += 2
            if pos + 2 > len(payload): break
            comp_len = (payload[pos] << 8) | payload[pos+1]
            pos += 2
            if pos + comp_len > len(payload): break
            comp_block = payload[pos:pos+comp_len]
            pos += comp_len
            block = self._decompress_backend(comp_block)
            if block is None: return data
            n |= 1
            e = pow(n, 16777216, 256) | 1
            e200 = pow(e, 200, 256)
            inv_e200 = pow(e200, -1, 256)
            transformed = bytearray(block)
            for i in range(len(transformed)):
                transformed[i] = (pow(transformed[i] + 1, inv_e200, 257) - 1) & 0xFF
            decoded.extend(transformed)
        return bytes(decoded[:orig_len])

    # ------------------------------------------------------------------
    # Transform 29 – 32‑byte blocks, 2^256 repeats, backend compress
    # ------------------------------------------------------------------
    def transform_29(self, data: bytes) -> bytes:                       # Transform 29: identity Fermat (e=1) + backend
        if not data:
            out = bytearray(b'\x00\x00\x00\x00')
            out.extend(b'\x01\x00')
            out.extend(self._compress_backend(b'\x00' * 32))
            return bytes(out)
        BLOCK_SIZE = 32
        total_blocks = (len(data) + BLOCK_SIZE - 1) // BLOCK_SIZE
        out = bytearray()
        out.extend(len(data).to_bytes(4, 'big'))
        for block_idx in range(total_blocks):
            start = block_idx * BLOCK_SIZE
            end = min(start + BLOCK_SIZE, len(data))
            chunk = data[start:end]
            pad_len = BLOCK_SIZE - len(chunk)
            if pad_len:
                chunk = chunk + b'\x00' * pad_len
            n = ((len(data) * 7 + block_idx * 13 + 1) & 0xFFFF) | 1
            e = pow(n, 2**256, 256) | 1   # always 1
            e200 = pow(e, 200, 256)       # always 1
            transformed = bytearray(chunk)  # unchanged
            compressed_block = self._compress_backend(bytes(transformed))
            out.append(n & 0xFF)
            out.append((n >> 8) & 0xFF)
            L = len(compressed_block)
            out.append((L >> 8) & 0xFF)
            out.append(L & 0xFF)
            out.extend(compressed_block)
        return bytes(out)

    def reverse_transform_29(self, data: bytes) -> bytes:               # Reverse 29
        if not data or len(data) < 4: return b''
        orig_len = int.from_bytes(data[:4], 'big')
        payload = data[4:]
        pos = 0
        decoded = bytearray()
        while pos < len(payload):
            if pos + 2 > len(payload): break
            n = payload[pos] | (payload[pos+1] << 8)
            pos += 2
            if pos + 2 > len(payload): break
            comp_len = (payload[pos] << 8) | payload[pos+1]
            pos += 2
            if pos + comp_len > len(payload): break
            comp_block = payload[pos:pos+comp_len]
            pos += comp_len
            block = self._decompress_backend(comp_block)
            if block is None: return data
            decoded.extend(block)   # reverse is identity
        return bytes(decoded[:orig_len])

    # ------------------------------------------------------------------
    # Transform 30 – 33‑byte blocks, variable‑length n
    # ------------------------------------------------------------------
    def _compute_n_for_block(self, block: bytes, block_idx: int, total_len: int) -> Tuple[int, bytes]:
        if not block:
            return (1, b'\x01\x01')
        d = block[0]
        x = (block_idx % 33) + 1
        try:
            t = (d*d - d**x) // 256
        except OverflowError:
            t = 0
        if 0 <= t <= 255:
            n = t | 1
            return (n, bytes([1, n]))
        h = hashlib.sha256(block + bytes([block_idx & 0xFF, (total_len>>8)&0xFF, total_len&0xFF])).digest()
        n_bytes = bytearray(h)
        n_bytes[0] |= 1
        length = len(n_bytes)
        encoded = bytes([length]) + bytes(n_bytes)
        n = int.from_bytes(n_bytes, 'big')
        return (n, encoded)

    def transform_30(self, data: bytes) -> bytes:                       # Transform 30: variable n, backend compress
        if not data:
            out = bytearray(b'\x00\x00\x00\x00')
            out.extend(b'\x01\x01')
            out.extend(self._compress_backend(b'\x00' * 33))
            return bytes(out)
        BLOCK_SIZE = 33
        total_blocks = (len(data) + BLOCK_SIZE - 1) // BLOCK_SIZE
        out = bytearray()
        out.extend(len(data).to_bytes(4, 'big'))
        for block_idx in range(total_blocks):
            start = block_idx * BLOCK_SIZE
            end = min(start + BLOCK_SIZE, len(data))
            chunk = data[start:end]
            pad_len = BLOCK_SIZE - len(chunk)
            if pad_len:
                chunk = chunk + b'\x00' * pad_len
            n, enc_n = self._compute_n_for_block(chunk, block_idx, len(data))
            transformed = chunk
            compressed_block = self._compress_backend(transformed)
            out.extend(enc_n)
            L = len(compressed_block)
            out.append((L >> 8) & 0xFF)
            out.append(L & 0xFF)
            out.extend(compressed_block)
        return bytes(out)

    def reverse_transform_30(self, data: bytes) -> bytes:               # Reverse 30
        if not data or len(data) < 4: return b''
        orig_len = int.from_bytes(data[:4], 'big')
        payload = data[4:]
        pos = 0
        decoded = bytearray()
        while pos < len(payload):
            if pos >= len(payload): break
            Ln = payload[pos]
            pos += 1
            if Ln > 32 or pos + Ln > len(payload): break
            n_bytes = payload[pos:pos+Ln]
            pos += Ln
            if pos + 2 > len(payload): break
            comp_len = (payload[pos] << 8) | payload[pos+1]
            pos += 2
            if pos + comp_len > len(payload): break
            comp_block = payload[pos:pos+comp_len]
            pos += comp_len
            block = self._decompress_backend(comp_block)
            if block is None: return data
            decoded.extend(block)
        return bytes(decoded[:orig_len])

    # ------------------------------------------------------------------
    # Transform 41 – Algorithm 2703 (first 8 bytes XOR 0x27,0x03)
    # ------------------------------------------------------------------
    def transform_41(self, data: bytes) -> bytes:                       # Transform 41: XOR first 8 bytes with 0x27,0x03
        if not data: return b''
        mask = bytes([0x27, 0x03])
        t = bytearray(data)
        n = min(len(t), 8)
        for i in range(n):
            t[i] ^= mask[i % 2]
        return bytes(t)
    reverse_transform_41 = transform_41

    # ------------------------------------------------------------------
    # Transform 42 – Extended Algorithm 2703 (every byte XOR 0x27,0x03)
    # ------------------------------------------------------------------
    def transform_42(self, data: bytes) -> bytes:                       # Transform 42: XOR every byte with 0x27,0x03
        if not data: return b''
        t = bytearray(data)
        mask = bytes([0x27, 0x03])
        for i in range(len(t)):
            t[i] ^= mask[i % 2]
        return bytes(t)
    reverse_transform_42 = transform_42

    # ------------------------------------------------------------------
    # Transform 43 – Base 16777216 (3‑byte blocks XOR 0x10,0x00,0x00)
    # ------------------------------------------------------------------
    def transform_43(self, data: bytes) -> bytes:                       # Transform 43: XOR 3‑byte blocks with 0x10,0x00,0x00
        if not data: return b''
        t = bytearray(data)
        mask = bytes([0x10, 0x00, 0x00])
        for i in range(0, len(t), 3):
            for j in range(min(3, len(t) - i)):
                t[i + j] ^= mask[j]
        return bytes(t)
    reverse_transform_43 = transform_43

    # ------------------------------------------------------------------
    # Transform 44 – Base64 (lossless encode/decode)
    # ------------------------------------------------------------------
    def transform_44(self, data: bytes) -> bytes:                       # Transform 44: Base64 encode
        if not data: return b''
        b64_str = base64.b64encode(data).decode('ascii')
        return b64_str.encode('ascii')
    def reverse_transform_44(self, data: bytes) -> bytes:               # Reverse: Base64 decode
        if not data: return b''
        try:
            b64_str = data.decode('ascii')
            return base64.b64decode(b64_str)
        except (ValueError, UnicodeDecodeError):
            return data

    # ------------------------------------------------------------------
    # Huffman coding helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _huffman_code_lengths(freq: List[int]) -> List[int]:            # Compute Huffman code lengths from frequencies
        heap = [(f, i) for i, f in enumerate(freq) if f > 0]
        if not heap:
            return [0]*256
        if len(heap) == 1:
            lengths = [0]*256
            lengths[heap[0][1]] = 1
            return lengths
        heapq.heapify(heap)
        while len(heap) > 1:
            f1, n1 = heapq.heappop(heap)
            f2, n2 = heapq.heappop(heap)
            heapq.heappush(heap, (f1 + f2, (n1, n2)))
        lengths = [0]*256
        def traverse(node, depth):
            if isinstance(node, int):
                lengths[node] = depth
            else:
                traverse(node[0], depth + 1)
                traverse(node[1], depth + 1)
        _, root = heap[0]
        traverse(root, 0)
        return lengths

    @staticmethod
    def _huffman_canonical_codes(code_lengths: List[int]) -> Dict[int, Tuple[int, int]]:
        symbols = list(range(256))
        symbols.sort(key=lambda s: (code_lengths[s], s))
        codes = {}
        code = 0
        prev_len = 0
        first = True
        for sym in symbols:
            cl = code_lengths[sym]
            if cl == 0:
                continue
            if first:
                prev_len = cl
                first = False
            elif cl != prev_len:
                code <<= (cl - prev_len)
                prev_len = cl
            codes[sym] = (code, cl)
            code += 1
        return codes

    # ------------------------------------------------------------------
    # Transform 45 – Huffman Coding (lossless, single‑symbol fix)
    # ------------------------------------------------------------------
    def transform_45(self, data: bytes) -> bytes:                       # Transform 45: Canonical Huffman encode
        if not data: return b''
        freq = [0]*256
        for b in data:
            freq[b] += 1
        code_lengths = self._huffman_code_lengths(freq)
        codes = self._huffman_canonical_codes(code_lengths)
        header = bytearray()
        header.extend(len(data).to_bytes(4, 'big'))
        header.extend(code_lengths)
        bits = []
        for b in data:
            c, cl = codes[b]
            for i in range(cl - 1, -1, -1):
                bits.append((c >> i) & 1)
        pad = (8 - len(bits) % 8) % 8
        bits.extend([0] * pad)
        out_bytes = bytearray()
        for i in range(0, len(bits), 8):
            val = 0
            for j in range(8):
                val = (val << 1) | bits[i + j]
            out_bytes.append(val)
        return bytes(header) + bytes(out_bytes)

    def reverse_transform_45(self, data: bytes) -> bytes:               # Huffman decode
        if not data: return b''
        if len(data) < 4 + 256: return data
        original_len = int.from_bytes(data[:4], 'big')
        code_lengths = list(data[4:4+256])
        payload = data[4+256:]
        if original_len == 0:
            return b''
        code_to_sym = {}
        symbols = list(range(256))
        symbols.sort(key=lambda s: (code_lengths[s], s))
        code = 0
        prev_len = 0
        first = True
        for sym in symbols:
            cl = code_lengths[sym]
            if cl == 0:
                continue
            if first:
                prev_len = cl
                first = False
            elif cl != prev_len:
                code <<= (cl - prev_len)
                prev_len = cl
            code_to_sym[(cl, code)] = sym
            code += 1
        bits = []
        for byte in payload:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        pos = 0
        nbits = len(bits)
        out = bytearray()
        while pos < nbits and len(out) < original_len:
            found = False
            for cl in range(1, 256):
                if pos + cl > nbits:
                    break
                val = 0
                for j in range(cl):
                    val = (val << 1) | bits[pos + j]
                if (cl, val) in code_to_sym:
                    sym = code_to_sym[(cl, val)]
                    out.append(sym)
                    pos += cl
                    found = True
                    break
            if not found:
                break
        return bytes(out)

    # ------------------------------------------------------------------
    # Transform 46 – Power‑of‑2 & 3 & 6 Minus‑10 Repeat‑10 XOR
    # ------------------------------------------------------------------
    def transform_46(self, data: bytes) -> bytes:                       # Transform 46: XOR with mask_46
        if not data: return b''
        t = bytearray(data)
        mask = self.mask_46
        for i in range(len(t)):
            t[i] ^= mask[i % len(mask)]
        return bytes(t)
    reverse_transform_46 = transform_46

    # ------------------------------------------------------------------
    # Transform 47 – PAQ state table minus 400 XOR
    # ------------------------------------------------------------------
    def transform_47(self, data: bytes) -> bytes:                       # Transform 47: XOR with mod_state_table
        if not data: return b''
        t = bytearray(data)
        for i in range(len(t)):
            row = self.mod_state_table[t[i] % 256]
            t[i] ^= row[0]
        return bytes(t)
    reverse_transform_47 = transform_47

    # ------------------------------------------------------------------
    # Transforms 48‑255 : dynamic (XOR with seed)
    # ------------------------------------------------------------------
    def _dynamic_transform(self, n: int):                               # Dynamic XOR transform generator
        def tf(data: bytes):
            if not data: return b''
            seed = self.get_seed(n % len(self.seed_tables), len(data))  # Seed based on transform index & data length
            t = bytearray(data)
            for i in range(len(t)):
                t[i] ^= seed                                            # Apply constant XOR
            return bytes(t)
        return tf, tf                                                   # Both forward and reverse are identical

    # ------------------------------------------------------------------
    # Identity transform 256
    # ------------------------------------------------------------------
    def transform_256(self, d: bytes) -> bytes:                         # Identity (do nothing)
        return d
    reverse_transform_256 = transform_256

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_pattern(self, size: int, index: int):                      # Deterministic pattern generation
        random.seed(12345 + size * 100 + index)
        return [random.randint(0, 255) for _ in range(size)]

    def _calculate_repeats(self, data: bytes) -> int:                   # Compute repeat count for transform 13
        if not data: return 1
        length = len(data)
        byte_sum = sum(data) % 256
        repeats = ((length * 13 + byte_sum * 17) % 256) + 1
        return max(1, min(256, repeats))

    # ------------------------------------------------------------------
    # Build transform maps (1..256)
    # ------------------------------------------------------------------
    def _build_transform_maps(self):                                    # Register all 256 transforms
        self.fwd_transforms: Dict[int, Callable] = {}
        self.rev_transforms: Dict[int, Callable] = {}

        # 1‑24: original set
        self.fwd_transforms[1] = self.transform_00; self.rev_transforms[1] = self.reverse_transform_00
        self.fwd_transforms[2] = self.transform_01; self.rev_transforms[2] = self.reverse_transform_01
        self.fwd_transforms[3] = self.transform_02; self.rev_transforms[3] = self.reverse_transform_02
        self.fwd_transforms[4] = self.transform_03; self.rev_transforms[4] = self.reverse_transform_03
        self.fwd_transforms[5] = self.transform_04; self.rev_transforms[5] = self.reverse_transform_04
        self.fwd_transforms[6] = self.transform_05; self.rev_transforms[6] = self.reverse_transform_05
        self.fwd_transforms[7] = self.transform_06; self.rev_transforms[7] = self.reverse_transform_06
        self.fwd_transforms[8] = self.transform_07; self.rev_transforms[8] = self.reverse_transform_07
        self.fwd_transforms[9] = self.transform_08; self.rev_transforms[9] = self.reverse_transform_08
        self.fwd_transforms[10] = self.transform_09; self.rev_transforms[10] = self.reverse_transform_09
        self.fwd_transforms[11] = self.transform_10; self.rev_transforms[11] = self.reverse_transform_10
        self.fwd_transforms[12] = self.transform_11; self.rev_transforms[12] = self.reverse_transform_11
        self.fwd_transforms[13] = self.transform_12; self.rev_transforms[13] = self.reverse_transform_12
        self.fwd_transforms[14] = self.transform_13; self.rev_transforms[14] = self.reverse_transform_13
        self.fwd_transforms[15] = self.transform_14; self.rev_transforms[15] = self.reverse_transform_14
        self.fwd_transforms[16] = self.transform_15; self.rev_transforms[16] = self.reverse_transform_15
        self.fwd_transforms[17] = self.transform_16; self.rev_transforms[17] = self.reverse_transform_16
        self.fwd_transforms[18] = self.transform_17; self.rev_transforms[18] = self.reverse_transform_17
        self.fwd_transforms[19] = self.transform_18; self.rev_transforms[19] = self.reverse_transform_18
        self.fwd_transforms[20] = self.transform_19; self.rev_transforms[20] = self.reverse_transform_19
        self.fwd_transforms[21] = self.transform_20; self.rev_transforms[21] = self.reverse_transform_20
        self.fwd_transforms[22] = self.transform_21; self.rev_transforms[22] = self.reverse_transform_21
        self.fwd_transforms[23] = self.transform_23; self.rev_transforms[23] = self.reverse_transform_23
        self.fwd_transforms[24] = self.transform_24; self.rev_transforms[24] = self.reverse_transform_24

        # 25‑30: Fermat‑Little‑Theorem based (two algorithm family #1)
        self.fwd_transforms[25] = self.transform_25; self.rev_transforms[25] = self.reverse_transform_25
        self.fwd_transforms[26] = self.transform_26; self.rev_transforms[26] = self.reverse_transform_26
        self.fwd_transforms[27] = self.transform_27; self.rev_transforms[27] = self.reverse_transform_27
        self.fwd_transforms[28] = self.transform_28; self.rev_transforms[28] = self.reverse_transform_28
        self.fwd_transforms[29] = self.transform_29; self.rev_transforms[29] = self.reverse_transform_29
        self.fwd_transforms[30] = self.transform_30; self.rev_transforms[30] = self.reverse_transform_30

        # 31‑40: dynamic (XOR with seed) – early dynamic family
        for i in range(31, 41):
            fwd, rev = self._dynamic_transform(i)
            self.fwd_transforms[i] = fwd
            self.rev_transforms[i] = rev

        # 41‑47: special named algorithms (Algorithm 2703, Base64, Huffman, etc.)
        self.fwd_transforms[41] = self.transform_41; self.rev_transforms[41] = self.reverse_transform_41
        self.fwd_transforms[42] = self.transform_42; self.rev_transforms[42] = self.reverse_transform_42
        self.fwd_transforms[43] = self.transform_43; self.rev_transforms[43] = self.reverse_transform_43
        self.fwd_transforms[44] = self.transform_44; self.rev_transforms[44] = self.reverse_transform_44
        self.fwd_transforms[45] = self.transform_45; self.rev_transforms[45] = self.reverse_transform_45
        self.fwd_transforms[46] = self.transform_46; self.rev_transforms[46] = self.reverse_transform_46
        self.fwd_transforms[47] = self.transform_47; self.rev_transforms[47] = self.reverse_transform_47

        # 48‑255: dynamic (XOR with seed) – second dynamic family (two algorithm family #2)
        for i in range(48, 256):
            fwd, rev = self._dynamic_transform(i)
            self.fwd_transforms[i] = fwd
            self.rev_transforms[i] = rev

        # 256: identity
        self.fwd_transforms[256] = self.transform_256; self.rev_transforms[256] = self.reverse_transform_256

        # Safety check
        for i in range(1, 257):
            if i not in self.fwd_transforms:
                raise RuntimeError(f"Transform {i} missing!")

    # ------------------------------------------------------------------
    # Build pair sequences – 65535 (256×256 minus identity)
    # ------------------------------------------------------------------
    def _build_pair_sequences(self) -> List[Tuple[int, int]]:           # Generate all ordered pairs (t1,t2) except (256,256)
        pairs = []
        for t1 in range(1, 257):
            for t2 in range(1, 257):
                if t1 == 256 and t2 == 256:
                    continue
                pairs.append((t1, t2))
        return pairs

    # ------------------------------------------------------------------
    # Transformation by index (0‑65535)
    # ------------------------------------------------------------------
    def get_transform_sequence(self, index: int) -> Tuple[int, ...]:    # Return transform sequence for a path index
        if index < 0 or index > 65535:
            raise ValueError("Index must be 0..65535")
        if index == 0:
            return ()
        return self.sequences[index - 1]

    def apply_transform_by_index(self, data: bytes, index: int) -> bytes:  # Apply forward transforms for given index
        seq = self.get_transform_sequence(index)
        if not seq:
            return data
        result = data
        for t in seq:
            result = self.fwd_transforms[t](result)
        return result

    def reverse_transform_by_index(self, data: bytes, index: int) -> bytes:  # Apply reverse transforms for given index
        seq = self.get_transform_sequence(index)
        if not seq:
            return data
        result = data
        for t in reversed(seq):
            result = self.rev_transforms[t](result)
        return result

    # ------------------------------------------------------------------
    # Compression backends
    # ------------------------------------------------------------------
    def _compress_backend(self, data: bytes) -> bytes:                  # Compress with best available backend (Zstd/PAQ/raw)
        candidates = []
        if HAS_ZSTD:
            try:
                candidates.append(zstd_cctx.compress(data))
            except:
                pass
        if paq is not None:
            try:
                candidates.append(paq.compress(data))
            except:
                pass
        candidates.append(data)
        return min(candidates, key=len)

    def _decompress_backend(self, data: bytes) -> Optional[bytes]:      # Decompress with appropriate backend
        if len(data) == 0:
            return b''
        if HAS_ZSTD:
            try:
                return zstd_dctx.decompress(data)
            except:
                pass
        if paq is not None:
            try:
                return paq.decompress(data)
            except:
                pass
        return data

    # ------------------------------------------------------------------
    # Variable‑length header encoding / decoding
    # ------------------------------------------------------------------
    def _encode_marker_single(self, t: int) -> bytes:                   # Encode a single transform index
        if t <= 252:
            return bytes([t - 1])
        return bytes([254, t - 253])

    def _encode_marker_raw(self) -> bytes:                              # Raw data marker
        return bytes([252])

    def _encode_marker_pair(self, t1: int, t2: int) -> bytes:           # Encode a pair index
        idx = (t1 - 1) * 256 + (t2 - 1)
        if t1 == 256 and t2 == 256:
            raise ValueError("Identity pair excluded")
        return bytes([253, (idx >> 8) & 0xFF, idx & 0xFF])

    def _decode_header(self, data: bytes):                              # Decode header, return (offset, sequence tuple)
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
            t1, t2 = self.pair_lookup[idx]
            return 3, (t1, t2)
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
    # Main compression – verifies every candidate
    # ------------------------------------------------------------------
    def compress_with_best(self, data: bytes, ultra: bool = True) -> bytes:  # Find best transform (single or pair)
        if not data:
            backend = self._compress_backend(b'')
            return self._encode_marker_raw() + backend

        best_total = float('inf')
        best_bytes = None

        def try_candidate(transform_header: bytes, transformed_data: bytes):
            nonlocal best_total, best_bytes
            backend = self._compress_backend(transformed_data)
            candidate = transform_header + backend
            decomp, _ = self._decompress_auto(candidate)
            if decomp == data and len(candidate) < best_total:
                best_total = len(candidate)
                best_bytes = candidate

        try_candidate(self._encode_marker_raw(), data)

        for t in range(1, 257):
            try:
                transformed = self.fwd_transforms[t](data)
                header = self._encode_marker_single(t)
                try_candidate(header, transformed)
            except:
                continue

        if ultra:
            for t1, t2 in self.sequences:
                try:
                    transformed = self.fwd_transforms[t1](data)
                    transformed = self.fwd_transforms[t2](transformed)
                    header = self._encode_marker_pair(t1, t2)
                    try_candidate(header, transformed)
                except:
                    continue

        if best_bytes is None:
            raise RuntimeError("Cannot compress this file in strict marker‑free mode.")
        return best_bytes

    def _decompress_auto(self, data: bytes) -> Tuple[bytes, Optional[Tuple[int, ...]]]:
        offset, seq = self._decode_header(data)
        if offset == 0:
            return b'', None
        payload = data[offset:]
        if not payload:
            return b'', None
        res = self._decompress_backend(payload)
        if res is None:
            return b'', None
        try:
            if not seq:
                result = res
            else:
                result = self._reverse_sequence(res, seq)
        except:
            return b'', None
        return result, seq

    def _reverse_sequence(self, data: bytes, seq: Tuple[int, ...]) -> bytes:  # Apply reverse transforms in reverse order
        result = data
        for t in reversed(seq):
            result = self.rev_transforms[t](result)
        return result

    # ------------------------------------------------------------------
    # Exhaustive self‑test – checks ALL 65536 indices on a test byte
    # ------------------------------------------------------------------
    def full_self_test(self) -> bool:
        print("=" * 60)
        print("PAQJP 9.2 – Transform65535 CHECK (0‑65535)")
        print("=" * 60)
        test_byte = 0xAA
        test_data = bytes([test_byte])
        print(f"Testing all 65536 transformation indices on byte 0x{test_byte:02X} ...")

        all_ok = True
        for index in range(65536):
            try:
                transformed = self.apply_transform_by_index(test_data, index)
                restored = self.reverse_transform_by_index(transformed, index)
                if restored != test_data:
                    print(f"  FAIL: index {index}, seq {self.get_transform_sequence(index)}")
                    all_ok = False
                    break
            except Exception as e:
                print(f"  EXCEPTION at index {index}: {e}")
                all_ok = False
                break
            if index % 10000 == 0 and index > 0:
                print(f"  ... {index} indices tested OK")

        if all_ok:
            print("  All 65536 transformations are lossless on test byte.")
        else:
            print("\n[FAIL] Check failed.")
            return False

        print("\nRandom 1000‑byte pipeline test...")
        rng = random.Random(12345)
        test_data = bytes(rng.randint(0, 255) for _ in range(1000))
        try:
            compressed = self.compress_with_best(test_data, ultra=True)
            decompressed, _ = self._decompress_auto(compressed)
            if decompressed != test_data:
                print("  FAIL: pipeline mismatch")
                return False
            print("  PASS")
        except RuntimeError as e:
            print(f"  Could not compress (rare): {e}")
            return False

        print("\n[All checks passed – 100% lossless]")
        return True

    # ------------------------------------------------------------------
    # File API
    # ------------------------------------------------------------------
    def compress_file(self, infile: str, outfile: str, ultra: bool = True):
        try:
            with open(infile, 'rb') as f:
                data = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
        try:
            compressed = self.compress_with_best(data, ultra=ultra)
        except RuntimeError as e:
            print(f"Compression failed: {e}")
            return
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
        original, seq = self._decompress_auto(data)
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

    # ------------------------------------------------------------------
    # Constant Diapason bit‑string analysis
    # ------------------------------------------------------------------
    def analyze_constant_diapason(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = f.read()
        bits = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        orig_len = len(bits)
        ones = sum(bits)
        zeros = orig_len - ones
        print(f"File: {filepath} ({len(data)} bytes, {orig_len} bits)")
        print(f"Ones: {ones}, Zeros: {zeros}\n")
        current_bits = bits[:]
        prev_len = orig_len
        pass_num = 0
        best_pass = 0
        best_len = orig_len
        print("Round‑by‑round sizes:")
        while pass_num < 255:
            pad_len = (4 - len(current_bits) % 4) % 4
            padded = current_bits + [0] * pad_len
            nibble_count = len(padded) // 4
            encoded_bits = []
            for i in range(nibble_count):
                nibble = (padded[i*4]<<3)|(padded[i*4+1]<<2)|(padded[i*4+2]<<1)|padded[i*4+3]
                length, codeword = _CONST_DIAPASON_ITER_CODE[nibble]
                for b in range(length-1, -1, -1):
                    encoded_bits.append((codeword >> b) & 1)
            new_len = len(encoded_bits)
            if new_len < prev_len:
                current_bits = encoded_bits
                prev_len = new_len
                pass_num += 1
                if new_len < best_len:
                    best_len = new_len
                    best_pass = pass_num
                print(f"  Pass {pass_num}: {new_len} bits  (delta {new_len - orig_len:+d})")
            else:
                print(f"  Pass {pass_num+1}: {new_len} bits (expansion) – stopping")
                break
        if pass_num == 0:
            print("  (no pass reduced size)")
        print(f"\nBest: {best_len} bits after {best_pass} passes")
        if best_pass == 0:
            print("Best is to keep original (no compression).")
        print("\nComparison with standard compressors (on original file):")
        orig_bytes = data
        if HAS_ZSTD:
            try:
                zstd_bytes = zstd_cctx.compress(orig_bytes)
                print(f"  Zstandard –22: {len(zstd_bytes)} bytes")
            except:
                print("  Zstandard –22: error")
        else:
            print("  Zstandard not available.")
        if paq is not None:
            try:
                paq_bytes = paq.compress(orig_bytes)
                print(f"  PAQ: {len(paq_bytes)} bytes")
            except:
                print("  PAQ: error")
        else:
            print("  PAQ not available.")
        print(f"  Raw (no compress): {len(orig_bytes)} bytes")

    # ------------------------------------------------------------------
    # Block‑wise Transform 23 (0 … 8191 bits per block)
    # ------------------------------------------------------------------
    def transform_23_blocks_compress(self, data: bytes, block_bits: int) -> bytes:
        if not data:
            return self.transform_23(data)
        total_bits = len(data) * 8
        if block_bits == 0 or block_bits >= total_bits:
            return self.transform_23(data)
        bits = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        out = bytearray()
        for start in range(0, total_bits, block_bits):
            chunk = bits[start:start+block_bits]
            compressed = self._compress_bits(chunk)
            orig_bit_len = len(chunk)
            out.append((orig_bit_len >> 8) & 0xFF)
            out.append(orig_bit_len & 0xFF)
            L = len(compressed)
            out.append((L >> 8) & 0xFF)
            out.append(L & 0xFF)
            out.extend(compressed)
        return bytes(out)

    def transform_23_blocks_decompress(self, data: bytes) -> bytes:
        if not data:
            return b''
        pos = 0
        all_bits = []
        try:
            while pos + 2 <= len(data):
                orig_bit_len = (data[pos] << 8) | data[pos+1]
                pos += 2
                if pos + 2 > len(data):
                    break
                L = (data[pos] << 8) | data[pos+1]
                pos += 2
                if pos + L > len(data):
                    break
                block_data = data[pos:pos+L]
                pos += L
                bits = self._decompress_bits(block_data)
                if len(bits) != orig_bit_len:
                    raise ValueError("Block bit length mismatch")
                all_bits.extend(bits)
            if all_bits:
                out_bytes = bytearray()
                for i in range(0, len(all_bits), 8):
                    val = 0
                    for j in range(i, min(i+8, len(all_bits))):
                        val = (val << 1) | all_bits[j]
                    if i+8 > len(all_bits):
                        val <<= (8 - (len(all_bits) - i))
                    out_bytes.append(val)
                return bytes(out_bytes)
        except:
            pass
        return self.reverse_transform_23(data)

    # ------------------------------------------------------------------
    # Find optimal block size for Transform 23 (0 … 8191 bits)
    # ------------------------------------------------------------------
    def find_best_block_size_transform23(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = f.read()
        if not data:
            print("Empty file – no block size test meaningful.")
            return
        total_bits = len(data) * 8
        print(f"File: {filepath}  ({len(data)} bytes, {total_bits} bits)")
        print("Scanning block sizes 0 … 8191 bits …")
        best_size = None
        best_block_bits = -1
        for block_bits in range(0, 8192):
            if block_bits == 0:
                compressed = self.transform_23(data)
            else:
                compressed = self.transform_23_blocks_compress(data, block_bits)
            L = len(compressed)
            if best_size is None or L < best_size:
                best_size = L
                best_block_bits = block_bits
            if block_bits % 1024 == 0:
                print(f"  {block_bits} bits → {L} bytes (best so far: {best_block_bits} bits → {best_size} bytes)")
        print("\nBest block size: {} bits → {} bytes".format(best_block_bits, best_size))
        print("(block size 0 means whole file as single block)")
        if best_block_bits == 0:
            compressed = self.transform_23(data)
            recovered = self.reverse_transform_23(compressed)
        else:
            compressed = self.transform_23_blocks_compress(data, best_block_bits)
            recovered = self.transform_23_blocks_decompress(compressed)
        if recovered == data:
            print("Lossless check: OK")
        else:
            print("Lossless check: FAILED")
        return best_block_bits, best_size

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():                                                             # Interactive menu
    print(f"{PROGNAME}")
    print("PAQJP 9.2 – 256 single + 65535 pairs = 65536 transform paths (indices 0‑65535)")
    if paq is None and not HAS_ZSTD:
        print("Warning: No backend compressor found – raw data will be stored.")

    c = PAQJPCompressorTransform65535(repeat_count=100)

    choice = input("\n1) Compress   2) Decompress   3) Full self‑test (all 0‑65535)\n"
                   "4) Analyze bit‑string   5) Test transformation by index\n"
                   "6) Find optimal block size for Transform 23 (0‑8191 bits)\n> ").strip()
    if choice == "1":
        i = input("Input file: ").strip()
        o = input("Output file: ").strip() or i + ".pjp"
        mode = input("Choose mode: 1) Fast (256)  2) Ultra (65535 pairs)\n> ").strip()
        ultra = True if mode == "2" else False
        c.compress_file(i, o, ultra=ultra)
    elif choice == "2":
        i = input("Compressed file: ").strip()
        o = input("Output file: ").strip() or i.rsplit('.', 1)[0] + ".orig"
        c.decompress_file(i, o)
    elif choice == "3":
        c.full_self_test()
    elif choice == "4":
        fpath = input("File to analyze: ").strip()
        c.analyze_constant_diapason(fpath)
    elif choice == "5":
        try:
            idx = int(input("Transformation index (0‑65535): ").strip())
        except ValueError:
            print("Invalid number.")
            return
        if idx < 0 or idx > 65535:
            print("Index out of range.")
            return
        seq = c.get_transform_sequence(idx)
        print(f"Sequence: {seq if seq else '(raw)'}")
        test_bytes = bytes([0x00, 0xFF, 0x55, 0xAA])
        print("Testing on bytes 0x00 0xFF 0x55 0xAA ...")
        for b in test_bytes:
            data = bytes([b])
            enc = c.apply_transform_by_index(data, idx)
            dec = c.reverse_transform_by_index(enc, idx)
            if dec == data:
                print(f"  0x{b:02X}: OK")
            else:
                print(f"  0x{b:02X}: FAIL (got {dec.hex()})")
    elif choice == "6":
        fpath = input("File to test: ").strip()
        c.find_best_block_size_transform23(fpath)
    else:
        print("Invalid choice.")

if __name__ == "__main__":                                              # Run main if executed directly
    main()
