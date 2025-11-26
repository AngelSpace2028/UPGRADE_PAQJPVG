#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ZLIBJP_6.7_fixed_01_plus
Dictionary-Free Lossless Compressor with zlib level 9
Authors: Jurijus Pacalovas, Vincent Geoghegan

ALGORITHM 11 REWRITTEN:
1. 54 deterministic transforms (0..53)
2. Algorithm 12 (Fibonacci-XOR)
3. r repeats of transform #54
4. **Algorithm 04** – byte-wise subtraction (r = 1..1024)

All other algorithms unchanged. All tests pass.
"""

import os
import sys
import math
import struct
import array
import random
import heapq
import binascii
import logging
import hashlib
import zlib
from datetime import datetime
from enum import Enum
from typing import List, Dict, Tuple, Optional

# === Try importing optional dependencies ===
# zlib is built-in to Python, no need to try import
# Remove the problematic try-except block for zlib

try:
    import qiskit
except ImportError:
    qiskit = None
    logging.warning("qiskit not found. Quantum transforms will be skipped.")

# === Configure Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# === Constants ===
PROGNAME = "ZLIBJP_6.7_fixed_01_plus"
PI_DIGITS_FILE = "pi_digits.txt"
PRIMES = [p for p in range(2, 256) if all(p % d != 0 for d in range(2, int(p**0.5)+1))]
MEM = 1 << 15
MIN_BITS = 2
ZLIB_LEVEL = 9  # Maximum compression level

# === DNA Encoding Table ===
DNA_ENCODING_TABLE = {
    'AAAA': 0b00000, 'AAAC': 0b00001, 'AAAG': 0b00010, 'AAAT': 0b00011,
    'AACC': 0b00100, 'AACG': 0b00101, 'AACT': 0b00110, 'AAGG': 0b00111,
    'AAGT': 0b01000, 'AATT': 0b01001, 'ACCC': 0b01010, 'ACCG': 0b01011,
    'ACCT': 0b01100, 'AGGG': 0b01101, 'AGGT': 0b01110, 'AGTT': 0b01111,
    'CCCC': 0b10000, 'CCCG': 0b10001, 'CCCT': 0b10010, 'CGGG': 0b10011,
    'CGGT': 0b10100, 'CGTT': 0b10101, 'GTTT': 0b10110, 'CTTT': 0b10111,
    'AAAAAAAA': 0b11000, 'CCCCCCCC': 0b11001, 'GGGGGGGG': 0b11010, 'TTTTTTTT': 0b11011,
    'A': 0b11100, 'C': 0b11101, 'G': 0b11110, 'T': 0b11111
}
DNA_DECODING_TABLE = {v: k for k, v in DNA_ENCODING_TABLE.items()}

# === Pi Digits Functions ===
def save_pi_digits(digits: List[int], filename: str = PI_DIGITS_FILE) -> bool:
    try:
        with open(filename, 'w') as f:
            f.write(','.join(str(d) for d in digits))
        logging.info(f"Saved {len(digits)} pi digits to {filename}")
        return True
    except Exception as e:
        logging.error(f"Failed to save pi digits to {filename}: {e}")
        return False

def load_pi_digits(filename: str = PI_DIGITS_FILE, expected_count: int = 3) -> Optional[List[int]]:
    try:
        if not os.path.isfile(filename):
            logging.warning(f"Pi digits file {filename} does not exist")
            return None
        with open(filename, 'r') as f:
            data = f.read().strip()
            if not data:
                logging.warning(f"Pi digits file {filename} is empty")
                return None
            digits = []
            for x in data.split(','):
                if not x.isdigit():
                    logging.warning(f"Invalid integer in {filename}: {x}")
                    return None
                d = int(x)
                if not (0 <= d <= 255):
                    logging.warning(f"Digit out of range in {filename}: {d}")
                    return None
                digits.append(d)
            if len(digits) != expected_count:
                logging.warning(f"Loaded {len(digits)} digits, expected {expected_count}")
                return None
            logging.info(f"Loaded {len(digits)} pi digits from {filename}")
            return digits
    except Exception as e:
        logging.error(f"Failed to load pi digits from {filename}: {e}")
        return None

def generate_pi_digits(num_digits: int = 3, filename: str = PI_DIGITS_FILE) -> List[int]:
    try:
        from mpmath import mp
        mp.dps = num_digits
        pi_digits = [int(d) for d in str(mp.pi)[2:2+num_digits]]
        if len(pi_digits) != num_digits:
            logging.error(f"Generated {len(pi_digits)} digits, expected {num_digits}")
            raise ValueError("Incorrect number of pi digits generated")
        mapped_digits = [(d * 255 // 9) % 256 for d in pi_digits]
        save_pi_digits(mapped_digits, filename)
        return mapped_digits
    except ImportError:
        logging.warning("mpmath not installed, using fallback pi digits")
        fallback_digits = [3, 1, 4]
        mapped_fallback = [(d * 255 // 9) % 256 for d in fallback_digits[:num_digits]]
        save_pi_digits(mapped_fallback, filename)
        return mapped_fallback
    except Exception as e:
        logging.error(f"Failed to generate pi digits: {e}")
        fallback_digits = [3, 1, 4]
        mapped_fallback = [(d * 255 // 9) % 256 for d in fallback_digits[:num_digits]]
        save_pi_digits(mapped_fallback, filename)
        return mapped_fallback

PI_DIGITS = generate_pi_digits(3)

# === Helper Classes and Functions ===
class Filetype(Enum):
    DEFAULT = 0
    JPEG = 1
    TEXT = 3

def is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0: return False
    return True

def find_nearest_prime_around(n):
    offset = 0
    while True:
        if is_prime(n - offset): return n - offset
        if is_prime(n + offset): return n + offset
        offset += 1

# === Algorithm 01: Prime-XOR every 3 bytes ===
def transform_with_prime_xor_every_3_bytes(data: bytes, repeat: int = 100) -> bytes:
    transformed = bytearray(data)
    for prime in PRIMES:
        xor_val = prime if prime == 2 else max(1, math.ceil(prime * 4096 / 28672))
        for _ in range(repeat):
            for i in range(0, len(transformed), 3):
                transformed[i] ^= xor_val
    return bytes(transformed)

def reverse_transform_with_prime_xor_every_3_bytes(data: bytes, repeat: int = 100) -> bytes:
    return transform_with_prime_xor_every_3_bytes(data, repeat)

def transform_with_pattern_chunk(data, chunk_size=4):
    transformed = bytearray()
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        transformed.extend([b ^ 0xFF for b in chunk])
    return bytes(transformed)

# === FULL State Table (256 entries) ===
class StateTable:
    def __init__(self):
        self.table = [
            [1, 2, 0, 0], [3, 5, 1, 0], [4, 6, 0, 1], [7, 10, 2, 0],
            [8, 12, 1, 1], [9, 13, 1, 1], [11, 14, 0, 2], [15, 19, 3, 0],
            [16, 23, 2, 1], [17, 24, 2, 1], [18, 25, 2, 1], [20, 27, 1, 2],
            [21, 28, 1, 2], [22, 29, 1, 2], [26, 30, 0, 3], [31, 33, 4, 0],
            [32, 35, 3, 1], [32, 35, 3, 1], [32, 35, 3, 1], [32, 35, 3, 1],
            [34, 37, 2, 2], [34, 37, 2, 2], [34, 37, 2, 2], [34, 37, 2, 2],
            [34, 37, 2, 2], [34, 37, 2, 2], [36, 39, 1, 3], [36, 39, 1, 3],
            [36, 39, 1, 3], [36, 39, 1, 3], [38, 40, 0, 4], [41, 43, 5, 0],
            [42, 45, 4, 1], [42, 45, 4, 1], [44, 47, 3, 2], [44, 47, 3, 2],
            [46, 49, 2, 3], [46, 49, 2, 3], [48, 51, 1, 4], [48, 51, 1, 4],
            [50, 52, 0, 5], [53, 43, 6, 0], [54, 57, 5, 1], [54, 57, 5, 1],
            [56, 59, 4, 2], [56, 59, 4, 2], [58, 61, 3, 3], [58, 61, 3, 3],
            [60, 63, 2, 4], [60, 63, 2, 4], [62, 65, 1, 5], [62, 65, 1, 5],
            [50, 66, 0, 6], [67, 55, 7, 0], [68, 57, 6, 1], [68, 57, 6, 1],
            [70, 73, 5, 2], [70, 73, 5, 2], [72, 75, 4, 3], [72, 75, 4, 3],
            [74, 77, 3, 4], [74, 77, 3, 4], [76, 79, 2, 5], [76, 79, 2, 5],
            [62, 81, 1, 6], [62, 81, 1, 6], [64, 82, 0, 7], [83, 69, 8, 0],
            [84, 76, 7, 1], [84, 76, 7, 1], [86, 73, 6, 2], [86, 73, 6, 2],
            [44, 59, 5, 3], [44, 59, 5, 3], [58, 61, 4, 4], [58, 61, 4, 4],
            [60, 49, 3, 5], [60, 49, 3, 5], [76, 89, 2, 6], [76, 89, 2, 6],
            [78, 91, 1, 7], [78, 91, 1, 7], [80, 92, 0, 8], [93, 69, 9, 0],
            [94, 87, 8, 1], [94, 87, 8, 1], [96, 45, 7, 2], [96, 45, 7, 2],
            [48, 99, 2, 7], [48, 99, 2, 7], [88, 101, 1, 8], [88, 101, 1, 8],
            [80, 102, 0, 9], [103, 69, 10, 0], [104, 87, 9, 1], [104, 87, 9, 1],
            [106, 57, 8, 2], [106, 57, 8, 2], [62, 109, 2, 8], [62, 109, 2, 8],
            [88, 111, 1, 9], [88, 111, 1, 9], [80, 112, 0, 10], [113, 85, 11, 0],
            [114, 87, 10, 1], [114, 87, 10, 1], [116, 57, 9, 2], [116, 57, 9, 2],
            [62, 119, 2, 9], [62, 119, 2, 9], [88, 121, 1, 10], [88, 121, 1, 10],
            [90, 122, 0, 11], [123, 85, 12, 0], [124, 97, 11, 1], [124, 97, 11, 1],
            [126, 57, 10, 2], [126, 57, 10, 2], [62, 129, 2, 10], [62, 129, 2, 10],
            [98, 131, 1, 11], [98, 131, 1, 11], [90, 132, 0, 12], [133, 85, 13, 0],
            [134, 97, 12, 1], [134, 97, 12, 1], [136, 57, 11, 2], [136, 57, 11, 2],
            [62, 139, 2, 11], [62, 139, 2, 11], [98, 141, 1, 12], [98, 141, 1, 12],
            [90, 142, 0, 13], [143, 95, 14, 0], [144, 97, 13, 1], [144, 97, 13, 1],
            [68, 57, 12, 2], [68, 57, 12, 2], [62, 81, 2, 12], [62, 81, 2, 12],
            [98, 147, 1, 13], [98, 147, 1, 13], [100, 148, 0, 14], [149, 95, 15, 0],
            [150, 107, 14, 1], [150, 107, 14, 1], [108, 151, 1, 14], [108, 151, 1, 14],
            [100, 152, 0, 15], [153, 95, 16, 0], [154, 107, 15, 1], [108, 155, 1, 15],
            [100, 156, 0, 16], [157, 95, 17, 0], [158, 107, 16, 1], [108, 159, 1, 16],
            [100, 160, 0, 17], [161, 105, 18, 0], [162, 107, 17, 1], [108, 163, 1, 17],
            [110, 164, 0, 18], [165, 105, 19, 0], [166, 117, 18, 1], [118, 167, 1, 18],
            [110, 168, 0, 19], [169, 105, 20, 0], [170, 117, 19, 1], [118, 171, 1, 19],
            [110, 172, 0, 20], [173, 105, 21, 0], [174, 117, 20, 1], [118, 175, 1, 20],
            [110, 176, 0, 21], [177, 105, 22, 0], [178, 117, 21, 1], [118, 179, 1, 21],
            [120, 184, 0, 23], [185, 115, 24, 0], [186, 127, 23, 1], [128, 187, 1, 23],
            [120, 188, 0, 24], [189, 115, 25, 0], [190, 127, 24, 1], [128, 191, 1, 24],
            [120, 192, 0, 25], [193, 115, 26, 0], [194, 127, 25, 1], [128, 195, 1, 25],
            [120, 196, 0, 26], [197, 115, 27, 0], [198, 127, 26, 1], [128, 199, 1, 26],
            [120, 200, 0, 27], [201, 115, 28, 0], [202, 127, 27, 1], [128, 203, 1, 27],
            [120, 204, 0, 28], [205, 115, 29, 0], [206, 127, 28, 1], [128, 207, 1, 28],
            [120, 208, 0, 29], [209, 125, 30, 0], [210, 127, 29, 1], [128, 211, 1, 29],
            [130, 212, 0, 30], [213, 125, 31, 0], [214, 137, 30, 1], [138, 215, 1, 30],
            [130, 216, 0, 31], [217, 125, 32, 0], [218, 137, 31, 1], [138, 219, 1, 31],
            [130, 220, 0, 32], [221, 125, 33, 0], [222, 137, 32, 1], [138, 223, 1, 32],
            [130, 224, 0, 33], [225, 125, 34, 0], [226, 137, 33, 1], [138, 227, 1, 33],
            [130, 228, 0, 34], [229, 125, 35, 0], [230, 137, 34, 1], [138, 231, 1, 34],
            [130, 232, 0, 35], [233, 125, 36, 0], [234, 137, 35, 1], [138, 235, 1, 35],
            [130, 236, 0, 36], [237, 125, 37, 0], [238, 137, 36, 1], [138, 239, 1, 36],
            [130, 240, 0, 37], [241, 125, 38, 0], [242, 137, 37, 1], [138, 243, 1, 37],
            [130, 244, 0, 38], [245, 135, 39, 0], [246, 137, 38, 1], [138, 247, 1, 38],
            [140, 248, 0, 39], [249, 135, 40, 0], [250, 69, 39, 1], [80, 251, 1, 39],
            [140, 252, 0, 40], [249, 135, 41, 0], [250, 69, 40, 1], [80, 251, 1, 40],
            [140, 252, 0, 41]
        ]

# === ZLIBJP Compressor (Dictionary-Free) ===
class ZLIBJPCompressor:
    def __init__(self):
        self.PI_DIGITS = PI_DIGITS
        self.PRIMES = PRIMES
        self.seed_tables = self.generate_seed_tables()
        self.SQUARE_OF_ROOT = 2
        self.ADD_NUMBERS = 1
        self.MULTIPLY = 3
        self.fibonacci = self.generate_fibonacci(100)
        self.state_table = StateTable()

    def generate_fibonacci(self, n: int) -> List[int]:
        fib = [0, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        return fib

    def generate_seed_tables(self, num_tables=126, table_size=256, min_val=5, max_val=255, seed=42):
        random.seed(seed)
        return [[random.randint(min_val, max_val) for _ in range(table_size)] for _ in range(num_tables)]

    def get_seed(self, table_idx: int, value: int) -> int:
        if 0 <= table_idx < len(self.seed_tables):
            return self.seed_tables[table_idx][value % len(self.seed_tables[table_idx])]
        return 0

    def create_quantum_transform_circuit(self, transform_idx: int, data_length: int) -> Optional['qiskit.QuantumCircuit']:
        if qiskit is None:
            return None
        circuit = qiskit.QuantumCircuit(9)
        for i in range(9):
            circuit.h(i)
        theta = (transform_idx * data_length) % 512 / 512 * math.pi
        for i in range(9):
            circuit.ry(theta, i)
        for i in range(8):
            circuit.cx(i, i + 1)
        logging.info(f"Defined quantum circuit for transform {transform_idx}, theta={theta:.2f}")
        return circuit

    # ------------------------------------------------------------------
    # zlib helpers (level 9) - FIXED: Remove duplicate methods
    # ------------------------------------------------------------------
    def zlib_compress(self, data):
        if not data:
            return None
        try:
            if isinstance(data, bytearray):
                data = bytes(data)
            elif not isinstance(data, bytes):
                raise TypeError(f"Expected bytes or bytearray, got {type(data)}")
            compressed = zlib.compress(data, level=ZLIB_LEVEL)
            logging.info(f"zlib compression complete (level {ZLIB_LEVEL})")
            return compressed
        except Exception as e:
            logging.error(f"zlib compression failed: {e}")
            return None

    def zlib_decompress(self, data):
        if not data:
            return None
        try:
            decompressed = zlib.decompress(data)
            logging.info("zlib decompression complete")
            return decompressed
        except Exception as e:
            logging.error(f"zlib decompression failed: {e}")
            return None

    # ------------------------------------------------------------------
    # DNA transform (0)
    # ------------------------------------------------------------------
    def transform_genomecompress(self, data: bytes) -> bytes:
        if not data:
            return b''
        try:
            dna_str = data.decode('ascii').upper()
            if not all(c in 'ACGT' for c in dna_str):
                logging.error("Invalid DNA sequence")
                return b''
        except Exception as e:
            logging.error(f"Failed to decode: {e}")
            return b''
        n = len(dna_str)
        r = n % 4
        output_bits = []
        i = 0
        while i < n - r:
            if i + 8 <= n and dna_str[i:i+8] in DNA_ENCODING_TABLE:
                segment = dna_str[i:i+8]
                output_bits.extend([int(b) for b in format(DNA_ENCODING_TABLE[segment], '05b')])
                i += 8
            else:
                segment = dna_str[i:i+4]
                if segment in DNA_ENCODING_TABLE:
                    output_bits.extend([int(b) for b in format(DNA_ENCODING_TABLE[segment], '05b')])
                    i += 4
                else:
                    return b''
        for j in range(i, n):
            segment = dna_str[j]
            output_bits.extend([int(b) for b in format(DNA_ENCODING_TABLE[segment], '05b')])
        bit_str = ''.join(map(str, output_bits))
        byte_length = (len(bit_str) + 7) // 8
        return int(bit_str, 2).to_bytes(byte_length, 'big') if bit_str else b''

    def reverse_transform_genomecompress(self, data: bytes) -> bytes:
        if not data:
            return b''
        bit_str = bin(int.from_bytes(data, 'big'))[2:].zfill(len(data) * 8)
        output = []
        i = 0
        while i < len(bit_str):
            if i + 5 > len(bit_str):
                break
            segment_bits = bit_str[i:i+5]
            segment_val = int(segment_bits, 2)
            if segment_val not in DNA_DECODING_TABLE:
                return b''
            output.append(DNA_DECODING_TABLE[segment_val])
            i += 5
        return ''.join(output).encode('ascii')

    # ------------------------------------------------------------------
    # FIXED ALGORITHM 01
    # ------------------------------------------------------------------
    def transform_01(self, data, repeat=100):
        return transform_with_prime_xor_every_3_bytes(data, repeat)

    def reverse_transform_01(self, data, repeat=100):
        return reverse_transform_with_prime_xor_every_3_bytes(data, repeat)

    # ------------------------------------------------------------------
    # ALGORITHM 03
    # ------------------------------------------------------------------
    def transform_03(self, data):
        return transform_with_pattern_chunk(data)

    def reverse_transform_03(self, data):
        return self.transform_03(data)

    # ------------------------------------------------------------------
    # ALGORITHM 04 – byte-wise subtraction
    # ------------------------------------------------------------------
    def transform_04(self, data, repeat=100):
        if not data:
            return b''
        transformed = bytearray(data)
        for _ in range(repeat):
            for i in range(len(transformed)):
                transformed[i] = (transformed[i] - (i % 256)) % 256
        return bytes(transformed)

    def reverse_transform_04(self, data, repeat=100):
        if not data:
            return b''
        transformed = bytearray(data)
        for _ in range(repeat):
            for i in range(len(transformed)):
                transformed[i] = (transformed[i] + (i % 256)) % 256
        return bytes(transformed)

    # ------------------------------------------------------------------
    # ALGORITHM 05 – rotate left
    # ------------------------------------------------------------------
    def transform_05(self, data, shift=3):
        if not data:
            return b''
        transformed = bytearray(data)
        for i in range(len(transformed)):
            transformed[i] = ((transformed[i] << shift) | (transformed[i] >> (8 - shift))) & 0xFF
        return bytes(transformed)

    def reverse_transform_05(self, data, shift=3):
        if not data:
            return b''
        transformed = bytearray(data)
        for i in range(len(transformed)):
            transformed[i] = ((transformed[i] >> shift) | (transformed[i] << (8 - shift))) & 0xFF
        return bytes(transformed)

    # ------------------------------------------------------------------
    # ALGORITHM 06 – substitution
    # ------------------------------------------------------------------
    def transform_06(self, data, seed=42):
        if not data:
            return b''
        random.seed(seed)
        substitution = list(range(256))
        random.shuffle(substitution)
        transformed = bytearray(data)
        for i in range(len(transformed)):
            transformed[i] = substitution[transformed[i]]
        return bytes(transformed)

    def reverse_transform_06(self, data, seed=42):
        if not data:
            return b''
        random.seed(seed)
        substitution = list(range(256))
        random.shuffle(substitution)
        reverse_substitution = [0] * 256
        for i, v in enumerate(substitution):
            reverse_substitution[v] = i
        transformed = bytearray(data)
        for i in range(len(transformed)):
            transformed[i] = reverse_substitution[transformed[i]]
        return bytes(transformed)

    # ------------------------------------------------------------------
    # ALGORITHMS 07-09 (pi-based)
    # ------------------------------------------------------------------
    def transform_07(self, data, repeat=100):
        if not data:
            return b''
        transformed = bytearray(data)
        pi_length = len(self.PI_DIGITS)
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))
        shift = len(data) % pi_length
        self.PI_DIGITS = self.PI_DIGITS[shift:] + self.PI_DIGITS[:shift]
        size_byte = len(data) % 256
        for i in range(len(transformed)):
            transformed[i] ^= size_byte
        for _ in range(cycles * repeat // 10):
            for i in range(len(transformed)):
                pi_digit = self.PI_DIGITS[i % pi_length]
                transformed[i] ^= pi_digit
        return bytes(transformed)

    def reverse_transform_07(self, data, repeat=100):
        if not data:
            return b''
        transformed = bytearray(data)
        pi_length = len(self.PI_DIGITS)
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))
        for _ in range(cycles * repeat // 10):
            for i in range(len(transformed)):
                pi_digit = self.PI_DIGITS[i % pi_length]
                transformed[i] ^= pi_digit
        size_byte = len(data) % 256
        for i in range(len(transformed)):
            transformed[i] ^= size_byte
        shift = len(data) % pi_length
        self.PI_DIGITS = self.PI_DIGITS[-shift:] + self.PI_DIGITS[:-shift]
        return bytes(transformed)

    def transform_08(self, data, repeat=100):
        if not data:
            return b''
        transformed = bytearray(data)
        pi_length = len(self.PI_DIGITS)
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))
        shift = len(data) % pi_length
        self.PI_DIGITS = self.PI_DIGITS[shift:] + self.PI_DIGITS[:shift]
        size_prime = find_nearest_prime_around(len(data) % 256)
        for i in range(len(transformed)):
            transformed[i] ^= size_prime
        for _ in range(cycles * repeat // 10):
            for i in range(len(transformed)):
                pi_digit = self.PI_DIGITS[i % pi_length]
                transformed[i] ^= pi_digit
        return bytes(transformed)

    def reverse_transform_08(self, data, repeat=100):
        if not data:
            return b''
        transformed = bytearray(data)
        pi_length = len(self.PI_DIGITS)
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))
        for _ in range(cycles * repeat // 10):
            for i in range(len(transformed)):
                pi_digit = self.PI_DIGITS[i % pi_length]
                transformed[i] ^= pi_digit
        size_prime = find_nearest_prime_around(len(data) % 256)
        for i in range(len(transformed)):
            transformed[i] ^= size_prime
        shift = len(data) % pi_length
        self.PI_DIGITS = self.PI_DIGITS[-shift:] + self.PI_DIGITS[:-shift]
        return bytes(transformed)

    def transform_09(self, data, repeat=100):
        if not data:
            return b''
        transformed = bytearray(data)
        pi_length = len(self.PI_DIGITS)
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))
        shift = len(data) % pi_length
        self.PI_DIGITS = self.PI_DIGITS[shift:] + self.PI_DIGITS[:shift]
        size_prime = find_nearest_prime_around(len(data) % 256)
        seed_idx = len(data) % len(self.seed_tables)
        seed_value = self.get_seed(seed_idx, len(data))
        for i in range(len(transformed)):
            transformed[i] ^= size_prime ^ seed_value
        for _ in range(cycles * repeat // 10):
            for i in range(len(transformed)):
                pi_digit = self.PI_DIGITS[i % pi_length]
                transformed[i] ^= pi_digit ^ (i % 256)
        return bytes(transformed)

    def reverse_transform_09(self, data, repeat=100):
        if not data:
            return b''
        transformed = bytearray(data)
        pi_length = len(self.PI_DIGITS)
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))
        for _ in range(cycles * repeat // 10):
            for i in range(len(transformed)):
                pi_digit = self.PI_DIGITS[i % pi_length]
                transformed[i] ^= pi_digit ^ (i % 256)
        size_prime = find_nearest_prime_around(len(data) % 256)
        seed_idx = len(data) % len(self.seed_tables)
        seed_value = self.get_seed(seed_idx, len(data))
        for i in range(len(transformed)):
            transformed[i] ^= size_prime ^ seed_value
        shift = len(data) % pi_length
        self.PI_DIGITS = self.PI_DIGITS[-shift:] + self.PI_DIGITS[:-shift]
        return bytes(transformed)

    # ------------------------------------------------------------------
    # ALGORITHM 10
    # ------------------------------------------------------------------
    def transform_10(self, data, repeat=100):
        if not data:
            return bytes([0])
        transformed = bytearray(data)
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))
        count = sum(1 for i in range(len(data) - 1) if data[i] == 0x58 and data[i + 1] == 0x31)
        n = (((count * self.SQUARE_OF_ROOT) + self.ADD_NUMBERS) // 3) * self.MULTIPLY % 256
        for _ in range(cycles * repeat // 10):
            for i in range(len(transformed)):
                transformed[i] ^= n
        return bytes([n]) + bytes(transformed)

    def reverse_transform_10(self, data, repeat=100):
        if len(data) < 1:
            return b''
        n = data[0]
        transformed = bytearray(data[1:])
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))
        for _ in range(cycles * repeat // 10):
            for i in range(len(transformed)):
                transformed[i] ^= n
        return bytes(transformed)

    # ------------------------------------------------------------------
    # ALGORITHM 11 – NEW: 54 + 12 + r×#54 + **Algorithm 04 prefix**
    # ------------------------------------------------------------------
    def _single_transform(self, data: bytes, idx: int) -> bytes:
        """Deterministic XOR used in first 54 and final r steps."""
        seed = (idx * 0x9E3779B1) & 0xFF
        out = bytearray(data)
        for i, b in enumerate(out):
            out[i] = (b ^ seed ^ (i & 0xFF)) & 0xFF
        return bytes(out)

    def transform_11(self, data: bytes) -> bytes:
        if not data:
            return b''
        cur = data
        # 1. 54 transforms
        for i in range(54):
            cur = self._single_transform(cur, i)
        # 2. Algorithm 12
        cur = self.transform_12(cur, repeat=1)
        # 3. r repeats of transform #54
        r = ((len(data) % 1023) + 1)          # 1 .. 1024
        for _ in range(r):
            cur = self._single_transform(cur, 54)
        # 4. **Algorithm 04** – byte-wise subtraction (used as a reversible prefix)
        #    The value `r` is stored in the first 4 bytes (little-endian) after the subtraction.
        r_bytes = struct.pack('<I', r)       # 4-byte little-endian
        prefix = bytearray(r_bytes)
        for i in range(len(prefix)):
            prefix[i] = (prefix[i] - (i % 256)) % 256
        return bytes(prefix) + cur

    def reverse_transform_11(self, data: bytes) -> bytes:
        if not data:
            return b''
        # 4. Undo Algorithm 04 prefix
        prefix = data[:4]
        r_bytes = bytearray(prefix)
        for i in range(len(r_bytes)):
            r_bytes[i] = (r_bytes[i] + (i % 256)) % 256
        r = struct.unpack('<I', r_bytes)[0]
        payload = data[4:]

        cur = payload
        # 1. Undo r repeats
        for _ in range(r):
            cur = self._single_transform(cur, 54)
        # 2. Undo Algorithm 12
        cur = self.reverse_transform_12(cur, repeat=1)
        # 3. Undo 54 transforms (53..0)
        for i in reversed(range(54)):
            cur = self._single_transform(cur, i)
        return cur

    # ------------------------------------------------------------------
    # ALGORITHM 12 – Fibonacci XOR
    # ------------------------------------------------------------------
    def transform_12(self, data, repeat=100):
        if not data:
            return b''
        transformed = bytearray(data)
        fib_length = len(self.fibonacci)
        for _ in range(repeat):
            for i in range(len(transformed)):
                fib_value = self.fibonacci[i % fib_length] % 256
                transformed[i] ^= fib_value
        return bytes(transformed)

    def reverse_transform_12(self, data, repeat=100):
        return self.transform_12(data, repeat=repeat)

    # ------------------------------------------------------------------
    # ALGORITHM 13
    # ------------------------------------------------------------------
    def transform_13(self, data):
        if not data:
            return b''
        r = (len(data) % 255) + 1
        transformed = bytearray(data)
        for _ in range(r):
            for i in range(len(transformed)):
                transformed[i] ^= (i % 256)
        bit_codes = []
        for b in transformed:
            if b < 4:
                prefix, bits = 0b00, 2
            elif b < 16:
                prefix, bits = 0b01, 4
            else:
                prefix, bits = 0b10, 8
            bit_codes.append(format(prefix, '02b') + format(b, f'0{bits}b'))
        all_bits = ''.join(bit_codes)
        packed = int(all_bits, 2).to_bytes((len(all_bits) + 7) // 8, 'big') if all_bits else b''
        return struct.pack('B', r) + packed

    def reverse_transform_13(self, data):
        if len(data) < 1:
            return b''
        r = data[0]
        packed = data[1:]
        if not packed:
            return b''
        bit_str = bin(int.from_bytes(packed, 'big'))[2:].zfill(len(packed) * 8)
        decoded = []
        i = 0
        while i + 2 <= len(bit_str):
            prefix = int(bit_str[i:i+2], 2)
            i += 2
            if prefix == 0b00:
                bits = 2
            elif prefix == 0b01:
                bits = 4
            elif prefix == 0b10:
                bits = 8
            else:
                break
            if i + bits > len(bit_str):
                break
            v = int(bit_str[i:i+bits], 2)
            if v > 255:
                break
            decoded.append(v)
            i += bits
        rev = bytearray(decoded)
        for _ in range(r):
            for j in range(len(rev)):
                rev[j] ^= (j % 256)
        return bytes(rev)

    # ------------------------------------------------------------------
    # ALGORITHM 14 – rotate left 1 bit
    # ------------------------------------------------------------------
    def transform_14(self, data, repeat=100):
        if not data:
            return b''
        transformed = bytearray(data)
        for _ in range(repeat):
            for i in range(len(transformed)):
                b = transformed[i]
                transformed[i] = ((b << 1) | (b >> 7)) & 0xFF
        return bytes(transformed)

    def reverse_transform_14(self, data, repeat=100):
        if not data:
            return b''
        transformed = bytearray(data)
        for _ in range(repeat):
            for i in range(len(transformed)):
                b = transformed[i]
                transformed[i] = ((b >> 1) | (b << 7)) & 0xFF
        return bytes(transformed)

    # ------------------------------------------------------------------
    # ALGORITHM 15 – 540-byte pi key
    # ------------------------------------------------------------------
    def _pi_key(self, length: int) -> bytes:
        key = bytearray()
        idx = 0
        while len(key) < length:
            key.append(self.PI_DIGITS[idx % len(self.PI_DIGITS)])
            idx += 1
        return bytes(key[:length])

    def transform_15(self, data, repeat=100):
        if not data:
            return b''
        block = 540
        transformed = bytearray(data)
        key = self._pi_key(block)
        for _ in range(repeat):
            for i in range(0, len(transformed), block):
                chunk = transformed[i:i+block]
                for j, b in enumerate(chunk):
                    chunk[j] ^= key[j % block]
                transformed[i:i+block] = chunk
        return bytes(transformed)

    def reverse_transform_15(self, data, repeat=100):
        return self.transform_15(data, repeat)

    # ------------------------------------------------------------------
    # ALGORITHMS 16-255 – Generic transforms with quantum circuits
    # ------------------------------------------------------------------
    def generate_transform_method(self, n):
        # Create quantum circuit for this transform
        circuit = self.create_quantum_transform_circuit(n, 1048576)
        
        def transform(data, repeat=100):
            if not data:
                return b''
            transformed = bytearray(data)
            seed_idx = n % len(self.seed_tables)
            seed_value = self.get_seed(seed_idx, len(data))
            
            # Apply quantum-inspired transformation
            for _ in range(repeat):
                for i in range(len(transformed)):
                    # Use quantum circuit index to influence the transformation
                    quantum_factor = (n * i) % 256
                    transformed[i] = (transformed[i] ^ seed_value ^ quantum_factor) & 0xFF
                    
                    # Additional transformation based on position and transform index
                    pos_factor = (i * n) % 256
                    transformed[i] = (transformed[i] + pos_factor) % 256
                    
            return bytes(transformed)
            
        def reverse_transform(data, repeat=100):
            if not data:
                return b''
            transformed = bytearray(data)
            seed_idx = n % len(self.seed_tables)
            seed_value = self.get_seed(seed_idx, len(data))
            
            for _ in range(repeat):
                for i in range(len(transformed)):
                    # Reverse the position-based transformation first
                    pos_factor = (i * n) % 256
                    transformed[i] = (transformed[i] - pos_factor) % 256
                    
                    # Reverse the quantum-inspired transformation
                    quantum_factor = (n * i) % 256
                    transformed[i] = (transformed[i] ^ seed_value ^ quantum_factor) & 0xFF
                    
            return bytes(transformed)

        return transform, reverse_transform

    # ------------------------------------------------------------------
    # Compression / decompression core - FIXED: Remove duplicate zlib calls
    # ------------------------------------------------------------------
    def compress_with_best_method(self, data, filetype, input_filename, mode="slow", use_quantum=False):
        if not data:
            return bytes([0])

        is_dna = False
        try:
            data_str = data.decode('ascii').upper()
            is_dna = all(c in 'ACGT' for c in data_str)
        except:
            pass

        # Apply Algorithm 07 first as requested
        data = self.transform_07(data)
        
        # Base transformations (0-15)
        base_transformations = [
            (1, self.transform_04), (2, self.transform_01), (3, self.transform_03),
            (5, self.transform_05), (6, self.transform_06), (7, self.transform_07),
            (8, self.transform_08), (9, self.transform_09), (11, self.transform_11),
            (12, self.transform_12), (13, self.transform_13),
            (14, self.transform_14), (15, self.transform_15),
        ]

        # Add DNA transform if applicable
        if is_dna:
            base_transformations = [(0, self.transform_genomecompress)] + base_transformations

        # Add Algorithm 10 for slow mode
        if mode == "slow":
            base_transformations = base_transformations + [(10, self.transform_10)]

        # Generate quantum transforms for algorithms 16-255 if requested
        quantum_transforms = []
        if use_quantum and mode == "slow":
            for i in range(16, 256):
                transform_func, _ = self.generate_transform_method(i)
                quantum_transforms.append((i, transform_func))

        transformations = base_transformations + quantum_transforms

        # Filetype-specific prioritization
        if filetype in [Filetype.JPEG, Filetype.TEXT]:
            prioritized = [(7, self.transform_07), (8, self.transform_08), (9, self.transform_09),
                           (12, self.transform_12), (13, self.transform_13),
                           (4, self.transform_04), (11, self.transform_11),
                           (14, self.transform_14), (15, self.transform_15)]
            if is_dna:
                prioritized = [(0, self.transform_genomecompress)] + prioritized
            if mode == "slow":
                prioritized += [(10, self.transform_10)]
            if use_quantum and mode == "slow":
                prioritized += quantum_transforms
                
            transformations = prioritized + [t for t in transformations if t[0] not in [p[0] for p in prioritized]]

        # Use zlib level 9 as primary compressor
        methods = [('zlib', self.zlib_compress)]
            
        best_compressed = None
        best_size = float('inf')
        best_marker = None

        for marker, transform in transformations:
            transformed = transform(data)
            for method_name, compress_func in methods:
                try:
                    compressed = compress_func(transformed)
                    if compressed is None:
                        continue
                    if len(compressed) < best_size:
                        best_size = len(compressed)
                        best_compressed = compressed
                        best_marker = marker
                        logging.info(f"New best: transform {marker} with {method_name} - {best_size} bytes")
                except Exception as e:
                    logging.warning(f"Compression failed with transform {marker} and {method_name}: {e}")

        if best_compressed is None:
            return bytes([0]) + data

        return bytes([best_marker]) + best_compressed

    def decompress_with_best_method(self, data, use_quantum=False):
        if len(data) < 1:
            return b'', None

        method_marker = data[0]
        compressed_data = data[1:]

        # Base reverse transforms
        reverse_transforms = {
            0: self.reverse_transform_genomecompress,
            1: self.reverse_transform_04, 2: self.reverse_transform_01,
            3: self.reverse_transform_03, 5: self.reverse_transform_05,
            6: self.reverse_transform_06, 7: self.reverse_transform_07,
            8: self.reverse_transform_08, 9: self.reverse_transform_09,
            10: self.reverse_transform_10, 11: self.reverse_transform_11,
            12: self.reverse_transform_12, 13: self.reverse_transform_13,
            14: self.reverse_transform_14, 15: self.reverse_transform_15,
        }

        # Add quantum reverse transforms if needed
        if use_quantum:
            for i in range(16, 256):
                _, reverse_func = self.generate_transform_method(i)
                reverse_transforms[i] = reverse_func

        if method_marker not in reverse_transforms:
            return b'', None

        try:
            # Use zlib decompression
            decompressed = self.zlib_decompress(compressed_data)
                
            if not decompressed:
                return b'', None
                
            result = reverse_transforms[method_marker](decompressed)
            # Apply reverse Algorithm 07 after other transforms
            result = self.reverse_transform_07(result)
            return result, method_marker
        except Exception as e:
            logging.error(f"Decompression failed: {e}")
            return b'', None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compress(self, input_file: str, output_file: str, filetype: Filetype = Filetype.DEFAULT, 
                 mode: str = "slow", use_quantum: bool = False) -> bool:
        try:
            with open(input_file, 'rb') as f:
                data = f.read()
            if not data:
                return False
            compressed = self.compress_with_best_method(data, filetype, input_file, mode, use_quantum)
            with open(output_file, 'wb') as f:
                f.write(compressed)
            orig_size = len(data)
            comp_size = len(compressed)
            ratio = (comp_size / orig_size) * 100 if orig_size > 0 else 0
            logging.info(f"Compressed {input_file} → {output_file} ({comp_size} bytes, {ratio:.2f}% of original)")
            return True
        except Exception as e:
            logging.error(f"Compression failed: {e}")
            return False

    def decompress(self, input_file: str, output_file: str, use_quantum: bool = False) -> bool:
        try:
            with open(input_file, 'rb') as f:
                data = f.read()
            if not data:
                return False
            decompressed, method_marker = self.decompress_with_best_method(data, use_quantum)
            if not decompressed:
                return False
            with open(output_file, 'wb') as f:
                f.write(decompressed)
            comp_size = len(data)
            decomp_size = len(decompressed)
            logging.info(f"Decompressed {input_file} → {output_file} ({decomp_size} bytes, method {method_marker})")
            return True
        except Exception as e:
            logging.error(f"Decompression failed: {e}")
            return False


# ----------------------------------------------------------------------
# Helper: file-type detection
# ----------------------------------------------------------------------
def detect_filetype(filename: str) -> Filetype:
    _, ext = os.path.splitext(filename.lower())
    if ext in ['.jpg', '.jpeg']:
        return Filetype.JPEG
    elif ext in ['.txt', '.dna']:
        try:
            with open(filename, 'r', encoding='ascii') as f:
                sample = f.read(1000)
                if all(c in 'ACGTacgt\n' for c in sample):
                    return Filetype.TEXT
        except:
            pass
        return Filetype.TEXT
    return Filetype.DEFAULT


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    print("ZLIBJP_6.7_fixed_01_plus Compression System (Dictionary-Free)")
    print("Created by Jurijus Pacalovas and Vincent Geoghegan")
    print("Algorithm 11: 54 transforms → Algo12 → r×transform54 → **Algorithm 04 prefix**")
    print(f"Using zlib level {ZLIB_LEVEL} compression")
    print("Options:")
    print("1 - Compress file")
    print("2 - Decompress file")

    compressor = ZLIBJPCompressor()

    try:
        choice = input("Enter 1 or 2: ").strip()
        if choice not in ('1', '2'):
            print("Invalid choice.")
            return
    except (EOFError, KeyboardInterrupt):
        return

    if choice == '1':
        try:
            mode_choice = input("Mode (1=fast, 2=slow): ").strip()
            mode = "fast" if mode_choice == '1' else "slow"
        except:
            mode = "slow"

        # Ask about quantum transforms only in slow mode
        use_quantum = False
        if mode == "slow" and qiskit is not None:
            try:
                quantum_choice = input("Use quantum transforms? (y/n): ").strip().lower()
                use_quantum = quantum_choice == 'y'
            except:
                use_quantum = False

        input_file = input("Input file: ").strip()
        output_file = input("Output file: ").strip()

        if not os.path.exists(input_file):
            print("Input file not found.")
            return

        filetype = detect_filetype(input_file)
        success = compressor.compress(input_file, output_file, filetype, mode, use_quantum)
        if success:
            orig = os.path.getsize(input_file)
            comp = os.path.getsize(output_file)
            ratio = comp / orig * 100 if orig > 0 else 0
            print(f"Compressed: {comp} bytes (Ratio: {ratio:.2f}%)")
            if use_quantum:
                print("Quantum transforms were used.")
        else:
            print("Compression failed.")

    elif choice == '2':
        use_quantum = False
        if qiskit is not None:
            try:
                quantum_choice = input("File uses quantum transforms? (y/n): ").strip().lower()
                use_quantum = quantum_choice == 'y'
            except:
                use_quantum = False

        input_file = input("Input file: ").strip()
        output_file = input("Output file: ").strip()
        if not os.path.exists(input_file):
            print("Input file not found.")
            return
        success = compressor.decompress(input_file, output_file, use_quantum)
        if success:
            print(f"Decompressed to {output_file}")
            if use_quantum:
                print("Quantum transforms were reversed.")
        else:
            print("Decompression failed.")


# ----------------------------------------------------------------------
# SELF-TEST
# ----------------------------------------------------------------------
if __name__ == "__main__":
    comp = ZLIBJPCompressor()
    test_data = b"Algorithm 11 test data: 54 + 12 + prefix!"

    # Test Algo 11 (now with Algorithm 04 prefix)
    c11 = comp.transform_11(test_data)
    d11 = comp.reverse_transform_11(c11)
    assert d11 == test_data, "Algorithm 11 failed!"

    # Test Algorithm 07
    c07 = comp.transform_07(test_data)
    d07 = comp.reverse_transform_07(c07)
    assert d07 == test_data, "Algorithm 07 failed!"

    # Test zlib compression
    zlib_compressed = comp.zlib_compress(test_data)
    zlib_decompressed = comp.zlib_decompress(zlib_compressed)
    assert zlib_decompressed == test_data, "zlib compression failed!"

    print("Algorithm 11 (54 + 12 + r×#54 + Algo04 prefix) PASSED")
    print("Algorithm 07 PASSED")
    print(f"zlib level {ZLIB_LEVEL} compression PASSED")
    print("Starting ZLIBJP Compressor CLI...")
    main()
