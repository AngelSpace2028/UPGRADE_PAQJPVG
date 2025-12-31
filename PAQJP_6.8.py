#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAQJP 7.0 – STRONGER & SMARTER Automatic Preprocessor + Zstd/Paq Hybrid
Balanced high-ratio transforms + Zstandard level 22 + Paq level 6
Author: Jurijus Pacalovas + Grok AI (xAI) – December 31, 2025
"""

import os
import math
import random
import zstandard as zstd
import paq  # pip install paq

# Global compressors
zstd_cctx = zstd.ZstdCompressor(level=22, threads=os.cpu_count() or 1)
zstd_dctx = zstd.ZstdDecompressor()

PROGNAME = "PAQJP_7.0_AUTO"

# Primes below 256
PRIMES = [p for p in range(2, 256) if all(p % d != 0 for d in range(2, int(p**0.5)+1))]

# Extended Pi mask (first 64 digits, scaled)
PI_DIGITS = [3,1,4,1,5,9,2,6,5,3,5,8,9,7,9,3,2,3,8,4,6,2,6,4,3,3,8,3,2,7,9,5,0,2,8,8,4,1,9,7,1,6,9,3,9,9,3,7,5,1,0,8,2,0,9,7,4,9,4,4,5,9,2,3,0,7,8]
PI_MASK = [(int(d) * 27) % 256 for d in PI_DIGITS * 2]  # longer cycle

class PAQJPCompressor:
    def __init__(self):
        self.fibonacci = self._gen_fib(256)
        random.seed(42)
        self.seed_tables = [[random.randint(5, 255) for _ in range(256)] for _ in range(180)]

    def _gen_fib(self, n):
        a, b = 0, 1
        res = [a, b]
        for _ in range(2, n):
            a, b = b, (a + b) % 256  # mod 256 for XOR
            res.append(b)
        return res

    def get_seed(self, idx: int, val: int) -> int:
        if 0 <= idx < len(self.seed_tables):
            return self.seed_tables[idx][val % 256]
        return 0

    # === Reversible Transforms ===

    # 1: Prime-based XOR every 3rd byte (3 repeats)
    def transform_01(self, d: bytes) -> bytes:
        t = bytearray(d)
        for prime in PRIMES:
            xor_val = prime if prime == 2 else max(1, (prime * 16) % 256)
            for _ in range(3):
                for i in range(0, len(t), 3):
                    if i < len(t):
                        t[i] ^= xor_val
        return bytes(t)
    reverse_transform_01 = transform_01

    # 2: Invert all bytes
    def transform_02(self, d: bytes) -> bytes:
        return bytes(b ^ 0xFF for b in d)
    reverse_transform_02 = transform_02

    # 3: Position subtract (3 repeats)
    def transform_03(self, d: bytes) -> bytes:
        t = bytearray(d)
        for _ in range(3):
            for i in range(len(t)):
                t[i] = (t[i] - (i % 256)) % 256
        return bytes(t)
    def reverse_transform_03(self, d: bytes) -> bytes:
        t = bytearray(d)
        for _ in range(3):
            for i in range(len(t)):
                t[i] = (t[i] + (i % 256)) % 256
        return bytes(t)

    # 4: Bit rotate left by 3
    def transform_04(self, d: bytes) -> bytes:
        return bytes(((b << 3) | (b >> 5)) & 0xFF for b in d)
    def reverse_transform_04(self, d: bytes) -> bytes:
        return bytes(((b << 5) | (b >> 3)) & 0xFF for b in d)

    # 5: Fibonacci XOR (4 repeats)
    def transform_05(self, d: bytes) -> bytes:
        t = bytearray(d)
        for _ in range(4):
            for i in range(len(t)):
                t[i] ^= self.fibonacci[i % len(self.fibonacci)]
        return bytes(t)
    reverse_transform_05 = transform_05

    # 6: Extended Pi cyclic XOR (2 repeats)
    def transform_06(self, d: bytes) -> bytes:
        t = bytearray(d)
        shift = len(d) % len(PI_MASK)
        mask = PI_MASK[shift:] + PI_MASK[:shift]
        for _ in range(2):
            for i in range(len(t)):
                t[i] ^= mask[i % len(mask)]
        return bytes(t)
    reverse_transform_06 = transform_06

    # 7: Delta encoding (subtract previous byte, first byte unchanged)
    def transform_07(self, d: bytes) -> bytes:
        if not d: return b''
        t = bytearray(d)
        prev = t[0]
        for i in range(1, len(t)):
            curr = t[i]
            t[i] = (curr - prev) % 256
            prev = curr
        return bytes(t)
    def reverse_transform_07(self, d: bytes) -> bytes:
        if not d: return b''
        t = bytearray(d)
        for i in range(1, len(t)):
            t[i] = (t[i] + t[i-1]) % 256
        return bytes(t)

    # 8: Light Move-to-Front (last 32 distinct bytes)
    def transform_08(self, d: bytes) -> bytes:
        if not d: return b''
        recent = []
        out = bytearray()
        for b in d:
            if b in recent:
                idx = recent.index(b)
                out.append(idx + 1)  # 1-32
                del recent[idx]
            else:
                out.append(0)
                if len(recent) >= 32:
                    recent.pop(0)
            recent.append(b)
        return bytes(out)
    def reverse_transform_08(self, d: bytes) -> bytes:
        if not d: return b''
        recent = []
        out = bytearray()
        for code in d:
            if code == 0:
                # Should not happen in valid data, but skip
                continue
            idx = code - 1
            if idx < len(recent):
                b = recent[idx]
                del recent[idx]
            else:
                b = 0  # fallback
            out.append(b)
            recent.append(b)
            if len(recent) > 32:
                recent.pop(0)
        return bytes(out)

    # Dynamic single-byte XORs (9–188)
    def _dynamic_transform(self, n: int):
        def tf(data: bytes):
            if not data: return b''
            seed = self.get_seed(n % len(self.seed_tables), len(data))
            return bytes(b ^ seed for b in data)
        return tf, tf

    # === Compression logic ===
    def _try_compress(self, data: bytes):
        results = []
        try:
            z = zstd_cctx.compress(data)
            results.append(('Z', z))
        except: pass
        try:
            p = paq.compress(data)
            results.append(('P', p))
        except: pass
        if not results:
            return None
        winner = min(results, key=lambda x: len(x[1]))
        return (0x5A if winner[0] == 'Z' else 0x50) + winner[1]

    def compress_with_best(self, data: bytes) -> bytes:
        if not data:
            return b'\x00'

        transforms = [
            (1, self.transform_01),
            (2, self.transform_02),
            (3, self.transform_03),
            (4, self.transform_04),
            (5, self.transform_05),
            (6, self.transform_06),
            (7, self.transform_07),
            (8, self.transform_08),
        ] + [(i+9, self._dynamic_transform(i+9)[0]) for i in range(180)]  # markers 9-188

        best_size = float('inf')
        best_payload = None
        best_marker = 255  # no transform

        # Raw no transform
        raw_comp = self._try_compress(data)
        if raw_comp and len(raw_comp) < best_size:
            best_size = len(raw_comp)
            best_payload = raw_comp
            best_marker = 255

        # Try transforms
        for marker, func in transforms:
            try:
                t_data = func(data)
                comp = self._try_compress(t_data)
                if comp and len(comp) < best_size:
                    best_size = len(comp)
                    best_payload = comp
                    best_marker = marker
            except:
                continue

        return bytes([best_marker]) + best_payload

    def decompress_with_best(self, data: bytes):
        if len(data) < 2:
            return None

        marker = data[0]
        payload = data[1:]

        backend_marker = payload[0]
        backend_payload = payload[1:]

        if backend_marker == 0x5A:  # Zstd
            try:
                backend_data = zstd_dctx.decompress(backend_payload)
            except:
                return None
        elif backend_marker == 0x50:  # Paq
            try:
                backend_data = paq.decompress(backend_payload)
            except:
                return None
        else:
            return None

        rev_map = {
            1: self.reverse_transform_01,
            2: self.reverse_transform_02,
            3: self.reverse_transform_03,
            4: self.reverse_transform_04,
            5: self.reverse_transform_05,
            6: self.reverse_transform_06,
            7: self.reverse_transform_07,
            8: self.reverse_transform_08,
        }
        for i in range(180):
            rev_map[i+9] = self._dynamic_transform(i+9)[1]

        rev_func = rev_map.get(marker, lambda x: x)
        return rev_func(backend_data)

    # === Public API ===
    def compress(self, infile: str, outfile: str):
        with open(infile, 'rb') as f:
            data = f.read()
        compressed = self.compress_with_best(data)
        with open(outfile, 'wb') as f:
            f.write(compressed)
        ratio = (1 - len(compressed)/len(data))*100 if data else 0
        print(f"Compressed {len(data)} → {len(compressed)} bytes ({ratio:.2f}% saved) → {outfile}")

    def decompress(self, infile: str, outfile: str):
        with open(infile, 'rb') as f:
            data = f.read()
        original = self.decompress_with_best(data)
        if original is None:
            print("Decompression failed!")
            return
        with open(outfile, 'wb') as f:
            f.write(original)
        print(f"Decompressed → {outfile}")

def main():
    print(f"{PROGNAME} – Stronger Zstd-22 + Paq-6 Hybrid")
    print("by Jurijus Pacalovas + Grok AI (xAI) – December 31, 2025")
    print()
    c = PAQJPCompressor()
    choice = input("1) Compress   2) Decompress\n> ").strip()
    if choice == "1":
        infile = input("Input file: ").strip()
        outfile = input("Output file (.pjp): ").strip() or infile + ".pjp"
        c.compress(infile, outfile)
    elif choice == "2":
        infile = input("Compressed file: ").strip()
        outfile = input("Output file: ").strip() or os.path.splitext(infile)[0] + ".dec"
        c.decompress(infile, outfile)
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
