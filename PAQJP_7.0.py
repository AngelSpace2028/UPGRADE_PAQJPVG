#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAQJP 7.0 – FULLY AUTOMATIC Preprocessor + Zstandard/Paq Hybrid (Ultimate Practical Edition)
The best achievable with pure Python: Smart multi-transform trial + Zstandard level 22 or Paq level 9
True PAQ-level compression requires native C++ binaries (paq8px latest ~v209), which are extremely slow but unbeatable in ratio.
This version gives excellent practical results, fast enough for daily use, while keeping the full StateTable and all transforms.
Author: Jurijus Pacalovas + Grok AI (xAI)
"""

import os
import math
import random
import zstandard as zstd
import paq
from typing import Optional

zstd_cctx = zstd.ZstdCompressor(level=22, threads=os.cpu_count() or 1)
zstd_dctx = zstd.ZstdDecompressor()

PROGNAME = "PAQJP_7.0_AUTO"

PRIMES = [p for p in range(2, 256) if all(p % d != 0 for d in range(2, int(p**0.5)+1))]

# ========================= Full StateTable (preserved as requested) =========================
class StateTable:
    def __init__(self):
        self.table = [[1,2,0,0],[3,5,1,0],[4,6,0,1],[7,10,2,0],[8,12,1,1],[9,13,1,1],[11,14,0,2],[15,19,3,0],
                      [16,23,2,1],[17,24,2,1],[18,25,2,1],[20,27,1,2],[21,28,1,2],[22,29,1,2],[26,30,0,3],[31,33,4,0],
                      [32,35,3,1],[32,35,3,1],[32,35,3,1],[32,35,3,1],[34,37,2,2],[34,37,2,2],[34,37,2,2],[34,37,2,2],
                      [34,37,2,2],[34,37,2,2],[36,39,1,3],[36,39,1,3],[36,39,1,3],[36,39,1,3],[38,40,0,4],[41,43,5,0],
                      [42,45,4,1],[42,45,4,1],[44,47,3,2],[44,47,3,2],[46,49,2,3],[46,49,2,3],[48,51,1,4],[48,51,1,4],
                      [50,52,0,5],[53,43,6,0],[54,57,5,1],[54,57,5,1],[56,59,4,2],[56,59,4,2],[58,61,3,3],[58,61,3,3],
                      [60,63,2,4],[60,63,2,4],[62,65,1,5],[62,65,1,5],[50,66,0,6],[67,55,7,0],[68,57,6,1],[68,57,6,1],
                      [70,73,5,2],[70,73,5,2],[72,75,4,3],[72,75,4,3],[74,77,3,4],[74,77,3,4],[76,79,2,5],[76,79,2,5],
                      [62,81,1,6],[62,81,1,6],[64,82,0,7],[83,69,8,0],[84,76,7,1],[84,76,7,1],[86,73,6,2],[86,73,6,2],
                      [44,59,5,3],[44,59,5,3],[58,61,4,4],[58,61,4,4],[60,49,3,5],[60,49,3,5],[76,89,2,6],[76,89,2,6],
                      [78,91,1,7],[78,91,1,7],[80,92,0,8],[93,69,9,0],[94,87,8,1],[94,87,8,1],[96,45,7,2],[96,45,7,2],
                      [48,99,2,7],[48,99,2,7],[88,101,1,8],[88,101,1,8],[80,102,0,9],[103,69,10,0],[104,87,9,1],[104,87,9,1],
                      [106,57,8,2],[106,57,8,2],[62,109,2,8],[62,109,2,8],[88,111,1,9],[88,111,1,9],[80,112,0,10],[113,85,11,0],
                      [114,87,10,1],[114,87,10,1],[116,57,9,2],[116,57,9,2],[62,119,2,9],[62,119,2,9],[88,121,1,10],[88,121,1,10],
                      [90,122,0,11],[123,85,12,0],[124,97,11,1],[124,97,11,1],[126,57,10,2],[126,57,10,2],[62,129,2,10],[62,129,2,10],
                      [98,131,1,11],[98,131,1,11],[90,132,0,12],[133,85,13,0],[134,97,12,1],[134,97,12,1],[136,57,11,2],[136,57,11,2],
                      [62,139,2,11],[62,139,2,11],[98,141,1,12],[98,141,1,12],[90,142,0,13],[143,95,14,0],[144,97,13,1],[144,97,13,1],
                      [68,57,12,2],[68,57,12,2],[62,81,2,12],[62,81,2,12],[98,147,1,13],[98,147,1,13],[100,148,0,14],[149,95,15,0],
                      [150,107,14,1],[150,107,14,1],[108,151,1,14],[108,151,1,14],[100,152,0,15],[153,95,16,0],[154,107,15,1],[108,155,1,15],
                      [100,156,0,16],[157,95,17,0],[158,107,16,1],[108,159,1,16],[100,160,0,17],[161,105,18,0],[162,107,17,1],[108,163,1,17],
                      [110,164,0,18],[165,105,19,0],[166,117,18,1],[118,167,1,18],[110,168,0,19],[169,105,20,0],[170,117,19,1],[118,171,1,19],
                      [110,172,0,20],[173,105,21,0],[174,117,20,1],[118,175,1,20],[110,176,0,21],[177,105,22,0],[178,117,21,1],[118,179,1,21],
                      [120,184,0,23],[185,115,24,0],[186,127,23,1],[128,187,1,23],[120,188,0,24],[189,115,25,0],[190,127,24,1],[128,191,1,24],
                      [120,192,0,25],[193,115,26,0],[194,127,25,1],[128,195,1,25],[120,196,0,26],[197,115,27,0],[198,127,26,1],[128,199,1,26],
                      [120,200,0,27],[201,115,28,0],[202,127,27,1],[128,203,1,27],[120,204,0,28],[205,115,29,0],[206,127,28,1],[128,207,1,28],
                      [120,208,0,29],[209,125,30,0],[210,127,29,1],[128,211,1,29],[130,212,0,30],[213,125,31,0],[214,137,30,1],[138,215,1,30],
                      [130,216,0,31],[217,125,32,0],[218,137,31,1],[138,219,1,31],[130,220,0,32],[221,125,33,0],[222,137,32,1],[138,223,1,32],
                      [130,224,0,33],[225,125,34,0],[226,137,33,1],[138,227,1,33],[130,228,0,34],[229,125,35,0],[230,137,34,1],[138,231,1,34],
                      [130,232,0,35],[233,125,36,0],[234,137,35,1],[138,235,1,35],[130,236,0,36],[237,125,37,0],[238,137,36,1],[138,239,1,36],
                      [130,240,0,37],[241,125,38,0],[242,137,37,1],[138,243,1,37],[130,244,0,38],[245,135,39,0],[246,137,38,1],[138,247,1,38],
                      [140,248,0,39],[249,135,40,0],[250,69,39,1],[80,251,1,39],[140,252,0,40],[249,135,41,0],[250,69,40,1],[80,251,1,40],
                      [140,252,0,41]]

# ========================= Helper functions =========================
def transform_with_prime_xor_every_3_bytes(data: bytes, repeat: int = 100) -> bytes:
    t = bytearray(data)
    for prime in PRIMES:
        xor_val = prime if prime == 2 else max(1, math.ceil(prime * 4096 / 28672))
        for _ in range(repeat):
            for i in range(0, len(t), 3):
                if i < len(t):
                    t[i] ^= xor_val
    return bytes(t)

def transform_with_pattern_chunk(data: bytes, chunk_size: int = 4) -> bytes:
    t = bytearray()
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        t.extend(b ^ 0xFF for b in chunk)
    return bytes(t)

# ========================= Main Compressor Class =========================
class PAQJPCompressor:
    def __init__(self):
        self.state_table = StateTable()  # Preserved
        self.fibonacci = self._gen_fib(100)
        random.seed(42)
        self.seed_tables = [[random.randint(5, 255) for _ in range(256)] for _ in range(126)]

    def _gen_fib(self, n):
        a, b = 0, 1
        res = [a, b]
        for _ in range(2, n):
            a, b = b, a + b
            res.append(b)
        return res

    def get_seed(self, idx: int, val: int) -> int:
        if 0 <= idx < len(self.seed_tables):
            return self.seed_tables[idx][val % 256]
        return 0

    # Reversible transforms
    def transform_01(self, d): return transform_with_prime_xor_every_3_bytes(d)
    def reverse_transform_01(self, d): return self.transform_01(d)

    def transform_03(self, d): return transform_with_pattern_chunk(d)
    def reverse_transform_03(self, d): return self.transform_03(d)

    def transform_04(self, d):
        t = bytearray(d)
        for i in range(len(t)): t[i] = (t[i] - (i%256)) % 256
        return bytes(t)
    def reverse_transform_04(self, d):
        t = bytearray(d)
        for i in range(len(t)): t[i] = (t[i] + (i%256)) % 256
        return bytes(t)

    def transform_05(self, d, s=3):
        t = bytearray(d)
        for i in range(len(t)): t[i] = ((t[i]<<s)|(t[i]>>(8-s)))&0xFF
        return bytes(t)
    def reverse_transform_05(self, d, s=3): return self.transform_05(d, (8-s)%8)

    def transform_12(self, d):
        t = bytearray(d)
        for i in range(len(t)): t[i] ^= self.fibonacci[i % len(self.fibonacci)] % 256
        return bytes(t)
    def reverse_transform_12(self, d): return self.transform_12(d)

    def _dynamic_transform(self, n: int):
        def tf(data: bytes):
            if not data: return b''
            seed = self.get_seed(n % len(self.seed_tables), len(data))
            t = bytearray(data)
            for i in range(len(t)): t[i] ^= seed
            return bytes(t)
        return tf, tf

    def compress_with_best(self, data: bytes) -> bytes:
        if not data: return bytes([0])

        transforms = [
            (1, self.transform_04),
            (2, self.transform_01),
            (3, self.transform_03),
            (5, self.transform_05),
            (12, self.transform_12),
        ] + [(i, self._dynamic_transform(i)[0]) for i in range(16, 64)]  # Balanced for speed

        best_size = float('inf')
        best_payload = b''
        best_marker = 255  # No transform

        # Also try no transform
        no_transform_func = lambda x: x
        t_data = no_transform_func(data)
        c_data_zstd = zstd_cctx.compress(t_data)
        size_zstd = len(c_data_zstd)
        c_data_paq = paq.compress(t_data)
        size_paq = len(c_data_paq)

        if size_zstd < best_size:
            best_size = size_zstd
            best_payload = c_data_zstd
            best_marker = 255

        if size_paq < best_size:
            best_size = size_paq
            best_payload = c_data_paq
            best_marker = 255

        for marker, func in transforms:
            try:
                t_data = func(data)
                c_data_zstd = zstd_cctx.compress(t_data)
                size_zstd = len(c_data_zstd)
                c_data_paq = paq.compress(t_data, level=9)
                size_paq = len(c_data_paq)

                if size_zstd < best_size:
                    best_size = size_zstd
                    best_payload = c_data_zstd
                    best_marker = marker

                if size_paq < best_size:
                    best_size = size_paq
                    best_payload = c_data_paq
                    best_marker = marker
            except:
                continue

        return bytes([best_marker]) + best_payload

    def decompress_with_best(self, data: bytes):
        if len(data) < 1: return b'', None
        marker = data[0]
        payload = data[1:]

        backend = None
        try:
            backend = zstd_dctx.decompress(payload)
        except zstd.ZstdError:
            pass

        if backend is None:
            try:
                backend = paq.decompress(payload)
            except paq.error:
                return None, None

        rev = {
            1: self.reverse_transform_04,
            2: self.reverse_transform_01,
            3: self.reverse_transform_03,
            5: self.reverse_transform_05,
            12: self.reverse_transform_12,
        }
        for i in range(16, 64):
            rev[i] = self._dynamic_transform(i)[1]

        rev_func = rev.get(marker, lambda x: x)
        return rev_func(backend), marker

    def compress(self, infile: str, outfile: str):
        with open(infile, 'rb') as f: data = f.read()
        compressed = self.compress_with_best(data)
        with open(outfile, 'wb') as f: f.write(compressed)
        ratio = (1 - len(compressed)/len(data))*100 if data else 0
        print(f"Compressed {len(data)} → {len(compressed)} bytes ({ratio:.2f}% saved) → {outfile}")

    def decompress(self, infile: str, outfile: str):
        with open(infile, 'rb') as f: data = f.read()
        original, _ = self.decompress_with_best(data)
        if original is None:
            print("Decompression failed!")
            return
        with open(outfile, 'wb') as f: f.write(original)
        print(f"Decompressed → {outfile}")

# ========================= Main =========================
def main():
    print(f"{PROGNAME} – Smart Preprocessor + Zstandard/Paq (Best Practical Python Compressor)")
    print("by Jurijus Pacalovas + Grok AI")
    print("Note: No native PAQ module exists in Python. For absolute maximum ratio (very slow), use external paq8px ~v209.")
    c = PAQJPCompressor()
    ch = input("1) Compress   2) Decompress\n> ").strip()
    if ch == "1":
        i = input("Input file: ").strip()
        o = input("Output file (.pjp): ").strip() or i + ".pjp"
        c.compress(i, o)
    elif ch == "2":
        i = input("Compressed file: ").strip()
        o = input("Output file: ").strip()
        c.decompress(i, o)

if __name__ == "__main__":
    main()
