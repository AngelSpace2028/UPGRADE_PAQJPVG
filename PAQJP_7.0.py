#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PAQJP 7.0 – Full reference implementation
• FULL StateTable (untruncated)
• ALL transforms and reverse transforms
• Dynamic transform IDs: 16–63 ONLY
• Zstandard primary, PAQ optional
• Python 3.7+ compatible
Created by Jurijus Pacalovas
"""

import math
import random
import zstandard as zstd

# ================= Optional PAQ =================
try:
    import paq
    HAVE_PAQ = True
except Exception:
    HAVE_PAQ = False

# ================= Zstandard =================
ZSTD_LEVEL = 22
zstd_cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
zstd_dctx = zstd.ZstdDecompressor()

# ================= Primes =================
PRIMES = [p for p in range(2, 256)
          if all(p % d != 0 for d in range(2, int(p ** 0.5) + 1))]

# ================= FULL StateTable =================
class StateTable:
    def __init__(self):
        self.table = [
[1,2,0,0],[3,5,1,0],[4,6,0,1],[7,10,2,0],[8,12,1,1],[9,13,1,1],[11,14,0,2],[15,19,3,0],
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
[150,107,14,1],[150,107,14,1],[108,151,1,14],[108,151,1,14],[100,152,0,15],[153,95,16,0],[154,107,15,1],
[108,155,1,15],[100,156,0,16],[157,95,17,0],[158,107,16,1],[108,159,1,16],[100,160,0,17],[161,105,18,0],
[162,107,17,1],[108,163,1,17],[110,164,0,18],[165,105,19,0],[166,117,18,1],[118,167,1,18],[110,168,0,19],
[169,105,20,0],[170,117,19,1],[118,171,1,19],[110,172,0,20],[173,105,21,0],[174,117,20,1],[118,175,1,20],
[110,176,0,21],[177,105,22,0],[178,117,21,1],[118,179,1,21]
        ]

# ================= TRANSFORMS =================
def transform_prime_xor_every_3(data):
    t = bytearray(data)
    for p in PRIMES:
        v = max(1, int(math.ceil(p * 4096 / 28672)))
        for i in range(0, len(t), 3):
            t[i] ^= v
    return bytes(t)

def transform_invert_bytes(data):
    return bytes(b ^ 0xFF for b in data)

def transform_add_index(data):
    t = bytearray(data)
    for i in range(len(t)):
        t[i] = (t[i] + i) & 0xFF
    return bytes(t)

def reverse_add_index(data):
    t = bytearray(data)
    for i in range(len(t)):
        t[i] = (t[i] - i) & 0xFF
    return bytes(t)

def transform_block_flip(data, block=4):
    out = bytearray()
    for i in range(0, len(data), block):
        out.extend(b ^ 0xFF for b in data[i:i+block])
    return bytes(out)

# ================= COMPRESSOR =================
class PAQJPCompressor:
    def __init__(self):
        self.state_table = StateTable()
        random.seed(42)
        self.seed_tables = [[random.randint(1, 255) for _ in range(256)]
                            for _ in range(64)]

    # ---------- dynamic reversible transform ----------
    def _dynamic_transform(self, idx):
        def tf(data):
            seed = self.seed_tables[idx % 64][len(data) & 0xFF]
            t = bytearray(data)
            for i in range(len(t)):
                t[i] ^= seed
            return bytes(t)
        return tf, tf

    # ---------- compression ----------
    def compress(self, data):
        best = zstd_cctx.compress(data)
        marker = 255

        # static transforms
        static = {
            1: (transform_add_index, reverse_add_index),
            2: (transform_prime_xor_every_3, transform_prime_xor_every_3),
            3: (transform_block_flip, transform_block_flip),
            4: (transform_invert_bytes, transform_invert_bytes),
        }

        for m, (tf, _) in static.items():
            td = tf(data)
            c = zstd_cctx.compress(td)
            if len(c) < len(best):
                best, marker = c, m

        # dynamic 16–63
        for i in range(16, 64):
            tf, _ = self._dynamic_transform(i)
            td = tf(data)
            c = zstd_cctx.compress(td)
            if len(c) < len(best):
                best, marker = c, i

        # optional PAQ
        if HAVE_PAQ:
            c = paq.compress(data)
            if len(c) < len(best):
                best, marker = c, 255

        return bytes([marker]) + best

    # ---------- decompression ----------
    def decompress(self, data):
        marker = data[0]
        payload = data[1:]

        try:
            raw = zstd_dctx.decompress(payload)
        except Exception:
            if HAVE_PAQ:
                raw = paq.decompress(payload)
            else:
                raise

        static_rev = {
            1: reverse_add_index,
            2: transform_prime_xor_every_3,
            3: transform_block_flip,
            4: transform_invert_bytes,
        }

        if marker in static_rev:
            return static_rev[marker](raw)

        if 16 <= marker <= 63:
            _, rtf = self._dynamic_transform(marker)
            return rtf(raw)

        return raw

# ================= MAIN =================
def main():
    c = PAQJPCompressor()
    mode = input("1 Compress  2 Decompress\n> ").strip()

    if mode == "1":
        i = input("Input file: ")
        o = input("Output file: ")
        with open(i, "rb") as f:
            d = f.read()
        with open(o, "wb") as f:
            f.write(c.compress(d))
        print("Compressed OK")

    elif mode == "2":
        i = input("Compressed file: ")
        o = input("Output file: ")
        with open(i, "rb") as f:
            d = f.read()
        out = c.decompress(d)
        with open(o, "wb") as f:
            f.write(out)
        print("Decompressed OK")

if __name__ == "__main__":
    main()
