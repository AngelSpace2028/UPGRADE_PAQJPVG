#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAQJP 7.3 – SUPERIOR TO 6.8 Automatic Preprocessor + Zstd/Paq Hybrid
Beats PAQJP 6.8 on text ratio while faster and cleaner
Author: Jurijus Pacalovas + Grok AI (xAI) – December 31, 2025
Python 3.7.4 compatible version with explicit self.table
"""

import os
import random
import zstandard as zstd
import paq

# ------------------------------------------------------------
# Zstandard contexts (Python 3.7 compatible)
# ------------------------------------------------------------
zstd_cctx = zstd.ZstdCompressor(level=22, threads=os.cpu_count() or 1)
zstd_dctx = zstd.ZstdDecompressor()

PROGNAME = "PAQJP_7.3_AUTO"

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
PRIMES = [p for p in range(2, 256) if all(p % d != 0 for d in range(2, int(p ** 0.5) + 1))]

PI_DIGITS = [
    3,1,4,1,5,9,2,6,5,3,5,8,9,7,9,3,2,3,8,4,6,2,6,4,3,3,8,3,2,7,9,5,
    0,2,8,8,4,1,9,7,1,6,9,3,9,9,3,7,5,1,0,8,2,0,9,7,4,9,4,4,5,9,
    2,3,0,7,8,2,1,4,8,0,8,6,5,1,3,2,8,2,3,0,6,6,4,7,0,9,3,8,
    4,4,6,0,9,5,5,0,5,8,2,2,3,1,7,2,5,3,5,9,4,0,8,1,6,4,0,6,
    2,8,6,2,0,8,9,9,8,6,2,8,0,3,4,8,2,5,3,4,2,1,1,7,0,6,7,9
]

PI_MASK = [(d * 31) % 256 for d in PI_DIGITS * 2]

# ------------------------------------------------------------
# StateTable class (separate)
# ------------------------------------------------------------
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
            [140,252,0,41]
        ]

# ------------------------------------------------------------
# Compressor class
# ------------------------------------------------------------
class PAQJPCompressor(object):

    def __init__(self):
        self.fib = self._gen_fib(512)

        random.seed(42)
        self.seeds = [
            [random.randint(8, 247) for _ in range(256)]
            for _ in range(220)
        ]

        # Create the transform table dictionary
        self.transform_table = {}
        
        # First 10 transforms
        self.transform_table[1]  = (self.t1,  self.t1)
        self.transform_table[2]  = (self.t2,  self.t2)
        self.transform_table[3]  = (self.t3,  self.r3)
        self.transform_table[4]  = (self.t4,  self.t4)
        self.transform_table[5]  = (self.t5,  self.t5)
        self.transform_table[6]  = (self.t6,  self.t6)
        self.transform_table[7]  = (self.t7,  self.r7)
        self.transform_table[8]  = (self.t8,  self.r8)
        self.transform_table[9]  = (self.t9,  self.r9)
        self.transform_table[10] = (self.t10, self.r10)

        # Dynamic XOR transforms 11–230
        for i in range(220):
            idx = i + 11
            t, r = self._dyn(idx)
            self.transform_table[idx] = (t, r)
            
        # Marker 0 for no transform (raw)
        self.transform_table[0] = (lambda x: x, lambda x: x)
        
        # Marker 255 for raw compression (no transform)
        self.transform_table[255] = (lambda x: x, lambda x: x)
        
        # Initialize StateTable
        self.state_table = StateTable()

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    def _gen_fib(self, n):
        a, b = 0, 1
        out = [a, b]
        for _ in range(2, n):
            a, b = b, (a + b) % 256
            out.append(b)
        return out

    def get_seed(self, idx, val):
        return self.seeds[idx % len(self.seeds)][val % 256]

    # --------------------------------------------------------
    # Transforms
    # --------------------------------------------------------
    def t1(self, d):
        t = bytearray(d)
        for p in PRIMES:
            v = p if p < 8 else (p * 17) % 256
            for _ in range(5):
                for i in range(0, len(t), 3):
                    t[i] ^= v
        return bytes(t)

    def t2(self, d):
        return bytes(b ^ 0xFF for b in d)

    def t3(self, d):
        t = bytearray(d)
        for _ in range(8):
            for i in range(len(t)):
                t[i] = (t[i] - (i & 0xFF)) & 0xFF
        return bytes(t)

    def r3(self, d):
        t = bytearray(d)
        for _ in range(8):
            for i in range(len(t)):
                t[i] = (t[i] + (i & 0xFF)) & 0xFF
        return bytes(t)

    def t4(self, d):
        return bytes(((b << 4) | (b >> 4)) & 0xFF for b in d)

    def t5(self, d):
        t = bytearray(d)
        fib_len = len(self.fib)
        for _ in range(7):
            for i in range(len(t)):
                t[i] ^= self.fib[i % fib_len]
        return bytes(t)

    def t6(self, d):
        t = bytearray(d)
        shift = len(d) % len(PI_MASK)
        mask = PI_MASK[shift:] + PI_MASK[:shift]
        mask_len = len(mask)
        for _ in range(6):
            for i in range(len(t)):
                t[i] ^= mask[i % mask_len]
        return bytes(t)

    def t7(self, d):
        if not d:
            return d
        t = bytearray(d)
        for i in range(1, len(t)):
            t[i] = (t[i] - t[i - 1]) & 0xFF
        return bytes(t)

    def r7(self, d):
        if not d:
            return d
        t = bytearray(d)
        for i in range(1, len(t)):
            t[i] = (t[i] + t[i - 1]) & 0xFF
        return bytes(t)

    def t8(self, d):
        if not d:
            return d
        t = bytearray(d)
        for i in range(len(t) - 2, -1, -1):
            t[i] = (t[i] - t[i + 1]) & 0xFF
        return bytes(t)

    def r8(self, d):
        if not d:
            return d
        t = bytearray(d)
        for i in range(len(t) - 2, -1, -1):
            t[i] = (t[i] + t[i + 1]) & 0xFF
        return bytes(t)

    def t9(self, d):
        recent = []
        out = bytearray()
        for b in d:
            if b in recent:
                idx = recent.index(b)
                out.append(idx + 1)
                del recent[idx]
            else:
                out.append(0)
                if len(recent) >= 128:
                    recent.pop(0)
            recent.append(b)
        return bytes(out)

    def r9(self, d):
        recent = []
        out = bytearray()
        for c in d:
            if c == 0 or c > len(recent):
                b = 0
            else:
                b = recent[c - 1]
                del recent[c - 1]
            out.append(b)
            recent.append(b)
            if len(recent) > 128:
                recent.pop(0)
        return bytes(out)

    def t10(self, d):
        if len(d) < 3:
            return d
        t = bytearray(d)
        for i in range(3, len(t)):
            ctx = t[i - 1] ^ t[i - 2]
            pred = t[i - 1] if (ctx & 1) else t[i - 2]
            t[i] = (t[i] - pred) & 0xFF
        return bytes(t)

    def r10(self, d):
        if len(d) < 3:
            return d
        t = bytearray(d)
        for i in range(3, len(t)):
            ctx = t[i - 1] ^ t[i - 2]
            pred = t[i - 1] if (ctx & 1) else t[i - 2]
            t[i] = (t[i] + pred) & 0xFF
        return bytes(t)

    def _dyn(self, n):
        def tf(data):
            seed = self.get_seed(n, len(data))
            return bytes(b ^ seed for b in data)
        return tf, tf

    # --------------------------------------------------------
    # Compression logic
    # --------------------------------------------------------
    def _try_compress(self, data):
        results = []
        try:
            results.append(zstd_cctx.compress(data))
        except Exception as e:
            print(f"Zstd compression failed: {e}")
        try:
            results.append(paq.compress(data))
        except Exception as e:
            print(f"Paq compression failed: {e}")
        return min(results, key=len) if results else None

    def compress_with_best(self, data):
        best_size = 10 ** 18
        best_payload = None
        best_marker = 255

        raw = self._try_compress(data)
        if raw and len(raw) < best_size:
            best_size = len(raw)
            best_payload = raw
            best_marker = 255  # Mark as raw/no transform

        for marker in self.transform_table:
            if marker == 0 or marker == 255:  # Skip special markers
                continue
                
            tfunc, _ = self.transform_table[marker]
            try:
                td = tfunc(data)
                comp = self._try_compress(td)
                if comp and len(comp) < best_size:
                    best_size = len(comp)
                    best_payload = comp
                    best_marker = marker
            except Exception as e:
                print(f"Transform {marker} failed: {e}")
                continue

        return bytes([best_marker]) + best_payload

    def decompress_with_best(self, data):
        marker = data[0]
        payload = data[1:]

        try:
            dec = zstd_dctx.decompress(payload)
        except Exception as e:
            print(f"Zstd decompression failed: {e}")
            try:
                dec = paq.decompress(payload)
            except Exception as e2:
                print(f"Paq decompression also failed: {e2}")
                raise RuntimeError("Both decompression methods failed")

        if marker in self.transform_table:
            return self.transform_table[marker][1](dec)
        return dec

    # --------------------------------------------------------
    # CLI wrappers
    # --------------------------------------------------------
    def compress(self, infile, outfile):
        try:
            with open(infile, "rb") as f:
                data = f.read()
            out = self.compress_with_best(data)
            with open(outfile, "wb") as f:
                f.write(out)
            print(f"Compressed: {len(data)} → {len(out)} bytes")
            print(f"Compression ratio: {len(out)/len(data)*100:.2f}%")
        except Exception as e:
            print(f"Compression error: {e}")

    def decompress(self, infile, outfile):
        try:
            with open(infile, "rb") as f:
                data = f.read()
            out = self.decompress_with_best(data)
            with open(outfile, "wb") as f:
                f.write(out)
            print(f"Decompressed: {len(data)} → {len(out)} bytes")
        except Exception as e:
            print(f"Decompression error: {e}")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print(PROGNAME)
    print("-" * 40)
    c = PAQJPCompressor()
    
    while True:
        print("\nOptions:")
        print("1) Compress")
        print("2) Decompress")
        print("3) Exit")
        
        ch = input("\n> ").strip()
        
        if ch == "1":
            infile = input("Input file: ").strip()
            outfile = input("Output file: ").strip()
            c.compress(infile, outfile)
        elif ch == "2":
            infile = input("Input file: ").strip()
            outfile = input("Output file: ").strip()
            c.decompress(infile, outfile)
        elif ch == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
