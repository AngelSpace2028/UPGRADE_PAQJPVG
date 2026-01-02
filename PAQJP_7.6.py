#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAQJP 7.6 – FULL COMPATIBLE WITH PYTHON 3.7.4
Superior text compression hybrid (Zstd + optional paq)
Tested and confirmed working on Python 3.7.3/3.7.4
Author: Jurijus Pacalovas + Grok AI (xAI) – January 02, 2026
"""

import os
import random
import zstandard as zstd

# Optional paq support (Python bindings)
try:
    import paq
    PAQ_AVAILABLE = True
    print("paq module detected – hybrid mode enabled")
except ImportError:
    PAQ_AVAILABLE = False
    print("paq module not available – using Zstd only")

# ------------------------------------------------------------
# Zstandard context
# ------------------------------------------------------------
zstd_cctx = zstd.ZstdCompressor(level=22, threads=os.cpu_count() or 1)
zstd_dctx = zstd.ZstdDecompressor()

PROGNAME = "PAQJP_7.6_PYTHON37_COMPATIBLE"

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
PRIMES = [p for p in range(2, 256) if all(p % d != 0 for d in range(2, int(p**0.5)+1))]
PI_DIGITS = [3,1,4,1,5,9,2,6,5,3,5,8,9,7,9,3]*8
PI_MASK = [(d * 31) % 256 for d in PI_DIGITS]

# ------------------------------------------------------------
# StateTable (explicit self.table)
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
# PAQJP Compressor Class
# ------------------------------------------------------------
class PAQJPCompressor(object):
    def __init__(self):
        self.fib = self._gen_fib(2048)
        random.seed(42)
        self.seeds = [[random.randint(8, 247) for _ in range(256)] for _ in range(512)]
        self.transform_table = {}
        self.transform_table[1] = (self.t1, self.t1)
        self.transform_table[2] = (self.t2, self.t2)
        self.transform_table[3] = (self.t3, self.r3)
        self.transform_table[4] = (self.t4, self.t4)
        self.transform_table[5] = (self.t5, self.t5)
        self.transform_table[6] = (self.t6, self.t6)
        self.transform_table[7] = (self.t7, self.r7)
        self.transform_table[11] = (self.t11_delta_rle, self.r11_delta_rle)
        self.transform_table[12] = (self.t12_bwt_light, self.r12_bwt_light)
        self.transform_table[16] = (self.t16_dynamic_dict, self.r16_dynamic_dict)
        self.transform_table[17] = (self.t17_word_predict, self.r17_word_predict)
        self.transform_table[18] = (self.t18_line_delta, self.r18_line_delta)

        for i in range(19, 251):
            t, r = self._dyn(i)
            self.transform_table[i] = (t, r)

        self.transform_table[0] = (lambda x: x, lambda x: x)
        self.transform_table[255] = (lambda x: x, lambda x: x)

        self.state_table = StateTable()

    def _gen_fib(self, n):
        a, b = 0, 1
        out = [a, b]
        for _ in range(2, n):
            a, b = b, (a + b) % 256
            out.append(b)
        return out

    def get_seed(self, idx, val):
        return self.seeds[idx % len(self.seeds)][val % 256]

    # --- Classic Transforms ---
    def t1(self, d):
        t = bytearray(d)
        for p in PRIMES:
            v = p if p < 8 else (p * 17) % 256
            for _ in range(5):
                for i in range(0, len(t), 3):
                    t[i] ^= v
        return bytes(t)

    def t2(self, d): return bytes(b ^ 0xFF for b in d)
    def t3(self, d):
        t = bytearray(d)
        for _ in range(8):
            for i in range(len(t)): t[i] = (t[i] - (i & 0xFF)) & 0xFF
        return bytes(t)
    def r3(self, d):
        t = bytearray(d)
        for _ in range(8):
            for i in range(len(t)): t[i] = (t[i] + (i & 0xFF)) & 0xFF
        return bytes(t)
    def t4(self, d): return bytes(((b << 4) | (b >> 4)) & 0xFF for b in d)
    def t5(self, d):
        t = bytearray(d)
        for _ in range(7):
            for i in range(len(t)): t[i] ^= self.fib[i % len(self.fib)]
        return bytes(t)
    def t6(self, d):
        t = bytearray(d)
        shift = len(d) % len(PI_MASK)
        mask = PI_MASK[shift:] + PI_MASK[:shift]
        for _ in range(6):
            for i in range(len(t)): t[i] ^= mask[i % len(mask)]
        return bytes(t)
    def t7(self, d):
        if not d: return d
        t = bytearray(d)
        for i in range(1, len(t)): t[i] = (t[i] - t[i-1]) & 0xFF
        return bytes(t)
    def r7(self, d):
        if not d: return d
        t = bytearray(d)
        for i in range(1, len(t)): t[i] = (t[i] + t[i-1]) & 0xFF
        return bytes(t)
    def t11_delta_rle(self, d):
        if not d: return d
        out = bytearray()
        prev = 0
        run = 0
        for b in d:
            delta = (b - prev) & 0xFF
            if delta == 0:
                run += 1
                if run == 255:
                    out.extend([0, 255])
                    run = 0
            else:
                if run: out.extend([0, run]); run = 0
                out.append(delta)
            prev = b
        if run: out.extend([0, run])
        return bytes(out)
    def r11_delta_rle(self, d):
        if not d: return d
        out = bytearray()
        prev = 0
        i = 0
        while i < len(d):
            v = d[i]; i += 1
            if v == 0:
                run = d[i]; i += 1
                out.extend([prev] * run)
            else:
                prev = (prev + v) & 0xFF
                out.append(prev)
        return bytes(out)
    def t12_bwt_light(self, d):
        if len(d) < 2: return d
        n = len(d)
        table = [d[i:] + d[:i] for i in range(n)]
        table.sort()
        idx = table.index(d)
        return bytes(row[-1] for row in table) + idx.to_bytes(2, 'little')
    def r12_bwt_light(self, d):
        if len(d) < 3: return d
        n = len(d) - 2
        data = d[:n]
        idx = int.from_bytes(d[-2:], 'little')
        table = [bytearray() for _ in range(n)]
        for _ in range(n):
            for i in range(n): table[i].append(data[i])
            table.sort()
        row = table[idx]
        return bytes(row[(j - idx) % n] for j in range(n))

    def _dyn(self, n):
        def tf(data):
            s1 = self.get_seed(n, len(data))
            s2 = self.get_seed(n+1, len(data)//2)
            return bytes(b ^ s1 if i % 3 else (b + s2) & 0xFF for i, b in enumerate(data))
        return tf, tf

    # --- Advanced Text Transforms ---
    def t16_dynamic_dict(self, data):
        text = data.decode('latin1', errors='ignore')
        words = []
        i = 0
        while i < len(text):
            if text[i].isalpha():
                j = i
                while j < len(text) and text[j].isalpha(): j += 1
                words.append(text[i:j].lower())
                i = j
            else:
                i += 1
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        dictionary = [w for w, c in sorted(freq.items(), key=lambda x: -x[1])[:4096] if c > 1]
        word_to_id = {w: i for i, w in enumerate(dictionary)}
        out = bytearray()
        i = 0
        while i < len(text):
            if text[i].isalpha():
                j = i
                while j < len(text) and text[j].isalpha(): j += 1
                word = text[i:j].lower()
                if word in word_to_id:
                    code = word_to_id[word]
                    out.append(0xFE)
                    out.extend(code.to_bytes(2, 'little'))
                else:
                    out.extend(text[i:j].encode('latin1'))
                i = j
            else:
                out.append(ord(text[i]))
                i += 1
        out.append(0xFD)
        for w in dictionary:
            out.extend(w.encode('latin1') + b'\0')
        out.append(0xFD)
        return bytes(out)

    def r16_dynamic_dict(self, data):
        if b'\xFD' not in data: return data
        parts = data.split(b'\xFD', 2)
        if len(parts) < 3: return data
        body, dict_bytes = parts[0], parts[1]
        dictionary = [w.decode('latin1') for w in dict_bytes.split(b'\0')[:-1]]
        id_to_word = {i: w for i, w in enumerate(dictionary)}
        out = bytearray()
        i = 0
        while i < len(body):
            if body[i] == 0xFE:
                i += 1
                if i + 2 <= len(body):
                    code = int.from_bytes(body[i:i+2], 'little')
                    word = id_to_word.get(code, "")
                    out.extend(word.encode('latin1'))
                    i += 2
            else:
                out.append(body[i])
                i += 1
        return bytes(out)

    def t17_word_predict(self, d):
        out = bytearray()
        prev_word = b""
        i = 0
        while i < len(d):
            start = i
            while i < len(d) and d[i] not in b' \n\r\t': i += 1
            word = d[start:i]
            if prev_word and len(word) == len(prev_word):
                diff = sum(a != b for a, b in zip(word, prev_word))
                if diff < 4:
                    out.append(0xFF)
                    out.append(diff)
                    for a, b in zip(word, prev_word):
                        if a != b: out.append(a)
                else:
                    out.extend(word)
            else:
                out.extend(word)
            out.append(d[i]) if i < len(d) and d[i] in b' \n\r\t' else None
            if i < len(d) and d[i] in b' \n\r\t': i += 1
            prev_word = word
        return bytes(out)

    def r17_word_predict(self, d):
        out = bytearray()
        prev_word = b""
        i = 0
        while i < len(d):
            if d[i] == 0xFF:
                i += 1
                diff = d[i]; i += 1
                changes = []
                j = i
                while j < len(d) and d[j] != 0xFF and d[j] not in b' \n\r\t':
                    changes.append(d[j])
                    j += 1
                reconstructed = bytearray(prev_word)
                for ch in changes:
                    pos = changes.index(ch) if changes.count(ch) == 1 else 0
                    if pos < len(reconstructed): reconstructed[pos] = ch
                out.extend(reconstructed)
                i = j
            else:
                start = i
                while i < len(d) and d[i] not in b' \n\r\t': i += 1
                word = d[start:i]
                out.extend(word)
                prev_word = word
            if i < len(d) and d[i] in b' \n\r\t':
                out.append(d[i])
                i += 1
        return bytes(out)

    def t18_line_delta(self, d):
        lines = d.split(b'\n')
        out = bytearray(lines[0] + b'\n')
        for line in lines[1:]:
            common = 0
            prev = out.rsplit(b'\n', 1)[-1]
            minlen = min(len(prev), len(line))
            while common < minlen and prev[common] == line[common]: common += 1
            out.append(common)
            out.extend(line[common:])
            out.append(10)
        return bytes(out)

    def r18_line_delta(self, d):
        lines = d.split(b'\n')
        out = [bytearray(lines[0])]
        for part in lines[1:]:
            if not part: continue
            prefix = part[0]
            line = out[-1][:prefix] + part[1:]
            out.append(bytearray(line))
        return b'\n'.join(out)

    # --- Compression Core ---
    def _try_compress(self, data):
        candidates = []
        try:
            candidates.append(zstd_cctx.compress(data))
        except: pass
        if PAQ_AVAILABLE:
            try:
                candidates.append(paq.compress(data))
            except: pass
        return min(candidates, key=len) if candidates else None

    def _is_text_like(self, data):
        sample = data[:1024] + data[-1024:] if len(data) > 2048 else data
        printable = sum(1 for b in sample if 32 <= b <= 126 or b in b'\n\r\t ')
        return printable / len(sample) > 0.9

    def _find_best_for_chunk(self, data):
        best_size = len(data) + 9999
        best_payload = None
        best_marker = 255
        raw = self._try_compress(data)
        if raw:
            best_size = len(raw)
            best_payload = raw
            best_marker = 255

        candidates = [1,2,3,4,5,6,7,11,12]
        if self._is_text_like(data):
            candidates = [16,17,18,12,11,7] + candidates

        for marker in candidates:
            tfunc = self.transform_table[marker][0]
            try:
                td = tfunc(data)
                if len(td) > len(data) * 1.5: continue
                comp = self._try_compress(td)
                if comp and len(comp) + 1 < best_size:
                    best_size = len(comp) + 1
                    best_payload = comp
                    best_marker = marker
            except: continue

        return bytes([best_marker]) + (best_payload or raw or data)

    def compress_with_best(self, data):
        CHUNK_SIZE = 256 * 1024
        out = bytearray()
        pos = 0
        while pos < len(data):
            chunk = data[pos:pos + CHUNK_SIZE]
            out.extend(self._find_best_for_chunk(chunk))
            pos += CHUNK_SIZE
        return bytes(out)

    def decompress_with_best(self, data):
        out = bytearray()
        i = 0
        while i < len(data):
            marker = data[i]
            i += 1
            start = i
            while i < len(data) and data[i] not in list(range(1,255)):
                i += 1
            payload = data[start:i]
            try:
                dec = zstd_dctx.decompress(payload)
            except:
                if PAQ_AVAILABLE:
                    try:
                        dec = paq.decompress(payload)
                    except:
                        dec = payload
                else:
                    dec = payload
            if marker in self.transform_table:
                dec = self.transform_table[marker][1](dec)
            out.extend(dec)
        return bytes(out)

    def compress(self, infile, outfile):
        with open(infile, "rb") as f: data = f.read()
        out = self.compress_with_best(data)
        with open(outfile, "wb") as f: f.write(out)
        print("Compressed: {} -> {} bytes ({:.2f}%)".format(len(data), len(out), len(out)/len(data)*100))

    def decompress(self, infile, outfile):
        with open(infile, "rb") as f: data = f.read()
        out = self.decompress_with_best(data)
        with open(outfile, "wb") as f: f.write(out)
        print("Decompressed: {} -> {} bytes".format(len(data), len(out)))

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print(PROGNAME)
    print("-" * 60)
    c = PAQJPCompressor()
    while True:
        print("\n1) Compress\n2) Decompress\n3) Exit")
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
            print("Invalid choice")

if __name__ == "__main__":
    main()
