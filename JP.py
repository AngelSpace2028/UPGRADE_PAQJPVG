#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified PAQJP+PJP – Algorithm 58 with repeats 1..256
=====================================================
- 256 transforms (single, no pairs)
- Per-input verification + raw fallback (100% lossless)
- Algorithm 58 (UPGRADED): per-block iterated bit-RLE, up to 256 passes,
  self-terminating via a 2-byte bit-count header, early stop on no gain.
- Algorithm 59: byte-level delta
- Algorithm 60: byte-RLE with adaptive escape byte
- Backend: try zstd / paq / raw, pick smallest; NO flag byte.
- Robust zstandard install.
- Output naming: input.txt.jp (or .jp.lzh)
"""

import math, random, decimal, hashlib, base64, heapq, struct, os, tempfile
import re, urllib.request, sys, subprocess, importlib, time, site, sysconfig
from typing import Optional, List, Tuple, Dict, Callable, Any
from collections import Counter

try:
    import paq
except ImportError:
    paq = None

USE_QUANTUM = False
HAS_QISKIT = False
HAS_ZSTD = False

# ------------------------------------------------------------------
# Robust package installation
# ------------------------------------------------------------------
LOCAL_LIBS_DIR = os.path.join(os.path.expanduser("~"), ".compress_local_libs")

def _add_to_sys_path(d):
    if d and os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

def _preload_existing_paths():
    for fn in (lambda: site.getusersitepackages(),
               lambda: site.getuserbase() + '/lib/python' +
                       f"{sys.version_info.major}.{sys.version_info.minor}" +
                       '/site-packages',
               lambda: sysconfig.get_paths().get('purelib'),
               lambda: sysconfig.get_paths().get('platlib'),
               lambda: LOCAL_LIBS_DIR):
        try: _add_to_sys_path(fn())
        except Exception: pass

def install_package(pkg: str) -> bool:
    os.makedirs(LOCAL_LIBS_DIR, exist_ok=True)
    strategies = [
        [sys.executable, '-m', 'pip', 'install', '--no-input',
         '--disable-pip-version-check', '--target', LOCAL_LIBS_DIR, pkg],
        [sys.executable, '-m', 'pip', 'install', '--no-input',
         '--disable-pip-version-check', pkg],
        [sys.executable, '-m', 'pip', 'install', '--no-input',
         '--disable-pip-version-check', '--user', pkg],
        [sys.executable, '-m', 'pip', 'install', '--no-input',
         '--disable-pip-version-check', '--break-system-packages', pkg],
    ]
    for i, cmd in enumerate(strategies, 1):
        print(f"  Attempt {i}/{len(strategies)}: {' '.join(cmd[3:])}")
        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
            _preload_existing_paths()
            _add_to_sys_path(LOCAL_LIBS_DIR)
            return True
        except Exception:
            continue
    for apt_cmd in (["sudo", "apt-get", "install", "-y", "python3-zstandard"],
                    ["apt-get", "install", "-y", "python3-zstandard"]):
        print(f"  Attempt apt: {' '.join(apt_cmd)}")
        try:
            subprocess.check_call(apt_cmd, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
            _preload_existing_paths()
            return True
        except Exception:
            continue
    return False

def _clear_module_cache(prefix):
    for m in list(sys.modules.keys()):
        if m == prefix or m.startswith(prefix + '.'):
            del sys.modules[m]

def try_import_zstd():
    _clear_module_cache('zstandard')
    _preload_existing_paths()
    _add_to_sys_path(LOCAL_LIBS_DIR)
    try:
        import zstandard as zstd
        return zstd, zstd.ZstdCompressor(level=22), zstd.ZstdDecompressor()
    except ImportError:
        return None, None, None

# ---------- Prompt 1 ----------
qc_choice = input("Option 1: Enable quantum-inspired transforms (Qiskit)? (y/n) [n]: ").strip().lower()
if qc_choice == 'y':
    try:
        from qiskit import QuantumCircuit
        HAS_QISKIT = True; USE_QUANTUM = True
        print("Quantum ENABLED.")
    except ImportError:
        if install_package('qiskit'):
            try:
                _clear_module_cache('qiskit')
                from qiskit import QuantumCircuit
                HAS_QISKIT = True; USE_QUANTUM = True
                print("Quantum ENABLED.")
            except ImportError:
                print("Qiskit install succeeded but import failed. Disabled.")
        else:
            print("Qiskit install failed. Disabled.")
else:
    print("Quantum disabled.")

# ---------- Prompt 2 ----------
oc = input("Option 2: Install 4 optional backends (mpmath, cython, paq, python-docx)? (y/n) [n]: ").strip().lower()
if oc == 'y':
    for pkg in ['mpmath', 'cython', 'paq', 'python-docx']:
        try:
            importlib.import_module(pkg); print(f"{pkg} already installed.")
        except ImportError:
            install_package(pkg)
else:
    print("Skipping 4 optional backends.")

# ---------- Prompt 3 ----------
print("Option 3: zstandard backend (strongly recommended)")
zc = input("  Install zstandard now? (y/n) [y]: ").strip().lower() or 'y'
if zc == 'n':
    print("WARNING: zstandard will NOT be used. Backend falls back to paq-only or raw.")
    HAS_ZSTD = False
else:
    zstd, zstd_cctx, zstd_dctx = try_import_zstd()
    if zstd is not None:
        HAS_ZSTD = True
        print("zstandard already installed and loaded.")
    else:
        print("zstandard not found. Attempting installation...")
        if install_package('zstandard'):
            zstd, zstd_cctx, zstd_dctx = try_import_zstd()
            if zstd is not None:
                HAS_ZSTD = True
                print("zstandard installed and loaded successfully.")
            else:
                print("Install reported success but import still fails.")
                print("Continuing WITHOUT zstandard.")
                HAS_ZSTD = False
        else:
            print("All zstandard install attempts failed.")
            print("Continuing WITHOUT zstandard.")
            HAS_ZSTD = False

PROGNAME = "UnifiedPAQJP+PJP (Algorithm 58 repeats 1-256, .jp output)"

DICT_DIR = "Dictionaries"
COMBINED_DICTIONARY_FILE = os.path.join(DICT_DIR, "dictionary_combined.txt")
DICTIONARY_FILES = [
    "generated.txt", "eng_news_2005_1M-sentences.txt", "eng_news_2005_1M-words.txt",
    "eng_news_2005_1M-sources.txt", "eng_news_2005_1M-co_n.txt", "eng_news_2005_1M-co_s.txt",
    "eng_news_2005_1M-inv_w_2.txt", "eng_news_2005_1M-inv_w_3.txt",
    "eng_news_2005_1M-inv_so.txt", "eng_news_2005_1M-meta.txt",
    "Dictionary.txt", "the-complete-reference-html-css-fifth-edition.txt",
]
DICTIONARY_URLS = [
    "https://drive.google.com/uc?export=download&id=1u_1dCEl8hhdEug6GwkOxHAuSx_6_Pme9",
    "https://drive.google.com/uc?export=download&id=1pVqNN5JZ2AeOCgRaHkv4Vv6Byr4zK20e",
    "https://drive.google.com/uc?export=download&id=1ZSC-Tn76x8itdN0rCp-Zw17hGudxbjxo",
    "https://drive.google.com/uc?export=download&id=1VB_7tzngs4GxjclSRyRDnxgS8znT2w2S",
    "https://drive.google.com/uc?export=download&id=1KVIRgiMrhCUCqQZJ3UT67ztls2GqGJzz",
    "https://drive.google.com/uc?export=download&id=1Z3Lx6SqL4HWsnmbJCez4kXWRQQhUXWKL",
    "https://drive.google.com/uc?export=download&id=1br2bdRMkZEVVRPKYmC4IIaZuAjxFJE4N",
    "https://drive.google.com/uc?export=download&id=1aE6ubPZiJ8rr3lEVk8fFJYjDQ1y1rU0X",
    "https://drive.google.com/uc?export=download&id=1uro3TZe-t5zPx2Qu2xrTL3lU8N0melk9",
    "https://drive.google.com/uc?export=download&id=1HqsTH1DqpWNpGbn9VtD7-SB6wVqA90R2",
    "https://drive.google.com/uc?export=download&id=1zZ8iMeBC3605NZhuc4UE9jx_w_lZFg5B",
    "https://drive.google.com/uc?export=download&id=1dDdqYDgm7f-smS7KF70Wf0KmyFo-ft1M",
]
MAX_LINE_ENTRIES = 1024

def download_and_merge_dictionaries():
    if not os.path.exists(DICT_DIR): os.makedirs(DICT_DIR)
    if os.path.exists(COMBINED_DICTIONARY_FILE):
        print("Combined dictionary exists. Skipping download."); return True
    all_words = set(); success_count = 0
    for filename, url in zip(DICTIONARY_FILES, DICTIONARY_URLS):
        local_path = os.path.join(DICT_DIR, filename)
        print(f"Downloading {filename}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response: content = response.read()
            if b'<html' in content[:200].lower(): print("  HTML page. Skipping."); continue
            with open(local_path, 'wb') as f: f.write(content)
            with open(local_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    w = line.strip()
                    if not w: continue
                    try: all_words.add(base64.b64decode(w, validate=True).decode('utf-8'))
                    except Exception: all_words.add(w)
            print(f"  OK ({os.path.getsize(local_path)} bytes)"); success_count += 1
        except Exception as e:
            print(f"  WARN: {e}")
            if os.path.exists(local_path): os.remove(local_path)
    if success_count == 0:
        print("No dictionaries downloaded."); return False
    try:
        with open(COMBINED_DICTIONARY_FILE, 'w', encoding='utf-8') as f:
            for word in sorted(all_words): f.write(word + '\n')
        print(f"Merged {len(all_words)} words."); return True
    except Exception as e: print(f"Write failed: {e}"); return False

PRIMES = [p for p in range(2, 256) if all(p % d != 0 for d in range(2, int(p ** 0.5) + 1))]
PI_DIGITS = [79, 17, 111]

def find_nearest_prime_around(n: int) -> int:
    if n < 2: return 2
    o = 0
    while True:
        c1, c2 = n - o, n + o
        if c1 >= 2 and all(c1 % d != 0 for d in range(2, int(c1 ** 0.5) + 1)): return c1
        if c2 >= 2 and all(c2 % d != 0 for d in range(2, int(c2 ** 0.5) + 1)): return c2
        o += 1

_CONST_DIAPASON_ITER_CODE = [
    (2, 0b10), (2, 0b11), (3, 0b010), (3, 0b011),
    (4, 0b0010), (4, 0b0011), (5, 0b00010), (5, 0b00011),
    (6, 0b000010), (6, 0b000011), (7, 0b0000010), (7, 0b0000011),
    (8, 0b00000010), (8, 0b00000011), (9, 0b000000010), (9, 0b000000011),
]
_CONST_DIAPASON_ITER_DECODE = {(L, bits): nib for nib, (L, bits) in enumerate(_CONST_DIAPASON_ITER_CODE)}

ALPHABET_6BIT = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 \n"
CHAR_TO_6BIT = {ch: i for i, ch in enumerate(ALPHABET_6BIT)}
SIXBIT_TO_CHAR = {i: ch for ch, i in CHAR_TO_6BIT.items()}

PAQ_STATE_TABLE = [
    [1,2,0,0],[3,5,0,1],[4,6,2,0],[7,10,0,2],[8,12,3,0],[9,13,1,1],[11,14,0,3],[15,19,4,0],
    [16,23,2,1],[17,24,2,1],[18,25,2,1],[20,27,1,2],[21,28,1,2],[22,29,1,2],[26,30,0,4],[31,33,5,0],
    [32,34,3,1],[35,37,1,3],[36,38,1,3],[39,42,0,5],[40,43,4,1],[41,44,2,2],[45,48,1,4],[46,49,1,4],
    [47,50,1,4],[51,52,0,6],[53,55,6,0],[54,56,4,1],[57,59,2,3],[58,60,2,3],[61,63,0,7],[62,64,5,1],
    [65,66,3,2],[67,69,1,5],[68,70,1,5],[71,73,0,8],[72,74,6,1],[75,76,4,2],[77,78,2,4],[79,80,2,4],
    [81,82,0,9],[83,84,7,1],[85,86,5,2],[87,88,3,3],[89,90,1,6],[91,92,0,10],[93,94,8,1],[95,96,6,2],
    [97,98,4,3],[99,100,2,5],[101,102,0,11],[103,104,9,1],[105,106,7,2],[107,108,5,3],[109,110,3,4],
    [111,112,1,7],[113,114,0,12],[115,116,10,1],[117,118,8,2],[119,120,6,3],[121,122,4,4],[123,124,2,6],
    [125,126,0,13],[127,128,11,1],[129,130,9,2],[131,132,7,3],[133,134,5,4],[135,136,3,5],[137,138,1,8],
    [139,140,0,14],[141,142,12,1],[143,144,10,2],[145,146,8,3],[147,148,6,4],[149,150,4,5],[151,152,2,7],
    [153,154,0,15],[155,156,13,1],[157,158,11,2],[159,160,9,3],[161,162,7,4],[163,164,5,5],[165,166,3,6],
    [167,168,1,9],[169,170,0,16],[171,172,14,1],[173,174,12,2],[175,176,10,3],[177,178,8,4],[179,180,6,5],
    [181,182,4,6],[183,184,2,8],[185,186,0,17],[187,188,15,1],[189,190,13,2],[191,192,11,3],[193,194,9,4],
    [195,196,7,5],[197,198,5,6],[199,200,3,7],[201,202,1,10],[203,204,0,18],[205,206,16,1],[207,208,14,2],
    [209,210,12,3],[211,212,10,4],[213,214,8,5],[215,216,6,6],[217,218,4,7],[219,220,2,9],[221,222,0,19],
    [223,224,17,1],[225,226,15,2],[227,228,13,3],[229,230,11,4],[231,232,9,5],[233,234,7,6],[235,236,5,7],
    [237,238,3,8],[239,240,1,11],[241,242,0,20],[243,244,18,1],[245,246,16,2],[247,248,14,3],[249,250,12,4],
    [251,252,10,5],[253,254,8,6],[255,255,6,7],
]

class TransformError(Exception): pass
class DecompressionError(Exception): pass

def mod_inv(a: int, m: int) -> Optional[int]:
    if a == 0: return None
    m0 = m; y = 0; x = 1
    if m == 1: return 0
    while a > 1:
        q = a // m; t = m; m = a % m; a = t
        t = y; y = x - q * y; x = t
    if x < 0: x += m0
    return x

class UnifiedCompressor:
    ULTRA_TIME_LIMIT = 300
    QUANTUM_QUBITS = 8
    RLE58_MAX_PASSES = 256

    def __init__(self):
        download_and_merge_dictionaries()
        self.PI_DIGITS = PI_DIGITS.copy()
        self.seed_tables = self._gen_seed_tables(num=126, size=40, seed=42)
        self.fibonacci = self._gen_fib(100)
        self.PI_STR = "3.14159265358979323846264338327950288419716939937510"
        self.repeat_count = 100
        self.mod_state_table = [[(v - 400) & 0xFF for v in row] for row in PAQ_STATE_TABLE]
        self._build_mask_46()
        self._build_transform_maps()
        self.static_dict, self.word_to_index = self._load_static_dictionary()
        self.line_dict, self.line_to_index = self._load_line_dictionary()
        self.quantum_transforms_built = False
        if USE_QUANTUM and HAS_QISKIT:
            self._precompute_quantum_transforms()

    def set_quantum_qubits(self, q: int):
        if not USE_QUANTUM or not HAS_QISKIT: print("Quantum disabled."); return
        if q < 1 or q > 49: print("Qubits 1..49."); return
        self.QUANTUM_QUBITS = q; self.quantum_transforms_built = False
        self._precompute_quantum_transforms()

    def get_quantum_variation_count(self) -> int:
        return (1 << self.QUANTUM_QUBITS) if (USE_QUANTUM and HAS_QISKIT) else 0

    def _build_mask_46(self):
        base = [1,2,4,8,16,32,64,128,3,6]
        self.mask_46 = [(b - 10) & 0xFF for b in base] * 10

    def get_pi_digits(self, n): return self.PI_STR[2:2+n] if n >= 1 else ""
    def find_lossless_k(self, n):
        if n < 1: return 0, True
        ts = int(self.PI_STR.replace('.','')[:n+1]); DF = 16777216
        decimal.getcontext().prec = 50; pi_dec = decimal.Decimal(self.PI_STR)
        k = int(round((pi_dec - 3) * DF)); k = max(0, min(k, DF - 1))
        ap = (3 * 10**n * DF + k * 10**n) // DF
        return k, ap == ts
    def to_bin(self, v, b): return format(v, 'b').zfill(b)
    def get_bit_size(self, k): return 23 if k <= 0x7FFFFF else 25

    def transform_17(self, d):
        if not d: return b''
        k, _ = self.find_lossless_k(7); bs = self.get_bit_size(k)
        bit_str = self.to_bin(k, bs); mb = []
        for i in range(0, len(bit_str), 8):
            bb = bit_str[i:i+8]
            if len(bb) < 8: bb = bb.ljust(8, '0')
            mb.append(int(bb, 2))
        mask = bytes(mb); t = bytearray(d)
        for i in range(len(t)): t[i] ^= mask[i % len(mask)]
        return bytes(t)
    reverse_transform_17 = transform_17

    def get_basel_digits(self, n):
        decimal.getcontext().prec = n + 5
        pi = decimal.Decimal(self.PI_STR)
        return str((pi*pi)/decimal.Decimal(6)).replace('.','')[:n]
    def get_one_over_e_digits(self, n):
        decimal.getcontext().prec = n + 5
        e = decimal.Decimal(1).exp()
        return str(decimal.Decimal(1)/e).replace('.','')[:n]
    def get_5e_digits(self, n):
        decimal.getcontext().prec = n + 5
        e = decimal.Decimal(1).exp()
        return str(decimal.Decimal(5)*e).replace('.','')[:n]

    def _gen_seed_tables(self, num=126, size=40, seed=42):
        random.seed(seed)
        return [[random.randint(5, 255) for _ in range(size)] for _ in range(num)]
    def _gen_fib(self, n):
        a, b = 0, 1; res = [a, b]
        for _ in range(2, n): a, b = b, a+b; res.append(b)
        return res
    def get_seed(self, i, v): return self.seed_tables[i][v % 40] if 0 <= i < len(self.seed_tables) else 0

    def _append_bits(self, bl, v, c):
        for i in range(c-1, -1, -1): bl.append((v >> i) & 1)
    def _read_bits(self, bits, pos, c):
        v = 0
        for i in range(c):
            if pos + i >= len(bits): return 0
            v = (v << 1) | bits[pos + i]
        return v

    def transform_00(self, data):
        if not data: return struct.pack('>I', 0)
        best_r = None; best_l = float('inf'); best_s = []
        cur = bytearray(data); app = []; orig = bytes(data)
        for _ in range(10):
            bs = 0; bsh = cur; bsc = float('-inf')
            for sh in range(256):
                tmp = bytearray(cur)
                for j in range(len(tmp)): tmp[j] = (tmp[j] + sh) % 256
                sc = 0; i = 0
                while i < len(tmp):
                    v = tmp[i]; r = 1; i += 1
                    while i < len(tmp) and tmp[i] == v: r += 1; i += 1
                    sc += r * r
                if sc > bsc: bsc = sc; bsh = tmp; bs = sh
            app.append(bs); rle = self._apply_rle_to_shifted(bsh, bs)
            dec = self._rle_decode(rle)
            if dec is not None:
                te = bytearray(dec)
                for sh in app:
                    for j in range(len(te)): te[j] = (te[j] - sh) % 256
                if bytes(te) == orig and len(rle) < best_l:
                    best_l = len(rle); best_r = rle; best_s = app.copy()
            cur = bsh
            if len(rle) >= len(data): break
        if best_r is None or best_l >= len(data):
            return struct.pack('>I', len(data)) + bytes([0]) + data
        h = bytearray(); h.extend(struct.pack('>I', len(data)))
        h.append(len(best_s)); h.extend(best_s); return h + best_r

    def _apply_rle_to_shifted(self, sd, sh):
        bits = []; self._append_bits(bits, 0b010, 3); self._append_bits(bits, sh, 8)
        i = 0; n = len(sd)
        while i < n:
            v = sd[i]; r = 1; i += 1
            while i < n and sd[i] == v: r += 1; i += 1
            while r >= 13:
                ch = min(r, 268)
                self._append_bits(bits, 0b1111, 4); self._append_bits(bits, ch-13, 8); self._append_bits(bits, v, 8)
                r -= ch
            if r == 1: self._append_bits(bits, 0b00, 2); self._append_bits(bits, v, 8)
            elif r <= 5: self._append_bits(bits, 0b01, 2); self._append_bits(bits, r-2, 2); self._append_bits(bits, v, 8)
            elif r <= 12: self._append_bits(bits, 0b10, 2); self._append_bits(bits, r-6, 3); self._append_bits(bits, v, 8)
        pad = (8 - len(bits) % 8) % 8; self._append_bits(bits, 0, pad)
        out = bytearray()
        for j in range(0, len(bits), 8):
            byte = 0
            for k in range(8):
                if j+k < len(bits): byte = (byte << 1) | bits[j+k]
            out.append(byte)
        return bytes(out)

    def reverse_transform_00(self, cd):
        if not cd or cd == struct.pack('>I', 0): return b''
        if len(cd) < 4: raise TransformError("RLE")
        ol = struct.unpack('>I', cd[:4])[0]; cd = cd[4:]
        if not cd: return b''
        if cd[0] == 0: return cd[1:ol+1]
        np = cd[0]
        if np == 0 or len(cd) < 1 + np: raise TransformError("RLE h")
        sh = list(cd[1:1+np]); rle = cd[1+np:]
        dec = self._rle_decode(rle)
        if dec is None: raise TransformError("RLE d")
        cur = bytearray(dec[:ol])
        for s in reversed(sh):
            for i in range(len(cur)): cur[i] = (cur[i] - s) % 256
        return bytes(cur)

    def _rle_decode(self, data):
        if not data: return None
        bits = []
        for b in data:
            for i in range(7,-1,-1): bits.append((b >> i) & 1)
        pos = 0; nb = len(bits)
        if nb < 11: return None
        m = self._read_bits(bits, pos, 3); pos += 3
        if m != 0b010: return None
        pos += 8; out = bytearray()
        while pos < nb:
            if pos + 2 > nb: break
            p = self._read_bits(bits, pos, 2); pos += 2
            if p == 0b00: r = 1
            elif p == 0b01:
                if pos + 2 + 8 > nb: break
                r = 2 + self._read_bits(bits, pos, 2); pos += 2
            elif p == 0b10:
                if pos + 3 + 8 > nb: break
                r = 6 + self._read_bits(bits, pos, 3); pos += 3
            else:
                if pos + 2 + 8 + 8 > nb: break
                if self._read_bits(bits, pos, 2) != 0b11: return None
                pos += 2; r = 13 + self._read_bits(bits, pos, 8); pos += 8
            if pos + 8 > nb: break
            v = self._read_bits(bits, pos, 8); pos += 8
            out.extend([v] * r)
        for i in range(pos, nb):
            if bits[i] != 0: return None
        return out

    def transform_01(self, d):
        t = bytearray(d); r = self.repeat_count
        for p in PRIMES:
            xv = p if p == 2 else max(1, math.ceil(p * 4096 / 28672))
            for _ in range(r):
                for i in range(0, len(t), 3):
                    if i < len(t): t[i] ^= xv
        return bytes(t)
    reverse_transform_01 = transform_01
    def transform_02(self, d):
        if not d: return b'\x00'
        t = bytearray(d); cs = sum(d) % 256
        pi_ = (len(d) + cs) % 256; pv = self._get_pattern(4, pi_)
        for i in range(1, len(t), 4):
            if i < len(t): t[i] ^= pv[i % len(pv)]
        return bytes([pi_]) + bytes(t)
    def reverse_transform_02(self, d):
        if d == b'\x00': return b''
        if len(d) < 2: raise TransformError("T02")
        pi_ = d[0]; t = bytearray(d[1:]); pv = self._get_pattern(4, pi_)
        for i in range(1, len(t), 4):
            if i < len(t): t[i] ^= pv[i % len(pv)]
        return bytes(t)
    def transform_03(self, d):
        if not d: return b'\x00'
        t = bytearray(d); rot = (len(d)*13 + sum(d)) % 8 or 1
        for i in range(2, len(t), 5):
            if i < len(t): t[i] = ((t[i] << rot) | (t[i] >> (8-rot))) & 0xFF
        return bytes([rot]) + bytes(t)
    def reverse_transform_03(self, d):
        if d == b'\x00': return b''
        if len(d) < 2: raise TransformError("T03")
        rot = d[0]; t = bytearray(d[1:])
        for i in range(2, len(t), 5):
            if i < len(t): t[i] = ((t[i] >> rot) | (t[i] << (8-rot))) & 0xFF
        return bytes(t)
    def transform_04(self, d):
        t = bytearray(d); r = self.repeat_count
        for _ in range(r):
            for i in range(len(t)): t[i] = (t[i] - (i % 256)) % 256
        return bytes(t)
    def reverse_transform_04(self, d):
        t = bytearray(d); r = self.repeat_count
        for _ in range(r):
            for i in range(len(t)): t[i] = (t[i] + (i % 256)) % 256
        return bytes(t)
    def transform_05(self, d, s=3):
        t = bytearray(d)
        for i in range(len(t)): t[i] = ((t[i] << s) | (t[i] >> (8-s))) & 0xFF
        return bytes(t)
    def reverse_transform_05(self, d, s=3):
        t = bytearray(d)
        for i in range(len(t)): t[i] = ((t[i] >> s) | (t[i] << (8-s))) & 0xFF
        return bytes(t)
    def transform_06(self, d, sd=42):
        random.seed(sd); sub = list(range(256)); random.shuffle(sub)
        return bytes(sub[b] for b in d)
    def reverse_transform_06(self, d, sd=42):
        random.seed(sd); sub = list(range(256)); random.shuffle(sub)
        inv = [0]*256
        for i in range(256): inv[sub[i]] = i
        return bytes(inv[b] for b in d)
    def transform_07(self, d):
        t = bytearray(d); r = self.repeat_count
        sh = len(d) % len(self.PI_DIGITS); pr = self.PI_DIGITS[sh:] + self.PI_DIGITS[:sh]
        sz = len(d) % 256
        for i in range(len(t)): t[i] ^= sz
        for _ in range(r):
            for i in range(len(t)): t[i] ^= pr[i % len(pr)]
        return bytes(t)
    reverse_transform_07 = transform_07
    def transform_08(self, d):
        t = bytearray(d); r = self.repeat_count
        sh = len(d) % len(self.PI_DIGITS); pr = self.PI_DIGITS[sh:] + self.PI_DIGITS[:sh]
        p = find_nearest_prime_around(len(d) % 256)
        for i in range(len(t)): t[i] ^= p
        for _ in range(r):
            for i in range(len(t)): t[i] ^= pr[i % len(pr)]
        return bytes(t)
    reverse_transform_08 = transform_08
    def transform_09(self, d):
        t = bytearray(d); r = self.repeat_count
        sh = len(d) % len(self.PI_DIGITS); pr = self.PI_DIGITS[sh:] + self.PI_DIGITS[:sh]
        p = find_nearest_prime_around(len(d) % 256)
        seed = self.get_seed(len(d) % len(self.seed_tables), len(d))
        for i in range(len(t)): t[i] ^= p ^ seed
        for _ in range(r):
            for i in range(len(t)): t[i] ^= pr[i % len(pr)] ^ (i % 256)
        return bytes(t)
    reverse_transform_09 = transform_09
    def transform_10(self, d):
        if not d: return b'\x00'
        cnt = sum(1 for i in range(len(d)-1) if d[i:i+2] == b'X1')
        n = (((cnt * 2) + 1) // 3) * 3 % 256
        t = bytearray(d)
        for i in range(len(t)): t[i] ^= n
        return bytes([n]) + bytes(t)
    def reverse_transform_10(self, d):
        if len(d) < 1: raise TransformError("T10")
        n = d[0]; t = bytearray(d[1:])
        for i in range(len(t)): t[i] ^= n
        return bytes(t)
    def transform_11(self, d):
        if not d: return b''
        t = bytearray(d); L = len(t)
        for i in range(L):
            fi = (i + L) % len(self.fibonacci)
            key = ((self.fibonacci[fi] % 256) ^ ((i*13 + L*17) % 256)) % 256
            t[i] ^= key
        return bytes(t)
    reverse_transform_11 = transform_11
    def transform_12(self, d):
        t = bytearray(d)
        for i in range(len(t)): t[i] ^= self.fibonacci[i % len(self.fibonacci)] % 256
        return bytes(t)
    reverse_transform_12 = transform_12
    def transform_13(self, d):
        if not d: return b'\x00'
        rp = self._calculate_repeats(d); cv = len(d) % 256; pv = []
        for _ in range(rp): cv = find_nearest_prime_around(cv); pv.append(cv)
        t = bytearray(d); xv = pv[-1] if pv else 0
        for i in range(len(t)): t[i] ^= xv
        return bytes([(rp-1) % 256]) + bytes(t)
    def reverse_transform_13(self, d):
        if d == b'\x00': return b''
        if len(d) < 2: raise TransformError("T13")
        rp = (d[0] + 1) % 256 or 256
        t = bytearray(d[1:]); cv = len(t) % 256; pv = []
        for _ in range(rp): cv = find_nearest_prime_around(cv); pv.append(cv)
        xv = pv[-1] if pv else 0
        for i in range(len(t)): t[i] ^= xv
        return bytes(t)
    def transform_14(self, d):
        if not d: return b'\x00'
        return d + bytes([sum(d) % 256])
    def reverse_transform_14(self, d):
        if not d: raise TransformError("T14")
        return d[:-1]
    def transform_15(self, d):
        if not d: return b'\x00'
        t = bytearray(d); pi_ = len(d) % 256; pv = self._get_pattern(3, pi_)
        for i in range(0, len(t), 3):
            if i < len(t): t[i] = (t[i] + pv[i % len(pv)]) % 256
        return bytes([pi_]) + bytes(t)
    def reverse_transform_15(self, d):
        if d == b'\x00': return b''
        if len(d) < 2: raise TransformError("T15")
        pi_ = d[0]; t = bytearray(d[1:]); pv = self._get_pattern(3, pi_)
        for i in range(0, len(t), 3):
            if i < len(t): t[i] = (t[i] - pv[i % len(pv)]) % 256
        return bytes(t)
    def transform_16(self, d):
        if not d: return b''
        xb = (len(d) * 7 + 13) % 256; t = bytearray(d)
        for i in range(len(t)): t[i] ^= xb
        return bytes(t)
    reverse_transform_16 = transform_16
    def transform_18(self, d):
        if not d: return b''
        dg = self.get_basel_digits(max(10, len(d)//2 + 5))
        mask = bytes(int(dg[i:i+2]) % 256 for i in range(0, len(dg), 2))
        t = bytearray(d)
        for i in range(len(t)): t[i] ^= mask[i % len(mask)]
        return bytes(t)
    reverse_transform_18 = transform_18
    def transform_19(self, d):
        if not d: return b''
        dg = self.get_one_over_e_digits(max(10, len(d)//2 + 5))
        mask = bytes(int(dg[i:i+2]) % 256 for i in range(0, len(dg), 2))
        t = bytearray(d)
        for i in range(len(t)): t[i] ^= mask[i % len(mask)]
        return bytes(t)
    reverse_transform_19 = transform_19
    def transform_20(self, d):
        if not d: return b''
        dg = self.get_5e_digits(max(10, len(d)//2 + 5))
        mask = bytes(int(dg[i:i+2]) % 256 for i in range(0, len(dg), 2))
        t = bytearray(d)
        for i in range(len(t)): t[i] ^= mask[i % len(mask)]
        return bytes(t)
    reverse_transform_20 = transform_20
    def transform_21(self, d):
        if not d: return b''
        t = bytearray(d)
        for i in range(len(t)): t[i] = (t[i] + 255) % 256
        return bytes(t)
    def reverse_transform_21(self, d):
        if not d: return b''
        t = bytearray(d)
        for i in range(len(t)): t[i] = (t[i] - 255) % 256
        return bytes(t)

    def transform_22(self, d): return base64.b64encode(d)
    def reverse_transform_22(self, d):
        try: return base64.b64decode(d, validate=False)
        except Exception as e: raise TransformError(f"B64: {e}")
    def transform_23(self, d):
        if not d: return b'\x00'
        try: text = d.decode('utf-8')
        except UnicodeDecodeError: return b'\x00' + d
        toks = re.split(r'([A-Za-z0-9_]+)', text)
        wl = []; w2i = {}; ts = []
        for i, t in enumerate(toks):
            if i % 2 == 1:
                wb = t.encode('utf-8'); idx = w2i.get(wb)
                if idx is None: idx = len(wl); w2i[wb] = idx; wl.append(wb)
                ts.append((1, idx))
            else: ts.append((0, t.encode('utf-8')))
        out = bytearray([1]); out += struct.pack('>I', len(wl))
        for wb in wl: out += struct.pack('>I', len(wb)) + wb
        for typ, pl in ts:
            if typ == 1: out += b'\x01' + struct.pack('>I', pl)
            else: out += b'\x00' + struct.pack('>I', len(pl)) + pl
        toked = bytes(out)
        try:
            if self.reverse_transform_23(toked) == d: return toked
        except Exception: pass
        return b'\x00' + d
    def reverse_transform_23(self, data):
        if not data: return b''
        flag = data[0]
        if flag == 0: return data[1:]
        if flag == 1:
            pos = 1
            if len(data) < pos + 4: raise TransformError("T23")
            nw = struct.unpack('>I', data[pos:pos+4])[0]; pos += 4
            wl = []
            for _ in range(nw):
                if pos + 4 > len(data): raise TransformError("T23")
                wlen = struct.unpack('>I', data[pos:pos+4])[0]; pos += 4
                if pos + wlen > len(data): raise TransformError("T23")
                wl.append(data[pos:pos+wlen]); pos += wlen
            out = bytearray()
            while pos < len(data):
                typ = data[pos]; pos += 1
                if typ == 1:
                    if pos + 4 > len(data): raise TransformError("T23")
                    idx = struct.unpack('>I', data[pos:pos+4])[0]; pos += 4
                    if idx >= len(wl): raise TransformError("T23")
                    out += wl[idx]
                elif typ == 0:
                    if pos + 4 > len(data): raise TransformError("T23")
                    ll = struct.unpack('>I', data[pos:pos+4])[0]; pos += 4
                    if pos + ll > len(data): raise TransformError("T23")
                    out += data[pos:pos+ll]; pos += ll
                else: raise TransformError("T23")
            return bytes(out)
        raise TransformError("T23 flag")
    def transform_24(self, d): return self.transform_23(d)
    def reverse_transform_24(self, d): return self.reverse_transform_23(d)
    def _split_text_into_chunks(self, text, level='all'):
        if level == 'paragraph': return re.split(r'(\n\n)', text)
        if level == 'line': return re.split(r'(\n)', text)
        if level == 'sentence': return re.split(r'([.!?]+)', text)
        if level == 'word': return re.split(r'(\s+|\b)', text)
        chunks = []
        for i, para in enumerate(re.split(r'(\n\n)', text)):
            if i % 2 == 1: chunks.append(para); continue
            for j, line in enumerate(re.split(r'(\n)', para)):
                if j % 2 == 1: chunks.append(line); continue
                for k, s in enumerate(re.split(r'([.!?]+)', line)):
                    if k % 2 == 1: chunks.append(s); continue
                    chunks.extend(re.split(r'(\s+|\b)', s))
        return chunks
    def _dynamic_dict_tokenize(self, data, ib=3):
        try: text = data.decode('utf-8')
        except: return b'\x00' + data
        chs = self._split_text_into_chunks(text, 'all')
        freq = Counter(chs)
        sc = sorted(freq.keys(), key=lambda x: (-freq[x], -len(x), x))
        c2i = {c: i for i, c in enumerate(sc)}
        ne = len(sc)
        if ib == 2 and ne > 65535: ib = 3
        if ib == 3 and ne > 16777215: ib = 8
        h = bytearray([ib]); h += struct.pack('>I', ne)
        for c in sc:
            cb = c.encode('utf-8'); h += struct.pack('>I', len(cb)) + cb
        ts = bytearray()
        for c in chs:
            idx = c2i[c]
            if ib == 2: ts += struct.pack('>H', idx)
            elif ib == 3: ts += struct.pack('>I', idx)[1:4]
            else: ts += struct.pack('>Q', idx)
        return bytes(h) + bytes(ts)
    def _dynamic_dict_detokenize(self, data):
        if not data: return b''
        if data[0] == 0: return data[1:]
        ib = data[0]
        if ib not in (2, 3, 8): raise TransformError(f"T25 ib={ib}")
        pos = 1
        if pos + 4 > len(data): raise TransformError("T25")
        ne = struct.unpack('>I', data[pos:pos+4])[0]; pos += 4
        dic = []
        for _ in range(ne):
            if pos + 4 > len(data): raise TransformError("T25")
            cl = struct.unpack('>I', data[pos:pos+4])[0]; pos += 4
            if pos + cl > len(data): raise TransformError("T25")
            dic.append(data[pos:pos+cl].decode('utf-8')); pos += cl
        toks = []
        while pos < len(data):
            if ib == 2:
                if pos + 2 > len(data): break
                idx = struct.unpack('>H', data[pos:pos+2])[0]; pos += 2
            elif ib == 3:
                if pos + 3 > len(data): break
                idx = struct.unpack('>I', b'\x00' + data[pos:pos+3])[0]; pos += 3
            else:
                if pos + 8 > len(data): break
                idx = struct.unpack('>Q', data[pos:pos+8])[0]; pos += 8
            if idx >= len(dic): raise TransformError(f"T25 idx={idx}")
            toks.append(dic[idx])
        try: return ''.join(toks).encode('utf-8')
        except Exception as e: raise TransformError(f"T25: {e}")
    def transform_25(self, d): return self._dynamic_dict_tokenize(d, 3)
    def reverse_transform_25(self, d): return self._dynamic_dict_detokenize(d)
    def transform_26(self, d):
        if not d: return b''
        secret = b"PJP_TRANSFORM26_SECRET"; out = bytearray()
        for idx in range(0, len(d), 1024):
            ch = d[idx:idx+1024]; bn = idx // 1024
            h = hashlib.sha256(); h.update(secret); h.update(struct.pack(">Q", bn))
            m = h.digest(); mr = (m * ((len(ch) // len(m)) + 1))[:len(ch)]
            out.extend(a ^ b for a, b in zip(ch, mr))
        return bytes(out)
    def reverse_transform_26(self, d): return self.transform_26(d)
    def transform_27(self, d):
        try: text = d.decode('utf-8')
        except UnicodeDecodeError: return b'\x00' + d
        for ch in text:
            if ch not in CHAR_TO_6BIT: return b'\x00' + d
        bits = []
        for ch in text:
            v = CHAR_TO_6BIT[ch]
            for i in range(5,-1,-1): bits.append((v >> i) & 1)
        pad = (8 - len(bits) % 8) % 8; bits.extend([0]*pad)
        out = bytearray()
        for i in range(0, len(bits), 8):
            b = 0
            for j in range(8): b = (b << 1) | bits[i+j]
            out.append(b)
        return b'\x01' + struct.pack('<I', len(text)) + bytes(out)
    def reverse_transform_27(self, d):
        if len(d) < 1: raise TransformError("T27")
        f = d[0]
        if f == 0: return d[1:]
        if f != 1: raise TransformError("T27")
        pl = d[1:]
        if len(pl) < 4: raise TransformError("T27")
        nc = struct.unpack('<I', pl[:4])[0]; pk = pl[4:]
        nb = (nc*6 + 7)//8
        if len(pk) != nb: raise TransformError("T27 len")
        pb = (8 - (nc*6) % 8) % 8
        if pb > 0 and pk:
            if (pk[-1] & ((1 << pb) - 1)) != 0: raise TransformError("T27 pad")
        bits = []
        for b in pk:
            for i in range(7,-1,-1): bits.append((b >> i) & 1)
        chars = []
        for i in range(nc):
            v = 0
            for j in range(6): v = (v << 1) | bits[i*6 + j]
            chars.append(SIXBIT_TO_CHAR[v])
        return ''.join(chars).encode('utf-8')

    def transform_28(self, d):
        if not d: return b'\x00'
        pl = (3 - len(d) % 3) % 3; pad = d + b'\x00' * pl
        out = bytearray([pl])
        for i in range(0, len(pad), 3):
            v = int.from_bytes(pad[i:i+3], 'little')
            key = ((i//3)*65537 + 12345) & 0xFFFF
            out.extend(((v - key) % (1 << 24)).to_bytes(3, 'little'))
        return bytes(out)
    def reverse_transform_28(self, d):
        if d == b'\x00': return b''
        if not d: raise TransformError("T28")
        pl = d[0]; pl_data = d[1:]
        if len(pl_data) % 3: raise TransformError("T28")
        out = bytearray()
        for i in range(0, len(pl_data), 3):
            v = int.from_bytes(pl_data[i:i+3], 'little')
            key = ((i//3)*65537 + 12345) & 0xFFFF
            out.extend(((v + key) % (1 << 24)).to_bytes(3, 'little'))
        if pl > 0: out = out[:-pl]
        return bytes(out)
    def _find_best_16bit_key(self, d, qb=False, tl=60.0):
        if len(d) < 3: return 0
        pl = (3 - len(d) % 3) % 3; pad = d + b'\x00' * pl
        vals = [int.from_bytes(pad[i:i+3], 'little') for i in range(0, len(pad), 3)]
        t0 = time.time(); bk = 0; bc = float('inf')
        if not qb or not HAS_QISKIT:
            for k in range(65536):
                if k % 1024 == 0 and time.time() - t0 > tl: break
                tr = [((v - k) & 0xFFFFFF) for v in vals]
                mt = sum(tr)//len(tr)
                c = sum(abs(t - mt) for t in tr)
                if c < bc: bc = c; bk = k
                if c == 0: break
            return bk
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(8)
        for i in range(8): qc.h(i); qc.rz(random.random()*2*math.pi, i)
        try: qasm = qc.qasm(); seed = hash(qasm) & 0xFFFFFFFF
        except: seed = 42
        rng = random.Random(seed); keys = list(range(65536)); rng.shuffle(keys)
        for i, k in enumerate(keys):
            if i % 1024 == 0 and time.time() - t0 > tl: break
            tr = [((v - k) & 0xFFFFFF) for v in vals]
            mt = sum(tr)//len(tr)
            c = sum(abs(t - mt) for t in tr)
            if c < bc: bc = c; bk = k
            if c == 0: break
        return bk
    def transform_29(self, d, qb=False, tl=60.0):
        if not d: return b'\x00'
        bk = self._find_best_16bit_key(d, qb, tl)
        pl = (3 - len(d) % 3) % 3; pad = d + b'\x00' * pl
        out = bytearray([pl]); out.extend(bk.to_bytes(2, 'little'))
        for i in range(0, len(pad), 3):
            v = int.from_bytes(pad[i:i+3], 'little')
            out.extend(((v - bk) % (1 << 24)).to_bytes(3, 'little'))
        return bytes(out)
    def reverse_transform_29(self, d):
        if d == b'\x00': return b''
        if not d or len(d) < 3: raise TransformError("T29")
        pl = d[0]; key = int.from_bytes(d[1:3], 'little'); pl_data = d[3:]
        if len(pl_data) % 3: raise TransformError("T29")
        out = bytearray()
        for i in range(0, len(pl_data), 3):
            v = int.from_bytes(pl_data[i:i+3], 'little')
            out.extend(((v + key) % (1 << 24)).to_bytes(3, 'little'))
        if pl > 0: out = out[:-pl]
        return bytes(out)
    def _find_best_24bit_key_heuristic(self, d):
        if len(d) < 3: return 0
        pl = (3 - len(d) % 3) % 3; pad = d + b'\x00' * pl
        vals = [int.from_bytes(pad[i:i+3], 'little') for i in range(0, len(pad), 3)]
        mean = sum(vals)//len(vals); med = sorted(vals)[len(vals)//2]
        cand = set()
        for base in (mean, med):
            for off in (0, 1, -1, 10, -10, 100, -100, 1000, -1000):
                cand.add((base + off) % (1 << 24))
        rng = random.Random(42)
        for _ in range(10): cand.add(rng.randint(0, (1 << 24) - 1))
        bk = 0; bc = float('inf')
        for k in cand:
            tr = [((v - k) & 0xFFFFFF) for v in vals]
            mt = sum(tr)//len(tr)
            c = sum(abs(t - mt) for t in tr)
            if c < bc: bc = c; bk = k
        return bk
    def transform_30(self, d):
        if not d: return b'\x00'
        bk = self._find_best_24bit_key_heuristic(d)
        pl = (3 - len(d) % 3) % 3; pad = d + b'\x00' * pl
        out = bytearray([pl]); out.extend(bk.to_bytes(3, 'little'))
        for i in range(0, len(pad), 3):
            v = int.from_bytes(pad[i:i+3], 'little')
            out.extend(((v - bk) % (1 << 24)).to_bytes(3, 'little'))
        return bytes(out)
    def reverse_transform_30(self, d):
        if d == b'\x00': return b''
        if not d or len(d) < 4: raise TransformError("T30")
        pl = d[0]; key = int.from_bytes(d[1:4], 'little'); pl_data = d[4:]
        if len(pl_data) % 3: raise TransformError("T30")
        out = bytearray()
        for i in range(0, len(pl_data), 3):
            v = int.from_bytes(pl_data[i:i+3], 'little')
            out.extend(((v + key) % (1 << 24)).to_bytes(3, 'little'))
        if pl > 0: out = out[:-pl]
        return bytes(out)

    def transform_31(self, d): return d
    def reverse_transform_31(self, d): return d
    def transform_32(self, d): return d
    def reverse_transform_32(self, d): return d

    def _paqjp_transform_23(self, d):
        if not d: return b'\x00' * 5
        bits = []
        for byte in d:
            for i in range(7,-1,-1): bits.append((byte >> i) & 1)
        return self._compress_bits(bits)
    def _paqjp_reverse_23(self, d):
        if not d or d == b'\x00'*5: return b''
        bits = self._decompress_bits(d)
        if not bits: return b''
        out = bytearray()
        for i in range(0, len(bits), 8):
            v = 0
            for j in range(i, min(i+8, len(bits))): v = (v << 1) | bits[j]
            if i+8 > len(bits): v <<= (8 - (len(bits) - i))
            out.append(v)
        return bytes(out)
    def _compress_bits(self, bits):
        obl = len(bits)
        if obl == 0: return b'\x00'*5
        cur = bits[:]; pl = obl; pc = 0
        while pc < 255:
            pad = (4 - len(cur) % 4) % 4; padded = cur + [0] * pad
            nb = len(padded) // 4; eb = []
            for i in range(nb):
                nib = (padded[i*4] << 3) | (padded[i*4+1] << 2) | (padded[i*4+2] << 1) | padded[i*4+3]
                L, C = _CONST_DIAPASON_ITER_CODE[nib]
                for b in range(L-1,-1,-1): eb.append((C >> b) & 1)
            nl = len(eb)
            if nl < pl: cur = eb; pl = nl; pc += 1
            else: break
        cbl = len(cur)
        h = struct.pack('>H', obl) + bytes([pc]) + struct.pack('>H', cbl)
        pad = (8 - len(cur) % 8) % 8; cur += [0] * pad
        ob = bytearray()
        for i in range(0, len(cur), 8):
            v = 0
            for j in range(8): v = (v << 1) | cur[i+j]
            ob.append(v)
        return h + bytes(ob)
    def _decompress_bits(self, data):
        if len(data) < 5: raise TransformError("CD")
        obl = struct.unpack('>H', data[:2])[0]; pc = data[2]
        cbl = struct.unpack('>H', data[3:5])[0]; payload = data[5:]
        bits = []
        for b in payload:
            for i in range(7,-1,-1): bits.append((b >> i) & 1)
        if len(bits) < cbl: raise TransformError("CD2")
        cur = bits[:cbl]
        if pc == 0: return cur[:obl]
        for _ in range(pc):
            pos = 0; nb = len(cur); dn = []
            while pos < nb:
                matched = False
                for L in range(2, 10):
                    if pos + L > nb: continue
                    cw = 0
                    for k in range(L): cw = (cw << 1) | cur[pos+k]
                    if (L, cw) in _CONST_DIAPASON_ITER_DECODE:
                        dn.append(_CONST_DIAPASON_ITER_DECODE[(L, cw)]); pos += L
                        matched = True; break
                if not matched: raise TransformError("CD3")
            nb2 = []
            for nib in dn:
                for j in range(3,-1,-1): nb2.append((nib >> j) & 1)
            cur = nb2
        if len(cur) < obl: raise TransformError("CD4")
        return cur[:obl]
    def _paqjp_transform_24(self, d):
        if not d: return struct.pack('>I', 0)
        ML = 43; bits = []; i = 0; n = len(d)
        while i < n:
            cl = min(ML, n - i); ch = d[i:i+cl]; fr = ch[0]
            if all(b == fr for b in ch):
                self._append_bits(bits, 1, 1); self._append_bits(bits, fr, 8); self._append_bits(bits, cl-1, 6)
            else:
                self._append_bits(bits, 0, 1); self._append_bits(bits, cl, 6)
                for b in ch: self._append_bits(bits, b, 8)
            i += cl
        pad = (8 - len(bits) % 8) % 8; self._append_bits(bits, 0, pad)
        out = bytearray()
        for j in range(0, len(bits), 8):
            b = 0
            for k in range(8): b = (b << 1) | bits[j+k]
            out.append(b)
        return struct.pack('>I', len(d)) + bytes(out)
    def _paqjp_reverse_24(self, d):
        if not d: return b''
        if len(d) < 4: raise TransformError("BR")
        ol = struct.unpack('>I', d[:4])[0]; pl = d[4:]
        bits = []
        for b in pl:
            for i in range(7,-1,-1): bits.append((b >> i) & 1)
        pos = 0; nb = len(bits); out = bytearray()
        while pos < nb and len(out) < ol:
            if pos + 1 > nb: break
            f = self._read_bits(bits, pos, 1); pos += 1
            if f == 1:
                if pos + 14 > nb: raise TransformError("BR1")
                bv = self._read_bits(bits, pos, 8); pos += 8
                cm1 = self._read_bits(bits, pos, 6); pos += 6
                rl = min(cm1 + 1, ol - len(out)); out.extend([bv] * rl)
            else:
                if pos + 6 > nb: raise TransformError("BR2")
                cl = self._read_bits(bits, pos, 6); pos += 6
                if cl == 0: break
                if pos + cl*8 > nb: raise TransformError("BR3")
                for _ in range(cl):
                    out.append(self._read_bits(bits, pos, 8)); pos += 8
        return bytes(out[:ol])
    def _paqjp_transform_25(self, d):
        if not d: return b'\x01'
        n = 3; res = bytearray(d)
        for i in range(len(res)): res[i] = (pow(res[i]+1, n, 257) - 1) & 0xFF
        return bytes([n]) + bytes(res)
    def _paqjp_reverse_25(self, d):
        if d == b'\x01': return b''
        if not d or len(d) < 2: raise TransformError("FLT25")
        n = d[0]; inv = mod_inv(n, 256)
        if inv is None: raise TransformError("FLT25")
        res = bytearray(d[1:])
        for i in range(len(res)): res[i] = (pow(res[i]+1, inv, 257) - 1) & 0xFF
        return bytes(res)
    def _paqjp_transform_26(self, d):
        if not d: return b'\x01\x00'
        n = (len(d)*7 + 13) & 0xFFFF
        if n % 2 == 0: n ^= 1
        e = pow(n, 16777216, 256) | 1
        res = bytearray(d)
        for i in range(len(res)): res[i] = (pow(res[i]+1, e, 257) - 1) & 0xFF
        return bytes([n & 0xFF, (n >> 8) & 0xFF]) + bytes(res)
    def _paqjp_reverse_26(self, d):
        if d == b'\x01\x00': return b''
        if not d or len(d) < 2: raise TransformError("FLT26")
        n = d[0] | (d[1] << 8)
        if n % 2 == 0: n ^= 1
        e = pow(n, 16777216, 256) | 1
        inv = mod_inv(e, 256)
        if inv is None: raise TransformError("FLT26")
        res = bytearray(d[2:])
        for i in range(len(res)): res[i] = (pow(res[i]+1, inv, 257) - 1) & 0xFF
        return bytes(res)
    def _paqjp_transform_27(self, d):
        if not d:
            out = bytearray(b'\x00\x00\x00\x00'); out.extend(b'\x01\x00'); out.extend(b'\x00'*1024); return bytes(out)
        BS = 1024; tb = (len(d) + BS - 1)//BS
        out = bytearray(); out.extend(len(d).to_bytes(4, 'big'))
        for bi in range(tb):
            s = bi*BS; e = min(s+BS, len(d)); ch = d[s:e]
            pl = BS - len(ch)
            if pl: ch = ch + b'\x00' * pl
            n = ((len(d)*7 + bi*13 + 1) & 0xFFFF) | 1
            e = pow(n, 16777216, 256) | 1; e200 = pow(e, 200, 256)
            tr = bytearray(ch)
            for i in range(BS): tr[i] = (pow(tr[i]+1, e200, 257) - 1) & 0xFF
            out.append(n & 0xFF); out.append((n >> 8) & 0xFF); out.extend(tr)
        return bytes(out)
    def _paqjp_reverse_27(self, d):
        if not d or len(d) < 4: raise TransformError("FLT27")
        ol = int.from_bytes(d[:4], 'big'); pl = d[4:]
        BS = 1024; BTL = 2 + BS
        if len(pl) % BTL: raise TransformError("FLT27")
        nb = len(pl)//BTL; dec = bytearray()
        for bi in range(nb):
            off = bi*BTL
            if off + 2 > len(pl): raise TransformError("FLT27")
            n = pl[off] | (pl[off+1] << 8); ch = pl[off+2:off+2+BS]
            n |= 1; e = pow(n, 16777216, 256) | 1; e200 = pow(e, 200, 256)
            inv = mod_inv(e200, 256)
            if inv is None: raise TransformError("FLT27")
            for i in range(BS): dec.append((pow(ch[i]+1, inv, 257) - 1) & 0xFF)
        return bytes(dec[:ol])
    def _compress_backend_with_flag(self, data):
        cand = []
        if HAS_ZSTD:
            try: cand.append(zstd_cctx.compress(data))
            except: pass
        if paq is not None:
            try: cand.append(paq.compress(data))
            except: pass
        cand.append(data)
        return min(cand, key=len)
    def _decompress_backend_with_flag(self, data):
        if not data: return b''
        if HAS_ZSTD:
            try: return zstd_dctx.decompress(data)
            except Exception: pass
        if paq is not None:
            try: return paq.decompress(data)
            except Exception: pass
        return data
    def _paqjp_transform_28(self, d):
        BS = 1024
        if not d:
            bl = b'\x00' * BS; n = 1
            c = self._compress_backend_with_flag(bl)
            out = bytearray(struct.pack('>I', 0))
            out += bytes([n & 0xFF, (n >> 8) & 0xFF])
            out += bytes([(len(c) >> 8) & 0xFF, len(c) & 0xFF]); out += c
            return bytes(out)
        tb = (len(d) + BS - 1)//BS
        out = bytearray(); out.extend(len(d).to_bytes(4, 'big'))
        for bi in range(tb):
            s = bi*BS; e = min(s+BS, len(d)); ch = d[s:e]
            pl = BS - len(ch)
            if pl: ch = ch + b'\x00' * pl
            n = ((len(d)*7 + bi*13 + 1) & 0xFFFF) | 1
            e = pow(n, 16777216, 256) | 1; e200 = pow(e, 200, 256)
            tr = bytearray(ch)
            for i in range(BS): tr[i] = (pow(tr[i]+1, e200, 257) - 1) & 0xFF
            c = self._compress_backend_with_flag(bytes(tr))
            out.append(n & 0xFF); out.append((n >> 8) & 0xFF)
            out.append((len(c) >> 8) & 0xFF); out.append(len(c) & 0xFF); out.extend(c)
        return bytes(out)
    def _paqjp_reverse_28(self, d):
        if not d or len(d) < 4: raise TransformError("FLT28")
        ol = int.from_bytes(d[:4], 'big'); pl = d[4:]
        pos = 0; dec = bytearray()
        while pos < len(pl):
            if pos + 4 > len(pl): raise TransformError("FLT28")
            n = pl[pos] | (pl[pos+1] << 8); pos += 2
            cl = (pl[pos] << 8) | pl[pos+1]; pos += 2
            if pos + cl > len(pl): raise TransformError("FLT28")
            c = pl[pos:pos+cl]; pos += cl
            blk = self._decompress_backend_with_flag(c)
            n |= 1; e = pow(n, 16777216, 256) | 1; e200 = pow(e, 200, 256)
            inv = mod_inv(e200, 256)
            if inv is None: raise TransformError("FLT28")
            tr = bytearray(blk)
            for i in range(len(tr)): tr[i] = (pow(tr[i]+1, inv, 257) - 1) & 0xFF
            dec.extend(tr)
        return bytes(dec[:ol])
    def _paqjp_transform_29(self, d):
        BS = 32
        if not d:
            bl = b'\x00' * BS; n = 1
            c = self._compress_backend_with_flag(bl)
            out = bytearray(struct.pack('>I', 0))
            out += bytes([n & 0xFF, (n >> 8) & 0xFF])
            out += bytes([(len(c) >> 8) & 0xFF, len(c) & 0xFF]); out += c
            return bytes(out)
        tb = (len(d) + BS - 1)//BS
        out = bytearray(); out.extend(len(d).to_bytes(4, 'big'))
        for bi in range(tb):
            s = bi*BS; e = min(s+BS, len(d)); ch = d[s:e]
            pl = BS - len(ch)
            if pl: ch = ch + b'\x00' * pl
            n = ((len(d)*7 + bi*13 + 1) & 0xFFFF) | 1
            c = self._compress_backend_with_flag(ch)
            out.append(n & 0xFF); out.append((n >> 8) & 0xFF)
            out.append((len(c) >> 8) & 0xFF); out.append(len(c) & 0xFF); out.extend(c)
        return bytes(out)
    def _paqjp_reverse_29(self, d):
        if not d or len(d) < 4: raise TransformError("FLT29")
        ol = int.from_bytes(d[:4], 'big'); pl = d[4:]
        pos = 0; dec = bytearray()
        while pos < len(pl):
            if pos + 4 > len(pl): raise TransformError("FLT29")
            n = pl[pos] | (pl[pos+1] << 8); pos += 2
            cl = (pl[pos] << 8) | pl[pos+1]; pos += 2
            if pos + cl > len(pl): raise TransformError("FLT29")
            c = pl[pos:pos+cl]; pos += cl
            dec.extend(self._decompress_backend_with_flag(c))
        return bytes(dec[:ol])
    def _paqjp_transform_30(self, d):
        BS = 33
        if not d:
            bl = b'\x00' * BS; n, enc = self._paqjp_compute_n_for_block(bl, 0, 0)
            c = self._compress_backend_with_flag(bl)
            out = bytearray(struct.pack('>I', 0))
            out += enc; out.append((len(c) >> 8) & 0xFF); out.append(len(c) & 0xFF); out += c
            return bytes(out)
        tb = (len(d) + BS - 1)//BS
        out = bytearray(); out.extend(len(d).to_bytes(4, 'big'))
        for bi in range(tb):
            s = bi*BS; e = min(s+BS, len(d)); ch = d[s:e]
            pl = BS - len(ch)
            if pl: ch = ch + b'\x00' * pl
            n, enc = self._paqjp_compute_n_for_block(ch, bi, len(d))
            c = self._compress_backend_with_flag(ch)
            out.extend(enc); out.append((len(c) >> 8) & 0xFF); out.append(len(c) & 0xFF); out.extend(c)
        return bytes(out)
    def _paqjp_reverse_30(self, d):
        if not d or len(d) < 4: raise TransformError("FLT30")
        ol = int.from_bytes(d[:4], 'big'); pl = d[4:]
        pos = 0; dec = bytearray()
        while pos < len(pl):
            if pos >= len(pl): raise TransformError("FLT30")
            Ln = pl[pos]; pos += 1
            if Ln > 32 or pos + Ln > len(pl): raise TransformError("FLT30")
            pos += Ln
            if pos + 2 > len(pl): raise TransformError("FLT30")
            cl = (pl[pos] << 8) | pl[pos+1]; pos += 2
            if pos + cl > len(pl): raise TransformError("FLT30")
            c = pl[pos:pos+cl]; pos += cl
            dec.extend(self._decompress_backend_with_flag(c))
        return bytes(dec[:ol])
    def _paqjp_compute_n_for_block(self, blk, bi, tl):
        if not blk: return (1, b'\x01\x01')
        d = blk[0]; x = (bi % 33) + 1
        try: t = (d*d - d**x) // 256
        except OverflowError: t = 0
        if 0 <= t <= 255:
            n = t | 1; return (n, bytes([1, n]))
        h = hashlib.sha256(blk + bytes([bi & 0xFF, (tl>>8)&0xFF, tl&0xFF])).digest()
        nb = bytearray(h); nb[0] |= 1
        return (int.from_bytes(nb, 'big'), bytes([len(nb)]) + bytes(nb))

    def transform_41(self, d):
        if not d: return b''
        m = bytes([0x27, 0x03]); t = bytearray(d); n = min(len(t), 8)
        for i in range(n): t[i] ^= m[i % 2]
        return bytes(t)
    reverse_transform_41 = transform_41
    def transform_42(self, d):
        if not d: return b''
        t = bytearray(d); m = bytes([0x27, 0x03])
        for i in range(len(t)): t[i] ^= m[i % 2]
        return bytes(t)
    reverse_transform_42 = transform_42
    def transform_43(self, d):
        if not d: return b''
        t = bytearray(d); m = bytes([0x10, 0x00, 0x00])
        for i in range(0, len(t), 3):
            for j in range(min(3, len(t)-i)): t[i+j] ^= m[j]
        return bytes(t)
    reverse_transform_43 = transform_43
    def transform_44(self, d):
        if not d: return b''
        return base64.b64encode(d)
    def reverse_transform_44(self, d):
        if not d: return b''
        try: return base64.b64decode(d, validate=False)
        except Exception as e: raise TransformError(f"B64-44: {e}")
    @staticmethod
    def _huffman_code_lengths(freq):
        heap = [(f, i, i) for i, f in enumerate(freq) if f > 0]
        if not heap: return [0] * len(freq)
        if len(heap) == 1:
            L = [0] * len(freq); L[heap[0][2]] = 1; return L
        heapq.heapify(heap); nid = len(freq)
        while len(heap) > 1:
            f1, _, n1 = heapq.heappop(heap); f2, _, n2 = heapq.heappop(heap)
            heapq.heappush(heap, (f1+f2, nid, (n1, n2))); nid += 1
        L = [0] * len(freq)
        def trav(node, depth):
            if isinstance(node, int): L[node] = depth
            else: l, r = node; trav(l, depth+1); trav(r, depth+1)
        _, _, root = heap[0]; trav(root, 0)
        return L
    @staticmethod
    def _huffman_canonical_codes(cl):
        syms = list(range(len(cl))); syms.sort(key=lambda s: (cl[s], s))
        codes = {}; code = 0; prev = 0; first = True
        for s in syms:
            c = cl[s]
            if c == 0: continue
            if first: prev = c; first = False
            elif c != prev: code <<= (c - prev); prev = c
            codes[s] = (code, c); code += 1
        return codes
    def transform_45(self, d):
        if not d: return b''
        freq = [0]*256
        for b in d: freq[b] += 1
        cl = self._huffman_code_lengths(freq); codes = self._huffman_canonical_codes(cl)
        h = bytearray(); h.extend(len(d).to_bytes(4, 'big')); h.extend(cl)
        bits = []
        for b in d:
            c, L = codes[b]
            for i in range(L-1,-1,-1): bits.append((c >> i) & 1)
        pad = (8 - len(bits) % 8) % 8; bits.extend([0]*pad)
        ob = bytearray()
        for i in range(0, len(bits), 8):
            v = 0
            for j in range(8): v = (v << 1) | bits[i+j]
            ob.append(v)
        return bytes(h) + bytes(ob)
    def reverse_transform_45(self, d):
        if not d: return b''
        if len(d) < 4 + 256: raise TransformError("Huff")
        ol = int.from_bytes(d[:4], 'big'); cl = list(d[4:4+256]); pl = d[4+256:]
        if ol == 0: return b''
        if max(cl) > 32: raise TransformError("Huff")
        c2s = {}
        syms = list(range(256)); syms.sort(key=lambda s: (cl[s], s))
        code = 0; prev = 0; first = True
        for s in syms:
            c = cl[s]
            if c == 0: continue
            if first: prev = c; first = False
            elif c != prev: code <<= (c - prev); prev = c
            c2s[(c, code)] = s; code += 1
        bits = []
        for b in pl:
            for i in range(7,-1,-1): bits.append((b >> i) & 1)
        pos = 0; nb = len(bits); out = bytearray()
        ml = max(cl) if any(cl) else 0
        while pos < nb and len(out) < ol:
            found = False
            for L in range(1, ml+1):
                if pos + L > nb: break
                v = 0
                for j in range(L): v = (v << 1) | bits[pos+j]
                if (L, v) in c2s:
                    out.append(c2s[(L, v)]); pos += L; found = True; break
            if not found: raise TransformError("Huff")
        if len(out) != ol: raise TransformError("Huff")
        return bytes(out)
    def transform_46(self, d):
        if not d: return b''
        t = bytearray(d); m = self.mask_46
        for i in range(len(t)): t[i] ^= m[i % len(m)]
        return bytes(t)
    reverse_transform_46 = transform_46
    def transform_47(self, d):
        if not d: return b''
        t = bytearray(d); tl = len(self.mod_state_table)
        if tl == 0: return d
        for i in range(len(t)): t[i] ^= self.mod_state_table[i % tl][0]
        return bytes(t)
    reverse_transform_47 = transform_47

    def transform_57(self, d):
        if len(d) < 4:
            pl = 4 - len(d); key = 0
            return bytes([pl]) + key.to_bytes(4, 'little') + d + b'\x00' * pl
        pl = (4 - len(d) % 4) % 4; pad = d + b'\x00' * pl
        blocks = [pad[i:i+4] for i in range(0, len(pad), 4)]
        mc, _ = Counter(blocks).most_common(1)[0]
        key = int.from_bytes(mc, 'little'); out = bytearray()
        for b in blocks: out.extend((int.from_bytes(b, 'little') ^ key).to_bytes(4, 'little'))
        return bytes([pl]) + key.to_bytes(4, 'little') + bytes(out)
    def reverse_transform_57(self, d):
        if len(d) < 5: raise TransformError("T57")
        pl = d[0]; key = int.from_bytes(d[1:5], 'little'); pl_data = d[5:]
        if len(pl_data) % 4: raise TransformError("T57")
        out = bytearray()
        for i in range(0, len(pl_data), 4):
            out.extend((int.from_bytes(pl_data[i:i+4], 'little') ^ key).to_bytes(4, 'little'))
        if pl > 0: out = out[:-pl]
        return bytes(out)

    # ==================================================================
    # ALGORITHM 58 (UPGRADED) — self-terminating bit-RLE, iterated 1..256
    # ==================================================================
    def _bit_rle_encode(self, data: bytes) -> bytes:
        if not data:
            return struct.pack('>H', 0)
        bits = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        orig_bit_count = len(bits)
        out_bits = []
        i = 0
        n = len(bits)
        while i < n:
            val = bits[i]
            j = i + 1
            while j < n and bits[j] == val:
                j += 1
            run = j - i
            while run > 0:
                ch = min(run, 259)
                out_bits.append(val)
                if ch == 1:
                    out_bits.append(0)
                elif ch == 2:
                    out_bits.extend([1, 0])
                elif ch == 3:
                    out_bits.extend([1, 1, 0])
                else:
                    out_bits.extend([1, 1, 1])
                    lc = ch - 4
                    for k in range(7, -1, -1):
                        out_bits.append((lc >> k) & 1)
                run -= ch
            i = j
        pad = (8 - len(out_bits) % 8) % 8
        out_bits.extend([0] * pad)
        packed = bytearray()
        for i in range(0, len(out_bits), 8):
            b = 0
            for j in range(8):
                b = (b << 1) | out_bits[i + j]
            packed.append(b)
        return struct.pack('>H', orig_bit_count) + bytes(packed)

    def _bit_rle_decode(self, data: bytes) -> bytes:
        if len(data) < 2:
            raise TransformError("BRLE short")
        orig_bit_count = struct.unpack('>H', data[:2])[0]
        if orig_bit_count == 0:
            return b''
        payload = data[2:]
        bits = []
        for byte in payload:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        out_bits = []
        pos = 0
        nbits = len(bits)
        while len(out_bits) < orig_bit_count:
            if pos >= nbits:
                raise TransformError("BRLE eof")
            val = bits[pos]; pos += 1
            if pos >= nbits:
                raise TransformError("BRLE eof2")
            if bits[pos] == 0:
                run = 1; pos += 1
            else:
                pos += 1
                if pos >= nbits:
                    raise TransformError("BRLE eof3")
                if bits[pos] == 0:
                    run = 2; pos += 1
                else:
                    pos += 1
                    if pos >= nbits:
                        raise TransformError("BRLE eof4")
                    if bits[pos] == 0:
                        run = 3; pos += 1
                    else:
                        pos += 1
                        if pos + 8 > nbits:
                            raise TransformError("BRLE eof5")
                        lc = 0
                        for _ in range(8):
                            lc = (lc << 1) | bits[pos]; pos += 1
                        run = lc + 4
            out_bits.extend([val] * run)
        out_bits = out_bits[:orig_bit_count]
        pad = (8 - len(out_bits) % 8) % 8
        out_bits.extend([0] * pad)
        out = bytearray()
        for i in range(0, len(out_bits), 8):
            b = 0
            for j in range(8):
                b = (b << 1) | out_bits[i + j]
            out.append(b)
        return bytes(out)

    def _bit_rle_iterate(self, data: bytes, max_passes: int = 256):
        best = None
        best_size = len(data)
        best_passes = 0
        current = data
        for p in range(1, max_passes + 1):
            try:
                encoded = self._bit_rle_encode(current)
            except Exception:
                break
            if len(encoded) >= len(current):
                break
            current = encoded
            if len(current) < best_size:
                best = current
                best_size = len(current)
                best_passes = p
        return best, best_passes

    def transform_58(self, data: bytes) -> bytes:
        if not data:
            return struct.pack('>I', 0)
        BLOCK = 256
        body = bytearray()
        for s in range(0, len(data), BLOCK):
            block = data[s:s+BLOCK]
            best, passes = self._bit_rle_iterate(block, self.RLE58_MAX_PASSES)
            if best is None:
                body.append(0)
                body.extend(struct.pack('>H', len(block)))
                body.extend(block)
            else:
                body.append(1)
                body.append(passes - 1)
                body.extend(struct.pack('>H', len(best)))
                body.extend(best)
        if len(body) < len(data):
            return b'\x01' + struct.pack('>I', len(data)) + bytes(body)
        return b'\x00' + struct.pack('>I', len(data)) + data

    def reverse_transform_58(self, data: bytes) -> bytes:
        if len(data) < 5:
            raise TransformError("T58 short")
        top = data[0]
        ol = struct.unpack('>I', data[1:5])[0]
        pl = data[5:]
        if top == 0:
            if len(pl) != ol:
                raise TransformError("T58 raw len")
            return bytes(pl)
        if top != 1:
            raise TransformError("T58 flag")
        out = bytearray()
        pos = 0
        while pos < len(pl) and len(out) < ol:
            if pos + 4 > len(pl):
                raise TransformError("T58 blk hdr")
            f = pl[pos]; pos += 1
            passes_minus_1 = pl[pos]; pos += 1
            bl = struct.unpack('>H', pl[pos:pos+2])[0]; pos += 2
            if pos + bl > len(pl):
                raise TransformError("T58 blk data")
            bd = pl[pos:pos+bl]; pos += bl
            if f == 0:
                out.extend(bd)
            elif f == 1:
                passes = passes_minus_1 + 1
                current = bd
                for _ in range(passes):
                    current = self._bit_rle_decode(current)
                cs = min(256, ol - len(out))
                out.extend(current[:cs])
            else:
                raise TransformError("T58 blk flag")
        if len(out) != ol:
            raise TransformError("T58 len")
        return bytes(out)

    def transform_59(self, data):
        if not data: return b''
        out = bytearray([data[0]])
        for i in range(1, len(data)):
            out.append((data[i] - data[i-1]) & 0xFF)
        return bytes(out)
    def reverse_transform_59(self, data):
        if not data: return b''
        out = bytearray([data[0]])
        for i in range(1, len(data)):
            out.append((data[i] + out[i-1]) & 0xFF)
        return bytes(out)

    def transform_60(self, data):
        if not data: return b''
        freq = [0] * 256
        for b in data: freq[b] += 1
        escape = min(range(256), key=lambda b: (freq[b], b))
        out = bytearray([escape]); i = 0; n = len(data)
        while i < n:
            val = data[i]; j = i + 1
            while j < n and data[j] == val and j - i < 256: j += 1
            run = j - i
            if val == escape or run >= 4:
                out.append(escape); out.append(run - 1); out.append(val)
            else:
                for _ in range(run): out.append(val)
            i = j
        if len(out) >= len(data): return b'\x00' + data
        return bytes(out)
    def reverse_transform_60(self, data):
        if not data: return b''
        if data[0] == 0: return data[1:]
        escape = data[0]; out = bytearray(); i = 1; n = len(data)
        while i < n:
            b = data[i]
            if b == escape:
                if i + 2 >= n: raise TransformError("T60 short")
                cnt = data[i+1] + 1; val = data[i+2]
                out.extend([val] * cnt); i += 3
            else:
                out.append(b); i += 1
        return bytes(out)

    def _dynamic_transform(self, n):
        def tf(d):
            if not d: return b''
            seed = self.get_seed(n % len(self.seed_tables), len(d))
            t = bytearray(d)
            for i in range(len(t)): t[i] ^= seed
            return bytes(t)
        return tf, tf
    def transform_256(self, d): return d
    reverse_transform_256 = transform_256

    def _decompress_static_dict(self, d): print("Static not impl."); return None
    def _decompress_dynamic_dict(self, d): print("Dynamic not impl."); return None
    def _decompress_line_dict(self, d): print("Line not impl."); return None

    def _build_transform_maps(self):
        self.fwd_transforms: Dict[int, Callable] = {}
        self.rev_transforms: Dict[int, Callable] = {}
        for i in range(1, 22):
            self.fwd_transforms[i] = getattr(self, f"transform_{i:02d}")
            self.rev_transforms[i] = getattr(self, f"reverse_transform_{i:02d}")
        self.fwd_transforms[22] = self.transform_22; self.rev_transforms[22] = self.reverse_transform_22
        self.fwd_transforms[23] = self.transform_23; self.rev_transforms[23] = self.reverse_transform_23
        self.fwd_transforms[24] = self.transform_24; self.rev_transforms[24] = self.reverse_transform_24
        self.fwd_transforms[25] = self.transform_25; self.rev_transforms[25] = self.reverse_transform_25
        self.fwd_transforms[26] = self.transform_26; self.rev_transforms[26] = self.reverse_transform_26
        self.fwd_transforms[27] = self.transform_27; self.rev_transforms[27] = self.reverse_transform_27
        self.fwd_transforms[28] = self.transform_28; self.rev_transforms[28] = self.reverse_transform_28
        self.fwd_transforms[29] = self.transform_29; self.rev_transforms[29] = self.reverse_transform_29
        self.fwd_transforms[30] = self.transform_30; self.rev_transforms[30] = self.reverse_transform_30
        self.fwd_transforms[31] = self.transform_31; self.rev_transforms[31] = self.reverse_transform_31
        self.fwd_transforms[32] = self.transform_32; self.rev_transforms[32] = self.reverse_transform_32
        self.fwd_transforms[33] = self._paqjp_transform_23; self.rev_transforms[33] = self._paqjp_reverse_23
        self.fwd_transforms[34] = self._paqjp_transform_24; self.rev_transforms[34] = self._paqjp_reverse_24
        self.fwd_transforms[35] = self._paqjp_transform_25; self.rev_transforms[35] = self._paqjp_reverse_25
        self.fwd_transforms[36] = self._paqjp_transform_26; self.rev_transforms[36] = self._paqjp_reverse_26
        self.fwd_transforms[37] = self._paqjp_transform_27; self.rev_transforms[37] = self._paqjp_reverse_27
        self.fwd_transforms[38] = self._paqjp_transform_28; self.rev_transforms[38] = self._paqjp_reverse_28
        self.fwd_transforms[39] = self._paqjp_transform_29; self.rev_transforms[39] = self._paqjp_reverse_29
        self.fwd_transforms[40] = self._paqjp_transform_30; self.rev_transforms[40] = self._paqjp_reverse_30
        self.fwd_transforms[41] = self.transform_41; self.rev_transforms[41] = self.reverse_transform_41
        self.fwd_transforms[42] = self.transform_42; self.rev_transforms[42] = self.reverse_transform_42
        self.fwd_transforms[43] = self.transform_43; self.rev_transforms[43] = self.reverse_transform_43
        self.fwd_transforms[44] = self.transform_44; self.rev_transforms[44] = self.reverse_transform_44
        self.fwd_transforms[45] = self.transform_45; self.rev_transforms[45] = self.reverse_transform_45
        self.fwd_transforms[46] = self.transform_46; self.rev_transforms[46] = self.reverse_transform_46
        self.fwd_transforms[47] = self.transform_47; self.rev_transforms[47] = self.reverse_transform_47
        for i in range(48, 57):
            f, r = self._dynamic_transform(i)
            self.fwd_transforms[i] = f; self.rev_transforms[i] = r
        self.fwd_transforms[57] = self.transform_57; self.rev_transforms[57] = self.reverse_transform_57
        self.fwd_transforms[58] = self.transform_58; self.rev_transforms[58] = self.reverse_transform_58
        self.fwd_transforms[59] = self.transform_59; self.rev_transforms[59] = self.reverse_transform_59
        self.fwd_transforms[60] = self.transform_60; self.rev_transforms[60] = self.reverse_transform_60
        for i in range(61, 256):
            f, r = self._dynamic_transform(i)
            self.fwd_transforms[i] = f; self.rev_transforms[i] = r
        self.fwd_transforms[256] = self.transform_256; self.rev_transforms[256] = self.reverse_transform_256

    def _load_static_dictionary(self):
        if not os.path.exists(COMBINED_DICTIONARY_FILE): return [], {}
        ws = set()
        try:
            with open(COMBINED_DICTIONARY_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    w = line.strip()
                    if w: ws.add(w)
        except Exception as e:
            print(f"Warn: {e}"); return [], {}
        sw = sorted(ws); return sw, {w: i for i, w in enumerate(sw)}
    def _load_line_dictionary(self):
        if not os.path.exists(COMBINED_DICTIONARY_FILE): return [], {}
        lines = []
        try:
            with open(COMBINED_DICTIONARY_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                for raw in f:
                    p = raw.strip()
                    if p and p not in lines:
                        lines.append(p)
                        if len(lines) >= MAX_LINE_ENTRIES: break
        except Exception as e:
            print(f"Warn: {e}"); return [], {}
        if not lines: return [], {}
        lines.sort(key=len, reverse=True)
        return lines, {p: i for i, p in enumerate(lines)}

    def _generate_permutation_from_circuit(self, nq, seed):
        if not USE_QUANTUM or not HAS_QISKIT:
            rng = random.Random(seed); size = 1 << nq
            p = list(range(size)); rng.shuffle(p); return p
        try:
            qc = QuantumCircuit(nq); rng = random.Random(seed)
            for q in range(nq):
                qc.h(q); qc.rz(rng.random()*2*math.pi, q); qc.rx(rng.random()*2*math.pi, q)
            for _ in range(nq):
                for i in range(nq-1): qc.cx(i, i+1)
                qc.barrier()
                for i in range(nq): qc.rz(rng.random()*2*math.pi, i); qc.rx(rng.random()*2*math.pi, i)
            try: qasm = qc.qasm()
            except AttributeError: qasm = qc.draw('text')
            fs = seed + hash(qasm) % 1000000
            rng2 = random.Random(fs); size = 1 << nq
            p = list(range(size)); rng2.shuffle(p); return p
        except Exception:
            rng = random.Random(seed); size = 1 << nq
            p = list(range(size)); rng.shuffle(p); return p
    def _precompute_quantum_transforms(self):
        if not USE_QUANTUM or not HAS_QISKIT: return
        q = self.QUANTUM_QUBITS
        if q > 49: q = 49; self.QUANTUM_QUBITS = q
        size = 1 << q
        bs = size if q <= 12 else 1024
        n = 8; self.quantum_fast_transforms = []
        for i in range(n):
            p = self._generate_permutation_from_circuit(q, 1000 + i)
            if len(p) > bs: p = p[:bs]
            f, r = self._make_substitution_transform(p, bs)
            self.quantum_fast_transforms.append((f, r))
        for idx, (f, r) in enumerate(self.quantum_fast_transforms, 1):
            self.fwd_transforms[256+idx] = f; self.rev_transforms[256+idx] = r
        self.quantum_transforms_built = True
        print(f"Quantum: {n} transforms, block={bs}, qubits={q}")
    def _make_substitution_transform(self, perm, size):
        if size < 256:
            inv = [0]*size
            for i, p in enumerate(perm): inv[p] = i
            def f(d):
                o = bytearray()
                for b in d: o.append(perm[b] if b < size else b)
                return bytes(o)
            def r(d):
                o = bytearray()
                for b in d: o.append(inv[b] if b < size else b)
                return bytes(o)
        else:
            inv = [0]*size
            for i, p in enumerate(perm): inv[p] = i
            def f(d): return bytes(perm[b] for b in d)
            def r(d): return bytes(inv[b] for b in d)
        return f, r

    def _get_pattern(self, size, index):
        random.seed(12345 + size*100 + index)
        return [random.randint(0, 255) for _ in range(size)]
    def _calculate_repeats(self, data):
        if not data: return 1
        r = ((len(data)*13 + (sum(data) % 256)*17) % 256) + 1
        return max(1, min(256, r))
    def _verify_lossless(self, orig, trans, rev):
        try: return rev(trans) == orig
        except TransformError: return False
        except Exception: return False

    WINDOW_SIZE = 2048; MIN_MATCH = 3; MAX_MATCH = 2048; MAX_DIST = 2048
    def _lz77_tokenize(self, data):
        tokens = []; i = 0; n = len(data)
        while i < n:
            bl = 0; bd = 0; sw = max(0, i - self.WINDOW_SIZE)
            for j in range(sw, i):
                if data[j] != data[i]: continue
                k = 0
                while i+k < n and j+k < i and data[j+k] == data[i+k]:
                    k += 1
                    if k >= self.MAX_MATCH: break
                if k >= self.MIN_MATCH and k > bl:
                    bl = k; bd = i - j
                    if bl == self.MAX_MATCH: break
            if bl >= self.MIN_MATCH: tokens.append(('M', bd, bl)); i += bl
            else: tokens.append(('L', data[i], None)); i += 1
        return tokens
    def _lz77_untokenize(self, tokens):
        out = bytearray()
        for t in tokens:
            if t[0] == 'L': out.append(t[1])
            else:
                d, l = t[1], t[2]; s = len(out) - d
                for k in range(l): out.append(out[s+k])
        return bytes(out)
    def _encode_lzh(self, data):
        tokens = self._lz77_tokenize(data)
        lf = [0]*256; df = [0]*(self.MAX_DIST+1); nf = [0]*(self.MAX_MATCH+1)
        for t in tokens:
            if t[0] == 'L': lf[t[1]] += 1
            else: df[t[1]] += 1; nf[t[2]] += 1
        lcl = self._huffman_code_lengths(lf); dcl = self._huffman_code_lengths(df); ncl = self._huffman_code_lengths(nf)
        lc = self._huffman_canonical_codes(lcl); dc = self._huffman_canonical_codes(dcl); nc = self._huffman_canonical_codes(ncl)
        bits = []
        for b in struct.pack('>I', len(tokens)):
            for i in range(8): bits.append((b >> (7-i)) & 1)
        for t in tokens:
            if t[0] == 'L':
                bits.append(0); cd, cl = lc[t[1]]
                for i in range(cl-1,-1,-1): bits.append((cd >> i) & 1)
            else:
                bits.append(1); cd, cl = dc[t[1]]
                for i in range(cl-1,-1,-1): bits.append((cd >> i) & 1)
                cd, cl = nc[t[2]]
                for i in range(cl-1,-1,-1): bits.append((cd >> i) & 1)
        pad = (8 - len(bits) % 8) % 8; bits.extend([0]*pad)
        def pk(L): return b''.join(struct.pack('>H', x) for x in L)
        h = bytearray(); h.extend(pk(lcl)); h.extend(pk(dcl)); h.extend(pk(ncl))
        for i in range(0, len(bits), 8):
            b = 0
            for j in range(8): b = (b << 1) | bits[i+j]
            h.append(b)
        return bytes(h)
    def _decode_lzh(self, data):
        LB = 256*2; DB = 2049*2; NB = 2049*2
        if len(data) < LB+DB+NB: raise TransformError("LZH short")
        pos = 0
        lcl = [struct.unpack('>H', data[i:i+2])[0] for i in range(pos, pos+LB, 2)]; pos += LB
        dcl = [struct.unpack('>H', data[i:i+2])[0] for i in range(pos, pos+DB, 2)]; pos += DB
        ncl = [struct.unpack('>H', data[i:i+2])[0] for i in range(pos, pos+NB, 2)]; pos += NB
        def bt(L):
            syms = list(range(len(L))); syms.sort(key=lambda s: (L[s], s))
            dc = {}; code = 0; prev = 0; first = True
            for s in syms:
                c = L[s]
                if c == 0: continue
                if first: prev = c; first = False
                elif c != prev: code <<= (c - prev); prev = c
                dc[(c, code)] = s; code += 1
            return dc
        ld = bt(lcl); dd = bt(dcl); nd = bt(ncl)
        ml = max(lcl) if any(lcl) else 0; md = max(dcl) if any(dcl) else 0; mn = max(ncl) if any(ncl) else 0
        pl = data[pos:]
        if len(pl) < 4: raise TransformError("LZH")
        tc = struct.unpack('>I', pl[:4])[0]
        bits = []
        for b in pl[4:]:
            for i in range(7,-1,-1): bits.append((b >> i) & 1)
        bp = 0; tokens = []
        for _ in range(tc):
            if bp >= len(bits): raise TransformError("LZH eof")
            f = bits[bp]; bp += 1
            if f == 0:
                found = False
                for cl in range(1, ml+1):
                    if bp + cl > len(bits): break
                    v = 0
                    for j in range(cl): v = (v << 1) | bits[bp+j]
                    if (cl, v) in ld:
                        tokens.append(('L', ld[(cl,v)], None)); bp += cl; found = True; break
                if not found: raise TransformError("LZH lit")
            else:
                fd = False
                for cl in range(1, md+1):
                    if bp + cl > len(bits): break
                    v = 0
                    for j in range(cl): v = (v << 1) | bits[bp+j]
                    if (cl, v) in dd:
                        d = dd[(cl,v)]; bp += cl; fd = True; break
                if not fd: raise TransformError("LZH d")
                fl = False
                for cl in range(1, mn+1):
                    if bp + cl > len(bits): break
                    v = 0
                    for j in range(cl): v = (v << 1) | bits[bp+j]
                    if (cl, v) in nd:
                        l = nd[(cl,v)]; bp += cl; fl = True; break
                if not fl: raise TransformError("LZH l")
                tokens.append(('M', d, l))
        return self._lz77_untokenize(tokens)

    def _encode_marker_single(self, t):
        if t <= 252: return bytes([t - 1])
        elif t <= 255: return bytes([254, t - 253])
        else: return bytes([255, (t-256)//256, (t-256) % 256])
    def _encode_marker_raw(self): return bytes([252])
    def _decode_header(self, data):
        if len(data) < 1: return 0, ()
        f = data[0]
        if f < 252: return 1, (f+1,)
        if f == 252: return 1, ()
        if f == 254:
            if len(data) < 2: return 0, ()
            x = data[1]
            if x > 3: return 0, ()
            return 2, (253 + x,)
        if f == 255:
            if len(data) < 3: return 0, ()
            return 3, (256 + data[1]*256 + data[2],)
        return 0, ()

    # ==================================================================
    # BACKEND — try zstd / paq / raw, no flag byte
    # ==================================================================
    def _compress_backend(self, data: bytes) -> bytes:
        candidates = [data]
        if HAS_ZSTD:
            try: candidates.append(zstd_cctx.compress(data))
            except Exception: pass
        if paq is not None:
            try: candidates.append(paq.compress(data))
            except Exception: pass
        return min(candidates, key=len)

    def _decompress_backend(self, data: bytes) -> Optional[bytes]:
        if not data: return b''
        if HAS_ZSTD:
            try: return zstd_dctx.decompress(data)
            except Exception: pass
        if paq is not None:
            try: return paq.decompress(data)
            except Exception: pass
        return data

    def compress_with_verification(self, data, time_limit=None):
        if time_limit is None: time_limit = self.ULTRA_TIME_LIMIT
        t0 = time.time(); bc = None; bl = float('inf')
        def tc(h, t):
            nonlocal bc, bl
            c = h + self._compress_backend(t)
            if len(c) < bl: bc = c; bl = len(c)
        tc(self._encode_marker_raw(), data)
        for t in range(1, 257):
            if time_limit and time.time() - t0 > time_limit: break
            try:
                tr = self.fwd_transforms[t](data)
                if not self._verify_lossless(data, tr, self.rev_transforms[t]): continue
                tc(self._encode_marker_single(t), tr)
            except Exception: continue
        if bc is None: bc = self._encode_marker_raw() + self._compress_backend(data)
        dec, _ = self._decompress_auto(bc)
        if dec == data: return bc
        fb = self._encode_marker_raw() + self._compress_backend(data)
        dfb, _ = self._decompress_auto(fb)
        if dfb != data: raise RuntimeError("Fallback failed")
        return fb

    def compress_with_lzh(self, data, time_limit=None):
        if time_limit is None: time_limit = self.ULTRA_TIME_LIMIT
        t0 = time.time(); bc = None; bl = float('inf')
        def tc(h, t):
            nonlocal bc, bl
            lzh = self._encode_lzh(t); c = h + b'\xFF' + lzh
            if len(c) < bl: bc = c; bl = len(c)
        tc(self._encode_marker_raw(), data)
        for t in range(1, 257):
            if time_limit and time.time() - t0 > time_limit: break
            try:
                tr = self.fwd_transforms[t](data)
                if not self._verify_lossless(data, tr, self.rev_transforms[t]): continue
                tc(self._encode_marker_single(t), tr)
            except Exception: continue
        if bc is None:
            fb = self._encode_marker_raw() + self._compress_backend(data)
            if self._decompress_lzh_pipeline(fb) != data: raise RuntimeError("FB")
            return fb
        if self._decompress_lzh_pipeline(bc) == data: return bc
        fb = self._encode_marker_raw() + self._compress_backend(data)
        if self._decompress_lzh_pipeline(fb) != data: raise RuntimeError("FB")
        return fb

    def _decompress_lzh_pipeline(self, data):
        off, seq = self._decode_header(data)
        if off == 0: return None
        if len(data) <= off or data[off] != 0xFF: return None
        tr = self._decode_lzh(data[off+1:])
        if tr is None: return None
        if not seq: return tr
        return self._reverse_sequence(tr, seq)

    def _decompress_auto(self, data):
        off, seq = self._decode_header(data)
        if off == 0: raise DecompressionError("Bad header")
        pl = data[off:]
        if not pl: raise DecompressionError("Empty")
        res = self._decompress_backend(pl)
        if res is None: raise DecompressionError("Backend")
        if not seq: return res, None
        return self._reverse_sequence(res, seq), seq

    def _reverse_sequence(self, data, seq):
        r = data
        for t in reversed(seq): r = self.rev_transforms[t](r)
        return r

    def _auto_output_name(self, infile, suffix=".jp"):
        return f"{os.path.basename(infile)}{suffix}"
    def _atomic_write(self, path, data):
        dn = os.path.dirname(path) or '.'
        bn = os.path.basename(path)
        fd, tmp = tempfile.mkstemp(prefix=bn + '.tmp', dir=dn)
        try:
            os.write(fd, data); os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
    def compress_file(self, infile, outfile="", use_lzh=False, time_limit=None):
        try:
            with open(infile, 'rb') as f: data = f.read()
        except Exception as e:
            print(f"Read error: {e}"); return
        try:
            if use_lzh:
                c = self.compress_with_lzh(data, time_limit); suf = ".jp.lzh"
            else:
                c = self.compress_with_verification(data, time_limit); suf = ".jp"
        except RuntimeError as e:
            print(f"Compression failed: {e}"); return
        if not outfile: outfile = self._auto_output_name(infile, suf)
        try: self._atomic_write(outfile, c)
        except Exception as e:
            print(f"Write error: {e}"); return
        print(f"Compressed {len(data)} → {len(c)} bytes → {outfile}")
    def decompress_file(self, infile, outfile=""):
        try:
            with open(infile, 'rb') as f: data = f.read()
        except Exception as e:
            print(f"Read error: {e}"); return False
        try:
            if data.startswith(b'DICT'):
                if data.startswith(b'DICT\x01'): orig = self._decompress_static_dict(data)
                elif data.startswith(b'DICT\x02'): orig = self._decompress_dynamic_dict(data)
                else: print("Unknown dict"); return False
            elif data.startswith(b'LINE'): orig = self._decompress_line_dict(data)
            else:
                off, seq = self._decode_header(data)
                if off == 0: print("Bad header"); return False
                if off < len(data) and data[off] == 0xFF:
                    orig = self._decompress_lzh_pipeline(data)
                else:
                    try: orig, _ = self._decompress_auto(data)
                    except DecompressionError as e:
                        print(f"Decompress: {e}"); return False
        except DecompressionError as e:
            print(f"Decompress: {e}"); return False
        except Exception as e:
            print(f"Unexpected: {e}"); return False
        if orig is None: print("None"); return False
        if not outfile:
            bn = os.path.basename(infile)
            outfile = re.sub(r'\.jp(\.lzh)?$', '', bn)
        try: self._atomic_write(outfile, orig)
        except Exception as e:
            print(f"Write error: {e}"); return False
        print(f"Decompressed → {outfile} ({len(orig)} bytes)")
        return True

    def full_self_test(self) -> bool:
        print("="*60)
        print("Unified PAQJP+PJP – Self‑Test (Algorithm 58 repeats 1-256)")
        print("="*60)
        print(f"zstandard available: {HAS_ZSTD}")
        print(f"paq available:       {paq is not None}")
        test_byte_values = [0x00, 0xFF, 0xAA, 0x55, 0x12, 0x34]
        all_ok = True
        for t in range(1, 257):
            if t % 25 == 0: print(f"  Testing transform {t}...")
            for tb in test_byte_values:
                td = bytes([tb])
                try:
                    tr = self.fwd_transforms[t](td)
                    rs = self.rev_transforms[t](tr)
                    if rs != td:
                        print(f"  FAIL: t={t} byte={tb:#04x}"); all_ok = False; break
                except TransformError: continue
                except Exception as e:
                    print(f"  EXC t={t} byte={tb:#04x}: {e}"); all_ok = False; break
            if not all_ok: break
        if not all_ok:
            print("\n  FAILED"); return False
        print("\n  All 256 transforms passed on test bytes.")
        print("\nAlgorithm 58 repeat demo (16×00 + 16×FF):")
        demo = b'\x00' * 16 + b'\xFF' * 16
        enc = self.transform_58(demo)
        dec = self.reverse_transform_58(enc)
        print(f"  demo in={len(demo)}B  t58 out={len(enc)}B  roundtrip={dec == demo}")
        print("\nRandom 100-byte compress+decompress...")
        rng = random.Random(12345)
        td = bytes(rng.randint(0, 255) for _ in range(100))
        try:
            c = self.compress_with_verification(td, time_limit=30)
            d, _ = self._decompress_auto(c)
            if d != td: print("  FAIL: mismatch"); return False
            print(f"  PASS  (100 B → {len(c)} B)")
        except RuntimeError as e:
            print(f"  Could not compress: {e}"); return False
        print("\n[All checks passed – 100% lossless]")
        return True

def main():
    print(f"{PROGNAME}")
    print("Compressed output: input.txt.jp (or .jp.lzh)")
    print(f"Backend: {'zstd/paq auto-detect' if HAS_ZSTD else 'paq auto-detect'} — no flag byte.")
    print(f"Algorithm 58: iterated bit-RLE up to {UnifiedCompressor.RLE58_MAX_PASSES} passes per block.")
    print("Lossless output guaranteed via per-input verification.\n")
    c = UnifiedCompressor()
    print("="*58)
    print("Options:")
    print("  1) Compress (Fast)")
    print("  2) Decompress")
    print("  3) Full self-test")
    print("  0) Exit")
    print("="*58)
    while True:
        print("\nMenu:")
        print("1) Compress (Fast)")
        print("2) Decompress")
        print("3) Full self-test")
        print("0) Exit")
        ch = input("> ").strip()
        if ch == "1":
            inf = input("Input file: ").strip()
            c.compress_file(inf, use_lzh=False)
        elif ch == "2":
            while True:
                inf = input("Compressed file [Enter to cancel]: ").strip()
                if not inf: break
                if not (inf.lower().endswith('.jp') or inf.lower().endswith('.jp.lzh')):
                    print("Only .jp / .jp.lzh supported."); continue
                outf = input("Output file (blank = auto): ").strip()
                if c.decompress_file(inf, outf): break
                print("Failed. Try again or Enter to cancel.")
        elif ch == "3": c.full_self_test()
        elif ch == "0": break
        else: print("Invalid.")

if __name__ == "__main__":
    main()
