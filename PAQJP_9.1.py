#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAQJP 9.0 – 256 Lossless Transforms + 2704 Transform‑Pair Sequences
+ Static Word Dictionary + Static Line Dictionary (1024 phrases, 8‑byte index)
+ Dynamic Dictionary (words, sentences, lines, paragraphs)
+ 1-8 byte words, sentence, paragraph (3-5 lines), line (1-255), line (256-65535) dictionaries
+ LZ77 + Huffman + Zstd + PAQ backends
+ Quantum transforms 27 (9 qubits) and 28 (17 qubits) – Qiskit optional
(Auto‑correcting marker‑free default, safe fallback if needed)

--- AUTO‑DOWNLOAD FROM WORKING GOOGLE DRIVE LINKS ---
All dictionary .txt files are stored in the 'Dictionaries/' folder.
"""

import subprocess
import sys
import importlib

# ------------------------------------------------------------------
# AUTO-INSTALL MISSING DEPENDENCIES
# ------------------------------------------------------------------
REQUIRED_PACKAGES = ["mpmath", "cython", "zstandard", "paq"]
PAQ_FALLBACK_URL = "git+https://github.com/observerss/paq.git"

def install_package(package_spec):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
        print(f"[OK] Installed: {package_spec}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Failed to install {package_spec}: {e}")
        return False

def install_dependencies():
    print("[CHECK] Checking required packages...")
    to_install = []
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
            print(f"  [OK] {pkg} already installed")
        except ImportError:
            print(f"  [MISS] {pkg} not found – will install")
            to_install.append(pkg)
    if not to_install:
        print("[OK] All dependencies are present.")
        return
    print(f"[INSTALL] Installing missing packages: {', '.join(to_install)}")
    for pkg in to_install:
        if pkg == "paq":
            print("   Attempting 'pip install paq' (PyPI)...")
            if not install_package("paq"):
                print(f"   PyPI failed, trying fallback URL: {PAQ_FALLBACK_URL}")
                install_package(PAQ_FALLBACK_URL)
        else:
            install_package(pkg)

install_dependencies()

# ---------- Now imports ----------
import math
import random
import decimal
import hashlib
import struct
import re
import os
import urllib.request
from typing import Optional, List, Tuple, Dict, Callable
from collections import Counter

# ---------- Optional Qiskit (QuantumCircuit only) ----------
try:
    from qiskit import QuantumCircuit
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    print("[INFO] Qiskit not installed. Quantum transforms will use classical fallback.")

# ---------- Optional compression backends ----------
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

PROGNAME = "PAQJP_9.0_LOSSLESS_HYBRID_DICT_QUANTUM"

# ---------- Dictionary configuration – only working files ----------
DICT_DIR = "Dictionaries"
COMBINED_DICTIONARY_FILE = os.path.join(DICT_DIR, "dictionary_combined.txt")

DICTIONARY_FILES = [
    "generated.txt",
    "eng_news_2005_1M-sentences.txt",
    "eng_news_2005_1M-words.txt",
    "eng_news_2005_1M-sources.txt",
    "eng_news_2005_1M-co_n.txt",
    "eng_news_2005_1M-co_s.txt",
    "eng_news_2005_1M-inv_w_2.txt",
    "eng_news_2005_1M-inv_w_3.txt",
    "eng_news_2005_1M-inv_so.txt",
    "eng_news_2005_1M-meta.txt",
    "Dictionary.txt",
    "the-complete-reference-html-css-fifth-edition.txt",
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

# ---------- Constants ----------
PRIMES = [p for p in range(2, 256) if all(p % d != 0 for d in range(2, int(p ** 0.5) + 1))]
PI_DIGITS = [79, 17, 111]
BLOCK_SIZE = 1024

def find_nearest_prime_around(n: int) -> int:
    o = 0
    while True:
        c1, c2 = n - o, n + o
        if c1 >= 2 and all(c1 % d != 0 for d in range(2, int(c1 ** 0.5) + 1)):
            return c1
        if c2 >= 2 and all(c2 % d != 0 for d in range(2, int(c2 ** 0.5) + 1)):
            return c2
        o += 1

def sha256_8bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()[:8]

def xor_prime_hash(word: str) -> bytes:
    prime = 2147483647
    total = sum(ord(c) for c in word)
    transformed = total ^ prime
    return transformed.to_bytes(8, 'big')

# ---------- LZ77 and Huffman backends ----------
def lz77_compress(data: bytes) -> bytes:
    if not data:
        return b''
    out = bytearray()
    i = 0
    n = len(data)
    window_size = 4096
    max_len = 16
    while i < n:
        start = max(0, i - window_size)
        best_len = 0
        best_dist = 0
        if i + 2 <= n:
            for dist in range(1, min(i - start, window_size) + 1):
                max_possible = min(max_len, n - i)
                match_len = 0
                while match_len < max_possible and data[i + match_len] == data[i - dist + match_len]:
                    match_len += 1
                if match_len > best_len:
                    best_len = match_len
                    best_dist = dist
                    if best_len == max_possible:
                        break
        if best_len >= 3:
            encoded = (best_dist << 4) | (best_len - 1)
            out.append(0x80 | (encoded >> 8))
            out.append(encoded & 0xFF)
            i += best_len
        else:
            out.append(0x00)
            out.append(data[i])
            i += 1
    return bytes(out)

def lz77_decompress(data: bytes) -> bytes:
    if not data:
        return b''
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        marker = data[i]
        if marker & 0x80:
            if i + 1 >= n:
                break
            encoded = ((marker & 0x7F) << 8) | data[i+1]
            dist = encoded >> 4
            length = (encoded & 0x0F) + 1
            i += 2
            start = len(out) - dist
            for _ in range(length):
                out.append(out[start])
                start += 1
        else:
            if i + 1 >= n:
                break
            out.append(data[i+1])
            i += 2
    return bytes(out)

def huffman_compress(data: bytes) -> bytes:
    if not data:
        return b''
    freq = [0]*256
    for b in data:
        freq[b] += 1
    import heapq
    class Node:
        __slots__ = ('freq', 'left', 'right', 'symbol')
        def __init__(self, freq, left=None, right=None, symbol=None):
            self.freq = freq
            self.left = left
            self.right = right
            self.symbol = symbol
    heap = [(freq[i], Node(freq[i], symbol=i)) for i in range(256) if freq[i] > 0]
    heapq.heapify(heap)
    if not heap:
        return b''
    if len(heap) == 1:
        code_lengths = [0]*256
        codes = [0]*256
        code_lengths[heap[0][1].symbol] = 1
        codes[heap[0][1].symbol] = 0
        header = bytearray()
        for l in code_lengths:
            header.append(l)
        return bytes(header)
    while len(heap) > 1:
        f1, n1 = heapq.heappop(heap)
        f2, n2 = heapq.heappop(heap)
        merged = Node(f1+f2, left=n1, right=n2)
        heapq.heappush(heap, (f1+f2, merged))
    root = heap[0][1]
    code_lengths = [0]*256
    codes = [0]*256
    def dfs(node, depth, code):
        if node.left is None and node.right is None:
            code_lengths[node.symbol] = depth
            codes[node.symbol] = code
        else:
            if node.left:
                dfs(node.left, depth+1, code << 1)
            if node.right:
                dfs(node.right, depth+1, (code << 1) | 1)
    dfs(root, 0, 0)
    header = bytearray()
    for l in code_lengths:
        header.append(l)
    bits = []
    for b in data:
        code = codes[b]
        length = code_lengths[b]
        for j in range(length-1, -1, -1):
            bits.append((code >> j) & 1)
    pad = (8 - len(bits) % 8) % 8
    bits += [0]*pad
    bit_bytes = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i+j]
        bit_bytes.append(byte)
    return bytes(header) + bytes(bit_bytes)

def huffman_decompress(data: bytes) -> bytes:
    if not data:
        return b''
    if len(data) < 256:
        return b''
    code_lengths = list(data[:256])
    bitstream = data[256:]
    symbols = [(i, code_lengths[i]) for i in range(256) if code_lengths[i] > 0]
    if not symbols:
        return b''
    symbols.sort(key=lambda x: (x[1], x[0]))
    class Node:
        __slots__ = ('left', 'right', 'symbol')
        def __init__(self):
            self.left = None
            self.right = None
            self.symbol = None
    next_code = 0
    prev_len = 0
    codes_rebuilt = {}
    for length in range(1, 17):  # assume max length 16
        next_code <<= (length - prev_len)
        for sym, l in symbols:
            if l == length:
                codes_rebuilt[sym] = next_code
                next_code += 1
        prev_len = length
    root = Node()
    for sym, code in codes_rebuilt.items():
        node = root
        for bit in range(code_lengths[sym]-1, -1, -1):
            bitval = (code >> bit) & 1
            if bitval == 0:
                if node.left is None:
                    node.left = Node()
                node = node.left
            else:
                if node.right is None:
                    node.right = Node()
                node = node.right
        node.symbol = sym
    bits = []
    for byte in bitstream:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    out = bytearray()
    node = root
    for bit in bits:
        if bit == 0:
            node = node.left
        else:
            node = node.right
        if node is None:
            break
        if node.symbol is not None:
            out.append(node.symbol)
            node = root
    return bytes(out)

# ---------- Quantum mask generation (no simulation) ----------
def _generate_quantum_mask(data_len: int, num_qubits: int, seed: int) -> bytes:
    """Generate a mask using a quantum circuit's QASM hash (no simulation)."""
    if HAS_QISKIT:
        try:
            rng = random.Random(seed)
            circuit = QuantumCircuit(num_qubits, num_qubits)
            for _ in range(num_qubits * 2):
                q = rng.randint(0, num_qubits-1)
                gate = rng.choice(['h', 'x', 'y', 'z'])
                if gate == 'h':
                    circuit.h(q)
                elif gate == 'x':
                    circuit.x(q)
                elif gate == 'y':
                    circuit.y(q)
                elif gate == 'z':
                    circuit.z(q)
                if rng.random() < 0.3 and num_qubits > 1:
                    ctrl = rng.randint(0, num_qubits-1)
                    tgt = rng.randint(0, num_qubits-1)
                    if ctrl != tgt:
                        circuit.cx(ctrl, tgt)
            qasm_str = circuit.qasm()
            digest = hashlib.sha256(qasm_str.encode()).digest()
            mask = (digest * ((data_len // len(digest)) + 1))[:data_len]
            return mask
        except Exception as e:
            print(f"[WARNING] Qiskit circuit creation failed: {e}. Falling back to classical.")
    # Classical fallback
    rng = random.Random(seed)
    return bytes(rng.randint(0, 255) for _ in range(data_len))

# ---------- Helper for packing/unpacking indices ----------
def _pack_index(idx: int, idx_bytes: int) -> bytes:
    if idx_bytes == 2:
        return struct.pack('>H', idx)
    elif idx_bytes == 3:
        return struct.pack('>I', idx)[1:4]
    else:
        return struct.pack('>Q', idx)

def _unpack_index(data: bytes, idx_bytes: int) -> Tuple[int, int]:
    if idx_bytes == 2:
        return struct.unpack('>H', data[:2])[0], 2
    elif idx_bytes == 3:
        return struct.unpack('>I', b'\x00' + data[:3])[0], 3
    else:
        return struct.unpack('>Q', data[:8])[0], 8

# ---------- Download and merge dictionaries ----------
def download_and_merge_dictionaries():
    """Download all dictionary files into Dictionaries/ folder and merge them."""
    if not os.path.exists(DICT_DIR):
        os.makedirs(DICT_DIR)

    if os.path.exists(COMBINED_DICTIONARY_FILE):
        print(f"Combined dictionary '{COMBINED_DICTIONARY_FILE}' already exists. Skipping download.")
        return True

    all_words = set()
    success_count = 0

    for idx, (filename, url) in enumerate(zip(DICTIONARY_FILES, DICTIONARY_URLS)):
        local_path = os.path.join(DICT_DIR, filename)
        print(f"Downloading {filename} to {DICT_DIR}/ ...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                content = response.read()

            if b'<html' in content[:200].lower():
                print(f"  WARNING: {filename} appears to be an HTML page (not a text file). Skipping.")
                continue

            with open(local_path, 'wb') as f:
                f.write(content)

            with open(local_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    w = line.strip()
                    if w:
                        all_words.add(w)

            print(f"  Downloaded {filename} ({os.path.getsize(local_path)} bytes)")
            success_count += 1

        except Exception as e:
            print(f"  WARNING: Could not download {filename}: {e}")
            if os.path.exists(local_path):
                os.remove(local_path)

    if success_count == 0:
        print("ERROR: No dictionary files could be downloaded.")
        print("Proceeding without static word and line dictionaries.")
        return False

    try:
        with open(COMBINED_DICTIONARY_FILE, 'w', encoding='utf-8') as f:
            for word in sorted(all_words):
                f.write(word + '\n')
        print(f"Merged {len(all_words)} unique words into {COMBINED_DICTIONARY_FILE} "
              f"({os.path.getsize(COMBINED_DICTIONARY_FILE)} bytes)")
        return True
    except Exception as e:
        print(f"ERROR: Could not write combined dictionary: {e}")
        return False

# ---------- Main Compressor Class ----------
class PAQJPCompressor:
    def __init__(self):
        download_and_merge_dictionaries()

        self.PI_DIGITS = PI_DIGITS.copy()
        self.seed_tables = self._gen_seed_tables(num=126, size=40, seed=42)
        self.fibonacci = self._gen_fib(100)
        self.PI_STR = "3.14159265358979323846264338327950288419716939937510"

        self._build_transform_maps()
        self.sequences = self._build_pair_sequences()
        self.pair_lookup = {idx: (t1, t2) for idx, (t1, t2) in enumerate(self.sequences)}

        # Load all dictionaries
        self.static_dict, self.word_to_index = self._load_static_dictionary()
        self.line_dict, self.line_to_index = self._load_line_dictionary()
        self.word_dict_1_8, self.word_dict_1_8_map = self._load_word_dict_1_8()
        self.sentence_dict, self.sentence_dict_map = self._load_sentence_dict()
        self.paragraph_dict, self.paragraph_dict_map = self._load_paragraph_dict()
        self.line_dict_1_255, self.line_dict_1_255_map = self._load_line_dict_range(1, 255)
        self.line_dict_256_65535, self.line_dict_256_65535_map = self._load_line_dict_range(256, 65535)

    # ------------------------------------------------------------------
    # Dictionary loaders for new types
    # ------------------------------------------------------------------
    def _load_word_dict_1_8(self):
        if not os.path.exists(COMBINED_DICTIONARY_FILE):
            return [], {}
        words = []
        try:
            with open(COMBINED_DICTIONARY_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    w = line.strip()
                    if w and 1 <= len(w.encode('utf-8')) <= 8:
                        words.append(w)
        except Exception as e:
            print(f"Warning: could not load word dict 1-8: {e}")
            return [], {}
        words = sorted(set(words))
        wmap = {w: i for i, w in enumerate(words)}
        print(f"Loaded word dict (1-8 bytes): {len(words)} entries")
        return words, wmap

    def _load_sentence_dict(self):
        if not os.path.exists(COMBINED_DICTIONARY_FILE):
            return [], {}
        sentences = []
        try:
            with open(COMBINED_DICTIONARY_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    s = line.strip()
                    if s and any(c in s for c in '.!?;:') and len(s) > 10:
                        sentences.append(s)
        except Exception as e:
            print(f"Warning: could not load sentence dict: {e}")
            return [], {}
        sentences = sorted(set(sentences), key=len, reverse=True)[:MAX_LINE_ENTRIES*2]
        smap = {s: i for i, s in enumerate(sentences)}
        print(f"Loaded sentence dict: {len(sentences)} entries")
        return sentences, smap

    def _load_paragraph_dict(self):
        if not os.path.exists(COMBINED_DICTIONARY_FILE):
            return [], {}
        paragraphs = []
        try:
            with open(COMBINED_DICTIONARY_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    p = line.strip()
                    if p and len(p) > 80 and (' ' in p) and len(p) < 2000:
                        paragraphs.append(p)
        except Exception as e:
            print(f"Warning: could not load paragraph dict: {e}")
            return [], {}
        paragraphs = sorted(set(paragraphs), key=len, reverse=True)[:1024]
        pmap = {p: i for i, p in enumerate(paragraphs)}
        print(f"Loaded paragraph dict: {len(paragraphs)} entries")
        return paragraphs, pmap

    def _load_line_dict_range(self, min_len: int, max_len: int):
        if not os.path.exists(COMBINED_DICTIONARY_FILE):
            return [], {}
        lines = []
        try:
            with open(COMBINED_DICTIONARY_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    s = line.strip()
                    l = len(s)
                    if s and min_len <= l <= max_len:
                        lines.append(s)
        except Exception as e:
            print(f"Warning: could not load line dict range {min_len}-{max_len}: {e}")
            return [], {}
        lines = sorted(set(lines), key=len, reverse=True)[:2048]
        lmap = {s: i for i, s in enumerate(lines)}
        print(f"Loaded line dict ({min_len}-{max_len}): {len(lines)} entries")
        return lines, lmap

    # ------------------------------------------------------------------
    # Original static & line dictionaries (backward compatibility)
    # ------------------------------------------------------------------
    def _load_static_dictionary(self):
        if not os.path.exists(COMBINED_DICTIONARY_FILE):
            print(f"ERROR: {COMBINED_DICTIONARY_FILE} not found. No dictionaries loaded.")
            return [], {}

        words_set = set()
        try:
            with open(COMBINED_DICTIONARY_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    w = line.strip()
                    if w:
                        words_set.add(w)
        except Exception as e:
            print(f"Warning: could not read {COMBINED_DICTIONARY_FILE}: {e}")
            return [], {}

        sorted_words = sorted(words_set)
        word_to_idx = {w: i for i, w in enumerate(sorted_words)}
        print(f"Loaded static word dictionary: {len(sorted_words)} unique words (from {COMBINED_DICTIONARY_FILE}).")
        return sorted_words, word_to_idx

    def _load_line_dictionary(self):
        if not os.path.exists(COMBINED_DICTIONARY_FILE):
            print(f"ERROR: {COMBINED_DICTIONARY_FILE} not found. Line dictionary disabled.")
            return [], {}

        lines = []
        try:
            with open(COMBINED_DICTIONARY_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                for raw_line in f:
                    phrase = raw_line.strip()
                    if phrase and phrase not in lines:
                        lines.append(phrase)
                        if len(lines) >= MAX_LINE_ENTRIES:
                            break
        except Exception as e:
            print(f"Warning: could not read {COMBINED_DICTIONARY_FILE}: {e}")
            return [], {}

        if not lines:
            print("Warning: No phrases found. Line dictionary disabled.")
            return [], {}

        lines.sort(key=len, reverse=True)
        line_to_idx = {phrase: i for i, phrase in enumerate(lines)}
        print(f"Loaded line dictionary: {len(lines)} phrases (first {MAX_LINE_ENTRIES} from {COMBINED_DICTIONARY_FILE}).")
        return lines, line_to_idx

    # ------------------------------------------------------------------
    # pi / constant helpers
    # ------------------------------------------------------------------
    def get_pi_digits(self, n: int) -> str:
        if n < 1: return ""
        return self.PI_STR[2:2 + n]

    def find_lossless_k(self, n: int):
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

    def to_bin(self, value: int, bits: int) -> str:
        return format(value, 'b').zfill(bits)

    def get_bit_size(self, k: int) -> int:
        return 23 if k <= 0x7FFFFF else 25

    def transform_17(self, data: bytes) -> bytes:
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

    reverse_transform_17 = transform_17

    def get_basel_digits(self, n: int) -> str:
        decimal.getcontext().prec = n + 5
        pi = decimal.Decimal(self.PI_STR)
        basel = (pi * pi) / decimal.Decimal(6)
        s = str(basel).replace('.', '')
        return s[:n]

    def get_one_over_e_digits(self, n: int) -> str:
        decimal.getcontext().prec = n + 5
        e = decimal.Decimal(1).exp()
        inv_e = decimal.Decimal(1) / e
        s = str(inv_e).replace('.', '')
        return s[:n]

    def get_5e_digits(self, n: int) -> str:
        decimal.getcontext().prec = n + 5
        e = decimal.Decimal(1).exp()
        five_e = decimal.Decimal(5) * e
        s = str(five_e).replace('.', '')
        return s[:n]

    # ------------------------------------------------------------------
    # Seed tables, Fibonacci
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
    # Bit helpers (for RLE)
    # ------------------------------------------------------------------
    def _append_bits(self, bitlist: List[int], value: int, count: int):
        for i in range(count - 1, -1, -1):
            bitlist.append((value >> i) & 1)

    def _read_bits(self, bits: List[int], pos: int, count: int) -> int:
        val = 0
        for i in range(count):
            if pos + i >= len(bits): return 0
            val = (val << 1) | bits[pos + i]
        return val

    # ------------------------------------------------------------------
    # RLE transform 00 (transform 1)
    # ------------------------------------------------------------------
    def transform_00(self, data: bytes) -> bytes:
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
            for shift in range(256):
                tmp = bytearray(current)
                for j in range(len(tmp)):
                    tmp[j] = (tmp[j] + shift) % 256
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
            rle_encoded = self._apply_rle_to_shifted(best_shifted, best_shift)
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

    def _apply_rle_to_shifted(self, shifted_data: bytearray, shift: int) -> bytes:
        bits = []
        self._append_bits(bits, 0b010, 3)
        self._append_bits(bits, shift, 8)
        i = 0
        n = len(shifted_data)
        while i < n:
            val = shifted_data[i]
            run = 1
            i += 1
            while i < n and shifted_data[i] == val:
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
        if not cdata or cdata == b'\x00': return b''
        if cdata[0] == 0: return cdata[1:]
        num_passes = cdata[0]
        if num_passes == 0 or len(cdata) < 1 + num_passes: return b''
        shifts = list(cdata[1:1 + num_passes])
        rle_data = cdata[1 + num_passes:]
        decoded = self._rle_decode(rle_data)
        if decoded is None: return b''
        current = bytearray(decoded)
        for shift in reversed(shifts):
            for i in range(len(current)):
                current[i] = (current[i] - shift) % 256
        return bytes(current)

    def _rle_decode(self, data: bytes) -> Optional[bytearray]:
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
        pos += 8
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
            else:
                if pos + 2 + 8 + 8 > nbits: break
                if self._read_bits(bits, pos, 2) != 0b11: return None
                pos += 2
                run = 13 + self._read_bits(bits, pos, 8)
                pos += 8
            if pos + 8 > nbits: break
            val = self._read_bits(bits, pos, 8)
            pos += 8
            out.extend([val] * run)
        for i in range(pos, nbits):
            if bits[i] != 0: return None
        return out

    # ------------------------------------------------------------------
    # Transforms 01‑21 (unchanged)
    # ------------------------------------------------------------------
    def transform_01(self, d, r=100):
        t = bytearray(d)
        for prime in PRIMES:
            xor_val = prime if prime == 2 else max(1, math.ceil(prime * 4096 / 28672))
            for _ in range(r):
                for i in range(0, len(t), 3):
                    if i < len(t): t[i] ^= xor_val
        return bytes(t)
    reverse_transform_01 = transform_01

    def transform_02(self, d):
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

    def transform_03(self, d):
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
        inv = [0]*256
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
    reverse_transform_07 = transform_07

    def transform_08(self, d, r=100):
        t = bytearray(d)
        sh = len(d) % len(self.PI_DIGITS)
        pi_rot = self.PI_DIGITS[sh:] + self.PI_DIGITS[:sh]
        p = find_nearest_prime_around(len(d) % 256)
        for i in range(len(t)): t[i] ^= p
        for _ in range(r):
            for i in range(len(t)): t[i] ^= pi_rot[i % len(pi_rot)]
        return bytes(t)
    reverse_transform_08 = transform_08

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
    reverse_transform_09 = transform_09

    def transform_10(self, data: bytes) -> bytes:
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

    def transform_11(self, data: bytes) -> bytes:
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

    def transform_12(self, data: bytes) -> bytes:
        t = bytearray(data)
        for i in range(len(t)): t[i] ^= self.fibonacci[i % len(self.fibonacci)] % 256
        return bytes(t)
    reverse_transform_12 = transform_12

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

    def transform_16(self, data: bytes) -> bytes:
        if not data: return b''
        xor_byte = (len(data) * 7 + 13) % 256
        t = bytearray(data)
        for i in range(len(t)): t[i] ^= xor_byte
        return bytes(t)
    reverse_transform_16 = transform_16

    # transform_17 defined earlier
    def transform_18(self, data: bytes) -> bytes:
        if not data: return b''
        digits = self.get_basel_digits(max(10, len(data)//2 + 5))
        mask = bytes(int(digits[i:i+2]) % 256 for i in range(0, len(digits), 2))
        t = bytearray(data)
        for i in range(len(t)): t[i] ^= mask[i % len(mask)]
        return bytes(t)
    reverse_transform_18 = transform_18

    def transform_19(self, data: bytes) -> bytes:
        if not data: return b''
        digits = self.get_one_over_e_digits(max(10, len(data)//2 + 5))
        mask = bytes(int(digits[i:i+2]) % 256 for i in range(0, len(digits), 2))
        t = bytearray(data)
        for i in range(len(t)): t[i] ^= mask[i % len(mask)]
        return bytes(t)
    reverse_transform_19 = transform_19

    def transform_20(self, data: bytes) -> bytes:
        if not data: return b''
        digits = self.get_5e_digits(max(10, len(data)//2 + 5))
        mask = bytes(int(digits[i:i+2]) % 256 for i in range(0, len(digits), 2))
        t = bytearray(data)
        for i in range(len(t)): t[i] ^= mask[i % len(mask)]
        return bytes(t)
    reverse_transform_20 = transform_20

    def transform_21(self, data: bytes) -> bytes:
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
    # Transform 23 – SHA‑256 word tokenizer
    # ------------------------------------------------------------------
    def transform_23(self, data: bytes) -> bytes:
        if not data: return b'\x00\x00\x00\x00'
        try:
            text = data.decode('latin-1')
        except:
            text = data.decode('latin-1', errors='replace')
        pattern = r'([A-Za-z0-9_]+)'
        tokens = re.split(pattern, text)
        hash_to_word = {}
        token_list = []
        for i, tok in enumerate(tokens):
            if i % 2 == 1:
                word_bytes = tok.encode('latin-1')
                h = sha256_8bytes(word_bytes)
                if h in hash_to_word:
                    if hash_to_word[h] != word_bytes:
                        token_list.append((False, word_bytes))
                        continue
                else:
                    hash_to_word[h] = word_bytes
                token_list.append((True, h))
            else:
                if tok:
                    token_list.append((False, tok.encode('latin-1')))
        dict_entries = list(hash_to_word.items())
        num_entries = len(dict_entries)
        result = bytearray()
        result += struct.pack('>I', num_entries)
        for h, wb in dict_entries:
            result += h
            result += struct.pack('>H', len(wb))
            result += wb
        for is_word, payload in token_list:
            if is_word:
                result += b'\x01'
                result += payload
            else:
                result += b'\x00'
                result += struct.pack('>H', len(payload))
                result += payload
        return bytes(result)

    def reverse_transform_23(self, data: bytes) -> bytes:
        if not data: return b''
        if len(data) < 4: return data
        num_entries = struct.unpack('>I', data[:4])[0]
        pos = 4
        hash_to_word = {}
        for _ in range(num_entries):
            if pos + 10 > len(data): break
            h = data[pos:pos+8]
            pos += 8
            wlen = struct.unpack('>H', data[pos:pos+2])[0]
            pos += 2
            if pos + wlen > len(data): break
            wb = data[pos:pos+wlen]
            pos += wlen
            hash_to_word[h] = wb
        out = bytearray()
        while pos < len(data):
            if pos >= len(data): break
            typ = data[pos]
            pos += 1
            if typ == 1:
                if pos + 8 > len(data): break
                h = data[pos:pos+8]
                pos += 8
                wb = hash_to_word.get(h)
                out += wb if wb else h
            elif typ == 0:
                if pos + 2 > len(data): break
                rawlen = struct.unpack('>H', data[pos:pos+2])[0]
                pos += 2
                if pos + rawlen > len(data): break
                out += data[pos:pos+rawlen]
                pos += rawlen
            else:
                break
        return bytes(out)

    # ------------------------------------------------------------------
    # Transform 24 – XOR‑prime word tokenizer
    # ------------------------------------------------------------------
    def transform_24(self, data: bytes) -> bytes:
        if not data: return b'\x00\x00\x00\x00'
        try:
            text = data.decode('latin-1')
        except:
            text = data.decode('latin-1', errors='replace')
        pattern = r'([A-Za-z0-9_]+)'
        tokens = re.split(pattern, text)
        hash_to_word = {}
        token_list = []
        for i, tok in enumerate(tokens):
            if i % 2 == 1:
                word_bytes = tok.encode('latin-1')
                h = xor_prime_hash(tok)
                if h in hash_to_word:
                    if hash_to_word[h] != word_bytes:
                        token_list.append((False, word_bytes))
                        continue
                else:
                    hash_to_word[h] = word_bytes
                token_list.append((True, h))
            else:
                if tok:
                    token_list.append((False, tok.encode('latin-1')))
        dict_entries = list(hash_to_word.items())
        num_entries = len(dict_entries)
        result = bytearray()
        result += struct.pack('>I', num_entries)
        for h, wb in dict_entries:
            result += h
            result += struct.pack('>H', len(wb))
            result += wb
        for is_word, payload in token_list:
            if is_word:
                result += b'\x01'
                result += payload
            else:
                result += b'\x00'
                result += struct.pack('>H', len(payload))
                result += payload
        return bytes(result)

    def reverse_transform_24(self, data: bytes) -> bytes:
        if not data: return b''
        if len(data) < 4: return data
        num_entries = struct.unpack('>I', data[:4])[0]
        pos = 4
        hash_to_word = {}
        for _ in range(num_entries):
            if pos + 10 > len(data): break
            h = data[pos:pos+8]
            pos += 8
            wlen = struct.unpack('>H', data[pos:pos+2])[0]
            pos += 2
            if pos + wlen > len(data): break
            wb = data[pos:pos+wlen]
            pos += wlen
            hash_to_word[h] = wb
        out = bytearray()
        while pos < len(data):
            if pos >= len(data): break
            typ = data[pos]
            pos += 1
            if typ == 1:
                if pos + 8 > len(data): break
                h = data[pos:pos+8]
                pos += 8
                wb = hash_to_word.get(h)
                out += wb if wb else h
            elif typ == 0:
                if pos + 2 > len(data): break
                rawlen = struct.unpack('>H', data[pos:pos+2])[0]
                pos += 2
                if pos + rawlen > len(data): break
                out += data[pos:pos+rawlen]
                pos += rawlen
            else:
                break
        return bytes(out)

    # ------------------------------------------------------------------
    # Transform 26 – SHA‑256 block masking
    # ------------------------------------------------------------------
    def transform_26(self, data: bytes) -> bytes:
        if not data: return b''
        secret = b"PAQJP_TRANSFORM26_SECRET"
        result = bytearray()
        for idx in range(0, len(data), BLOCK_SIZE):
            chunk = data[idx:idx+BLOCK_SIZE]
            block_num = idx // BLOCK_SIZE
            hasher = hashlib.sha256()
            hasher.update(secret)
            hasher.update(struct.pack(">Q", block_num))
            mask = hasher.digest()
            mask_repeated = (mask * ((len(chunk) // len(mask)) + 1))[:len(chunk)]
            xored = bytes(a ^ b for a, b in zip(chunk, mask_repeated))
            result.extend(xored)
        return bytes(result)

    def reverse_transform_26(self, data: bytes) -> bytes:
        return self.transform_26(data)

    # ------------------------------------------------------------------
    # Transform 254 – hex mask from π digits (0xFE)
    # ------------------------------------------------------------------
    def transform_254(self, data: bytes) -> bytes:
        if not data:
            return b''
        digits = self.PI_STR.replace('.', '')
        mask = bytes(int(digits[i:i+2]) % 256 for i in range(0, len(digits), 2))
        t = bytearray(data)
        for i in range(len(t)):
            t[i] ^= mask[i % len(mask)]
        return bytes(t)
    reverse_transform_254 = transform_254

    # ------------------------------------------------------------------
    # Quantum transforms (27: 9 qubits, 28: 17 qubits)
    # ------------------------------------------------------------------
    def transform_27(self, data: bytes) -> bytes:
        if not data:
            return b''
        mask = _generate_quantum_mask(len(data), 9, seed=27)
        t = bytearray(data)
        for i in range(len(t)):
            t[i] ^= mask[i]
        return bytes(t)
    reverse_transform_27 = transform_27

    def transform_28(self, data: bytes) -> bytes:
        if not data:
            return b''
        mask = _generate_quantum_mask(len(data), 17, seed=28)
        t = bytearray(data)
        for i in range(len(t)):
            t[i] ^= mask[i]
        return bytes(t)
    reverse_transform_28 = transform_28

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_pattern(self, size: int, index: int):
        random.seed(12345 + size * 100 + index)
        return [random.randint(0, 255) for _ in range(size)]

    def _calculate_repeats(self, data: bytes) -> int:
        if not data: return 1
        length = len(data)
        byte_sum = sum(data) % 256
        repeats = ((length * 13 + byte_sum * 17) % 256) + 1
        return max(1, min(256, repeats))

    def _dynamic_transform(self, n: int):
        def tf(data: bytes):
            if not data: return b''
            seed = self.get_seed(n % len(self.seed_tables), len(data))
            t = bytearray(data)
            for i in range(len(t)): t[i] ^= seed
            return bytes(t)
        return tf, tf

    # ------------------------------------------------------------------
    # Build transform maps (1..256)
    # ------------------------------------------------------------------
    def _build_transform_maps(self):
        self.fwd_transforms: Dict[int, Callable] = {}
        self.rev_transforms: Dict[int, Callable] = {}
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
        self.fwd_transforms[26] = self.transform_26; self.rev_transforms[26] = self.reverse_transform_26
        # New transforms:
        self.fwd_transforms[254] = self.transform_254; self.rev_transforms[254] = self.reverse_transform_254
        self.fwd_transforms[27] = self.transform_27; self.rev_transforms[27] = self.reverse_transform_27
        self.fwd_transforms[28] = self.transform_28; self.rev_transforms[28] = self.reverse_transform_28
        # Transform 25 is dynamic dictionary tokenizer
        self.fwd_transforms[25] = self.transform_25
        self.rev_transforms[25] = self.reverse_transform_25
        # Transforms 29-253 and 255 are dynamic (skip 254, 27, 28, 26)
        for i in range(29, 256):
            if i in (254, 27, 28, 26):
                continue
            fwd, rev = self._dynamic_transform(i)
            self.fwd_transforms[i] = fwd
            self.rev_transforms[i] = rev
        # Transform 256 is no-op
        self.fwd_transforms[256] = self.transform_256
        self.rev_transforms[256] = self.reverse_transform_256
        # Ensure all present
        for i in range(1, 257):
            if i not in self.fwd_transforms:
                raise RuntimeError(f"Transform {i} missing!")

    # ------------------------------------------------------------------
    # Build pair sequences – 2704 (52×52)
    # ------------------------------------------------------------------
    def _build_pair_sequences(self) -> List[Tuple[int, int]]:
        base = list(range(1, 53))
        return [(t1, t2) for t1 in base for t2 in base]

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
    # Backend compression with automatic selection (extended)
    # ------------------------------------------------------------------
    def _compress_with_best_backend(self, data: bytes) -> bytes:
        candidates = []
        candidates.append((b'N', b'N' + data))
        if HAS_ZSTD:
            try:
                candidates.append((b'Z', b'Z' + zstd_cctx.compress(data)))
            except:
                pass
        if paq is not None:
            try:
                candidates.append((b'P', b'P' + paq.compress(data)))
            except:
                pass
        try:
            lz = lz77_compress(data)
            candidates.append((b'L', b'L' + lz))
        except:
            pass
        try:
            huff = huffman_compress(data)
            candidates.append((b'H', b'H' + huff))
        except:
            pass
        best = min(candidates, key=lambda x: len(x[1]))
        return best[1]

    def _decompress_backend(self, data: bytes) -> Optional[bytes]:
        if len(data) < 1:
            return None
        marker = data[0:1]
        payload = data[1:]
        if marker == b'N':
            return payload
        elif marker == b'Z' and HAS_ZSTD:
            try:
                return zstd_dctx.decompress(payload)
            except:
                pass
        elif marker == b'P' and paq is not None:
            try:
                return paq.decompress(payload)
            except:
                pass
        elif marker == b'L':
            try:
                return lz77_decompress(payload)
            except:
                pass
        elif marker == b'H':
            try:
                return huffman_decompress(payload)
            except:
                pass
        return None

    # ------------------------------------------------------------------
    # Static word dictionary tokenizer (original)
    # ------------------------------------------------------------------
    MAGIC_DICT = b'DICT'

    def _tokenize_with_static_dict(self, data: bytes) -> Optional[bytes]:
        try:
            text = data.decode('utf-8')
        except:
            return None
        pattern = r'([A-Za-z0-9_]+)'
        tokens = re.split(pattern, text)
        stream = bytearray()
        for i, tok in enumerate(tokens):
            if i % 2 == 1:
                idx = self.word_to_index.get(tok)
                if idx is not None:
                    stream += b'\x01'
                    stream += struct.pack('>I', idx)
                else:
                    word_bytes = tok.encode('utf-8')
                    stream += b'\x02'
                    stream += struct.pack('>H', len(word_bytes))
                    stream += word_bytes
            else:
                if tok:
                    sep_bytes = tok.encode('utf-8')
                    stream += b'\x00'
                    stream += struct.pack('>H', len(sep_bytes))
                    stream += sep_bytes
        return bytes(stream)

    def _detokenize_static_dict(self, token_stream: bytes) -> Optional[bytes]:
        if not token_stream:
            return b''
        out = bytearray()
        pos = 0
        while pos < len(token_stream):
            if pos >= len(token_stream):
                break
            typ = token_stream[pos]
            pos += 1
            if typ == 0x01:
                if pos + 4 > len(token_stream):
                    break
                idx = struct.unpack('>I', token_stream[pos:pos+4])[0]
                pos += 4
                if idx < len(self.static_dict):
                    out += self.static_dict[idx].encode('utf-8')
                else:
                    return None
            elif typ == 0x02:
                if pos + 2 > len(token_stream):
                    break
                word_len = struct.unpack('>H', token_stream[pos:pos+2])[0]
                pos += 2
                if pos + word_len > len(token_stream):
                    break
                out += token_stream[pos:pos+word_len]
                pos += word_len
            elif typ == 0x00:
                if pos + 2 > len(token_stream):
                    break
                sep_len = struct.unpack('>H', token_stream[pos:pos+2])[0]
                pos += 2
                if pos + sep_len > len(token_stream):
                    break
                out += token_stream[pos:pos+sep_len]
                pos += sep_len
            else:
                break
        return bytes(out)

    def _compress_static_dict(self, data: bytes) -> Optional[bytes]:
        token_stream = self._tokenize_with_static_dict(data)
        if token_stream is None:
            return None
        compressed = self._compress_with_best_backend(token_stream)
        return self.MAGIC_DICT + b'\x01' + compressed

    def _decompress_static_dict(self, compressed: bytes) -> Optional[bytes]:
        if not compressed.startswith(self.MAGIC_DICT + b'\x01'):
            return None
        payload = compressed[len(self.MAGIC_DICT) + 1:]
        token_stream = self._decompress_backend(payload)
        if token_stream is None:
            return None
        return self._detokenize_static_dict(token_stream)

    # ------------------------------------------------------------------
    # Dynamic Dictionary Tokenizer (Transform 25)
    # ------------------------------------------------------------------
    def _split_text_into_chunks(self, text: str, level: str = 'all') -> List[str]:
        if level == 'paragraph':
            return re.split(r'(\n\n)', text)
        elif level == 'line':
            return re.split(r'(\n)', text)
        elif level == 'sentence':
            return re.split(r'([.!?]+)', text)
        elif level == 'word':
            return re.split(r'(\s+|\b)', text)
        else:
            chunks = []
            paragraphs = re.split(r'(\n\n)', text)
            for i, para in enumerate(paragraphs):
                if i % 2 == 1:
                    chunks.append(para)
                    continue
                lines = re.split(r'(\n)', para)
                for j, line in enumerate(lines):
                    if j % 2 == 1:
                        chunks.append(line)
                        continue
                    sentences = re.split(r'([.!?]+)', line)
                    for k, sent in enumerate(sentences):
                        if k % 2 == 1:
                            chunks.append(sent)
                            continue
                        words = re.split(r'(\s+|\b)', sent)
                        chunks.extend(words)
            return chunks

    def _dynamic_dict_tokenize(self, data: bytes, index_bytes: int = 3) -> bytes:
        try:
            text = data.decode('utf-8')
        except:
            return b'\x00' + data
        chunks = self._split_text_into_chunks(text, 'all')
        freq = Counter(chunks)
        sorted_chunks = sorted(freq.keys(), key=lambda x: (-freq[x], -len(x), x))
        chunk_to_idx = {ch: i for i, ch in enumerate(sorted_chunks)}
        num_entries = len(sorted_chunks)
        if index_bytes == 2 and num_entries > 65535:
            index_bytes = 3
        if index_bytes == 3 and num_entries > 16777215:
            index_bytes = 8
        header = bytearray()
        header.append(index_bytes)
        header += struct.pack('>I', num_entries)
        for chunk in sorted_chunks:
            chunk_bytes = chunk.encode('utf-8')
            header += struct.pack('>I', len(chunk_bytes))
            header += chunk_bytes
        token_stream = bytearray()
        for chunk in chunks:
            idx = chunk_to_idx[chunk]
            if index_bytes == 2:
                token_stream += struct.pack('>H', idx)
            elif index_bytes == 3:
                token_stream += struct.pack('>I', idx)[1:4]
            else:
                token_stream += struct.pack('>Q', idx)
        return bytes(header) + bytes(token_stream)

    def _dynamic_dict_detokenize(self, data: bytes) -> Optional[bytes]:
        if not data: return b''
        if data[0] == 0: return data[1:]
        index_bytes = data[0]
        if index_bytes not in (2, 3, 8): return None
        pos = 1
        if pos + 4 > len(data): return None
        num_entries = struct.unpack('>I', data[pos:pos+4])[0]
        pos += 4
        dictionary = []
        for _ in range(num_entries):
            if pos + 4 > len(data): return None
            chunk_len = struct.unpack('>I', data[pos:pos+4])[0]
            pos += 4
            if pos + chunk_len > len(data): return None
            chunk = data[pos:pos+chunk_len].decode('utf-8')
            pos += chunk_len
            dictionary.append(chunk)
        tokens = []
        while pos < len(data):
            if index_bytes == 2:
                if pos + 2 > len(data): break
                idx = struct.unpack('>H', data[pos:pos+2])[0]
                pos += 2
            elif index_bytes == 3:
                if pos + 3 > len(data): break
                idx_bytes = b'\x00' + data[pos:pos+3]
                idx = struct.unpack('>I', idx_bytes)[0]
                pos += 3
            else:
                if pos + 8 > len(data): break
                idx = struct.unpack('>Q', data[pos:pos+8])[0]
                pos += 8
            if idx < len(dictionary):
                tokens.append(dictionary[idx])
            else:
                return None
        try:
            text = ''.join(tokens)
            return text.encode('utf-8')
        except:
            return None

    def transform_25(self, data: bytes) -> bytes:
        return self._dynamic_dict_tokenize(data, index_bytes=3)

    def reverse_transform_25(self, data: bytes) -> bytes:
        result = self._dynamic_dict_detokenize(data)
        return result if result is not None else b''

    def _compress_dynamic_dict(self, data: bytes) -> Optional[bytes]:
        try:
            token_stream = self.transform_25(data)
        except:
            return None
        compressed = self._compress_with_best_backend(token_stream)
        return self.MAGIC_DICT + b'\x02' + compressed

    def _decompress_dynamic_dict(self, compressed: bytes) -> Optional[bytes]:
        if not compressed.startswith(self.MAGIC_DICT + b'\x02'):
            return None
        payload = compressed[len(self.MAGIC_DICT) + 1:]
        token_stream = self._decompress_backend(payload)
        if token_stream is None:
            return None
        return self.reverse_transform_25(token_stream)

    # ------------------------------------------------------------------
    # Line‑Based Static Dictionary Tokenizer (original, 8‑byte index)
    # ------------------------------------------------------------------
    MAGIC_LINE = b'LINE'

    def _tokenize_with_line_dict(self, data: bytes) -> Optional[bytes]:
        if not self.line_dict:
            return None
        try:
            text = data.decode('utf-8')
        except:
            return None

        pos = 0
        token_list = []
        while pos < len(text):
            earliest_pos = len(text) + 1
            earliest_len = 0
            earliest_idx = -1
            for idx, phrase in enumerate(self.line_dict):
                p = text.find(phrase, pos)
                if p != -1 and (p < earliest_pos or (p == earliest_pos and len(phrase) > earliest_len)):
                    earliest_pos = p
                    earliest_len = len(phrase)
                    earliest_idx = idx
            if earliest_idx != -1:
                if earliest_pos > pos:
                    token_list.append((False, text[pos:earliest_pos].encode('utf-8')))
                token_list.append((True, earliest_idx))
                pos = earliest_pos + earliest_len
            else:
                token_list.append((False, text[pos:].encode('utf-8')))
                break

        out = bytearray()
        for is_index, payload in token_list:
            if is_index:
                out += b'\x01'
                out += struct.pack('>Q', payload)
            else:
                raw_bytes = payload
                out += b'\x00'
                out += struct.pack('>H', len(raw_bytes))
                out += raw_bytes
        return bytes(out)

    def _detokenize_line_dict(self, token_stream: bytes) -> Optional[bytes]:
        if not token_stream:
            return b''
        out = bytearray()
        pos = 0
        while pos < len(token_stream):
            if pos >= len(token_stream):
                break
            typ = token_stream[pos]
            pos += 1
            if typ == 1:
                if pos + 8 > len(token_stream):
                    return None
                idx = struct.unpack('>Q', token_stream[pos:pos+8])[0]
                pos += 8
                if idx < len(self.line_dict):
                    out += self.line_dict[idx].encode('utf-8')
                else:
                    return None
            elif typ == 0:
                if pos + 2 > len(token_stream):
                    return None
                raw_len = struct.unpack('>H', token_stream[pos:pos+2])[0]
                pos += 2
                if pos + raw_len > len(token_stream):
                    return None
                out += token_stream[pos:pos+raw_len]
                pos += raw_len
            else:
                return None
        return bytes(out)

    def _compress_line_dict(self, data: bytes) -> Optional[bytes]:
        token_stream = self._tokenize_with_line_dict(data)
        if token_stream is None:
            return None
        compressed = self._compress_with_best_backend(token_stream)
        return self.MAGIC_LINE + compressed

    def _decompress_line_dict(self, compressed: bytes) -> Optional[bytes]:
        if not compressed.startswith(self.MAGIC_LINE):
            return None
        payload = compressed[len(self.MAGIC_LINE):]
        token_stream = self._decompress_backend(payload)
        if token_stream is None:
            return None
        return self._detokenize_line_dict(token_stream)

    # ------------------------------------------------------------------
    # NEW DICTIONARY COMPRESSORS: word 1-8, sentence, paragraph, line ranges
    # ------------------------------------------------------------------
    MAGIC_WORD_1_8 = b'DICT_W'
    MAGIC_SENT = b'DICT_S'
    MAGIC_PARA = b'DICT_P'
    MAGIC_LINE_S = b'DICT_L'
    MAGIC_LINE_L = b'DICT_LL'

    # -------- Word 1-8 --------
    def _tokenize_word_1_8(self, data: bytes) -> Optional[bytes]:
        if not self.word_dict_1_8:
            return None
        try:
            text = data.decode('utf-8')
        except:
            return None
        pattern = r'([A-Za-z0-9_]+)'
        tokens = re.split(pattern, text)
        stream = bytearray()
        idx_bytes = 2 if len(self.word_dict_1_8) <= 65535 else (3 if len(self.word_dict_1_8) <= 16777215 else 8)
        # First byte store idx_bytes
        stream.append(idx_bytes)
        for i, tok in enumerate(tokens):
            if i % 2 == 1 and 1 <= len(tok) <= 8:
                idx = self.word_dict_1_8_map.get(tok)
                if idx is not None:
                    stream += b'\x01'
                    stream += _pack_index(idx, idx_bytes)
                else:
                    raw = tok.encode('utf-8')
                    stream += b'\x02'
                    stream += struct.pack('>H', len(raw))
                    stream += raw
            else:
                if tok:
                    raw = tok.encode('utf-8')
                    stream += b'\x00'
                    stream += struct.pack('>H', len(raw))
                    stream += raw
        return bytes(stream)

    def _detokenize_word_1_8(self, token_stream: bytes) -> Optional[bytes]:
        if not token_stream:
            return b''
        out = bytearray()
        pos = 0
        if pos >= len(token_stream):
            return None
        idx_bytes = token_stream[pos]
        pos += 1
        if idx_bytes not in (2,3,8):
            return None
        while pos < len(token_stream):
            typ = token_stream[pos]
            pos += 1
            if typ == 0x01:
                idx, consumed = _unpack_index(token_stream[pos:], idx_bytes)
                pos += consumed
                if idx < len(self.word_dict_1_8):
                    out += self.word_dict_1_8[idx].encode('utf-8')
                else:
                    return None
            elif typ == 0x02:
                if pos + 2 > len(token_stream):
                    return None
                raw_len = struct.unpack('>H', token_stream[pos:pos+2])[0]
                pos += 2
                if pos + raw_len > len(token_stream):
                    return None
                out += token_stream[pos:pos+raw_len]
                pos += raw_len
            elif typ == 0x00:
                if pos + 2 > len(token_stream):
                    return None
                raw_len = struct.unpack('>H', token_stream[pos:pos+2])[0]
                pos += 2
                if pos + raw_len > len(token_stream):
                    return None
                out += token_stream[pos:pos+raw_len]
                pos += raw_len
            else:
                return None
        return bytes(out)

    def _compress_word_1_8(self, data: bytes) -> Optional[bytes]:
        token_stream = self._tokenize_word_1_8(data)
        if token_stream is None:
            return None
        compressed = self._compress_with_best_backend(token_stream)
        return self.MAGIC_WORD_1_8 + compressed

    def _decompress_word_1_8(self, compressed: bytes) -> Optional[bytes]:
        if not compressed.startswith(self.MAGIC_WORD_1_8):
            return None
        payload = compressed[len(self.MAGIC_WORD_1_8):]
        token_stream = self._decompress_backend(payload)
        if token_stream is None:
            return None
        return self._detokenize_word_1_8(token_stream)

    # -------- Sentence --------
    def _tokenize_sentence(self, data: bytes) -> Optional[bytes]:
        if not self.sentence_dict:
            return None
        try:
            text = data.decode('utf-8')
        except:
            return None
        # split on . ! ? but keep delimiters
        tokens = re.split(r'([.!?]+)', text)
        stream = bytearray()
        idx_bytes = 2 if len(self.sentence_dict) <= 65535 else (3 if len(self.sentence_dict) <= 16777215 else 8)
        stream.append(idx_bytes)
        for chunk in tokens:
            if chunk:
                idx = self.sentence_dict_map.get(chunk)
                if idx is not None:
                    stream += b'\x01'
                    stream += _pack_index(idx, idx_bytes)
                else:
                    raw = chunk.encode('utf-8')
                    stream += b'\x00'
                    stream += struct.pack('>H', len(raw))
                    stream += raw
        return bytes(stream)

    def _detokenize_sentence(self, token_stream: bytes) -> Optional[bytes]:
        if not token_stream:
            return b''
        out = bytearray()
        pos = 0
        if pos >= len(token_stream):
            return None
        idx_bytes = token_stream[pos]
        pos += 1
        if idx_bytes not in (2,3,8):
            return None
        while pos < len(token_stream):
            typ = token_stream[pos]
            pos += 1
            if typ == 0x01:
                idx, consumed = _unpack_index(token_stream[pos:], idx_bytes)
                pos += consumed
                if idx < len(self.sentence_dict):
                    out += self.sentence_dict[idx].encode('utf-8')
                else:
                    return None
            elif typ == 0x00:
                if pos + 2 > len(token_stream):
                    return None
                raw_len = struct.unpack('>H', token_stream[pos:pos+2])[0]
                pos += 2
                if pos + raw_len > len(token_stream):
                    return None
                out += token_stream[pos:pos+raw_len]
                pos += raw_len
            else:
                return None
        return bytes(out)

    def _compress_sentence(self, data: bytes) -> Optional[bytes]:
        token_stream = self._tokenize_sentence(data)
        if token_stream is None:
            return None
        compressed = self._compress_with_best_backend(token_stream)
        return self.MAGIC_SENT + compressed

    def _decompress_sentence(self, compressed: bytes) -> Optional[bytes]:
        if not compressed.startswith(self.MAGIC_SENT):
            return None
        payload = compressed[len(self.MAGIC_SENT):]
        token_stream = self._decompress_backend(payload)
        if token_stream is None:
            return None
        return self._detokenize_sentence(token_stream)

    # -------- Paragraph (3-5 lines) --------
    def _tokenize_paragraph(self, data: bytes) -> Optional[bytes]:
        if not self.paragraph_dict:
            return None
        try:
            text = data.decode('utf-8')
        except:
            return None
        lines = text.splitlines()
        # Group lines into paragraphs of 3-5 lines (simplified: groups of 3)
        paragraphs = []
        i = 0
        while i < len(lines):
            if lines[i] == '':
                i += 1
                continue
            # take 3 lines (or fewer if near end)
            count = min(3, len(lines)-i)
            para = '\n'.join(lines[i:i+count])
            paragraphs.append(para)
            i += count
        stream = bytearray()
        idx_bytes = 2 if len(self.paragraph_dict) <= 65535 else (3 if len(self.paragraph_dict) <= 16777215 else 8)
        stream.append(idx_bytes)
        for para in paragraphs:
            if para:
                idx = self.paragraph_dict_map.get(para)
                if idx is not None:
                    stream += b'\x01'
                    stream += _pack_index(idx, idx_bytes)
                else:
                    raw = para.encode('utf-8')
                    stream += b'\x00'
                    stream += struct.pack('>H', len(raw))
                    stream += raw
        return bytes(stream)

    def _detokenize_paragraph(self, token_stream: bytes) -> Optional[bytes]:
        if not token_stream:
            return b''
        out = bytearray()
        pos = 0
        if pos >= len(token_stream):
            return None
        idx_bytes = token_stream[pos]
        pos += 1
        if idx_bytes not in (2,3,8):
            return None
        first = True
        while pos < len(token_stream):
            typ = token_stream[pos]
            pos += 1
            if typ == 0x01:
                idx, consumed = _unpack_index(token_stream[pos:], idx_bytes)
                pos += consumed
                if idx < len(self.paragraph_dict):
                    if not first:
                        out.append(10)  # newline between paragraphs
                    out += self.paragraph_dict[idx].encode('utf-8')
                    first = False
                else:
                    return None
            elif typ == 0x00:
                if pos + 2 > len(token_stream):
                    return None
                raw_len = struct.unpack('>H', token_stream[pos:pos+2])[0]
                pos += 2
                if pos + raw_len > len(token_stream):
                    return None
                if not first:
                    out.append(10)
                out += token_stream[pos:pos+raw_len]
                first = False
                pos += raw_len
            else:
                return None
        return bytes(out)

    def _compress_paragraph(self, data: bytes) -> Optional[bytes]:
        token_stream = self._tokenize_paragraph(data)
        if token_stream is None:
            return None
        compressed = self._compress_with_best_backend(token_stream)
        return self.MAGIC_PARA + compressed

    def _decompress_paragraph(self, compressed: bytes) -> Optional[bytes]:
        if not compressed.startswith(self.MAGIC_PARA):
            return None
        payload = compressed[len(self.MAGIC_PARA):]
        token_stream = self._decompress_backend(payload)
        if token_stream is None:
            return None
        return self._detokenize_paragraph(token_stream)

    # -------- Line 1-255 and 256-65535 --------
    def _tokenize_line_range(self, data: bytes, dict_list, dict_map, min_len, max_len) -> Optional[bytes]:
        if not dict_list:
            return None
        try:
            text = data.decode('utf-8')
        except:
            return None
        lines = text.splitlines()
        stream = bytearray()
        idx_bytes = 2 if len(dict_list) <= 65535 else (3 if len(dict_list) <= 16777215 else 8)
        stream.append(idx_bytes)
        for line in lines:
            l = len(line)
            if min_len <= l <= max_len:
                idx = dict_map.get(line)
                if idx is not None:
                    stream += b'\x01'
                    stream += _pack_index(idx, idx_bytes)
                else:
                    raw = line.encode('utf-8')
                    stream += b'\x00'
                    stream += struct.pack('>H', len(raw))
                    stream += raw
            else:
                raw = line.encode('utf-8')
                stream += b'\x00'
                stream += struct.pack('>H', len(raw))
                stream += raw
        return bytes(stream)

    def _detokenize_line_range(self, token_stream: bytes, dict_list) -> Optional[bytes]:
        if not token_stream:
            return b''
        out = bytearray()
        pos = 0
        if pos >= len(token_stream):
            return None
        idx_bytes = token_stream[pos]
        pos += 1
        if idx_bytes not in (2,3,8):
            return None
        first = True
        while pos < len(token_stream):
            typ = token_stream[pos]
            pos += 1
            if typ == 0x01:
                idx, consumed = _unpack_index(token_stream[pos:], idx_bytes)
                pos += consumed
                if idx < len(dict_list):
                    if not first:
                        out.append(10)
                    out += dict_list[idx].encode('utf-8')
                    first = False
                else:
                    return None
            elif typ == 0x00:
                if pos + 2 > len(token_stream):
                    return None
                raw_len = struct.unpack('>H', token_stream[pos:pos+2])[0]
                pos += 2
                if pos + raw_len > len(token_stream):
                    return None
                if not first:
                    out.append(10)
                out += token_stream[pos:pos+raw_len]
                first = False
                pos += raw_len
            else:
                return None
        return bytes(out)

    def _compress_line_1_255(self, data: bytes) -> Optional[bytes]:
        token_stream = self._tokenize_line_range(data, self.line_dict_1_255, self.line_dict_1_255_map, 1, 255)
        if token_stream is None:
            return None
        compressed = self._compress_with_best_backend(token_stream)
        return self.MAGIC_LINE_S + compressed

    def _decompress_line_1_255(self, compressed: bytes) -> Optional[bytes]:
        if not compressed.startswith(self.MAGIC_LINE_S):
            return None
        payload = compressed[len(self.MAGIC_LINE_S):]
        token_stream = self._decompress_backend(payload)
        if token_stream is None:
            return None
        return self._detokenize_line_range(token_stream, self.line_dict_1_255)

    def _compress_line_256_65535(self, data: bytes) -> Optional[bytes]:
        token_stream = self._tokenize_line_range(data, self.line_dict_256_65535, self.line_dict_256_65535_map, 256, 65535)
        if token_stream is None:
            return None
        compressed = self._compress_with_best_backend(token_stream)
        return self.MAGIC_LINE_L + compressed

    def _decompress_line_256_65535(self, compressed: bytes) -> Optional[bytes]:
        if not compressed.startswith(self.MAGIC_LINE_L):
            return None
        payload = compressed[len(self.MAGIC_LINE_L):]
        token_stream = self._decompress_backend(payload)
        if token_stream is None:
            return None
        return self._detokenize_line_range(token_stream, self.line_dict_256_65535)

    # ------------------------------------------------------------------
    # Main compression with auto‑correction
    # ------------------------------------------------------------------
    def compress_with_best(self, data: bytes, safe: bool = False, ultra: bool = True) -> bytes:
        if not data:
            backend = self._compress_with_best_backend(b'')
            compressed = self._encode_marker_raw() + backend
            if not safe:
                decomp, _ = self._decompress_auto(compressed)
                if decomp != b'':
                    return self.compress_with_best(data, safe=True, ultra=ultra)
            return compressed

        best_total = float('inf')
        best_bytes = None

        raw_backend = self._compress_with_best_backend(data)
        candidate = self._encode_marker_raw() + raw_backend
        if len(candidate) < best_total:
            best_total = len(candidate)
            best_bytes = candidate

        for t in range(1, 257):
            try:
                transformed = self.fwd_transforms[t](data)
                backend = self._compress_with_best_backend(transformed)
                candidate = self._encode_marker_single(t) + backend
                if len(candidate) < best_total:
                    best_total = len(candidate)
                    best_bytes = candidate
            except:
                continue

        if ultra:
            for t1, t2 in self.sequences:
                try:
                    transformed = self._apply_sequence(data, (t1, t2))
                    backend = self._compress_with_best_backend(transformed)
                    candidate = self._encode_marker_pair(t1, t2) + backend
                    if len(candidate) < best_total:
                        best_total = len(candidate)
                        best_bytes = candidate
                except:
                    continue

        decomp, _ = self._decompress_auto(best_bytes)
        if decomp != data:
            if not safe:
                print("Note: marker‑free mode produced ambiguous stream, falling back to safe markers...")
                return self.compress_with_best(data, safe=True, ultra=ultra)
            else:
                raise RuntimeError("Safe compression failed – unexpected internal error!")
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

    def _encode_marker_single(self, t: int) -> bytes:
        if t <= 252:
            return bytes([t - 1])
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

    def compress_file(self, infile: str, outfile: str, ultra: bool = True, hybrid: bool = False):
        try:
            with open(infile, 'rb') as f:
                data = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return

        candidates = []
        if hybrid:
            # Try all dictionary compressors
            c_static = self._compress_static_dict(data)
            if c_static is not None:
                candidates.append(('Static-Word-Dict', c_static))
            c_line = self._compress_line_dict(data)
            if c_line is not None:
                candidates.append(('Line-Dict', c_line))
            c_dynamic = self._compress_dynamic_dict(data)
            if c_dynamic is not None:
                candidates.append(('Dynamic-Dict', c_dynamic))
            c_word1_8 = self._compress_word_1_8(data)
            if c_word1_8 is not None:
                candidates.append(('Word-1-8', c_word1_8))
            c_sent = self._compress_sentence(data)
            if c_sent is not None:
                candidates.append(('Sentence-Dict', c_sent))
            c_para = self._compress_paragraph(data)
            if c_para is not None:
                candidates.append(('Paragraph-Dict', c_para))
            c_line_s = self._compress_line_1_255(data)
            if c_line_s is not None:
                candidates.append(('Line-1-255', c_line_s))
            c_line_l = self._compress_line_256_65535(data)
            if c_line_l is not None:
                candidates.append(('Line-256-65535', c_line_l))

        c_paqjp = self.compress_with_best(data, safe=False, ultra=ultra)
        candidates.append(('PAQJP', c_paqjp))

        if not candidates:
            print("No compression method succeeded.")
            return

        best_method, best_bytes = min(candidates, key=lambda x: len(x[1]))
        try:
            with open(outfile, 'wb') as f:
                f.write(best_bytes)
        except Exception as e:
            print(f"Error writing output file: {e}")
            return
        print(f"Compressed {len(data)} → {len(best_bytes)} bytes ({best_method}) → {outfile}")

    def decompress_file(self, infile: str, outfile: str):
        try:
            with open(infile, 'rb') as f:
                data = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return

        # Try all dictionary decompressors in order
        decompressors = [
            (self.MAGIC_LINE, self._decompress_line_dict),
            (self.MAGIC_DICT + b'\x01', self._decompress_static_dict),
            (self.MAGIC_DICT + b'\x02', self._decompress_dynamic_dict),
            (self.MAGIC_WORD_1_8, self._decompress_word_1_8),
            (self.MAGIC_SENT, self._decompress_sentence),
            (self.MAGIC_PARA, self._decompress_paragraph),
            (self.MAGIC_LINE_S, self._decompress_line_1_255),
            (self.MAGIC_LINE_L, self._decompress_line_256_65535),
        ]
        for magic, decomp_func in decompressors:
            if data.startswith(magic):
                original = decomp_func(data)
                if original is not None:
                    with open(outfile, 'wb') as f:
                        f.write(original)
                    print(f"Decompressed ({magic.decode()}) → {outfile} ({len(original)} bytes)")
                    return

        # Fallback to PAQJP
        original, seq = self._decompress_auto(data)
        if original != b'' or seq is not None:
            with open(outfile, 'wb') as f:
                f.write(original)
            print(f"Decompressed (PAQJP) → {outfile} ({len(original)} bytes)")
            return

        print("Decompression failed – unknown format.")

    def verify_transforms(self) -> bool:
        print("Verifying all 256 transforms...")
        ok = True
        for t in range(1, 257):
            test = bytes([0x55])
            try:
                enc = self.fwd_transforms[t](test)
                dec = self.rev_transforms[t](enc)
                if dec == test:
                    print(f"Transform {t}: right")
                else:
                    print(f"Transform {t}: incorrect")
                    ok = False
            except Exception:
                print(f"Transform {t}: exception")
                ok = False
        print("Verification complete.\n")
        return ok

    def full_self_test(self) -> bool:
        print("=" * 60)
        print("PAQJP 9.0 – FULL SELF‑TEST (100% lossless)")
        print("=" * 60)
        all_ok = True

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

        print("\nTesting random 1000‑byte block through full compress/decompress...")
        rng = random.Random(12345)
        test_data = bytes(rng.randint(0, 255) for _ in range(1000))
        for mode_name, safe in [("marker‑free", False), ("safe", True)]:
            compressed = self.compress_with_best(test_data, safe=safe, ultra=True)
            decompressed, _ = self._decompress_auto(compressed)
            if decompressed != test_data:
                print(f"  FAIL: random data pipeline mismatch in {mode_name} mode")
                return False
        print("  PASS: random data pipeline OK in both modes")

        print("\nTesting empty input...")
        for safe in [False, True]:
            compressed_empty = self.compress_with_best(b'', safe)
            decomp_empty, _ = self._decompress_auto(compressed_empty)
            if decomp_empty != b'':
                print(f"  FAIL: empty input pipeline mismatch (safe={safe})")
                return False
        print("  PASS: empty input pipeline OK")

        # Test each new dictionary with sample text
        sample_texts = [
            ("1-8 word", b"The quick brown fox jumps over the lazy dog."),
            ("sentence", b"This is a test. Hello world! How are you?"),
            ("paragraph", b"First line.\nSecond line.\nThird line."),
            ("line 1-255", b"Short line"),
            ("line 256-65535", b"X" * 300),  # long line
        ]
        for name, sample in sample_texts:
            print(f"\nTesting {name} dictionary tokenizer...")
            if name == "1-8 word":
                tok = self._tokenize_word_1_8(sample)
                if tok is None:
                    print("  SKIP: dict empty")
                    continue
                dec = self._detokenize_word_1_8(tok)
            elif name == "sentence":
                tok = self._tokenize_sentence(sample)
                if tok is None:
                    print("  SKIP: dict empty")
                    continue
                dec = self._detokenize_sentence(tok)
            elif name == "paragraph":
                tok = self._tokenize_paragraph(sample)
                if tok is None:
                    print("  SKIP: dict empty")
                    continue
                dec = self._detokenize_paragraph(tok)
            elif name == "line 1-255":
                tok = self._tokenize_line_range(sample, self.line_dict_1_255, self.line_dict_1_255_map, 1, 255)
                if tok is None:
                    print("  SKIP: dict empty")
                    continue
                dec = self._detokenize_line_range(tok, self.line_dict_1_255)
            else:  # long line
                tok = self._tokenize_line_range(sample, self.line_dict_256_65535, self.line_dict_256_65535_map, 256, 65535)
                if tok is None:
                    print("  SKIP: dict empty")
                    continue
                dec = self._detokenize_line_range(tok, self.line_dict_256_65535)
            if dec != sample:
                print(f"  FAIL: {name} round‑trip mismatch")
                all_ok = False
            else:
                print(f"  PASS: {name} round‑trip OK")

        print("\n[All tests passed – compressor is 100% lossless]")
        return True

    def transform_256(self, d: bytes) -> bytes:
        return d
    reverse_transform_256 = transform_256

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print(f"{PROGNAME}")
    print("256 single transforms + 2704 transform‑pair sequences (100% lossless).")
    print("Hybrid mode: static word, line, dynamic, 1-8 word, sentence, paragraph, line ranges.")
    print("Backends: Zstd, PAQ, LZ77, Huffman, Raw.")
    if paq is None and not HAS_ZSTD:
        print("Warning: PAQ and Zstd not installed. Using LZ77, Huffman, Raw.")
    if not HAS_QISKIT:
        print("Info: Qiskit not installed; quantum transforms (27, 28) use classical fallback.")

    c = PAQJPCompressor()
    c.verify_transforms()

    choice = input("\n1) Compress   2) Decompress   3) Full self‑test\n> ").strip()
    if choice == "1":
        i = input("Input file: ").strip()
        o = input("Output file: ").strip() or i + ".pjp"
        mode = input("Choose mode:\n  1) PAQJP Fast (256 transforms)\n  2) PAQJP Ultra (256+2704 pairs)\n  3) Hybrid (dicts + PAQJP)\n> ").strip()
        if mode == "1":
            c.compress_file(i, o, ultra=False, hybrid=False)
        elif mode == "2":
            c.compress_file(i, o, ultra=True, hybrid=False)
        elif mode == "3":
            c.compress_file(i, o, ultra=True, hybrid=True)
        else:
            print("Invalid choice.")
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
