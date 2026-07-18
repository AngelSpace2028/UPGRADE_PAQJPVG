#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PJP – 256 Lossless Transforms + 2704 Transform‑Pair Sequences
+ Hybrid Dictionary Mode + Quantum Transforms + Base64 + 6‑bit Text
+ Transforms 28–30 + .docx transforms 31–32
+ Zaden Block Optimization (Options 9 & 10)
  Option  9: test passes 0..max_passes with a single time limit & block size.
  Option 10: EXHAUSTIVE – test ALL combinations of passes (0..max_passes),
             time limit (1..300 s) and block size (1..256 bytes).
  Both pick the absolute smallest result among Absolute and Zaden variants.
  Zaden magic: 0x33 (2 passes), 0x34 (≤255 passes), 0x35 (any passes).
  MOD: Options 1,2,3,4,8 now EXCLUDE transforms 33,34,35,36.
       Options 9,10 keep the default set.
"""

import math
import random
import decimal
import hashlib
import struct
import re
import os
import urllib.request
import sys
import subprocess
import importlib
import tempfile
import base64
import zipfile
import io
import xml.etree.ElementTree as ET
import time
from typing import Optional, List, Tuple, Dict, Callable, Set
from collections import Counter, defaultdict

# ------------------------------------------------------------------
# Helper: install a single package via pip (silent, auto)
# ------------------------------------------------------------------
def install_package(pkg: str) -> bool:
    print(f"Installing {pkg}...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])
        print(f"Successfully installed {pkg}")
        return True
    except Exception as e:
        print(f"Failed to install {pkg}: {e}")
        return False

# ------------------------------------------------------------------
# 1. Ask about quantum transforms – auto‑install if missing
# ------------------------------------------------------------------
USE_QUANTUM = False
HAS_QISKIT = False

quantum_choice = input("Enable quantum‑inspired transforms (requires Qiskit)? (y/n): ").strip().lower()
if quantum_choice == 'y':
    try:
        from qiskit import QuantumCircuit
        HAS_QISKIT = True
        USE_QUANTUM = True
        print("Quantum transforms ENABLED (Qiskit already installed).")
    except ImportError:
        print("Qiskit not found. Installing automatically...")
        if install_package('qiskit'):
            try:
                from qiskit import QuantumCircuit
                HAS_QISKIT = True
                USE_QUANTUM = True
                print("Quantum transforms ENABLED after automatic installation.")
            except ImportError:
                print("Qiskit installation succeeded but import failed – quantum transforms disabled.")
        else:
            print("Automatic installation failed – quantum transforms disabled.")
else:
    print("Quantum transforms disabled.")

# ------------------------------------------------------------------
# 2. Ask about other optional compression backends (zstandard, paq, etc.)
# ------------------------------------------------------------------
other_choice = input("Install other optional compression backends (zstandard, paq, mpmath, cython, python-docx)? (y/n): ").strip().lower()
if other_choice == 'y':
    for pkg in ['mpmath', 'zstandard', 'cython', 'paq', 'python-docx']:
        try:
            importlib.import_module(pkg)
        except ImportError:
            install_package(pkg)
else:
    print("Skipping other backends.")

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

# ---------- (Re‑import Qiskit if it was just installed) ----------
if USE_QUANTUM and not HAS_QISKIT:
    try:
        from qiskit import QuantumCircuit
        HAS_QISKIT = True
    except ImportError:
        USE_QUANTUM = False
        print("Quantum transforms disabled because Qiskit could not be imported.")

PROGNAME = "PJP"

# ---------- Dictionary configuration ----------
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

def download_and_merge_dictionaries():
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
                    if not w:
                        continue
                    try:
                        decoded = base64.b64decode(w, validate=True)
                        decoded_str = decoded.decode('utf-8')
                        all_words.add(decoded_str)
                    except Exception:
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

# ---------- 6‑bit alphabet for transform 27 (exactly 64 chars) ----------
ALPHABET_6BIT = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"    # 26
    "abcdefghijklmnopqrstuvwxyz"    # 26
    "0123456789"                    # 10
    " \n"                           # 2  (space and newline)
)  # Total = 64
assert len(ALPHABET_6BIT) == 64
CHAR_TO_6BIT = {ch: i for i, ch in enumerate(ALPHABET_6BIT)}
SIXBIT_TO_CHAR = {i: ch for ch, i in CHAR_TO_6BIT.items()}

# ---------- Main Compressor Class ----------
class PJPCompressor:
    def __init__(self):
        download_and_merge_dictionaries()

        self.PI_DIGITS = PI_DIGITS.copy()
        self.seed_tables = self._gen_seed_tables(num=126, size=40, seed=42)
        self.fibonacci = self._gen_fib(100)
        self.PI_STR = "3.14159265358979323846264338327950288419716939937510"

        self._build_transform_maps()
        self.sequences = self._build_pair_sequences()
        self.pair_lookup = {idx: (t1, t2) for idx, (t1, t2) in enumerate(self.sequences)}

        self.static_dict, self.word_to_index = self._load_static_dictionary()
        self.line_dict, self.line_to_index = self._load_line_dictionary()

        # Precompute quantum permutations if enabled
        if USE_QUANTUM and HAS_QISKIT:
            self._precompute_quantum_transforms()

    # ... (all previous methods unchanged, only the ones we modify are shown below)
    # For brevity, I'll include the full class but only the changed methods will be highlighted.
    # The full code is available in the previous answer; I'll replace the relevant parts.

    # (The entire class definition from the previous full code is inserted here.
    #  To keep the answer length manageable, I'll only rewrite the key methods.)

    def _build_transform_maps(self):
        # ... same as before ...
        pass

    # --------------------------------------------------
    # compress_with_best – now accepts an exclusion set
    # --------------------------------------------------
    def compress_with_best(self, data: bytes, safe: bool = False, ultra: bool = True,
                           include_28: bool = False, include_29: bool = False,
                           include_30: bool = False,
                           exclude_transforms: Optional[Set[int]] = None) -> bytes:
        if not data:
            backend = self._compress_backend(b'', safe)
            compressed = self._encode_marker_raw() + backend
            if not safe:
                decomp, _ = self._decompress_auto(compressed)
                if decomp != b'':
                    return self.compress_with_best(data, safe=True, ultra=ultra,
                                                   include_28=include_28, include_29=include_29,
                                                   include_30=include_30,
                                                   exclude_transforms=exclude_transforms)
            return compressed

        if exclude_transforms is None:
            exclude_transforms = set()

        best_total = float('inf')
        best_bytes = None

        # single transforms 1..256 minus excluded ones
        single_transforms = [t for t in range(1, 257) if t not in exclude_transforms]

        # filter out 28-30 if disallowed
        if not include_28:
            single_transforms = [t for t in single_transforms if t != 28]
        if not include_29:
            single_transforms = [t for t in single_transforms if t != 29]
        if not include_30:
            single_transforms = [t for t in single_transforms if t != 30]

        # quantum transforms
        if USE_QUANTUM and HAS_QISKIT:
            fast_quantum = range(257, 266)
            single_transforms.extend(fast_quantum)
            if ultra:
                single_transforms.extend(range(266, 283))

        # pairs – use the original sequences (unchanged)
        allowed_pairs = self.sequences
        if not include_28:
            allowed_pairs = [seq for seq in allowed_pairs if 28 not in seq]
        if not include_29:
            allowed_pairs = [seq for seq in allowed_pairs if 29 not in seq]
        if not include_30:
            allowed_pairs = [seq for seq in allowed_pairs if 30 not in seq]

        # raw
        raw_backend = self._compress_backend(data, safe)
        candidate = self._encode_marker_raw() + raw_backend
        if len(candidate) < best_total:
            best_total = len(candidate)
            best_bytes = candidate

        # singles
        for t in single_transforms:
            try:
                transformed = self.fwd_transforms[t](data)
                backend = self._compress_backend(transformed, safe)
                candidate = self._encode_marker_single(t) + backend
                if len(candidate) < best_total:
                    best_total = len(candidate)
                    best_bytes = candidate
            except:
                continue

        # pairs
        if ultra:
            for t1, t2 in allowed_pairs:
                try:
                    transformed = self._apply_sequence(data, (t1, t2))
                    backend = self._compress_backend(transformed, safe)
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
                return self.compress_with_best(data, safe=True, ultra=ultra,
                                               include_28=include_28, include_29=include_29,
                                               include_30=include_30,
                                               exclude_transforms=exclude_transforms)
            else:
                raise RuntimeError("Safe compression failed – unexpected internal error!")
        return best_bytes

    # --------------------------------------------------
    # compress_file – pass exclude_transforms
    # --------------------------------------------------
    def compress_file(self, infile: str, outfile: str, ultra: bool = True, hybrid: bool = False,
                      include_28: bool = False, include_29: bool = False,
                      include_30: bool = False,
                      exclude_transforms: Optional[Set[int]] = None):
        try:
            with open(infile, 'rb') as f:
                data = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return

        if hybrid:
            best_bytes, method = self._compress_hybrid_bytes(data, exclude_transforms=exclude_transforms)
        else:
            candidates = []
            c_pjp = self.compress_with_best(data, safe=False, ultra=ultra,
                                            include_28=include_28, include_29=include_29,
                                            include_30=include_30,
                                            exclude_transforms=exclude_transforms)
            candidates.append(('PJP', c_pjp))
            best_method, best_bytes = min(candidates, key=lambda x: len(x[1]))
            method = best_method

        with open(outfile, 'wb') as f:
            f.write(best_bytes)
        print(f"Compressed {len(data)} → {len(best_bytes)} bytes ({method}) → {outfile}")

    # --------------------------------------------------
    # _compress_hybrid_bytes – propagate exclude
    # --------------------------------------------------
    def _compress_hybrid_bytes(self, data: bytes, exclude_transforms: Optional[Set[int]] = None) -> Tuple[bytes, str]:
        candidates = []
        c_static = self._compress_static_dict(data)
        if c_static is not None:
            candidates.append(('Static-Word-Dict', c_static))
        c_line = self._compress_line_dict(data)
        if c_line is not None:
            candidates.append(('Line-Dict', c_line))
        c_dynamic = self._compress_dynamic_dict(data)
        if c_dynamic is not None:
            candidates.append(('Dynamic-Dict', c_dynamic))
        c_pjp = self.compress_with_best(data, safe=False, ultra=True,
                                        include_28=True, include_29=True, include_30=True,
                                        exclude_transforms=exclude_transforms)
        candidates.append(('PJP-Absolute', c_pjp))
        best_method, best_bytes = min(candidates, key=lambda x: len(x[1]))
        return best_bytes, best_method

    # --------------------------------------------------
    # Option 9 (unchanged, uses default exclude=None)
    # --------------------------------------------------
    def compress_with_best_plus_block(self, infile: str, outfile: str,
                                      block_size: int = 256,
                                      quantum_boost: bool = False,
                                      time_limit_per_block: float = 60.0,
                                      max_passes: int = 2):
        # ... identical to previous version, calls compress_with_best without exclude_transforms ...
        pass  # (full code would be here)

    # --------------------------------------------------
    # Option 10 (exhaustive) – also uses default exclude
    # --------------------------------------------------
    def compress_with_best_exhaustive(self, infile: str, outfile: str,
                                      max_passes: int = 10,
                                      quantum_boost: bool = False):
        # ... calls compress_with_best without exclude ...
        pass

    # (remaining methods unchanged)

# ------------------------------------------------------------
# Main menu – passes the exclude set for 1/2/3/4/8
# ------------------------------------------------------------
def main():
    # ... same as before, but in the choices we pass exclude_transforms:
    EXCLUDE_33_36 = {33, 34, 35, 36}

    # inside menu:
    if choice == 1:
        c.compress_file(i, o, ultra=False, hybrid=False,
                        include_28=False, include_29=False, include_30=False,
                        exclude_transforms=EXCLUDE_33_36)
    elif choice == 2:
        c.compress_file(i, o, ultra=True, hybrid=False,
                        include_28=False, include_29=False, include_30=False,
                        exclude_transforms=EXCLUDE_33_36)
    elif choice == 3:
        c.compress_file(i, o, ultra=True, hybrid=True,
                        include_28=False, include_29=False, include_30=False,
                        exclude_transforms=EXCLUDE_33_36)
    elif choice == 4:
        c.compress_file(i, o, ultra=True, hybrid=True,
                        include_28=True, include_29=True, include_30=True,
                        exclude_transforms=EXCLUDE_33_36)
    elif choice == 8:
        c.compress_file(i, o, ultra=False, hybrid=False,
                        include_28=False, include_29=False, include_30=False,
                        exclude_transforms=EXCLUDE_33_36)
    # Options 5,6,7,9,10 unchanged

    # ...

if __name__ == "__main__":
    main()
