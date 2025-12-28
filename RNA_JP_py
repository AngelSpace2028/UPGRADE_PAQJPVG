import os
import zstandard as zstd

print("Zstandard (zstd) Level 22 Compressor/Decompressor")
print("Real high-performance lossless compression")

mode = input("c for compress, e for extract/decompress: ").lower()
if mode not in ["c", "e"]:
    print("Invalid choice")
    raise SystemExit

if mode == "c":
    file_path = input("Enter file path to compress: ")
    if not os.path.exists(file_path):
        print("File not found!")
        raise SystemExit
    output_path = file_path + ".zst"
    with open(file_path, "rb") as f:
        data = f.read()
    cctx = zstd.ZstdCompressor(level=22, threads=os.cpu_count() or 1)
    compressed = cctx.compress(data)
    with open(output_path, "wb") as f:
        f.write(compressed)
    ratio = len(compressed) / len(data) if data else 0
    print(f"Compressed: {file_path} -> {output_path}")
    print(f"Original: {len(data):,} bytes | Compressed: {len(compressed):,} bytes | Ratio: {ratio:.2%}")
else:
    file_path = input("Enter .zst file to extract: ")
    if not os.path.exists(file_path):
        print("File not found!")
        raise SystemExit
    if file_path.endswith(".zst"):
        default_name = file_path[:-4]
    else:
        default_name = file_path + ".decompressed"
    output_path = input(f"Enter output file name (default: {default_name}): ").strip()
    if not output_path:
        output_path = default_name
    with open(file_path, "rb") as f:
        data = f.read()
    dctx = zstd.ZstdDecompressor()
    decompressed = dctx.decompress(data)
    with open(output_path, "wb") as f:
        f.write(decompressed)
    ratio = len(data) / len(decompressed) if decompressed else 0
    print(f"Decompressed: {file_path} -> {output_path}")
    print(f"Compressed: {len(data):,} bytes | Original: {len(decompressed):,} bytes | Ratio: {ratio:.2%}")
    delete = input("\nDelete the compressed file? (y/n): ").lower().strip()
    if delete in ["y", "yes"]:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except OSError as e:
            print(f"Could not delete {file_path}: {e}")
