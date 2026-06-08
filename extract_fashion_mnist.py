import os
import gzip
import shutil

raw_dir = r"./data/FashionMNIST/raw"

files = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

for filename in files:
    src = os.path.join(raw_dir, filename)
    dst = os.path.join(raw_dir, filename[:-3])  # 去掉 .gz

    print(f"Extracting: {src} -> {dst}")
    with gzip.open(src, "rb") as f_in:
        with open(dst, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

print("Done.")