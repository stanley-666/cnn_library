#!/usr/bin/bash
set -e  # 一有錯誤就中斷腳本

echo "=== Building C project ==="
make

echo "=== Running C inference ==="
./main

echo "=== Displaying PGM result with Python ==="
python3 ./pytools/display_pgm.py
python3 ./pytools/display_npy.py