import argparse
import os
import shutil
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Copy kernel from hw1 to hw2 directory")
    parser.add_argument(
        "--hw1-dir",
        type=str,
        required=True,
        help="Path to the hw1 directory"
    )
    parser.add_argument(
        "--hw2-dir",
        type=str,
        required=True,
        help="Path to the hw2 directory"
    )

    args = parser.parse_args()

    hw1_dir = os.path.abspath(args.hw1_dir)
    hw2_dir = os.path.abspath(args.hw2_dir)

    if not os.path.isdir(hw1_dir):
        print(f"Err: {hw1_dir} is not a valid directory", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(hw2_dir):
        print(f"Err: {hw2_dir} is not a valid directory", file=sys.stderr)
        sys.exit(1)

    src_file = os.path.join(hw1_dir, "src/combine.cu")
    dst_file = os.path.join(hw2_dir, "src/combine.cu")
    so_file = os.path.join(hw2_dir, "minitorch/cuda_kernels/combine.so")

    if not os.path.isfile(src_file):
        print(f"Error: {src_file} does not exist", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    os.makedirs(os.path.dirname(so_file), exist_ok=True)
    shutil.copy2(src_file, dst_file)


    commands = [
        ["nvcc", "-o", so_file, "--shared", dst_file, "-Xcompiler", "-fPIC"]
    ]

    for cmd in commands:
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
