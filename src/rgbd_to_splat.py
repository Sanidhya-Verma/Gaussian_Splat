"""
src/rgbd_to_splat.py
Master runner script: Loads any RGB-D image pair (or multi-view JSON),
passes it through engine.py, and writes the .ply file using writer.py.
"""

import os
import json
import argparse
import numpy as np
from PIL import Image

from engine import extract_surfels, build_map
from writer import save_splat_ply


def main():
    parser = argparse.ArgumentParser(description="Convert any RGB-D image or multi-view rig into a 3DGS .ply file")
    parser.add_argument("--rgb", type=str, help="Path to RGB image (.png or .jpg)")
    parser.add_argument("--depth", type=str, help="Path to 16-bit Depth image (.png)")
    parser.add_argument("--output", type=str, default="output.ply", help="Output .ply filename")
    parser.add_argument("--fx", type=float, default=525.0, help="Camera focal length fx")
    parser.add_argument("--fy", type=float, default=525.0, help="Camera focal length fy")
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="Divisor to convert depth pixels to meters (1000 for mm)")
    args = parser.parse_args()

    # 1. Load the 2D images from disk into NumPy arrays
    rgb_img = np.array(Image.open(args.rgb).convert("RGB"), dtype=np.float32)
    depth_meters = np.array(Image.open(args.depth), dtype=np.float32) / args.depth_scale

    rows, cols = depth_meters.shape
    intrinsics = (args.fx, args.fy, cols / 2.0, rows / 2.0)

    # 2. Send arrays into engine.py -> returns the `surfel_data` dictionary of numbers
    surfel_data = extract_surfels(rgb_img, depth_meters, intrinsics)

    # 3. Send the dictionary into writer.py -> pads to 62 properties and writes the .ply file
    save_splat_ply(args.output, surfel_data)


if __name__ == "__main__":
    main()