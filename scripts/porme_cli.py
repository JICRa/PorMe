# SPDX-License-Identifier: MIT
import argparse
from pathlib import Path
from PIL import Image
from src.porme.io import load_images, save_image
from src.porme.segmentation import segment_pores
from src.porme.metrics import pore_fraction
from src.porme.visualization import overlay_mask

def main():
    """"
    ap = argparse.ArgumentParser(description="PorMe demo CLI (skeleton).")
    ap.add_argument("--input", type=Path, required=True, help="Folder with images")
    ap.add_argument("--out", type=Path, required=True, help="Output folder")
    args = ap.parse_args()

    imgs = load_images(args.input)
    args.out.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(imgs):
        mask = segment_pores(img)
        pf = pore_fraction(mask)
        comp = overlay_mask(img, mask, alpha=0.35)
        comp.save(args.out / f"overlay_{i:03d}.png")
        print(f"[{i:03d}] pore fraction = {pf:.3f}")
    """
if __name__ == "__main__":
    main()
