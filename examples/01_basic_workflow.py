# SPDX-License-Identifier: MIT
from pathlib import Path
from src.porme.io import load_images, save_image
from src.porme.segmentation import segment_pores
from src.porme.metrics import pore_fraction
from src.porme.visualization import overlay_mask

DATA = Path(__file__).parent / "data" / "test_images"

def run():
    """
    imgs = load_images(DATA)
    for i, img in enumerate(imgs):
        mask = segment_pores(img)
        pf = pore_fraction(mask)
        comp = overlay_mask(img, mask, alpha=0.35)
        out = Path(__file__).parent / f"result_{i:03d}.png"
        save_image(comp, out)
        print(f"Image {i}: pore fraction = {pf:.3f}; saved {out}")
    """
if __name__ == "__main__":
    run()
