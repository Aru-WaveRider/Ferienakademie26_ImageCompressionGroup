"""Make compression errors visible: zoomed crops + error maps per qp.

Run after run_image.py (it reads the recon_qpXX.png files it wrote):
  python projects/01-image-compression/compare_recon.py
  python projects/01-image-compression/compare_recon.py --qps 0 63 --crop 400 100 128
"""
import argparse
import os

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")                     # no screen in the container: draw to a file
import matplotlib.pyplot as plt


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="/work/Dataset/kodim19.png")
    ap.add_argument("--recon-dir", default="/work/outputs/01-image-compression")
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
    ap.add_argument("--crop", type=int, nargs=3, default=[300, 150, 192],
                    metavar=("TOP", "LEFT", "SIZE"), help="zoom region in pixels")
    ap.add_argument("--err-max", type=float, default=30.0,
                    help="error value shown as brightest colour (same for every qp)")
    args = ap.parse_args()

    orig = load(args.image)
    top, left, s = args.crop

    recs = {}
    for qp in args.qps:
        path = os.path.join(args.recon_dir, f"recon_qp{qp:02d}.png")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} -- run run_image.py so that qp {qp} is included")
        recs[qp] = load(path)

    n = len(args.qps) + 1
    fig, ax = plt.subplots(2, n, figsize=(3.2 * n, 7))

    # column 0: the original, zoomed crop on top, full image with crop box below
    ax[0, 0].imshow(orig[top:top + s, left:left + s].astype(np.uint8), interpolation="nearest")
    ax[0, 0].set_title("original (zoom)")
    ax[1, 0].imshow(orig.astype(np.uint8))
    ax[1, 0].add_patch(plt.Rectangle((left, top), s, s, fill=False, ec="red", lw=1.5))
    ax[1, 0].set_title("zoom location")

    # one column per qp: zoomed reconstruction on top, error map below
    for i, qp in enumerate(args.qps, start=1):
        rec = recs[qp]
        mse = np.mean((orig - rec) ** 2)
        psnr = 10 * np.log10(255.0 ** 2 / mse)
        ax[0, i].imshow(rec[top:top + s, left:left + s].astype(np.uint8), interpolation="nearest")
        ax[0, i].set_title(f"qp {qp}  ({psnr:.2f} dB)")
        err = np.abs(orig - rec).mean(axis=2)          # average error over R, G, B
        im = ax[1, i].imshow(err, cmap="inferno", vmin=0, vmax=args.err_max)
        ax[1, i].set_title(f"|error|  qp {qp}")

    for a in ax.flat:
        a.axis("off")
    fig.colorbar(im, ax=ax[1, :].tolist(), shrink=0.8, label="absolute error (0-255 scale)")

    out = os.path.join(args.recon_dir, "compare_recon.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()