"""DCVC-UF vs. standard JPEG at the same file size: zoomed crops + error maps.

Run after run_image.py -- it reads rd.json and the recon_qpXX.png files:
  python projects/01-image-compression/jpeg_compare.py
  python projects/01-image-compression/jpeg_compare.py --crop 550 150 160
"""
import argparse
import io
import json
import os

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")                     # no screen in the container: draw to a file
import matplotlib.pyplot as plt


def to_arr(img):
    return np.asarray(img.convert("RGB"), dtype=np.float32)


def psnr(a, b):
    return 10 * np.log10(255.0 ** 2 / np.mean((a - b) ** 2))


def jpeg_encode(img, quality):
    """Encode in memory; return (file bytes, decoded image)."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    data = buf.getvalue()
    return data, Image.open(io.BytesIO(data))


def jpeg_at_size(img, target_bytes):
    """Highest JPEG quality (1-95) whose file fits in target_bytes (binary search).
    Returns (quality, bytes, decoded, fits). If even quality 1 is too big, fits=False."""
    lo, hi, best = 1, 95, None
    while lo <= hi:
        q = (lo + hi) // 2
        data, dec = jpeg_encode(img, q)
        if len(data) <= target_bytes:
            best, lo = (q, data, dec), q + 1
        else:
            hi = q - 1
    if best is None:
        data, dec = jpeg_encode(img, 1)
        return 1, data, dec, False
    return (*best, True)


def auto_crop(err, s):
    """Top-left corner of the s x s window with the largest total error."""
    c = np.pad(err.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    sums = c[s:, s:] - c[:-s, s:] - c[s:, :-s] + c[:-s, :-s]
    top, left = np.unravel_index(np.argmax(sums), sums.shape)
    return int(top), int(left)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="/work/Dataset/kodim19.png")
    ap.add_argument("--recon-dir", default="/work/outputs/01-image-compression")
    ap.add_argument("--crop", type=int, nargs=3, default=None, metavar=("TOP", "LEFT", "SIZE"),
                    help="zoom region; default: where DCVC's lowest-qp error is highest")
    ap.add_argument("--crop-size", type=int, default=160, help="zoom size for the automatic crop")
    ap.add_argument("--err-max", type=float, default=30.0,
                    help="error value shown as brightest colour (same for every panel)")
    args = ap.parse_args()

    with open(os.path.join(args.recon_dir, "rd.json")) as f:
        points = sorted(json.load(f)["points"], key=lambda p: p["qp"])

    src = Image.open(args.image).convert("RGB")
    orig = to_arr(src)

    rows = []
    print(f"{'qp':>4} {'DCVC kB':>8} {'DCVC PSNR':>10} | {'JPEG q':>6} {'JPEG kB':>8} {'JPEG PSNR':>10}")
    print("-" * 56)
    for p in points:
        qp = p["qp"]
        path = os.path.join(args.recon_dir, f"recon_qp{qp:02d}.png")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} -- rerun run_image.py")
        dcvc = to_arr(Image.open(path))
        target = p["kbytes"] * 1024            # DCVC's estimated size in bytes
        q, data, dec, fits = jpeg_at_size(src, target)
        jpg = to_arr(dec)
        with open(os.path.join(args.recon_dir, f"jpeg_qp{qp:02d}.jpg"), "wb") as f:
            f.write(data)                      # the real JPEG file, to open yourself
        row = dict(qp=qp, dcvc=dcvc, jpg=jpg, jq=q, fits=fits,
                   d_kb=p["kbytes"], j_kb=len(data) / 1024,
                   d_psnr=psnr(orig, dcvc), j_psnr=psnr(orig, jpg))
        rows.append(row)
        note = "" if fits else "  (JPEG can't get this small)"
        print(f"{qp:>4} {row['d_kb']:>8.1f} {row['d_psnr']:>8.2f}dB | {q:>6} "
              f"{row['j_kb']:>8.1f} {row['j_psnr']:>8.2f}dB{note}")

    if args.crop:
        top, left, s = args.crop
    else:
        s = args.crop_size
        top, left = auto_crop(np.abs(orig - rows[0]["dcvc"]).mean(axis=2), s)
    print(f"\nzoom: --crop {top} {left} {s}")

    def zoom(a):
        return a[top:top + s, left:left + s].astype(np.uint8)

    fig, ax = plt.subplots(len(rows) + 1, 4, figsize=(13, 3.3 * (len(rows) + 1)))
    ax[0, 0].imshow(zoom(orig), interpolation="nearest")
    ax[0, 0].set_title("original (zoom)")
    ax[0, 1].imshow(orig.astype(np.uint8))
    ax[0, 1].add_patch(plt.Rectangle((left, top), s, s, fill=False, ec="red", lw=1.5))
    ax[0, 1].set_title("zoom location")

    for i, r in enumerate(rows, start=1):
        jtitle = f"JPEG q{r['jq']}  {r['j_kb']:.1f} kB  {r['j_psnr']:.2f} dB"
        if not r["fits"]:
            jtitle += "\n(over budget)"
        ax[i, 0].imshow(zoom(r["dcvc"]), interpolation="nearest")
        ax[i, 0].set_title(f"DCVC qp{r['qp']}  {r['d_kb']:.1f} kB  {r['d_psnr']:.2f} dB")
        ax[i, 1].imshow(zoom(r["jpg"]), interpolation="nearest")
        ax[i, 1].set_title(jtitle)
        ax[i, 2].imshow(np.abs(orig - r["dcvc"]).mean(axis=2), cmap="inferno", vmin=0, vmax=args.err_max)
        ax[i, 2].set_title(f"|error| DCVC qp{r['qp']}")
        im = ax[i, 3].imshow(np.abs(orig - r["jpg"]).mean(axis=2), cmap="inferno", vmin=0, vmax=args.err_max)
        ax[i, 3].set_title(f"|error| JPEG q{r['jq']}")

    for a in ax.flat:
        a.axis("off")
    fig.colorbar(im, ax=ax[1:, :].ravel().tolist(), shrink=0.6, label="absolute error (0-255 scale)")

    out = os.path.join(args.recon_dir, "jpeg_compare.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()