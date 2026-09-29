import json
import matplotlib.pyplot as plt

with open("outputs/01-image-compression/rd.json", "r") as f:
    data = json.load(f)

points = data["points"]
bpp = [p["bpp"] for p in points]
psnr = [p["psnr_rgb"] for p in points]
qp = [p["qp"] for p in points]

plt.figure(figsize=(7, 5))
plt.plot(bpp, psnr, marker="o", color="blue", linewidth=2, label="DCVC-UF Intra (DMCI)")

for i, txt in enumerate(qp):
    plt.annotate(f"QP {txt}", (bpp[i], psnr[i]), textcoords="offset points", xytext=(0, 8), ha="center")

plt.xlabel("Bitrate (bpp)")
plt.ylabel("PSNR-RGB (dB)")
plt.title(f"Rate-Distortion Curve: {data['image'].split('/')[-1]}")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("outputs/01-image-compression/rd_curve.png", dpi=150)
print("Saved RD curve plot to outputs/01-image-compression/rd_curve.png")