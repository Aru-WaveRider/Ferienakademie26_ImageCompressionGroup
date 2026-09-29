import sys
import torch
from PIL import Image
from torchvision.transforms import ToTensor

# Add scripts directory to path to reach uf_codec
sys.path.append('scripts/bitstream')
import uf_codec
from src.models.image_model import DMCI

print("Initializing DMCI model...")
device = "cpu"
net = DMCI().to(device)

# Load pretrained weights
checkpoint_path = "weights/dcvc/cvpr2026_image.pth.tar"
ckpt = torch.load(checkpoint_path, map_location=device)
if "state_dict" in ckpt:
    ckpt = ckpt["state_dict"]
net.load_state_dict(ckpt)
net.eval()

# Prepare codec engine
print("Preparing uf_codec...")
net = uf_codec.prepare(net, skip_thres=0.15)

# Load test image
img_path = "Dataset/kodim19.png"  # or check path in projects/01-image-compression/
try:
    img = Image.open(img_path).convert("RGB")
except FileNotFoundError:
    print(f"Could not find {img_path}, using random tensor for verification")
    img_tensor = torch.rand(1, 3, 256, 256)
else:
    img_tensor = ToTensor()(img).unsqueeze(0)

# Compress to real bitstream bytes
qp = 21
print(f"Compressing at qp={qp}...")
bitstream, x_hat = uf_codec.compress(net, img_tensor, qp)
print(f"Compression complete! Real bitstream size: {len(bitstream)} bytes")

# Decompress directly from the byte payload
print("Decompressing from bytes alone...")
x_recon = uf_codec.decompress(net, bitstream)
print(f"Decompressed tensor shape: {x_recon.shape}")
print("Real bitstream test SUCCESS!")