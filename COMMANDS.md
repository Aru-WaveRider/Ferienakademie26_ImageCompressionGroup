# Team Quickstart & Commands Guide

## 1. Launch the Environment
Open Git Bash in the project folder and start the Docker container:
\`\`\`bash
docker compose run --rm image-compression bash
\`\`\`

## 2. Test Real Bitstream CPU Codec
\`\`\`bash
python projects/01-image-compression/test_bitstream.py
\`\`\`

## 3. Run Rate-Distortion Sweeps

### Baseline (Kodak 19):
\`\`\`bash
python projects/01-image-compression/run_image.py --image Dataset/kodim19.png --rate-num 4 --save-recon
\`\`\`

### Custom Test Images:
\`\`\`bash
python projects/01-image-compression/run_image.py --image Dataset/artificial.ppm --rate-num 4 --save-recon
\`\`\`

## 4. Plot the RD Curve
\`\`\`bash
python projects/01-image-compression/plot_rd.py
\`\`\`
