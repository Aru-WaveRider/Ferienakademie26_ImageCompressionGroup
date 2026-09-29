# Potential Feature Ideas for DCVC-UF Image Compression

This document outlines four potential features that can be built on top of the inference-only DCVC-UF image compression codebase. Since we are not training the model, these features focus on modifying the forward pass, extracting insights, or improving the evaluation pipeline.

## 1. Region of Interest (RoI) Coding
**Concept**: In standard compression, the entire image is quantized uniformly based on a single Quality Parameter (QP). RoI coding dynamically applies fine quantization (high quality) to important regions (e.g., faces, text, or a central subject) and coarse quantization (high compression) to the background.

**Implementation Strategy**:
- **Where to edit**: `DMCI.forward_one_frame()` in `third_party/DCVC/src/models/image_model.py`.
- **How to do it**: 
  - The model currently scales the entire latent `y` uniformly using `curr_q_enc`. We can introduce a spatial mask (e.g., a tensor of 0s and 1s) to modulate this quantization step spatially.
  - You can generate this mask programmatically (e.g., a simple center-crop bounding box) or by passing the image through a lightweight saliency/face detector before compression.
  - This allows us to achieve massive bitrate savings on backgrounds while keeping the subject crystal clear.

## 2. Latent Space & Spatial Prior Visualizer
**Concept**: The latent representations (`y` and `z`) and the spatial prior parameters (scales and means) are typically hidden inside the black box of the forward pass. Extracting and visualizing them as heatmaps provides deep insights into how the model mathematically "understands" the image and where it decides to spend bits.

**Implementation Strategy**:
- **Where to edit**: `forward_prior_4x` in `common_model.py` and a new script in `projects/01-image-compression/`.
- **How to do it**:
  - Modify the return dictionaries in the model to bubble up the `scales_hat` (which represents the predicted variance/entropy) and `y_hat` (the quantized features).
  - Create a Python script (`visualize_latents.py`) that runs the model on an image, averages the channels of `scales_hat`, and uses `matplotlib` to plot it as a 2D heatmap. 
  - Areas with high predicted variance will light up, typically aligning with sharp edges and complex textures.

## 3. Advanced Perceptual Metrics Integration
**Concept**: Currently, `run_image.py` calculates quality using PSNR (Peak Signal-to-Noise Ratio). PSNR is purely mathematical and often penalizes structural shifts that the human eye doesn't even notice. Integrating metrics like LPIPS (Learned Perceptual Image Patch Similarity) or MS-SSIM provides a Rate-Distortion (RD) curve that actually correlates with human perception.

**Implementation Strategy**:
- **Where to edit**: `projects/01-image-compression/run_image.py`.
- **How to do it**:
  - Add libraries to compute MS-SSIM and LPIPS (the repository's README mentions LPIPS is already used in Task 6b, so the AlexNet weights will be available).
  - Calculate these metrics alongside PSNR inside the rate sweep loop.
  - Export the new metrics to `rd.json` so they can be plotted. This is especially useful for comparing DCVC-UF's visual quality against classical codecs like VTM, which often optimize for PSNR at the expense of visual sharpness.

## 4. Noise-Resilient Pre-processing for High-ISO Images
**Concept**: Your dataset contains high-ISO night shots (`nightshot_iso_1600.ppm`) and RAW images. Camera noise is inherently random, making it impossible for the entropy model to predict. This causes the bitrate to skyrocket (the model wastes bits trying to perfectly reconstruct the noise). 

**Implementation Strategy**:
- **Where to edit**: `projects/01-image-compression/run_image.py`.
- **How to do it**:
  - Introduce an optional pre-processing flag (e.g., `--denoise`).
  - Use OpenCV (e.g., `cv2.fastNlMeansDenoisingColored`) or a basic smoothing filter to remove grain before the image is converted to YCbCr and passed to the model.
  - The goal is to measure and prove the **bitrate savings** (BPP reduction) when compressing the denoised image vs. the noisy image, demonstrating how critical clean input data is for learned entropy models.
