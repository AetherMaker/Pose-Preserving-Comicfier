# Pose-Preserving Comicfier - Gradio App

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/Mer-o) <!-- TODO: change the link -->

This repository contains the code for a Gradio web application that transforms input images into a specific retro Western comic book style while preserving the original pose. It uses Stable Diffusion v1.5, ControlNet (OpenPose + Tile), and specific LoRAs.

This application refactors the workflow initially developed in a [Kaggle Notebook](https://github.com/mehran-khani/SD-Controlnet-Comic-Styler) into a deployable web app.

## Features

*   **Pose Preservation:** Uses ControlNet OpenPose to accurately maintain the pose from the input image.
*   **Retro Comic Style Transfer:** Applies specific LoRAs (`night_comic_V06.safetensors` & `add_detail.safetensors`) for a 1940s Western comic aesthetic with enhanced details.
*   **Tiling Upscaling:** Implements ControlNet Tile for 2x high-resolution output (1024x1024), improving detail consistency over large images.
*   **Simplified UI:** Easy-to-use interface with only an image upload and generate button.
*   **Fixed Parameters:** Generation uses pre-defined, optimized parameters (steps, guidance, strength, prompts) based on the original notebook implementation for consistent results.
*   **Dynamic Backgrounds:** The background elements in the generated image are randomized for variety in the low-resolution stage.
*   **Broad Image Support:** Accepts common formats like JPG, PNG, WEBP, and HEIC (requires `pillow-heif`).

## Technology Stack

*   **Python 3**
*   **Gradio:** Web UI framework.
*   **PyTorch:** Core ML framework.
*   **Hugging Face Libraries:**
    *   `diffusers`: Stable Diffusion pipelines, ControlNet integration.
    *   `transformers`: Underlying model components.
    *   `accelerate`: Hardware acceleration utilities.
    *   `peft`: LoRA loading and management.
*   **ControlNet:**
    *   OpenPose Detector (`controlnet_aux`)
    *   OpenPose ControlNet Model (`lllyasviel/sd-controlnet-openpose`)
    *   Tile ControlNet Model (`lllyasviel/control_v11f1e_sd15_tile`)
*   **Base Model:** `runwayml/stable-diffusion-v1-5`
*   **LoRAs Used:**
    *   Style: [Western Comics Style](https://civitai.com/models/1081588/western-comics-style) (using `night_comic_V06.safetensors`)
    *   Detail: [Detail Tweaker LoRA](https://civitai.com/models/58390/detail-tweaker-lora-lora) (using `add_detail.safetensors`)
*   **Image Processing:** `Pillow`, `pillow-heif`, `numpy`, `opencv-python-headless`
*   **Dependencies:** `matplotlib`, `mediapipe` (required by `controlnet_aux`)

## Workflow Overview

1.  **Image Preparation (`image_utils.py`):** Input image is loaded (supports HEIC), converted to RGB, EXIF data handled, and force-resized to 512x512.
2.  **Pose Detection (`pipelines.py`):** An OpenPose map is extracted from the resized image using `controlnet_aux`.
3.  **Low-Resolution Generation (`pipelines.py`):**
    *   An SDv1.5 Img2Img pipeline with Pose ControlNet is dynamically loaded.
    *   Prompts are generated (`prompts.py`) with a fixed base/style and a *randomized* background element.
    *   Style and Detail LoRAs are applied.
    *   A 512x512 image is generated using fixed parameters.
    *   The pipeline is unloaded to conserve VRAM.
4.  **High-Resolution Tiling (`pipelines.py`):**
    *   The 512x512 image is upscaled 2x (to 1024x1024) using bicubic interpolation (creating a blurry base).
    *   An SDv1.5 Img2Img pipeline with Tile ControlNet is dynamically loaded.
    *   Tile-specific prompts (excluding the random background) are used.
    *   Style and Detail LoRAs are applied (potentially with different weights).
    *   The image is processed in overlapping 1024x1024 tiles.
    *   Processed tiles are blended back together using an alpha mask (`image_utils.py`).
    *   The pipeline is unloaded.
5.  **Output (`app.py`):** The final 1024x1024 image is displayed in the Gradio UI.

## How to Run Locally

*(Requires sufficient RAM/CPU or compatible GPU, Python 3.8+, and Git)*

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mehran-khani/Pose-Preserving-Comicfier.git
    cd Pose-Preserving-Comicfier
    ```
2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    # .\.venv\Scripts\Activate.ps1
    # .\.venv\Scripts\activate.bat
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: PyTorch installation might require specific commands depending on your OS/CUDA setup if using a local GPU. See PyTorch website.)*
4.  **Download LoRA files:**
    *   Create a folder named `loras` in the project root.
    *   Download `night_comic_V06.safetensors` (from Civitai link above) and place it in the `loras` folder.
    *   Download `add_detail.safetensors` (from Civitai link above) and place it in the `loras` folder.
5.  **Run the Gradio app:**
    ```bash
    python app.py
    ```
6.  Open the local URL provided (e.g., `http://127.0.0.1:7860`) in your browser. *(Note: Execution will be very slow without a suitable GPU).*

## Deployment to Hugging Face Spaces

This app is designed for deployment on Hugging Face Spaces, ideally with GPU hardware.

1.  Ensure all code (`*.py`), `requirements.txt`, `.gitignore`, and the `loras` folder (containing the `.safetensors` files) are committed and pushed to this GitHub repository.
2.  Create a new Space on Hugging Face ([huggingface.co/new-space](https://huggingface.co/new-space)).
3.  Choose an owner, Space name, and select "Gradio" as the Space SDK.
4.  Select desired hardware (e.g., "T4 small" under GPU options). Note compute costs may apply.
5.  Choose "Use existing GitHub repository".
6.  Enter the URL of this GitHub repository.
7.  Click "Create Space". The Space will build the environment from `requirements.txt` and run `app.py`. Monitor the build and runtime logs for any issues.

## Limitations

*   **Speed:** Generation requires significant time (minutes), especially on shared/free GPU hardware, due to the multi-stage process and dynamic model loading between stages. CPU execution is impractically slow.
*   **VRAM:** While optimized with dynamic pipeline unloading, the process still requires considerable GPU VRAM (>10GB peak). Out-of-memory errors are possible on lower-VRAM GPUs.
*   **Fixed Style:** The artistic style (prompts, LoRAs, parameters) is fixed in the code to replicate the notebook's specific output and cannot be changed via the UI.

## License

MIT License