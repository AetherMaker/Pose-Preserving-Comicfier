"""
Handles the loading and management of necessary AI models from Hugging Face Hub.

Provides functions to load models once at startup and access them throughout
the application, managing device placement (CPU/GPU) and data types.
Optimized for typical Hugging Face Space GPU environments.
"""

import torch
from diffusers import ControlNetModel
from controlnet_aux import OpenposeDetector
import gc

# --- Configuration ---
# Automatically detect CUDA availability and set appropriate device/dtype
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
    print(f"CUDA available. Using Device: {DEVICE}, Dtype: {DTYPE}")
    try:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Couldn't get GPU name: {e}")
else:
    DEVICE = "cpu"
    DTYPE = torch.float32
    print(f"CUDA not available. Using Device: {DEVICE}, Dtype: {DTYPE}")


# Model IDs from Hugging Face Hub
# BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5" # Base SD model ID needed by pipelines
OPENPOSE_DETECTOR_ID = 'lllyasviel/ControlNet' # Preprocessor model repo
CONTROLNET_POSE_MODEL_ID = "lllyasviel/sd-controlnet-openpose" # OpenPose ControlNet weights
CONTROLNET_TILE_MODEL_ID = "lllyasviel/control_v11f1e_sd15_tile" # Tile ControlNet weights

_openpose_detector = None
_controlnet_pose = None
_controlnet_tile = None
_models_loaded = False

# --- Loading Function ---

def load_models(force_reload=False):
    """
    Loads the OpenPose detector (to CPU) and ControlNet models (to configured DEVICE).

    This function should typically be called once when the application starts.
    It checks if models are already loaded to prevent redundant loading unless
    `force_reload` is True.

    Args:
        force_reload (bool): If True, forces reloading even if models are already loaded.

    Returns:
        bool: True if all models were loaded successfully (or already were), False otherwise.
    """
    global _openpose_detector, _controlnet_pose, _controlnet_tile, _models_loaded

    if _models_loaded and not force_reload:
        print("Models already loaded.")
        return True

    print(f"--- Loading Models ---")
    if DEVICE == "cuda":
        print("Performing initial CUDA cache clear...")
        gc.collect()
        torch.cuda.empty_cache()

    # 1. OpenPose Detector
    try:
        print(f"Loading OpenPose Detector from {OPENPOSE_DETECTOR_ID} to CPU...")
        _openpose_detector = OpenposeDetector.from_pretrained(OPENPOSE_DETECTOR_ID)
        print("OpenPose detector loaded successfully (on CPU).")
    except Exception as e:
        print(f"ERROR: Failed to load OpenPose Detector: {e}")
        _models_loaded = False
        return False

    # 2. ControlNet Models
    try:
        print(f"Loading ControlNet Pose Model from {CONTROLNET_POSE_MODEL_ID} to {DEVICE} ({DTYPE})...")
        _controlnet_pose = ControlNetModel.from_pretrained(
            CONTROLNET_POSE_MODEL_ID, torch_dtype=DTYPE
        )
        _controlnet_pose.to(DEVICE)
        print("ControlNet Pose model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load ControlNet Pose Model: {e}")
        _models_loaded = False
        return False

    try:
        print(f"Loading ControlNet Tile Model from {CONTROLNET_TILE_MODEL_ID} to {DEVICE} ({DTYPE})...")
        _controlnet_tile = ControlNetModel.from_pretrained(
            CONTROLNET_TILE_MODEL_ID, torch_dtype=DTYPE
        )
        _controlnet_tile.to(DEVICE)
        print("ControlNet Tile model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load ControlNet Tile Model: {e}")
        _models_loaded = False
        return False

    _models_loaded = True
    print("--- All prerequisite models loaded successfully. ---")
    if DEVICE == "cuda":
        print("Performing post-load CUDA cache clear...")
        gc.collect()
        torch.cuda.empty_cache()
    return True

# --- Getter Functions ---

def get_openpose_detector():
    if not _models_loaded: load_models()
    return _openpose_detector

def get_controlnet_pose():
    if not _models_loaded: load_models()
    return _controlnet_pose

def get_controlnet_tile():
    if not _models_loaded: load_models()
    return _controlnet_tile

def get_device():
    return DEVICE

def get_dtype():
    return DTYPE

def are_models_loaded():
    return _models_loaded