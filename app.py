"""
Main application script for the Gradio interface.

This script initializes the application, loads prerequisite models via model_loader,
defines the user interface using Gradio Blocks, and orchestrates the multi-stage
image generation process by calling functions from the pipelines module.
"""

import gradio as gr
import gradio.themes as gr_themes
import time
import os
import random
# --- Imports from our custom modules ---
try:
    from image_utils import prepare_image
    from model_loader import load_models, are_models_loaded
    from pipelines import run_pose_detection, run_low_res_generation, run_hires_tiling, cleanup_memory
    print("Helper modules imported successfully.")
except ImportError as e:
    print(f"ERROR: Failed to import required local modules: {e}")
    print("Please ensure prompts.py, image_utils.py, model_loader.py, and pipelines.py are in the same directory.")
    raise SystemExit(f"Module import failed: {e}")

# --- Constants & UI Configuration ---
DEFAULT_SEED = 1024
DEFAULT_STEPS_LOWRES = 30
DEFAULT_GUIDANCE_LOWRES = 8.0
DEFAULT_STRENGTH_LOWRES = 0.05
DEFAULT_CN_SCALE_LOWRES = 1.0

DEFAULT_STEPS_HIRES = 20
DEFAULT_GUIDANCE_HIRES = 8.0
DEFAULT_STRENGTH_HIRES = 0.75
DEFAULT_CN_SCALE_HIRES = 1.0

# OUTPUT_DIR = "outputs"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load Prerequisite Models at Startup ---
if not are_models_loaded():
    print("Initial model loading required...")
    load_successful = load_models()
    if not load_successful:
        print("FATAL: Failed to load prerequisite models. The application may not work correctly.")
else:
    print("Models were already loaded.")


# --- Main Processing Function ---
def generate_full_pipeline(
    input_image_path,
    progress=gr.Progress(track_tqdm=True)
    ):
    """
    Orchestrates the entire image generation workflow.

    This function is called when the user clicks the 'Generate' button in the UI.
    It takes inputs from the UI, calls the necessary processing steps in sequence
    (prepare, detect pose, low-res gen, hi-res gen), updates the progress bar,
    and returns the final generated image.

    Args:
        input_image_path (str): Path to the uploaded input image file.
        seed (int): Random seed for generation.
        steps_lowres (int): Inference steps for the low-resolution stage.
        guidance_lowres (float): Guidance scale for the low-resolution stage.
        strength_lowres (float): Img2Img strength for the low-resolution stage.
        cn_scale_lowres (float): ControlNet scale for the low-resolution stage.
        steps_hires (int): Inference steps per tile for the high-resolution stage.
        guidance_hires (float): Guidance scale for the high-resolution stage.
        strength_hires (float): Img2Img strength for the high-resolution stage.
        cn_scale_hires (float): ControlNet scale for the high-resolution stage.
        progress (gr.Progress): Gradio progress tracking object.

    Returns:
        PIL.Image.Image | None: The final generated high-resolution image,
        or the low-resolution image as a fallback if
        tiling fails, or None if critical errors occur early.

    Raises:
        gr.Error: If critical steps like image preparation or pose detection fail.
        gr.Warning: If hi-res tiling fails but low-res succeeded (returns low-res).
    """
    print(f"\n--- Starting New Generation Run ---")
    run_start_time = time.time()

    current_seed = DEFAULT_SEED
    if current_seed == -1:
        current_seed = random.randint(0, 9999999)
        print(f"Using Random Seed: {current_seed}")
    else:
        print(f"Using Fixed Seed: {current_seed}")

    low_res_image = None
    final_image = None

    try:
        progress(0.05, desc="Preparing Input Image...")
        resized_input_image = prepare_image(input_image_path, target_size=512)
        if resized_input_image is None:
            raise gr.Error("Failed to load or prepare the input image. Check format/corruption.")

        progress(0.15, desc="Detecting Pose...")
        pose_map = run_pose_detection(resized_input_image)
        if pose_map is None:
            raise gr.Error("Failed to detect pose from the input image.")
        # try: pose_map.save(os.path.join(OUTPUT_DIR, f"pose_map_{current_seed}.png"))
        # except Exception as save_e: print(f"Warning: Could not save pose map: {save_e}")


        progress(0.25, desc="Starting Low-Res Generation...")
        low_res_image = run_low_res_generation(
            resized_input_image=resized_input_image,
            pose_map=pose_map,
            seed=int(current_seed),
            steps=int(DEFAULT_STEPS_LOWRES),
            guidance_scale=float(DEFAULT_GUIDANCE_LOWRES),
            strength=float(DEFAULT_STRENGTH_LOWRES),
            controlnet_scale=float(DEFAULT_CN_SCALE_LOWRES),
            progress=progress
        )
        print("Low-res generation stage completed successfully.")
        # try: low_res_image.save(os.path.join(OUTPUT_DIR, f"lowres_output_{current_seed}.png"))
        # except Exception as save_e: print(f"Warning: Could not save low-res image: {save_e}")
        progress(0.45, desc="Low-Res Generation Complete.")


        progress(0.50, desc="Starting Hi-Res Tiling...")
        final_image = run_hires_tiling(
            low_res_image=low_res_image,
            seed=int(current_seed),
            steps=int(DEFAULT_STEPS_HIRES),
            guidance_scale=float(DEFAULT_GUIDANCE_HIRES),
            strength=float(DEFAULT_STRENGTH_HIRES),
            controlnet_scale=float(DEFAULT_CN_SCALE_HIRES),
            upscale_factor=2,
            tile_size=1024,
            tile_stride=1024,
            progress=progress
        )
        print("Hi-res tiling stage completed successfully.")
        # try: final_image.save(os.path.join(OUTPUT_DIR, f"hires_output_{current_seed}.png"))
        # except Exception as save_e: print(f"Warning: Could not save final image: {save_e}")

        progress(1.0, desc="Complete!")

    except gr.Error as e:
        print(f"Gradio Error occurred: {e}")
        if final_image is None and low_res_image is not None and ("tiling" in str(e).lower() or "hi-res" in str(e).lower()):
            gr.Warning(f"High-resolution upscaling failed ({e}). Returning low-resolution image.")
            final_image = low_res_image
        else:
            raise e
    except Exception as e:
        print(f"An unexpected error occurred in generate_full_pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"An unexpected error occurred: {e}")
    finally:
        print("Running final cleanup check...")
        cleanup_memory()
        run_end_time = time.time()
        print(f"--- Full Pipeline Run Finished in {run_end_time - run_start_time:.2f} seconds ---")

    return final_image


# --- Gradio Interface Definition ---

theme = gr_themes.Soft(primary_hue=gr_themes.colors.blue, secondary_hue=gr_themes.colors.sky)

# New, improved Markdown description
DESCRIPTION = f"""
<div style="text-align: center;">
    <h1 style="font-family: Impact, Charcoal, sans-serif; font-size: 280%; font-weight: 900; margin-bottom: 16px;"> 
    Pose-Preserving Comicfier
    </h1>
    <p style="margin-bottom: 12; font-size: 94%">
    Transform your photos into the gritty style of a 1940s Western comic! This app uses (Stable Diffusion + ControlNet)
    to apply the artistic look while keeping the original pose intact. Just upload your image and click Generate!
    </p>
    <p style="font-size: 85%;"><em>(Generation currently runs on CPU and can take several minutes. Please be patient! Prompts & parameters are fixed.)</em></p>
    <p style="font-size: 80%; color: grey;">
    <a href="https://github.com/mehran-khani/Pose-Preserving-Comicfier" target="_blank">[View Project on GitHub]</a> | 
    <a href="https://huggingface.co/spaces/Mer-o/Pose-Preserving-Comicfier/discussions" target="_blank">[Report an Issue]</a> 
    </p> 
</div>
"""

EXAMPLE_IMAGES_DIR = "examples"
EXAMPLE_IMAGES = [
    os.path.join(EXAMPLE_IMAGES_DIR, "example1.jpg"),
    os.path.join(EXAMPLE_IMAGES_DIR, "example2.jpg"),
    os.path.join(EXAMPLE_IMAGES_DIR, "example3.jpg"),
    os.path.join(EXAMPLE_IMAGES_DIR, "example4.jpg"),
    os.path.join(EXAMPLE_IMAGES_DIR, "example5.jpg"),
    os.path.join(EXAMPLE_IMAGES_DIR, "example6.jpg"),
]
EXAMPLE_IMAGES = [img for img in EXAMPLE_IMAGES if os.path.exists(img)]

CUSTOM_CSS = """
/* Target the container div Gradio uses for the Image component */
.gradio-image {
    width: 100%;   /* Ensure the container fills the column width */
    height: 100%;  /* Ensure the container fills the height set by the component (e.g., height=400) */
    overflow: hidden; /* Hide any potential overflow before object-fit applies */
}

/* Target the actual <img> tag inside the container */
.gradio-image img { 
    display: block;    /* Remove potential bottom spacing */
    width: 100%;       /* Force image width to match container */
    height: 100%;      /* Force image height to match container */
    object-fit: cover; /* Scale/crop image to cover this forced W/H */
} 

footer { visibility: hidden } 
"""

with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="Pose-Preserving Comicfier") as demo:
    gr.HTML(DESCRIPTION)

    with gr.Row():
        # Input Column
        with gr.Column(scale=1, min_width=350):
            # REMOVED height=400
            input_image = gr.Image(
                type="filepath",
                label="Upload Your Image Here"
            )
            generate_button = gr.Button("Generate Comic Image", variant="primary")

        # Output Column
        with gr.Column(scale=1, min_width=350):
             # REMOVED height=400
            output_image = gr.Image(
                type="pil",
                label="Generated Comic Image",
                interactive=False
            )


    # Examples Section
    if EXAMPLE_IMAGES:
        gr.Examples(
            examples=EXAMPLE_IMAGES,
            inputs=[input_image],
            outputs=[output_image],
            fn=generate_full_pipeline,
            cache_examples=False
        )

    generate_button.click(
        fn=generate_full_pipeline,
        inputs=[input_image],
        outputs=[output_image],
        api_name="generate"
    )


# --- Launch the Gradio App ---
if __name__ == "__main__":
    if not are_models_loaded():
        print("Attempting to load models before launch...")
        if not load_models():
             print("FATAL: Model loading failed on launch. App may not function.")

    print("Attempting to launch Gradio demo...")
    demo.queue().launch(debug=False, share=False)
    print("Gradio app launched. Access it at the URL provided above.")
