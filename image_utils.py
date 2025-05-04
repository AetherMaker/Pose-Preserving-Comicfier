"""
Contains utility functions for image loading, preparation, and manipulation.
Includes HEIC image format support via the optional 'pillow-heif' library.
"""

from PIL import Image, ImageOps, ImageDraw
import os

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    print("HEIC opener registered successfully using pillow-heif.")
    _heic_support = True
except ImportError:
    print("Warning: pillow-heif not installed. HEIC/HEIF support will be disabled.")
    _heic_support = False


print("Loading Image Utils...")

def prepare_image(image_filepath, target_size=512):
    """
    Prepares an input image file for the diffusion pipeline.

    Loads an image from the given filepath (supports standard formats like
    JPG, PNG, WEBP, and HEIC/HEIF),
    ensures it's in RGB format, handles EXIF orientation, and performs
    a forced resize to a square target_size, ignoring the original aspect ratio.

    Args:
        image_filepath (str): The path to the image file.
        target_size (int): The target dimension for both width and height.

    Returns:
        PIL.Image.Image | None: The prepared image as a PIL Image object in RGB format,
                               or None if loading or processing fails.
    """
    if image_filepath is None:
        print("Warning: prepare_image received None filepath.")
        return None

    if not isinstance(image_filepath, str) or not os.path.exists(image_filepath):
         print(f"Error: Invalid filepath provided to prepare_image: {image_filepath}")
         if isinstance(image_filepath, Image.Image):
             print("Warning: Received PIL Image instead of filepath, proceeding...")
             image = image_filepath
         else:
            return None
    else:
        # --- Load Image from Filepath ---
        print(f"Loading image from path: {image_filepath}")
        try:
            image = Image.open(image_filepath)
        except ImportError as e:
             print(f"ImportError during Image.open: {e}. Is pillow-heif installed?")
             print("Cannot process image format.")
             return None
        except Exception as e:
            print(f"Error opening image file {image_filepath} with PIL: {e}")
            return None

    # --- Process PIL Image ---
    try:
        image = ImageOps.exif_transpose(image)

        image = image.convert("RGB")

        original_width, original_height = image.size

        final_width = target_size
        final_height = target_size

        resized_image = image.resize((final_width, final_height), Image.LANCZOS)

        print(f"Original size: ({original_width}, {original_height}), FORCED Resized to: ({final_width}, {final_height})")
        return resized_image
    except Exception as e:
        print(f"Error during PIL image processing steps: {e}")
        return None

def create_blend_mask(tile_size=1024, overlap=256):
    """
    Creates a feathered blending mask (alpha mask) for smooth tile stitching.

    Generates a square mask where the edges have a linear gradient ramp within
    the specified overlap zone, and the central area is fully opaque.
    Assumes overlap occurs equally on all four sides.

    Args:
        tile_size (int): The dimension (width and height) of the tiles being processed.
        overlap (int): The number of pixels that overlap between adjacent tiles.

    Returns:
        PIL.Image.Image: The blending mask as a PIL Image object in 'L' (grayscale) mode.
                         White (255) areas are fully opaque, black (0) are transparent,
                         gray values provide blending.
    """
    if overlap >= tile_size // 2:
        print("Warning: Overlap is large relative to tile size, mask generation might be suboptimal.")
        overlap = tile_size // 2 - 1

    mask = Image.new("L", (tile_size, tile_size), 0)
    draw = ImageDraw.Draw(mask)

    if overlap > 0:
        for i in range(overlap):
            alpha = int(255 * (i / float(overlap)))

            # Left edge ramp
            draw.line([(i, 0), (i, tile_size)], fill=alpha)
            # Right edge ramp
            draw.line([(tile_size - 1 - i, 0), (tile_size - 1 - i, tile_size)], fill=alpha)
            # Top edge ramp
            draw.line([(0, i), (tile_size, i)], fill=alpha)
            # Bottom edge ramp
            draw.line([(0, tile_size - 1 - i), (tile_size, tile_size - 1 - i)], fill=alpha)

    center_start = overlap
    center_end_x = tile_size - overlap
    center_end_y = tile_size - overlap

    if center_end_x > center_start and center_end_y > center_start:
        draw.rectangle( (center_start, center_start, center_end_x - 1, center_end_y - 1), fill=255 )
    else:
        center_x, center_y = tile_size // 2, tile_size // 2
        draw.point((center_x, center_y), fill=255)
        if tile_size % 2 == 0:
             draw.point((center_x-1, center_y), fill=255)
             draw.point((center_x, center_y-1), fill=255)
             draw.point((center_x-1, center_y-1), fill=255)


    print(f"Blend mask created (Size: {tile_size}x{tile_size}, Overlap: {overlap})")
    return mask