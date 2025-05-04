import random

"""
Defines fixed prompts and provides a function to generate
randomized prompts for each run, mirroring the original notebook behavior.
Used by the main pipeline functions.
"""

BASE_PROMPT = "detailed face portrait, accurate facial features, natural features, clear eyes, keep the gender same as the input image"

STYLE_PROMPT = r"((LIMITED PALETTE)), ((RETRO COMIC)), ((1940S \(STYLE\))), ((WESTERN COMICS \(STYLE\))), ((NIGHT COMIC)), detailed illustration, sharp lines, sfw"

BASE_NEGATIVE_PROMPT = (
    "generic face, distorted features, unrealistic face, bad anatomy, extra limbs, fused fingers, poorly drawn hands, poorly drawn face, "
    "text, signature, watermark, letters, words, username, artist name, speech bubble, multiple panels, "
    "ugly, disfigured, deformed, low quality, worst quality, blurry, jpeg artifacts, noisy, "
    "weapon, gun, knife, violence, gore, blood, injury, mutilated, horrific, nsfw, nude, naked, explicit, sexual, lingerie, bikini, suggestive, provocative, disturbing, scary, offensive, illegal, unlawful"
)

# --- Background Generation Elements ---
BG_SETTINGS = [
    "on a futuristic city street at night", "in a retro sci-fi control room", "in a dusty western saloon",
    "in front of an abstract energy field", "in a neon-lit alleyway", "in a stark cyberpunk cityscape",
    "with speed lines background", "in a manga panel frame", "in a dimly lit laboratory",
    "against a dramatic explosive background", "in a cluttered artist studio", "in a dynamic action scene"
]

BG_DETAILS = [
    "detailed background", "cinematic lighting", "dramatic shadows",
    "high contrast", "low angle shot", "dynamic composition", "atmospheric perspective", "intricate details"
]

def get_prompts_for_run():
    """
    Generates the prompts needed for one generation cycle,
    including a newly randomized background for the low-res stage.
    Returns prompts suitable for low-res and hi-res stages.
    """
    # --- Low-Res Prompt Generation ---
    chosen_bg_setting = random.choice(BG_SETTINGS)
    chosen_bg_detail = random.choice(BG_DETAILS)
    background_prompt = f"{chosen_bg_setting}, {chosen_bg_detail}"
    positive_prompt_lowres = f"{BASE_PROMPT}, {STYLE_PROMPT}, {background_prompt}"

    # --- Tile Prompt Generation ---
    positive_prompt_tile = f"{BASE_PROMPT}, {STYLE_PROMPT}"

    negative_prompt_tile = (
        BASE_NEGATIVE_PROMPT +
        ", blurry face, distorted face, mangled face, bad face, low quality, blurry"
    )

    return positive_prompt_lowres, BASE_NEGATIVE_PROMPT, positive_prompt_tile, negative_prompt_tile