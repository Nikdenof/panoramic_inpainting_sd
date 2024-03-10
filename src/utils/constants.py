SD_SEED = 32  # for reproducibility
DILATE_RADIUS = 3
DILATE_ITERATION = 5

MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
MODEL_PATH = "../../models/sam_vit_h_4b8939.pth"

# SKY
DINO_SKY_PROMPT = "sky"
SKY_PROMPT = "Blue_sky, realistic, fully_clear, clear_edges, high_resolution, endless_blue_sky, only_blue_white, clouds"
SKY_NEGATIVE_PROMPT = (
    "low_resolution, ugly, unrealistic, text, birds, branches, leaves, full_of_stuff,"
    "buildings, roof, structural extension, additional upper levels, trees, debris, sprigs, "
    "twigs, green_branches, black_twigs, green_tall_grass, flats, multi_storey_buildings, sun, rays"
)
SKY_COLOR_RGB = [135, 206, 235]
SKY_STRENGTH = 0.8
SKY_GUIDANCE = 13.5
# SKY_COLOR_BGR = SKY_COLOR_RGB[::-1]

# GRASS
DINO_GRASS_PROMPT = "grass"
GRASS_PROMPT = "green_grass, realistic, summer_grass, high_resolution, lawn, greenest, summer"
GRASS_NEGATIVE_PROMPT = (
    "low_resolution, ugly, unrealistic, branches, gold_leaves, structural_extension, trees, debris, sprigs, "
    "black_twigs, blank_spots, grass_bare_patches, mud, ground"
)
GRASS_COLOR_RGB = [126, 200, 80]
GRASS_STRENGTH = 0.4
GRASS_GUIDANCE = 10
# GRASS_COLOR_BGR = GRASS_COLOR_RGB[::-1]

DINO2SD_DICT = {
    DINO_SKY_PROMPT: [SKY_PROMPT, SKY_NEGATIVE_PROMPT, SKY_COLOR_RGB, SKY_STRENGTH, SKY_GUIDANCE],
    DINO_GRASS_PROMPT: [GRASS_PROMPT, GRASS_NEGATIVE_PROMPT, GRASS_COLOR_RGB, GRASS_STRENGTH, GRASS_GUIDANCE],
}
