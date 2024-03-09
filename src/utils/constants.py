SD_SEED = 42  # for reproducibility

MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
MODEL_PATH = "../../models/sam_vit_h_4b8939.pth"

# SKY
DINO_SKY_PROMPT = "sky"
SKY_PROMPT = "Blue_sky, realistic, fully_clear, clear_edges, high_resolution, endless_blue_sky, only_blue_white"
SKY_NEGATIVE_PROMPT = (
    "low_resolution, ugly, unrealistic, birds, branches, leaves, full_of_stuff, many_clouds, "
    "buildings, roof, clouds, structural extension, additional upper levels, trees, debris, sprigs, "
    "twigs, green_branches, black_twigs, green_tall_grass, flats, multi_storey_buildings"
)

# GRASS
DINO_GRASS_PROMPT = "grass"
GRASS_PROMPT = "green_grass, realistic, summer_grass, high_resolution, lawn"
GRASS_NEGATIVE_PROMPT = (
    "low_resolution, ugly, unrealistic, branches, gold_leaves, structural_extension, trees, debris, sprigs, "
    "black_twigs, blank_spots, grass_bare_patches, mud, ground"
)

DINO2SD_DICT = {
    DINO_SKY_PROMPT: [SKY_PROMPT, SKY_NEGATIVE_PROMPT],
    DINO_GRASS_PROMPT: [GRASS_PROMPT, GRASS_NEGATIVE_PROMPT],
}
