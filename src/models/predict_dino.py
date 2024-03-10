import torch
from huggingface_hub import hf_hub_download
from GSAM.GroundingDINO.groundingdino.util.inference import load_image

from GSAM.GroundingDINO.groundingdino.models import build_model
from GSAM.GroundingDINO.groundingdino.util.inference import (
    annotate,
    predict,
)
from GSAM.GroundingDINO.groundingdino.util.slconfig import SLConfig
from GSAM.GroundingDINO.groundingdino.util.utils import clean_state_dict

ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"


def load_model_hf(repo_id, filename, ckpt_config_filename, device="cpu"):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    args.device = device
    model = build_model(args)

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


def get_dino_model(device):
    groundingdino_model = load_model_hf(
        ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device
    )
    return groundingdino_model


def detect(
    image, image_source, text_prompt, model, box_threshold=0.4, text_threshold=0.25
):
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    annotated_frame = annotate(
        image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
    )
    annotated_frame = annotated_frame[..., ::-1]
    return annotated_frame, boxes, phrases


if __name__ == "__main__":
    local_image_path = "../../data/raw/test_images/faces_plates.jpg"
    image_source, image = load_image(local_image_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    groundingdino_model = get_dino_model(device)

    classes2detect = [
        "grass",
        "sky",
        "licence_plate",
        "person",
        "person_face",
        # "faces",
        # "physiognomy",
        # "facial_features",
        # "facial_expression",
    ]
    TEXT_PROMPT = f"{'. '.join(classes2detect)}."
    annotated_frame, detected_boxes, output_phrases = detect(
        image, image_source, text_prompt=TEXT_PROMPT, model=groundingdino_model
    )



    print()
