import torch
from segment_anything import build_sam, SamPredictor

from GSAM.GroundingDINO.groundingdino.util import box_ops
from src.data.file_download import download_file
from src.utils.constants import MODEL_URL, MODEL_PATH


def get_sam_model(device):
    download_file(MODEL_URL, MODEL_PATH)
    return SamPredictor(build_sam(checkpoint=MODEL_PATH).to(device))


def segment(image, sam_model, boxes, device):
    sam_model.set_image(image)
    H, W, _ = image.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    transformed_boxes = sam_model.transform.apply_boxes_torch(
        boxes_xyxy.to(device), image.shape[:2]
    )
    masks, _, _ = sam_model.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    return masks.cpu()
