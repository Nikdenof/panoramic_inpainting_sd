from GSAM.GroundingDINO.groundingdino.util.inference import (
    annotate,
    load_image,
    predict,
)
from src.models.grounded_inpaint import inpaint_image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import torch
import io
from pydantic import BaseModel
from src.models.predict_sam import get_sam_model, segment
from diffusers import StableDiffusionInpaintPipeline
from src.models.predict_dino import detect, get_dino_model
from src.utils.constants import SD_SEED, DINO2SD_DICT, DILATE_RADIUS, DILATE_ITERATION

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
groundingdino_model = get_dino_model(device)
sam_predictor = get_sam_model(device)
sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
).to(device)


class SimpleSuccessResponse(BaseModel):
    success: bool
    message: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get(
    "/test_pong",
    response_model=SimpleSuccessResponse,
)
def index():
    return SimpleSuccessResponse(success=True, message="pong")


@app.post("/upload_image/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    image_source, image_tensor = load_image(io.BytesIO(contents))

    result_image = inpaint_image(
        image_source,
        image_tensor,
        groundingdino_model,
        sam_predictor,
        sd_pipe,
        device,
        DINO2SD_DICT,
    )

    img_byte_arr = io.BytesIO()
    result_image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    # Running the FastAPI app with uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=5002)

    # uvicorn.run(app, host="127.0.0.1", port=8000)
