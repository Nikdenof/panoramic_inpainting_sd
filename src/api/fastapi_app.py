from GSAM.GroundingDINO.groundingdino.util.inference import (
    annotate,
    load_image,
    predict,
)
from src.models.grounded_inpaint import inpaint_image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import io
from pydantic import BaseModel

app = FastAPI()


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
    image, image_tensor = load_image(io.BytesIO(contents))

    result_image = inpaint_image(image, image_tensor)

    img_byte_arr = io.BytesIO()
    result_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/png")

if __name__ == "__main__":
    import uvicorn

    # Running the FastAPI app with uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=5002)

    # uvicorn.run(app, host="127.0.0.1", port=8000)
