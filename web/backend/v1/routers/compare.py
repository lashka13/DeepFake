from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Annotated
from io import BytesIO
from PIL import Image
from core.model import compare_images

router = APIRouter()
 
@router.post("/compare_images/")
async def compare_images_api(
    first_image: Annotated[UploadFile, File(...)],
    second_image: Annotated[UploadFile, File(...)]
):
    try:
        # Convert uploaded files to PIL images
        img1 = Image.open(BytesIO(await first_image.read())).convert("RGB")
        img2 = Image.open(BytesIO(await second_image.read())).convert("RGB")

        # Compare images
        sim_scores = await compare_images(img1, img2)

        return {"similarity": int(sim_scores*100000)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing images: {str(e)}")