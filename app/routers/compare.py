from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List, Optional
from app.services.vlm_service import VLMService
from app.utils.image_loader import load_image_from_upload
from app.models.difference_response import DifferenceResponse

router = APIRouter()
# instantiate a (singleton) VLMService; it will lazy-load the model
vlm = VLMService()

@router.post("/compare-images", response_model=DifferenceResponse)
async def compare_images(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    model_id: Optional[str] = None,  # optional override
    use_hf_api: Optional[bool] = False
):
    """
    Compare two images and return human-readable bullet points describing differences.
    """
    try:
        pil1 = await load_image_from_upload(image1)
        pil2 = await load_image_from_upload(image2)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # You can optionally pass a model_id to use a different HF model
    result = vlm.compare_images(
        pil1,
        pil2,
        model_id=model_id,
        use_hf_api=use_hf_api
    )
    return result
