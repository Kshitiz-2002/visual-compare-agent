from fastapi import APIRouter, File, UploadFile, HTTPException
from app.utils.image_loader import load_image_from_upload
from app.services.agent_service import AgentService
from app.models.difference_response import DifferenceResponse

router = APIRouter()
agent = AgentService()


@router.post("/compare-images", response_model=DifferenceResponse)
async def compare_images(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
):
    """
    Visual comparison endpoint.
    Uses an internal agent loop (first-pass + refinement) to produce bullet-point differences.
    """
    try:
        img1 = await load_image_from_upload(image1)
        img2 = await load_image_from_upload(image2)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    result = agent.run(img1, img2)
    return result
