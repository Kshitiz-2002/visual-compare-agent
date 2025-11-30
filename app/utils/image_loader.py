from fastapi import UploadFile
from PIL import Image
import io

async def load_image_from_upload(upload: UploadFile) -> Image.Image:
    """
    Load a FastAPI UploadFile into a PIL Image and ensure RGB.
    """
    contents = await upload.read()
    img = Image.open(io.BytesIO(contents))
    return img.convert("RGB")