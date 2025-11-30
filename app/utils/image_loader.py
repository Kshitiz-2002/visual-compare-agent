from fastapi import UploadFile
from PIL import Image
import io

async def load_image_from_upload(upload: UploadFile) -> Image.Image:
    data = await upload.read()
    return Image.open(io.BytesIO(data)).convert("RGB")
