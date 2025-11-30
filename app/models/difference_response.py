from pydantic import BaseModel
from typing import List, Optional

class DifferenceResponse(BaseModel):
    bullets: List[str]
    explanation: str
    confidence: Optional[float] = None  # 0..1 heuristic (SSIM)
    added_elements: Optional[List[str]] = []
    removed_elements: Optional[List[str]] = []
