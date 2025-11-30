from typing import List
from pydantic import BaseModel

class DifferenceResponse(BaseModel):
    bullets: List[str]
    explanation: str
    confidence: float
    added_elements: List[str]
    removed_elements: List[str]
