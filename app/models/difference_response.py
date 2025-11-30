from pydantic import BaseModel 
from typing import List, Optional


class DifferenceResponse(BaseModel):
    bullets: []
    explanation: str 
    confidence: Optional[float] = None
    added_elements: Optional[List[str]] = None
    removed_elements: Optional[List[str]] = None 