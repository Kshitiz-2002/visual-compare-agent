import torch
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Union, List

from transformers import AutoModelForCausalLM, AutoProcessor
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2  # pixel-diff heuristic

from app.models.difference_response import DifferenceResponse


SYSTEM_PROMPT = """You are an expert visual difference assistant.

You will receive:
- A brief pixel-difference summary
- A description of Image A
- A description of Image B

Your job is to list ONLY the differences between Image A and Image B.

Rules:
- Use bullet points (one change per line)
- Be short and clear
- Categorize when possible using prefixes:
  - Added:
  - Removed:
  - Changed:
  - Moved:
  - Resized:
- Do NOT describe things that are identical
- Do NOT output JSON or numbered lists
- Avoid hedging words like "maybe", "seems", "likely"
"""


@dataclass
class LocalModelMeta:
    model_id: str
    processor: any
    model: any
    device: str


class VLMService:

    def __init__(self):
        self._local_meta: Optional[LocalModelMeta] = None
        self.default_model_id = "vikhyatk/moondream2"

    # ---------- Model loading ----------

    def ensure_local_model(self):
        if self._local_meta:
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[VLM] Loading {self.default_model_id} on {device}...")

        processor = AutoProcessor.from_pretrained(
            self.default_model_id,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.default_model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)

        print("[VLM] Model ready.")
        self._local_meta = LocalModelMeta(
            model_id=self.default_model_id,
            processor=processor,
            model=model,
            device=device,
        )

    # ---------- Output utilities ----------

    def _normalize_output(self, raw: Union[str, dict]) -> str:
        """Normalize model response whether dict or string."""
        if isinstance(raw, str):
            return raw.strip()

        if isinstance(raw, dict):
            # Moondream-style keys
            for key in ["answer", "text", "response", "output", "generated_text"]:
                if key in raw:
                    return str(raw[key]).strip()
            # nested reasoning.text
            if "reasoning" in raw and isinstance(raw["reasoning"], dict) and "text" in raw["reasoning"]:
                return str(raw["reasoning"]["text"]).strip()

        return str(raw).strip()

    # ---------- Vision utilities ----------

    def _compute_confidence_ssim(self, imgA: Image.Image, imgB: Image.Image) -> float:
        """Heuristic similarity 0..1 based on SSIM."""
        def to_arr(img):
            return np.array(img.convert("L").resize((512, 512))).astype("float32") / 255.0

        try:
            a = to_arr(imgA)
            b = to_arr(imgB)
            score = ssim(a, b, data_range=1.0)
            score = float(max(0.0, min(1.0, score)))
            return round(score, 3)
        except Exception as e:
            print(f"[VLM] SSIM failed: {e}")
            return 0.5

    def _get_pixel_diff_summary(self, imgA: Image.Image, imgB: Image.Image) -> str:
        """Rudimentary pixel-level diff summary for the prompt."""
        a = cv2.cvtColor(np.array(imgA.convert("RGB")), cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(np.array(imgB.convert("RGB")), cv2.COLOR_RGB2GRAY)

        # align shapes
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = a[:h, :w]
        b = b[:h, :w]

        diff = cv2.absdiff(a, b)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        diff_percent = (np.count_nonzero(thresh) / thresh.size) * 100.0

        if diff_percent < 1:
            return "Pixel-level: almost identical (less than 1% difference)."
        if diff_percent < 5:
            return "Pixel-level: small visual modifications (1-5% of pixels differ)."
        if diff_percent < 15:
            return "Pixel-level: moderate layout or content changes (5-15% difference)."
        return "Pixel-level: significant visual and/or layout changes (over 15% difference)."

    # ---------- Core tools ----------

    def describe_image(self, img: Image.Image) -> str:
        """Runs single-image Moondream captioning / description."""
        self.ensure_local_model()
        meta = self._local_meta
        img = img.convert("RGB")

        raw = meta.model.query(img, "Describe this image clearly and completely.", meta.processor)
        return self._normalize_output(raw)

    def compare_images(self, imgA: Image.Image, imgB: Image.Image) -> DifferenceResponse:
        """
        Main comparison function:
        - describe both images
        - compute pixel diff summary
        - ask model for differences only
        - post-process into bullets + categories
        """
        self.ensure_local_model()
        print("[VLM] Describing Image A...")
        descA = self.describe_image(imgA)

        print("[VLM] Describing Image B...")
        descB = self.describe_image(imgB)

        print("[VLM] Computing pixel difference summary...")
        pixel_summary = self._get_pixel_diff_summary(imgA, imgB)

        compare_prompt = f"""
{SYSTEM_PROMPT}

Pixel difference summary:
{pixel_summary}

###
Image A description:
{descA}

###
Image B description:
{descB}

Now list ONLY the visual differences as bullet points using the required categories.
""".strip()

        print("[VLM] Running comparison reasoning...")
        meta = self._local_meta
        raw = meta.model.query(imgB.convert("RGB"), compare_prompt, meta.processor)
        result = self._normalize_output(raw)

        confidence = self._compute_confidence_ssim(imgA, imgB)

        # ---- Bullet extraction ----
        raw_lines = result.splitlines()
        bullets: List[str] = []
        for line in raw_lines:
            clean = line.strip("•-* ").strip()
            if len(clean) <= 3:
                continue
            # skip meta lines
            if clean.lower().startswith(("image a", "image b", "pixel-level")):
                continue

            # ensure it has a category prefix if possible
            lower = clean.lower()
            if any(word in lower for word in ["added", "new", "introduced", "appears"]):
                if not lower.startswith("added"):
                    clean = f"Added: {clean}"
            elif any(word in lower for word in ["removed", "deleted", "missing", "gone"]):
                if not lower.startswith("removed"):
                    clean = f"Removed: {clean}"
            elif any(word in lower for word in ["changed", "different", "modified", "replaced"]):
                if not lower.startswith("changed"):
                    clean = f"Changed: {clean}"

            bullets.append(clean)

        # hallucination filter: remove super-vague statements
        hallucination_markers = ["maybe", "seems", "likely", "appears to", "probably"]
        bullets = [
            b for b in bullets
            if not any(m in b.lower() for m in hallucination_markers)
        ]

        # classification
        added = [b for b in bullets if b.lower().startswith("added")]
        removed = [b for b in bullets if b.lower().startswith("removed")]

        return DifferenceResponse(
            bullets=bullets,
            explanation=result,
            confidence=confidence,
            added_elements=added,
            removed_elements=removed,
        )
