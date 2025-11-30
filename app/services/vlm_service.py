import os
import io
import base64
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from PIL import Image
import torch
from pydantic import BaseModel
from app.models.difference_response import DifferenceResponse
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Transformers imports (local VLM path)
from transformers import AutoProcessor, AutoModelForImageTextToText

# Optional: requests if using HuggingFace Inference API
import requests

# System prompt (internal)
SYSTEM_PROMPT = (
    "You are an expert visual inspector. Compare the two images carefully and describe ALL differences "
    "between them in simple bullet points. Use concise human language. Examples: "
    "'The main heading is missing', 'The button color changed from blue to green', 'Left-side image is missing'. "
    "Do NOT output JSON or analysis steps — only bullet points (one-per-line)."
)

# A safe user-visible template to send to the model
def build_prompt() -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Image A: (original)\nImage B: (edited)\n\n"
        "Please list all differences in bullet points. Also label each bullet with 'Added' or 'Removed' when applicable. "
        "If colors changed, say 'color changed from X to Y'. If text/heading removed, say 'heading removed'."
    )

@dataclass
class LocalModelMeta:
    model_id: str
    processor: Any
    model: Any
    device: str

class VLMService:
    """
    VLM wrapper that supports:
      - local inference using transformers (AutoProcessor + AutoModelForImageTextToText)
      - remote inference via Hugging Face Inference API (optional)
    """

    def __init__(self):
        self._local_meta: Optional[LocalModelMeta] = None
        # default model to try locally (small/medium recommended for testing)
        self.default_model_id = os.environ.get("MODEL_ID", "OpenGVLab/InternVL3-1B-hf")

    def ensure_local_model(self, model_id: Optional[str] = None):
        """
        Lazy-load local model (processor + model).
        Use device_map to allow GPU if available.
        """
        if self._local_meta and (model_id is None or self._local_meta.model_id == model_id):
            return

        model_id = model_id or self.default_model_id
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[VLMService] Loading model {model_id} on {device} (this may take a while)...")
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        # AutoModelForImageTextToText is the class used on many InternVL / image->text checkpoints
        model = AutoModelForImageTextToText.from_pretrained(model_id, trust_remote_code=True, device_map="auto" if torch.cuda.is_available() else None)
        model.eval()
        self._local_meta = LocalModelMeta(model_id=model_id, processor=processor, model=model, device=device)

    def compare_images(self, imgA: Image.Image, imgB: Image.Image, model_id: Optional[str] = None, use_hf_api: bool = False) -> DifferenceResponse:
        """
        Main comparison entrypoint.
        If use_hf_api is True, try Hugging Face Inference API (requires HF_TOKEN).
        Otherwise use local transformers model (AutoProcessor + AutoModelForImageTextToText).
        Also compute a lightweight confidence score using SSIM as an extra signal.
        """
        # 1) compute heuristic confidence (0..1) via SSIM on grayscale & resized images
        conf = self._compute_confidence_ssim(imgA, imgB)

        # 2) Build prompt
        prompt = build_prompt()

        # 3) Ask the model (local or HF API)
        if use_hf_api:
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                raise RuntimeError("HF_TOKEN environment variable required for Hugging Face Inference API mode.")
            text_out = self._call_hf_inference_api(imgA, imgB, prompt, model_id=model_id)
        else:
            # local
            self.ensure_local_model(model_id=model_id)
            text_out = self._call_local_model(imgA, imgB, prompt)

        # 4) Postprocess model output into bullet list (split lines, minimal cleanup)
        bullets, added, removed = self._postprocess_text_to_bullets(text_out)

        return DifferenceResponse(
            bullets=bullets,
            explanation="\n".join(bullets) if bullets else text_out,
            confidence=round(float(conf), 3),
            added_elements=added,
            removed_elements=removed,
        )

    def _call_local_model(self, imgA: Image.Image, imgB: Image.Image, prompt: str) -> str:
        """
        Local model flow: processor + model.generate
        Many image->text processors accept a list of images; we pass a composite prompt and both images in sequence.
        """
        meta = self._local_meta
        if meta is None:
            raise RuntimeError("Local model not loaded")

        # Some processors expect 'images' list and 'text' field with template; we'll feed prompt and both images.
        images = [imgA.convert("RGB"), imgB.convert("RGB")]
        # Add a short separator text so model knows both images belong to the same prompt
        # Format: provide the same prompt once and the processor will attach images to it
        inputs = meta.processor(images=images, text=prompt, return_tensors="pt").to(meta.device)
        # Generation
        with torch.no_grad():
            gen = meta.model.generate(**inputs, max_new_tokens=256)
            out = meta.model.decode(gen[0], skip_special_tokens=True)
        return out

    def _call_hf_inference_api(self, imgA: Image.Image, imgB: Image.Image, prompt: str, model_id: Optional[str] = None) -> str:
        """
        Simple wrapper for Hugging Face Inference API. This is a fallback path.
        Hugging Face's inference endpoint expects particular JSON; adapt as needed for the chosen model.
        """
        model = model_id or self.default_model_id
        token = os.environ.get("HF_TOKEN")
        endpoint = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {token}"}

        # encode images to base64
        def to_b64(img: Image.Image):
            buff = io.BytesIO()
            img.save(buff, format="PNG")
            return base64.b64encode(buff.getvalue()).decode("utf-8")

        payload = {
            "inputs": prompt,
            # Many inference endpoints accept 'image' or 'images' in inputs JSON — user may need to adapt
            "options": {"wait_for_model": True},
            "data": {
                "images": [to_b64(imgA), to_b64(imgB)]
            }
        }
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # response format can vary; we try common keys:
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        # Fallback: stringify
        return str(data)

    def _postprocess_text_to_bullets(self, text: str):
        """
        Convert model output to a list of short human bullets.
        Also classify bullets into added/removed when obvious keywords appear.
        """
        if not text:
            return [], [], []
        # split by lines; keep lines that look like bullets or short sentences
        lines = []
        for line in text.splitlines():
            line = line.strip(" \t-•\u2022")
            if not line:
                continue
            # discard long model footers
            if len(line.split()) > 60:
                # keep but summarize
                line = " ".join(line.split()[:30]) + "..."
            lines.append(line)

        added = [l for l in lines if any(k in l.lower() for k in ["added", "new", "appeared", "inserted"])]
        removed = [l for l in lines if any(k in l.lower() for k in ["removed", "missing", "deleted", "gone"])]
        return lines, added, removed

    def _compute_confidence_ssim(self, imgA: Image.Image, imgB: Image.Image) -> float:
        """
        Heuristic confidence: compute SSIM (grayscale) between images resized to 512x512.
        Return a 0..1 score where 1 == identical.
        This is an auxiliary signal only; not model confidence.
        """
        def to_gray_arr(img: Image.Image, size=(512,512)):
            i = img.convert("L").resize(size)
            arr = np.array(i).astype(np.float32) / 255.0
            return arr
        a = to_gray_arr(imgA)
        b = to_gray_arr(imgB)
        try:
            s = ssim(a, b)
            # ssim returns -1..1 sometimes; clamp to 0..1
            s = float(np.clip(s, 0.0, 1.0))
            return s
        except Exception:
            # fallback: naive pixel diff
            diff = np.mean(np.abs(a - b))
            return float(max(0.0, 1.0 - diff))
