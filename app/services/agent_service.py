from PIL import Image
from app.services.vlm_service import VLMService
from app.models.difference_response import DifferenceResponse


class AgentService:
    """
    Visual comparison agent:
    - First pass via VLMService.compare_images
    - If result looks weak, do a refinement pass
    - Final pass merges & structures bullets
    """

    def __init__(self):
        self.vlm = VLMService()

    def run(self, imgA: Image.Image, imgB: Image.Image) -> DifferenceResponse:
        print("[Agent] Step 1 — First Pass Analysis")
        first = self.vlm.compare_images(imgA, imgB)

        if self._looks_good(first):
            print("[Agent] First pass good enough.")
            return self._final_pass(first)

        print("[Agent] Step 2 — Refinement pass")
        refined = self._refine(imgA, imgB, first)

        if self._looks_good(refined):
            print("[Agent] Refinement improved result.")
            return self._final_pass(refined)

        print("[Agent] Using first pass as fallback.")
        return self._final_pass(first)

    # ---------- heuristics ----------

    def _looks_good(self, result: DifferenceResponse) -> bool:
        if not result.bullets:
            return False

        if len(result.bullets) >= 3:
            return True

        joined = " ".join(result.bullets).lower()
        if "no difference" in joined or "no significant difference" in joined:
            return False

        return len(result.bullets) >= 2

    # ---------- refinement & finalization ----------

    def _refine(self, imgA: Image.Image, imgB: Image.Image, initial: DifferenceResponse) -> DifferenceResponse:
        """
        Ask the model to improve clarity & coverage given the initial bullets.
        """
        self.vlm.ensure_local_model()
        meta = self.vlm._local_meta

        refinement_prompt = (
            "You previously attempted to list differences between two images.\n\n"
            "Your previous difference list was:\n" +
            "\n".join(f"- {b}" for b in initial.bullets) +
            "\n\nNow produce a BETTER difference list that:\n"
            "- Is more complete and precise\n"
            "- Uses bullet points\n"
            "- Uses prefixes Added: / Removed: / Changed: / Moved: / Resized: when appropriate\n"
            "- Does NOT repeat the same information multiple times\n"
        )

        print("[Agent] Asking model for refined differences...")
        raw = meta.model.query(imgB.convert("RGB"), refinement_prompt, meta.processor)
        normalized = self.vlm._normalize_output(raw)

        refined_lines = [
            line.strip("-• ").strip()
            for line in normalized.splitlines()
            if len(line.strip()) > 3
        ]

        return DifferenceResponse(
            bullets=refined_lines,
            explanation=normalized,
            confidence=initial.confidence,
            added_elements=[b for b in refined_lines if "added" in b.lower()],
            removed_elements=[b for b in refined_lines if "removed" in b.lower()],
        )

    def _final_pass(self, result: DifferenceResponse) -> DifferenceResponse:
        """
        Final cleanup:
        - deduplicate bullets
        - structure explanation nicely
        """
        # dedupe but preserve order
        unique_bullets = list(dict.fromkeys(result.bullets))

        added = [b for b in unique_bullets if b.lower().startswith("added")]
        removed = [b for b in unique_bullets if b.lower().startswith("removed")]
        changed = [b for b in unique_bullets if b.lower().startswith("changed")]

        sections = []
        if added:
            sections.append("Added:\n" + "\n".join(f"- {x}" for x in added))
        if removed:
            sections.append("Removed:\n" + "\n".join(f"- {x}" for x in removed))
        if changed:
            sections.append("Changed:\n" + "\n".join(f"- {x}" for x in changed))

        explanation = "\n\n".join(sections) if sections else "\n".join(unique_bullets)

        return DifferenceResponse(
            bullets=unique_bullets,
            explanation=explanation,
            confidence=result.confidence,
            added_elements=added,
            removed_elements=removed,
        )
