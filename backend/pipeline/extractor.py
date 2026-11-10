import hashlib
import re
from typing import Literal
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Stage 1: Fast deterministic junk filter (Saves LLM quota and keeps training set clean)
JUNK_PATTERNS = re.compile(
    r"\b(box only|empty box|cooler only|heatsink only|for parts|not working|"
    r"no display|artifact|code 43|broken|defective|ζητειται|ζητείται|μόνο κουτί|για ανταλλακτικά)\b",
    re.IGNORECASE,
)

def compute_content_hash(title: str, description: str) -> str:
    normalized = re.sub(r"\s+", " ", f"{title} {description}".lower()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

def passes_fast_gate(title: str, description: str, price: float) -> bool:
    if price < 25.0:  # Drops cable/bracket listings miscategorized as hardware
        return False
    return not bool(JUNK_PATTERNS.search(f"{title} {description}"))


# Stage 2: Batch Structured Extraction Schema for unseen content_hashes
class HardwareFeatureItem(BaseModel):
    content_hash: str
    category: Literal["GPU", "CPU", "LAPTOP", "ACCESSORY_JUNK"]
    canonical_sku: str = Field(
        description="Standardized chip SKU only, e.g. 'RTX 3080 10GB', 'RTX 3080 12GB', 'RX 6800 XT 16GB', 'RYZEN 7 7800X3D'. Distinguish VRAM variants."
    )
    brand_tier: Literal["FE_REF", "LOW", "MID", "FLAGSHIP"] = Field(
        description="AIB cooling/build tier: LOW (Ventus/Eagle/Biostar), MID (TUF/Gaming X/Pulse), FLAGSHIP (Strix/Suprim/Nitro+)"
    )
    vram_or_ram_gb: int | None
    storage_gb: int | None
    condition_score: int = Field(ge=0, le=4, description="0=Defective/Parts, 1=Fair, 2=Good, 3=Mint/Complete, 4=Sealed")
    has_warranty: bool
    has_original_box: bool
    is_mined_or_repaired: bool = Field(description="True if mining rig history, repasted/opened, or repaired")
    is_valid_hardware: bool = Field(description="False if listing is just a box, cooler, waterblock, or 'wanted' ad")

class BatchExtractionResponse(BaseModel):
    items: list[HardwareFeatureItem]


def extract_hardware_batch(client: genai.Client, batch: list[dict]) -> list[HardwareFeatureItem]:
    """
    Expects batch = [{"content_hash": "...", "title": "...", "description": "..."}]
    Processes up to 25 listings in a single structured call.
    """
    prompt = (
        "Extract standardized hardware features from these second-hand marketplace listings. "
        "Normalize canonical_sku strictly to the base chip + VRAM/variant so identical chips group together.\n\n"
        f"Listings:\n{batch}"
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BatchExtractionResponse,
            temperature=0.0,
        ),
    )
    return BatchExtractionResponse.model_validate_json(response.text).items