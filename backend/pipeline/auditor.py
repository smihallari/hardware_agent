from typing import Literal
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


class DealAuditItem(BaseModel):
    listing_id: int
    audit_verdict: Literal["BUY", "REJECT_TRAP", "MANUAL_REVIEW"]
    trap_flags: list[
        Literal[
            "UNTESTED_CLAIM",
            "THERMAL_OR_CRASH_UNDER_LOAD",
            "PARTIAL_PORT_OR_SLOT_FAILURE",
            "NO_BUYER_PROTECTION_PAYMENT",
            "SKU_DOWNGRADE_IN_DESCRIPTION",
            "BIOS_LOCKED_OR_MDM",
            "ZERO_FEEDBACK_HIGH_VALUE",
        ]
    ] = Field(default_factory=list)
    arbitrage_rationale: str = Field(
        description="Max 2 sentences: state the net flip margin vs Q50 and explain either why the discount is legitimate (e.g., urgent sale, no box) or what specific defect disqualifies it."
    )


class BatchAuditResponse(BaseModel):
    audits: list[DealAuditItem]


def audit_flagged_deals(
    client: genai.Client, candidates: list[dict]
) -> list[DealAuditItem]:
    """
    Expects candidates with keys:
    listing_id, canonical_sku, title, description, seller_rating,
    evaluated_price, q10, q50, expected_net_profit
    """
    if not candidates:
        return []

    prompt = (
        "You are a hardware arbitrage risk auditor. Each listing below is priced below the "
        "10th percentile market floor (Q10). Inspect the full title, description, and seller rating "
        "for hidden defects or transaction scams (e.g. 'untested', 'works fine unless gaming', "
        "'one fan broken', 'bank transfer only without shipping protection', or title/body SKU mismatch).\n"
        "- Mark 'BUY' ONLY if the item is fully functional and the discount is explainable (quick cash, upgrade, missing box).\n"
        "- Mark 'REJECT_TRAP' if any functional defect, 'untested' excuse, or scam payment condition exists.\n"
        "- Mark 'MANUAL_REVIEW' if description is too vague to confirm full stability.\n\n"
        f"Candidates:\n{candidates}"
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BatchAuditResponse,
            temperature=0.0,
        ),
    )
    return BatchAuditResponse.model_validate_json(response.text).audits