import json
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
        description="Max 2 sentences in English: state net margin vs Q50 and explain either why the discount looks legitimate or the exact defect/risk disqualifying it."
    )


class BatchAuditResponse(BaseModel):
    audits: list[DealAuditItem]


def audit_flagged_deals(
    client: genai.Client, candidates: list[dict]
) -> list[DealAuditItem]:
    if not candidates:
        return []

    sanitized = [
        {**c, "description": (c.get("description") or "")[:1500]}
        for c in candidates
    ]

    prompt = (
        "You are a European second-hand GPU risk auditor. Listings may be in English, German, French, or Italian.\n"
        "Each listing below is priced below the 10th percentile market floor (Q10). Inspect title, full description, "
        "and seller_rating for hidden traps: 'defekt', 'Bastler', 'nicht getestet', 'HS', 'non testé', 'guasto', "
        "'artifacting', 'crashes under load', 'broken fan/port', 'Überweisung / bank transfer only', "
        "empty boxes, waterblocks, or title/body SKU downgrades.\n"
        "- Mark 'BUY' ONLY if the card is described as working with no functional caveats.\n"
        "- Mark 'REJECT_TRAP' if any defect, untested excuse, accessory-only trap, or off-platform payment risk exists.\n"
        "- Mark 'MANUAL_REVIEW' if the description is too vague/short to confirm functionality.\n\n"
        f"Candidates:\n{json.dumps(sanitized, ensure_ascii=False)}"
    )

    from google.genai import errors

    for model_name in ("gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.6-flash"):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BatchAuditResponse,
                    temperature=0.0,
                ),
            )
            return BatchAuditResponse.model_validate_json(response.text).audits
        except errors.ServerError:
            continue
    return []
