import os
from datetime import datetime
from typing import Literal
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import psycopg
from psycopg.rows import dict_row
from pydantic import BaseModel

app = FastAPI(title="Hardware Arbitrage Engine API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def get_db() -> psycopg.Connection:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise HTTPException(status_code=500, detail="DATABASE_URL is not configured")
    return psycopg.connect(db_url, row_factory=dict_row)


class FunnelStatsOut(BaseModel):
    total_listings: int
    cleared_listings: int
    evaluated_listings: int
    triggered_llm_gate: int
    confirmed_buys: int
    rejected_traps: int
    llm_calls_saved_pct: float
    last_evaluated_at: datetime | None


class DealOut(BaseModel):
    listing_id: int
    source: str
    url: str
    title: str
    description: str | None
    seller_rating: float | None
    category: str
    canonical_sku: str
    brand_tier: str
    condition_score: int
    has_warranty: bool
    has_original_box: bool
    is_mined_or_repaired: bool
    evaluated_price: float
    q10_price: float
    q50_price: float
    q90_price: float
    expected_net_profit: float
    discount_vs_q50_pct: float
    triggered_llm_gate: bool
    audit_verdict: str | None
    trap_flags: list[str]
    arbitrage_rationale: str | None
    evaluated_at: datetime


class SkuBenchmarkOut(BaseModel):
    category: str
    canonical_sku: str
    active_count: int
    cleared_count: int
    median_q10: float
    median_q50: float
    median_q90: float


@app.get("/api/stats", response_model=FunnelStatsOut)
def get_pipeline_stats() -> FunnelStatsOut:
    with get_db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM raw_listings) AS total_listings,
                (SELECT COUNT(*) FROM raw_listings WHERE is_cleared = TRUE) AS cleared_listings,
                COUNT(*) AS evaluated_listings,
                COUNT(*) FILTER (WHERE triggered_llm_gate = TRUE) AS triggered_llm_gate,
                COUNT(*) FILTER (WHERE audit_verdict = 'BUY') AS confirmed_buys,
                COUNT(*) FILTER (WHERE audit_verdict = 'REJECT_TRAP') AS rejected_traps,
                MAX(evaluated_at) AS last_evaluated_at
            FROM deal_evaluations;
            """
        )
        row = cur.fetchone()

    evaluated = row["evaluated_listings"] or 0
    gated = row["triggered_llm_gate"] or 0
    saved_pct = round((1.0 - (gated / evaluated)) * 100, 1) if evaluated > 0 else 0.0

    return FunnelStatsOut(
        total_listings=row["total_listings"] or 0,
        cleared_listings=row["cleared_listings"] or 0,
        evaluated_listings=evaluated,
        triggered_llm_gate=gated,
        confirmed_buys=row["confirmed_buys"] or 0,
        rejected_traps=row["rejected_traps"] or 0,
        llm_calls_saved_pct=saved_pct,
        last_evaluated_at=row["last_evaluated_at"],
    )


@app.get("/api/deals", response_model=list[DealOut])
def list_deals(
    verdict: Literal["ALL", "BUY", "REJECT_TRAP", "MANUAL_REVIEW"] = "ALL",
    category: Literal["ALL", "GPU", "CPU", "LAPTOP"] = "ALL",
    sku: str | None = None,
    only_gated: bool = Query(True, description="Only show sub-Q10 anomalies that triggered the LLM gate"),
    limit: int = Query(50, ge=1, le=200),
) -> list[DealOut]:
    clauses = ["r.is_cleared = FALSE"]
    params: list = []

    if only_gated:
        clauses.append("d.triggered_llm_gate = TRUE")
    if verdict != "ALL":
        clauses.append("d.audit_verdict = %s")
        params.append(verdict)
    if category != "ALL":
        clauses.append("f.category = %s")
        params.append(category)
    if sku:
        clauses.append("f.canonical_sku = %s")
        params.append(sku)

    where_sql = " AND ".join(clauses)
    params.append(limit)

    query = f"""
        SELECT
            d.listing_id,
            r.source,
            r.url,
            r.title,
            r.description,
            r.seller_rating,
            f.category,
            f.canonical_sku,
            f.brand_tier,
            f.condition_score,
            f.has_warranty,
            f.has_original_box,
            f.is_mined_or_repaired,
            d.evaluated_price,
            d.q10_price,
            d.q50_price,
            d.q90_price,
            d.expected_net_profit,
            ROUND(((d.q50_price - d.evaluated_price) / NULLIF(d.q50_price, 0)) * 100, 1) AS discount_vs_q50_pct,
            d.triggered_llm_gate,
            d.audit_verdict,
            COALESCE(d.trap_flags, '{{}}') AS trap_flags,
            d.arbitrage_rationale,
            d.evaluated_at
        FROM deal_evaluations d
        JOIN raw_listings r ON r.id = d.listing_id
        JOIN extracted_features f ON f.content_hash = r.content_hash
        WHERE {where_sql}
        ORDER BY
            CASE WHEN d.audit_verdict = 'BUY' THEN 0 ELSE 1 END,
            d.expected_net_profit DESC
        LIMIT %s;
    """

    with get_db() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    return [DealOut(**row) for row in rows]


@app.get("/api/skus", response_model=list[SkuBenchmarkOut])
def list_sku_benchmarks() -> list[SkuBenchmarkOut]:
    with get_db() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                f.category,
                f.canonical_sku,
                COUNT(*) FILTER (WHERE r.is_cleared = FALSE) AS active_count,
                COUNT(*) FILTER (WHERE r.is_cleared = TRUE) AS cleared_count,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY d.q10_price)::numeric, 2) AS median_q10,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY d.q50_price)::numeric, 2) AS median_q50,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY d.q90_price)::numeric, 2) AS median_q90
            FROM deal_evaluations d
            JOIN raw_listings r ON r.id = d.listing_id
            JOIN extracted_features f ON f.content_hash = r.content_hash
            GROUP BY f.category, f.canonical_sku
            ORDER BY f.category, median_q50 DESC;
            """
        )
        rows = cur.fetchall()

    return [SkuBenchmarkOut(**row) for row in rows]