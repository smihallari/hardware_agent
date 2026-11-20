import os
from typing import Literal
import psycopg
from psycopg.rows import dict_row
from fastmcp import FastMCP

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

mcp = FastMCP("hardware-arbitrage-db")


def _get_conn() -> psycopg.Connection:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set.")
    return psycopg.connect(db_url, row_factory=dict_row)


@mcp.tool
def get_top_deals(
    sku: str | None = None,
    source: str | None = None,
    verdict: Literal["ALL", "BUY", "REJECT_TRAP", "MANUAL_REVIEW"] = "BUY",
    max_price_eur: float | None = None,
    limit: int = 10,
) -> list[dict]:
    """Query active sub-Q10 hardware deals from Neon PostgreSQL with optional filters."""
    clauses = ["r.is_cleared = FALSE", "d.triggered_llm_gate = TRUE"]
    params: list = []

    if verdict != "ALL":
        clauses.append("d.audit_verdict = %s")
        params.append(verdict)
    if sku:
        clauses.append("f.canonical_sku ILIKE %s")
        params.append(f"%{sku}%")
    if source:
        clauses.append("r.source = %s")
        params.append(source.lower())
    if max_price_eur is not None:
        clauses.append("d.evaluated_price <= %s")
        params.append(max_price_eur)

    params.append(max(1, min(limit, 50)))
    sql = f"""
        SELECT
            d.listing_id,
            r.source,
            f.canonical_sku,
            f.brand_tier,
            r.title,
            r.url,
            r.seller_rating,
            d.evaluated_price::float AS asking_plus_shipping_eur,
            d.q10_price::float AS q10_eur,
            d.q50_price::float AS q50_eur,
            d.expected_net_profit::float AS expected_net_profit_eur,
            d.audit_verdict,
            COALESCE(d.trap_flags, '{{}}') AS trap_flags,
            d.arbitrage_rationale
        FROM deal_evaluations d
        JOIN raw_listings r ON r.id = d.listing_id
        JOIN extracted_features f ON f.content_hash = r.content_hash
        WHERE {" AND ".join(clauses)}
        ORDER BY d.expected_net_profit DESC
        LIMIT %s;
    """
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


@mcp.tool
def get_sku_benchmarks(sku: str | None = None) -> list[dict]:
    """Return median Q10, Q50, Q90 prices and active/cleared listing counts per SKU."""
    where_extra = "AND f.canonical_sku ILIKE %s" if sku else ""
    params = [f"%{sku}%"] if sku else []

    sql = f"""
        SELECT
            f.canonical_sku,
            COUNT(*) FILTER (WHERE r.is_cleared = FALSE) AS active_count,
            COUNT(*) FILTER (WHERE r.is_cleared = TRUE) AS cleared_count,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY d.q10_price)::numeric, 2)::float AS median_q10_eur,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY d.q50_price)::numeric, 2)::float AS median_q50_eur,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY d.q90_price)::numeric, 2)::float AS median_q90_eur
        FROM raw_listings r
        JOIN extracted_features f ON f.content_hash = r.content_hash
        LEFT JOIN deal_evaluations d ON d.listing_id = r.id
        WHERE f.is_valid_hardware = TRUE {where_extra}
        GROUP BY f.canonical_sku
        HAVING COUNT(d.listing_id) > 0
        ORDER BY median_q50_eur DESC;
    """
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


@mcp.tool
def get_listing_price_history(listing_id: int) -> dict:
    """Return full description and chronological price snapshots for a specific listing_id."""
    with _get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.id, r.source, r.title, r.url, r.description, r.seller_rating,
                   r.first_seen_at::text, r.last_seen_at::text, r.is_cleared
            FROM raw_listings r
            WHERE r.id = %s;
            """,
            (listing_id,),
        )
        listing = cur.fetchone()
        if not listing:
            return {"error": f"Listing {listing_id} not found."}

        cur.execute(
            """
            SELECT asking_price::float, shipping_cost::float, scraped_at::text
            FROM listing_snapshots
            WHERE listing_id = %s
            ORDER BY scraped_at ASC;
            """,
            (listing_id,),
        )
        listing["snapshots"] = cur.fetchall()
        return listing


if __name__ == "__main__":
    mcp.run()  # Defaults to stdio transport for Claude Desktop / Cursor / VS Code