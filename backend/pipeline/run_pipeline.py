import os
import psycopg
from psycopg.rows import dict_row
from google import genai

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from backend.pipeline.ingest import (
    DEFAULT_MARKETPLACES,
    TARGET_GPU_QUERIES,
    EbayBrowseClient,
    upsert_and_track_snapshots,
)
from backend.pipeline.extractor import extract_deterministic
from backend.pipeline.valuation import QuantileArbitrageModel
from backend.pipeline.auditor import audit_flagged_deals


def run_pipeline() -> None:
    db_url = os.environ.get("DATABASE_URL")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable is missing.")

    marketplaces = tuple(
        m.strip().upper()
        for m in os.environ.get("EBAY_MARKETPLACES", ",".join(DEFAULT_MARKETPLACES)).split(",")
        if m.strip()
    )

    ebay = EbayBrowseClient()
    gemini = genai.Client(api_key=gemini_key) if gemini_key else None

    try:
        with psycopg.connect(db_url, row_factory=dict_row,prepare_threshold=None, autocommit=True) as conn:
            print(f"[1/5] Ingesting live USED Buy-It-Now GPUs from {marketplaces}...")
            for mp in marketplaces:
                mp_listings = []
                sweep_complete = True
                for query in TARGET_GPU_QUERIES:
                    try:
                        mp_listings.extend(ebay.search_used_gpus(mp, query, limit=200))
                    except Exception as exc:
                        sweep_complete = False
                        print(f"  WARN {mp} query='{query}' failed: {exc}")

                count = upsert_and_track_snapshots(
                    conn,
                    source=mp.lower(),
                    listings=mp_listings,
                    mark_missing_cleared=sweep_complete,
                )
                print(f"  {mp}: upserted {count} unique listings (complete_sweep={sweep_complete})")

            print("[2/5] Running deterministic feature extraction on unparsed listings...")
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT ON (r.content_hash)
                        r.content_hash,
                        r.title,
                        r.description,
                        r.raw_payload->>'conditionId' AS condition_id,
                        s.asking_price
                    FROM raw_listings r
                    LEFT JOIN extracted_features f ON f.content_hash = r.content_hash
                    JOIN LATERAL (
                        SELECT asking_price FROM listing_snapshots
                        WHERE listing_id = r.id ORDER BY scraped_at DESC LIMIT 1
                    ) s ON TRUE
                    WHERE f.content_hash IS NULL;
                    """
                )
                unparsed = cur.fetchall()

            if unparsed:
                with conn.cursor() as cur:
                    for row in unparsed:
                        feat = extract_deterministic(
                            content_hash=row["content_hash"],
                            title=row["title"],
                            description=row["description"] or "",
                            asking_price=float(row["asking_price"]),
                            condition_id=row["condition_id"],
                        )
                        cur.execute(
                            """
                            INSERT INTO extracted_features (
                                content_hash, category, canonical_sku, brand_tier,
                                vram_or_ram_gb, storage_gb, condition_score,
                                has_warranty, has_original_box, is_mined_or_repaired, is_valid_hardware
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (content_hash) DO NOTHING;
                            """,
                            (
                                feat.content_hash, feat.category, feat.canonical_sku, feat.brand_tier,
                                feat.vram_or_ram_gb, feat.storage_gb, feat.condition_score,
                                feat.has_warranty, feat.has_original_box, feat.is_mined_or_repaired,
                                feat.is_valid_hardware,
                            ),
                        )
                conn.commit()
            print(f"  Extracted {len(unparsed)} new listings deterministically ($0 LLM cost).")

            print("[3/5] Running 5-fold OOF LightGBM Quantile Valuation & Baseline Benchmark...")
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        r.id,
                        r.source,
                        r.seller_rating,
                        r.is_cleared,
                        f.category,
                        f.canonical_sku,
                        f.brand_tier,
                        COALESCE(f.vram_or_ram_gb, 0) AS vram_or_ram_gb,
                        f.condition_score,
                        f.has_warranty,
                        f.has_original_box,
                        f.is_mined_or_repaired,
                        EXTRACT(DAY FROM (r.last_seen_at - r.first_seen_at))::int AS days_on_market,
                        (s.asking_price + s.shipping_cost) AS total_price
                    FROM raw_listings r
                    JOIN extracted_features f ON f.content_hash = r.content_hash
                    JOIN LATERAL (
                        SELECT asking_price, shipping_cost FROM listing_snapshots
                        WHERE listing_id = r.id ORDER BY scraped_at DESC LIMIT 1
                    ) s ON TRUE
                    WHERE f.is_valid_hardware = TRUE;
                    """
                )
                dataset = cur.fetchall()

            model = QuantileArbitrageModel()
            predictions, report = model.fit_and_evaluate_oof(dataset)
            if not predictions or not report:
                print(f"  Not enough valid active listings yet ({len(dataset)} total). Skipping evaluation.")
                return

            print(
                f"  Benchmark (n={report.n_evaluated}): "
                f"Q10 Pinball LGB={report.lgb_q10_pinball} vs Naive={report.naive_q10_pinball} | "
                f"Q50 Pinball LGB={report.lgb_q50_pinball} vs Naive={report.naive_q50_pinball} | "
                f"Q10 Coverage LGB={report.lgb_q10_coverage:.1%} vs Naive={report.naive_q10_coverage:.1%}"
            )

            with conn.cursor() as cur:
                for p in predictions:
                    cur.execute(
                        """
                        INSERT INTO deal_evaluations (
                            listing_id, evaluated_price, q10_price, q50_price, q90_price,
                            expected_net_profit, triggered_llm_gate
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (listing_id) DO UPDATE SET
                            evaluated_price = EXCLUDED.evaluated_price,
                            q10_price = EXCLUDED.q10_price,
                            q50_price = EXCLUDED.q50_price,
                            q90_price = EXCLUDED.q90_price,
                            expected_net_profit = EXCLUDED.expected_net_profit,
                            triggered_llm_gate = EXCLUDED.triggered_llm_gate,
                            evaluated_at = NOW();
                        """,
                        (
                            p.listing_id, p.evaluated_price, p.q10, p.q50, p.q90,
                            p.expected_net_profit, p.trigger_llm_gate,
                        ),
                    )
            conn.commit()

            print("[4/5] Hydrating full descriptions & running Post-Gate LLM Audit...")
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT d.listing_id, r.source, r.external_id, f.canonical_sku,
                           r.title, r.description, r.seller_rating,
                           d.evaluated_price, d.q10_price, d.q50_price, d.expected_net_profit
                    FROM deal_evaluations d
                    JOIN raw_listings r ON r.id = d.listing_id
                    JOIN extracted_features f ON f.content_hash = r.content_hash
                    WHERE r.is_cleared = FALSE
                      AND d.triggered_llm_gate = TRUE
                      AND d.audit_verdict IS NULL
                    ORDER BY d.expected_net_profit DESC
                    LIMIT 30;
                    """
                )
                gated_rows = cur.fetchall()

            # --- Replace the hydration + audit write blocks in Step 4 ---
            hydrated_updates: list[tuple[str, int]] = []
            candidates = []
            for r in gated_rows:
                full_desc = r["description"] or ""
                try:
                    hydrated = ebay.fetch_item_detail(r["source"].upper(), r["external_id"])
                    if hydrated and len(hydrated) > len(full_desc):
                        full_desc = hydrated
                        hydrated_updates.append((full_desc, r["listing_id"]))
                except Exception as exc:
                    print(f"  WARN detail hydration failed for {r['external_id']}: {exc}")

                candidates.append({
                    "listing_id": r["listing_id"],
                    "canonical_sku": r["canonical_sku"],
                    "title": r["title"],
                    "description": full_desc,
                    "seller_rating": float(r["seller_rating"] or 98.0),
                    "evaluated_price": float(r["evaluated_price"]),
                    "q10": float(r["q10_price"]),
                    "q50": float(r["q50_price"]),
                    "expected_net_profit": float(r["expected_net_profit"]),
                })

            if hydrated_updates:
                with psycopg.connect(db_url) as write_conn, write_conn.cursor() as cur:
                    cur.executemany(
                        "UPDATE raw_listings SET description = %s WHERE id = %s;",
                        hydrated_updates,
                    )
                    write_conn.commit()

            if candidates and gemini:
                all_audits = []
                for i in range(0, len(candidates), 15):
                    all_audits.extend(audit_flagged_deals(gemini, candidates[i : i + 15]))

                with psycopg.connect(db_url) as write_conn, write_conn.cursor() as cur:
                    cur.executemany(
                        """
                        UPDATE deal_evaluations
                        SET audit_verdict = %s, trap_flags = %s, arbitrage_rationale = %s
                        WHERE listing_id = %s;
                        """,
                        [
                            (a.audit_verdict, a.trap_flags, a.arbitrage_rationale, a.listing_id)
                            for a in all_audits
                        ],
                    )
                    write_conn.commit()
                print(f"  Audited {len(all_audits)} gated sub-Q10 deals.")
            else:
                print(f"  Skipped LLM audit (gated={len(candidates)}, gemini_configured={bool(gemini)}).")

            print("[5/5] Pipeline run completed successfully.")
    finally:
        ebay.close()


if __name__ == "__main__":
    run_pipeline()