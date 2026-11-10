import os
import numpy as np
import psycopg
from google import genai
from backend.pipeline.ingest import seed_historical_baseline
from backend.pipeline.extractor import extract_hardware_batch
from backend.pipeline.valuation import QuantileArbitrageModel
from backend.pipeline.auditor import audit_flagged_deals

def run_pipeline() -> None:
    db_url = os.environ.get("DATABASE_URL")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable is missing.")
    
    client = genai.Client(api_key=gemini_key) if gemini_key else None

    print("[1/5] Connecting to Neon PostgreSQL and seeding baseline...")
    with psycopg.connect(db_url) as conn:
        seed_historical_baseline(conn)

        print("[2/5] Extracting features for unparsed listings...")
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.content_hash, r.title, r.description
                FROM raw_listings r
                LEFT JOIN extracted_features f ON f.content_hash = r.content_hash
                WHERE f.content_hash IS NULL
                LIMIT 50;
                """
            )
            unparsed = cur.fetchall()

        if unparsed and client:
            batch = [{"content_hash": row[0], "title": row[1], "description": row[2]} for row in unparsed]
            extracted_items = extract_hardware_batch(client, batch)
            with conn.cursor() as cur:
                for item in extracted_items:
                    cur.execute(
                        """
                        INSERT INTO extracted_features (
                            content_hash, category, canonical_sku, brand_tier,
                            vram_or_ram_gb, storage_gb, condition_score,
                            has_warranty, has_original_box, is_mined_or_repaired, is_valid_hardware
                        )
                        VALUES (%s, %s, %s, %s, %s, NULL, %s, %s, %s, %s, %s)
                        ON CONFLICT (content_hash) DO NOTHING;
                        """,
                        (
                            item.content_hash, item.category, item.canonical_sku, item.brand_tier,
                            item.vram_or_ram_gb, item.condition_score, item.has_warranty,
                            item.has_original_box, item.is_mined_or_repaired, item.is_valid_hardware
                        ),
                    )
            conn.commit()

        print("[3/5] Training LightGBM Quantile Models & Evaluating Market...")
        with conn.cursor() as cur:
            # Fetch training dataset (historical / cleared + active valid hardware)
            cur.execute(
                """
                SELECT
                    f.category, f.canonical_sku, f.brand_tier,
                    COALESCE(f.vram_or_ram_gb, 0), COALESCE(f.storage_gb, 0),
                    f.condition_score, f.has_warranty::int, f.has_original_box::int,
                    f.is_mined_or_repaired::int,
                    EXTRACT(DAY FROM (r.last_seen_at - r.first_seen_at))::int AS days_on_market,
                    s.asking_price + s.shipping_cost AS total_price,
                    r.is_cleared
                FROM raw_listings r
                JOIN extracted_features f ON f.content_hash = r.content_hash
                JOIN LATERAL (
                    SELECT asking_price, shipping_cost FROM listing_snapshots
                    WHERE listing_id = r.id ORDER BY scraped_at DESC LIMIT 1
                ) s ON TRUE
                WHERE f.is_valid_hardware = TRUE;
                """
            )
            rows = cur.fetchall()

        if not rows:
            print("No valid listings found for training.")
            return

        # Build feature arrays
        X_data, y_prices, cleared_flags, dom_list = [], [], [], []
        for row in rows:
            cat, sku, tier, vram, stor, cond, warr, box, mined, dom, price, cleared = row
            # Categorical encoding hash lookup for LightGBM
            X_data.append([str(cat), str(sku), str(tier), int(vram), int(stor), int(cond), int(warr), int(box), int(mined), int(dom)])
            y_prices.append(float(price))
            cleared_flags.append(bool(cleared))
            dom_list.append(int(dom))

        X_mat = np.array(X_data, dtype=object)
        y_arr = np.array(y_prices, dtype=np.float32)
        cleared_arr = np.array(cleared_flags, dtype=bool)
        dom_arr = np.array(dom_list, dtype=np.int32)

        model = QuantileArbitrageModel()
        model.fit(X_mat, y_arr, cleared_arr, dom_arr)

        # Evaluate active listings for arbitrage
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.id, f.category, f.canonical_sku, f.brand_tier,
                       COALESCE(f.vram_or_ram_gb, 0), COALESCE(f.storage_gb, 0),
                       f.condition_score, f.has_warranty::int, f.has_original_box::int,
                       f.is_mined_or_repaired::int,
                       EXTRACT(DAY FROM (r.last_seen_at - r.first_seen_at))::int AS days_on_market,
                       s.asking_price + s.shipping_cost AS total_price
                FROM raw_listings r
                JOIN extracted_features f ON f.content_hash = r.content_hash
                JOIN LATERAL (
                    SELECT asking_price, shipping_cost FROM listing_snapshots
                    WHERE listing_id = r.id ORDER BY scraped_at DESC LIMIT 1
                ) s ON TRUE
                WHERE r.is_cleared = FALSE AND f.is_valid_hardware = TRUE;
                """
            )
            active_rows = cur.fetchall()

        if active_rows:
            act_ids = [r[0] for r in active_rows]
            act_X = np.array([[str(r[1]), str(r[2]), str(r[3]), int(r[4]), int(r[5]), int(r[6]), int(r[7]), int(r[8]), int(r[9]), int(r[10])] for r in active_rows], dtype=object)
            act_prices = np.array([float(r[11]) for r in active_rows], dtype=np.float32)

            predictions = model.evaluate_candidates(act_ids, act_X, act_prices)

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
                        (p.listing_id, p.evaluated_price, p.q10, p.q50, p.q90, p.expected_net_profit, p.trigger_llm_gate)
                    )
            conn.commit()

        print("[4/5] Running Post-Gate LLM Audit on sub-Q10 anomalies...")
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.listing_id, f.canonical_sku, r.title, r.description, r.seller_rating,
                       d.evaluated_price, d.q10_price, d.q50_price, d.expected_net_profit
                FROM deal_evaluations d
                JOIN raw_listings r ON r.id = d.listing_id
                JOIN extracted_features f ON f.content_hash = r.content_hash
                WHERE d.triggered_llm_gate = TRUE AND d.audit_verdict IS NULL;
                """
            )
            gated_rows = cur.fetchall()

        if gated_rows and client:
            candidates = [
                {
                    "listing_id": r[0], "canonical_sku": r[1], "title": r[2], "description": r[3],
                    "seller_rating": float(r[4] or 98.0), "evaluated_price": float(r[5]),
                    "q10": float(r[6]), "q50": float(r[7]), "expected_net_profit": float(r[8])
                }
                for r in gated_rows
            ]
            audits = audit_flagged_deals(client, candidates)
            with conn.cursor() as cur:
                for a in audits:
                    cur.execute(
                        """
                        UPDATE deal_evaluations
                        SET audit_verdict = %s, trap_flags = %s, arbitrage_rationale = %s
                        WHERE listing_id = %s;
                        """,
                        (a.audit_verdict, a.trap_flags, a.arbitrage_rationale, a.listing_id)
                    )
            conn.commit()

        print("[5/5] Pipeline run completed successfully.")

if __name__ == "__main__":
    run_pipeline()