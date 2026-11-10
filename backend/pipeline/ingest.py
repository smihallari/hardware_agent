from dataclasses import dataclass
import json
import random
import urllib.request
import psycopg
from backend.pipeline.extractor import compute_content_hash, passes_fast_gate


@dataclass
class ScrapedListing:
    source: str
    external_id: str
    url: str
    title: str
    description: str
    seller_id: str
    seller_rating: float
    asking_price: float
    shipping_cost: float
    raw_payload: dict


def upsert_and_track_snapshots(
    conn: psycopg.Connection,
    source: str,
    listings: list[ScrapedListing],
) -> list[int]:
    """
    Upserts live listings, records price changes in listing_snapshots,
    and marks missing recent listings from the same source as is_cleared = TRUE.
    Returns active listing_ids that passed the Stage 1 regex gate.
    """
    seen_external_ids: list[str] = []
    valid_listing_ids: list[int] = []

    with conn.cursor() as cur:
        for item in listings:
            seen_external_ids.append(item.external_id)
            c_hash = compute_content_hash(item.title, item.description)

            cur.execute(
                """
                INSERT INTO raw_listings (
                    source, external_id, url, title, description,
                    seller_id, seller_rating, content_hash, raw_payload,
                    first_seen_at, last_seen_at, is_cleared
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW(), FALSE)
                ON CONFLICT (source, external_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    content_hash = EXCLUDED.content_hash,
                    last_seen_at = NOW(),
                    is_cleared = FALSE
                RETURNING id;
                """,
                (
                    item.source,
                    item.external_id,
                    item.url,
                    item.title,
                    item.description,
                    item.seller_id,
                    item.seller_rating,
                    c_hash,
                    json.dumps(item.raw_payload),
                ),
            )
            listing_id = cur.fetchone()[0]

            # Only insert a new snapshot row if it's the first snapshot or the price changed
            cur.execute(
                """
                SELECT asking_price, shipping_cost
                FROM listing_snapshots
                WHERE listing_id = %s
                ORDER BY scraped_at DESC
                LIMIT 1;
                """,
                (listing_id,),
            )
            last_snap = cur.fetchone()
            if not last_snap or (
                float(last_snap[0]) != item.asking_price
                or float(last_snap[1]) != item.shipping_cost
            ):
                cur.execute(
                    """
                    INSERT INTO listing_snapshots (listing_id, asking_price, shipping_cost)
                    VALUES (%s, %s, %s);
                    """,
                    (listing_id, item.asking_price, item.shipping_cost),
                )

            if passes_fast_gate(item.title, item.description, item.asking_price):
                valid_listing_ids.append(listing_id)

        # Mark listings from this source that disappeared within 14 days as market-cleared
        if seen_external_ids:
            cur.execute(
                """
                UPDATE raw_listings
                SET is_cleared = TRUE
                WHERE source = %s
                  AND NOT (external_id = ANY(%s))
                  AND is_cleared = FALSE
                  AND (last_seen_at - first_seen_at) <= INTERVAL '14 days';
                """,
                (source, seen_external_ids),
            )

    conn.commit()
    return valid_listing_ids


def fetch_ebay_browse_feed(oauth_token: str, query: str = "RTX 3080") -> list[ScrapedListing]:
    """
    Fetches live active Buy-It-Now hardware listings from eBay Browse REST API.
    """
    url = (
        "https://api.ebay.com/buy/browse/v1/item_summary/search"
        f"?q={urllib.parse.quote(query)}&filter=buyingOptions:{{FIXED_PRICE}}&limit=50"
    )
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {oauth_token}",
            "X-EBAY-C-MARKETPLACE-ID": "EBAY_DE",  # EU Euro pricing
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    out: list[ScrapedListing] = []
    for row in data.get("itemSummaries", []):
        price_val = float(row.get("price", {}).get("value", 0.0))
        ship_opts = row.get("shippingOptions", [{}])
        ship_val = float(ship_opts[0].get("shippingCost", {}).get("value", 0.0)) if ship_opts else 0.0
        seller = row.get("seller", {})

        out.append(
            ScrapedListing(
                source="ebay_de",
                external_id=row["itemId"],
                url=row.get("itemWebUrl", ""),
                title=row.get("title", ""),
                description=row.get("shortDescription", row.get("title", "")),
                seller_id=seller.get("username", "unknown"),
                seller_rating=float(seller.get("feedbackPercentage", 98.0)),
                asking_price=price_val,
                shipping_cost=ship_val,
                raw_payload=row,
            )
        )
    return out


def seed_historical_baseline(conn: psycopg.Connection) -> None:
    """
    Populates Neon with realistic historical EU hardware listings + features
    so LightGBM has immediate quantile training support on initial deployment.
    """
    rng = random.Random(42)
    skus = [
        ("GPU", "RTX 3080 10GB", 10, 390.0),
        ("GPU", "RTX 4070 SUPER 12GB", 12, 540.0),
        ("GPU", "RX 6800 XT 16GB", 16, 375.0),
        ("GPU", "RTX 3070 8GB", 8, 265.0),
        ("CPU", "RYZEN 7 7800X3D", 0, 345.0),
        ("CPU", "RYZEN 7 5800X3D", 0, 255.0),
    ]
    tiers = [("LOW", -20.0, "MSI Ventus"), ("MID", 0.0, "ASUS TUF"), ("FLAGSHIP", 30.0, "ROG Strix")]

    with conn.cursor() as cur:
        for i in range(180):
            cat, sku, vram, base_p = skus[i % len(skus)]
            tier_code, tier_adj, tier_name = tiers[i % len(tiers)]
            is_cleared = rng.random() < 0.65
            cond = rng.choice([2, 3, 3, 4])

            # Inject 6 genuine under-Q10 deals and 4 subtle defective traps
            if i < 6:
                price = round(base_p + tier_adj - rng.uniform(75, 110), 2)
                desc = f"{tier_name} {sku} in great working condition. Upgraded to 4090, quick sale, includes original box."
                is_cleared = False
            elif i < 10:
                price = round(base_p + tier_adj - rng.uniform(85, 120), 2)
                desc = f"{tier_name} {sku}. Boots to desktop fine but crashes in DX12 games after 10 minutes. Sold as-is, bank transfer only."
                is_cleared = False
            else:
                overprice_bias = 0.0 if is_cleared else rng.uniform(35.0, 80.0)
                price = round(base_p + tier_adj + (cond - 2.5) * 15 + overprice_bias + rng.gauss(0, 14), 2)
                desc = f"Selling my {tier_name} {sku}, works flawlessly, never mined, low temps."

            title = f"{tier_name} {sku} - {'With Box' if cond >= 3 else 'Good Condition'}"
            c_hash = compute_content_hash(title, desc)

            cur.execute(
                """
                INSERT INTO raw_listings (
                    source, external_id, url, title, description,
                    seller_id, seller_rating, content_hash, raw_payload,
                    first_seen_at, last_seen_at, is_cleared
                )
                VALUES (
                    'seed_eu', %s, %s, %s, %s, %s, %s, %s, '{}'::jsonb,
                    NOW() - (%s || ' days')::interval, NOW(), %s
                )
                ON CONFLICT (source, external_id) DO NOTHING
                RETURNING id;
                """,
                (
                    f"seed-{i}",
                    f"https://www.ebay.de/itm/{100000 + i}",
                    title,
                    desc,
                    f"seller_{i % 40}",
                    99.2 if i >= 10 else 88.5,
                    c_hash,
                    str(rng.randint(2, 9) if is_cleared else rng.randint(15, 30)),
                    is_cleared,
                ),
            )
            row = cur.fetchone()
            if not row:
                continue

            cur.execute(
                "INSERT INTO listing_snapshots (listing_id, asking_price, shipping_cost) VALUES (%s, %s, 8.50);",
                (row[0], price),
            )
            cur.execute(
                """
                INSERT INTO extracted_features (
                    content_hash, category, canonical_sku, brand_tier,
                    vram_or_ram_gb, storage_gb, condition_score,
                    has_warranty, has_original_box, is_mined_or_repaired, is_valid_hardware
                )
                VALUES (%s, %s, %s, %s, %s, NULL, %s, %s, %s, FALSE, TRUE)
                ON CONFLICT (content_hash) DO NOTHING;
                """,
                (c_hash, cat, sku, tier_code, vram, cond, cond == 4, cond >= 3),
            )
    conn.commit()