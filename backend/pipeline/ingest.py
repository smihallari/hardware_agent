import base64
from dataclasses import dataclass
import html
import json
import os
import re
import time
from typing import Any
import httpx
import psycopg
from backend.pipeline.extractor import compute_content_hash

EBAY_TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
EBAY_BROWSE_SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
EBAY_BROWSE_ITEM_URL = "https://api.ebay.com/buy/browse/v1/item/{item_id}"

# Leaf category for Graphics/Video Cards (identical across EBAY_DE, EBAY_FR, EBAY_IT)
GPU_CATEGORY_ID = "27386"

# High-liquidity GPUs so quantile estimation has real sample density
TARGET_GPU_QUERIES = [
    "RTX 3060 12GB",
    "RTX 3070",
    "RTX 3080",
    "RTX 4070",
    "RX 6700 XT",
    "RX 6800 XT",
]

DEFAULT_MARKETPLACES = ("EBAY_DE", "EBAY_FR", "EBAY_IT")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


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
    raw_payload: dict[str, Any]


def strip_html(raw_html: str) -> str:
    text = _HTML_TAG_RE.sub(" ", raw_html or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


class EbayBrowseClient:
    def __init__(self) -> None:
        self.client_id = os.environ["EBAY_CLIENT_ID"]
        self.client_secret = os.environ["EBAY_CLIENT_SECRET"]
        self._token: str | None = None
        self._token_expires_at: float = 0.0
        self._http = httpx.Client(timeout=20.0)

    def close(self) -> None:
        self._http.close()

    def _get_access_token(self) -> str:
        if self._token and time.time() < self._token_expires_at - 60:
            return self._token

        creds = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode("utf-8")
        ).decode("utf-8")
        resp = self._http.post(
            EBAY_TOKEN_URL,
            headers={
                "Authorization": f"Basic {creds}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "grant_type": "client_credentials",
                "scope": "https://api.ebay.com/oauth/api_scope",
            },
        )
        resp.raise_for_status()
        payload = resp.json()
        self._token = payload["access_token"]
        self._token_expires_at = time.time() + int(payload.get("expires_in", 7200))
        return self._token

    def _headers(self, marketplace_id: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._get_access_token()}",
            "X-EBAY-C-MARKETPLACE-ID": marketplace_id,
        }

    def search_used_gpus(
        self, marketplace_id: str, query: str, limit: int = 200
    ) -> list[ScrapedListing]:
        params = {
            "q": query,
            "category_ids": GPU_CATEGORY_ID,
            "filter": "buyingOptions:{FIXED_PRICE},conditions:{USED}",
            "fieldgroups": "EXTENDED",
            "limit": min(limit, 200),
        }
        resp = self._http.get(
            EBAY_BROWSE_SEARCH_URL,
            headers=self._headers(marketplace_id),
            params=params,
        )
        resp.raise_for_status()
        rows = resp.json().get("itemSummaries", [])

        source = marketplace_id.lower()
        out: list[ScrapedListing] = []
        for row in rows:
            price_obj = row.get("price", {})
            if price_obj.get("currency") != "EUR":
                continue
            # Explicitly drop eBay conditionId 7000 ("For parts or not working")
            if str(row.get("conditionId", "")) == "7000":
                continue

            try:
                asking_price = float(price_obj.get("value", 0.0))
            except (TypeError, ValueError):
                continue
            if asking_price <= 0:
                continue

            ship_opts = row.get("shippingOptions") or []
            shipping_cost = 0.0
            if ship_opts:
                ship_obj = ship_opts[0].get("shippingCost", {})
                if ship_obj.get("currency", "EUR") == "EUR":
                    try:
                        shipping_cost = float(ship_obj.get("value", 0.0))
                    except (TypeError, ValueError):
                        shipping_cost = 0.0

            seller = row.get("seller") or {}
            try:
                rating = min(100.0, max(0.0, float(seller.get("feedbackPercentage", 98.0))))
            except (TypeError, ValueError):
                rating = 98.0

            title = row.get("title", "").strip()
            short_desc = row.get("shortDescription", "").strip()

            out.append(
                ScrapedListing(
                    source=source,
                    external_id=row["itemId"],
                    url=row.get("itemWebUrl", ""),
                    title=title,
                    description=short_desc or title,
                    seller_id=seller.get("username", "unknown"),
                    seller_rating=rating,
                    asking_price=asking_price,
                    shipping_cost=shipping_cost,
                    raw_payload=row,
                )
            )
        return out

    def fetch_item_detail(self, marketplace_id: str, item_id: str) -> str | None:
        """Lazy full-description hydration called ONLY for sub-Q10 candidates."""
        resp = self._http.get(
            EBAY_BROWSE_ITEM_URL.format(item_id=item_id),
            headers=self._headers(marketplace_id),
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        payload = resp.json()

        parts: list[str] = []
        if payload.get("conditionDescription"):
            parts.append(f"Condition note: {payload['conditionDescription']}")
        if payload.get("description"):
            parts.append(strip_html(payload["description"]))
        elif payload.get("shortDescription"):
            parts.append(payload["shortDescription"])

        combined = " | ".join(p for p in parts if p)
        return combined[:2500] if combined else None


def upsert_and_track_snapshots(
    conn: psycopg.Connection,
    source: str,
    listings: list[ScrapedListing],
    mark_missing_cleared: bool = True,
) -> int:
    """
    Deduplicates by external_id, upserts active listings, logs price changes,
    and marks listings absent from a complete marketplace sweep as is_cleared.
    """
    deduped = {item.external_id: item for item in listings}
    seen_external_ids = list(deduped.keys())

    with conn.cursor() as cur:
        for item in deduped.values():
            c_hash = compute_content_hash(item.title, item.description)

            # Preserve previously hydrated full description if longer than search snippet
            cur.execute(
                """
                INSERT INTO raw_listings (
                    source, external_id, url, title, description,
                    seller_id, seller_rating, content_hash, raw_payload,
                    first_seen_at, last_seen_at, is_cleared
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, NOW(), NOW(), FALSE)
                ON CONFLICT (source, external_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    description = CASE
                        WHEN LENGTH(COALESCE(raw_listings.description, '')) > LENGTH(COALESCE(EXCLUDED.description, ''))
                        THEN raw_listings.description
                        ELSE EXCLUDED.description
                    END,
                    seller_rating = EXCLUDED.seller_rating,
                    raw_payload = EXCLUDED.raw_payload,
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
            listing_id = cur.fetchone()["id"]

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
                float(last_snap["asking_price"]) != item.asking_price
                or float(last_snap["shipping_cost"]) != item.shipping_cost
            ):
                cur.execute(
                    """
                    INSERT INTO listing_snapshots (listing_id, asking_price, shipping_cost)
                    VALUES (%s, %s, %s);
                    """,
                    (listing_id, item.asking_price, item.shipping_cost),
                )

        if mark_missing_cleared and seen_external_ids:
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
    return len(seen_external_ids)