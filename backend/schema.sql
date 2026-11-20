-- 1. Raw ingestion & cross-marketplace tracking
CREATE TABLE IF NOT EXISTS raw_listings (
    id BIGSERIAL PRIMARY KEY,
    source VARCHAR(32) NOT NULL,          -- 'ebay_de', 'ebay_fr', 'ebay_it'
    external_id VARCHAR(128) NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    seller_id VARCHAR(128),
    seller_rating NUMERIC(5,2),           -- Fixed: allows 100.00%
    content_hash CHAR(64) NOT NULL,
    raw_payload JSONB NOT NULL,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_cleared BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (source, external_id)
);

CREATE INDEX IF NOT EXISTS idx_raw_content_hash ON raw_listings(content_hash);
CREATE INDEX IF NOT EXISTS idx_raw_cleared ON raw_listings(is_cleared, last_seen_at);

-- 2. Time-series price & velocity tracking
CREATE TABLE IF NOT EXISTS listing_snapshots (
    id BIGSERIAL PRIMARY KEY,
    listing_id BIGINT NOT NULL REFERENCES raw_listings(id) ON DELETE CASCADE,
    asking_price NUMERIC(10,2) NOT NULL,
    shipping_cost NUMERIC(10,2) NOT NULL DEFAULT 0.00,
    scraped_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_snapshots_listing_time ON listing_snapshots(listing_id, scraped_at DESC);

-- 3. Hash-cached tabular features (Deterministic upstream of LightGBM)
CREATE TABLE IF NOT EXISTS extracted_features (
    content_hash CHAR(64) PRIMARY KEY,
    category VARCHAR(32) NOT NULL,
    canonical_sku VARCHAR(64) NOT NULL,
    brand_tier VARCHAR(16) NOT NULL,
    vram_or_ram_gb SMALLINT,
    storage_gb SMALLINT,
    condition_score SMALLINT NOT NULL,
    has_warranty BOOLEAN NOT NULL DEFAULT FALSE,
    has_original_box BOOLEAN NOT NULL DEFAULT FALSE,
    is_mined_or_repaired BOOLEAN NOT NULL DEFAULT FALSE,
    is_valid_hardware BOOLEAN NOT NULL DEFAULT TRUE,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4. Downstream LightGBM Quantiles & Post-Gate LLM Audit
CREATE TABLE IF NOT EXISTS deal_evaluations (
    listing_id BIGINT PRIMARY KEY REFERENCES raw_listings(id) ON DELETE CASCADE,
    evaluated_price NUMERIC(10,2) NOT NULL,
    q10_price NUMERIC(10,2) NOT NULL,
    q50_price NUMERIC(10,2) NOT NULL,
    q90_price NUMERIC(10,2) NOT NULL,
    expected_net_profit NUMERIC(10,2) NOT NULL,
    triggered_llm_gate BOOLEAN NOT NULL DEFAULT FALSE,
    audit_verdict VARCHAR(16),
    trap_flags TEXT[],
    arbitrage_rationale TEXT,
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);