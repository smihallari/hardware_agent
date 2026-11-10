-- 1. Raw ingestion & cross-post tracking
CREATE TABLE raw_listings (
    id BIGSERIAL PRIMARY KEY,
    source VARCHAR(32) NOT NULL,          -- e.g., 'ebay', 'insomnia', 'vendora'
    external_id VARCHAR(128) NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    seller_id VARCHAR(128),
    seller_rating NUMERIC(4,2),
    content_hash CHAR(64) NOT NULL,       -- SHA-256(lower(title || description))
    raw_payload JSONB NOT NULL,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_cleared BOOLEAN NOT NULL DEFAULT FALSE, -- True when listing disappears < 14 days
    UNIQUE (source, external_id)
);

CREATE INDEX idx_raw_content_hash ON raw_listings(content_hash);
CREATE INDEX idx_raw_cleared ON raw_listings(is_cleared, last_seen_at);

-- 2. Time-series price & velocity tracking
CREATE TABLE listing_snapshots (
    id BIGSERIAL PRIMARY KEY,
    listing_id BIGINT NOT NULL REFERENCES raw_listings(id) ON DELETE CASCADE,
    asking_price NUMERIC(10,2) NOT NULL,
    shipping_cost NUMERIC(10,2) NOT NULL DEFAULT 0.00,
    scraped_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_snapshots_listing_time ON listing_snapshots(listing_id, scraped_at DESC);

-- 3. Hash-cached tabular features (Upstream of LightGBM)
CREATE TABLE extracted_features (
    content_hash CHAR(64) PRIMARY KEY,
    category VARCHAR(32) NOT NULL,        -- 'GPU', 'CPU', 'LAPTOP', 'ACCESSORY_JUNK'
    canonical_sku VARCHAR(64) NOT NULL,   -- e.g., 'RTX 3080 10GB', 'RYZEN 7 5800X3D'
    brand_tier VARCHAR(16) NOT NULL,      -- 'FE_REF', 'LOW', 'MID', 'FLAGSHIP' (e.g., ROG Strix)
    vram_or_ram_gb SMALLINT,
    storage_gb SMALLINT,
    condition_score SMALLINT NOT NULL,    -- 0=Broken/Parts, 1=Heavily Used, 2=Good, 3=Like New/Boxed, 4=Sealed
    has_warranty BOOLEAN NOT NULL DEFAULT FALSE,
    has_original_box BOOLEAN NOT NULL DEFAULT FALSE,
    is_mined_or_repaired BOOLEAN NOT NULL DEFAULT FALSE,
    is_valid_hardware BOOLEAN NOT NULL DEFAULT TRUE, -- False for "Box only", "Cooler only", "Wanted"
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4. Downstream LightGBM Quantiles & Post-Gate LLM Audit
CREATE TABLE deal_evaluations (
    listing_id BIGINT PRIMARY KEY REFERENCES raw_listings(id) ON DELETE CASCADE,
    evaluated_price NUMERIC(10,2) NOT NULL, -- asking_price + shipping_cost
    q10_price NUMERIC(10,2) NOT NULL,
    q50_price NUMERIC(10,2) NOT NULL,
    q90_price NUMERIC(10,2) NOT NULL,
    expected_net_profit NUMERIC(10,2) NOT NULL, -- (q50_price * (1 - platform_fee)) - evaluated_price
    triggered_llm_gate BOOLEAN NOT NULL DEFAULT FALSE,
    audit_verdict VARCHAR(16),            -- 'BUY', 'REJECT_TRAP', 'MANUAL_REVIEW'
    trap_flags TEXT[],                    -- e.g., {'THERMAL_ARTIFACTING', 'NO_PAYPAL_PROTECTION'}
    arbitrage_rationale TEXT,
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);