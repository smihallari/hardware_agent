from dataclasses import dataclass
import lightgbm as lgb
import numpy as np

CATEGORICAL_COLS = ["category", "canonical_sku", "brand_tier"]
NUMERIC_COLS = [
    "vram_or_ram_gb",
    "storage_gb",
    "condition_score",
    "has_warranty",
    "has_original_box",
    "is_mined_or_repaired",
    "days_on_market",
]

@dataclass
class ValuationPrediction:
    listing_id: int
    evaluated_price: float
    q10: float
    q50: float
    q90: float
    expected_net_profit: float
    trigger_llm_gate: bool


def compute_survivorship_weights(is_cleared: np.ndarray, days_on_market: np.ndarray) -> np.ndarray:
    """
    Fast-clearing listings represent true liquidity.
    Stagnant active listings (>14 days) are down-weighted so LightGBM doesn't learn scalper prices.
    """
    weights = np.ones_like(days_on_market, dtype=np.float32)
    weights[is_cleared & (days_on_market <= 14)] = 3.0
    weights[(~is_cleared) & (days_on_market > 14)] = 0.35
    return weights


class QuantileArbitrageModel:
    def __init__(self, resale_fee_pct: float = 0.10, min_net_profit: float = 35.0):
        self.resale_fee_pct = resale_fee_pct
        self.min_net_profit = min_net_profit
        self.models: dict[float, lgb.Booster] = {}

    def fit(
        self,
        X: np.ndarray,
        y_price: np.ndarray,
        is_cleared: np.ndarray,
        days_on_market: np.ndarray,
    ) -> None:
        weights = compute_survivorship_weights(is_cleared, days_on_market)
        train_set = lgb.Dataset(
            X,
            label=y_price,
            weight=weights,
            feature_name=CATEGORICAL_COLS + NUMERIC_COLS,
            categorical_feature=CATEGORICAL_COLS,
            free_raw_data=False,
        )

        for alpha in (0.10, 0.50, 0.90):
            params = {
                "objective": "quantile",
                "alpha": alpha,
                "metric": "quantile",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_data_in_leaf": 8,  # Prevents overfitting on rare GPU SKUs
                "verbose": -1,
            }
            self.models[alpha] = lgb.train(params, train_set, num_boost_round=250)

    def evaluate_candidates(
        self,
        listing_ids: list[int],
        X_cand: np.ndarray,
        total_Delivered_prices: np.ndarray,
    ) -> list[ValuationPrediction]:
        q10_preds = self.models[0.10].predict(X_cand)
        q50_preds = self.models[0.50].predict(X_cand)
        q90_preds = self.models[0.90].predict(X_cand)

        results: list[ValuationPrediction] = []
        for lid, ask, q10, q50, q90 in zip(
            listing_ids, total_Delivered_prices, q10_preds, q50_preds, q90_preds
        ):
            # Enforce monotonicity in case quantile trees cross on sparse SKUs
            q10_clean = float(min(q10, q50))
            q50_clean = float(q50)
            q90_clean = float(max(q50, q90))

            net_resale = q50_clean * (1.0 - self.resale_fee_pct)
            expected_profit = round(net_resale - float(ask), 2)

            # Gate Rule: Must beat the 10th percentile floor AND clear minimum euro profit after fees
            is_arbitrage_candidate = (float(ask) < q10_clean) and (expected_profit >= self.min_net_profit)

            results.append(
                ValuationPrediction(
                    listing_id=lid,
                    evaluated_price=round(float(ask), 2),
                    q10=round(q10_clean, 2),
                    q50=round(q50_clean, 2),
                    q90=round(q90_clean, 2),
                    expected_net_profit=expected_profit,
                    trigger_llm_gate=is_arbitrage_candidate,
                )
            )
        return results