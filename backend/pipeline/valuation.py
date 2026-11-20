from dataclasses import dataclass
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder

CATEGORICAL_COLS = ["source", "category", "canonical_sku", "brand_tier"]
NUMERIC_COLS = [
    "vram_or_ram_gb",
    "perf_score",
    "launch_year",
    "msrp_eur",
    "condition_score",
    "has_warranty",
    "has_original_box",
    "is_mined_or_repaired",
    "seller_rating",
]
ALL_FEATURES = CATEGORICAL_COLS + NUMERIC_COLS
CAT_INDICES = list(range(len(CATEGORICAL_COLS)))

# Static hardware priors: (relative_1440p_perf_score, launch_year, launch_msrp_eur)
GPU_SPECS: dict[str, tuple[float, int, float]] = {
    "RTX 4070 TI SUPER 16GB": (148.0, 2024, 889.0),
    "RTX 4070 TI 12GB": (136.0, 2023, 899.0),
    "RTX 4070 SUPER 12GB": (127.0, 2024, 659.0),
    "RTX 4070 12GB": (108.0, 2023, 659.0),
    "RTX 3080 TI 12GB": (124.0, 2021, 1199.0),
    "RTX 3080 12GB": (115.0, 2022, 850.0),
    "RTX 3080 10GB": (108.0, 2020, 719.0),
    "RTX 3070 TI 8GB": (88.0, 2021, 619.0),
    "RTX 3070 8GB": (82.0, 2020, 519.0),
    "RTX 3060 TI 8GB": (72.0, 2020, 419.0),
    "RTX 3060 12GB": (54.0, 2021, 329.0),
    "RTX 3060 8GB": (46.0, 2022, 309.0),
    "RX 6800 XT 16GB": (110.0, 2020, 649.0),
    "RX 6800 16GB": (96.0, 2020, 579.0),
    "RX 6750 XT 12GB": (80.0, 2022, 549.0),
    "RX 6700 XT 12GB": (74.0, 2021, 479.0),
}


@dataclass
class ValuationPrediction:
    listing_id: int
    evaluated_price: float
    q10: float
    q50: float
    q90: float
    expected_net_profit: float
    trigger_llm_gate: bool


@dataclass
class BenchmarkReport:
    n_evaluated: int
    lgb_q10_pinball: float
    naive_q10_pinball: float
    lgb_q50_pinball: float
    naive_q50_pinball: float
    lgb_q10_coverage: float
    naive_q10_coverage: float


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


def compute_survivorship_weights(
    is_cleared: np.ndarray, days_on_market: np.ndarray
) -> np.ndarray:
    """
    Capped weighting (2.0x / 0.5x) until cleared-vs-delisted noise is validated.
    days_on_market is ONLY used here, never as a predictive feature.
    """
    weights = np.ones_like(days_on_market, dtype=np.float32)
    weights[is_cleared & (days_on_market <= 14)] = 2.0
    weights[(~is_cleared) & (days_on_market > 14)] = 0.50
    return weights


def build_feature_matrices(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    cat_rows, num_rows = [], []
    for r in rows:
        sku = str(r["canonical_sku"])
        perf, year, msrp = GPU_SPECS.get(sku, (75.0, 2021, 500.0))
        cat_rows.append([
            str(r["source"]),
            str(r["category"]),
            sku,
            str(r["brand_tier"]),
        ])
        num_rows.append([
            float(r["vram_or_ram_gb"] or 0),
            perf,
            float(year),
            msrp,
            float(r["condition_score"]),
            float(int(bool(r["has_warranty"]))),
            float(int(bool(r["has_original_box"]))),
            float(int(bool(r["is_mined_or_repaired"]))),
            float(r["seller_rating"] if r["seller_rating"] is not None else 98.0),
        ])
    return np.array(cat_rows, dtype=object), np.array(num_rows, dtype=np.float32)


class QuantileArbitrageModel:
    def __init__(self, resale_fee_pct: float = 0.10, min_net_profit: float = 35.0):
        self.resale_fee_pct = resale_fee_pct
        self.min_net_profit = min_net_profit

    def _fit_fold_models(
        self, X_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray
    ) -> dict[float, lgb.Booster]:
        train_set = lgb.Dataset(
            X_train,
            label=y_train,
            weight=w_train,
            feature_name=ALL_FEATURES,
            categorical_feature=CAT_INDICES,
            free_raw_data=False,
        )
        models: dict[float, lgb.Booster] = {}
        min_leaf = max(5, min(15, len(y_train) // 25))
        for alpha in (0.10, 0.50, 0.90):
            params = {
                "objective": "quantile",
                "alpha": alpha,
                "metric": "quantile",
                "learning_rate": 0.05,
                "num_leaves": 15,
                "min_data_in_leaf": min_leaf,
                "feature_fraction": 0.85,
                "verbose": -1,
            }
            models[alpha] = lgb.train(params, train_set, num_boost_round=180)
        return models

    def fit_and_evaluate_oof(
        self, rows: list[dict]
    ) -> tuple[list[ValuationPrediction], BenchmarkReport | None]:
        """
        Scores active listings strictly out-of-fold (5-fold CV) while keeping
        all cleared listings in every training fold, then benchmarks LightGBM
        against a naive per-SKU empirical quantile baseline on the exact same folds.
        """
        active_indices = [i for i, r in enumerate(rows) if not r["is_cleared"]]
        cleared_indices = [i for i, r in enumerate(rows) if r["is_cleared"]]

        if len(active_indices) < 20:
            return [], None

        X_cat_all, X_num_all = build_feature_matrices(rows)
        y_all = np.array([float(r["total_price"]) for r in rows], dtype=np.float32)
        cleared_all = np.array([bool(r["is_cleared"]) for r in rows], dtype=bool)
        dom_all = np.array([int(r["days_on_market"]) for r in rows], dtype=np.int32)
        weights_all = compute_survivorship_weights(cleared_all, dom_all)
        skus_all = np.array([str(r["canonical_sku"]) for r in rows], dtype=object)

        n_splits = min(5, len(active_indices) // 4)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        oof_q10 = np.zeros(len(active_indices), dtype=np.float32)
        oof_q50 = np.zeros(len(active_indices), dtype=np.float32)
        oof_q90 = np.zeros(len(active_indices), dtype=np.float32)
        naive_q10 = np.zeros(len(active_indices), dtype=np.float32)
        naive_q50 = np.zeros(len(active_indices), dtype=np.float32)

        active_idx_arr = np.array(active_indices, dtype=int)
        cleared_idx_arr = np.array(cleared_indices, dtype=int)

        for train_pos, val_pos in kf.split(active_idx_arr):
            train_global = np.concatenate([cleared_idx_arr, active_idx_arr[train_pos]])
            val_global = active_idx_arr[val_pos]

            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_tr_cat = enc.fit_transform(X_cat_all[train_global]).astype(np.float32)
            X_va_cat = enc.transform(X_cat_all[val_global]).astype(np.float32)

            X_tr = np.hstack([X_tr_cat, X_num_all[train_global]])
            X_va = np.hstack([X_va_cat, X_num_all[val_global]])

            models = self._fit_fold_models(X_tr, y_all[train_global], weights_all[train_global])
            oof_q10[val_pos] = models[0.10].predict(X_va)
            oof_q50[val_pos] = models[0.50].predict(X_va)
            oof_q90[val_pos] = models[0.90].predict(X_va)

            # Out-of-fold naive per-SKU percentile baseline
            global_q10 = float(np.quantile(y_all[train_global], 0.10))
            global_q50 = float(np.quantile(y_all[train_global], 0.50))
            sku_q10_map: dict[str, float] = {}
            sku_q50_map: dict[str, float] = {}
            for sku in np.unique(skus_all[train_global]):
                mask = skus_all[train_global] == sku
                if np.sum(mask) >= 5:
                    sku_q10_map[sku] = float(np.quantile(y_all[train_global][mask], 0.10))
                    sku_q50_map[sku] = float(np.quantile(y_all[train_global][mask], 0.50))

            for local_i, g_idx in zip(val_pos, val_global):
                sku = skus_all[g_idx]
                naive_q10[local_i] = sku_q10_map.get(sku, global_q10)
                naive_q50[local_i] = sku_q50_map.get(sku, global_q50)

        y_active = y_all[active_idx_arr]
        report = BenchmarkReport(
            n_evaluated=len(active_indices),
            lgb_q10_pinball=round(pinball_loss(y_active, oof_q10, 0.10), 2),
            naive_q10_pinball=round(pinball_loss(y_active, naive_q10, 0.10), 2),
            lgb_q50_pinball=round(pinball_loss(y_active, oof_q50, 0.50), 2),
            naive_q50_pinball=round(pinball_loss(y_active, naive_q50, 0.50), 2),
            lgb_q10_coverage=round(float(np.mean(y_active < oof_q10)), 3),
            naive_q10_coverage=round(float(np.mean(y_active < naive_q10)), 3),
        )

        predictions: list[ValuationPrediction] = []
        for pos, g_idx in enumerate(active_idx_arr):
            row = rows[g_idx]
            ask = float(row["total_price"])
            q50_clean = float(oof_q50[pos])
            q10_clean = float(min(oof_q10[pos], q50_clean))
            q90_clean = float(max(oof_q90[pos], q50_clean))

            net_resale = q50_clean * (1.0 - self.resale_fee_pct)
            expected_profit = round(net_resale - ask, 2)
            trigger = (ask < q10_clean) and (expected_profit >= self.min_net_profit)

            predictions.append(
                ValuationPrediction(
                    listing_id=int(row["id"]),
                    evaluated_price=round(ask, 2),
                    q10=round(q10_clean, 2),
                    q50=round(q50_clean, 2),
                    q90=round(q90_clean, 2),
                    expected_net_profit=expected_profit,
                    trigger_llm_gate=trigger,
                )
            )

        return predictions, report