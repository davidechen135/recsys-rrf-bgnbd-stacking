"""
Customer Lifetime Value and Future Value Out-of-Time Evaluation Protocol
Author: Chen Davide (A0351457B)
Course: IE5005 Data Analytics for Industrial Engineers
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lifetimes.utils import calibration_and_holdout_data
from src.config import CALIBRATION_END_DATE, OUTPUT_DIR
from src.models.clv import (
    KMeansCLV,
    LinearRegressionCLV,
    RidgeCLV,
    KNNCLV,
    RandomForestCLV,
    GradientBoostingCLV,
    BGNBDGammaGammaCLV,
    ProbabilisticGuidedHybridCLV
)

class CLVEvaluator:
    """
    Executes Out-of-Time (OOT) holdout validation for customer value forecasting.
    """

    def __init__(self, df_cust: pd.DataFrame):
        self.df_cust = df_cust

    def evaluate_all(self) -> pd.DataFrame:
        max_date = self.df_cust['InvoiceDate'].max()
        cal_hold = calibration_and_holdout_data(
            self.df_cust,
            customer_id_col='CustomerID',
            datetime_col='InvoiceDate',
            monetary_value_col='TotalRevenue',
            calibration_period_end=CALIBRATION_END_DATE,
            observation_period_end=max_date,
            freq='D'
        )

        holdout_tx = self.df_cust[self.df_cust['InvoiceDate'] > CALIBRATION_END_DATE]
        actual_future_spend = holdout_tx.groupby('CustomerID')['TotalRevenue'].sum()
        cal_hold['actual_future_spend'] = cal_hold.index.map(actual_future_spend).fillna(0.0)

        X = pd.DataFrame({
            'Recency_Cal': cal_hold['recency_cal'],
            'Tenure_Cal': cal_hold['T_cal'],
            'Frequency_Cal': cal_hold['frequency_cal'],
            'Monetary_Cal': cal_hold['monetary_value_cal'].fillna(0.0),
            'Days_Inactive': cal_hold['T_cal'] - cal_hold['recency_cal']
        })
        y = cal_hold['actual_future_spend'].values

        holdout_days = (pd.to_datetime(max_date) - pd.to_datetime(CALIBRATION_END_DATE)).days

        models = {
            "Lecture 8: K-Means Segmentation (Cluster Mean)": (KMeansCLV(), "Lecture Curriculum Method", "Unsupervised Partitioning"),
            "Lecture 5: Multiple Linear Regression (OLS)": (LinearRegressionCLV(), "Lecture Curriculum Method", "Parametric Linear Regression"),
            "Lecture 7: Ridge Regression (L2 Regularization)": (RidgeCLV(), "Lecture Curriculum Method", "Regularized Linear Shrinkage"),
            "Lecture 7: K-Nearest Neighbors (KNN Regressor)": (KNNCLV(), "Lecture Curriculum Method", "Non-parametric Instance-based"),
            "Advanced ML 1: Random Forest Regressor (100 Trees)": (RandomForestCLV(), "Advanced Extra-curricular ML", "Ensemble Decision Tree Bagging"),
            "Advanced ML 2: Gradient Boosted Trees (GBDT)": (GradientBoostingCLV(), "Advanced Extra-curricular ML", "Gradient Boosting (Stage-wise Optimization)"),
            "Frontier SOTA: BG/NBD + Gamma-Gamma Probabilistic": (BGNBDGammaGammaCLV(), "Frontier SOTA DS / Ad Targeting", "Continuous-time Latent Stochastic Process"),
            "Proposed SOTA: Probabilistic-Guided Stacking Ensemble": (ProbabilisticGuidedHybridCLV(), "Proposed SOTA Hybrid", "Stacking BG/NBD Priors with Gradient Boosting Trees")
        }

        print(f"[CLVEvaluator] Training and evaluating {len(models)} CLV models across Q4 holdout...")
        top_10_actual_thresh = np.percentile(y, 90)
        top_10_actual_mask = y >= top_10_actual_thresh
        total_actual_holdout_spend = y.sum()

        records = []
        for name, (model, cat, paradigm) in models.items():
            model.fit(X, y, cal_df=cal_hold)
            pred = model.predict(X, cal_df=cal_hold, holdout_days=holdout_days)
            pred = np.nan_to_num(pred, nan=0.0)

            mae = mean_absolute_error(y, pred)
            rmse = np.sqrt(mean_squared_error(y, pred))
            spearman_corr, _ = spearmanr(y, pred)

            # Precision in identifying the actual top 10% high-value accounts
            top_10_pred_thresh = np.percentile(pred, 90)
            top_10_pred_mask = pred >= top_10_pred_thresh
            top_decile_precision = (top_10_actual_mask & top_10_pred_mask).sum() / top_10_pred_mask.sum() * 100.0

            # Revenue Capture @ Top 20% Budget: If marketing targets top 20% predicted spenders, what % of actual revenue is captured?
            top_20_pred_thresh = np.percentile(pred, 80)
            top_20_pred_mask = pred >= top_20_pred_thresh
            revenue_capture_top20 = (y[top_20_pred_mask].sum() / total_actual_holdout_spend * 100.0) if total_actual_holdout_spend > 0 else 0.0

            records.append({
                "Model": name,
                "Holdout MAE (£)": f"£{mae:.2f}",
                "Holdout RMSE (£)": f"£{rmse:.2f}",
                "Spearman Rank Correlation (ρ)": f"{spearman_corr:.4f}",
                "Top-Decile Precision (%)": f"{top_decile_precision:.2f}%",
                "Revenue Capture @ Top 20% Budget (%)": f"{revenue_capture_top20:.2f}%",
                "Category": cat,
                "Model Paradigm": paradigm
            })

        df_bench = pd.DataFrame(records)
        df_bench.to_csv(OUTPUT_DIR / "benchmark_clv_expanded.csv", index=False)
        return df_bench
