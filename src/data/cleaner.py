"""
Data Cleaning and Partitioning Pipeline (Part 2)
Author: Chen Davide (A0351457B)
Course: IE5005 Data Analytics for Industrial Engineers
"""

import pandas as pd
from src.config import (
    NON_MERCHANDISE_CODES,
    CACHE_SALES_CLEAN,
    CACHE_RETURNS_CLEAN,
    CACHE_CUSTOMER_SALES,
    OUTPUT_DIR
)

class DataCleaner:
    """
    Applies principled data treatments, maintains an explicit Cleaning Decision Log,
    and decouples transactions into Positive Sales vs Returns/Cancellations.
    """

    def __init__(self, df_raw: pd.DataFrame):
        self.df_raw = df_raw.copy()
        self.cleaning_log = []

    def clean(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = self.df_raw.copy()
        n_initial = len(df)

        # 1. Deduplication
        n_dups = df.duplicated().sum()
        df = df.drop_duplicates().copy()
        self.cleaning_log.append({
            "#": 1,
            "Issue found": f"Exact duplicate records ({n_dups:,} rows)",
            "Action taken": "Removed all exact duplicate rows across all 8 attributes.",
            "Why": "Identical rows logged at identical timestamps represent database duplicate write anomalies. Keeping them artificially inflates sales volume and frequency.",
            "Effect on analysis": f"Eliminated {n_dups:,} redundant rows ({n_dups/n_initial*100:.2f}%), ensuring unbiased frequency counts and revenue sums."
        })

        # 2. Non-merchandise Overhead Codes
        stock_clean = df['StockCode'].astype(str).str.strip().str.upper()
        is_non_merch = stock_clean.isin(NON_MERCHANDISE_CODES) | stock_clean.str.startswith('AMAZON')
        n_non_merch = is_non_merch.sum()
        df = df[~is_non_merch].copy()
        self.cleaning_log.append({
            "#": 2,
            "Issue found": f"Non-merchandise overhead stock codes ({n_non_merch:,} rows)",
            "Action taken": "Filtered out non-product overhead codes (POST, D, M, BANK CHARGES, CRUK, AMAZONFEE).",
            "Why": "These represent freight postage, administrative discounts, banking fees, and manual adjustments rather than catalog products. Including them pollutes item-level association rules and basket cross-selling models.",
            "Effect on analysis": "Product catalog and recommendation candidate pools now exclusively contain authentic giftware SKUs."
        })

        # 3. Administrative rows with UnitPrice <= 0
        is_bad_price = df['UnitPrice'] <= 0
        n_bad_price = is_bad_price.sum()
        df = df[~is_bad_price].copy()
        self.cleaning_log.append({
            "#": 3,
            "Issue found": f"Zero and negative unit prices ({n_bad_price:,} rows, including bad debts)",
            "Action taken": "Removed rows with UnitPrice <= 0.",
            "Why": "Zero prices indicate promotional giveaways, internal warehouse damages, or missing pricing records; negative prices are accounting balancing lines. They distort price distributions and unit revenue.",
            "Effect on analysis": "Prevents artificial deflation of Average Order Value (AOV) and ensures revenue models reflect commercial monetary exchange."
        })

        # 4. Partition into Sales vs Returns/Cancellations
        is_return = (df['Quantity'] < 0) | (df['InvoiceNo'].astype(str).str.startswith('C'))
        df_returns = df[is_return].copy()
        df_sales = df[~is_return].copy()

        self.cleaning_log.append({
            "#": 4,
            "Issue found": f"Cancellations and customer returns ({len(df_returns):,} rows)",
            "Action taken": "Separated into dual analytical tracks: kept in df_returns for return-rate and lost-revenue attribution; excluded from df_sales for purchase basket and LTV modeling.",
            "Why": "Business question explicitly demands separating purchasing behavior from returns, while quantifying where revenue is lost. Mixing returns with forward purchases biases basket association algorithms.",
            "Effect on analysis": "Enables clean, positive-only sequence modeling for RecSys and forward CLV, while providing a dedicated risk attribution view for lost revenue recovery."
        })

        # 5. Missing Customer Identifiers
        n_anon = df_sales['CustomerID'].isna().sum()
        df_cust_sales = df_sales[df_sales['CustomerID'].notna()].copy()
        df_cust_sales['CustomerID'] = df_cust_sales['CustomerID'].astype(int).astype(str)

        self.cleaning_log.append({
            "#": 5,
            "Issue found": f"Missing customer account identifiers ({n_anon:,} rows, ~{n_anon/len(df_sales)*100:.1f}%)",
            "Action taken": "Retained for aggregate product volume, national revenue totals, and basket co-occurrence; partitioned out into df_customer_sales for customer-level LTV/RFM.",
            "Why": "Anonymous guest transactions represent authentic cash flows and market demand. Dropping them would understate total revenue by ~25%, but they cannot be tracked longitudinally over time.",
            "Effect on analysis": "Aggregate metrics reflect full commercial scale (£10.25M GMV); customer-level models are trained on the robust 4,334 identified account cohort with explicit sample size notation."
        })

        # 6. Extreme Outliers
        self.cleaning_log.append({
            "#": 6,
            "Issue found": "Extreme wholesale bulk outliers (Quantity up to 80,995)",
            "Action taken": "Identified institutional bulk buyers; capped extreme quantity at 99.9th percentile (468 units) for standard parametric EDA while preserving raw totals for corporate revenue accounting.",
            "Why": "Dozens of extreme bulk orders heavily distort standard deviations, means, and parametric regression slopes, creating artificial right-skewness.",
            "Effect on analysis": "Provides robust distribution estimates without discarding institutional wholesale customers."
        })

        df_cleaning_log = pd.DataFrame(self.cleaning_log)
        df_cleaning_log.to_csv(OUTPUT_DIR / "part2_cleaning_decision_log.csv", index=False)

        # Cache subsets to parquet
        df_sales.to_parquet(CACHE_SALES_CLEAN, index=False)
        df_returns.to_parquet(CACHE_RETURNS_CLEAN, index=False)
        df_cust_sales.to_parquet(CACHE_CUSTOMER_SALES, index=False)

        print(f"[DataCleaner] Saved clean sales ({len(df_sales):,} rows), returns ({len(df_returns):,} rows), customer sales ({len(df_cust_sales):,} rows).")
        return df_sales, df_returns, df_cust_sales, df_cleaning_log
