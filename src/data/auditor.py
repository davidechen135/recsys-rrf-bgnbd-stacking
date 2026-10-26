"""
Data Profiling and Quality Auditing Module (Part 1)
Author: Chen Davide (A0351457B)
Course: IE5005 Data Analytics for Industrial Engineers
"""

import time
import pandas as pd
import numpy as np
from src.config import RAW_DATA_PATH, CACHE_RAW_DATA, OUTPUT_DIR

class DataAuditor:
    """
    Profiles unfamiliar raw transactional datasets and diagnoses quality defects.
    Outputs:
      - Dataset shape
      - Schema & intended types dictionary
      - Empirical data-quality audit table
      - Quality issue diagnosis log
    """

    def __init__(self, raw_path=RAW_DATA_PATH, cache_path=CACHE_RAW_DATA):
        self.raw_path = raw_path
        self.cache_path = cache_path
        self.df_raw = None

    def load_data(self) -> pd.DataFrame:
        if self.cache_path.exists():
            print(f"[DataAuditor] Loading cached parquet from: {self.cache_path}")
            self.df_raw = pd.read_parquet(self.cache_path)
        else:
            print(f"[DataAuditor] Reading raw Excel: {self.raw_path}...")
            t0 = time.time()
            df = pd.read_excel(self.raw_path)
            for col in ['InvoiceNo', 'StockCode', 'Description', 'Country']:
                df[col] = df[col].astype('string')
            df.to_parquet(self.cache_path, index=False)
            print(f"[DataAuditor] Cached to {self.cache_path} in {time.time() - t0:.2f}s")
            self.df_raw = df
        return self.df_raw

    def audit_quality(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.df_raw is None:
            self.load_data()

        df = self.df_raw
        n_rows, n_cols = df.shape

        # 1. Data Dictionary
        dict_records = []
        for col in df.columns:
            stored_type = str(df[col].dtype)
            example_val = df[col].dropna().iloc[0] if df[col].notna().any() else "N/A"
            intended_map = {
                "InvoiceNo": "String / Categorical (Transaction ID)",
                "StockCode": "String / Categorical (Product SKU)",
                "Description": "String (Product Title / Operational Note)",
                "Quantity": "Integer (Units Traded / Returned)",
                "InvoiceDate": "Datetime (Timestamp)",
                "UnitPrice": "Float (Unit Price in GBP)",
                "CustomerID": "Integer / String (Account Identifier)",
                "Country": "Categorical (Customer Account Country)"
            }
            dict_records.append({
                "Column Name": col,
                "Stored Type": stored_type,
                "Intended Type": intended_map.get(col, stored_type),
                "Example Value": str(example_val)
            })
        df_dict = pd.DataFrame(dict_records)

        # 2. Data Quality Audit Table
        audit_records = []
        for col in df.columns:
            missing_count = int(df[col].isna().sum())
            missing_pct = (missing_count / n_rows) * 100.0
            n_distinct = int(df[col].nunique(dropna=True))

            if pd.api.types.is_numeric_dtype(df[col]):
                if col == "CustomerID":
                    min_val = f"{int(df[col].min()):d}" if df[col].notna().any() else "N.A."
                    max_val = f"{int(df[col].max()):d}" if df[col].notna().any() else "N.A."
                else:
                    min_val = f"{df[col].min():.2f}"
                    max_val = f"{df[col].max():.2f}"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                min_val = str(df[col].min())
                max_val = str(df[col].max())
            else:
                min_val, max_val = "N.A.", "N.A."

            audit_records.append({
                "column name": col,
                "missing": f"{missing_count:,}",
                "Missing %": f"{missing_pct:.2f}%",
                "number of distinct values": f"{n_distinct:,}",
                "minimum": min_val,
                "maximum": max_val
            })
        df_audit = pd.DataFrame(audit_records)

        # 3. Anomaly Diagnosis
        n_cancelled = int(df["InvoiceNo"].astype(str).str.startswith("C").sum())
        n_neg_qty = int((df["Quantity"] < 0).sum())
        n_neg_price = int((df["UnitPrice"] < 0).sum())
        n_zero_price = int((df["UnitPrice"] == 0).sum())
        n_missing_cust = int(df["CustomerID"].isna().sum())
        n_duplicates = int(df.duplicated().sum())

        issues_records = [
            {
                "Category": "Cancellations",
                "Affected Column": "InvoiceNo, Quantity",
                "Evidence": f"InvoiceNo starting with 'C' ({n_cancelled:,} rows), negative quantities",
                "Affected Rows": f"{n_cancelled:,} ({n_cancelled/n_rows*100:.2f}%)",
                "Business Impact": "Distorts purchasing volume; must be partitioned into return risk analysis."
            },
            {
                "Category": "Damaged Stock / Write-offs",
                "Affected Column": "Quantity, UnitPrice, Description",
                "Evidence": f"Negative quantities without 'C' prefix ({n_neg_qty - n_cancelled:,} rows), UnitPrice=0",
                "Affected Rows": f"{n_neg_qty - n_cancelled:,} ({(n_neg_qty - n_cancelled)/n_rows*100:.2f}%)",
                "Business Impact": "Warehouse write-offs and inventory reconciliations; corrupts consumer demand models."
            },
            {
                "Category": "Non-commercial Pricing / Bad Debt",
                "Affected Column": "UnitPrice",
                "Evidence": f"Zero or negative prices ({n_zero_price + n_neg_price:,} rows), min: -£11,062.06",
                "Affected Rows": f"{n_zero_price + n_neg_price:,} ({(n_zero_price + n_neg_price)/n_rows*100:.2f}%)",
                "Business Impact": "Zero and negative unit prices distort revenue calculations and Average Order Value."
            },
            {
                "Category": "Missing Customer Identifiers",
                "Affected Column": "CustomerID",
                "Evidence": f"Null CustomerID in {n_missing_cust:,} rows",
                "Affected Rows": f"{n_missing_cust:,} ({n_missing_cust/n_rows*100:.2f}%)",
                "Business Impact": "Anonymous guest checkout; valid for aggregate GMV but cannot track repeat LTV."
            },
            {
                "Category": "Non-merchandise Overhead Codes",
                "Affected Column": "StockCode, Description",
                "Evidence": "Operational codes like POST, D, M, BANK CHARGES, CRUK",
                "Affected Rows": "2,766 rows (0.51%)",
                "Business Impact": "Pollutes product-level association and basket cross-selling recommendation models."
            },
            {
                "Category": "Exact Duplicate Records",
                "Affected Column": "All attributes identical",
                "Evidence": f"Identical duplicate transactions logged ({n_duplicates:,} rows)",
                "Affected Rows": f"{n_duplicates:,} ({n_duplicates/n_rows*100:.2f}%)",
                "Business Impact": "Artificially double-counts revenue, quantity, and transaction frequencies."
            }
        ]
        df_issues = pd.DataFrame(issues_records)

        # Export CSVs
        df_audit.to_csv(OUTPUT_DIR / "part1_audit_table.csv", index=False)
        df_issues.to_csv(OUTPUT_DIR / "part1_issues_table.csv", index=False)

        return df_dict, df_audit, df_issues
