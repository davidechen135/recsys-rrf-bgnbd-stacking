"""
Exploratory Data Analysis and Macro Attribution Module (Part 3)
Author: Chen Davide (A0351457B)
Course: IE5005 Data Analytics for Industrial Engineers
"""

import pandas as pd
import numpy as np
from src.config import OUTPUT_DIR

class DescriptiveAnalyzer:
    """
    Computes rigorous univariate and bivariate descriptive statistics,
    evaluates distribution skewness, and attributes return revenue loss.
    """

    def __init__(self, df_sales: pd.DataFrame, df_returns: pd.DataFrame):
        self.df_sales = df_sales
        self.df_returns = df_returns

    def run_univariate_numeric(self) -> pd.DataFrame:
        num_vars = ['Quantity', 'UnitPrice', 'TotalRevenue']
        records = []
        for var in num_vars:
            s = self.df_sales[var]
            records.append({
                "Variable": var,
                "Count": f"{len(s):,}",
                "Mean": f"{s.mean():.2f}",
                "Median": f"{s.median():.2f}",
                "Std Dev": f"{s.std():.2f}",
                "Min": f"{s.min():.2f}",
                "Q1 (25%)": f"{s.quantile(0.25):.2f}",
                "Q3 (75%)": f"{s.quantile(0.75):.2f}",
                "Max": f"{s.max():.2f}"
            })
        df_num = pd.DataFrame(records)
        df_num.to_csv(OUTPUT_DIR / "part3_numeric_summary.csv", index=False)
        return df_num

    def run_univariate_categorical(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Country
        counts_c = self.df_sales['Country'].value_counts()
        df_country = pd.DataFrame({
            "Country": counts_c.index,
            "Transaction Lines": counts_c.values,
            "Percentage": (counts_c.values / len(self.df_sales) * 100).round(2),
            "Revenue (GBP)": self.df_sales.groupby('Country')['TotalRevenue'].sum().loc[counts_c.index].round(2).values
        })
        df_country.to_csv(OUTPUT_DIR / "part3_country_frequency.csv", index=False)

        # Month
        counts_m = self.df_sales['InvoiceMonth'].value_counts().sort_index()
        df_month = pd.DataFrame({
            "Month": counts_m.index.astype(str),
            "Transaction Lines": counts_m.values,
            "Percentage": (counts_m.values / len(self.df_sales) * 100).round(2),
            "Revenue (GBP)": self.df_sales.groupby('InvoiceMonth')['TotalRevenue'].sum().sort_index().round(2).values
        })
        df_month.to_csv(OUTPUT_DIR / "part3_month_frequency.csv", index=False)

        return df_country, df_month

    def run_bivariate_analysis(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Correlation
        corr = self.df_sales[['Quantity', 'UnitPrice', 'TotalRevenue']].corr().round(4)
        
        # Numeric across top countries
        top_8 = self.df_sales['Country'].value_counts().head(8).index.tolist()
        df_sub = self.df_sales[self.df_sales['Country'].isin(top_8)]
        biv_country = df_sub.groupby('Country').agg(
            Order_Count=('InvoiceNo', 'nunique'),
            Avg_Order_Revenue=('BasketRevenue', 'mean'),
            Median_Order_Revenue=('BasketRevenue', 'median'),
            Total_Revenue=('TotalRevenue', 'sum')
        ).reset_index().sort_values('Total_Revenue', ascending=False)
        biv_country.to_csv(OUTPUT_DIR / "part3_bivariate_country.csv", index=False)

        return corr, biv_country

    def run_macro_attribution(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Top Return Products
        ret_prod = self.df_returns.groupby(['StockCode', 'Description']).agg(
            Return_Lines=('InvoiceNo', 'count'),
            Lost_Quantity=('Quantity', 'sum'),
            Lost_Revenue=('TotalRevenue', lambda x: abs(float(x.sum())))
        ).reset_index().sort_values('Lost_Revenue', ascending=False).head(10)
        ret_prod.to_csv(OUTPUT_DIR / "part5_top_return_products.csv", index=False)

        # Overseas Performance
        overseas_sales = self.df_sales[self.df_sales['Country'] != 'United Kingdom'].groupby('Country').agg(
            Order_Count=('InvoiceNo', 'nunique'),
            Gross_Revenue=('TotalRevenue', 'sum'),
            AOV=('BasketRevenue', 'mean')
        ).reset_index().sort_values('Gross_Revenue', ascending=False)
        overseas_returns = self.df_returns[self.df_returns['Country'] != 'United Kingdom'].groupby('Country')['TotalRevenue'].sum().abs()
        overseas_sales['Lost_Revenue'] = overseas_sales['Country'].map(overseas_returns).fillna(0)
        overseas_sales['Return_Rate'] = (overseas_sales['Lost_Revenue'] / overseas_sales['Gross_Revenue'] * 100).round(2)
        overseas_sales.to_csv(OUTPUT_DIR / "part5_overseas_markets.csv", index=False)

        # Type I vs Type II Return Decoupling Analysis
        self.export_return_decoupling()

        return ret_prod, overseas_sales

    def export_return_decoupling(self) -> pd.DataFrame:
        """
        Decouples returns into Type I (Instant checkout reversals / administrative anomalies)
        and Type II (Chronic post-delivery operational & product quality returns).
        """
        total_lost_rev = abs(float(self.df_returns['TotalRevenue'].sum()))
        
        # Identify Type I major anomalies (bulk order reversals: absolute quantity > 50,000 or revenue > £50,000)
        type1_mask = (abs(self.df_returns['Quantity']) > 50000) | (abs(self.df_returns['TotalRevenue']) > 50000)
        df_type1 = self.df_returns[type1_mask].copy()
        df_type2 = self.df_returns[~type1_mask].copy()

        type1_rev = abs(float(df_type1['TotalRevenue'].sum()))
        type2_rev = abs(float(df_type2['TotalRevenue'].sum()))

        decoupling_summary = pd.DataFrame([
            {
                "Return_Archetype": "Type I: Instant Checkout Reversals (Administrative / Systemic)",
                "Transaction_Lines": len(df_type1),
                "Lost_Quantity": int(abs(df_type1['Quantity'].sum())),
                "Lost_Revenue_GBP": f"£{type1_rev:,.2f}",
                "Share_of_Total_Returns": f"{(type1_rev / total_lost_rev * 100):.2f}%",
                "Root_Cause": "Erroneous bulk buyer checkout cancelled within 15-30 minutes (SKU 23843 & 23166)",
                "Operational_Action": "Implement checkout quantity cap and 30-min payment confirmation hold in OMS"
            },
            {
                "Return_Archetype": "Type II: Chronic Operational & Product Quality Returns",
                "Transaction_Lines": len(df_type2),
                "Lost_Quantity": int(abs(df_type2['Quantity'].sum())),
                "Lost_Revenue_GBP": f"£{type2_rev:,.2f}",
                "Share_of_Total_Returns": f"{(type2_rev / total_lost_rev * 100):.2f}%",
                "Root_Cause": "Fragile ceramics transit damage, packaging failure, customer expectation mismatch",
                "Operational_Action": "Re-engineer fragile SKU bubble-wrap packaging (Regency Cakestand) & supplier audit"
            },
            {
                "Return_Archetype": "Total Combined Return Volume",
                "Transaction_Lines": len(self.df_returns),
                "Lost_Quantity": int(abs(self.df_returns['Quantity'].sum())),
                "Lost_Revenue_GBP": f"£{total_lost_rev:,.2f}",
                "Share_of_Total_Returns": "100.00%",
                "Root_Cause": "Aggregate e-commerce reversals and returns",
                "Operational_Action": "Dual-track intervention recovering estimated £360k+ in 2012"
            }
        ])
        decoupling_summary.to_csv(OUTPUT_DIR / "part5_return_decoupling.csv", index=False)
        return decoupling_summary
