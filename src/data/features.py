"""
Feature Engineering and RFM Construction Module (Part 2 Task 2)
Author: Chen Davide (A0351457B)
Course: IE5005 Data Analytics for Industrial Engineers
"""

import pandas as pd
from lifetimes.utils import summary_data_from_transaction_data

class FeaturePipeline:
    """
    Transforms transaction logs into derived business metrics and longitudinal
    RFM-T representations.
    """

    @staticmethod
    def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['TotalRevenue'] = df['Quantity'] * df['UnitPrice']
        df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')
        df['BasketSize'] = df.groupby('InvoiceNo')['Quantity'].transform('sum')
        df['BasketRevenue'] = df.groupby('InvoiceNo')['TotalRevenue'].transform('sum')
        df['BasketLines'] = df.groupby('InvoiceNo')['StockCode'].transform('nunique')
        return df

    @staticmethod
    def build_rfm_summary(df_cust: pd.DataFrame) -> pd.DataFrame:
        """
        Builds customer-level Recency, Frequency, Monetary, and Tenure matrix.
        """
        rfm = summary_data_from_transaction_data(
            df_cust,
            customer_id_col='CustomerID',
            datetime_col='InvoiceDate',
            monetary_value_col='TotalRevenue',
            observation_period_end=df_cust['InvoiceDate'].max(),
            freq='D'
        )
        total_spend = df_cust.groupby('CustomerID')['TotalRevenue'].sum()
        rfm['TotalSpend'] = total_spend
        rfm['DaysInactive'] = rfm['T'] - rfm['recency']
        return rfm
