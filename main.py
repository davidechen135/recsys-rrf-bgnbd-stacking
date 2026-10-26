"""
Unified Command-Line Interface Driver for IE5005 Mini Project
Author: Chen Davide (A0351457B)
Course: IE5005 Data Analytics for Industrial Engineers

Usage:
  python main.py --stage all
  python main.py --stage audit
  python main.py --stage clean
  python main.py --stage eda
  python main.py --stage benchmarks
  python main.py --stage figures
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.auditor import DataAuditor
from src.data.cleaner import DataCleaner
from src.data.features import FeaturePipeline
from src.eda.descriptive import DescriptiveAnalyzer
from src.evaluation.recsys_eval import RecSysEvaluator
from src.evaluation.clv_eval import CLVEvaluator
from src.visualization.publication_plots import FigureGenerator

def main():
    parser = argparse.ArgumentParser(description="IE5005 Data Analytics Pipeline: E-Commerce RecSys & Ad Targeting")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "audit", "clean", "eda", "benchmarks", "figures"],
        help="Pipeline stage to execute: 'all', 'audit', 'clean', 'eda', 'benchmarks', 'figures'"
    )
    args = parser.parse_args()

    print("\n" + "="*80)
    print(f"IE5005 DATA ANALYTICS PIPELINE | EXECUTING STAGE: {args.stage.upper()}")
    print("="*80)

    # Stage 1: Data Audit
    if args.stage in ["all", "audit"]:
        print("\n>>> [STAGE 1] Running Data Quality Audit...")
        auditor = DataAuditor()
        df_dict, df_audit, df_issues = auditor.audit_quality()
        print("[STAGE 1 COMPLETE] Data dictionary and quality audit tables exported.")

    # Stage 2: Data Cleaning & Feature Engineering
    if args.stage in ["all", "clean", "eda", "benchmarks", "figures"]:
        print("\n>>> [STAGE 2] Running Data Cleaning and Feature Pipeline...")
        auditor = DataAuditor()
        df_raw = auditor.load_data()
        cleaner = DataCleaner(df_raw)
        df_sales, df_returns, df_cust_sales, cleaning_log = cleaner.clean()
        
        # Enrich features
        df_sales = FeaturePipeline.add_transaction_features(df_sales)
        df_returns = FeaturePipeline.add_transaction_features(df_returns)
        df_cust_sales = FeaturePipeline.add_transaction_features(df_cust_sales)
        print("[STAGE 2 COMPLETE] Clean sales, returns, and customer datasets saved.")

    # Stage 3: Exploratory Data Analysis & Attribution
    if args.stage in ["all", "eda"]:
        print("\n>>> [STAGE 3] Running Exploratory Data Analysis (EDA)...")
        eda = DescriptiveAnalyzer(df_sales, df_returns)
        df_num = eda.run_univariate_numeric()
        df_c, df_m = eda.run_univariate_categorical()
        corr, biv_country = eda.run_bivariate_analysis()
        ret_prod, overseas = eda.run_macro_attribution()
        print("[STAGE 3 COMPLETE] EDA descriptive summaries and attribution tables exported.")

    # Stage 4: Rigorous Benchmarks
    if args.stage in ["all", "benchmarks"]:
        print("\n>>> [STAGE 4] Executing Comprehensive Benchmark Experiments...")
        print("--- Running Candidate Retrieval Benchmark (Matrix 1)...")
        recsys_eval = RecSysEvaluator(df_sales)
        df_rec_matrix = recsys_eval.evaluate_all()
        print(df_rec_matrix.to_string(index=False))

        print("\n--- Running Customer CLV Prediction Benchmark (Matrix 2)...")
        clv_eval = CLVEvaluator(df_cust_sales)
        df_clv_matrix = clv_eval.evaluate_all()
        print(df_clv_matrix.to_string(index=False))
        print("[STAGE 4 COMPLETE] All benchmarks executed and saved.")

    # Stage 5: Publication Figures
    if args.stage in ["all", "figures"]:
        print("\n>>> [STAGE 5] Rendering All 8 Publication-Grade Visualizations...")
        fig_gen = FigureGenerator(df_sales, df_returns, df_cust_sales)
        fig_gen.generate_all()
        print("[STAGE 5 COMPLETE] All 8 figures rendered and synced to artifact gallery.")

    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()
