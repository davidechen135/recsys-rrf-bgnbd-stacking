"""
Global Configuration and Hyperparameters for E-Commerce RecSys & Ad Targeting Pipeline
Author: Chen Davide (A0351457B)
Course: IE5005 Data Analytics for Industrial Engineers
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path("/Users/davide/Developer/coursework/IE5005/data")
RAW_DATA_PATH = DATA_DIR / "Dataset A Online Retail.xlsx"

OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = PROJECT_ROOT / "figures"
ARTIFACT_DIR = Path("/Users/davide/.gemini/antigravity/brain/b85bcc32-bb92-4778-91d1-69046a9a1323")

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Parquet Caches
CACHE_RAW_DATA = OUTPUT_DIR / "raw_data.parquet"
CACHE_SALES_CLEAN = OUTPUT_DIR / "df_sales_clean.parquet"
CACHE_RETURNS_CLEAN = OUTPUT_DIR / "df_returns_clean.parquet"
CACHE_CUSTOMER_SALES = OUTPUT_DIR / "df_customer_sales.parquet"

# Data Governance Rules
NON_MERCHANDISE_CODES = [
    'POST', 'D', 'M', 'BANK CHARGES', 'CRUK', 'PADS', 'DOT', 'AMAZONFEE', 'S', 'B'
]

# Random Seed & Reproducibility
RANDOM_SEED = 42

# Time-Series Split Dates (Out-of-Time Validation)
CALIBRATION_END_DATE = "2011-08-31"

# SOTA Model Hyperparameters
ITEM2VEC_PARAMS = {
    "vector_size": 64,
    "window": 10,
    "min_count": 3,
    "sg": 1,          # Skip-Gram
    "negative": 5,    # Negative sampling
    "workers": 1,     # Single-worker for deterministic reproducibility
    "epochs": 15,
    "seed": RANDOM_SEED
}

BGNBD_PARAMS = {
    "penalizer_coef": 0.01
}

GAMMA_GAMMA_PARAMS = {
    "penalizer_coef": 0.01
}

# Visualization Global Style
PLOT_CONFIG = {
    "font_family": ['DejaVu Sans', 'Arial', 'Helvetica'],
    "edge_color": '#333333',
    "line_width": 0.8,
    "grid_color": '#EAEAEA',
    "grid_style": '--',
    "grid_alpha": 0.8,
    "dpi": 300
}
