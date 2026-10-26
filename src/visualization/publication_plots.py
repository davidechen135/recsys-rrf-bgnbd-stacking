"""
Publication-Grade Visualization Generation Module (CVPR / Academic Standard)
Produces All 8 Empirical Charts adhering to Top-Tier Conference Aesthetics:
- Times/Serif Typography matching formal publications
- Minimalist Academic Palette (Oxford Navy, Slate, Charcoal, Muted Terracotta)
- No Redundant In-Plot Titles (Context fully delegated to Figure Captions)
- High-DPI Vector Crispness & Subtle Grids
Author: Chen Davide (A0351457B)
Course: IE5005 Data Analytics for Industrial Engineers, National University of Singapore
"""

import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from src.config import (
    FIGURES_DIR,
    ARTIFACT_DIR,
    OUTPUT_DIR,
    PLOT_CONFIG,
    BGNBD_PARAMS,
    GAMMA_GAMMA_PARAMS
)
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

# Apply CVPR / Top-Tier Conference Styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.edgecolor'] = '#2b2b2b'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#e2e8f0'
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.alpha'] = 0.7
plt.rcParams['axes.labelsize'] = 9.5
plt.rcParams['xtick.labelsize'] = 8.5
plt.rcParams['ytick.labelsize'] = 8.5
plt.rcParams['legend.fontsize'] = 8.0

# Academic CVPR Palette
NAVY_PRIMARY = '#1b365d'
SLATE_SECONDARY = '#4a7bb0'
CRIMSON_ACCENT = '#8b263e'
FOREST_ACCENT = '#2d6a4f'
CHARCOAL_BASE = '#333333'
MUTED_GRAY = '#64748b'
LIGHT_GRAY_FILL = '#f8fafc'


class FigureGenerator:
    """
    Renders the complete suite of 8 empirical figures adhering strictly to
    top-tier conference aesthetics (CVPR / KDD / RecSys).
    """

    def __init__(self, df_sales: pd.DataFrame, df_returns: pd.DataFrame, df_cust_sales: pd.DataFrame):
        self.df_sales = df_sales
        self.df_returns = df_returns
        self.df_cust_sales = df_cust_sales

    def generate_all(self):
        self.plot_distribution()
        self.plot_pareto()
        self.plot_monthly_trend()
        self.plot_rfm_relationship()
        self.plot_budget_quadrants()
        self.plot_recsys_benchmark()
        self.plot_clv_benchmark()
        self.plot_overseas_bubble()
        self._sync_artifacts()

    def plot_distribution(self):
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        rev_log = np.log10(self.df_sales['TotalRevenue'].clip(lower=0.1))
        sns.histplot(rev_log, bins=50, kde=True, color=NAVY_PRIMARY, ax=ax, edgecolor='white', alpha=0.75, linewidth=0.5)
        median_rev = self.df_sales['TotalRevenue'].median()
        mean_rev = self.df_sales['TotalRevenue'].mean()
        
        median_x = np.log10(median_rev)
        mean_x = np.log10(mean_rev)
        ax.axvline(median_x, color=FOREST_ACCENT, linestyle='--', linewidth=1.8, label=f'Median: £{median_rev:.2f} (log10={median_x:.2f})')
        ax.axvline(mean_x, color=CRIMSON_ACCENT, linestyle='-', linewidth=1.8, label=f'Mean: £{mean_rev:.2f} (log10={mean_x:.2f})')
        
        # Callout boxes with semi-transparent white backgrounds
        y_max = ax.get_ylim()[1]
        ax.text(median_x - 0.08, y_max * 0.85, f'Median: £{median_rev:.2f}', 
                color=FOREST_ACCENT, ha='right', va='center', fontsize=8.2, weight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=FOREST_ACCENT, alpha=0.92, linewidth=0.8))
        ax.text(mean_x + 0.08, y_max * 0.72, f'Mean: £{mean_rev:.2f}', 
                color=CRIMSON_ACCENT, ha='left', va='center', fontsize=8.2, weight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=CRIMSON_ACCENT, alpha=0.92, linewidth=0.8))
        
        ax.set_xlabel(r"Transaction Line Revenue: $\log_{10}(\mathrm{Revenue\ in\ GBP})$")
        ax.set_ylabel("Empirical Frequency (Lines)")
        ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.95, loc='upper right')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig1_distribution_revenue.png")
        plt.close()

    def plot_pareto(self):
        cust_rev = self.df_cust_sales.groupby('CustomerID')['TotalRevenue'].sum().sort_values(ascending=False).reset_index()
        cust_rev['Cumulative_Revenue'] = cust_rev['TotalRevenue'].cumsum()
        cust_rev['Cumulative_Pct'] = (cust_rev['Cumulative_Revenue'] / cust_rev['TotalRevenue'].sum()) * 100
        cust_rev['Customer_Pct'] = (np.arange(1, len(cust_rev) + 1) / len(cust_rev)) * 100
        top_10_pct = cust_rev.loc[cust_rev['Customer_Pct'] <= 10, 'Cumulative_Pct'].iloc[-1]

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        ax.plot(cust_rev['Customer_Pct'], cust_rev['Cumulative_Pct'], color=NAVY_PRIMARY, linewidth=2.2, label='Empirical Concentration')
        ax.plot([0, 100], [0, 100], color='#888888', linestyle=':', linewidth=1.2, label='Uniform Baseline (45°)')
        ax.axvline(10, color=CRIMSON_ACCENT, linestyle='--', linewidth=1.2)
        ax.axhline(top_10_pct, color=CRIMSON_ACCENT, linestyle='--', linewidth=1.2)
        ax.scatter([10], [top_10_pct], color=CRIMSON_ACCENT, s=45, zorder=5)
        ax.annotate(f'Top 10% Accounts generate {top_10_pct:.1f}% GMV', xy=(10, top_10_pct), xytext=(22, top_10_pct - 10),
                    arrowprops=dict(facecolor=CRIMSON_ACCENT, shrink=0.06, width=0.9, headwidth=4.5),
                    fontweight='semibold', fontsize=8.2, color=CRIMSON_ACCENT)
        ax.set_xlabel("Cumulative Fraction of Customer Accounts (%)")
        ax.set_ylabel("Cumulative Fraction of Attributed GMV (%)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        ax.xaxis.set_major_formatter(ticker.PercentFormatter())
        ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#cccccc')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig2_composition_pareto.png")
        plt.close()

    def plot_monthly_trend(self):
        m_sales = self.df_sales.groupby('InvoiceMonth')['TotalRevenue'].sum()
        m_returns = self.df_returns.groupby('InvoiceMonth')['TotalRevenue'].sum().abs()
        months_str = [str(m) for m in m_sales.index]
        cancel_rate = (m_returns / (m_sales + m_returns)) * 100

        fig, ax1 = plt.subplots(figsize=(9.5, 4.8), dpi=300)
        ax1.set_xlabel('Invoice Observation Month (Dec 2010 – Dec 2011)')
        ax1.set_ylabel('Gross Sales GMV (£k GBP)', color=NAVY_PRIMARY)
        line1 = ax1.plot(range(len(months_str)), m_sales.values / 1000, color=NAVY_PRIMARY, marker='o', linewidth=2.0, markersize=5, label='Gross Sales (£k)')
        ax1.tick_params(axis='y', labelcolor=NAVY_PRIMARY)
        ax1.set_xticks(range(len(months_str)))
        ax1.set_xticklabels(months_str, rotation=35, ha='right')
        ax1.set_ylim(400, 1650)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Order Cancellation Rate (%)', color=CRIMSON_ACCENT)
        line2 = ax2.plot(range(len(months_str)), cancel_rate.values, color=CRIMSON_ACCENT, marker='s', linestyle='--', linewidth=1.8, markersize=5, label='Cancellation Rate (%)')
        ax2.tick_params(axis='y', labelcolor=CRIMSON_ACCENT)
        ax2.yaxis.set_major_formatter(ticker.PercentFormatter())
        ax2.set_ylim(0, 28)

        # Annotate Type I cancellation spikes and Nov peak in clean academic style
        ax2.annotate('Jan 2011 Spike (12.0%):\n£77.5k Type I Administrative\n(Cust 12346, 16-min reversal)',
                     xy=(1, cancel_rate.values[1]), xytext=(1.4, 18.0),
                     arrowprops=dict(facecolor=CRIMSON_ACCENT, shrink=0.08, width=0.8, headwidth=4.5),
                     fontsize=7.8, color=CRIMSON_ACCENT,
                     bbox=dict(boxstyle="square,pad=0.25", fc="white", ec='#cccccc', lw=0.7))

        ax1.annotate('Nov Peak: £1,453k\n(+100.5% vs Aug base)',
                     xy=(11, m_sales.values[11] / 1000), xytext=(5.0, 1420),
                     arrowprops=dict(facecolor=NAVY_PRIMARY, shrink=0.06, width=0.8, headwidth=4.5),
                     fontsize=7.8, color=NAVY_PRIMARY,
                     bbox=dict(boxstyle="square,pad=0.25", fc="white", ec='#cccccc', lw=0.7))

        ax2.annotate('Dec 2011 Spike (22.1%):\n£168.5k Type I Administrative\n(Cust 16446, 12-min reversal)',
                     xy=(12, cancel_rate.values[12]), xytext=(5.0, 17.0),
                     arrowprops=dict(facecolor=CRIMSON_ACCENT, shrink=0.08, width=0.8, headwidth=4.5),
                     fontsize=7.8, color=CRIMSON_ACCENT,
                     bbox=dict(boxstyle="square,pad=0.25", fc="white", ec='#cccccc', lw=0.7))

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig3_change_monthly_trend.png")
        plt.close()

    def plot_rfm_relationship(self):
        cust_summary = self.df_cust_sales.groupby('CustomerID').agg(
            Frequency=('InvoiceNo', 'nunique'),
            Monetary=('TotalRevenue', 'sum'),
            Recency_Days=('InvoiceDate', lambda d: (self.df_sales['InvoiceDate'].max() - d.max()).days)
        ).reset_index()

        fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=300)
        # Reduced size and alpha blending (0.38) to eliminate overplotting in dense low-frequency region
        scatter = ax.scatter(np.log1p(cust_summary['Frequency']), np.log10(cust_summary['Monetary'].clip(lower=1)),
                             c=cust_summary['Recency_Days'], cmap='cividis', alpha=0.38, s=16, edgecolors='none')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Recency: Days Since Last Transaction (0 = Active, 370 = Inactive)', fontsize=8.2)
        ax.set_xlabel(r"Purchase Velocity: $\log(\mathrm{Order\ Frequency} + 1)$")
        ax.set_ylabel(r"Cumulative Spend: $\log_{10}(\mathrm{Monetary\ Spend\ in\ GBP})$")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig4_relationship_rfm_scatter.png")
        plt.close()

    def plot_budget_quadrants(self):
        max_date = self.df_cust_sales['InvoiceDate'].max()
        cust_summary = self.df_cust_sales.groupby('CustomerID').agg(
            Frequency=('InvoiceNo', 'nunique'),
            Monetary=('TotalRevenue', 'sum'),
            Recency_Days=('InvoiceDate', lambda d: (max_date - d.max()).days)
        ).reset_index()

        # Compute forward 1-year CLV via BG/NBD and Gamma-Gamma
        rfm = summary_data_from_transaction_data(
            self.df_cust_sales,
            customer_id_col='CustomerID',
            datetime_col='InvoiceDate',
            monetary_value_col='TotalRevenue',
            observation_period_end=max_date,
            freq='D'
        )
        bgf = BetaGeoFitter(penalizer_coef=BGNBD_PARAMS["penalizer_coef"])
        rep = rfm[rfm['frequency'] > 0]
        bgf.fit(rep['frequency'], rep['recency'], rep['T'])

        ggf = GammaGammaFitter(penalizer_coef=GAMMA_GAMMA_PARAMS["penalizer_coef"])
        ret = rfm[(rfm['frequency'] > 0) & (rfm['monetary_value'] > 0)]
        ggf.fit(ret['frequency'], ret['monetary_value'])

        pred_trans = bgf.conditional_expected_number_of_purchases_up_to_time(365, rfm['frequency'], rfm['recency'], rfm['T'])
        pred_monetary = ggf.conditional_expected_average_profit(rfm['frequency'], rfm['monetary_value']).fillna(ret['monetary_value'].median())
        rfm['Forward_CLV'] = (pred_trans * pred_monetary).values

        cust_merged = cust_summary.merge(rfm[['Forward_CLV']], left_on='CustomerID', right_index=True)
        clv_thresh = 2499.72
        rec_thresh = 60

        def assign_quad(row):
            active = row['Recency_Days'] <= rec_thresh
            high_val = row['Forward_CLV'] >= clv_thresh
            if active and high_val:
                return 'Q1'
            elif (not active) and high_val:
                return 'Q2'
            elif active and (not high_val):
                return 'Q3'
            else:
                return 'Q4'

        cust_merged['Quadrant'] = cust_merged.apply(assign_quad, axis=1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.6), dpi=300)
        
        # Left Panel: 4-Quadrant Crosshair Scatter (Academic Muted Palette)
        colors = {'Q1': FOREST_ACCENT, 'Q2': CRIMSON_ACCENT, 'Q3': NAVY_PRIMARY, 'Q4': MUTED_GRAY}
        for q_name, q_color in colors.items():
            sub = cust_merged[cust_merged['Quadrant'] == q_name]
            alpha_val = 0.30 if q_name == 'Q4' else 0.65
            size_val = 16 if q_name == 'Q4' else 26
            ax1.scatter(sub['Recency_Days'], np.log10(sub['Forward_CLV'].clip(lower=1)), 
                        c=q_color, alpha=alpha_val, s=size_val, edgecolors='none', label=q_name)

        ax1.axvline(rec_thresh, color='#111111', linestyle='--', linewidth=1.2, alpha=0.7)
        ax1.axhline(np.log10(clv_thresh), color='#111111', linestyle='--', linewidth=1.2, alpha=0.7)

        # In-plot callouts placed with clean minimalist styling
        ax1.text(12, 5.08, 'Q1: Core Champions\n(Active VIP: 744 Accts)\nOrganic VIP Care (15% Budget)', 
                 fontsize=8.0, color=FOREST_ACCENT, 
                 bbox=dict(boxstyle='square,pad=0.25', facecolor='white', edgecolor='#cccccc', alpha=0.92))
        ax1.text(145, 5.08, 'Q2: At-Risk High-LTV\n(Dormant VIP: 89 Accts)\n1st Paid Ad Priority (50% Budget)', 
                 fontsize=8.0, color=CRIMSON_ACCENT, 
                 bbox=dict(boxstyle='square,pad=0.25', facecolor='white', edgecolor='#cccccc', alpha=0.92))
        ax1.text(6, 1.05, 'Q3: Promising Growth\n(Active Mid: 1,664 Accts)\nCross-Sell Recs (25% Budget)', 
                 fontsize=8.0, color=NAVY_PRIMARY, 
                 bbox=dict(boxstyle='square,pad=0.25', facecolor='white', edgecolor='#cccccc', alpha=0.92))
        ax1.text(155, 1.05, 'Q4: Low-Value Inactive\n(Dormant Low: 1,837 Accts)\nAd Suppressed / Drip (10% Budget)', 
                 fontsize=8.0, color=CHARCOAL_BASE, 
                 bbox=dict(boxstyle='square,pad=0.25', facecolor='white', edgecolor='#cccccc', alpha=0.92))

        ax1.set_xlabel("Customer Inactivity / Recency (Days)")
        ax1.set_ylabel(r"Forward 1-Year Value: $\log_{10}(\mathrm{Predicted\ CLV\ in\ GBP})$")
        ax1.set_xlim(-10, 385)
        ax1.set_ylim(0.75, 5.75)

        # Right Panel: Budget Allocation Bar Chart
        quad_summary = pd.read_csv(OUTPUT_DIR / "part5_ad_budget_quadrants.csv")
        quad_labels = [
            "Q1: Core Champions\n(Active VIP)",
            "Q2: At-Risk High-LTV\n(Dormant VIP)",
            "Q3: Promising Growth\n(Active Growth)",
            "Q4: Inactive Low-Val\n(Dormant Small)"
        ]
        quad_pcts = [
            quad_summary.loc[quad_summary['Ad_Budget_Quadrant'].str.startswith('Q1'), 'Pct_CLV'].values[0],
            quad_summary.loc[quad_summary['Ad_Budget_Quadrant'].str.startswith('Q2'), 'Pct_CLV'].values[0],
            quad_summary.loc[quad_summary['Ad_Budget_Quadrant'].str.startswith('Q3'), 'Pct_CLV'].values[0],
            quad_summary.loc[quad_summary['Ad_Budget_Quadrant'].str.startswith('Q4'), 'Pct_CLV'].values[0]
        ]
        quad_totals = ["£6.03M", "£404k", "£1.55M", "£680k"]
        bars = ax2.bar(quad_labels, quad_pcts, color=[FOREST_ACCENT, CRIMSON_ACCENT, NAVY_PRIMARY, MUTED_GRAY], edgecolor='#333333', linewidth=0.6, width=0.52)
        for idx, bar in enumerate(bars):
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval + 1.2, f"{yval:.1f}%\n({quad_totals[idx]})", 
                     ha='center', va='bottom', fontsize=8.0)
        ax2.set_ylabel("Share of Total Forward 1-Year CLV Potential (%)")
        ax2.set_ylim(0, max(quad_pcts) * 1.25)
        ax2.tick_params(axis='x', rotation=10)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig5_model_clv_budget_allocation.png")
        plt.close()

    def plot_recsys_benchmark(self):
        df_rec = pd.read_csv(OUTPUT_DIR / "benchmark_recsys_expanded.csv")
        df_rec['HitRate5_val'] = df_rec['HitRate@5 (%)'].str.rstrip('%').astype(float)
        df_rec['HitRate10_val'] = df_rec['HitRate@10 (%)'].str.rstrip('%').astype(float)
        df_rec['Coverage_val'] = df_rec['Catalog Coverage (%)'].str.rstrip('%').astype(float)

        short_names = [
            "Popularity\n(Baseline)", 
            "Apriori\n(Lec 8)", 
            "Hierarchical\n(Lec 8)", 
            "ItemCF\n(Classic CF)", 
            "SVD Fact\n(Latent)", 
            "Item2Vec\n(Frontier)", 
            "Hybrid RRF\n(Proposed)"
        ]
        fig, ax1 = plt.subplots(figsize=(11.5, 5.2), dpi=300)
        x = np.arange(len(short_names))
        width = 0.28
        rects1 = ax1.bar(x - width/2, df_rec['HitRate5_val'], width, label='HitRate@5 (%)', color=NAVY_PRIMARY, edgecolor='#333333', linewidth=0.6)
        rects2 = ax1.bar(x + width/2, df_rec['HitRate10_val'], width, label='HitRate@10 (%)', color=SLATE_SECONDARY, edgecolor='#333333', linewidth=0.6)
        ax1.set_ylabel('Offline Recommendation Hit Rate (%)', color=NAVY_PRIMARY)
        ax1.set_xticks(x)
        ax1.set_xticklabels(short_names)
        ax1.yaxis.set_major_formatter(ticker.PercentFormatter())
        ax1.set_ylim(0, max(df_rec['HitRate10_val']) * 1.35)

        ax2 = ax1.twinx()
        line_cov = ax2.plot(x, df_rec['Coverage_val'], color=CRIMSON_ACCENT, marker='o', linewidth=2.0, markersize=6, label='Catalog Coverage (%)')
        ax2.set_ylabel('Catalog Long-Tail Coverage (%)', color=CRIMSON_ACCENT)
        ax2.yaxis.set_major_formatter(ticker.PercentFormatter())
        ax2.set_ylim(0, 96)
        ax2.tick_params(axis='y', labelcolor=CRIMSON_ACCENT)

        # Dual annotations with high legibility and bold callouts
        ax2.annotate('Item2Vec (Semantic Diversity):\n73.33% Long-Tail Coverage\n13.26 bits Novelty', 
                     xy=(5, df_rec['Coverage_val'].iloc[5]),
                     xytext=(1.2, 75), arrowprops=dict(facecolor=CRIMSON_ACCENT, shrink=0.06, width=1.0, headwidth=5.0),
                     fontsize=8.5, color=CRIMSON_ACCENT,
                     bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=CRIMSON_ACCENT, lw=0.9, alpha=0.95))

        hybrid_idx = len(short_names) - 1
        ax1.annotate('Proposed Hybrid RRF (SOTA):\n18.64% HitRate@10 (+11.1% gain)\n58.51% Coverage | 0.104 NDCG', 
                     xy=(hybrid_idx, df_rec['HitRate10_val'].iloc[hybrid_idx] + 0.8),
                     xytext=(4.1, 21.0), arrowprops=dict(facecolor=NAVY_PRIMARY, shrink=0.06, width=1.0, headwidth=5.0),
                     fontsize=8.5, color=NAVY_PRIMARY,
                     bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=NAVY_PRIMARY, lw=0.9, alpha=0.95))

        for rect in rects1:
            h = rect.get_height()
            ax1.text(rect.get_x() + rect.get_width()/2., h + 0.3, f'{h:.1f}%', ha='center', va='bottom', fontsize=7.2)
        for rect in rects2:
            h = rect.get_height()
            ax1.text(rect.get_x() + rect.get_width()/2., h + 0.3, f'{h:.1f}%', ha='center', va='bottom', fontsize=7.2)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig6_recsys_benchmark_comparison.png")
        plt.close()

    def plot_clv_benchmark(self):
        df_clv = pd.read_csv(OUTPUT_DIR / "benchmark_clv_expanded.csv")
        df_clv['MAE_val'] = df_clv['Holdout MAE (£)'].str.replace('£', '').astype(float)
        df_clv['Precision_val'] = df_clv['Top-Decile Precision (%)'].str.rstrip('%').astype(float)

        short_names = [
            "K-Means\n(Lec 8)", 
            "OLS Reg\n(Lec 5)", 
            "Ridge Reg\n(Lec 7)", 
            "KNN Reg\n(Lec 7)", 
            "Random Forest\n(Bagging)", 
            "GBDT\n(Boosting)", 
            "BG/NBD Prob\n(Frontier)", 
            "Hybrid Stacking\n(Proposed)"
        ]
        fig, ax1 = plt.subplots(figsize=(12, 5.3), dpi=300)
        x = np.arange(len(short_names))
        width = 0.52
        bar_colors = [MUTED_GRAY, MUTED_GRAY, MUTED_GRAY, MUTED_GRAY, SLATE_SECONDARY, SLATE_SECONDARY, FOREST_ACCENT, NAVY_PRIMARY]
        bars = ax1.bar(x, df_clv['MAE_val'], width, color=bar_colors, edgecolor='#333333', linewidth=0.6, alpha=0.88, label='Holdout MAE (£ GBP) [Bars, Left]')
        ax1.set_ylabel('Holdout Q4 Prediction Error (MAE in £ GBP) [Lower is Better]', color='#333333')
        ax1.set_xticks(x)
        ax1.set_xticklabels(short_names)
        ax1.set_ylim(0, max(df_clv['MAE_val']) * 1.28)

        ax2 = ax1.twinx()
        line_prec = ax2.plot(x, df_clv['Precision_val'], color=CRIMSON_ACCENT, marker='s', linewidth=2.0, markersize=6, label='Top-Decile Precision (%) [Line, Right]')
        ax2.set_ylabel('Top-Decile Value Capture Precision (%) [Higher is Better]', color=CRIMSON_ACCENT)
        ax2.yaxis.set_major_formatter(ticker.PercentFormatter())
        ax2.set_ylim(15, 76)
        ax2.tick_params(axis='y', labelcolor=CRIMSON_ACCENT)

        for idx, bar in enumerate(bars):
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., 340, f'£{h:.0f}', ha='center', va='center', fontsize=7.8, color='white' if idx >= 4 else '#111111', fontweight='semibold')

        for idx, val in enumerate(df_clv['Precision_val']):
            ax2.text(x[idx], val + 1.8, f'{val:.1f}%', ha='center', va='bottom', fontsize=7.5, color=CRIMSON_ACCENT)

        hybrid_idx = len(short_names) - 1
        ax2.annotate('Proposed Stacking Ensemble (SOTA):\nLowest MAE (£593.25, -£352 error cut)\n59.82% Top-Decile Precision\n72.30% Revenue Capture @ 20% Budget', 
                     xy=(hybrid_idx, df_clv['Precision_val'].iloc[hybrid_idx]),
                     xytext=(3.1, 64.0), arrowprops=dict(facecolor=CRIMSON_ACCENT, shrink=0.08, width=1.0, headwidth=5.0),
                     fontsize=8.5, color=CRIMSON_ACCENT,
                     bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=CRIMSON_ACCENT, lw=0.9, alpha=0.95))

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, facecolor='white', edgecolor='#cccccc')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig7_clv_benchmark_comparison.png")
        plt.close()

    def plot_overseas_bubble(self):
        df_overseas = pd.read_csv(OUTPUT_DIR / "part5_overseas_markets.csv")
        top_markets = df_overseas.head(12).copy()

        fig, ax = plt.subplots(figsize=(11, 5.8), dpi=300)
        bubble_sizes = (top_markets['Gross_Revenue'] / 160).clip(lower=90, upper=2200)
        
        scatter = ax.scatter(
            top_markets['Order_Count'], 
            top_markets['AOV'], 
            s=bubble_sizes, 
            c=top_markets['Return_Rate'], 
            cmap='viridis', 
            alpha=0.75, 
            edgecolors='#222222', 
            linewidth=1.0,
            zorder=3
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Return & Cancellation Rate (%)', fontsize=8.8)

        custom_label_pos = {
            'Australia': (56, 9750),
            'Netherlands': (138, 8450),
            'Japan': (19, 3650),
            'Switzerland': (96, 2850),
            'Sweden': (12, 2380),
            'Norway': (12, 1020),
            'Portugal': (50, -250),
            'Spain': (132, 1480),
            'Belgium': (145, 200),
            'EIRE': (284, 2150),
            'France': (370, 1650),
            'Germany': (455, 1650),
        }

        for _, row in top_markets.iterrows():
            country = row['Country']
            x_pos = row['Order_Count']
            y_pos = row['AOV']
            gmv_k = row['Gross_Revenue'] / 1000
            tx, ty = custom_label_pos.get(country, (x_pos, y_pos + 300))

            ax.annotate(
                f"{country}\n(£{gmv_k:.0f}k)", 
                xy=(x_pos, y_pos),
                xytext=(tx, ty),
                ha='center', 
                va='center', 
                fontsize=7.8,
                color='#111111',
                arrowprops=dict(arrowstyle="-", color='#666666', lw=0.6),
                bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="#cccccc", alpha=0.92, lw=0.5),
                zorder=4
            )

        ax.set_xlabel("Wholesale Order Transaction Velocity (Number of Invoices)")
        ax.set_ylabel("Average Order Value (Line-Weighted AOV in £ GBP)")
        ax.set_xlim(-15, 520)
        ax.set_ylim(-600, 11500)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig8_overseas_market_bubble.png")
        plt.close()

    def _sync_artifacts(self):
        for fig_name in [
            "fig1_distribution_revenue.png",
            "fig2_composition_pareto.png",
            "fig3_change_monthly_trend.png",
            "fig4_relationship_rfm_scatter.png",
            "fig5_model_clv_budget_allocation.png",
            "fig6_recsys_benchmark_comparison.png",
            "fig7_clv_benchmark_comparison.png",
            "fig8_overseas_market_bubble.png"
        ]:
            src_file = FIGURES_DIR / fig_name
            if src_file.exists():
                shutil.copy(src_file, ARTIFACT_DIR / fig_name)
