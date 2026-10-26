import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def generate_teaser_figure(output_path):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman'],
        'mathtext.fontset': 'stix',
        'font.size': 8.0
    })

    # High-resolution panorama with generous margins
    fig, ax = plt.subplots(figsize=(11.6, 3.2), dpi=300)
    ax.set_xlim(0, 116)
    ax.set_ylim(0, 32)
    ax.axis('off')

    # Color Palette (CVPR Top-Tier Academic)
    NAVY = '#1b365d'
    SLATE = '#4a7bb0'
    FOREST = '#2d6a4f'
    CRIMSON = '#8b263e'
    AMBER = '#b45309'
    CHARCOAL = '#1a1a1a'
    BG_CARD = '#f8fafc'
    BORDER_CARD = '#94a3b8'
    BG_SUB = '#ffffff'

    # -------------------------------------------------------------
    # Module 1: Data Auditing & Anomaly Decoupling (x: 1 to 27)
    # -------------------------------------------------------------
    box1 = patches.FancyBboxPatch((1, 0.8), 26, 30.4, boxstyle="round,pad=0.4,rounding_size=1.0",
                                  facecolor=BG_CARD, edgecolor=BORDER_CARD, linewidth=1.0)
    ax.add_patch(box1)
    
    h1 = patches.FancyBboxPatch((1, 27.2), 26, 4.0, boxstyle="round,pad=0.2,rounding_size=0.6",
                                 facecolor=NAVY, edgecolor='none')
    ax.add_patch(h1)
    ax.text(14, 29.2, "1. AUDITING & DECOUPLING", color='white', weight='bold', fontsize=8.2, ha='center', va='center')

    ax.text(14, 24.8, "UCI Corpus: N = 541,909\n4,334 Accounts | £10.25M GMV", 
            ha='center', va='center', fontsize=6.8, color=CHARCOAL, style='italic')
    
    sub1a = patches.FancyBboxPatch((2, 16.5), 24, 6.5, boxstyle="round,pad=0.3,rounding_size=0.5",
                                   facecolor=BG_SUB, edgecolor=SLATE, linewidth=0.8)
    ax.add_patch(sub1a)
    ax.text(3, 21.0, "Track A: Clean Merchandise", weight='bold', fontsize=6.8, color=SLATE)
    ax.text(3, 18.2, "• 522,709 sales lines (£10.25M)\n• 23,260 baskets | 3,665 SKUs", fontsize=6.2, color=CHARCOAL)

    sub1b = patches.FancyBboxPatch((2, 1.8), 24, 13.5, boxstyle="round,pad=0.3,rounding_size=0.5",
                                   facecolor=BG_SUB, edgecolor=CRIMSON, linewidth=0.8)
    ax.add_patch(sub1b)
    ax.text(3, 13.2, "Track B: Forensic Anomaly Audit", weight='bold', fontsize=6.8, color=CRIMSON)
    ax.text(3, 10.5, "Type I: Instant Admin Reversals\n  £245.7k (51.6%) | Cust 16446/12346\n  Remedy: OMS 30-min hold policy", fontsize=6.0, color=CHARCOAL)
    ax.text(3, 4.8, "Type II: Chronic Physical Returns\n  £230.3k (48.4%) | Regency Cakestand\n  Remedy: Packaging re-engineering", fontsize=6.0, color=CHARCOAL)

    # Arrow 1 -> 2
    ax.annotate('', xy=(29.2, 16), xytext=(27.2, 16),
                arrowprops=dict(facecolor=NAVY, edgecolor=NAVY, width=1.5, headwidth=5, headlength=5))

    # -------------------------------------------------------------
    # Module 2: Candidate Retrieval RecSys (Matrix 1) (x: 29.5 to 55.5)
    # -------------------------------------------------------------
    box2 = patches.FancyBboxPatch((29.5, 0.8), 26, 30.4, boxstyle="round,pad=0.4,rounding_size=1.0",
                                  facecolor=BG_CARD, edgecolor=BORDER_CARD, linewidth=1.0)
    ax.add_patch(box2)
    
    h2 = patches.FancyBboxPatch((29.5, 27.2), 26, 4.0, boxstyle="round,pad=0.2,rounding_size=0.6",
                                 facecolor=NAVY, edgecolor='none')
    ax.add_patch(h2)
    ax.text(42.5, 29.2, "2. CANDIDATE RETRIEVAL", color='white', weight='bold', fontsize=8.2, ha='center', va='center')

    ax.text(42.5, 24.8, "Leave-One-Out Evaluation\n3,655 Holdout Test Baskets", 
            ha='center', va='center', fontsize=6.8, color=CHARCOAL, style='italic')

    sub2a = patches.FancyBboxPatch((30.5, 11.5), 24, 11.5, boxstyle="round,pad=0.3,rounding_size=0.5",
                                   facecolor=BG_SUB, edgecolor=SLATE, linewidth=0.8)
    ax.add_patch(sub2a)
    ax.text(31.5, 21.0, "Multi-Channel Pipelines:", weight='bold', fontsize=6.8, color=SLATE)
    ax.text(31.5, 18.2, "• ItemCF: 16.78% Hit@10 | 55.0% Cov", fontsize=6.2, color=CHARCOAL)
    ax.text(31.5, 15.5, "• Truncated SVD: 13.44% Hit@10", fontsize=6.2, color=CHARCOAL)
    ax.text(31.5, 12.8, "• Item2Vec: 73.3% Cov (Max Novelty)", fontsize=6.2, color=CHARCOAL)

    sub2b = patches.FancyBboxPatch((30.5, 1.8), 24, 8.5, boxstyle="round,pad=0.3,rounding_size=0.5",
                                   facecolor='#f0fdf4', edgecolor=FOREST, linewidth=0.9)
    ax.add_patch(sub2b)
    ax.text(31.5, 8.2, "Proposed: Hybrid RRF Fusion", weight='bold', fontsize=7.0, color=FOREST)
    ax.text(31.5, 5.8, "• HitRate@10: 18.64% (+11.1% gain)", weight='bold', fontsize=6.2, color=FOREST)
    ax.text(31.5, 3.4, "• Coverage: 58.51% (Pareto Dominance)", fontsize=6.0, color=CHARCOAL)

    # Arrow 2 -> 3
    ax.annotate('', xy=(57.7, 16), xytext=(55.7, 16),
                arrowprops=dict(facecolor=NAVY, edgecolor=NAVY, width=1.5, headwidth=5, headlength=5))

    # -------------------------------------------------------------
    # Module 3: Forward CLV Forecasting (Matrix 2) (x: 58 to 84.5)
    # -------------------------------------------------------------
    box3 = patches.FancyBboxPatch((58, 0.8), 26.5, 30.4, boxstyle="round,pad=0.4,rounding_size=1.0",
                                  facecolor=BG_CARD, edgecolor=BORDER_CARD, linewidth=1.0)
    ax.add_patch(box3)

    h3 = patches.FancyBboxPatch((58, 27.2), 26.5, 4.0, boxstyle="round,pad=0.2,rounding_size=0.6",
                                 facecolor=NAVY, edgecolor='none')
    ax.add_patch(h3)
    ax.text(71.25, 29.2, "3. FORWARD CLV BENCHMARK", color='white', weight='bold', fontsize=8.2, ha='center', va='center')

    ax.text(71.25, 24.8, "Out-of-Time Q4 Evaluation\n3,309 Calibration Accounts", 
            ha='center', va='center', fontsize=6.8, color=CHARCOAL, style='italic')

    sub3a = patches.FancyBboxPatch((59, 11.5), 24.5, 11.5, boxstyle="round,pad=0.3,rounding_size=0.5",
                                   facecolor=BG_SUB, edgecolor=SLATE, linewidth=0.8)
    ax.add_patch(sub3a)
    ax.text(60, 21.0, "Model Architectures Evaluated:", weight='bold', fontsize=6.8, color=SLATE)
    ax.text(60, 18.2, "• Baselines: K-Means (£945), OLS (£842)", fontsize=6.2, color=CHARCOAL)
    ax.text(60, 15.5, "• Stochastic: BG/NBD + GG (MAE £642)", fontsize=6.2, color=CHARCOAL)
    ax.text(60, 12.8, "• Dual Trees: Random Forest + GBDT(L1)", fontsize=6.2, color=CHARCOAL)

    sub3b = patches.FancyBboxPatch((59, 1.8), 24.5, 8.5, boxstyle="round,pad=0.3,rounding_size=0.5",
                                   facecolor='#f0fdf4', edgecolor=FOREST, linewidth=0.9)
    ax.add_patch(sub3b)
    ax.text(60, 8.2, "Proposed: Probabilistic Stacking", weight='bold', fontsize=7.0, color=FOREST)
    ax.text(60, 5.8, "• Holdout MAE: £593.25 (-£352 cut)", weight='bold', fontsize=6.2, color=FOREST)
    ax.text(60, 3.4, "• Captures 72.30% of Q4 Total GMV", fontsize=6.0, color=CHARCOAL)

    # Arrow 3 -> 4
    ax.annotate('', xy=(86.7, 16), xytext=(84.7, 16),
                arrowprops=dict(facecolor=NAVY, edgecolor=NAVY, width=1.5, headwidth=5, headlength=5))

    # -------------------------------------------------------------
    # Module 4: Knapsack Budget Optimization & Policy (x: 87 to 115)
    # -------------------------------------------------------------
    box4 = patches.FancyBboxPatch((87, 0.8), 28, 30.4, boxstyle="round,pad=0.4,rounding_size=1.0",
                                  facecolor=BG_CARD, edgecolor=BORDER_CARD, linewidth=1.0)
    ax.add_patch(box4)

    h4 = patches.FancyBboxPatch((87, 27.2), 28, 4.0, boxstyle="round,pad=0.2,rounding_size=0.6",
                                 facecolor=NAVY, edgecolor='none')
    ax.add_patch(h4)
    ax.text(101, 29.2, "4. 2012 CAPITAL ALLOCATION", color='white', weight='bold', fontsize=8.2, ha='center', va='center')

    ax.text(101, 24.8, "Constrained Knapsack Optimization\nmax sum y_hat_i * Delta R_i  s.t. sum c_i <= B", 
            ha='center', va='center', fontsize=6.5, color=CHARCOAL, style='italic')

    sub4a = patches.FancyBboxPatch((88, 11.5), 26, 11.5, boxstyle="round,pad=0.3,rounding_size=0.5",
                                   facecolor=BG_SUB, edgecolor=AMBER, linewidth=0.8)
    ax.add_patch(sub4a)
    ax.text(89, 21.0, "Four-Quadrant Capital Policy:", weight='bold', fontsize=6.8, color=AMBER)
    ax.text(89, 18.2, "• Q2 Dormant VIP (89 accts): 50% (TOP)", weight='bold', fontsize=6.2, color=CHARCOAL)
    ax.text(89, 15.5, "• Q1 Active VIP: 15% (Zero Ad Waste)", fontsize=6.2, color=CHARCOAL)
    ax.text(89, 12.8, "• Q3 Growth: 25% | Q4 Inactive: 10%", fontsize=6.2, color=CHARCOAL)

    sub4b = patches.FancyBboxPatch((88, 1.8), 26, 8.5, boxstyle="round,pad=0.3,rounding_size=0.5",
                                   facecolor='#fefce8', edgecolor=AMBER, linewidth=0.9)
    ax.add_patch(sub4b)
    ax.text(89, 8.2, "Macro Commercial Payoff:", weight='bold', fontsize=7.0, color=AMBER)
    ax.text(89, 5.8, "• £360k+ Leaking Revenue Recovered", weight='bold', fontsize=6.2, color=CHARCOAL)
    ax.text(89, 3.4, "• Netherlands Hub (AOV £7,780, 0.14% ret)", fontsize=6.0, color=CHARCOAL)

    plt.tight_layout(pad=0.08)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"[Teaser] Successfully generated perfectly-padded {output_path}")

if __name__ == '__main__':
    out = Path("/Users/davide/Developer/coursework/IE5005/mini_project/figures/fig0_teaser_pipeline.png")
    generate_teaser_figure(out)
