"""
Recommender System Offline Evaluation Protocol (Leave-One-Out Basket Completion)
Author: Chen Davide (A0351457B)
Course: IE5005 Data Analytics for Industrial Engineers
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import RANDOM_SEED, OUTPUT_DIR
from src.models.recommender import (
    PopularityRecommender,
    AprioriRecommender,
    HierarchicalClusteringRecommender,
    ItemCFRecommender,
    TruncatedSVDRecommender,
    Item2VecRecommender,
    HybridEnsembleRecommender
)

class RecSysEvaluator:
    """
    Executes standard offline Leave-One-Out evaluation on unseen shopping carts.
    """

    def __init__(self, df_sales: pd.DataFrame):
        self.df_sales = df_sales

    def evaluate_all(self) -> pd.DataFrame:
        basket_sizes = self.df_sales.groupby('InvoiceNo')['StockCode'].nunique()
        valid_invoices = basket_sizes[basket_sizes >= 2].index
        df_valid = self.df_sales[self.df_sales['InvoiceNo'].isin(valid_invoices)].copy()

        baskets_dict = df_valid.groupby('InvoiceNo')['StockCode'].apply(lambda x: list(set(x))).to_dict()
        invoices = list(baskets_dict.keys())

        np.random.seed(RANDOM_SEED)
        train_invs, test_invs = train_test_split(invoices, test_size=0.20, random_state=RANDOM_SEED)
        train_baskets = [baskets_dict[inv] for inv in train_invs]

        train_df = df_valid[df_valid['InvoiceNo'].isin(train_invs)]
        item_freq = train_df.groupby('StockCode')['InvoiceNo'].nunique()
        vocab = set(item_freq[item_freq >= 3].index)

        # Calculate item probabilities for novelty computation
        total_item_occurrences = item_freq[item_freq.index.isin(vocab)].sum()
        item_probs = (item_freq[item_freq.index.isin(vocab)] / total_item_occurrences).to_dict()
        default_prob = 1.0 / (total_item_occurrences + 1)

        models = {
            "Baseline: Popularity (Top Sellers)": (PopularityRecommender(), "Heuristic Baseline", "Frequency Heuristic (Zero Context)"),
            "Lecture 8: Apriori Association Rules": (AprioriRecommender(), "Lecture Curriculum Method", "Frequent Pattern Mining (Support & Lift)"),
            "Lecture 8: Hierarchical Clustering (Ward)": (HierarchicalClusteringRecommender(), "Lecture Curriculum Method", "Agglomerative Hierarchical Dendrogram"),
            "Classic Benchmark: ItemCF (Jaccard)": (ItemCFRecommender(), "Classic Industry Benchmark", "Neighborhood Collaborative Filtering"),
            "Latent Factor: Matrix Factorization (SVD)": (TruncatedSVDRecommender(), "Linear Latent Factor", "Low-Rank Matrix Approximation"),
            "Frontier SOTA: Item2Vec (Neural Skip-Gram)": (Item2VecRecommender(), "Frontier SOTA Deep Learning", "Continuous Latent Representation & Dense Retrieval"),
            "Proposed SOTA: Multi-Channel Hybrid Ensemble": (HybridEnsembleRecommender(), "Proposed SOTA (Multi-Channel RRF)", "Reciprocal Rank Fusion (ItemCF + SVD + Item2Vec)")
        }

        print(f"[RecSysEvaluator] Training {len(models)} models on {len(train_invs):,} baskets...")
        for name, (model, _, _) in models.items():
            model.fit(train_baskets, vocab)

        print(f"[RecSysEvaluator] Evaluating on {len(test_invs):,} test baskets...")
        hits_5 = {m: 0 for m in models}
        hits_10 = {m: 0 for m in models}
        ndcg_10 = {m: 0.0 for m in models}
        mrr = {m: 0.0 for m in models}
        novelty_sum = {m: 0.0 for m in models}
        catalogs = {m: set() for m in models}

        n_eval = 0
        for inv in test_invs:
            b = [it for it in baskets_dict[inv] if it in vocab]
            if len(b) < 2:
                continue
            target = np.random.choice(b)
            context = [it for it in b if it != target]
            n_eval += 1

            for m_name, (model, _, _) in models.items():
                recs = model.recommend(context, top_k=10)
                catalogs[m_name].update(recs)
                if target in recs[:5]: 
                    hits_5[m_name] += 1
                if target in recs[:10]: 
                    hits_10[m_name] += 1
                    pos = recs.index(target)
                    ndcg_10[m_name] += 1.0 / np.log2(pos + 2)
                if target in recs: 
                    mrr[m_name] += 1.0 / (recs.index(target) + 1)
                
                # Novelty calculation (self-information in bits: -log2(p))
                if recs:
                    rec_novelties = [-np.log2(item_probs.get(it, default_prob)) for it in recs]
                    novelty_sum[m_name] += np.mean(rec_novelties)

        records = []
        for m_name, (_, cat, paradigm) in models.items():
            records.append({
                "Model": m_name,
                "HitRate@5 (%)": f"{(hits_5[m_name] / n_eval * 100):.2f}%",
                "HitRate@10 (%)": f"{(hits_10[m_name] / n_eval * 100):.2f}%",
                "NDCG@10": f"{(ndcg_10[m_name] / n_eval):.4f}",
                "MRR": f"{(mrr[m_name] / n_eval):.4f}",
                "Novelty (bits)": f"{(novelty_sum[m_name] / n_eval):.2f}",
                "Catalog Coverage (%)": f"{(len(catalogs[m_name]) / len(vocab) * 100):.2f}%",
                "Category": cat,
                "Theoretical Paradigm": paradigm
            })

        df_bench = pd.DataFrame(records)
        df_bench.to_csv(OUTPUT_DIR / "benchmark_recsys_expanded.csv", index=False)
        return df_bench
