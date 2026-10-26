"""
Candidate Retrieval Recommender Models Module (Task 1)
Implements Baselines, Lecture 8 Curriculum Methods, Classic CF, SOTA Item2Vec,
and the SOTA Multi-Channel Hybrid Ensemble Recommender.
Author: Chen Davide (A0351457B)
Course: IE5005 Data Analytics for Industrial Engineers
"""

from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from gensim.models import Word2Vec
from src.config import ITEM2VEC_PARAMS

class BaseRecommender(ABC):
    """Abstract Base Class for Recommender Candidate Retrieval."""
    
    @abstractmethod
    def fit(self, train_baskets: list[list[str]], vocab: set[str]):
        pass

    @abstractmethod
    def recommend(self, context_items: list[str], top_k: int = 10) -> list[str]:
        pass


class PopularityRecommender(BaseRecommender):
    """Heuristic Frequency Baseline (Global Top Sellers)."""
    
    def __init__(self):
        self.popular_items = []

    def fit(self, train_baskets: list[list[str]], vocab: set[str]):
        freq = defaultdict(int)
        for b in train_baskets:
            for it in b:
                if it in vocab:
                    freq[it] += 1
        self.popular_items = [it for it, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)]

    def recommend(self, context_items: list[str], top_k: int = 10) -> list[str]:
        return [it for it in self.popular_items if it not in context_items][:top_k]


class AprioriRecommender(BaseRecommender):
    """IE5005 Lecture 08: Frequent Itemset Association Rules (Support & Lift)."""
    
    def __init__(self, min_co_occur: int = 3):
        self.min_co_occur = min_co_occur
        self.rules = defaultdict(list)
        self.fallback = []

    def fit(self, train_baskets: list[list[str]], vocab: set[str]):
        n_baskets = len(train_baskets)
        item_counts = defaultdict(int)
        pair_counts = defaultdict(int)

        for b in train_baskets:
            b_clean = [it for it in b if it in vocab]
            for it in b_clean:
                item_counts[it] += 1
            for i in range(len(b_clean)):
                for j in range(i + 1, len(b_clean)):
                    pair_counts[(b_clean[i], b_clean[j])] += 1
                    pair_counts[(b_clean[j], b_clean[i])] += 1

        self.fallback = [it for it, _ in sorted(item_counts.items(), key=lambda x: x[1], reverse=True)]

        for (it1, it2), cnt in pair_counts.items():
            if cnt >= self.min_co_occur:
                supp1 = item_counts[it1] / n_baskets
                supp2 = item_counts[it2] / n_baskets
                lift = (cnt / n_baskets) / (supp1 * supp2)
                self.rules[it1].append((it2, lift))

        for it1 in self.rules:
            self.rules[it1] = [it2 for it2, _ in sorted(self.rules[it1], key=lambda x: x[1], reverse=True)[:20]]

    def recommend(self, context_items: list[str], top_k: int = 10) -> list[str]:
        scores = defaultdict(float)
        for it in context_items:
            for rank_idx, cand in enumerate(self.rules.get(it, [])):
                if cand not in context_items:
                    scores[cand] += 1.0 / (rank_idx + 1)
        recs = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
        if len(recs) < top_k:
            for p in self.fallback:
                if p not in context_items and p not in recs:
                    recs.append(p)
                if len(recs) == top_k:
                    break
        return recs


class HierarchicalClusteringRecommender(BaseRecommender):
    """IE5005 Lecture 08: Agglomerative Hierarchical Dendrogram Clustering."""
    
    def __init__(self, n_clusters: int = 50):
        self.n_clusters = n_clusters
        self.item2cluster = {}
        self.cluster_items = defaultdict(list)
        self.fallback = []

    def fit(self, train_baskets: list[list[str]], vocab: set[str]):
        vocab_list = sorted(list(vocab))
        item2idx = {it: idx for idx, it in enumerate(vocab_list)}
        
        rows, cols, data = [], [], []
        for inv_idx, b in enumerate(train_baskets):
            for it in b:
                if it in item2idx:
                    rows.append(inv_idx)
                    cols.append(item2idx[it])
                    data.append(1.0)
        user_item_matrix = csr_matrix((data, (rows, cols)), shape=(len(train_baskets), len(vocab_list)))
        svd = TruncatedSVD(n_components=64, random_state=42)
        item_emb = svd.fit_transform(user_item_matrix.T)
        norm_emb = item_emb / (np.linalg.norm(item_emb, axis=1, keepdims=True) + 1e-9)

        hier = AgglomerativeClustering(n_clusters=self.n_clusters, metric='euclidean', linkage='ward')
        labels = hier.fit_predict(norm_emb)

        item_freq = defaultdict(int)
        for b in train_baskets:
            for it in b:
                if it in vocab:
                    item_freq[it] += 1
        self.fallback = [it for it, _ in sorted(item_freq.items(), key=lambda x: x[1], reverse=True)]

        for it_idx, c_lbl in enumerate(labels):
            code = vocab_list[it_idx]
            self.item2cluster[code] = c_lbl
            self.cluster_items[c_lbl].append((code, item_freq[code]))

        for c in self.cluster_items:
            self.cluster_items[c] = [it for it, _ in sorted(self.cluster_items[c], key=lambda x: x[1], reverse=True)]

    def recommend(self, context_items: list[str], top_k: int = 10) -> list[str]:
        clusters = [self.item2cluster[it] for it in context_items if it in self.item2cluster]
        scores = defaultdict(int)
        for c in clusters:
            for cand in self.cluster_items[c][:15]:
                if cand not in context_items:
                    scores[cand] += 1
        recs = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
        if len(recs) < top_k:
            for p in self.fallback:
                if p not in context_items and p not in recs:
                    recs.append(p)
                if len(recs) == top_k:
                    break
        return recs


class ItemCFRecommender(BaseRecommender):
    """Classic Industry Benchmark: Amazon 2003 Item-to-Item Collaborative Filtering (Jaccard)."""
    
    def __init__(self, min_co_occur: int = 2):
        self.min_co_occur = min_co_occur
        self.sim_matrix = defaultdict(list)
        self.fallback = []

    def fit(self, train_baskets: list[list[str]], vocab: set[str]):
        item_counts = defaultdict(int)
        pair_counts = defaultdict(int)

        for b in train_baskets:
            b_clean = [it for it in b if it in vocab]
            for it in b_clean:
                item_counts[it] += 1
            for i in range(len(b_clean)):
                for j in range(i + 1, len(b_clean)):
                    pair_counts[(b_clean[i], b_clean[j])] += 1
                    pair_counts[(b_clean[j], b_clean[i])] += 1

        self.fallback = [it for it, _ in sorted(item_counts.items(), key=lambda x: x[1], reverse=True)]

        for (it1, it2), cnt in pair_counts.items():
            if cnt >= self.min_co_occur:
                jaccard = cnt / (item_counts[it1] + item_counts[it2] - cnt)
                self.sim_matrix[it1].append((it2, jaccard))

        for it1 in self.sim_matrix:
            self.sim_matrix[it1] = [it2 for it2, _ in sorted(self.sim_matrix[it1], key=lambda x: x[1], reverse=True)[:25]]

    def recommend(self, context_items: list[str], top_k: int = 10) -> list[str]:
        scores = defaultdict(float)
        for it in context_items:
            for cand in self.sim_matrix.get(it, []):
                if cand not in context_items:
                    scores[cand] += 1.0
        recs = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
        if len(recs) < top_k:
            for p in self.fallback:
                if p not in context_items and p not in recs:
                    recs.append(p)
                if len(recs) == top_k:
                    break
        return recs


class TruncatedSVDRecommender(BaseRecommender):
    """Linear Latent Factor Recommender (Truncated SVD Matrix Factorization)."""
    
    def __init__(self, n_components: int = 64):
        self.n_components = n_components
        self.vocab_list = []
        self.item2idx = {}
        self.item_norm_emb = None
        self.fallback = []

    def fit(self, train_baskets: list[list[str]], vocab: set[str]):
        self.vocab_list = sorted(list(vocab))
        self.item2idx = {it: idx for idx, it in enumerate(self.vocab_list)}
        
        rows, cols, data = [], [], []
        item_counts = defaultdict(int)
        for inv_idx, b in enumerate(train_baskets):
            for it in b:
                if it in self.item2idx:
                    rows.append(inv_idx)
                    cols.append(self.item2idx[it])
                    data.append(1.0)
                    item_counts[it] += 1

        self.fallback = [it for it, _ in sorted(item_counts.items(), key=lambda x: x[1], reverse=True)]

        user_item_matrix = csr_matrix((data, (rows, cols)), shape=(len(train_baskets), len(self.vocab_list)))
        svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        emb = svd.fit_transform(user_item_matrix.T)
        self.item_norm_emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

    def recommend(self, context_items: list[str], top_k: int = 10) -> list[str]:
        context_indices = [self.item2idx[it] for it in context_items if it in self.item2idx]
        if context_indices:
            q_vec = self.item_norm_emb[context_indices].mean(axis=0)
            q_vec /= (np.linalg.norm(q_vec) + 1e-9)
            sims = np.dot(self.item_norm_emb, q_vec)
            top_indices = np.argsort(sims)[::-1]
            return [self.vocab_list[idx] for idx in top_indices if self.vocab_list[idx] not in context_items][:top_k]
        return [it for it in self.fallback if it not in context_items][:top_k]


class Item2VecRecommender(BaseRecommender):
    """Frontier Dense Embedding Recommender (Word2Vec Skip-Gram on Basket Sessions)."""
    
    def __init__(self, params: dict = None):
        self.params = params or ITEM2VEC_PARAMS
        self.model = None
        self.fallback = []

    def fit(self, train_baskets: list[list[str]], vocab: set[str]):
        clean_baskets = [[it for it in b if it in vocab] for b in train_baskets]
        item_counts = defaultdict(int)
        for b in clean_baskets:
            for it in b:
                item_counts[it] += 1
        self.fallback = [it for it, _ in sorted(item_counts.items(), key=lambda x: x[1], reverse=True)]

        self.model = Word2Vec(
            sentences=clean_baskets,
            vector_size=self.params["vector_size"],
            window=self.params["window"],
            min_count=self.params["min_count"],
            sg=self.params["sg"],
            negative=self.params["negative"],
            workers=self.params["workers"],
            epochs=self.params["epochs"],
            seed=self.params["seed"]
        )

    def recommend(self, context_items: list[str], top_k: int = 10) -> list[str]:
        valid_context = [it for it in context_items if it in self.model.wv]
        if valid_context:
            sims = self.model.wv.most_similar(positive=valid_context, topn=top_k * 4)
            return [sim_it for sim_it, _ in sims if sim_it not in context_items][:top_k]
        return [it for it in self.fallback if it not in context_items][:top_k]


class HybridEnsembleRecommender(BaseRecommender):
    """
    PROPOSED SOTA: Multi-Channel Hybrid Retrieval (Reciprocal Rank Fusion of ItemCF + SVD + Item2Vec).
    Achieves Pareto dominance by uniting high local accuracy with broad semantic catalog discovery.
    """

    def __init__(self, itemcf_weight: float = 0.50, svd_weight: float = 0.25, i2v_weight: float = 0.25):
        self.itemcf = ItemCFRecommender()
        self.svd = TruncatedSVDRecommender()
        self.i2v = Item2VecRecommender()
        self.w_cf = itemcf_weight
        self.w_svd = svd_weight
        self.w_i2v = i2v_weight

    def fit(self, train_baskets: list[list[str]], vocab: set[str]):
        self.itemcf.fit(train_baskets, vocab)
        self.svd.fit(train_baskets, vocab)
        self.i2v.fit(train_baskets, vocab)

    def recommend(self, context_items: list[str], top_k: int = 10) -> list[str]:
        recs_cf = self.itemcf.recommend(context_items, top_k=25)
        recs_svd = self.svd.recommend(context_items, top_k=25)
        recs_i2v = self.i2v.recommend(context_items, top_k=25)

        # Reciprocal Rank Fusion (RRF) with channel weights
        rrf_scores = defaultdict(float)
        k_rrf = 30
        for rank, item in enumerate(recs_cf):
            rrf_scores[item] += self.w_cf / (k_rrf + rank + 1)
        for rank, item in enumerate(recs_svd):
            rrf_scores[item] += self.w_svd / (k_rrf + rank + 1)
        for rank, item in enumerate(recs_i2v):
            rrf_scores[item] += self.w_i2v / (k_rrf + rank + 1)

        sorted_recs = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [it for it in sorted_recs if it not in context_items][:top_k]
