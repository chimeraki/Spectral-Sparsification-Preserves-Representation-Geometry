#!/usr/bin/env python3
"""
final_fashion_cora_paul15_figs.py

Runs real-data experiments on exactly:
- FashionMNIST
- Cora
- Paul15

and generates the final figures:
1. fig_final_hidden_gram_to_nn20
2. fig_final_knn_heatmap
3. fig_final_centroids
4. fig_final_procrustes
5. fig_final_twopanel_chain_vertical_from_csv

Also saves:
- datasets.csv
- empirical_rows.csv
- final_metrics_fashion_cora_paul15.csv
- viz_payload_<dataset>_c*.npz
- summary_metrics.json

Notes:
- FashionMNIST: real image data, kNN graph built from PCA features
- Cora: real citation graph from PyG
- Paul15: real single-cell data from Scanpy, kNN graph built from expression features
- To keep this laptop-friendly, geometry metrics are computed on subsampled test nodes
- Scatter figures use repeated sparsifier draws per budget
- Qualitative figures use one representative payload per budget (rep == 0)

Install:
    pip install torch torchvision torch-geometric scikit-learn scipy pandas matplotlib scanpy anndata

Usage:
    python final_fashion_cora_paul15_figs.py --quick --outdir final_real_quick
    python final_fashion_cora_paul15_figs.py --outdir final_real
"""

import argparse
import copy
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as stats
from scipy.linalg import orthogonal_procrustes
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


DATASET_ORDER = ["FashionMNIST", "Cora", "Paul15"]
DATASET_COLORS = {
    "FashionMNIST": "#1f77b4",
    "Cora": "#9467bd",
    "Paul15": "#2ca02c",
}


# ============================================================
# style / utils
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def print_check(msg: str):
    print(f"[check] {msg}", flush=True)


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2))


def set_style():
    plt.rcParams.update({
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "figure.titlesize": 12,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "grid.linewidth": 0.7,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_both(fig, outdir: Path, stem: str):
    fig.savefig(outdir / f"{stem}.pdf")
    fig.savefig(outdir / f"{stem}.png", dpi=300)
    plt.close(fig)


def light_ticks(ax):
    ax.tick_params(axis="both", which="major", length=4, width=0.8)


def normalize_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def fit_line(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return None
    return np.polyfit(x, y, 1)


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


# ============================================================
# sparse helpers
# ============================================================

def symmetrize_weighted(adj: sp.spmatrix) -> sp.csr_matrix:
    coo = ((adj + adj.T) * 0.5).astype(np.float64).tocoo()
    mask = coo.row != coo.col
    out = sp.csr_matrix((coo.data[mask], (coo.row[mask], coo.col[mask])), shape=adj.shape, dtype=np.float64)
    out.eliminate_zeros()
    return out


def adjacency_from_edges(n, u, v, w):
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    vals = np.concatenate([w, w])
    mask = rows != cols
    out = sp.csr_matrix((vals[mask], (rows[mask], cols[mask])), shape=(n, n), dtype=np.float64)
    out.eliminate_zeros()
    return out


def symmetric_edge_list(adj: sp.csr_matrix):
    coo = sp.triu(adj, k=1).tocoo()
    return coo.row.astype(int), coo.col.astype(int), coo.data.astype(np.float64)


def normalized_adjacency_with_self_loops(adj: sp.csr_matrix) -> sp.csr_matrix:
    A = adj + sp.eye(adj.shape[0], format="csr", dtype=np.float64)
    deg = np.asarray(A.sum(axis=1)).reshape(-1)
    inv_sqrt = np.power(np.maximum(deg, 1e-12), -0.5)
    D = sp.diags(inv_sqrt)
    return (D @ A @ D).tocsr()


def combinatorial_laplacian(adj: sp.csr_matrix) -> sp.csr_matrix:
    deg = np.asarray(adj.sum(axis=1)).reshape(-1)
    return (sp.diags(deg) - adj).tocsr()


def to_torch_sparse(mat: sp.csr_matrix) -> torch.Tensor:
    coo = mat.tocoo()
    idx = np.vstack([coo.row, coo.col])
    i = torch.tensor(idx, dtype=torch.long)
    v = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(i, v, size=coo.shape).coalesce()


# ============================================================
# data containers / loaders
# ============================================================

@dataclass
class DatasetPack:
    name: str
    adj: sp.csr_matrix
    x: np.ndarray
    y: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray


def make_masks(y, train_per_class, val_per_class, seed=0):
    rng = np.random.default_rng(seed)
    train = np.zeros(len(y), dtype=bool)
    val = np.zeros(len(y), dtype=bool)
    test = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        ids = np.where(y == c)[0]
        rng.shuffle(ids)
        a = min(train_per_class, len(ids))
        b = min(val_per_class, max(0, len(ids) - a))
        train[ids[:a]] = True
        val[ids[a:a+b]] = True
        test[ids[a+b:]] = True
    return train, val, test


def build_weighted_knn_graph(X: np.ndarray, k: int = 15) -> sp.csr_matrix:
    n = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n), metric="euclidean")
    nbrs.fit(X)
    dists, inds = nbrs.kneighbors(X)
    dists = dists[:, 1:]
    inds = inds[:, 1:]
    sigma = dists[:, -1] + 1e-12

    rows, cols, vals = [], [], []
    for i in range(n):
        for dj, j in zip(dists[i], inds[i]):
            w = math.exp(-(dj ** 2) / (sigma[i] * sigma[int(j)] + 1e-12))
            rows.append(i)
            cols.append(int(j))
            vals.append(float(w))
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
    return symmetrize_weighted(A)


def maybe_reduce_features(X: np.ndarray, max_dim=64, seed=0):
    X = np.asarray(X, dtype=np.float32)
    if X.shape[1] <= max_dim:
        return X
    return PCA(n_components=max_dim, random_state=seed).fit_transform(X).astype(np.float32)


def load_fashion_mnist(seed=0, n_per_class=220, pca_dim=50, k=15):
    from torchvision import datasets
    print_check("Loading FashionMNIST")
    ds = datasets.FashionMNIST(root="./fashion_mnist_data", train=True, download=True)
    X = ds.data.numpy().reshape(len(ds), -1).astype(np.float32) / 255.0
    y = np.array(ds.targets, dtype=np.int64)

    rng = np.random.default_rng(seed)
    keep = []
    for c in range(10):
        ids = np.where(y == c)[0]
        rng.shuffle(ids)
        keep.extend(ids[:n_per_class].tolist())
    keep = np.array(sorted(keep))
    X = X[keep]
    y = y[keep]

    if pca_dim is not None and pca_dim < X.shape[1]:
        X = PCA(n_components=pca_dim, random_state=seed).fit_transform(X).astype(np.float32)

    A = build_weighted_knn_graph(X, k=k)
    train, val, test = make_masks(y, train_per_class=20, val_per_class=40, seed=seed)
    return DatasetPack("FashionMNIST", A, X, y, train, val, test)


def load_cora(seed=0):
    from torch_geometric.datasets import Planetoid
    print_check("Loading Cora")
    ds = Planetoid(root="./pyg_planetoid", name="Cora")
    data = ds[0]
    X = data.x.cpu().numpy().astype(np.float32)
    y = data.y.cpu().numpy().astype(np.int64)
    edge_index = data.edge_index.cpu().numpy()
    u, v = edge_index[0], edge_index[1]
    w = np.ones(len(u), dtype=np.float64)
    A = sp.csr_matrix((w, (u, v)), shape=(X.shape[0], X.shape[0]), dtype=np.float64)
    A = symmetrize_weighted(A)
    train, val, test = make_masks(y, train_per_class=20, val_per_class=50, seed=seed)
    return DatasetPack("Cora", A, X, y, train, val, test)


def load_paul15(seed=0):
    import scanpy as sc
    print_check("Loading Paul15")
    adata = sc.datasets.paul15()
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    X = maybe_reduce_features(X.astype(np.float32), max_dim=64, seed=seed)

    label_col = None
    for cand in ["paul15_clusters", "clusters", "louvain", "cell_type"]:
        if cand in adata.obs.columns:
            label_col = cand
            break
    if label_col is None:
        raise RuntimeError("Could not find a usable label column in paul15")

    labels = adata.obs[label_col].astype(str).to_numpy()
    _, y = np.unique(labels, return_inverse=True)
    y = y.astype(np.int64)

    A = build_weighted_knn_graph(X, k=15)
    train, val, test = make_masks(y, train_per_class=20, val_per_class=30, seed=seed)
    return DatasetPack("Paul15", A, X, y, train, val, test)


# ============================================================
# sparsifier
# ============================================================

def approximate_effective_resistance_scores(adj: sp.csr_matrix, rank: int = 16) -> np.ndarray:
    n = adj.shape[0]
    L = combinatorial_laplacian(adj).astype(np.float64)
    k = min(rank + 1, n - 1)
    try:
        vals, vecs = sp.linalg.eigsh(L, k=k, which="SM")
    except Exception:
        vals, vecs = np.linalg.eigh(L.toarray())
        vals, vecs = vals[:k], vecs[:, :k]

    vals = np.asarray(vals)
    vecs = np.asarray(vecs)
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    mask = vals > 1e-8
    vals = vals[mask]
    vecs = vecs[:, mask]
    if len(vals) == 0:
        vals = np.array([1.0])
        vecs = np.zeros((n, 1))
    Phi = vecs / np.sqrt(vals.reshape(1, -1))

    u, v, w = symmetric_edge_list(adj)
    diff = Phi[u] - Phi[v]
    r = np.sum(diff * diff, axis=1)
    return np.maximum(w * r, 1e-12)


def approximate_resistance_sparsify_by_q(adj: sp.csr_matrix, q: int, seed: int = 0, rank: int = 16) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    u, v, w = symmetric_edge_list(adj)
    scores = approximate_effective_resistance_scores(adj, rank=rank)
    probs = scores / scores.sum()
    q = max(1, int(q))
    samples = rng.choice(len(u), size=q, replace=True, p=probs)
    counts = np.bincount(samples, minlength=len(u))
    keep = np.where(counts > 0)[0]
    new_w = w[keep] * counts[keep] / np.maximum(q * probs[keep], 1e-12)
    return adjacency_from_edges(adj.shape[0], u[keep], v[keep], new_w)


def approximate_empirical_epsilon(adj: sp.csr_matrix, adj_t: sp.csr_matrix, probe_rank: int = 32) -> float:
    n = adj.shape[0]
    r = min(probe_rank, n - 2)
    rng = np.random.default_rng(0)
    P = rng.normal(size=(n, r))
    P = P - P.mean(axis=0, keepdims=True)
    L = combinatorial_laplacian(adj)
    Lt = combinatorial_laplacian(adj_t)
    num = np.sum(P * (Lt @ P), axis=0)
    den = np.sum(P * (L @ P), axis=0)
    den = np.maximum(den, 1e-12)
    ratio = num / den
    return float(max(0.0, max(1.0 - ratio.min(), ratio.max() - 1.0)))


# ============================================================
# model / training
# ============================================================

class TwoLayerGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.lin2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, A_hat, x, return_hidden=False):
        h = torch.sparse.mm(A_hat, x)
        h = F.relu(self.lin1(h))
        z = torch.sparse.mm(A_hat, h)
        out = self.lin2(z)
        if return_hidden:
            return out, h
        return out


def train_teacher(data: DatasetPack, adj: sp.csr_matrix, hidden_dim: int, epochs: int, lr: float, seed: int):
    set_seed(seed)
    device = torch.device("cpu")
    A_hat = to_torch_sparse(normalized_adjacency_with_self_loops(adj)).to(device)
    x = torch.tensor(data.x, dtype=torch.float32, device=device)
    y = torch.tensor(data.y, dtype=torch.long, device=device)
    train_mask = torch.tensor(data.train_mask, dtype=torch.bool, device=device)
    val_mask = torch.tensor(data.val_mask, dtype=torch.bool, device=device)
    test_mask = torch.tensor(data.test_mask, dtype=torch.bool, device=device)

    model = TwoLayerGCN(x.shape[1], hidden_dim, int(y.max().item()) + 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_state, best_val = None, -1e18
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(A_hat, x)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(A_hat, x)
            val_loss = F.cross_entropy(val_logits[val_mask], y[val_mask]).item()
        if -val_loss > best_val:
            best_val = -val_loss
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % max(5, epochs // 3) == 0 or epoch == epochs:
            print_check(f"{data.name} teacher epoch {epoch}/{epochs} val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, hidden = model(A_hat, x, return_hidden=True)
        acc = (logits.argmax(dim=1)[test_mask] == y[test_mask]).float().mean().item()
    return model, hidden.cpu().numpy(), float(acc)


# ============================================================
# geometry metrics
# ============================================================

def subsample_test_embeddings(hidden, test_mask, max_points=500, seed=0):
    ids = np.where(test_mask)[0]
    rng = np.random.default_rng(seed)
    if len(ids) > max_points:
        ids = np.sort(rng.choice(ids, size=max_points, replace=False))
    return hidden[ids], ids


def relative_gram_error_fro(Z, Zt):
    G = Z @ Z.T
    Gt = Zt @ Zt.T
    return float(np.linalg.norm(G - Gt, ord="fro") / (np.linalg.norm(G, ord="fro") + 1e-12))


def topk_neighbors(Z, k=20):
    Z = normalize_rows(Z)
    S = Z @ Z.T
    np.fill_diagonal(S, -np.inf)
    return np.argpartition(-S, kth=min(k, S.shape[1] - 1), axis=1)[:, :k]


def mean_knn_overlap(Z, Zt, k=20):
    Nd = topk_neighbors(Z, k=k)
    Ns = topk_neighbors(Zt, k=k)
    vals = []
    for a in range(Z.shape[0]):
        vals.append(len(set(Nd[a].tolist()) & set(Ns[a].tolist())) / float(k))
    return float(np.mean(vals))


def per_anchor_knn_overlap(Z, Zt, anchors, k=20):
    Nd = topk_neighbors(Z, k=k)
    Ns = topk_neighbors(Zt, k=k)
    vals = []
    for a in anchors:
        vals.append(len(set(Nd[a].tolist()) & set(Ns[a].tolist())) / float(k))
    return np.array(vals, dtype=float)


def procrustes_align(P_dense, P_sparse):
    X = P_dense - P_dense.mean(axis=0, keepdims=True)
    Y = P_sparse - P_sparse.mean(axis=0, keepdims=True)
    R, _ = orthogonal_procrustes(Y, X)
    return X, Y @ R


# ============================================================
# experiment
# ============================================================

def run_dataset_pipeline(data: DatasetPack, c_list, reps, hidden_dim, teacher_epochs, teacher_lr,
                         approx_rank, hidden_eval_subsample, seed):
    print_check(f"=== Running pipeline for {data.name} ===")
    teacher_model, dense_hidden, teacher_acc = train_teacher(
        data, data.adj, hidden_dim=hidden_dim, epochs=teacher_epochs, lr=teacher_lr, seed=seed
    )

    n = data.adj.shape[0]
    scale = n * math.log(max(n, 2))
    m0 = len(symmetric_edge_list(data.adj)[0])

    rows = []
    viz_payloads = []

    total = len(c_list) * reps
    done = 0

    for rep in range(reps):
        for idx, c in enumerate(c_list):
            q = int(round(c * scale))
            sparse_adj = approximate_resistance_sparsify_by_q(
                data.adj, q=q, seed=seed + 1000 * rep + 37 * idx, rank=approx_rank
            )
            eps_emp = approximate_empirical_epsilon(data.adj, sparse_adj, probe_rank=32)
            keep_ratio = len(symmetric_edge_list(sparse_adj)[0]) / float(m0)

            device = torch.device("cpu")
            A_sparse = to_torch_sparse(normalized_adjacency_with_self_loops(sparse_adj)).to(device)
            x = torch.tensor(data.x, dtype=torch.float32, device=device)
            with torch.no_grad():
                _, sparse_hidden = teacher_model(A_sparse, x, return_hidden=True)
            sparse_hidden = sparse_hidden.cpu().numpy()

            Zd, ids = subsample_test_embeddings(dense_hidden, data.test_mask, max_points=hidden_eval_subsample, seed=100 + rep)
            Zs = sparse_hidden[ids]
            gram = relative_gram_error_fro(Zd, Zs)
            nn20 = mean_knn_overlap(Zd, Zs, k=min(20, len(Zd) - 1))

            rows.append({
                "dataset": data.name,
                "rep": rep,
                "budget_multiplier": c,
                "sample_count_q": q,
                "keep_ratio": keep_ratio,
                "eps_emp": eps_emp,
                "hidden_gram_error": gram,
                "nn_overlap_20": nn20,
                "teacher_acc": teacher_acc,
                "n": n,
                "m_unique": m0,
                "feature_dim": data.x.shape[1],
                "classes": len(np.unique(data.y)),
            })

            if rep == 0:
                viz_payloads.append({
                    "dataset": data.name,
                    "budget_multiplier": c,
                    "eps_emp": eps_emp,
                    "dense_hidden": dense_hidden,
                    "sparse_hidden": sparse_hidden,
                    "labels": data.y.copy(),
                    "test_mask": data.test_mask.copy(),
                })

            done += 1
            if done == 1 or done % max(2, total // 6) == 0 or done == total:
                print_check(
                    f"{data.name} {done}/{total}: rep={rep}, c={c:.2f}, "
                    f"eps={eps_emp:.3f}, gram={gram:.3f}, nn20={nn20:.3f}"
                )

    return rows, viz_payloads


# ============================================================
# plotting
# ============================================================

def plot_hidden_gram_to_nn20(df: pd.DataFrame, outdir: Path):
    present = [d for d in DATASET_ORDER if d in set(df["dataset"])]
    fig, axes = plt.subplots(1, len(present), figsize=(4.1 * len(present), 3.25), squeeze=False)

    for j, ds in enumerate(present):
        ax = axes[0, j]
        sdf = df[df["dataset"] == ds].copy()
        x = sdf["hidden_gram_error"].to_numpy()
        y = sdf["nn_overlap_20"].to_numpy()
        color = DATASET_COLORS.get(ds, None)

        ax.scatter(x, y, s=18, alpha=0.68, color=color)
        coef = fit_line(x, y)
        if coef is not None:
            xx = np.linspace(x.min(), x.max(), 200)
            ax.plot(xx, coef[0] * xx + coef[1], linestyle="--", color=color)
        rho = spearman(x, y)
        ax.set_title(f"{ds}\n" + rf"$\rho_s={rho:.2f}$", pad=8)
        ax.set_xlabel("hidden Gram distortion")
        ax.set_ylabel("NN overlap@20")
        light_ticks(ax)

    fig.tight_layout()
    save_both(fig, outdir, "fig_final_hidden_gram_to_nn20")


def plot_twopanel_vertical(df: pd.DataFrame, outdir: Path):
    present = [d for d in DATASET_ORDER if d in set(df["dataset"])]
    fig, axes = plt.subplots(len(present), 2, figsize=(8.7, 3.1 * len(present)), squeeze=False)

    for i, ds in enumerate(present):
        sdf = df[df["dataset"] == ds].copy()
        color = DATASET_COLORS.get(ds, None)

        ax = axes[i, 0]
        x = sdf["eps_emp"].to_numpy()
        y = sdf["hidden_gram_error"].to_numpy()
        ax.scatter(x, y, s=18, alpha=0.68, color=color)
        coef = fit_line(x, y)
        if coef is not None:
            xx = np.linspace(x.min(), x.max(), 200)
            ax.plot(xx, coef[0] * xx + coef[1], linestyle="--", color=color)
        rho = spearman(x, y)
        ax.set_title(f"{ds}: graph distortion → hidden geometry\n" + rf"$\rho_s={rho:.2f}$", pad=8)
        ax.set_xlabel(r"empirical graph distortion $\varepsilon_{\mathrm{emp}}$")
        ax.set_ylabel("hidden Gram distortion")
        light_ticks(ax)

        ax = axes[i, 1]
        x = sdf["hidden_gram_error"].to_numpy()
        y = sdf["nn_overlap_20"].to_numpy()
        ax.scatter(x, y, s=18, alpha=0.68, color=color)
        coef = fit_line(x, y)
        if coef is not None:
            xx = np.linspace(x.min(), x.max(), 200)
            ax.plot(xx, coef[0] * xx + coef[1], linestyle="--", color=color)
        rho = spearman(x, y)
        ax.set_title(f"{ds}: hidden geometry → neighborhood preservation\n" + rf"$\rho_s={rho:.2f}$", pad=8)
        ax.set_xlabel("hidden Gram distortion")
        ax.set_ylabel("NN overlap@20")
        light_ticks(ax)

    fig.tight_layout()
    save_both(fig, outdir, "fig_final_twopanel_chain_vertical_from_csv")


def plot_knn_heatmap(viz_payloads_by_budget, outdir: Path, max_points=700, k=20, n_anchors=20):
    present = [d for d in DATASET_ORDER if d in viz_payloads_by_budget]
    fig, axes = plt.subplots(len(present), 1, figsize=(8.8, 2.4 * len(present)), squeeze=False)

    for i, ds in enumerate(present):
        payloads = viz_payloads_by_budget[ds]
        ids = np.where(payloads[0]["test_mask"])[0]
        rng = np.random.default_rng(2)
        if len(ids) > max_points:
            ids = np.sort(rng.choice(ids, size=max_points, replace=False))

        rng = np.random.default_rng(3)
        anchors = np.sort(rng.choice(len(ids), size=min(n_anchors, len(ids)), replace=False))

        M = []
        labels = []
        for p in payloads:
            Zd = p["dense_hidden"][ids]
            Zs = p["sparse_hidden"][ids]
            M.append(per_anchor_knn_overlap(Zd, Zs, anchors, k=min(k, len(ids) - 1)))
            labels.append(f'c={p["budget_multiplier"]:.2f}\nε={p["eps_emp"]:.2f}')
        M = np.stack(M, axis=1)

        ax = axes[i, 0]
        im = ax.imshow(M, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(f"{ds}: kNN-neighborhood agreement", pad=8)
        ax.set_xlabel("sparsification level")
        ax.set_ylabel("anchor")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticks(np.arange(len(anchors)))
        ax.set_yticklabels([str(a) for a in anchors], fontsize=8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label(f"overlap@{k}")

    fig.tight_layout()
    save_both(fig, outdir, "fig_final_knn_heatmap")


def plot_centroids(viz_payloads_by_budget, outdir: Path, pick="middle", max_points=900):
    present = [d for d in DATASET_ORDER if d in viz_payloads_by_budget]
    fig, axes = plt.subplots(1, len(present), figsize=(4.4 * len(present), 4.0), squeeze=False)

    for j, ds in enumerate(present):
        payloads = viz_payloads_by_budget[ds]
        p = payloads[0] if pick == "first" else payloads[-1] if pick == "last" else payloads[len(payloads) // 2]

        ids = np.where(p["test_mask"])[0]
        rng = np.random.default_rng(4)
        if len(ids) > max_points:
            ids = np.sort(rng.choice(ids, size=max_points, replace=False))

        Zd = p["dense_hidden"][ids]
        Zs = p["sparse_hidden"][ids]
        y = p["labels"][ids]

        classes = np.unique(y)
        Cd = np.stack([Zd[y == c].mean(axis=0) for c in classes], axis=0)
        Cs = np.stack([Zs[y == c].mean(axis=0) for c in classes], axis=0)

        pca = PCA(n_components=2, random_state=0)
        pca.fit(np.vstack([Cd, Cs]))
        Pd = pca.transform(Cd)
        Ps = pca.transform(Cs)

        ax = axes[0, j]
        ax.scatter(Pd[:, 0], Pd[:, 1], s=90, marker="o", color=DATASET_COLORS.get(ds, None), label="dense")
        ax.scatter(Ps[:, 0], Ps[:, 1], s=90, marker="x", color="#d62728", label="sparse")
        for t, c in enumerate(classes):
            ax.plot([Pd[t, 0], Ps[t, 0]], [Pd[t, 1], Ps[t, 1]], linestyle="--", alpha=0.45, color="gray")
            ax.text(Pd[t, 0], Pd[t, 1], str(int(c)), fontsize=8)
        ax.set_title(f"{ds}: class-centroid geometry")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        light_ticks(ax)

    axes[0, 0].legend(frameon=False, loc="best")
    fig.tight_layout()
    save_both(fig, outdir, "fig_final_centroids")


def plot_procrustes(viz_payloads_by_budget, outdir: Path, pick="middle", max_points=700):
    present = [d for d in DATASET_ORDER if d in viz_payloads_by_budget]
    fig, axes = plt.subplots(len(present), 1, figsize=(6.0, 3.7 * len(present)), squeeze=False)

    for i, ds in enumerate(present):
        payloads = viz_payloads_by_budget[ds]
        p = payloads[0] if pick == "first" else payloads[-1] if pick == "last" else payloads[len(payloads) // 2]

        ids = np.where(p["test_mask"])[0]
        rng = np.random.default_rng(5)
        if len(ids) > max_points:
            ids = np.sort(rng.choice(ids, size=max_points, replace=False))

        Zd = p["dense_hidden"][ids]
        Zs = p["sparse_hidden"][ids]
        y = p["labels"][ids]

        pca = PCA(n_components=2, random_state=0)
        pca.fit(np.vstack([Zd, Zs]))
        Pd = pca.transform(Zd)
        Ps = pca.transform(Zs)
        Pd0, Ps0 = procrustes_align(Pd, Ps)

        ax = axes[i, 0]
        ax.scatter(Pd0[:, 0], Pd0[:, 1], c=y, s=10, cmap="tab10", alpha=0.55, marker="o", label="dense")
        ax.scatter(Ps0[:, 0], Ps0[:, 1], c=y, s=10, cmap="tab10", alpha=0.55, marker="x", label="sparse")
        ax.set_title(f"{ds}: Procrustes-aligned dense vs sparse embeddings")
        ax.set_xlabel("aligned PC1")
        ax.set_ylabel("aligned PC2")
        light_ticks(ax)

    axes[0, 0].legend(frameon=False, loc="best")
    fig.tight_layout()
    save_both(fig, outdir, "fig_final_procrustes")


# ============================================================
# main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="final_real")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    set_style()
    set_seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print_check(f"Output dir: {outdir}")

    if args.quick:
        c_list = [0.6, 1.0, 1.5]
        reps = 2
        teacher_epochs = 14
        teacher_lr = 0.01
        approx_rank = 12
        hidden_eval_subsample = 400
        fashion_n_per_class = 180
        max_plot_points = 500
    else:
        c_list = [0.45, 0.60, 0.80, 1.00, 1.25, 1.50]
        reps = 3
        teacher_epochs = 20
        teacher_lr = 0.01
        approx_rank = 16
        hidden_eval_subsample = 500
        fashion_n_per_class = 220
        max_plot_points = 650

    loaders = [
        ("FashionMNIST", lambda: load_fashion_mnist(seed=args.seed, n_per_class=fashion_n_per_class, pca_dim=50, k=15)),
        ("Cora", lambda: load_cora(seed=args.seed + 1)),
        ("Paul15", lambda: load_paul15(seed=args.seed + 2)),
    ]

    dataset_rows = []
    all_rows = []
    viz_payloads_by_budget = {}

    for i, (name, fn) in enumerate(loaders):
        data = fn()

        dataset_rows.append({
            "dataset": data.name,
            "n": int(data.adj.shape[0]),
            "m_unique": int(len(symmetric_edge_list(data.adj)[0])),
            "feature_dim": int(data.x.shape[1]),
            "classes": int(len(np.unique(data.y))),
        })

        hidden_dim = 48 if data.x.shape[1] <= 64 else 64
        rows, payloads = run_dataset_pipeline(
            data=data,
            c_list=c_list,
            reps=reps,
            hidden_dim=hidden_dim,
            teacher_epochs=teacher_epochs,
            teacher_lr=teacher_lr,
            approx_rank=approx_rank,
            hidden_eval_subsample=hidden_eval_subsample,
            seed=args.seed + 100 * i
        )
        all_rows.extend(rows)
        viz_payloads_by_budget[data.name] = payloads

        for p in payloads:
            np.savez_compressed(
                outdir / f"viz_payload_{data.name}_c{p['budget_multiplier']:.2f}.npz",
                dense_hidden=p["dense_hidden"],
                sparse_hidden=p["sparse_hidden"],
                labels=p["labels"],
                test_mask=p["test_mask"],
                eps_emp=np.array([p["eps_emp"]], dtype=float),
                budget_multiplier=np.array([p["budget_multiplier"]], dtype=float),
            )

    write_csv(outdir / "datasets.csv", dataset_rows)
    write_csv(outdir / "empirical_rows.csv", all_rows)

    final_metrics = [{
        "dataset": r["dataset"],
        "rep": r["rep"],
        "budget_multiplier": r["budget_multiplier"],
        "eps_emp": r["eps_emp"],
        "hidden_gram_error": r["hidden_gram_error"],
        "nn_overlap_20": r["nn_overlap_20"],
    } for r in all_rows]
    write_csv(outdir / "final_metrics_fashion_cora_paul15.csv", final_metrics)

    df = pd.DataFrame(final_metrics)
    plot_hidden_gram_to_nn20(df, outdir)
    plot_twopanel_vertical(df, outdir)
    plot_knn_heatmap(viz_payloads_by_budget, outdir, max_points=max_plot_points, k=20, n_anchors=20)
    plot_centroids(viz_payloads_by_budget, outdir, pick="middle", max_points=min(900, max_plot_points))
    plot_procrustes(viz_payloads_by_budget, outdir, pick="middle", max_points=max_plot_points)

    summary = {
        "seed": args.seed,
        "quick": args.quick,
        "c_list": c_list,
        "reps": reps,
        "teacher_epochs": teacher_epochs,
        "teacher_lr": teacher_lr,
        "approx_rank": approx_rank,
        "hidden_eval_subsample": hidden_eval_subsample,
        "fashion_n_per_class": fashion_n_per_class,
        "max_plot_points": max_plot_points,
        "datasets_present": sorted(list(set(df["dataset"]))),
    }
    save_json(outdir / "summary_metrics.json", summary)
    print_check("Done")


if __name__ == "__main__":
    main()
