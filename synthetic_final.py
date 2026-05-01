#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.spatial.distance import cdist


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    outdir: str = "synthetic_paper_ready_fig1_fig2"

    # graph sizes
    n_per_sbm: int = 80
    n_per_geo: int = 80

    # SBM params
    p_in: float = 0.20
    p_out: float = 0.05
    weight_low: float = 0.8
    weight_high: float = 1.2

    # geometric params
    geo_dim: int = 20
    geo_k: int = 30

    # class-structured features
    feature_dim_sbm: int = 32
    feature_dim_geo: int = 20
    feature_noise_std: float = 0.5

    # Laplacian polynomial graph filter used in both Fig. 1 and Fig. 2
    # Kept theorem-aligned but numerically milder than I + L + 1/2 L^2.
    poly_coeffs: Tuple[float, ...] = (1.0, -0.60, 0.15)

    # Fig. 1 dimensions
    fig1_hidden_dim_sbm: int = 48
    fig1_hidden_dim_geo: int = 32
    fig1_out_dim_sbm: int = 24
    fig1_out_dim_geo: int = 16

    # Fig. 2 training params
    fig2_epochs: int = 100
    one_layer_lr: float = 0.01
    two_layer_lr: float = 0.003
    weight_decay: float = 1e-3
    grad_clip: float = 5.0
    one_layer_out_dim: int = 16
    two_layer_hidden_dim: int = 32
    two_layer_out_dim: int = 16

    # Moderate distortion regime for paper-ready plots
    target_eps_list_fig1: Tuple[float, ...] = (0.18, 0.26, 0.36, 0.48, 0.60, 0.72)
    target_eps_list_fig2: Tuple[float, ...] = (0.18, 0.26, 0.36, 0.48, 0.60, 0.72)

    # Search grid q = floor(c n log n)
    c_grid_min: float = 0.8
    c_grid_max: float = 8.0
    c_grid_num: int = 24

    repeats_per_target: int = 5
    master_seed: int = 0


# ============================================================
# Small utilities
# ============================================================


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def print_check(msg: str):
    print(f"[check] {msg}", flush=True)


def rel_fro_error(A: np.ndarray, B: np.ndarray) -> float:
    return np.linalg.norm(A - B, ord="fro") / (np.linalg.norm(A, ord="fro") + 1e-12)


def clip_grad(G: np.ndarray, max_norm: float) -> np.ndarray:
    nrm = np.linalg.norm(G, ord="fro")
    if nrm > max_norm:
        G = G * (max_norm / (nrm + 1e-12))
    return G


# ============================================================
# Graph operators
# ============================================================


def combinatorial_laplacian(A: np.ndarray) -> np.ndarray:
    d = A.sum(axis=1)
    return np.diag(d) - A


def scaled_combinatorial_laplacian_operator(
    A: np.ndarray, scale: float | None = None, eps: float = 1e-12
) -> Tuple[np.ndarray, float]:
    """Return L / scale, with a common dense-graph scale shared by dense/sparse operators."""
    L = combinatorial_laplacian(A)
    if scale is None:
        scale = float(np.linalg.norm(L, ord=2))
    scale = max(float(scale), eps)
    return L / scale, scale


def polynomial_filter(M: np.ndarray, coeffs: Tuple[float, ...]) -> np.ndarray:
    n = M.shape[0]
    out = coeffs[0] * np.eye(n)
    power = np.eye(n)
    for a in coeffs[1:]:
        power = power @ M
        out += a * power
    return out


def fit_line(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 100)
    yy = slope * xx + intercept
    return slope, intercept, xx, yy


# ============================================================
# Synthetic graph/data generators
# ============================================================


def make_weighted_sbm(
    n_per: int,
    p_in: float,
    p_out: float,
    weight_low: float,
    weight_high: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(4), n_per)
    n = len(labels)
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if labels[i] == labels[j] else p_out
            if rng.random() < p:
                w = rng.uniform(weight_low, weight_high)
                A[i, j] = A[j, i] = w
    return A, labels


def make_geometric_graph(
    n_per: int,
    d: int,
    k: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    centers = np.zeros((4, d))
    block = max(1, d // 4)
    for c in range(4):
        start = c * block
        end = min(d, start + block)
        centers[c, start:end] = 2.5

    Xraw = []
    labels = []
    for c in range(4):
        pts = centers[c] + 0.9 * rng.standard_normal((n_per, d))
        Xraw.append(pts)
        labels.extend([c] * n_per)

    Xraw = np.vstack(Xraw)
    labels = np.array(labels)
    D = cdist(Xraw, Xraw)
    kth = np.sort(D, axis=1)[:, k + 1]
    sigma = np.median(kth) + 1e-12

    n = Xraw.shape[0]
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        nbrs = np.argsort(D[i])[1:k + 1]
        for j in nbrs:
            w = np.exp(-(D[i, j] ** 2) / (2 * sigma ** 2))
            A[i, j] = max(A[i, j], w)
            A[j, i] = max(A[j, i], w)
    return A, labels


def make_class_structured_features(
    labels: np.ndarray,
    d: int,
    noise_std: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = noise_std * rng.standard_normal((len(labels), d))
    block = max(1, d // 4)
    for i, c in enumerate(labels):
        start = c * block
        end = min(d, start + block)
        X[i, start:end] += 2.0
    return X


# ============================================================
# Effective-resistance sparsification (precompute once per graph)
# ============================================================


def precompute_er_sampler(A: np.ndarray, family_name: str = ""):
    n = A.shape[0]
    edges = [(i, j, A[i, j]) for i in range(n) for j in range(i + 1, n) if A[i, j] > 0]
    if len(edges) == 0:
        raise ValueError("Graph has no edges.")

    print_check(f"{family_name}: precomputing exact ER probabilities for {len(edges)} edges")
    L = combinatorial_laplacian(A)
    L_pinv = np.linalg.pinv(L)

    scores = []
    for i, j, w in edges:
        R_ij = L_pinv[i, i] + L_pinv[j, j] - 2.0 * L_pinv[i, j]
        scores.append(max(w * R_ij, 0.0))

    scores = np.asarray(scores, dtype=float)
    if scores.sum() <= 0:
        probs = np.ones_like(scores) / len(scores)
    else:
        probs = scores / scores.sum()

    print_check(
        f"{family_name}: ER probability stats min={probs.min():.3e}, median={np.median(probs):.3e}, max={probs.max():.3e}"
    )
    return edges, probs


def sample_er_sparsifier_from_precomputed(
    A_shape: Tuple[int, int],
    edges,
    probs: np.ndarray,
    q: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(edges), size=q, replace=True, p=probs)
    counts = np.bincount(idxs, minlength=len(edges))
    As = np.zeros(A_shape, dtype=float)

    for idx, count in enumerate(counts):
        if count == 0:
            continue
        i, j, w = edges[idx]
        p = probs[idx]
        new_w = count * w / (q * p)
        As[i, j] += new_w
        As[j, i] += new_w
    return As


# ============================================================
# Empirical spectral distortion
# ============================================================


def empirical_epsilon_exact(L: np.ndarray, Ls: np.ndarray, eps: float = 1e-10):
    n = L.shape[0]
    Q = np.eye(n)[:, 1:]  # drop the constant vector direction after eigensolve-like restriction
    # Better: use basis orthogonal to ones
    one = np.ones((n, 1)) / np.sqrt(n)
    B = np.eye(n) - one @ one.T
    vals, vecs = eigh(B)
    Q = vecs[:, vals > 1e-8]
    A = Q.T @ Ls @ Q
    Bm = Q.T @ L @ Q
    lam = eigh(A, Bm, eigvals_only=True)
    lam_min = float(np.min(lam))
    lam_max = float(np.max(lam))
    eps_emp = max(1.0 - lam_min, lam_max - 1.0)
    return float(eps_emp), lam_min, lam_max


# ============================================================
# Fig. 1 evaluation
# ============================================================


def evaluate_fig1_metrics(
    A: np.ndarray,
    X: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    coeffs: Tuple[float, ...],
    c: float,
    seed: int,
    er_sampler,
) -> Dict[str, float]:
    n = A.shape[0]
    q = max(1, int(np.floor(c * n * np.log(n))))
    As = sample_er_sparsifier_from_precomputed(A.shape, er_sampler[0], er_sampler[1], q=q, seed=seed)

    L = combinatorial_laplacian(A)
    Ls = combinatorial_laplacian(As)
    eps_emp, lam_min, lam_max = empirical_epsilon_exact(L, Ls)

    Lbar, scale = scaled_combinatorial_laplacian_operator(A)
    Lsbar, _ = scaled_combinatorial_laplacian_operator(As, scale=scale)
    P = polynomial_filter(Lbar, coeffs)
    Ps = polynomial_filter(Lsbar, coeffs)

    # Relative filter error for more interpretable plotting
    filter_rel_err = np.linalg.norm(P - Ps, ord=2) / (np.linalg.norm(P, ord=2) + 1e-12)

    H = P @ X @ W1
    Z = H @ W2
    Hs = Ps @ X @ W1
    Zs = Hs @ W2

    rep_err = rel_fro_error(Z, Zs)
    gram_err = rel_fro_error(Z @ Z.T, Zs @ Zs.T)
    edge_frac = np.count_nonzero(np.triu(As, 1)) / max(np.count_nonzero(np.triu(A, 1)), 1)

    return {
        "q": q,
        "eps_emp": eps_emp,
        "lam_min": lam_min,
        "lam_max": lam_max,
        "filter_rel_err": float(filter_rel_err),
        "rep_err": float(rep_err),
        "gram_err": float(gram_err),
        "edge_frac": float(edge_frac),
    }


# ============================================================
# Fig. 2 training models (polynomial Laplacian graph shift)
# ============================================================


def run_one_layer_training(
    S: np.ndarray,
    Ss: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    out_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    W = 0.10 * rng.standard_normal((X.shape[1], out_dim))
    Ws = W.copy()

    traj_dense = []
    traj_sparse = []
    rel_gap = []

    for t in range(epochs + 1):
        F = S @ X @ W
        Fs = Ss @ X @ Ws

        traj_dense.append(F.copy())
        traj_sparse.append(Fs.copy())
        rel_gap.append(np.linalg.norm(W - Ws, ord="fro") / (np.linalg.norm(W, ord="fro") + 1e-12))

        if t == epochs:
            break

        G = X.T @ S.T @ (F - Y) / X.shape[0] + weight_decay * W
        Gs = X.T @ Ss.T @ (Fs - Y) / X.shape[0] + weight_decay * Ws
        G = clip_grad(G, grad_clip)
        Gs = clip_grad(Gs, grad_clip)
        W -= lr * G
        Ws -= lr * Gs

    return np.array(rel_gap), np.array(traj_dense), np.array(traj_sparse)


def relu(U: np.ndarray):
    return np.maximum(U, 0.0)


def run_two_layer_training(
    S: np.ndarray,
    Ss: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    hidden_dim: int,
    out_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    W1 = 0.10 * rng.standard_normal((X.shape[1], hidden_dim))
    W2 = 0.10 * rng.standard_normal((hidden_dim, out_dim))
    W1s = W1.copy()
    W2s = W2.copy()

    rel_gap = []
    traj_dense = []
    traj_sparse = []

    for t in range(epochs + 1):
        A1 = S @ X @ W1
        H = relu(A1)
        F = S @ H @ W2

        A1s = Ss @ X @ W1s
        Hs = relu(A1s)
        Fs = Ss @ Hs @ W2s

        traj_dense.append(F.copy())
        traj_sparse.append(Fs.copy())
        num = np.linalg.norm(W1 - W1s, ord="fro") ** 2 + np.linalg.norm(W2 - W2s, ord="fro") ** 2
        den = np.linalg.norm(W1, ord="fro") ** 2 + np.linalg.norm(W2, ord="fro") ** 2 + 1e-12
        rel_gap.append(np.sqrt(num / den))

        if t == epochs:
            break

        E = (F - Y) / X.shape[0]
        G2 = H.T @ S.T @ E + weight_decay * W2
        dH = (S.T @ E) @ W2.T
        dA1 = dH * (A1 > 0)
        G1 = X.T @ S.T @ dA1 + weight_decay * W1

        Es = (Fs - Y) / X.shape[0]
        G2s = Hs.T @ Ss.T @ Es + weight_decay * W2s
        dHs = (Ss.T @ Es) @ W2s.T
        dA1s = dHs * (A1s > 0)
        G1s = X.T @ Ss.T @ dA1s + weight_decay * W1s

        G1 = clip_grad(G1, grad_clip)
        G2 = clip_grad(G2, grad_clip)
        G1s = clip_grad(G1s, grad_clip)
        G2s = clip_grad(G2s, grad_clip)

        W1 -= lr * G1
        W2 -= lr * G2
        W1s -= lr * G1s
        W2s -= lr * G2s

    return np.array(rel_gap), np.array(traj_dense), np.array(traj_sparse)


# ============================================================
# Budget targeting helpers
# ============================================================


def select_budget_for_target(A: np.ndarray, er_sampler, c_grid: np.ndarray, target_eps: float, seed_base: int):
    L = combinatorial_laplacian(A)
    best = None
    best_gap = np.inf
    n = A.shape[0]
    for j, c in enumerate(c_grid):
        q = max(1, int(np.floor(c * n * np.log(n))))
        As = sample_er_sparsifier_from_precomputed(A.shape, er_sampler[0], er_sampler[1], q=q, seed=seed_base + j)
        eps_emp, lam_min, lam_max = empirical_epsilon_exact(L, combinatorial_laplacian(As))
        gap = abs(eps_emp - target_eps)
        if gap < best_gap:
            best_gap = gap
            best = {
                "c": float(c),
                "q": q,
                "eps_emp": float(eps_emp),
                "lam_min": float(lam_min),
                "lam_max": float(lam_max),
                "gap": float(gap),
            }
    return best


# ============================================================
# Family runners
# ============================================================


def run_fig1_family(
    family_name: str,
    A: np.ndarray,
    X: np.ndarray,
    cfg: Config,
    hidden_dim: int,
    out_dim: int,
    weight_seed: int,
    search_seed: int,
    repeat_seed_base: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print_check(f"{family_name} Fig1: starting")
    er_sampler = precompute_er_sampler(A, family_name=f"{family_name} Fig1")
    c_grid = np.linspace(cfg.c_grid_min, cfg.c_grid_max, cfg.c_grid_num)

    rng = np.random.default_rng(weight_seed)
    W1 = 0.25 * rng.standard_normal((X.shape[1], hidden_dim))
    W2 = 0.25 * rng.standard_normal((hidden_dim, out_dim))

    selected = []
    for i, target in enumerate(cfg.target_eps_list_fig1):
        out = select_budget_for_target(A, er_sampler, c_grid, target, search_seed + 1000 * i)
        out["family"] = family_name
        out["target_eps"] = target
        selected.append(out)
        print_check(
            f"{family_name} Fig1 target={target:.2f}: selected c={out['c']:.3f}, q={out['q']}, eps={out['eps_emp']:.3f}"
        )

    selected_df = pd.DataFrame(selected)

    rows = []
    for _, sel in selected_df.iterrows():
        c = float(sel["c"])
        target = float(sel["target_eps"])
        for r in range(cfg.repeats_per_target):
            seed = repeat_seed_base + 10000 * int(round(100 * target)) + r
            met = evaluate_fig1_metrics(A, X, W1, W2, cfg.poly_coeffs, c, seed, er_sampler)
            rows.append({
                "family": family_name,
                "target_eps": target,
                "repeat": r,
                **met,
            })
            print_check(
                f"{family_name} Fig1 target={target:.2f} repeat={r}: eps={met['eps_emp']:.3f}, rel_filter={met['filter_rel_err']:.3f}, rep={met['rep_err']:.3f}, gram={met['gram_err']:.3f}"
            )

    return selected_df, pd.DataFrame(rows)


def run_fig2_family(
    family_name: str,
    A: np.ndarray,
    X: np.ndarray,
    cfg: Config,
    search_seed: int,
    repeat_seed_base: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print_check(f"{family_name} Fig2: starting")
    er_sampler = precompute_er_sampler(A, family_name=f"{family_name} Fig2")
    c_grid = np.linspace(cfg.c_grid_min, cfg.c_grid_max, cfg.c_grid_num)

    selected = []
    for i, target in enumerate(cfg.target_eps_list_fig2):
        out = select_budget_for_target(A, er_sampler, c_grid, target, search_seed + 1000 * i)
        out["family"] = family_name
        out["target_eps"] = target
        selected.append(out)
        print_check(
            f"{family_name} Fig2 target={target:.2f}: selected c={out['c']:.3f}, q={out['q']}, eps={out['eps_emp']:.3f}"
        )

    selected_df = pd.DataFrame(selected)
    traj_rows = []
    final_rows = []

    # fixed teacher target Y based on dense graph
    Lbar, scale = scaled_combinatorial_laplacian_operator(A)
    S = polynomial_filter(Lbar, cfg.poly_coeffs)
    rng = np.random.default_rng(cfg.master_seed + 4242)
    Y = S @ X @ (0.30 * rng.standard_normal((X.shape[1], cfg.one_layer_out_dim)))

    for _, sel in selected_df.iterrows():
        c = float(sel["c"])
        target = float(sel["target_eps"])
        for r in range(cfg.repeats_per_target):
            seed = repeat_seed_base + 10000 * int(round(100 * target)) + r
            q = max(1, int(np.floor(c * A.shape[0] * np.log(A.shape[0]))))
            As = sample_er_sparsifier_from_precomputed(A.shape, er_sampler[0], er_sampler[1], q=q, seed=seed)
            eps_emp, _, _ = empirical_epsilon_exact(combinatorial_laplacian(A), combinatorial_laplacian(As))
            Lsbar, _ = scaled_combinatorial_laplacian_operator(As, scale=scale)
            Ss = polynomial_filter(Lsbar, cfg.poly_coeffs)

            gap1, _, _ = run_one_layer_training(
                S=S, Ss=Ss, X=X, Y=Y,
                out_dim=cfg.one_layer_out_dim,
                epochs=cfg.fig2_epochs,
                lr=cfg.one_layer_lr,
                weight_decay=cfg.weight_decay,
                grad_clip=cfg.grad_clip,
                seed=seed + 100,
            )
            gap2, _, _ = run_two_layer_training(
                S=S, Ss=Ss, X=X, Y=Y,
                hidden_dim=cfg.two_layer_hidden_dim,
                out_dim=cfg.two_layer_out_dim,
                epochs=cfg.fig2_epochs,
                lr=cfg.two_layer_lr,
                weight_decay=cfg.weight_decay,
                grad_clip=cfg.grad_clip,
                seed=seed + 200,
            )

            for epoch, val in enumerate(gap1):
                traj_rows.append({
                    "family": family_name,
                    "target_eps": target,
                    "repeat": r,
                    "eps_emp": eps_emp,
                    "epoch": epoch,
                    "model": "one-layer",
                    "rel_gap": float(val),
                })
            for epoch, val in enumerate(gap2):
                traj_rows.append({
                    "family": family_name,
                    "target_eps": target,
                    "repeat": r,
                    "eps_emp": eps_emp,
                    "epoch": epoch,
                    "model": "two-layer",
                    "rel_gap": float(val),
                })

            final_rows.append({
                "family": family_name,
                "target_eps": target,
                "repeat": r,
                "q": q,
                "eps_emp": eps_emp,
                "model": "one-layer",
                "final_rel_gap": float(gap1[-1]),
            })
            final_rows.append({
                "family": family_name,
                "target_eps": target,
                "repeat": r,
                "q": q,
                "eps_emp": eps_emp,
                "model": "two-layer",
                "final_rel_gap": float(gap2[-1]),
            })
            print_check(
                f"{family_name} Fig2 target={target:.2f} repeat={r}: eps={eps_emp:.3f}, final_gap_one={gap1[-1]:.3f}, final_gap_two={gap2[-1]:.3f}"
            )

    return selected_df, pd.DataFrame(traj_rows), pd.DataFrame(final_rows)


# ============================================================
# Plotting
# ============================================================


def plot_fig1(df: pd.DataFrame, save_pdf: str, save_png: str):
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    })

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2))
    metrics = [
        ("filter_rel_err", r"$\|p(\bar L)-p(\widetilde{\bar L})\|_2 / \|p(\bar L)\|_2$", "Relative Filter Error"),
        ("rep_err", r"$\|Z-\widetilde Z\|_F / \|Z\|_F$", "Relative Representation"),
        ("gram_err", r"$\|ZZ^\top-\widetilde Z\widetilde Z^\top\|_F / \|ZZ^\top\|_F$", "Relative Gram"),
    ]

    for ax, (metric, ylabel, title) in zip(axes, metrics):
        for family in ["SBM", "Geometric"]:
            sub = df[df["family"] == family].copy()
            grp = sub.groupby("target_eps").agg(
                eps_mean=("eps_emp", "mean"),
                y_mean=(metric, "mean"),
                y_std=(metric, "std"),
            ).reset_index().sort_values("eps_mean")

            ax.errorbar(
                grp["eps_mean"],
                grp["y_mean"],
                yerr=grp["y_std"].fillna(0.0),
                fmt="o",
                capsize=3,
                label=family,
            )

            fit = fit_line(grp["eps_mean"].values, grp["y_mean"].values)
            if fit is not None:
                slope, intercept, xx, yy = fit
                ax.plot(xx, yy, "--", linewidth=1.4, label=f"{family} fit (slope={slope:.2f})")

        ax.set_xlabel(r"Empirical sparsification distortion $\epsilon_{\mathrm{emp}}$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    axes[0].legend(loc="best")
    fig.suptitle("Figure 1: controlled sparsification validation", y=1.03)
    fig.tight_layout()
    fig.savefig(save_pdf, bbox_inches="tight")
    fig.savefig(save_png, dpi=250, bbox_inches="tight")
    plt.close(fig)



def plot_fig2(df_traj: pd.DataFrame, df_final: pd.DataFrame, save_pdf: str, save_png: str):
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 8,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.3))

    for row_idx, family in enumerate(["SBM", "Geometric"]):
        fam_final = df_final[df_final["family"] == family].copy()
        ordered_targets = fam_final.groupby("target_eps")["eps_emp"].mean().sort_values().index.tolist()
        picked = [ordered_targets[0], ordered_targets[len(ordered_targets)//2], ordered_targets[-1]] if len(ordered_targets) >= 3 else ordered_targets

        # Trajectory panel: actual mean trajectories only, no dashed straight-line fits.
        ax = axes[row_idx, 0]
        fam_traj = df_traj[(df_traj["family"] == family) & (df_traj["target_eps"].isin(picked))].copy()

        for model, marker in [("one-layer", "o"), ("two-layer", "s")]:
            for t_eps in picked:
                sub = fam_traj[(fam_traj["model"] == model) & (fam_traj["target_eps"] == t_eps)]
                grp = sub.groupby("epoch").agg(
                    gap_mean=("rel_gap", "mean"),
                    gap_std=("rel_gap", "std"),
                    eps_mean=("eps_emp", "mean"),
                ).reset_index()
                eps_mean = grp["eps_mean"].iloc[0]
                ax.plot(
                    grp["epoch"],
                    grp["gap_mean"],
                    marker=marker,
                    markersize=2.5,
                    linewidth=1.6,
                    label=f"{model}, target={t_eps:.2f}, eps≈{eps_mean:.2f}",
                )

        ax.set_title(f"{family}: Relative Parameter-Gap Trajectories")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Relative Parameter Gap")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

        # Final-gap panel: points + error bars only, no dashed fitted lines.
        ax2 = axes[row_idx, 1]
        for model in ["one-layer", "two-layer"]:
            sub = fam_final[fam_final["model"] == model]
            grp = sub.groupby("target_eps").agg(
                eps_mean=("eps_emp", "mean"),
                gap_mean=("final_rel_gap", "mean"),
                gap_std=("final_rel_gap", "std"),
            ).reset_index().sort_values("eps_mean")
            ax2.errorbar(
                grp["eps_mean"],
                grp["gap_mean"],
                yerr=grp["gap_std"].fillna(0.0),
                fmt="o",
                capsize=3,
                label=model,
            )
        ax2.set_title(f"{family}: Final Relative Gap vs Sparsification")
        ax2.set_xlabel(r"Empirical sparsification distortion $\epsilon_{\mathrm{emp}}$")
        ax2.set_ylabel("Final Relative Gap")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="best")

    fig.suptitle("Figure 2: polynomial-Laplacian training-trajectory perturbation", y=1.02)
    fig.tight_layout()
    fig.savefig(save_pdf, bbox_inches="tight")
    fig.savefig(save_png, dpi=250, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================


def main():
    cfg = Config()
    ensure_dir(cfg.outdir)
    print_check("Starting paper-ready synthetic Fig. 1 / Fig. 2 generation")
    print_check(json.dumps(asdict(cfg), indent=2))

    # Build graphs and node features
    A_sbm, y_sbm = make_weighted_sbm(
        cfg.n_per_sbm, cfg.p_in, cfg.p_out, cfg.weight_low, cfg.weight_high, seed=cfg.master_seed + 1
    )
    X_sbm = make_class_structured_features(
        y_sbm, cfg.feature_dim_sbm, cfg.feature_noise_std, seed=cfg.master_seed + 2
    )
    print_check(f"SBM graph: n={A_sbm.shape[0]}, m={np.count_nonzero(np.triu(A_sbm, 1))}")

    A_geo, y_geo = make_geometric_graph(
        cfg.n_per_geo, cfg.geo_dim, cfg.geo_k, seed=cfg.master_seed + 3
    )
    X_geo = make_class_structured_features(
        y_geo, cfg.feature_dim_geo, cfg.feature_noise_std, seed=cfg.master_seed + 4
    )
    print_check(f"Geometric graph: n={A_geo.shape[0]}, m={np.count_nonzero(np.triu(A_geo, 1))}")

    # Fig. 1
    _, fig1_sbm = run_fig1_family(
        "SBM", A_sbm, X_sbm, cfg,
        hidden_dim=cfg.fig1_hidden_dim_sbm,
        out_dim=cfg.fig1_out_dim_sbm,
        weight_seed=cfg.master_seed + 10,
        search_seed=cfg.master_seed + 20,
        repeat_seed_base=cfg.master_seed + 30,
    )
    _, fig1_geo = run_fig1_family(
        "Geometric", A_geo, X_geo, cfg,
        hidden_dim=cfg.fig1_hidden_dim_geo,
        out_dim=cfg.fig1_out_dim_geo,
        weight_seed=cfg.master_seed + 40,
        search_seed=cfg.master_seed + 50,
        repeat_seed_base=cfg.master_seed + 60,
    )
    df_fig1 = pd.concat([fig1_sbm, fig1_geo], ignore_index=True)
    df_fig1.to_csv(os.path.join(cfg.outdir, "fig1_data.csv"), index=False)
    plot_fig1(
        df_fig1,
        save_pdf=os.path.join(cfg.outdir, "figure1_paper_ready.pdf"),
        save_png=os.path.join(cfg.outdir, "figure1_paper_ready.png"),
    )

    # Fig. 2
    _, traj_sbm, final_sbm = run_fig2_family(
        "SBM", A_sbm, X_sbm, cfg,
        search_seed=cfg.master_seed + 70,
        repeat_seed_base=cfg.master_seed + 80,
    )
    _, traj_geo, final_geo = run_fig2_family(
        "Geometric", A_geo, X_geo, cfg,
        search_seed=cfg.master_seed + 90,
        repeat_seed_base=cfg.master_seed + 100,
    )
    df_traj = pd.concat([traj_sbm, traj_geo], ignore_index=True)
    df_final = pd.concat([final_sbm, final_geo], ignore_index=True)
    df_traj.to_csv(os.path.join(cfg.outdir, "fig2_traj_data.csv"), index=False)
    df_final.to_csv(os.path.join(cfg.outdir, "fig2_final_data.csv"), index=False)
    plot_fig2(
        df_traj,
        df_final,
        save_pdf=os.path.join(cfg.outdir, "figure2_paper_ready.pdf"),
        save_png=os.path.join(cfg.outdir, "figure2_paper_ready.png"),
    )

    print_check(f"Done. Outputs written to: {cfg.outdir}")


if __name__ == "__main__":
    main()
