"""
基于交通流数据，用 MVGAE 学习传感器功能邻接矩阵的工具函数。
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch_geometric.utils import negative_sampling

from model.PYG_MVGAE_model import build_pyg_mvgae


@dataclass
class MVGAEAdjConfig:
    epochs: int = 300
    lr: float = 1e-3
    latent: int = 32
    num_heads: int = 4
    hidden: Optional[int] = None
    gcn_layers: int = 2
    patience: int = 20
    corr_threshold: float = 0.6
    corr_top_k: int = 15
    adj_threshold: float = 0.3
    adj_top_k: int = 20
    train_ratio: float = 0.6
    seed: int = 42


def mvgae_config_from_parser(config) -> MVGAEAdjConfig:
    """从 ConfigParser 的 [MVGAE] 节读取参数（缺省用 MVGAEAdjConfig 默认值）。"""
    defaults = MVGAEAdjConfig()
    if not config.has_section('MVGAE'):
        return defaults

    section = config['MVGAE']

    def _opt(name: str, cast, default):
        if config.has_option('MVGAE', name):
            return cast(section[name])
        return default

    hidden = _opt('hidden', int, defaults.hidden) if config.has_option('MVGAE', 'hidden') else None
    return MVGAEAdjConfig(
        epochs=_opt('epochs', int, defaults.epochs),
        lr=_opt('lr', float, defaults.lr),
        latent=_opt('latent', int, defaults.latent),
        num_heads=_opt('num_heads', int, defaults.num_heads),
        hidden=hidden,
        gcn_layers=_opt('gcn_layers', int, defaults.gcn_layers),
        patience=_opt('patience', int, defaults.patience),
        corr_threshold=_opt('corr_threshold', float, defaults.corr_threshold),
        corr_top_k=_opt('corr_top_k', int, defaults.corr_top_k),
        adj_threshold=_opt('adj_threshold', float, defaults.adj_threshold),
        adj_top_k=_opt('adj_top_k', int, defaults.adj_top_k),
        train_ratio=_opt('train_ratio', float, defaults.train_ratio),
        seed=_opt('seed', int, defaults.seed),
    )



def load_raw_traffic_series(
    graph_signal_matrix_filename: str,
    feature_index: int = 0,
    train_ratio: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    从原始 npz 读取交通序列，并仅使用训练段以避免信息泄露。

    Returns
    -------
    train_series: (T_train, N)
    full_series: (T, N)
    """
    data = np.load(graph_signal_matrix_filename)["data"]
    series = data[:, :, feature_index].astype(np.float32)
    split = max(1, int(series.shape[0] * train_ratio))
    return series[:split], series


def build_traffic_node_features(
    train_series: np.ndarray,
    num_stats: int = 4,
) -> np.ndarray:
    """
    由训练段交通流构造节点特征 [N, F]。
    默认包含 mean / std / max / min；若 num_stats 更大则追加分位数特征。
    """
    stats = [
        train_series.mean(axis=0),
        train_series.std(axis=0),
        train_series.max(axis=0),
        train_series.min(axis=0),
    ]
    if num_stats > 4:
        stats.extend(
            [
                np.quantile(train_series, 0.25, axis=0),
                np.quantile(train_series, 0.75, axis=0),
            ]
        )
    features = np.stack(stats[:num_stats], axis=1).astype(np.float32)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (features - mean) / std


def load_distance_adjacency(
    distance_df_filename: str,
    num_of_vertices: int,
    id_filename: Optional[str] = None,
) -> np.ndarray:
    """读取距离 CSV，得到 0/1 道路邻接矩阵。"""
    adj = np.zeros((num_of_vertices, num_of_vertices), dtype=np.float32)
    id_dict = None
    if id_filename:
        with open(id_filename, "r", encoding="utf-8") as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split("\n"))}

    with open(distance_df_filename, "r", encoding="utf-8") as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j = int(row[0]), int(row[1])
            if id_dict is not None:
                i, j = id_dict[i], id_dict[j]
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    return adj


def adjacency_to_edge_index(adj: np.ndarray) -> Tensor:
    """稠密邻接矩阵 -> PyG edge_index（无自环、双向去重为单向）。"""
    src, dst = np.where(adj > 0)
    mask = src < dst
    src, dst = src[mask], dst[mask]
    edge_index = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0)
    return torch.from_numpy(edge_index).long()


def edge_index_to_adjacency(edge_index: Tensor, num_nodes: int) -> np.ndarray:
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    adj[src, dst] = 1.0
    return adj


def build_correlation_edges(
    train_series: np.ndarray,
    threshold: float = 0.6,
    top_k: Optional[int] = None,
) -> np.ndarray:
    """
    由训练段 Pearson 相关系数构造功能邻接（正边监督）。
    在训练段上算传感器两两相关系数
    超过 corr_threshold（默认 0.6）且每节点最多保留 corr_top_k（默认 15）条
    用途：链路预测的正样本监督
    含义：「交通流模式上谁和谁更相关」
    """
    corr = np.corrcoef(train_series.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)

    num_nodes = corr.shape[0]
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    if top_k is not None and top_k > 0:
        for i in range(num_nodes):
            order = np.argsort(-corr[i])
            kept = 0
            for j in order:
                if i == j:
                    continue
                if corr[i, j] >= threshold:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
                    kept += 1
                if kept >= top_k:
                    break
    else:
        mask = corr >= threshold
        adj[mask] = 1.0

    return adj


def merge_prior_and_functional_edges(
    prior_adj: np.ndarray,
    functional_adj: np.ndarray,
) -> np.ndarray:
    """先验道路边 + 功能相关边，作为 MVGAE 链路预测正样本。"""
    merged = np.clip(prior_adj + functional_adj, 0.0, 1.0)
    np.fill_diagonal(merged, 0.0)
    return merged


def sparsify_adjacency(
    adj: np.ndarray,
    threshold: Optional[float] = None,
    top_k: Optional[int] = None,
) -> np.ndarray:
    """对 MVGAE 输出的加权邻接做稀疏化。"""
    out = adj.copy()
    np.fill_diagonal(out, 0.0)
    num_nodes = out.shape[0]

    if top_k is not None and top_k > 0:
        sparse = np.zeros_like(out)
        for i in range(num_nodes):
            row = out[i].copy()
            if threshold is not None:
                row[row < threshold] = 0.0
            order = np.argsort(-row)
            kept = 0
            for j in order:
                if row[j] <= 0:
                    break
                sparse[i, j] = row[j]
                sparse[j, i] = max(sparse[j, i], row[j])
                kept += 1
                if kept >= top_k:
                    break
        out = sparse
    elif threshold is not None:
        out[out < threshold] = 0.0

    out = np.maximum(out, out.T)
    np.fill_diagonal(out, 0.0)
    return out.astype(np.float32)


def numpy_adj_to_torch(x: np.ndarray, device: torch.device) -> Tensor:
    return torch.from_numpy(x).float().to(device)


@torch.no_grad()
def generate_adjacency_from_mvgae(
    model,
    x: Tensor,
    edge_index: Tensor,
    threshold: Optional[float] = None,
    top_k: Optional[int] = 20,
) -> np.ndarray:
    """调用 MVGAE 解码器生成并稀疏化邻接矩阵。"""
    adj = model.predict_adjacency(x, edge_index).cpu().numpy()
    return sparsify_adjacency(adj, threshold=threshold, top_k=top_k)


def train_mvgae_link_predictor(
    model,
    x: Tensor,
    prior_edge_index: Tensor,
    pos_edge_index: Tensor,
    device: torch.device,
    epochs: int = 300,
    lr: float = 1e-3,
    patience: int = 20,
    eval_every: int = 10,
    val_ratio: float = 0.1,
    neg_ratio: int = 1,
    kl_scale: str = "per_node",
    subsample_size: int = 4096,
) -> dict:
    """
    在交通传感器图上训练 MVGAE，用于后续邻接推断。
    """
    num_nodes = x.size(0)
    num_edges = pos_edge_index.size(1)
    perm = torch.randperm(num_edges, device=device)
    val_size = max(1, int(num_edges * val_ratio))
    val_pos = pos_edge_index[:, perm[:val_size]]
    train_pos = pos_edge_index[:, perm[val_size:]]

    if train_pos.size(1) == 0:
        train_pos = pos_edge_index
        val_pos = pos_edge_index[:, :val_size]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_val_auc = -1.0
    counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        z = model.encode(x, prior_edge_index)

        if train_pos.size(1) > subsample_size:
            idx = torch.randperm(train_pos.size(1), device=device)[:subsample_size]
            pos = train_pos[:, idx]
        else:
            pos = train_pos

        neg = negative_sampling(
            edge_index=pos,
            num_nodes=num_nodes,
            num_neg_samples=pos.size(1) * neg_ratio,
        )

        loss_recon = model.recon_loss(z, pos, neg)
        loss_kl = model.kl_loss()
        if kl_scale == "per_node":
            loss_kl = loss_kl / num_nodes
        loss = loss_recon + loss_kl
        loss.backward()
        optimizer.step()

        if epoch % eval_every == 0:
            val_neg = negative_sampling(
                edge_index=val_pos,
                num_nodes=num_nodes,
                num_neg_samples=val_pos.size(1) * neg_ratio,
            )
            val_auc, val_ap = model.test(
                model.encode(x, prior_edge_index),
                val_pos,
                val_neg,
            )
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return {"best_val_auc": best_val_auc}


def build_mvgae_adjacency(
    graph_signal_matrix_filename: str,
    adj_filename: str,
    num_of_vertices: int,
    device: torch.device,
    id_filename: Optional[str] = None,
    cfg: Optional[MVGAEAdjConfig] = None,
    save_path: Optional[str] = None,
) -> tuple[np.ndarray, dict]:
    """
    训练 MVGAE 并返回稀疏化后的邻接矩阵；可选保存为 .npy。

    Returns
    -------
    adj_mx, stats
    """
    cfg = cfg or MVGAEAdjConfig()
    torch.manual_seed(cfg.seed)

    train_series, _ = load_raw_traffic_series(
        graph_signal_matrix_filename,
        feature_index=0,
        train_ratio=cfg.train_ratio,
    )  # 只用前 60% 时间步，防泄露
    print(f"train_series shape: {train_series.shape}") # (10195, 307)
    node_features = build_traffic_node_features(train_series)  # 4 个统计特征：均值、标准差、最大值、最小值
    print(f"node_features shape: {node_features.shape}") # (307, 4)
    x = numpy_adj_to_torch(node_features, device)
    print(f"x shape: {x.shape}") # (307, 4)

    prior_adj = load_distance_adjacency(adj_filename, num_of_vertices, id_filename) # 先验图 prior_adj（距离道路边）
    print(f"prior_adj shape: {prior_adj.shape}")  # (307, 307)
    functional_adj = build_correlation_edges(
        train_series,
        threshold=cfg.corr_threshold,
        top_k=cfg.corr_top_k,
    ) # 功能图 functional_adj（Pearson 相关边）
    print(f"functional_adj shape: {functional_adj.shape}")  # (307, 307)
    pos_adj = merge_prior_and_functional_edges(prior_adj, functional_adj)

    prior_edge_index = adjacency_to_edge_index(prior_adj).to(device)
    pos_edge_index = adjacency_to_edge_index(pos_adj).to(device)

    hidden_dim = cfg.hidden if cfg.hidden is not None else 2 * cfg.latent
    model = build_pyg_mvgae(
        in_channels=node_features.shape[1],
        hidden_channels=hidden_dim,
        out_channels=cfg.latent,
        num_heads=cfg.num_heads,
        num_gcn_layers=cfg.gcn_layers,
    ).to(device)

    train_stats = train_mvgae_link_predictor(
        model=model,
        x=x,
        prior_edge_index=prior_edge_index,
        pos_edge_index=pos_edge_index,
        device=device,
        epochs=cfg.epochs,
        lr=cfg.lr,
        patience=cfg.patience,
    )

    learned_adj = generate_adjacency_from_mvgae(
        model,
        x,
        prior_edge_index,
        threshold=cfg.adj_threshold,
        top_k=cfg.adj_top_k,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        np.save(save_path, learned_adj)

    edge_count = int((learned_adj > 0).sum() // 2)
    stats = {
        'best_val_auc': train_stats['best_val_auc'],
        'prior_edges': int(prior_adj.sum() // 2),
        'functional_edges': int(functional_adj.sum() // 2),
        'learned_edges': edge_count,
        'save_path': save_path,
    }
    return learned_adj, stats


def resolve_adjacency_for_mstgcn(
    adj_filename: str,
    graph_signal_matrix_filename: str,
    num_of_vertices: int,
    device: torch.device,
    id_filename: Optional[str] = None,
    use_mvgae: bool = False,
    mvgae_adj_filename: Optional[str] = None,
    mvgae_cfg: Optional[MVGAEAdjConfig] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    解析 MSTGCN 使用的邻接矩阵。

    - use_mvgae=False: 距离邻接
    - use_mvgae=True: 优先加载 mvgae_adj_filename；不存在或 mvgae_retrain 时现场训练
    """
    from lib.utils import get_adjacency_matrix

    distance_adj, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)

    if not use_mvgae:
        return distance_adj, distance_adj

    cache_path = mvgae_adj_filename
    if cache_path and os.path.isfile(cache_path):
        mvgae_adj = np.load(cache_path).astype(np.float32)
        if mvgae_adj.shape != distance_adj.shape:
            raise ValueError(
                f'MVGAE adj shape {mvgae_adj.shape} != expected {distance_adj.shape}'
            )
        print('Loaded cached MVGAE adjacency:', cache_path)
        print('  distance edges (undirected):', int(np.sum(distance_adj > 0) // 2))
        print('  MVGAE edges (undirected):', int(np.sum(mvgae_adj > 0) // 2))
        return mvgae_adj, distance_adj

    print('Training MVGAE adjacency (in-process)...')
    mvgae_adj, stats = build_mvgae_adjacency(
        graph_signal_matrix_filename=graph_signal_matrix_filename,
        adj_filename=adj_filename,
        num_of_vertices=num_of_vertices,
        device=device,
        id_filename=id_filename,
        cfg=mvgae_cfg,
        save_path=cache_path,
    )
    print(
        f"MVGAE done | val_auc={stats['best_val_auc']:.4f} | "
        f"prior={stats['prior_edges']} func={stats['functional_edges']} "
        f"out={stats['learned_edges']}"
    )
    if cache_path:
        print('Saved MVGAE adjacency:', cache_path)
    return mvgae_adj, distance_adj