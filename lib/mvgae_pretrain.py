"""
MVGAE 无监督预训练：路网图 A + 节点静态属性 S → Z_init。
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

from model.PYG_MVGAE_model import build_shared_mvgae


@dataclass
class MVGAEPretrainConfig:
    epochs: int = 300
    lr: float = 1e-3
    latent: int = 32
    num_heads: int = 4
    hidden: Optional[int] = None
    gcn_layers: int = 2
    fusion_dim: Optional[int] = None
    patience: int = 20
    seed: int = 42
    kl_weight: float = 1.0
    diversity_weight: float = 0.1
    static_feature_dim: int = 4
    # True=变分（μ/logstd + KL）；False=确定性 GAE（消融：单头图自编码器）
    variational: bool = True
    # 第一阶段使用的路网边比例（1.0/0.5/0.25）；稳定性实验用，默认全图
    pretrain_data_ratio: float = 1.0


def _as_bool(value, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "y", "on")


def mvgae_pretrain_config_from_parser(config) -> MVGAEPretrainConfig:
    defaults = MVGAEPretrainConfig()
    if not config.has_section("MVGAE"):
        return defaults

    section = config["MVGAE"]

    def _opt(name: str, cast, default):
        if config.has_option("MVGAE", name):
            return cast(section[name])
        return default

    hidden = _opt("hidden", int, defaults.hidden) if config.has_option("MVGAE", "hidden") else None
    fusion_dim = (
        _opt("fusion_dim", int, defaults.fusion_dim)
        if config.has_option("MVGAE", "fusion_dim")
        else None
    )
    variational = defaults.variational
    if config.has_option("MVGAE", "variational"):
        variational = _as_bool(section["variational"], default=True)

    pretrain_data_ratio = defaults.pretrain_data_ratio
    if config.has_option("MVGAE", "pretrain_data_ratio"):
        pretrain_data_ratio = float(section["pretrain_data_ratio"])

    return MVGAEPretrainConfig(
        epochs=_opt("epochs", int, defaults.epochs),
        lr=_opt("lr", float, defaults.lr),
        latent=_opt("latent", int, defaults.latent),
        num_heads=_opt("num_heads", int, defaults.num_heads),
        hidden=hidden,
        gcn_layers=_opt("gcn_layers", int, defaults.gcn_layers),
        fusion_dim=fusion_dim,
        patience=_opt("patience", int, defaults.patience),
        seed=_opt("seed", int, defaults.seed),
        kl_weight=_opt("kl_weight", float, defaults.kl_weight),
        diversity_weight=_opt("diversity_weight", float, defaults.diversity_weight),
        static_feature_dim=_opt("static_feature_dim", int, defaults.static_feature_dim),
        variational=variational,
        pretrain_data_ratio=pretrain_data_ratio,
    )


def load_distance_adjacency(
    distance_df_filename: str,
    num_of_vertices: int,
    id_filename: Optional[str] = None,
) -> np.ndarray:
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


def build_static_node_features(
    adj: np.ndarray,
    num_features: int = 4,
) -> np.ndarray:
    """
    由路网邻接矩阵 A 构造节点静态属性 S。
    特征：度、平均邻居度、二阶邻居连通强度、局部聚类系数近似。
    """
    degree = adj.sum(axis=1)
    neighbor_degree = adj @ degree
    neighbor_degree = neighbor_degree / np.maximum(degree, 1.0)

    adj2 = adj @ adj
    np.fill_diagonal(adj2, 0.0)
    second_order = adj2.sum(axis=1) / np.maximum(degree, 1.0)

    triangles = np.diag(adj @ adj @ adj) / 2.0
    clustering = triangles / np.maximum(degree * (degree - 1.0), 1.0)

    stats = [degree, neighbor_degree, second_order, clustering]
    if num_features > 4:
        stats.extend(
            [
                np.log1p(degree),
                adj.sum(axis=0),
            ]
        )
    features = np.stack(stats[:num_features], axis=1).astype(np.float32)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (features - mean) / std


def adjacency_to_edge_index(adj: np.ndarray) -> Tensor:
    src, dst = np.where(adj > 0)
    mask = src < dst
    src, dst = src[mask], dst[mask]
    edge_index = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0)
    return torch.from_numpy(edge_index).long()


def subsample_bidirected_edges(
    edge_index: Tensor,
    ratio: float,
    seed: int,
) -> Tensor:
    """
    按比例随机保留无向边，再还原为双向 edge_index。
    ratio>=1 时原样返回；至少保留 1 条无向边。
    """
    ratio = float(ratio)
    if ratio >= 1.0 - 1e-12:
        return edge_index
    if ratio <= 0.0:
        raise ValueError("pretrain_data_ratio 必须 > 0")

    src, dst = edge_index[0], edge_index[1]
    undirected = src < dst
    u, v = src[undirected], dst[undirected]
    num_u = int(u.numel())
    if num_u == 0:
        return edge_index

    keep = max(1, int(round(num_u * ratio)))
    keep = min(keep, num_u)
    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(num_u, generator=g)[:keep]
    if u.device.type != "cpu":
        perm = perm.to(u.device)
    u_keep, v_keep = u[perm], v[perm]
    return torch.stack(
        [torch.cat([u_keep, v_keep], dim=0), torch.cat([v_keep, u_keep], dim=0)],
        dim=0,
    )


def numpy_to_torch(x: np.ndarray, device: torch.device) -> Tensor:
    return torch.from_numpy(x).float().to(device)


def train_mvgae_pretrain(
    model,
    x: Tensor,
    edge_index: Tensor,
    pos_edge_index: Tensor,
    device: torch.device,
    epochs: int = 300,
    lr: float = 1e-3,
    patience: int = 20,
    eval_every: int = 10,
    val_ratio: float = 0.1,
    neg_ratio: int = 1,
    kl_weight: float = 1.0,
    diversity_weight: float = 0.1,
    subsample_size: int = 4096,
) -> dict:
    """图结构重构 + KL + 头间多样性联合优化。"""
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
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        z = model.encode(x, edge_index)

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
        loss_kl = model.kl_loss() / num_nodes
        loss_div = model.diversity_loss()
        loss = loss_recon + kl_weight * loss_kl + diversity_weight * loss_div
        loss.backward()
        optimizer.step()

        if epoch % eval_every == 0:
            val_neg = negative_sampling(
                edge_index=val_pos,
                num_nodes=num_nodes,
                num_neg_samples=val_pos.size(1) * neg_ratio,
            )
            val_auc, val_ap = model.test(
                model.encode(x, edge_index),
                val_pos,
                val_neg,
            )
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "recon": float(loss_recon.item()),
                    "kl": float(loss_kl.item()),
                    "diversity": float(loss_div.item()),
                    "val_auc": val_auc,
                    "val_ap": val_ap,
                }
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

    return {"best_val_auc": best_val_auc, "history": history}


def pretrain_mvgae(
    adj_filename: str,
    num_of_vertices: int,
    device: torch.device,
    id_filename: Optional[str] = None,
    cfg: Optional[MVGAEPretrainConfig] = None,
    z_init_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> tuple[np.ndarray, dict]:
    """
    预训练 MVGAE 并导出 Z_init 与模型 checkpoint。

    Returns
    -------
    z_init : [N, fusion_dim]
    stats : dict
    """
    cfg = cfg or MVGAEPretrainConfig()
    torch.manual_seed(cfg.seed)

    prior_adj = load_distance_adjacency(adj_filename, num_of_vertices, id_filename)
    static_features = build_static_node_features(prior_adj, num_features=cfg.static_feature_dim)
    x = numpy_to_torch(static_features, device)

    edge_index_full = adjacency_to_edge_index(prior_adj).to(device)
    # 稳定性实验：仅用部分边做预训练（消息传递 + 重构）；导出 Z 时仍在全图上编码
    edge_index = subsample_bidirected_edges(
        edge_index_full, cfg.pretrain_data_ratio, cfg.seed
    )
    pos_edge_index = edge_index.clone()
    n_full = int((edge_index_full[0] < edge_index_full[1]).sum().item())
    n_used = int((edge_index[0] < edge_index[1]).sum().item())

    hidden_dim = cfg.hidden if cfg.hidden is not None else 2 * cfg.latent
    kl_weight = cfg.kl_weight if cfg.variational else 0.0
    diversity_weight = cfg.diversity_weight if cfg.num_heads >= 2 else 0.0
    print(
        "[MVGAE pretrain] num_heads=%d variational=%s latent=%d fusion_dim=%s "
        "pretrain_data_ratio=%.4f edges=%d/%d"
        % (
            cfg.num_heads,
            cfg.variational,
            cfg.latent,
            cfg.fusion_dim,
            cfg.pretrain_data_ratio,
            n_used,
            n_full,
        )
    )
    model = build_shared_mvgae(
        in_channels=static_features.shape[1],
        hidden_channels=hidden_dim,
        out_channels=cfg.latent,
        num_heads=cfg.num_heads,
        num_gcn_layers=cfg.gcn_layers,
        fusion_out_dim=cfg.fusion_dim,
        variational=cfg.variational,
    ).to(device)

    train_stats = train_mvgae_pretrain(
        model=model,
        x=x,
        edge_index=edge_index,
        pos_edge_index=pos_edge_index,
        device=device,
        epochs=cfg.epochs,
        lr=cfg.lr,
        patience=cfg.patience,
        kl_weight=kl_weight,
        diversity_weight=diversity_weight,
    )

    # 编码器冻结后供 Hybrid：在完整路网上导出 Z（节点数与预测阶段一致）
    z_init = model.compute_z_init(x, edge_index_full).cpu().numpy()

    if z_init_path:
        os.makedirs(os.path.dirname(z_init_path) or ".", exist_ok=True)
        np.save(z_init_path, z_init.astype(np.float32))

    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "z_init": z_init,
                "static_features": static_features,
                "config": cfg.__dict__,
            },
            checkpoint_path,
        )

    stats = {
        "best_val_auc": train_stats["best_val_auc"],
        "z_init_shape": z_init.shape,
        "prior_edges": int(prior_adj.sum() // 2),
        "pretrain_edges": n_used,
        "pretrain_data_ratio": float(cfg.pretrain_data_ratio),
        "z_init_path": z_init_path,
        "checkpoint_path": checkpoint_path,
    }
    return z_init, stats


def load_z_init(
    z_init_path: str,
    device: torch.device,
) -> Tensor:
    z = np.load(z_init_path).astype(np.float32)
    return torch.from_numpy(z).float().to(device)


def _cfg_from_checkpoint_dict(raw: dict, fallback: Optional[MVGAEPretrainConfig] = None) -> MVGAEPretrainConfig:
    """从 checkpoint['config'] 或当前配置重建预训练超参。"""
    base = fallback or MVGAEPretrainConfig()
    if not raw:
        return base

    def _get(name, cast, default):
        if name not in raw or raw[name] is None:
            return default
        return cast(raw[name])

    return MVGAEPretrainConfig(
        epochs=_get("epochs", int, base.epochs),
        lr=_get("lr", float, base.lr),
        latent=_get("latent", int, base.latent),
        num_heads=_get("num_heads", int, base.num_heads),
        hidden=_get("hidden", int, base.hidden) if raw.get("hidden") is not None else base.hidden,
        gcn_layers=_get("gcn_layers", int, base.gcn_layers),
        fusion_dim=_get("fusion_dim", int, base.fusion_dim) if raw.get("fusion_dim") is not None else base.fusion_dim,
        patience=_get("patience", int, base.patience),
        seed=_get("seed", int, base.seed),
        kl_weight=_get("kl_weight", float, base.kl_weight),
        diversity_weight=_get("diversity_weight", float, base.diversity_weight),
        static_feature_dim=_get("static_feature_dim", int, base.static_feature_dim),
        variational=_as_bool(raw.get("variational"), default=base.variational),
    )


def load_mvgae_for_finetune(
    checkpoint_path: str,
    device: torch.device,
    adj_filename: str,
    num_of_vertices: int,
    id_filename: Optional[str] = None,
    cfg: Optional[MVGAEPretrainConfig] = None,
) -> tuple:
    """
    加载预训练 MVGAE 供第二阶段联合微调。

    Returns
    -------
    model, static_x [N, F], edge_index [2, E]
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "联合微调需要 MVGAE checkpoint: %s（请先跑预训练或开启 auto_pretrain）"
            % checkpoint_path
        )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt_cfg = _cfg_from_checkpoint_dict(ckpt.get("config") or {}, fallback=cfg)

    if "static_features" in ckpt and ckpt["static_features"] is not None:
        static_features = np.asarray(ckpt["static_features"], dtype=np.float32)
    else:
        prior_adj = load_distance_adjacency(adj_filename, num_of_vertices, id_filename)
        static_features = build_static_node_features(
            prior_adj, num_features=ckpt_cfg.static_feature_dim
        )

    prior_adj = load_distance_adjacency(adj_filename, num_of_vertices, id_filename)
    edge_index = adjacency_to_edge_index(prior_adj).to(device)
    x = numpy_to_torch(static_features, device)

    if x.size(0) != num_of_vertices:
        raise ValueError(
            "checkpoint 静态特征节点数 %d 与 num_of_vertices=%d 不一致"
            % (x.size(0), num_of_vertices)
        )

    hidden_dim = ckpt_cfg.hidden if ckpt_cfg.hidden is not None else 2 * ckpt_cfg.latent
    model = build_shared_mvgae(
        in_channels=static_features.shape[1],
        hidden_channels=hidden_dim,
        out_channels=ckpt_cfg.latent,
        num_heads=ckpt_cfg.num_heads,
        num_gcn_layers=ckpt_cfg.gcn_layers,
        fusion_out_dim=ckpt_cfg.fusion_dim,
        variational=ckpt_cfg.variational,
    )
    state = ckpt.get("model_state_dict") or ckpt
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    print(
        "[joint finetune] loaded MVGAE from %s | heads=%d variational=%s fusion_dim=%s"
        % (checkpoint_path, ckpt_cfg.num_heads, ckpt_cfg.variational, model.latent_dim)
    )
    return model, x, edge_index


def build_mvgae_from_scratch(
    device: torch.device,
    adj_filename: str,
    num_of_vertices: int,
    id_filename: Optional[str] = None,
    cfg: Optional[MVGAEPretrainConfig] = None,
    seed: Optional[int] = None,
) -> tuple:
    """
    端到端联合训练：随机初始化 MVGAE（不加载预训练），与 Hybrid 一起优化。

    Returns
    -------
    model, static_x [N, F], edge_index [2, E]
    """
    cfg = cfg or MVGAEPretrainConfig()
    if seed is not None:
        torch.manual_seed(int(seed))

    prior_adj = load_distance_adjacency(adj_filename, num_of_vertices, id_filename)
    static_features = build_static_node_features(prior_adj, num_features=cfg.static_feature_dim)
    x = numpy_to_torch(static_features, device)
    edge_index = adjacency_to_edge_index(prior_adj).to(device)

    hidden_dim = cfg.hidden if cfg.hidden is not None else 2 * cfg.latent
    model = build_shared_mvgae(
        in_channels=static_features.shape[1],
        hidden_channels=hidden_dim,
        out_channels=cfg.latent,
        num_heads=cfg.num_heads,
        num_gcn_layers=cfg.gcn_layers,
        fusion_out_dim=cfg.fusion_dim,
        variational=cfg.variational,
    ).to(device)
    print(
        "[end-to-end] random-init MVGAE | heads=%d variational=%s fusion_dim=%s"
        % (cfg.num_heads, cfg.variational, model.latent_dim)
    )
    return model, x, edge_index


def transfer_encode_z_init(
    source_checkpoint_path: str,
    target_adj_filename: str,
    target_num_of_vertices: int,
    device: torch.device,
    target_id_filename: Optional[str] = None,
    cfg: Optional[MVGAEPretrainConfig] = None,
    z_init_save_path: Optional[str] = None,
) -> Tensor:
    """
    空间表征迁移：加载源域 MVGAE 编码器权重，用目标域路网 (A,S) 重新编码得到 Z。

    注意：迁移的是编码器参数，不是源域 z_init 矩阵（节点数可不同）。
    """
    if not os.path.isfile(source_checkpoint_path):
        raise FileNotFoundError(
            "迁移需要源域 MVGAE checkpoint: %s\n"
            "请先在源数据集上完成预训练（例如 PEMS04 完整实验 / train_mvgae_pretrain.py）。"
            % source_checkpoint_path
        )

    ckpt = torch.load(source_checkpoint_path, map_location="cpu")
    ckpt_cfg = _cfg_from_checkpoint_dict(ckpt.get("config") or {}, fallback=cfg)

    prior_adj = load_distance_adjacency(
        target_adj_filename, target_num_of_vertices, target_id_filename
    )
    static_features = build_static_node_features(
        prior_adj, num_features=ckpt_cfg.static_feature_dim
    )
    if static_features.shape[0] != target_num_of_vertices:
        raise ValueError(
            "目标图静态特征节点数 %d 与 num_of_vertices=%d 不一致"
            % (static_features.shape[0], target_num_of_vertices)
        )

    x = numpy_to_torch(static_features, device)
    edge_index = adjacency_to_edge_index(prior_adj).to(device)

    hidden_dim = ckpt_cfg.hidden if ckpt_cfg.hidden is not None else 2 * ckpt_cfg.latent
    model = build_shared_mvgae(
        in_channels=static_features.shape[1],
        hidden_channels=hidden_dim,
        out_channels=ckpt_cfg.latent,
        num_heads=ckpt_cfg.num_heads,
        num_gcn_layers=ckpt_cfg.gcn_layers,
        fusion_out_dim=ckpt_cfg.fusion_dim,
        variational=ckpt_cfg.variational,
    )
    state = ckpt.get("model_state_dict") or ckpt
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        z_init = model.encode_z_init(x, edge_index)

    print(
        "[transfer] source=%s → target_nodes=%d | Z shape=%s | heads=%d fusion_dim=%s"
        % (
            source_checkpoint_path,
            target_num_of_vertices,
            tuple(z_init.shape),
            ckpt_cfg.num_heads,
            int(z_init.size(1)),
        )
    )

    if z_init_save_path:
        os.makedirs(os.path.dirname(z_init_save_path) or ".", exist_ok=True)
        np.save(z_init_save_path, z_init.detach().cpu().numpy().astype(np.float32))
        print("[transfer] saved Z to", z_init_save_path)

    return z_init
