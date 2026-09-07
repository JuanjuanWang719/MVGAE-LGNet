# -*- coding:utf-8 -*-
"""节点表征一致性：平均逐点余弦相似度 Sim(Z_a, Z_b)。"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor


ArrayLike = Union[np.ndarray, Tensor]


def _to_numpy(z: ArrayLike) -> np.ndarray:
    if isinstance(z, Tensor):
        return z.detach().cpu().numpy().astype(np.float64)
    return np.asarray(z, dtype=np.float64)


def mean_node_cosine_similarity(
    z_a: ArrayLike,
    z_b: ArrayLike,
    eps: float = 1e-12,
) -> float:
    """
    Sim(Z_a, Z_b) = (1/N) sum_i cos(z_a_i, z_b_i)

    用于稳定性实验：Z_100 vs Z_r（r∈{75,50,25}）。
    """
    a = _to_numpy(z_a)
    b = _to_numpy(z_b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Z 应为 [N, D]，得到 %s 与 %s" % (a.shape, b.shape))
    if a.shape != b.shape:
        raise ValueError("Z 形状不一致: %s vs %s" % (a.shape, b.shape))

    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = np.maximum(na * nb, eps)
    cos = np.sum(a * b, axis=1) / denom
    return float(np.mean(cos))


def compute_z_init_similarity_to_baseline(
    z_init: Optional[ArrayLike],
    baseline_path: Optional[str],
) -> Optional[dict]:
    """
    若提供 baseline npy 路径且 z_init 非空，计算与基线的平均余弦相似度。
    基线文件不存在时返回 None（并打印提示），不中断训练。
    """
    if z_init is None or not baseline_path:
        return None
    baseline_path = str(baseline_path).strip()
    if not baseline_path:
        return None

    import os

    if not os.path.isfile(baseline_path):
        print(
            "[z-sim] baseline 不存在，跳过 Sim(Z_100, Z_r): %s\n"
            "  请先跑 r100（或至少生成该 npy），再跑 r75/r50/r25。"
            % baseline_path
        )
        return None

    z_base = np.load(baseline_path).astype(np.float64)
    sim = mean_node_cosine_similarity(z_base, z_init)
    info = {
        "z_init_sim_to_r100": sim,
        "z_init_similarity_baseline": os.path.abspath(baseline_path),
        "z_init_similarity_formula": "mean_node_cosine",
    }
    print(
        "[z-sim] Sim(Z_100, Z_r) = %.6f | baseline=%s | Z shape=%s"
        % (sim, baseline_path, tuple(_to_numpy(z_init).shape))
    )
    return info
