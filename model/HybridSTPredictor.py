# -*- coding:utf-8 -*-
"""
局部-全局混合时空预测模型：
- 可选节点嵌入 Z（MVGAE 预训练 / 随机初始化；消融可完全关闭）；
- 局部分支：Graph WaveNet（与 Graph-WaveNet 官方实现一致）；
- 全局分支：Temporal Transformer 建模长程时间依赖；
- 门控融合机制合并两路表示并输出未来交通流。
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.gwnet_utils import build_supports
from model.gwnet import gwnet


class LearnableNodeEmbedding(nn.Module):
    """可训练节点嵌入矩阵 Z；完整实验由 MVGAE 的 Z_init 初始化，消融可为随机初始化。"""

    def __init__(self, z_init: torch.Tensor):
        super().__init__()
        if z_init.dim() != 2:
            raise ValueError("z_init 应为 [N, D]")
        self.embedding = nn.Parameter(z_init.clone())

    @property
    def num_nodes(self) -> int:
        return self.embedding.size(0)

    @property
    def embed_dim(self) -> int:
        return self.embedding.size(1)

    def forward(self) -> torch.Tensor:
        return self.embedding


class GraphWaveNetBranch(nn.Module):
    """Graph WaveNet 局部分支（封装官方 gwnet）。"""

    def __init__(
        self,
        device: torch.device,
        num_nodes: int,
        in_channels: int,
        embed_dim: int,
        adj_mx,
        seq_len: int,
        out_channels: int = 64,
        nhid: int = 32,
        dropout: float = 0.3,
        blocks: int = 4,
        layers: int = 2,
        gcn_bool: bool = True,
        addaptadj: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.in_dim = in_channels + embed_dim
        supports = build_supports(adj_mx, device, adjtype="doubletransition")
        aptinit = supports[0]

        self.gwnet = gwnet(
            device=device,
            num_nodes=num_nodes,
            dropout=dropout,
            supports=supports,
            gcn_bool=gcn_bool,
            addaptadj=addaptadj,
            aptinit=aptinit,
            in_dim=self.in_dim,
            out_dim=out_channels,
            residual_channels=nhid,
            dilation_channels=nhid,
            skip_channels=nhid * 8,
            end_channels=nhid * 16,
            blocks=blocks,
            layers=layers,
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, node_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x : (B, N, F, T)
        node_embed : (N, D) 或 None（仅交通序列）
        return : (B, N, C_out, T)
        """
        b, n, _, t = x.shape
        if self.embed_dim > 0:
            if node_embed is None:
                raise ValueError("embed_dim>0 时需要 node_embed")
            z = node_embed.unsqueeze(0).unsqueeze(-1).expand(b, n, -1, t)
            x_cat = torch.cat([x, z], dim=2)
        else:
            x_cat = x
        x_in = x_cat.permute(0, 2, 1, 3)
        x_in = F.pad(x_in, (1, 0, 0, 0))
        out = self.gwnet(x_in)
        if out.size(3) != self.seq_len:
            out = F.interpolate(
                out,
                size=(out.size(2), self.seq_len),
                mode="bilinear",
                align_corners=False,
            )
        return out.permute(0, 2, 1, 3)


class TemporalTransformerBranch(nn.Module):
    """Temporal Transformer 全局分支（沿时间维自注意力）。"""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_proj = nn.Linear(in_channels + embed_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, d_model)
        self.out_channels = d_model

    def forward(self, x: torch.Tensor, node_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x : (B, N, F, T)
        return : (B, N, C_out, T)
        """
        b, n, f, t = x.shape
        x_seq = x.permute(0, 1, 3, 2)
        if self.embed_dim > 0:
            if node_embed is None:
                raise ValueError("embed_dim>0 时需要 node_embed")
            z = node_embed.unsqueeze(0).unsqueeze(2).expand(b, -1, t, -1)
            h = torch.cat([x_seq, z], dim=-1)
        else:
            h = x_seq
        h = self.input_proj(h)
        h = h.reshape(b * n, t, -1)
        h = self.transformer(h)
        h = self.out_proj(h)
        h = h.reshape(b, n, t, -1).permute(0, 1, 3, 2)
        return h


class GatedFusion(nn.Module):
    """门控融合局部与全局时空表示（自适应）。"""

    def __init__(self, local_dim: int, global_dim: int, out_dim: int):
        super().__init__()
        self.local_proj = nn.Conv2d(local_dim, out_dim, kernel_size=(1, 1))
        self.global_proj = nn.Conv2d(global_dim, out_dim, kernel_size=(1, 1))
        self.gate = nn.Conv2d(local_dim + global_dim, out_dim, kernel_size=(1, 1))

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        """
        local_feat / global_feat : (B, N, C, T)
        return : (B, N, out_dim, T)
        """
        local_c = local_feat.permute(0, 2, 1, 3)
        global_c = global_feat.permute(0, 2, 1, 3)
        local_p = self.local_proj(local_c)
        global_p = self.global_proj(global_c)
        gate = torch.sigmoid(self.gate(torch.cat([local_c, global_c], dim=1)))
        fused = gate * local_p + (1.0 - gate) * global_p
        return fused.permute(0, 2, 1, 3)


class SumFusion(nn.Module):
    """简单加和融合：两路投影到同维后相加（消融：无门控）。"""

    def __init__(self, local_dim: int, global_dim: int, out_dim: int):
        super().__init__()
        self.local_proj = nn.Conv2d(local_dim, out_dim, kernel_size=(1, 1))
        self.global_proj = nn.Conv2d(global_dim, out_dim, kernel_size=(1, 1))

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        local_c = local_feat.permute(0, 2, 1, 3)
        global_c = global_feat.permute(0, 2, 1, 3)
        fused = self.local_proj(local_c) + self.global_proj(global_c)
        return fused.permute(0, 2, 1, 3)


def normalize_fusion_mode(fusion_mode: str) -> str:
    mode = (fusion_mode or "gated").strip().lower()
    aliases = {
        "gated": "gated",
        "gate": "gated",
        "adaptive": "gated",
        "sum": "sum",
        "add": "sum",
        "addition": "sum",
    }
    if mode not in aliases:
        raise ValueError("未知 fusion_mode=%r，可选: gated / sum" % (fusion_mode,))
    return aliases[mode]


def build_branch_fusion(fusion_mode: str, local_dim: int, global_dim: int, out_dim: int) -> nn.Module:
    mode = normalize_fusion_mode(fusion_mode)
    if mode == "sum":
        return SumFusion(local_dim, global_dim, out_dim)
    return GatedFusion(local_dim, global_dim, out_dim)


class SingleBranchProj(nn.Module):
    """单分支消融：将唯一分支特征投影到 fusion_dim。"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat : (B, N, C, T) -> (B, N, out_dim, T)"""
        return self.proj(feat.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)


class PeriodFusion(nn.Module):
    """ASTGCN 式多周期融合：week / day / hour 各路表示合并为单路。"""

    def __init__(self, fusion_dim: int, num_periods: int):
        super().__init__()
        self.num_periods = num_periods
        self.proj = nn.Conv2d(fusion_dim * num_periods, fusion_dim, kernel_size=(1, 1))
        self.gate = nn.Conv2d(fusion_dim * num_periods, num_periods, kernel_size=(1, 1))

    def forward(self, period_feats: List[torch.Tensor]) -> torch.Tensor:
        """
        period_feats : list of (B, N, fusion_dim, T)，长度 = num_periods
        return : (B, N, fusion_dim, T)
        """
        stacked = torch.cat([feat.permute(0, 2, 1, 3) for feat in period_feats], dim=1)
        gates = torch.softmax(self.gate(stacked), dim=1)
        weighted = 0.0
        for i, feat in enumerate(period_feats):
            weighted = weighted + gates[:, i : i + 1] * feat.permute(0, 2, 1, 3)
        mixed = self.proj(stacked) + weighted
        return mixed.permute(0, 2, 1, 3)


def count_input_periods(num_of_weeks: int, num_of_days: int, num_of_hours: int) -> int:
    return int(num_of_weeks > 0) + int(num_of_days > 0) + int(num_of_hours > 0)


def split_period_inputs(
    x: torch.Tensor,
    num_of_weeks: int,
    num_of_days: int,
    num_of_hours: int,
    period_len: int,
) -> List[torch.Tensor]:
    """按 prepareData 拼接顺序 [week, day, hour] 切分输入。"""
    periods: List[torch.Tensor] = []
    start = 0
    if num_of_weeks > 0:
        periods.append(x[..., start : start + period_len])
        start += period_len
    if num_of_days > 0:
        periods.append(x[..., start : start + period_len])
        start += period_len
    if num_of_hours > 0:
        periods.append(x[..., start : start + period_len])
    if not periods:
        raise ValueError("至少需要一个时间周期输入")
    return periods


class HybridSTPredictor(nn.Module):
    """
    混合时空预测器。
    输入 (B, N, F_in, T_in)，输出 (B, N, T_out)。
    多周期时可选 ASTGCN 式分路：week/day/hour 各用 12 步独立编码再融合。
    use_node_embed=False 时仅使用历史交通序列（无节点嵌入拼接）。
    use_local_branch / use_global_branch 可单独关闭（消融单分支）。
    传入 mvgae 时：每个 forward 可微计算 Z，联合微调 MVGAE（消融）。
    """

    def __init__(
        self,
        device: torch.device,
        adj_mx,
        in_channels: int,
        seq_len: int,
        num_for_predict: int,
        z_init: Optional[torch.Tensor] = None,
        use_node_embed: bool = True,
        num_nodes: Optional[int] = None,
        local_hidden: int = 64,
        global_d_model: int = 64,
        global_nhead: int = 4,
        global_layers: int = 2,
        gwnet_blocks: int = 4,
        gwnet_layers: int = 2,
        gwnet_nhid: int = 32,
        gcn_bool: bool = True,
        addaptadj: bool = True,
        fusion_dim: int = 64,
        dropout: float = 0.1,
        gwnet_dropout: float = 0.3,
        num_of_weeks: int = 0,
        num_of_days: int = 0,
        num_of_hours: int = 1,
        period_split: bool = True,
        fusion_mode: str = "gated",
        use_local_branch: bool = True,
        use_global_branch: bool = True,
        mvgae: Optional[nn.Module] = None,
        mvgae_x: Optional[torch.Tensor] = None,
        mvgae_edge_index: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.use_node_embed = bool(use_node_embed)
        self.use_local_branch = bool(use_local_branch)
        self.use_global_branch = bool(use_global_branch)
        if not self.use_local_branch and not self.use_global_branch:
            raise ValueError("use_local_branch 与 use_global_branch 不能同时为 False")
        self.fusion_mode = normalize_fusion_mode(fusion_mode)

        self.mvgae = None
        self.node_embed = None
        if self.use_node_embed and mvgae is not None:
            if mvgae_x is None or mvgae_edge_index is None:
                raise ValueError("联合微调 MVGAE 时需要提供 mvgae_x 与 mvgae_edge_index")
            self.mvgae = mvgae
            self.register_buffer("mvgae_x", mvgae_x.detach().clone().float())
            self.register_buffer("mvgae_edge_index", mvgae_edge_index.detach().clone().long())
            embed_dim = int(mvgae.latent_dim)
            n_nodes = int(mvgae_x.size(0))
        elif self.use_node_embed:
            if z_init is None:
                raise ValueError("use_node_embed=True 时必须提供 z_init 或 mvgae")
            self.node_embed = LearnableNodeEmbedding(z_init)
            embed_dim = self.node_embed.embed_dim
            n_nodes = z_init.size(0)
        else:
            embed_dim = 0
            if num_nodes is not None:
                n_nodes = int(num_nodes)
            else:
                n_nodes = int(adj_mx.shape[0])

        self.num_of_weeks = num_of_weeks
        self.num_of_days = num_of_days
        self.num_of_hours = num_of_hours
        self.num_periods = count_input_periods(num_of_weeks, num_of_days, num_of_hours)
        self.period_split = period_split and self.num_periods > 1
        self.period_len = num_for_predict
        branch_seq_len = self.period_len if self.period_split else seq_len

        if self.period_split and seq_len != self.num_periods * self.period_len:
            raise ValueError(
                f"分路模式要求 seq_len={seq_len} 等于 "
                f"num_periods({self.num_periods}) * period_len({self.period_len})"
            )

        self.local_branch = None
        if self.use_local_branch:
            self.local_branch = GraphWaveNetBranch(
                device=device,
                num_nodes=n_nodes,
                in_channels=in_channels,
                embed_dim=embed_dim,
                adj_mx=adj_mx,
                seq_len=branch_seq_len,
                out_channels=local_hidden,
                nhid=gwnet_nhid,
                dropout=gwnet_dropout,
                blocks=gwnet_blocks,
                layers=gwnet_layers,
                gcn_bool=gcn_bool,
                addaptadj=addaptadj,
            )

        self.global_branch = None
        if self.use_global_branch:
            self.global_branch = TemporalTransformerBranch(
                in_channels=in_channels,
                embed_dim=embed_dim,
                d_model=global_d_model,
                nhead=global_nhead,
                num_layers=global_layers,
                seq_len=branch_seq_len,
                dropout=dropout,
            )

        if self.use_local_branch and self.use_global_branch:
            self.fusion = build_branch_fusion(
                self.fusion_mode,
                local_dim=self.local_branch.out_channels,
                global_dim=self.global_branch.out_channels,
                out_dim=fusion_dim,
            )
            self.single_proj = None
        elif self.use_local_branch:
            self.fusion = None
            self.single_proj = SingleBranchProj(self.local_branch.out_channels, fusion_dim)
        else:
            self.fusion = None
            self.single_proj = SingleBranchProj(self.global_branch.out_channels, fusion_dim)

        self.period_fusion = (
            PeriodFusion(fusion_dim, self.num_periods) if self.period_split else None
        )
        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(fusion_dim, num_for_predict, kernel_size=(1, branch_seq_len)),
        )
        self.num_for_predict = num_for_predict
        self.seq_len = seq_len

    def _get_node_embed(self) -> Optional[torch.Tensor]:
        if self.mvgae is not None:
            return self.mvgae.encode_z_init(self.mvgae_x, self.mvgae_edge_index)
        if self.node_embed is not None:
            return self.node_embed()
        return None

    def _encode_period(self, x: torch.Tensor, z: Optional[torch.Tensor]) -> torch.Tensor:
        if self.use_local_branch and self.use_global_branch:
            local_feat = self.local_branch(x, z)
            global_feat = self.global_branch(x, z)
            return self.fusion(local_feat, global_feat)
        if self.use_local_branch:
            return self.single_proj(self.local_branch(x, z))
        return self.single_proj(self.global_branch(x, z))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self._get_node_embed()
        if self.period_split:
            period_inputs = split_period_inputs(
                x,
                self.num_of_weeks,
                self.num_of_days,
                self.num_of_hours,
                self.period_len,
            )
            period_feats = [self._encode_period(x_p, z) for x_p in period_inputs]
            fused = self.period_fusion(period_feats)
        else:
            fused = self._encode_period(x, z)
        out = self.output_layer(fused.permute(0, 2, 1, 3))
        return out.squeeze(-1).permute(0, 2, 1)


def make_hybrid_model(
    device: torch.device,
    adj_mx,
    in_channels: int,
    seq_len: int,
    num_for_predict: int,
    z_init: Optional[torch.Tensor] = None,
    use_node_embed: bool = True,
    num_nodes: Optional[int] = None,
    local_hidden: int = 64,
    global_d_model: int = 64,
    global_nhead: int = 4,
    global_layers: int = 2,
    gwnet_blocks: int = 4,
    gwnet_layers: int = 2,
    gwnet_nhid: int = 32,
    gcn_bool: bool = True,
    addaptadj: bool = True,
    fusion_dim: int = 64,
    dropout: float = 0.1,
    gwnet_dropout: float = 0.3,
    local_layers: Optional[int] = None,
    num_of_weeks: int = 0,
    num_of_days: int = 0,
    num_of_hours: int = 1,
    period_split: bool = True,
    fusion_mode: str = "gated",
    use_local_branch: bool = True,
    use_global_branch: bool = True,
    mvgae: Optional[nn.Module] = None,
    mvgae_x: Optional[torch.Tensor] = None,
    mvgae_edge_index: Optional[torch.Tensor] = None,
    preserve_mvgae_weights: bool = True,
) -> HybridSTPredictor:
    """local_layers 已弃用，保留兼容：若提供则映射为 gwnet_blocks=local_layers, gwnet_layers=2。
    preserve_mvgae_weights=True：跳过对 MVGAE 的 Xavier（联合微调预训练权重）；
    False：端到端随机初始化时对 MVGAE 也做 Xavier。
    """
    if local_layers is not None:
        gwnet_blocks = local_layers
        gwnet_layers = 2

    if z_init is not None and z_init.device != device:
        z_init = z_init.to(device)

    model = HybridSTPredictor(
        device=device,
        adj_mx=adj_mx,
        z_init=z_init,
        use_node_embed=use_node_embed,
        num_nodes=num_nodes,
        in_channels=in_channels,
        seq_len=seq_len,
        num_for_predict=num_for_predict,
        local_hidden=local_hidden,
        global_d_model=global_d_model,
        global_nhead=global_nhead,
        global_layers=global_layers,
        gwnet_blocks=gwnet_blocks,
        gwnet_layers=gwnet_layers,
        gwnet_nhid=gwnet_nhid,
        gcn_bool=gcn_bool,
        addaptadj=addaptadj,
        fusion_dim=fusion_dim,
        dropout=dropout,
        gwnet_dropout=gwnet_dropout,
        num_of_weeks=num_of_weeks,
        num_of_days=num_of_days,
        num_of_hours=num_of_hours,
        period_split=period_split,
        fusion_mode=fusion_mode,
        use_local_branch=use_local_branch,
        use_global_branch=use_global_branch,
        mvgae=mvgae,
        mvgae_x=mvgae_x,
        mvgae_edge_index=mvgae_edge_index,
    ).to(device)

    skip_ids = set()
    if model.node_embed is not None:
        skip_ids.add(id(model.node_embed.embedding))
    if model.mvgae is not None and preserve_mvgae_weights:
        for p in model.mvgae.parameters():
            skip_ids.add(id(p))
    for p in model.parameters():
        if p.dim() > 1 and id(p) not in skip_ids:
            nn.init.xavier_uniform_(p)
    return model
