"""
多头变分图自编码器（MVGAE）：

- 共享底层 GCN 骨干在路网图 A 上传播节点静态属性 S；
- 多个变分编码头在共享表示上从不同潜在子空间建模；
- 融合机制生成节点初始嵌入 Z_init；
- 预训练损失：图结构重构 + KL 散度 + 头间多样性约束。

消融：num_heads=1 且 variational=False → 单头确定性图自编码器（GAE）。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import InnerProductDecoder

MAX_LOGSTD = 10
EPS = 1e-15


class SharedGCNBackbone(nn.Module):
    """共享 GCN 骨干：GCN(S, A) → 隐藏表示 h。"""

    def __init__(self, in_channels: int, hidden_channels: int, num_gcn_layers: int = 2):
        super().__init__()
        if num_gcn_layers < 1:
            raise ValueError("num_gcn_layers 至少为 1")

        convs: list[GCNConv] = []
        convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_gcn_layers - 1):
            convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs = nn.ModuleList(convs)
        self.hidden_channels = hidden_channels

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.relu(h)
        return h


class VariationalHead(nn.Module):
    """单头变分投影：h → (μ, logstd)。"""

    def __init__(self, hidden_channels: int, out_channels: int):
        super().__init__()
        self.mu_proj = nn.Linear(hidden_channels, out_channels)
        self.logstd_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        return self.mu_proj(h), self.logstd_proj(h)


class DeterministicHead(nn.Module):
    """单头确定性投影（经典 GAE）：h → μ；logstd 返回全零占位。"""

    def __init__(self, hidden_channels: int, out_channels: int):
        super().__init__()
        self.mu_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        mu = self.mu_proj(h)
        return mu, torch.zeros_like(mu)


class HeadFusion(nn.Module):
    """多头注意力融合：将各头 μ 融合为 Z_init。"""

    def __init__(self, num_heads: int, latent_dim: int, out_dim: Optional[int] = None):
        super().__init__()
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.out_dim = out_dim if out_dim is not None else latent_dim
        self.score = nn.Linear(latent_dim, 1)
        if self.out_dim != latent_dim:
            self.proj = nn.Linear(latent_dim, self.out_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, mu: Tensor) -> Tensor:
        """
        Parameters
        ----------
        mu : [N, num_heads, latent_dim]

        Returns
        -------
        z_init : [N, out_dim]
        """
        weights = F.softmax(self.score(mu).squeeze(-1), dim=1)
        fused = (mu * weights.unsqueeze(-1)).sum(dim=1)
        return self.proj(fused)


class SharedMultiHeadGCNEncoder(nn.Module):
    """共享 GCN 骨干 + 多个并行编码头（变分或确定性）。"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_heads: int,
        num_gcn_layers: int = 2,
        fusion_out_dim: Optional[int] = None,
        variational: bool = True,
    ):
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads 至少为 1")

        self.num_heads = num_heads
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.variational = bool(variational)
        self.backbone = SharedGCNBackbone(in_channels, hidden_channels, num_gcn_layers)
        head_cls = VariationalHead if self.variational else DeterministicHead
        self.heads = nn.ModuleList(
            [head_cls(hidden_channels, out_channels) for _ in range(num_heads)]
        )
        self.fusion = HeadFusion(num_heads, out_channels, fusion_out_dim)

    @property
    def fusion_dim(self) -> int:
        return self.fusion.out_dim

    def forward(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        h = self.backbone(x, edge_index)
        mus: list[Tensor] = []
        logstds: list[Tensor] = []
        for head in self.heads:
            mu, logstd = head(h)
            mus.append(mu)
            logstds.append(logstd)
        mu_stacked = torch.stack(mus, dim=1)
        logstd_stacked = torch.stack(logstds, dim=1)
        z_init = self.fusion(mu_stacked)
        return mu_stacked, logstd_stacked, z_init


class MVGAEEncoderHead(nn.Module):
    """兼容旧接口：独立 GCN 头（保留供邻接学习脚本使用）。"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_gcn_layers: int = 2,
    ):
        super().__init__()
        self.backbone = SharedGCNBackbone(in_channels, hidden_channels, num_gcn_layers)
        self.head = VariationalHead(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        h = self.backbone(x, edge_index)
        return self.head(h)


class MultiHeadGCNEncoder(nn.Module):
    """兼容旧接口：num_heads 路独立 GCN 头。"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_heads: int,
        num_gcn_layers: int = 2,
    ):
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads 至少为 1")
        self.num_heads = num_heads
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.variational = True
        self.heads = nn.ModuleList(
            [
                MVGAEEncoderHead(
                    in_channels,
                    hidden_channels,
                    out_channels,
                    num_gcn_layers=num_gcn_layers,
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        mus: list[Tensor] = []
        logstds: list[Tensor] = []
        for head in self.heads:
            mu, logstd = head(x, edge_index)
            mus.append(mu)
            logstds.append(logstd)
        return torch.stack(mus, dim=1), torch.stack(logstds, dim=1)


class MVGAE(nn.Module):
    """
    Multi-head VGAE / 单头 GAE。
    - shared_backbone=True：共享 GCN + 融合 Z_init（预训练推荐）；
    - shared_backbone=False：独立多头（邻接学习兼容模式）；
    - variational=False：确定性编码（无 KL / 无重参数化采样）。
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: Optional[nn.Module] = None,
        shared_backbone: bool = False,
        variational: Optional[bool] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder if decoder is not None else InnerProductDecoder()
        self.shared_backbone = shared_backbone
        self.num_heads = encoder.num_heads
        if variational is None:
            variational = bool(getattr(encoder, "variational", True))
        self.variational = bool(variational)
        self.__mu__: Tensor | None = None
        self.__logstd__: Tensor | None = None
        self.__z_init__: Tensor | None = None

    @property
    def latent_dim(self) -> int:
        if self.shared_backbone:
            return self.encoder.fusion_dim
        return self.encoder.out_channels * self.num_heads

    def reset_parameters(self) -> None:
        for m in self.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if not self.variational:
            return mu
        logstd = logstd.clamp(max=MAX_LOGSTD)
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        return mu

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if self.shared_backbone:
            self.__mu__, self.__logstd__, self.__z_init__ = self.encoder(x, edge_index)
            if self.variational:
                self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
            z_heads = self.reparametrize(self.__mu__, self.__logstd__)
            return z_heads
        self.__mu__, self.__logstd__ = self.encoder(x, edge_index)
        if self.variational:
            self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        return self.reparametrize(self.__mu__, self.__logstd__)

    @property
    def z_init(self) -> Tensor:
        if self.__z_init__ is None:
            raise RuntimeError("请先调用 encode()")
        return self.__z_init__

    def compute_z_init(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """推理阶段获取融合嵌入 Z_init（使用 μ；无梯度）。"""
        self.eval()
        with torch.no_grad():
            return self.encode_z_init(x, edge_index)

    def encode_z_init(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """可微融合嵌入 Z（与 compute_z_init 同公式，供第二阶段联合微调）。"""
        if self.shared_backbone:
            mu, logstd, z_init = self.encoder(x, edge_index)
            self.__mu__ = mu
            self.__logstd__ = logstd
            self.__z_init__ = z_init
            return z_init
        mu, logstd = self.encoder(x, edge_index)
        self.__mu__ = mu
        self.__logstd__ = logstd
        return mu.reshape(mu.size(0), -1)

    def kl_loss(
        self,
        mu: Optional[Tensor] = None,
        logstd: Optional[Tensor] = None,
    ) -> Tensor:
        if not self.variational:
            mu = self.__mu__ if mu is None else mu
            assert mu is not None
            return mu.new_zeros(())
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(max=MAX_LOGSTD)
        assert mu is not None and logstd is not None
        kl = -0.5 * torch.sum(1 + 2 * logstd - mu.pow(2) - logstd.exp().pow(2), dim=-1)
        return kl.mean()

    def diversity_loss(self, mu: Optional[Tensor] = None) -> Tensor:
        """
        头间多样性约束：惩罚不同编码头 μ 之间的余弦相似度。
        mu : [N, num_heads, latent_dim]
        """
        mu = self.__mu__ if mu is None else mu
        assert mu is not None
        num_heads = mu.size(1)
        if num_heads < 2:
            return mu.new_zeros(())

        loss = mu.new_zeros(())
        count = 0
        for i in range(num_heads):
            zi = F.normalize(mu[:, i], dim=-1)
            for j in range(i + 1, num_heads):
                zj = F.normalize(mu[:, j], dim=-1)
                loss = loss + (zi * zj).sum(dim=-1).pow(2).mean()
                count += 1
        return loss / max(count, 1)

    def _head_recon_loss(self, z: Tensor, pos_edge_index: Tensor, neg_edge_index: Tensor) -> Tensor:
        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()
        return pos_loss + neg_loss

    def recon_loss(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Optional[Tensor] = None,
    ) -> Tensor:
        if neg_edge_index is None:
            raise ValueError("MVGAE.recon_loss 需要显式传入 neg_edge_index")

        if z.dim() == 2:
            return self._head_recon_loss(z, pos_edge_index, neg_edge_index)

        losses = [self._head_recon_loss(z[:, h], pos_edge_index, neg_edge_index) for h in range(z.size(1))]
        return torch.stack(losses).mean()

    def _fused_edge_pred(self, z: Tensor, edge_index: Tensor) -> Tensor:
        preds = torch.stack(
            [self.decoder(z[:, h], edge_index, sigmoid=True) for h in range(z.size(1))],
            dim=0,
        )
        return preds.mean(dim=0)

    @torch.no_grad()
    def test(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
    ) -> tuple[float, float]:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
        except ImportError as e:
            raise ImportError("评估 AUC/AP 需要 scikit-learn") from e

        if z.dim() == 2:
            pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
            neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        else:
            pos_pred = self._fused_edge_pred(z, pos_edge_index)
            neg_pred = self._fused_edge_pred(z, neg_edge_index)

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0).cpu().numpy()
        pred = torch.cat([pos_pred, neg_pred], dim=0).cpu().numpy()
        return float(roc_auc_score(y, pred)), float(average_precision_score(y, pred))

    def encode_fused(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """返回融合嵌入 Z_init。"""
        if self.shared_backbone:
            return self.compute_z_init(x, edge_index)
        z = self.encode(x, edge_index)
        if z.dim() == 3:
            return z.reshape(z.size(0), -1)
        return z

    @torch.no_grad()
    def predict_adjacency(
        self,
        x: Tensor,
        edge_index: Tensor,
        symmetrize: bool = True,
    ) -> Tensor:
        self.eval()
        z = self.encode(x, edge_index)
        if z.dim() == 3:
            head_adjs = [torch.sigmoid(z[:, h] @ z[:, h].T) for h in range(z.size(1))]
            adj = torch.stack(head_adjs, dim=0).mean(dim=0)
        else:
            adj = torch.sigmoid(z @ z.T)
        if symmetrize:
            adj = (adj + adj.T) * 0.5
        adj.fill_diagonal_(0.0)
        return adj


def build_shared_mvgae(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_heads: int = 4,
    num_gcn_layers: int = 2,
    fusion_out_dim: Optional[int] = None,
    variational: bool = True,
) -> MVGAE:
    """构建共享骨干 MVGAE / 单头 GAE（预训练阶段）。"""
    encoder = SharedMultiHeadGCNEncoder(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_heads=num_heads,
        num_gcn_layers=num_gcn_layers,
        fusion_out_dim=fusion_out_dim,
        variational=variational,
    )
    return MVGAE(encoder, shared_backbone=True, variational=variational)


def build_pyg_mvgae(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_heads: int = 4,
    num_gcn_layers: int = 2,
) -> MVGAE:
    """构建独立多头 MVGAE（邻接学习兼容）。"""
    encoder = MultiHeadGCNEncoder(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_heads=num_heads,
        num_gcn_layers=num_gcn_layers,
    )
    return MVGAE(encoder, shared_backbone=False, variational=True)
