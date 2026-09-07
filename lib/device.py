"""训练设备解析（GPU / CPU）。"""
from __future__ import annotations

import os

import torch


def resolve_device(ctx: str = "0", prefer_cuda: bool = True) -> torch.device:
    """
    根据配置 ctx 选择训练设备。
    ctx 对应 CUDA_VISIBLE_DEVICES，例如 "0" 使用第一块 GPU。
    """
    if prefer_cuda and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ctx)
        if torch.cuda.device_count() < 1:
            print("warning: CUDA 可用但 device_count=0，回退到 CPU")
            return torch.device("cpu")
        device = torch.device("cuda:0")
        print("Using GPU:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)
        return device

    os.environ["CUDA_VISIBLE_DEVICES"] = str(ctx)
    if prefer_cuda:
        print(
            "warning: 未检测到 CUDA。当前 PyTorch:",
            torch.__version__,
            "\n  请安装 GPU 版: pip install torch --index-url https://download.pytorch.org/whl/cu124",
        )
    return torch.device("cpu")
