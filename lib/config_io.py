"""Configuration file helpers (no heavy ML dependencies)."""
from __future__ import annotations

import configparser
import os


def read_config(config_path: str) -> configparser.ConfigParser:
    """Read .conf with UTF-8 (avoids GBK decode errors on Windows).

    Raises FileNotFoundError / ValueError with a clear message when the path
    is wrong or the file is missing required sections (common on unsynced servers).
    """
    path = os.path.abspath(config_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "配置文件不存在: %s\n"
            "请确认路径正确，且已同步到服务器（例如 configurations/decouple/）。\n"
            "当前工作目录: %s"
            % (path, os.getcwd())
        )

    config = configparser.ConfigParser()
    read_ok = config.read(path, encoding="utf-8")
    if not read_ok:
        raise ValueError("无法读取配置文件（编码或格式异常）: %s" % path)

    if "Data" not in config:
        sections = list(config.sections())
        raise KeyError(
            "配置缺少 [Data] 段: %s\n"
            "已解析到的段: %s\n"
            "常见原因：服务器上该 conf 未同步、路径指错、或文件内容为空。"
            % (path, sections if sections else "(无)")
        )
    return config
