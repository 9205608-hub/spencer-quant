"""单一真相源配置加载。所有模块通过 load_config() 取参数，禁止散落硬编码。"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["root"] = ROOT
    cfg["data_dir"] = ROOT / cfg["data_dir"]
    cfg["output_dir"] = ROOT / cfg["output_dir"]
    return cfg
