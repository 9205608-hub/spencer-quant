"""因子注册表 + 缓存。

两条硬规矩：
1. 因子函数只接收 store，输出 date×code 宽表 —— 输入输出契约唯一。
2. 缓存新鲜度判据 = 「因子末端 == 数据末端」。末端不齐一律重算，
   杜绝"数据更新了、因子还是旧的"这类静默腐烂。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from ..data.store import WideStore

_REGISTRY: dict[str, Callable[[WideStore], pd.DataFrame]] = {}


def factor(name: str):
    """注册装饰器: @factor("mom_20_5")"""
    def deco(fn):
        if name in _REGISTRY:
            raise KeyError(f"因子重名: {name}")
        _REGISTRY[name] = fn
        fn.factor_name = name
        return fn
    return deco


def list_factors() -> list[str]:
    return sorted(_REGISTRY)


def compute(name: str, store: WideStore, cache_dir: Path, use_cache: bool = True) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{name}.parquet"
    data_end = store.end_date()

    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
        df.index = pd.to_datetime(df.index)
        if df.index[-1] == data_end:
            return df
        print(f"[factor] {name} 缓存末端 {df.index[-1].date()} != 数据末端 {data_end.date()}, 重算")

    df = _REGISTRY[name](store)
    assert df.index[-1] == data_end, (
        f"{name}: 因子末端 {df.index[-1].date()} != 数据末端 {data_end.date()} —— "
        f"因子必须铺满到数据末端"
    )
    df.to_parquet(cache)
    return df
