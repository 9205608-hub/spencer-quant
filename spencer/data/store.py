"""宽表存储：每个字段一张 date×code 矩阵，一个 parquet 文件。

这是 qlib 式数据层思想的最小实现：研究端 99% 的操作是"取一个字段的
全市场面板"，宽表让这一步变成 O(1) 的单文件读取。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

WIDE_FIELDS = ["open", "high", "low", "close", "adj_close", "volume",
               "amount", "turn", "is_st", "is_trading"]


class WideStore:
    def __init__(self, wide_dir: Path):
        self.dir = Path(wide_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, pd.DataFrame] = {}

    def path(self, field: str) -> Path:
        return self.dir / f"{field}.parquet"

    def save(self, field: str, df: pd.DataFrame) -> None:
        assert df.index.is_monotonic_increasing, f"{field}: 日期索引必须升序"
        assert df.index.is_unique and df.columns.is_unique, f"{field}: 索引/列重复"
        df.to_parquet(self.path(field))
        self._cache[field] = df

    def load(self, field: str) -> pd.DataFrame:
        if field not in self._cache:
            df = pd.read_parquet(self.path(field))
            df.index = pd.to_datetime(df.index)
            self._cache[field] = df
        return self._cache[field]

    def fields(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.parquet"))

    def end_date(self) -> pd.Timestamp:
        """数据末端 —— 因子缓存新鲜度判据的锚点。"""
        return self.load("close").index[-1]


def build_wide_store(long_path: Path, wide_dir: Path) -> WideStore:
    long_df = pd.read_parquet(long_path)
    store = WideStore(wide_dir)
    for field in WIDE_FIELDS:
        wide = long_df.pivot(index="date", columns="code", values=field).sort_index()
        store.save(field, wide)
    print(f"[store] 宽表建成: {len(store.fields())} 字段, "
          f"{wide.shape[0]} 交易日 × {wide.shape[1]} 股票, 末端 {store.end_date().date()}")
    return store
