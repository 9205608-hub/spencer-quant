"""组合构造 (M9 第二步: 信号 → 目标持仓) —— 与执行端的唯一接口。

v0.2 实现"带缓冲区的 top-N 等权"(工业界最常用的低换手规则之一):
- 信号排进前 top_n → 买入候选;
- 已持有的票只要仍在前 buffer_n (buffer_n > top_n) 就继续持有;
- 这样避免排名在边界附近抖动造成的无谓换手。

输出 date×code 权重矩阵(行和=1), 以及任意一天的目标持仓表 csv ——
执行端(券商 QMT/PTrade 或平台策略)只需要读这张表下单, 研究端与执行端
在这里解耦。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def topn_buffer_weights(signal: pd.DataFrame, top_n: int = 50,
                        buffer_n: int = 80, rebal_days: int = 5) -> pd.DataFrame:
    """带缓冲区的 top-N 等权目标权重(只在调仓日变动, 其余日延续)。"""
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    held: list = []
    rebal_dates = set(signal.index[::rebal_days])
    for dt in signal.index:
        if dt in rebal_dates:
            row = signal.loc[dt].dropna().sort_values(ascending=False)
            if len(row) >= buffer_n:
                buffer_set = set(row.index[:buffer_n])
                keep = [c for c in held if c in buffer_set]
                need = top_n - len(keep)
                adds = [c for c in row.index if c not in keep][:max(need, 0)]
                held = keep + adds
        if held:
            weights.loc[dt, held] = 1.0 / len(held)
    return weights


def turnover_series(weights: pd.DataFrame) -> pd.Series:
    """逐日单边换手 = sum|Δw| / 2。"""
    return weights.diff().abs().sum(axis=1) / 2


def export_targets(weights: pd.DataFrame, date, out_csv: Path) -> pd.DataFrame:
    """导出某日目标持仓表(执行端接口)。"""
    w = weights.loc[date]
    tbl = w[w > 0].sort_values(ascending=False).rename("weight").reset_index()
    tbl.columns = ["code", "weight"]
    tbl.to_csv(out_csv, index=False)
    return tbl
