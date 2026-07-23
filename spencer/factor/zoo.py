"""示范因子。每个因子 = 一个纯函数: store -> date×code 宽表。

三个因子覆盖三种典型构造形态：
- mom_20_5:  纯价格窗口因子（经典动量, 跳过近5日反转段）
- vol_20:    滚动统计因子
- chip_age:  递推状态因子（教科书级筹码概念: 换手率衰减的平均持仓天数）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import factor
from ..data.store import WideStore


@factor("mom_20_5")
def mom_20_5(store: WideStore) -> pd.DataFrame:
    adj = store.load("adj_close")
    return adj.shift(5) / adj.shift(25) - 1.0


@factor("vol_20")
def vol_20(store: WideStore) -> pd.DataFrame:
    adj = store.load("adj_close")
    return adj.pct_change(fill_method=None).rolling(20, min_periods=15).std()


@factor("rev_5")
def rev_5(store: WideStore) -> pd.DataFrame:
    """短期反转(过去5日收益, 方向交给 orient)。"""
    adj = store.load("adj_close")
    return adj / adj.shift(5) - 1.0


@factor("turn_surge_5_60")
def turn_surge_5_60(store: WideStore) -> pd.DataFrame:
    """换手异动: 近5日均换手 / 近60日均换手。"""
    turn = store.load("turn")
    return turn.rolling(5, min_periods=3).mean() / \
        turn.rolling(60, min_periods=40).mean().where(lambda x: x > 1e-8)


@factor("amihud_20")
def amihud_20(store: WideStore) -> pd.DataFrame:
    """Amihud 非流动性: |日收益|/成交额 的20日均值(经典公开因子)。"""
    adj = store.load("adj_close")
    amount = store.load("amount")
    illiq = (adj.pct_change(fill_method=None).abs() / amount.where(amount > 0)) * 1e9
    return illiq.rolling(20, min_periods=12).mean()


@factor("px_pos_250")
def px_pos_250(store: WideStore) -> pd.DataFrame:
    """价格位置: 当前价在过去250日高低区间中的分位(52周高低点思想)。"""
    adj = store.load("adj_close")
    lo = adj.rolling(250, min_periods=120).min()
    hi = adj.rolling(250, min_periods=120).max()
    return (adj - lo) / (hi - lo).where(lambda x: x > 1e-12)


@factor("chip_age")
def chip_age(store: WideStore) -> pd.DataFrame:
    """平均持仓天数: age_t = (1 - turn_t) * (age_{t-1} + 1).

    直觉: 每天有 turn 比例的筹码换到新手里(龄清零), 其余筹码龄+1。
    公开概念(筹码分布/平均持仓成本线的教科书推导), 递推只用当日及
    之前的换手率, PIT 安全。上市初期龄被面板起点截断, 用 60 日预热
    期屏蔽(与 config.eval.warmup_days 一致的思想, 这里独立硬约束)。
    """
    turn = store.load("turn").fillna(0.0).clip(0.0, 1.0)
    close = store.load("close")

    ages = np.zeros(turn.shape[1])
    rows = []
    for i in range(len(turn.index)):
        ages = (1.0 - turn.values[i]) * (ages + 1.0)
        rows.append(ages.copy())
    df = pd.DataFrame(rows, index=turn.index, columns=turn.columns)

    valid_days = close.notna().cumsum()
    df[close.isna() | (valid_days < 60)] = np.nan
    return df
