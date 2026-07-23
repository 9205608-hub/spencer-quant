"""多因子合成 (M9 第一步: 因子 → 单一信号)。

两种合成:
- equal_weight: 各因子截面排名等权平均。稳健基线, 无需估计任何参数。
- icir_weight:  用**过去可得的** IC 走势加权。PIT 细节是这里的全部难点:
  T 日因子的 IC 要等 T+horizon+1 日标签完成才可知, 所以加权用的 IC 序列
  必须先 shift(horizon+2) 再取滚动均值 —— 否则就是用未来的 IC 加权过去
  的因子(隐蔽前视, 30问#15 的变体)。

工程细节: 不同因子预热期不同(如递推因子60日/风格250日), 合成对每个
(date,code) 格子用"当格可用因子"的加权平均(逐格分母), 而不是要求全员
到齐 —— 否则合成信号的覆盖被最慢的因子拖死。

约定: 输入因子必须已经 orient 过(值大=看多), 合成不再管方向。
"""
from __future__ import annotations

import pandas as pd

from ..eval.panel import ic_series


def equal_weight(factors: dict[str, pd.DataFrame], min_count: int = 2) -> pd.DataFrame:
    num = den = None
    for df in factors.values():
        r = df.rank(axis=1, pct=True)
        m = r.notna().astype(float)
        num = r.fillna(0.0) if num is None else num + r.fillna(0.0)
        den = m if den is None else den + m
    return (num / den.where(den >= min_count))


def icir_weight(factors: dict[str, pd.DataFrame], fwd: pd.DataFrame,
                horizon: int, window: int = 252, min_periods: int = 120,
                min_count: int = 2) -> pd.DataFrame:
    """滚动 ICIR 加权(负权截断为0 —— 已 orient 的因子权重为负说明近期失效)。"""
    num = den = None
    for name, f in factors.items():
        ic = ic_series(f, fwd).reindex(f.index)
        ic_known = ic.shift(horizon + 2)                    # 标签完成日之后才可用
        mu = ic_known.rolling(window, min_periods=min_periods).mean()
        sd = ic_known.rolling(window, min_periods=min_periods).std()
        w = (mu / sd).clip(lower=0.0)

        r = f.rank(axis=1, pct=True)
        m = r.notna().mul(w, axis=0).fillna(0.0)
        t = r.fillna(0.0).mul(w, axis=0).fillna(0.0)
        num = t if num is None else num + t
        den = m if den is None else den + m
    cnt = None
    for f in factors.values():
        c = f.notna().astype(float)
        cnt = c if cnt is None else cnt + c
    return (num / den.where(den > 0)).where(cnt >= min_count)
