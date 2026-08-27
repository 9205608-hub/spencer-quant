"""自算风格因子 —— Barra CNE5 公开白皮书思想的简化实现。

本模块提供五个行情风格: SIZE / BETA / MOMENTUM / VOLATILITY / LIQUIDITY。
第六风格 VALUE(BTOP) 需要财报 PIT 管道, 由 risk/fundamental.py 实现
(30问#33 欠账已补), 集成方将其并入本模块产出的风格 dict。

口径备注:
- 市场收益用可得样本的等权均值(自洽, 不引入指数数据源)。
- beta 用 250 日滚动 cov/var, 同一估计量口径(先算 E[xy]-E[x]E[y]),
  避免 cov 与 var 用不同修正系数导致的系统偏差。
- 所有风格值只用 T 日及以前的数据, PIT 安全。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.store import WideStore


def size_proxy(store: WideStore) -> pd.DataFrame:
    """对数流通市值代理: close × volume / turn (换手率定义反解流通股本)。"""
    close, volume, turn = (store.load(c) for c in ("close", "volume", "turn"))
    cap = close * volume / turn.where(turn > 1e-6)
    return np.log(cap).ffill(limit=20)


def build_styles(store: WideStore, beta_window: int = 250) -> dict[str, pd.DataFrame]:
    adj = store.load("adj_close")
    turn = store.load("turn")
    ret = adj.pct_change(fill_method=None)
    mkt = ret.mean(axis=1)

    mp = beta_window // 2 + 25
    e_xy = ret.mul(mkt, axis=0).rolling(beta_window, min_periods=mp).mean()
    e_x = ret.rolling(beta_window, min_periods=mp).mean()
    e_y = mkt.rolling(beta_window, min_periods=mp).mean()
    var = (mkt ** 2).rolling(beta_window, min_periods=mp).mean() - e_y ** 2
    beta = (e_xy - e_x.mul(e_y, axis=0)).div(var, axis=0)

    styles = {
        "size": size_proxy(store),
        "beta": beta,
        "momentum": adj.shift(21) / adj.shift(273) - 1.0,          # 12-1月动量
        "volatility": ret.rolling(60, min_periods=40).std(),
        "liquidity": np.log(turn.rolling(21, min_periods=10).mean().where(lambda x: x > 1e-8)),
    }
    return styles


def load_industry(csv_path, codes: pd.Index) -> pd.Series:
    """code → 行业名 (快照口径, 欠账见30问#31)。缺失映射为'未分类'。"""
    df = pd.read_csv(csv_path)
    m = df.set_index("code")["industry"]
    return m.reindex(codes).fillna("未分类")
