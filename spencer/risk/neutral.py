"""行业 + 风格中性化: 逐日截面回归取残差。

两个用途, 同一个函数:
1. 中性化【因子】—— 剥掉行业与风格暴露, 看还剩多少独立信息;
2. 中性化【标签(前瞻收益)】—— 得到风格中性收益(Barra 纯因子收益思想),
   用它算 IC = 剥掉行业与风格 β 之后的"纯 alpha 读数"。
   这是工业界"多档中性化读数"的通用做法: 同一个因子在
   raw / 市值中性 / 行业+全风格中性 三档下的 IC 阶梯, 直接量出
   它有多少收益其实是风格搭便车。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..factor.ops import winsorize_mad, zscore


def residualize(panel: pd.DataFrame,
                styles: dict[str, pd.DataFrame],
                industry: pd.Series | None = None,
                min_names: int = 50) -> pd.DataFrame:
    """panel ~ 行业哑变量 + 标准化风格, 返回逐日 OLS 残差。

    - 风格先截面去极值+标准化(避免风格自身的极端值劫持回归);
    - 行业哑变量张满截距空间, 故不再加常数项;
    - 单日有效样本 < max(min_names, 自由度+10) 时该日整行 NaN。
    """
    prep = {k: zscore(winsorize_mad(v)) for k, v in styles.items()}

    if industry is not None:
        industry = industry.reindex(panel.columns).fillna("未分类")
        cats = pd.Categorical(industry)
        cat_ids = np.asarray(cats.codes)
        n_ind = len(cats.categories)
    else:
        cat_ids, n_ind = None, 0

    out = pd.DataFrame(np.nan, index=panel.index, columns=panel.columns)
    style_names = list(prep)

    for dt in panel.index:
        y = panel.loc[dt]
        xs = [prep[k].loc[dt] for k in style_names]
        valid = y.notna()
        for x in xs:
            valid &= x.notna()
        n = int(valid.sum())
        k_free = n_ind + len(xs) + 1
        if n < max(min_names, k_free + 10):
            continue

        cols = []
        if cat_ids is not None:
            ids = cat_ids[valid.values]
            D = np.zeros((n, n_ind))
            D[np.arange(n), ids] = 1.0
            D = D[:, D.sum(axis=0) > 0]        # 当日无成员的行业列剔除
            cols.append(D)
        else:
            cols.append(np.ones((n, 1)))
        cols.extend(x[valid].values.reshape(-1, 1) for x in xs)

        X = np.hstack(cols)
        beta, *_ = np.linalg.lstsq(X, y[valid].values, rcond=None)
        out.loc[dt, valid] = y[valid].values - X @ beta
    return out
