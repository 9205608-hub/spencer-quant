"""模型因子 (M8 初版): 特征 → 梯度提升树 → 预测值当因子。

结构(每一条都是 PIT 决定):
- 特征 = 已注册因子 + 五风格, 逐日截面 zscore(跨股票可比, 不跨时间);
- 标签 = 中性化前瞻收益的截面排名(稳健, 剥掉行业与风格β后的纯 alpha 目标);
- 训练 = 逐年 walk-forward: 预测第 Y 年只用 [Y-train_years, Y) 的滚动窗,
  且窗末端再砍掉 embargo 个交易日(标签窗跨期泄漏的隔离带);
- 模型 = sklearn HistGradientBoosting(CPU 友好, 无需调参起步)。

诚实声明: 这是"模型因子管线正确性"的初版, 不是调优过的预测器。超参
全部取库默认并写死在这里; 任何调参都必须走台账记 N。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..factor.ops import zscore, winsorize_mad


def walk_forward_model_factor(features: dict[str, pd.DataFrame],
                              label: pd.DataFrame,
                              train_years: int = 3,
                              embargo_days: int = 10,
                              seed: int = 7,
                              verbose: bool = True) -> pd.DataFrame:
    feats = {k: zscore(winsorize_mad(v)) for k, v in features.items()}
    names = list(feats)
    idx = label.index
    cols = label.columns
    lbl = label.rank(axis=1, pct=True)

    years = sorted(set(idx.year))
    out = pd.DataFrame(np.nan, index=idx, columns=cols)

    def stack(dates) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.stack([feats[k].loc[dates].values.ravel() for k in names], axis=1)
        y = lbl.loc[dates].values.ravel()
        ok = ~np.isnan(y)
        for j in range(X.shape[1]):
            ok &= ~np.isnan(X[:, j])
        return X[ok], y[ok], ok

    for test_year in years:
        train_dates = idx[(idx.year >= test_year - train_years) & (idx.year < test_year)]
        if len(train_dates) > embargo_days:
            train_dates = train_dates[:-embargo_days]
        test_dates = idx[idx.year == test_year]
        if len(train_dates) < 300 or len(test_dates) == 0:
            continue

        Xtr, ytr, _ = stack(train_dates)
        if len(ytr) < 50_000:
            continue
        model = HistGradientBoostingRegressor(random_state=seed)
        model.fit(Xtr, ytr)

        Xte = np.stack([feats[k].loc[test_dates].values.ravel() for k in names], axis=1)
        ok = ~np.isnan(Xte).any(axis=1)
        pred = np.full(len(Xte), np.nan)
        if ok.sum():
            pred[ok] = model.predict(Xte[ok])
        out.loc[test_dates] = pred.reshape(len(test_dates), len(cols))
        if verbose:
            print(f"  [ml] {test_year}: train {len(ytr):,} rows "
                  f"({train_dates[0].date()}→{train_dates[-1].date()}), "
                  f"predict {len(test_dates)} days")
    return out
