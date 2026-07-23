"""风格协方差与特异风险 (M6 收尾): Σ ≈ B F B' + diag(spec²)。

结构是 Barra 公开白皮书的标准分解, 估计方法全部公开:
- 因子收益: 逐日 Fama-MacBeth 截面回归 fwd1 ~ 行业哑变量 + 标准化风格,
  取风格系数序列。行业因子收益不进 F —— v1.0 只做风格块, 行业块的风险
  并入特异项(已知近似, 明示: 行业相关性被低估, 特异项被高估, 错的方向
  是保守的 —— 组合会被劝更靠近基准, 不会被怂恿冒险);
- F: 因子收益的 EWMA 协方差(halflife 默认 90 日, EWMA 是 RiskMetrics 的
  公开做法);
- 特异方差: 截面回归残差平方的 EWMA(halflife 60 日), 逐股。

PIT 记账(本模块的全部难点): fwd1[t] 的收益在 t+2 日收盘才实现 →
t 日"可用"的因子收益序列最多到 t-2 行。sigma_at() 内部做 shift, 调用方
只管给日期; 测试里用"篡改未来行不改变 sigma_at 输出"钉死这条。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..factor.ops import winsorize_mad, zscore


def style_factor_returns(fwd1: pd.DataFrame,
                         styles: dict[str, pd.DataFrame],
                         industry: pd.Series | None = None,
                         min_names: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐日截面回归, 返回 (style_ret: date×k, resid: date×code)。

    与 neutral.residualize 同一套回归, 区别是这里要的是**系数**(因子收益)
    而不是残差 —— 两者都保留, 一次回归两份产出。
    """
    prep = {k: zscore(winsorize_mad(v)) for k, v in styles.items()}
    names = list(prep)

    if industry is not None:
        industry = industry.reindex(fwd1.columns).fillna("未分类")
        cats = pd.Categorical(industry)
        cat_ids = np.asarray(cats.codes)
        n_ind = len(cats.categories)
    else:
        cat_ids, n_ind = None, 0

    ret_rows, resid = {}, pd.DataFrame(np.nan, index=fwd1.index, columns=fwd1.columns)
    for dt in fwd1.index:
        y = fwd1.loc[dt]
        xs = [prep[k].loc[dt] for k in names]
        valid = y.notna()
        for x in xs:
            valid &= x.notna()
        n = int(valid.sum())
        if n < max(min_names, n_ind + len(xs) + 10):
            continue
        cols = []
        if cat_ids is not None:
            ids = cat_ids[valid.values]
            D = np.zeros((n, n_ind))
            D[np.arange(n), ids] = 1.0
            keep = D.sum(axis=0) > 0
            D = D[:, keep]
            cols.append(D)
        else:
            cols.append(np.ones((n, 1)))
        X = np.hstack(cols + [x[valid].values.reshape(-1, 1) for x in xs])
        beta, *_ = np.linalg.lstsq(X, y[valid].values, rcond=None)
        ret_rows[dt] = beta[-len(names):]                 # 风格系数在末尾, 顺序=names
        resid.loc[dt, valid] = y[valid].values - X @ beta
    style_ret = pd.DataFrame(ret_rows, index=names).T
    return style_ret, resid


def sigma_at(date, styles: dict[str, pd.DataFrame],
             style_ret: pd.DataFrame, resid: pd.DataFrame,
             halflife_f: int = 90, halflife_s: int = 60,
             lag: int = 2, min_obs: int = 120):
    """t 日可用的 (B, F, spec, codes), 直接喂 optimizer.solve 的因子结构 Σ。

    - B: t 日的标准化风格暴露(风格值本身 PIT, 当日可用);
    - F: 截至 t-lag 的因子收益 EWMA 协方差(lag=2 = fwd1 实现滞后, 保守);
    - spec: 截至 t-lag 的残差平方 EWMA 开根(日频特异波动率);
    - codes: 上述三者全部有效的股票交集(顺序对齐)。
    """
    date = pd.Timestamp(date)
    usable = style_ret.loc[:date]
    if lag > 0:
        usable = usable.iloc[:-lag] if len(usable) > lag else usable.iloc[:0]
    usable = usable.dropna(how="any")
    if len(usable) < min_obs:
        raise ValueError(f"因子收益可用样本 {len(usable)} < {min_obs}, 无法估计 F")

    F = usable.ewm(halflife=halflife_f).cov().loc[usable.index[-1]].to_numpy()
    # 数值护栏: EWMA 协方差理论上 PSD, 浮点噪声可能给出 -1e-18 级特征值
    w, V = np.linalg.eigh(F)
    F = (V * np.clip(w, 0.0, None)) @ V.T

    sv = resid.pow(2).ewm(halflife=halflife_s, min_periods=min_obs // 2).mean()
    sv_row = sv.loc[:usable.index[-1]].iloc[-1]

    prep = {k: zscore(winsorize_mad(v)) for k, v in styles.items()}
    B_row = pd.DataFrame({k: prep[k].loc[date] for k in style_ret.columns})

    valid = B_row.notna().all(axis=1) & sv_row.reindex(B_row.index).notna() \
        & (sv_row.reindex(B_row.index) > 0)
    codes = B_row.index[valid].tolist()
    B = B_row.loc[codes].to_numpy()
    spec = np.sqrt(sv_row.reindex(codes).to_numpy())
    return B, F, spec, codes
