"""横截面算子：去极值 / 标准化 / 排名 / 中性化 / 方向翻正。

全部按行(单个交易日的截面)操作, 不跨时间 —— 跨时间的操作属于因子构造，
不属于预处理，边界必须清楚。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_mad(df: pd.DataFrame, k: float = 5.0) -> pd.DataFrame:
    """MAD 去极值: 截面 median ± k * 1.4826 * MAD."""
    med = df.median(axis=1)
    mad = (df.sub(med, axis=0)).abs().median(axis=1) * 1.4826
    lower = med - k * mad
    upper = med + k * mad
    return df.clip(lower=lower, upper=upper, axis=0)


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """截面百分位排名 (0,1]。对厚尾因子比 zscore 稳。"""
    return df.rank(axis=1, pct=True)


def neutralize(df: pd.DataFrame, *styles: pd.DataFrame) -> pd.DataFrame:
    """逐日 OLS 残差中性化: factor ~ 1 + style1 + style2 + ...

    styles 是同形状的 date×code 宽表(如对数市值)。返回残差面板。
    """
    out = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    for dt in df.index:
        y = df.loc[dt]
        X_cols = [s.loc[dt] for s in styles]
        valid = y.notna()
        for x in X_cols:
            valid &= x.notna()
        n = int(valid.sum())
        if n < len(X_cols) + 10:
            continue
        X = np.column_stack([np.ones(n)] + [x[valid].values for x in X_cols])
        beta, *_ = np.linalg.lstsq(X, y[valid].values, rcond=None)
        out.loc[dt, valid] = y[valid].values - X @ beta
    return out


def orient(factor_df: pd.DataFrame, fwd: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """方向翻正: 保证「值大 = 看多」。返回 (翻正后因子, 符号)。

    交付/入库前的惯例动作。符号由全样本 IC 均值决定 —— 这是一次样本内
    决策, 报告里必须披露(quickstart 会打印)。
    """
    ic = factor_df.rank(axis=1).corrwith(fwd.rank(axis=1), axis=1).mean()
    sign = 1 if ic >= 0 else -1
    return factor_df * sign, sign
