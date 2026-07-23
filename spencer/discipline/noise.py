"""噪声对照: 真因子必须跑赢一群"纯噪声但同形状"的分身。

概念出处(全公开): 置换检验(permutation test)的标准做法; 也是姊妹项目
alpha-court 的核心理念之一, 此处为 Spencer 原生实现。

为什么: IC 阈值是一把静态尺子, 但"多好才算好"其实取决于数据本身 ——
同一个 0.02 的 IC, 在宽截面低噪声市场里可能显著, 在窄截面高噪声市场里
可能就是运气。噪声对照把尺子换成动态的: 用真因子自己的形状(逐日覆盖、
缺失结构、边际分布全保留)造 n 个截面内随机置换的分身, 只摧毁"股票-因子
值"的配对 —— 分身的 |IC| 分布就是"这套数据里纯运气能拿到多少"的实测,
真因子的经验 p 值 = (1 + #{|IC_噪声| ≥ |IC_真|}) / (n + 1)。

这比高斯噪声臂更严格: 高斯臂连覆盖结构都不像真因子, 赢它不光彩。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..eval.panel import ic_series


def permute_within_rows(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """逐行(逐个截面)置换非缺失值, NaN 位置不动 —— 保形状摧毁配对。"""
    out = df.to_numpy(copy=True)
    for i in range(out.shape[0]):
        mask = ~np.isnan(out[i])
        vals = out[i][mask]
        rng.shuffle(vals)
        out[i][mask] = vals
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def noise_control(factor_df: pd.DataFrame, fwd: pd.DataFrame,
                  n_arms: int = 20, min_names: int = 30,
                  seed: int = 7) -> dict:
    """返回 {real_abs_ic, noise_abs_ic_mean/std/max, p_value, n_arms}。

    判读: p_value ≤ 1/(n_arms+1) 是本方法能给出的最小值(真因子赢了所有
    分身); p 大 = 读数和纯运气难以区分。n_arms=20 时最小 p≈0.048, 要更
    细的分辨率就加臂数(计算量线性涨)。
    """
    rng = np.random.default_rng(seed)
    real = abs(float(ic_series(factor_df, fwd, min_names).mean()))
    noise = []
    for _ in range(n_arms):
        arm = permute_within_rows(factor_df, rng)
        noise.append(abs(float(ic_series(arm, fwd, min_names).mean())))
    noise = np.asarray(noise)
    p = (1 + int((noise >= real).sum())) / (n_arms + 1)
    return {"real_abs_ic": round(real, 4),
            "noise_abs_ic_mean": round(float(noise.mean()), 4),
            "noise_abs_ic_std": round(float(noise.std()), 4),
            "noise_abs_ic_max": round(float(noise.max()), 4),
            "p_value": round(p, 4), "n_arms": n_arms}
