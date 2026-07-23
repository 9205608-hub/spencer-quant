"""标准评估面板 (alphalens tear sheet 思想的自实现)。

一个因子一张面板, 固定输出六件套:
  1. RankIC 序列与累计曲线      4. 秩自相关(换手代理)
  2. 汇总统计(IC均值/ICIR/t/N)  5. 分层收益(单调性检查)
  3. 逐年一致性表 + 月度热力图   6. 多空组合累计曲线

判读铁律: 看逐年一致性, 不看裸均值; 单年驱动 = 假信号。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data.store import WideStore


# ---------- 前瞻收益 ----------

def forward_returns(store: WideStore, horizon: int) -> pd.DataFrame:
    """T 日因子对应 T+1 → T+1+horizon 的收益: shift(-(1+h))/shift(-1) - 1.

    跳过 T+1 开盘前无法建仓的那一天 —— 前视偏差第一道闸。收益轴用后复权价。
    """
    adj = store.load("adj_close")
    return adj.shift(-(1 + horizon)) / adj.shift(-1) - 1.0


def forward_return_1d(store: WideStore) -> pd.DataFrame:
    adj = store.load("adj_close")
    return adj.shift(-2) / adj.shift(-1) - 1.0


# ---------- IC 族 ----------

def ic_series(factor_df: pd.DataFrame, fwd: pd.DataFrame, min_names: int = 30) -> pd.Series:
    both = factor_df.notna() & fwd.notna()
    fr = factor_df[both].rank(axis=1)
    rr = fwd[both].rank(axis=1)
    ic = fr.corrwith(rr, axis=1)
    ic[both.sum(axis=1) < min_names] = np.nan
    return ic.dropna()


def ic_summary(ic: pd.Series, horizon: int) -> dict:
    n = len(ic)
    mean, std = ic.mean(), ic.std()
    # 重叠窗口收益导致IC序列自相关, t值按有效样本数 n/horizon 保守缩水
    n_eff = max(n / max(horizon, 1), 1.0)
    return {
        "ic_mean": round(float(mean), 4),
        "ic_ir_daily": round(float(mean / std), 3),
        "t_stat_conservative": round(float(mean / std * np.sqrt(n_eff)), 2),
        "n_days": n,
    }


def yearly_table(ic: pd.Series) -> pd.DataFrame:
    g = ic.groupby(ic.index.year)
    df = pd.DataFrame({"ic_mean": g.mean(), "ic_ir": g.mean() / g.std(), "n": g.size()})
    df["positive"] = (df["ic_mean"] > 0)
    return df.round(4)


def monthly_matrix(ic: pd.Series) -> pd.DataFrame:
    return ic.groupby([ic.index.year, ic.index.month]).mean().unstack().round(4)


# ---------- 换手与分层 ----------

def rank_autocorr(factor_df: pd.DataFrame, lag: int = 5) -> float:
    r = factor_df.rank(axis=1, pct=True)
    ac = r.corrwith(r.shift(lag), axis=1).mean()
    return round(float(ac), 4)


def quantile_returns(factor_df: pd.DataFrame, fwd1: pd.DataFrame, q: int = 5) -> pd.DataFrame:
    """按因子分 q 层的等权次日收益序列, 列 = Q1(低)...Qq(高)."""
    pct = factor_df.rank(axis=1, pct=True)
    bucket = np.ceil(pct * q)
    out = {}
    for b in range(1, q + 1):
        out[f"Q{b}"] = fwd1.where(bucket == b).mean(axis=1)
    df = pd.DataFrame(out)
    df["LS"] = df[f"Q{q}"] - df["Q1"]
    return df.dropna(how="all")


# ---------- 面板总装 ----------

def run_panel(name: str, factor_df: pd.DataFrame, store: WideStore,
              horizon: int, q: int, min_names: int, outdir: Path,
              fwd: pd.DataFrame | None = None,
              fwd1: pd.DataFrame | None = None) -> dict:
    """fwd 可注入外部标签(如风格中性化后的收益) —— 多档读数用同一面板函数,
    保证「对比必同函数同参数」。"""
    if fwd is None:
        fwd = forward_returns(store, horizon)
    if fwd1 is None:
        fwd1 = forward_return_1d(store)

    ic = ic_series(factor_df, fwd, min_names)
    summ = ic_summary(ic, horizon)
    yr = yearly_table(ic)
    mm = monthly_matrix(ic)
    qret = quantile_returns(factor_df, fwd1, q)
    ac = rank_autocorr(factor_df)

    result = {
        "factor": name, **summ,
        "rank_autocorr_5d": ac,
        "yearly_all_positive": bool(yr["positive"].all()),
        "ls_ann_ret_gross": round(float(qret["LS"].mean() * 252), 4),
    }

    outdir.mkdir(parents=True, exist_ok=True)
    _plot(name, ic, mm, qret, outdir / f"panel_{name}.png")
    yr.to_csv(outdir / f"yearly_{name}.csv")

    print(f"\n== {name} ==")
    print(f"  IC均值 {summ['ic_mean']}  日频ICIR {summ['ic_ir_daily']}  "
          f"保守t {summ['t_stat_conservative']}  样本 {summ['n_days']}天")
    print(f"  秩自相关(5d) {ac}   多空年化(毛) {result['ls_ann_ret_gross']:.1%}")
    print(f"  逐年: {' '.join(f'{y}:{v:+.4f}' for y, v in yr['ic_mean'].items())}"
          f"   全正={result['yearly_all_positive']}")
    return result


def _plot(name: str, ic: pd.Series, mm: pd.DataFrame, qret: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"factor panel: {name}", fontsize=14)

    ax = axes[0][0]
    ic.cumsum().plot(ax=ax, lw=1.2)
    ax.set_title(f"cumulative RankIC (mean={ic.mean():.4f})")
    ax.axhline(0, color="gray", lw=0.6)

    ax = axes[0][1]
    im = ax.imshow(mm.values, aspect="auto", cmap="RdYlGn",
                   vmin=-abs(mm).max().max(), vmax=abs(mm).max().max())
    ax.set_title("monthly IC heatmap")
    ax.set_yticks(range(len(mm.index)), mm.index)
    ax.set_xticks(range(len(mm.columns)), mm.columns)
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1][0]
    (qret.drop(columns="LS").mean() * 252).plot.bar(ax=ax)
    ax.set_title("annualized return by quantile (gross)")
    ax.axhline(0, color="gray", lw=0.6)

    ax = axes[1][1]
    qret["LS"].cumsum().plot(ax=ax, lw=1.2, color="darkred")
    ax.set_title("long-short cumulative (gross, top-bottom)")
    ax.axhline(0, color="gray", lw=0.6)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
