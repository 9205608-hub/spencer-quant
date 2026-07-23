"""标准评估面板 (alphalens tear sheet 思想的自实现)。

一个因子一张面板, 固定输出六件套:
  1. RankIC 序列与累计曲线      4. 秩自相关(换手代理)
  2. 汇总统计(IC均值/ICIR/t/N)  5. 分层收益(单调性检查)
  3. 逐年一致性表 + 月度热力图   6. 多空组合累计曲线

t 值给两套读数 (都是对"重叠窗口导致 IC 序列自相关"的修正, 粗细各一):
  - t_stat_conservative: 有效样本数按 n/horizon 缩水的粗修正 (保守下界);
  - t_stat_nw: Newey-West (1987) HAC 修正 (Bartlett 核, lag=horizon)。
另提供 horizon_profile: 同一因子跨多个持有期的 IC 衰减曲线
(alphalens 多 horizon 读数思想)。

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


def newey_west_tstat(x: pd.Series, lag: int) -> float:
    """Newey-West (1987) HAC 修正 t 值: 检验序列均值是否显著非零。

    为什么需要: horizon>1 时前瞻收益窗口重叠, IC 序列强自相关, 朴素
    t = mean/std*sqrt(n) 把相关样本当独立样本用 —— 分母被低估, t 被虚高。
    修正方法 = 用 Bartlett 核加权的长程方差替代朴素方差 (公开公式):

        S = γ0 + 2 * Σ_{l=1..L} (1 - l/(L+1)) * γ_l ,  γ_l = lag-l 自协方差
        t = mean / sqrt(S / n)

    Bartlett 权 (1 - l/(L+1)) 保证 S 非负 —— 这正是 Newey & West (1987)
    的核心贡献; 自协方差按惯例除 n 而非 n-l (有偏但方差更小, 且保 PSD)。
    lag 取 horizon 是重叠窗口场景的标准选择: h 日重叠在理论上恰好造成
    h 阶以内的自相关 (MA(h-1) 结构), 更远的自相关不是重叠造成的。
    极端情况: 样本<2 或长程方差非正(常数序列等退化情形) 返回 NaN,
    宁缺毋滥, 不给假 t 值。
    """
    v = x.dropna().to_numpy(dtype=float)
    n = v.size
    if n < 2:
        return float("nan")
    lag = int(max(0, min(lag, n - 1)))
    d = v - v.mean()
    s = d @ d / n  # γ0
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        s += 2.0 * w * (d[l:] @ d[:-l]) / n
    if s <= 0:
        return float("nan")
    return float(v.mean() / np.sqrt(s / n))


def ic_summary(ic: pd.Series, horizon: int) -> dict:
    n = len(ic)
    mean, std = ic.mean(), ic.std()
    # 重叠窗口收益导致IC序列自相关, 给两套修正并排读:
    #   保守缩水(粗, 有效样本数按 n/horizon 折算) + Newey-West(细, lag=horizon)。
    # 为什么留两套: 粗修正是下界式的安全垫, NW 是标准答案; 两者分歧大
    # 本身就是诊断信号(说明自相关结构偏离纯重叠假设)。
    n_eff = max(n / max(horizon, 1), 1.0)
    return {
        "ic_mean": round(float(mean), 4),
        "ic_ir_daily": round(float(mean / std), 3),
        "t_stat_conservative": round(float(mean / std * np.sqrt(n_eff)), 2),
        "t_stat_nw": round(newey_west_tstat(ic, lag=horizon), 2),
        "n_days": n,
    }


def yearly_table(ic: pd.Series) -> pd.DataFrame:
    g = ic.groupby(ic.index.year)
    df = pd.DataFrame({"ic_mean": g.mean(), "ic_ir": g.mean() / g.std(), "n": g.size()})
    df["positive"] = (df["ic_mean"] > 0)
    return df.round(4)


def monthly_matrix(ic: pd.Series) -> pd.DataFrame:
    return ic.groupby([ic.index.year, ic.index.month]).mean().unstack().round(4)


# ---------- IC 衰减曲线 (horizon profile) ----------

def horizon_profile(factor_df: pd.DataFrame, store: WideStore,
                    horizons=(1, 5, 10, 20), min_names: int = 30) -> pd.DataFrame:
    """同一因子跨多个持有期的 IC 衰减曲线 (alphalens 多 horizon 读数思想)。

    返回 DataFrame(index=horizon, columns=[ic_mean, icir, t_nw])。

    为什么: 单一 horizon 的读数看不出信号的时间结构 —— 衰减快的信号
    必须靠高换手兑现(成本吃得起吗?), 衰减慢的信号才扛得住低频调仓。
    每个 horizon 各自用 forward_returns 重算前瞻收益(同一道防前视闸),
    t 值用 Newey-West, lag=该 horizon (重叠窗口的标准 lag 选择)。

    已知近似(明示不藏): 各 horizon 的前瞻收益窗口相互嵌套 —— fwd(10)
    包含 fwd(5) 的全部收益日 —— 所以长 horizon 的读数天然含短 horizon
    信号被更多噪声日稀释后的残留。这条曲线读的是"形状"(峰在哪、衰减
    多快), 不是逐点独立显著性。icir 与 ic_summary 的 ic_ir_daily 同口径
    (日频 mean/std, 不年化), 保证跨函数数字可并排。
    """
    rows = []
    for h in horizons:
        fwd = forward_returns(store, h)
        ic = ic_series(factor_df, fwd, min_names)
        mean, std = ic.mean(), ic.std()
        rows.append({
            "ic_mean": round(float(mean), 4),
            "icir": round(float(mean / std), 3),
            "t_nw": round(newey_west_tstat(ic, lag=h), 2),
        })
    return pd.DataFrame(rows, index=pd.Index(list(horizons), name="horizon"))


def ic_decay_plot(profile: pd.DataFrame, name: str, path: Path) -> None:
    """IC 衰减曲线落盘 (matplotlib Agg, 只写文件不弹窗)。

    柱 = ic_mean (左轴), 折线 = t_nw (右轴)。为什么双轴: IC 均值给幅度,
    NW-t 给可信度, 两者一起看才能区分"信号弱"和"样本不够"。
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.arange(len(profile.index))
    ax.bar(xs, profile["ic_mean"].values, width=0.5, color="steelblue",
           label="ic_mean")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xticks(xs, [str(h) for h in profile.index])
    ax.set_xlabel("horizon (days)")
    ax.set_ylabel("ic_mean")
    ax2 = ax.twinx()
    ax2.plot(xs, profile["t_nw"].values, color="darkred", marker="o",
             lw=1.2, label="t_nw")
    ax2.set_ylabel("Newey-West t")
    ax.set_title(f"IC decay: {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


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
          f"保守t {summ['t_stat_conservative']}  NW-t {summ['t_stat_nw']}  "
          f"样本 {summ['n_days']}天")
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
