"""统计纪律: DSR 缩水夏普 / E[max SR] 期望最大夏普 / PBO(CSCV) 过拟合概率。

台账(ledger.py)记录「一共试了 N 次」, 本模块负责把 N 变成惩罚:
试得越多, 单个「最好结果」越可能只是噪声的极值。两条互补的刀:

- DSR: 把观测 Sharpe 和「N 次纯噪声试验的期望最大 Sharpe」对比 ——
  回答「这个最好成绩比瞎蒙 N 次的冠军好吗」。
- PBO: 直接度量「样本内选最优」这个动作本身的过拟合概率 ——
  回答「IS 冠军在 OOS 掉到下半区的频率有多高」。

方法出处(全部公开文献):
- PSR 概率夏普 (Probabilistic Sharpe Ratio):
  Bailey & Lopez de Prado (2012) "The Sharpe Ratio Efficient Frontier";
  非正态修正项源自 Mertens (2002) 的 Sharpe 估计量渐近方差。
- DSR 缩水夏普 (Deflated Sharpe Ratio) 与期望最大夏普 E[max SR]:
  Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting and Non-Normality"。
- PBO / CSCV 组合对称交叉验证 (Combinatorially Symmetric Cross-Validation):
  Bailey, Borwein, Lopez de Prado & Zhu (2017) "The Probability of Backtest
  Overfitting", Journal of Computational Finance。
- 汇总性介绍见 AFML (Lopez de Prado, 2018,《Advances in Financial Machine
  Learning》) 第 11 / 14 章。

频率约定(容易翻车, 写死): 本模块所有 Sharpe 都是「与 T 同频的每期 Sharpe」
—— 日收益就传日频 Sharpe(mean/std, 不年化), T = 观测期数。年化 Sharpe 传进来
公式全错。峰度参数是原始峰度(正态=3), pandas .kurt() 给的是超额峰度要 +3。

实现依赖: 标准正态 CDF/分位数用标准库 statistics.NormalDist —— 刻意不引入
scipy, 保持 requirements 不变。
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

_PHI = NormalDist()                      # 标准正态
_EULER_GAMMA = 0.5772156649015329        # Euler-Mascheroni 常数 γ


# ---------- 期望最大夏普 (选择偏差的基准线) ----------

def expected_max_sharpe(n_trials: int, T: int, var_trials: float | None = None) -> float:
    """N 次独立纯噪声试验里, 期望的最大 Sharpe (DSR 的比较基准 SR0)。

    公式 (Bailey & Lopez de Prado 2014, 极值理论渐近近似):
        E[max SR] ≈ sqrt(V) * [ (1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e)) ]
    其中 γ 为 Euler-Mascheroni 常数, V 为各次试验 Sharpe 估计量的方差。

    为什么 var_trials 默认 1/T: 零假设(收益均值为0的 iid 序列)下, 每期 Sharpe
    估计量的方差 ≈ (1 + SR²/2)/T, 代入 SR=0 即 1/T。若手上有全部 N 次试验的
    实测 Sharpe, 传它们的样本方差 V[{SR_n}] 更贴近论文原式 —— 试验彼此相关时
    实测方差通常小于 1/T, 用默认值是偏保守(基准更高、更难过)的一侧。

    已知近似(明示不藏): 这是渐近式, N 小时高估 —— N=10 时约高 2%
    (近似 1.574 vs 精确 1.539), N 越大越准; N<=1 没有选择效应, 直接返回 0。
    """
    if n_trials <= 1:
        return 0.0
    if var_trials is None:
        var_trials = 1.0 / T
    z1 = _PHI.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _PHI.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(var_trials) * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2)


# ---------- PSR / DSR ----------

def probabilistic_sharpe(observed_sharpe: float, benchmark_sharpe: float, T: int,
                         skew: float = 0.0, kurt: float = 3.0) -> float:
    """PSR: 真实 Sharpe 超过基准 SR* 的概率 (Bailey & Lopez de Prado 2012)。

        PSR = Φ[ (SR - SR*)·sqrt(T-1) / sqrt(1 - γ₃·SR + (γ₄-1)/4·SR²) ]

    分母是非正态收益下 Sharpe 估计量的渐近标准差修正(Mertens 2002):
    负偏 / 肥尾都会放大 Sharpe 的估计误差, 从而压低 PSR —— 这正是想要的惩罚。

    参数口径: SR 与 T 同频不年化; kurt 是原始峰度(正态=3)。
    极端偏度/峰度组合可能使分母平方项非正(越出公式适用域), 此时返回 nan
    而不是硬算 —— 让调用者显式面对, 不静默给数。
    """
    sr = float(observed_sharpe)
    denom_sq = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if denom_sq <= 0.0 or T < 2:
        return float("nan")
    z = (sr - float(benchmark_sharpe)) * math.sqrt(T - 1.0) / math.sqrt(denom_sq)
    return _PHI.cdf(z)


def deflated_sharpe(observed_sharpe: float, n_trials: int, T: int,
                    skew: float = 0.0, kurt: float = 3.0,
                    var_trials: float | None = None) -> dict:
    """DSR: 对「N 次试验里挑出来的最好 Sharpe」做多重检验缩水。

    DSR = PSR(SR0), SR0 = expected_max_sharpe(N, T) —— 基准不再是 0, 而是
    「纯噪声试 N 次的期望冠军成绩」。DSR > 0.95 才算过 95% 置信 (Bailey &
    Lopez de Prado 2014)。

    返回 dict 而不是裸浮点(与 run_panel/layered_backtest 的返回风格一致):
    - dsr:          缩水后的显著性概率, 判定用这个
    - sr0:          期望最大夏普基准(与 T 同频)
    - psr_vs_zero:  不缩水、对基准 0 的朴素 PSR —— 与 dsr 的落差就是
                    「选择偏差吃掉了多少显著性」, 落差本身是信息
    - n_trials, T:  留痕, 报告 dsr 时必须一起报(N 从台账来, 见
                    trial_n_from_ledger)
    """
    sr0 = expected_max_sharpe(n_trials, T, var_trials=var_trials)
    return {
        "dsr": probabilistic_sharpe(observed_sharpe, sr0, T, skew, kurt),
        "sr0": sr0,
        "psr_vs_zero": probabilistic_sharpe(observed_sharpe, 0.0, T, skew, kurt),
        "n_trials": int(n_trials),
        "T": int(T),
    }


# ---------- PBO: CSCV 组合对称交叉验证 ----------

def pbo_cscv(returns_matrix, n_splits: int = 16, chunk: int = 4096) -> dict:
    """CSCV 过拟合概率 PBO (Bailey, Borwein, Lopez de Prado & Zhu 2017)。

    输入: T×N 收益矩阵(N 个策略/因子配置的同频收益, DataFrame 或 ndarray)。
    做法: 时间轴切成 n_splits 个等长块, 枚举全部 C(S, S/2) 种「一半块做
    样本内(IS)/另一半做样本外(OOS)」的组合; 每种组合取 IS 里 Sharpe 最高的
    策略 n*, 看它在 OOS 的相对秩 ω = rank/(N+1), logit λ = ln(ω/(1-ω));
    PBO = P(λ < 0) = IS 冠军在 OOS 掉进下半区的频率。
    纯噪声 ≈ 0.5, 真实技能 → 0, 系统性过拟合 → 1。

    实现与近似(全部明示):
    - 组内评价指标 = 每期 Sharpe(mean/std, ddof=1), 与论文基准用法一致;
    - 时间轴尾部不足整块的行直接丢弃(块等长是 CSCV 的结构要求);
    - 块内保持时间连续、块间组合打乱 —— 这是 CSCV 对时序结构的妥协, 论文
      原文即如此(每块内部保序);
    - 性能: 预计算每块的 sum/sum², 组合层用 0/1 矩阵乘法合成, 不重复扫
      原矩阵; 组合按 chunk 分批枚举控内存(S=16 时共 12870 个组合);
    - 不支持 NaN: 上游先对齐/dropna, 这里 assert 挡住而不是静默传播。
    """
    M = returns_matrix.to_numpy(dtype=float) if isinstance(returns_matrix, pd.DataFrame) \
        else np.asarray(returns_matrix, dtype=float)
    assert M.ndim == 2, "returns_matrix 必须是 T×N 二维矩阵"
    assert n_splits >= 2 and n_splits % 2 == 0, "n_splits 必须是 >=2 的偶数"
    T_raw, N = M.shape
    assert N >= 2, "至少要有 2 个策略才谈得上『选最优』"
    L = T_raw // n_splits                      # 每块行数
    assert L >= 2, f"每块至少2行: T={T_raw} 撑不起 n_splits={n_splits}"
    M = M[: L * n_splits]
    assert not np.isnan(M).any(), "收益矩阵含 NaN, 上游先对齐/dropna"

    S = n_splits
    T_half = L * S // 2                        # IS 与 OOS 各自的行数(对称)
    blocks = M.reshape(S, L, N)
    bsum = blocks.sum(axis=1)                  # (S, N) 每块和
    bsq = (blocks ** 2).sum(axis=1)            # (S, N) 每块平方和
    tot_sum, tot_sq = bsum.sum(axis=0), bsq.sum(axis=0)

    def _sharpe(s: np.ndarray, sq: np.ndarray) -> np.ndarray:
        mean = s / T_half
        var = (sq - T_half * mean ** 2) / (T_half - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return mean / np.sqrt(var)

    n_c = math.comb(S, S // 2)
    omega = np.empty(n_c)
    it = itertools.combinations(range(S), S // 2)
    pos = 0
    while True:
        batch = list(itertools.islice(it, chunk))
        if not batch:
            break
        idx = np.asarray(batch)                              # (B, S/2)
        mask = np.zeros((len(batch), S))
        mask[np.arange(len(batch))[:, None], idx] = 1.0      # 0/1 组合矩阵
        is_sr = _sharpe(mask @ bsum, mask @ bsq)             # (B, N)
        oos_sr = _sharpe(tot_sum - mask @ bsum, tot_sq - mask @ bsq)
        n_star = np.argmax(np.nan_to_num(is_sr, nan=-np.inf), axis=1)
        star = oos_sr[np.arange(len(batch)), n_star]
        # ω 含自身(rank>=1), 故永不为 0 或 1, logit 有限
        omega[pos:pos + len(batch)] = \
            (oos_sr <= star[:, None]).sum(axis=1) / (N + 1.0)
        pos += len(batch)

    lam = np.log(omega / (1.0 - omega))
    return {
        "pbo": float((lam < 0).mean()),
        "n_combinations": n_c,
        "n_strategies": N,
        "n_periods_used": L * S,
        "n_splits": S,
        "oos_rank_mean": float(omega.mean()),   # 纯噪声≈0.5, 越高越好
        "lambda_mean": float(lam.mean()),
    }


# ---------- 台账桥接: N 从哪里来 ----------

def trial_n_from_ledger(ledger_path: Path | str | None = None,
                        factor: str | None = None) -> int:
    """从实验台账读当前累计试验次数 N —— deflated_sharpe 的 n_trials 输入。

    为什么必须从台账读而不是手填: 手填的 N 永远偏小(人只记得「正式」的那几次,
    忘了废掉的草稿), 台账 append-only 才是 N 的下界真相源。
    默认路径 = 项目根/research_ledger.csv, 与 examples/quickstart.py 同一约定。
    factor 传 None = 全台账计数(跨因子挑最优时用全 N); 传因子名 = 只数该因子
    (单因子调参场景)。台账不存在返回 0。
    """
    from .ledger import trial_count          # 只 import 不修改, 单一计数实现
    if ledger_path is None:
        from ..config import ROOT
        ledger_path = ROOT / "research_ledger.csv"
    return trial_count(Path(ledger_path), factor=factor)


def dsr_from_ledger(observed_sharpe: float, T: int,
                    skew: float = 0.0, kurt: float = 3.0,
                    ledger_path: Path | str | None = None,
                    factor: str | None = None) -> dict:
    """一步到位: 台账 N + 观测 Sharpe → DSR。

    N 按台账实际计数, 但至少为 1(眼前这次也是一次试验); 结果里带上
    n_trials_from_ledger 留痕, 报告时 N 和 DSR 必须一起出现。
    """
    n = max(trial_n_from_ledger(ledger_path, factor=factor), 1)
    out = deflated_sharpe(observed_sharpe, n, T, skew=skew, kurt=kurt)
    out["n_trials_from_ledger"] = n
    return out
