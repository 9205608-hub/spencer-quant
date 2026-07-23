"""M4 统计纪律层断言: DSR / E[max SR] / PBO(CSCV) / 台账桥接。

跑法: python tests/test_m4_stats.py  (合成数据, 零网络)

四条主断言(对应任务验收):
  1. 纯噪声 100 次试验挑最好 → 朴素 PSR 看着显著, DSR 打回原形;
  2. 真信号 + 噪声混合 → 真信号 DSR 存活;
  3. 随机收益矩阵 PBO ≈ 0.5 (±0.15), 有真技能列 PBO → 0;
  4. 公式数值与文献量级一致(E[max] 蒙特卡洛复算 + PSR 结构性质)。
"""
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spencer.discipline.stats import (deflated_sharpe, dsr_from_ledger,
                                      expected_max_sharpe, pbo_cscv,
                                      probabilistic_sharpe,
                                      trial_n_from_ledger)

T = 1250          # 约5年日频
N_TRIALS = 100
VOL = 0.01        # 日波动1%


def _sr_skew_kurt(x: np.ndarray):
    """每期 Sharpe + 偏度 + 原始峰度(pandas kurt 是超额峰度, +3 还原)。"""
    s = pd.Series(x)
    return float(s.mean() / s.std()), float(s.skew()), float(s.kurt() + 3.0)


def test_dsr_kills_noise_champion():
    """纯噪声 100 个试验里挑最好的: 朴素 PSR 显著, DSR 应让显著性消失。"""
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0, VOL, size=(T, N_TRIALS))
    srs = rets.mean(axis=0) / rets.std(axis=0, ddof=1)
    best = int(np.argmax(srs))
    sr, sk, ku = _sr_skew_kurt(rets[:, best])

    naive = probabilistic_sharpe(sr, 0.0, T, sk, ku)
    d = deflated_sharpe(sr, N_TRIALS, T, sk, ku)
    # 不做多重检验校正: 冠军看起来"过了95%置信" —— 这就是赢家诅咒
    assert naive > 0.95, f"朴素PSR应虚假显著, got {naive:.4f}"
    # DSR 把基准从0抬到E[max SR]: 显著性应消失(冠军成绩≈纯噪声期望冠军)
    assert d["dsr"] < 0.90, f"噪声冠军DSR应不显著, got {d['dsr']:.4f}"
    assert d["dsr"] < naive, "DSR 必须比朴素 PSR 更严"
    assert d["sr0"] > 0, "N>1 时期望最大夏普基准应为正"
    print(f"dsr_kills_noise OK (naive PSR {naive:.3f} -> DSR {d['dsr']:.3f}, "
          f"sr0 {d['sr0']:.4f})")


def test_dsr_true_signal_survives():
    """真信号(日SR=0.2, 年化约3.2)混在同样的 100 次试验预算里 → DSR 存活。"""
    rng = np.random.default_rng(7)
    signal = rng.normal(0.002, VOL, size=T)      # 真实日均收益 20bp
    sr, sk, ku = _sr_skew_kurt(signal)
    d = deflated_sharpe(sr, N_TRIALS, T, sk, ku)
    assert d["dsr"] > 0.95, f"真信号DSR应存活, got {d['dsr']:.4f}"
    # 同预算下真信号必须比噪声冠军更显著
    rng2 = np.random.default_rng(42)
    noise = rng2.normal(0.0, VOL, size=(T, N_TRIALS))
    srs = noise.mean(axis=0) / noise.std(axis=0, ddof=1)
    nsr, nsk, nku = _sr_skew_kurt(noise[:, int(np.argmax(srs))])
    d_noise = deflated_sharpe(nsr, N_TRIALS, T, nsk, nku)
    assert d["dsr"] > d_noise["dsr"]
    print(f"dsr_signal_survives OK (DSR {d['dsr']:.4f} > 0.95)")


def test_pbo_random_near_half():
    """纯随机矩阵: IS冠军的OOS秩应均匀分布 → PBO ≈ 0.5 (±0.15)。"""
    rng = np.random.default_rng(11)
    rets = rng.normal(0.0, VOL, size=(512, 50))
    res = pbo_cscv(rets, n_splits=16)
    assert res["n_combinations"] == math.comb(16, 8) == 12870
    assert res["n_periods_used"] == 512
    assert 0.35 <= res["pbo"] <= 0.65, f"随机矩阵PBO应≈0.5, got {res['pbo']:.3f}"
    assert 0.4 <= res["oos_rank_mean"] <= 0.6
    print(f"pbo_random OK (PBO {res['pbo']:.3f}, oos_rank_mean "
          f"{res['oos_rank_mean']:.3f})")


def test_pbo_skill_low():
    """混入一列真技能(日SR=0.3): IS 总选中它、OOS 它仍在上半区 → PBO 应接近 0。
    同时验证 DataFrame 输入与 ndarray 等价。"""
    rng = np.random.default_rng(23)
    rets = rng.normal(0.0, VOL, size=(512, 50))
    rets[:, 0] += 0.003                          # 第0列 = 真信号
    res = pbo_cscv(rets, n_splits=16)
    assert res["pbo"] < 0.15, f"有真技能列时PBO应低, got {res['pbo']:.3f}"
    df = pd.DataFrame(rets)
    res_df = pbo_cscv(df, n_splits=16)
    assert abs(res_df["pbo"] - res["pbo"]) < 1e-12, "DataFrame/ndarray 应同结果"
    print(f"pbo_skill OK (PBO {res['pbo']:.3f})")


def test_formula_magnitudes():
    """公式数值与文献量级一致:
    E[max] 渐近式对蒙特卡洛复算(独立验证, 不循环论证) + PSR 结构性质。"""
    # 标准正态 N 个取最大的期望: 文献近似 N=10≈1.57, N=100≈2.53 (精确值略低)
    emax10 = expected_max_sharpe(10, T=1)        # var=1/T=1 → 纯 z 尺度
    emax100 = expected_max_sharpe(100, T=1)
    assert 1.45 < emax10 < 1.70, f"E[max z] N=10 量级错: {emax10:.3f}"
    assert 2.35 < emax100 < 2.65, f"E[max z] N=100 量级错: {emax100:.3f}"
    # 蒙特卡洛复算(4000 次抽样的均值, 标准误约0.007): 渐近式偏差应 <0.1
    rng = np.random.default_rng(3)
    mc = rng.standard_normal(size=(4000, 100)).max(axis=1).mean()
    assert abs(emax100 - mc) < 0.10, f"渐近式 {emax100:.3f} vs MC {mc:.3f}"
    # 单调性: 试验越多基准越高; 样本越长(T大)同样N的噪声冠军越小
    assert expected_max_sharpe(1000, 252) > expected_max_sharpe(100, 252) \
        > expected_max_sharpe(10, 252)
    assert expected_max_sharpe(100, 252) < expected_max_sharpe(100, 63)
    assert expected_max_sharpe(1, 252) == 0.0, "N=1 无选择效应, 基准=0"
    # PSR 结构性质(论文明示): SR=基准 → 0.5; 负偏度/肥尾压低显著性
    assert abs(probabilistic_sharpe(0.1, 0.1, 500) - 0.5) < 1e-12
    assert probabilistic_sharpe(0.1, 0.0, 500, skew=-1.0) \
        < probabilistic_sharpe(0.1, 0.0, 500, skew=0.0)
    assert probabilistic_sharpe(0.1, 0.0, 500, kurt=10.0) \
        < probabilistic_sharpe(0.1, 0.0, 500, kurt=3.0)
    # 越出适用域(分母非正)必须给 nan 不硬算
    assert math.isnan(probabilistic_sharpe(2.0, 0.0, 500, skew=5.0))
    # N=1 时 DSR 退化为对 0 的 PSR
    d1 = deflated_sharpe(0.1, 1, 500)
    assert abs(d1["dsr"] - probabilistic_sharpe(0.1, 0.0, 500)) < 1e-12
    print(f"formula_magnitudes OK (E[max z]: N=10 {emax10:.3f}, "
          f"N=100 {emax100:.3f} vs MC {mc:.3f})")


def test_ledger_bridge():
    """台账桥接: N 从 research_ledger.csv 读, 缺文件=0, 按因子过滤, DSR一步到位。"""
    from spencer.discipline.ledger import log_run
    with tempfile.TemporaryDirectory() as tmp:
        ledger = Path(tmp) / "research_ledger.csv"
        assert trial_n_from_ledger(ledger) == 0, "台账不存在应返回0"
        for i in range(5):
            log_run(ledger, {"factor": "mom", "ic_mean": 0.01 * i}, note="t")
        log_run(ledger, {"factor": "rev", "ic_mean": 0.02})
        assert trial_n_from_ledger(ledger) == 6
        assert trial_n_from_ledger(ledger, factor="mom") == 5
        d = dsr_from_ledger(0.15, T=1250, ledger_path=ledger)
        assert d["n_trials_from_ledger"] == 6
        # N=6 的缩水必须比 N=1 严, 比 N=100 松 (单调惩罚)
        assert deflated_sharpe(0.15, 1, 1250)["dsr"] > d["dsr"] \
            > deflated_sharpe(0.15, 100, 1250)["dsr"]
        # 台账为空时按 N=1 兜底(眼前这次也是一次试验)
        d0 = dsr_from_ledger(0.15, T=1250, ledger_path=Path(tmp) / "none.csv")
        assert d0["n_trials_from_ledger"] == 1
    print("ledger_bridge OK")


if __name__ == "__main__":
    test_dsr_kills_noise_champion()
    test_dsr_true_signal_survives()
    test_pbo_random_near_half()
    test_pbo_skill_low()
    test_formula_magnitudes()
    test_ledger_bridge()
    print("ALL GREEN (m4_stats)")
