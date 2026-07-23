"""M3 评估层测试: Newey-West 修正 t 值 + IC 衰减曲线 (horizon_profile)。

跑法: python tests/test_m3_eval.py  (零依赖, 合成数据, 不碰网络)

覆盖:
  1. 正自相关 IC 序列上 NW-t < 朴素 t (修正真的在压缩虚高的 t);
  2. 白噪声上 NW-t ≈ 朴素 t (没自相关就几乎不该动), lag=0 时精确相等;
  3. ic_summary 向后兼容: 旧字段原样保留, 新增 t_stat_nw;
  4. horizon_profile 在"只在 h=5 有信号"的构造数据上形状正确、峰在 5;
  5. ic_decay_plot 落盘成功 (Agg 后端);
  6. run_panel 原签名可跑 (向后兼容冒烟);
  7. 退化输入 (常数序列/超短序列) 返回 NaN 不返回假 t。
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spencer.eval.panel import (forward_returns, horizon_profile, ic_decay_plot,
                                ic_series, ic_summary, newey_west_tstat, run_panel)


class FakeStore:
    def __init__(self, frames):
        self.frames = frames

    def load(self, field):
        return self.frames[field]

    def end_date(self):
        return self.frames["close"].index[-1]


def _ar1_series(n=500, phi=0.7, mu=0.03, sigma=0.02, seed=0) -> pd.Series:
    """构造均值为正、AR(1) 正自相关的合成 IC 序列。"""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    eps = rng.normal(0, sigma, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return pd.Series(x + mu, index=pd.bdate_range("2020-01-01", periods=n))


def test_nw_shrinks_t_on_autocorr():
    ic = _ar1_series()
    n = len(ic)
    naive_t = float(ic.mean() / ic.std() * np.sqrt(n))
    nw_t = newey_west_tstat(ic, lag=10)
    assert nw_t > 0, f"正均值序列 NW-t 应为正, got {nw_t}"
    assert nw_t < naive_t, f"正自相关下 NW-t 应小于朴素 t: {nw_t} vs {naive_t}"
    # 修正幅度要实打实: phi=0.7 的长程方差放大数倍, t 至少砍 30%
    assert nw_t < 0.7 * naive_t, f"NW 修正幅度过小: {nw_t} vs {naive_t}"
    print(f"nw_shrinks_t OK (naive {naive_t:.2f} -> nw {nw_t:.2f})")


def test_nw_matches_naive_on_white_noise():
    rng = np.random.default_rng(42)
    ic = pd.Series(rng.normal(0.02, 0.05, 2000),
                   index=pd.bdate_range("2018-01-01", periods=2000))
    naive_t = float(ic.mean() / ic.std() * np.sqrt(len(ic)))
    # lag=0 时 NW 退化为朴素 t (仅 ddof 差异, 2000 样本下可忽略)
    nw0 = newey_west_tstat(ic, lag=0)
    assert abs(nw0 - naive_t) / naive_t < 1e-3, f"lag=0 应退化为朴素 t: {nw0} vs {naive_t}"
    # 白噪声上 lag=5 的修正应很小
    nw5 = newey_west_tstat(ic, lag=5)
    assert 0.8 < nw5 / naive_t < 1.2, f"白噪声上 NW 不应大动: {nw5} vs {naive_t}"
    print(f"nw_white_noise OK (naive {naive_t:.2f}, lag0 {nw0:.2f}, lag5 {nw5:.2f})")


def test_nw_degenerate_inputs():
    idx = pd.bdate_range("2024-01-01", periods=50)
    const = pd.Series(0.5, index=idx)
    assert np.isnan(newey_west_tstat(const, lag=5)), "常数序列应返回 NaN"
    short = pd.Series([0.1], index=idx[:1])
    assert np.isnan(newey_west_tstat(short, lag=5)), "单点序列应返回 NaN"
    # lag 超过样本长度不允许越界崩溃
    tiny = pd.Series([0.1, 0.2, 0.15], index=idx[:3])
    t = newey_west_tstat(tiny, lag=100)
    assert np.isfinite(t) or np.isnan(t)
    print("nw_degenerate OK")


def test_ic_summary_backward_compat():
    ic = _ar1_series(n=400, seed=1)
    h = 5
    summ = ic_summary(ic, horizon=h)
    # 旧字段一个不少
    for key in ("ic_mean", "ic_ir_daily", "t_stat_conservative", "n_days"):
        assert key in summ, f"旧字段 {key} 丢失"
    # 旧字段公式原样: mean/std*sqrt(n/h)
    n = len(ic)
    expect = round(float(ic.mean() / ic.std() * np.sqrt(max(n / h, 1.0))), 2)
    assert summ["t_stat_conservative"] == expect, "t_stat_conservative 公式被改动"
    # 新字段存在且与直接调用一致
    assert "t_stat_nw" in summ
    assert summ["t_stat_nw"] == round(newey_west_tstat(ic, lag=h), 2)
    print("ic_summary_compat OK")


def _make_h5_signal_panel(t=300, n=100, beta=0.01, sigma=0.02, seed=11):
    """构造"只在 h=5 有信号"的合成面板。

    因子 F_T (逐日独立的截面正态) 只预测 T+5→T+6 那一天的收益:
        r_t = 噪声 + beta * F_{t-6}
    则按 forward_returns 的对齐 (fwd(h) = T+1→T+1+h 收益):
      - fwd(1) 覆盖 T+1→T+2, 不含信号日 → IC≈0;
      - fwd(5) 覆盖 T+1→T+6, 恰好含信号日 → IC 峰值;
      - fwd(10)/fwd(20) 窗口嵌套含同一信号日但被更多噪声日稀释 → IC 递减。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=t)
    codes = [f"s{i:03d}" for i in range(n)]
    F = pd.DataFrame(rng.normal(size=(t, n)), index=dates, columns=codes)
    noise = rng.normal(0, sigma, (t, n))
    r = pd.DataFrame(noise, index=dates, columns=codes) + beta * F.shift(6).fillna(0.0)
    px = 100 * np.exp(r.cumsum())
    store = FakeStore({"adj_close": px, "close": px})
    return F, store


def test_horizon_profile_shape_and_peak():
    F, store = _make_h5_signal_panel()
    horizons = [1, 5, 10, 20]
    prof = horizon_profile(F, store, horizons=horizons, min_names=30)

    # 形状: index=horizons, 三列定序
    assert list(prof.index) == horizons, f"index 应为 {horizons}, got {list(prof.index)}"
    assert list(prof.columns) == ["ic_mean", "icir", "t_nw"], f"列错: {list(prof.columns)}"
    assert prof.shape == (4, 3)
    assert prof.index.name == "horizon"

    # 信号结构: h=1 无信号, 峰在 h=5, 之后被稀释递减
    assert abs(prof.loc[1, "ic_mean"]) < 0.03, f"h=1 应无信号, got {prof.loc[1, 'ic_mean']}"
    assert prof.loc[5, "ic_mean"] == prof["ic_mean"].max(), "峰应在 h=5"
    assert prof.loc[5, "ic_mean"] > 0.10, f"h=5 信号读数过弱: {prof.loc[5, 'ic_mean']}"
    assert prof.loc[5, "ic_mean"] > prof.loc[10, "ic_mean"] > prof.loc[20, "ic_mean"], \
        "嵌套窗口稀释应使 IC 随 horizon 递减"
    assert prof.loc[5, "t_nw"] > 3, f"h=5 的 NW-t 应显著: {prof.loc[5, 't_nw']}"
    print(f"horizon_profile OK (ic: " +
          " ".join(f"h{h}={prof.loc[h, 'ic_mean']:+.3f}" for h in horizons) + ")")


def test_ic_decay_plot_writes_file():
    F, store = _make_h5_signal_panel(t=150, n=60)
    prof = horizon_profile(F, store, horizons=[1, 5, 10], min_names=20)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ic_decay_test.png"
        ic_decay_plot(prof, "synthetic_h5", path)
        assert path.exists() and path.stat().st_size > 0, "衰减图未落盘"
    print("ic_decay_plot OK")


def test_run_panel_backward_compat_smoke():
    """run_panel 原签名原样可跑, 结果 dict 新增 t_stat_nw 且旧键不丢。"""
    F, store = _make_h5_signal_panel(t=200, n=60)
    with tempfile.TemporaryDirectory() as td:
        res = run_panel("smoke", F, store, horizon=5, q=5,
                        min_names=20, outdir=Path(td))
        assert (Path(td) / "panel_smoke.png").exists()
        assert (Path(td) / "yearly_smoke.csv").exists()
    for key in ("factor", "ic_mean", "ic_ir_daily", "t_stat_conservative",
                "n_days", "rank_autocorr_5d", "yearly_all_positive",
                "ls_ann_ret_gross"):
        assert key in res, f"run_panel 旧键 {key} 丢失"
    assert "t_stat_nw" in res, "run_panel 结果应含 t_stat_nw"
    print("run_panel_smoke OK")


if __name__ == "__main__":
    test_nw_shrinks_t_on_autocorr()
    test_nw_matches_naive_on_white_noise()
    test_nw_degenerate_inputs()
    test_ic_summary_backward_compat()
    test_horizon_profile_shape_and_peak()
    test_ic_decay_plot_writes_file()
    test_run_panel_backward_compat_smoke()
    print("ALL GREEN (m3_eval)")
