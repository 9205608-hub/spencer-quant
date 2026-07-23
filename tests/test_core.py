"""核心断言: 算子正确性 + 前视偏差闸门 + IC 管道自洽。

跑法: python tests/test_core.py  (零依赖, 不碰网络)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spencer.factor.ops import cs_rank, neutralize, winsorize_mad, zscore
from spencer.eval.panel import forward_returns, ic_series


class FakeStore:
    def __init__(self, frames):
        self.frames = frames

    def load(self, field):
        return self.frames[field]

    def end_date(self):
        return self.frames["close"].index[-1]


def make_panel(t=120, n=80, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=t)
    codes = [f"s{i:03d}" for i in range(n)]
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (t, n)), axis=0)),
                      index=dates, columns=codes)
    return dates, codes, px


def test_ops():
    dates, codes, px = make_panel()
    f = px.pct_change(fill_method=None)

    z = zscore(f)
    assert z.iloc[5:].mean(axis=1).abs().max() < 1e-10, "zscore 截面均值应为0"
    assert (z.iloc[5:].std(axis=1) - 1).abs().max() < 1e-10, "zscore 截面std应为1"

    r = cs_rank(f)
    assert r.iloc[5:].max(axis=1).eq(1.0).all(), "cs_rank 最大值应为1"

    w = winsorize_mad(f.copy())
    assert w.iloc[5:].abs().max().max() <= f.iloc[5:].abs().max().max() + 1e-12

    size = np.log(px)
    resid = neutralize(f, size)
    for dt in resid.index[5:10]:
        v = resid.loc[dt].dropna()
        s = size.loc[dt, v.index]
        corr = np.corrcoef(v.values, s.values)[0, 1]
        assert abs(corr) < 1e-8, f"中性化残差与风格相关性应为0, got {corr}"
    print("ops OK")


def test_forward_returns_no_lookahead():
    dates, codes, px = make_panel()
    store = FakeStore({"adj_close": px, "close": px})
    fwd = forward_returns(store, horizon=5)
    # 手工核对一格: T 日的 fwd = adj[T+6]/adj[T+1] - 1
    t = 10
    expect = px.iloc[t + 6, 0] / px.iloc[t + 1, 0] - 1
    assert abs(fwd.iloc[t, 0] - expect) < 1e-12, "前瞻收益对齐错位"
    # 末端 horizon+1 天必须是 NaN (未来还没发生)
    assert fwd.iloc[-(5 + 1):, 0].isna().all(), "末端未来收益应为NaN"
    print("forward_returns OK")


def test_ic_pipeline():
    dates, codes, px = make_panel()
    store = FakeStore({"adj_close": px, "close": px})
    fwd = forward_returns(store, horizon=5)
    # 用未来收益本身当因子 → IC 应≈1 (管道自洽性上限测试)
    ic = ic_series(fwd, fwd, min_names=30)
    assert ic.min() > 0.999, "自身IC应为1"
    # 纯噪声因子 → |IC均值| 应接近 0
    rng = np.random.default_rng(1)
    noise = pd.DataFrame(rng.normal(size=px.shape), index=px.index, columns=px.columns)
    ic2 = ic_series(noise, fwd, min_names=30)
    assert abs(ic2.mean()) < 0.05, f"噪声因子IC均值应≈0, got {ic2.mean()}"
    print("ic_pipeline OK")


def test_residualize():
    from spencer.risk.neutral import residualize
    dates, codes, px = make_panel()
    rng = np.random.default_rng(3)
    f = px.pct_change(fill_method=None)
    styles = {"size": np.log(px),
              "vol": f.rolling(20, min_periods=10).std()}
    industry = pd.Series([f"ind{i % 6}" for i in range(len(codes))], index=codes)

    resid = residualize(f, styles, industry, min_names=30)
    from spencer.factor.ops import winsorize_mad, zscore
    prep = {k: zscore(winsorize_mad(v)) for k, v in styles.items()}
    for dt in resid.index[40:44]:
        v = resid.loc[dt].dropna()
        for k in styles:
            s = prep[k].loc[dt, v.index]
            corr = np.corrcoef(v.values, s.values)[0, 1]
            assert abs(corr) < 1e-6, f"残差与风格{k}相关性应为0, got {corr}"
        for g, grp in v.groupby(industry.reindex(v.index)):
            assert abs(grp.mean()) < 1e-8, f"行业{g}组内残差均值应为0"
    print("residualize OK")


def test_backtest():
    from spencer.backtest.layered import layered_backtest
    dates, codes, px = make_panel(t=200)
    store = FakeStore({"adj_close": px, "close": px})
    from spencer.eval.panel import forward_return_1d
    fwd1 = forward_return_1d(store)
    # 用未来收益当因子(作弊上限): 净多空必须显著为正, 且成本越高净值越低
    cheat = fwd1.copy()
    lo = layered_backtest(cheat, fwd1, q=5, rebal_days=5, cost_bps=0)
    hi = layered_backtest(cheat, fwd1, q=5, rebal_days=5, cost_bps=50)
    assert lo["ls_net"]["ann_ret"] > 0.5, "作弊因子净多空应大幅为正"
    assert hi["ls_net"]["ann_ret"] < lo["ls_net"]["ann_ret"], "成本应单调侵蚀净收益"
    print("backtest OK")


def test_strategy():
    from spencer.strategy.composite import equal_weight, icir_weight
    from spencer.strategy.portfolio import topn_buffer_weights, turnover_series
    dates, codes, px = make_panel(t=200)
    store = FakeStore({"adj_close": px, "close": px})
    from spencer.eval.panel import forward_returns
    fwd = forward_returns(store, 5)
    f1 = px.pct_change(5, fill_method=None)
    f2 = -px.pct_change(20, fill_method=None)

    comp = equal_weight({"a": f1, "b": f2})
    assert comp.max().max() <= 1.0 + 1e-9 and comp.min().min() >= 0.0
    # 逐格分母: f1 有值 f2 缺值的格子, 合成应仍有值(min_count=1)
    comp1 = equal_weight({"a": f1, "b": f2.mask(f2.notna())}, min_count=1)
    assert comp1.iloc[30:].notna().sum().sum() > 0

    ic_w = icir_weight({"a": f1, "b": f2}, fwd, horizon=5, window=60, min_periods=30)
    assert ic_w.shape == f1.shape

    w_nobuf = topn_buffer_weights(comp, top_n=10, buffer_n=10, rebal_days=5)
    w_buf = topn_buffer_weights(comp, top_n=10, buffer_n=20, rebal_days=5)
    s = w_buf.iloc[20:].sum(axis=1)
    assert ((s - 1).abs() < 1e-9).all(), "权重行和应为1"
    to_nobuf = turnover_series(w_nobuf).sum()
    to_buf = turnover_series(w_buf).sum()
    assert to_buf <= to_nobuf + 1e-9, f"缓冲区应降低换手 {to_buf} vs {to_nobuf}"
    print(f"strategy OK (turnover buffer {to_buf:.1f} <= nobuffer {to_nobuf:.1f})")


if __name__ == "__main__":
    test_ops()
    test_forward_returns_no_lookahead()
    test_ic_pipeline()
    test_residualize()
    test_backtest()
    test_strategy()
    print("ALL GREEN")
