"""优化器回测的记账与形状断言(微型合成世界, 秒级)。

跑法: python3 tests/test_opt_bt.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spencer.backtest.opt_bt import optimizer_backtest
from spencer.risk.covariance import style_factor_returns


def test_opt_bt_smoke():
    rng = np.random.default_rng(2)
    t, n = 260, 40
    dates = pd.bdate_range("2024-01-01", periods=t)
    codes = [f"s{i:03d}" for i in range(n)]
    b = rng.normal(0, 1, n)
    f = rng.normal(0, 0.004, t)
    fwd1 = pd.DataFrame(np.outer(f, b) + rng.normal(0, 0.01, (t, n)),
                        index=dates, columns=codes)
    styles = {"sty": pd.DataFrame(np.tile(b, (t, 1)), index=dates, columns=codes)}
    sr, resid = style_factor_returns(fwd1, styles, None, min_names=20)
    signal = pd.DataFrame(rng.normal(size=(t, n)), index=dates, columns=codes)
    uni = pd.DataFrame(True, index=dates, columns=codes)

    r = optimizer_backtest(signal, fwd1, uni, styles, sr, resid,
                           rebal_days=20, lam=20.0, tau=0.0,
                           cap=0.2, bound=0.5, warmup_obs=60, min_members=20,
                           max_iter=800, verbose_every=0)
    assert r["n_rebalances"] >= 5, r
    w = r["weights"]
    active_days = w.abs().sum(axis=1) > 0
    sums = w.loc[active_days].sum(axis=1)
    assert (sums - 1).abs().max() < 1e-4, "活跃日权重应归一"
    assert (w.values <= 0.2 + 1e-6).all(), "个股上限被突破"
    assert len(r["net_series"]) > 100
    # τ 惩罚应单调降低换手
    r2 = optimizer_backtest(signal, fwd1, uni, styles, sr, resid,
                            rebal_days=20, lam=20.0, tau=0.01,
                            cap=0.2, bound=0.5, warmup_obs=60, min_members=20,
                            max_iter=800, verbose_every=0)
    assert r2["turnover_ann_oneside"] <= r["turnover_ann_oneside"] + 1e-9, \
        f"τ 应压换手: {r2['turnover_ann_oneside']} vs {r['turnover_ann_oneside']}"
    print(f"opt_bt OK (τ=0 换手 {r['turnover_ann_oneside']}x → τ=0.01 换手 "
          f"{r2['turnover_ann_oneside']}x)")


if __name__ == "__main__":
    test_opt_bt_smoke()
    print("ALL GREEN (opt_bt)")
