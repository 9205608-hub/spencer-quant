"""M5 回测层补齐测试: 可成交性掩码 + 冲击成本项 + 向后兼容回归锚。

跑法: python tests/test_m5_bt.py  (零依赖, 不碰网络, 全部合成数据)

覆盖:
  1. tradability_masks 一字板判定(主板/ST/创业板改革前后/科创板/非一字/NaN);
  2. 不传新参数时与 v0.1 实现逐位一致(参考实现内嵌本文件当回归锚);
  3. 想买而执行日不可买 → 确实没进持仓(仓位摊给可买的);
  4. 想卖而执行日不可卖 → 滞留到首个可卖执行日才卖出, 补卖成本计在该信号日;
  5. 执行日对表: 掩码按 T+1(交易实际发生日)取值 —— T 日正常/T+1 一字必须
     拦住, T 日一字/T+1 打开不得多余排除; 末行执行日越界默认可成交;
  6. impact_bps_extra 与 cost_bps 的加和等价性 + 成本/冲击单调性(两条路径)。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spencer.backtest.layered import layered_backtest, tradability_masks


class FakeStore:
    def __init__(self, frames):
        self.frames = frames

    def load(self, field):
        return self.frames[field]


# ---------- 1. 一字板可成交性 ----------

def test_tradability_masks():
    dates = pd.to_datetime(["2020-08-18", "2020-08-19", "2020-08-20", "2020-08-21",
                            "2020-08-24", "2020-08-25", "2020-08-26"])
    d0, d1, d2, d3, d4, d5, d6 = dates
    codes = ["sh.600001", "sz.000002", "sz.300001", "sh.688001"]  # 主板/主板ST/创业板/科创板

    close = pd.DataFrame(10.0, index=dates, columns=codes)
    open_ = pd.DataFrame(10.0, index=dates, columns=codes)
    high = pd.DataFrame(10.0, index=dates, columns=codes)
    low = pd.DataFrame(10.0, index=dates, columns=codes)
    is_st = pd.DataFrame(0, index=dates, columns=codes)
    is_st["sz.000002"] = 1

    def set_ohl(dt, code, o, h, l):
        open_.loc[dt, code], high.loc[dt, code], low.loc[dt, code] = o, h, l

    set_ohl(d0, "sh.600001", 11.0, 11.0, 11.0)   # 首日无昨收 → 判不了 → 默认可交易
    set_ohl(d1, "sh.600001", 11.0, 11.0, 11.0)   # +10% 一字涨停
    set_ohl(d1, "sz.000002", 10.49, 10.49, 10.49)  # ST +4.9% > 4.8% → 封
    set_ohl(d1, "sz.300001", 10.49, 10.49, 10.49)  # 非ST +4.9% < 9.8% → 不封
    set_ohl(d2, "sh.600001", 11.0, 11.1, 11.0)   # +10% 但盘中打开(非一字) → 可买
    set_ohl(d2, "sz.300001", 11.5, 11.5, 11.5)   # 创业板改革前 +15% > 9.8% → 封
    set_ohl(d2, "sh.688001", 11.5, 11.5, 11.5)   # 科创板改革前(已知近似: 统一2020-08边界) → 封
    set_ohl(d4, "sz.300001", 11.5, 11.5, 11.5)   # 改革后 +15% < 19.8% → 可买
    set_ohl(d4, "sh.688001", 12.0, 12.0, 12.0)   # +20% >= 19.8% → 封
    set_ohl(d5, "sh.600001", 9.0, 9.0, 9.0)      # -10% 一字跌停 → 不可卖
    set_ohl(d5, "sz.000002", 9.51, 9.51, 9.51)   # ST -4.9% 一字跌停 → 不可卖
    set_ohl(d6, "sz.000002", 10.49, 10.49, 10.49)  # ST 阈值不随改革日期变
    for df in (open_, high, low, close):          # 缺数据 → 默认可交易
        df.loc[d6, "sz.300001"] = np.nan

    store = FakeStore({"open": open_, "high": high, "low": low,
                       "close": close, "is_st": is_st})
    cb, cs = tradability_masks(store)

    assert cb.loc[d0, "sh.600001"], "首日无昨收应默认可交易"
    assert not cb.loc[d1, "sh.600001"] and cs.loc[d1, "sh.600001"], "一字涨停: 不可买/可卖"
    assert not cb.loc[d1, "sz.000002"], "ST 4.9% 应判一字涨停"
    assert cb.loc[d1, "sz.300001"], "非ST 4.9% 不应封板"
    assert cb.loc[d2, "sh.600001"], "盘中打开的涨停应视为可买"
    assert not cb.loc[d2, "sz.300001"], "创业板改革前 15% 应判封板"
    assert not cb.loc[d2, "sh.688001"], "科创板按统一 2020-08 边界(已知近似)改革前判封"
    assert cb.loc[d4, "sz.300001"], "创业板改革后 15% 不应封板"
    assert not cb.loc[d4, "sh.688001"], "改革后 20% 应判封板"
    assert cb.loc[d5, "sh.600001"] and not cs.loc[d5, "sh.600001"], "一字跌停: 可买/不可卖"
    assert not cs.loc[d5, "sz.000002"], "ST 一字跌停应不可卖"
    assert not cb.loc[d6, "sz.000002"], "ST 阈值与改革日期无关"
    assert cb.loc[d6, "sz.300001"] and cs.loc[d6, "sz.300001"], "NaN 行情应默认可交易"
    assert cb.loc[d3].all() and cs.loc[d3].all(), "无事日应全可交易"
    print("tradability_masks OK")


# ---------- 2. 向后兼容: 与 v0.1 参考实现逐位一致 ----------

def _layered_backtest_v01(factor, fwd1, q=5, rebal_days=5, cost_bps=15.0):
    """v0.1 原实现逐字拷贝, 当回归锚 —— 新代码不传掩码时必须与它逐位一致。"""
    cost = cost_bps / 1e4
    idx = factor.index.intersection(fwd1.index)
    factor, fwd1 = factor.loc[idx], fwd1.loc[idx]
    pct = factor.rank(axis=1, pct=True)
    rebal_dates = idx[::rebal_days]

    def leg(select_mask_at):
        rets, turns = {}, {}
        members = set()
        next_i = 0
        for dt in idx:
            if next_i < len(rebal_dates) and dt == rebal_dates[next_i]:
                row = pct.loc[dt].dropna()
                new = set(row[select_mask_at(row)].index)
                if new:
                    turns[dt] = 1.0 if not members else 1 - len(members & new) / len(new)
                    members = new
                next_i += 1
            if members:
                rets[dt] = fwd1.loc[dt, list(members)].mean()
        turns = pd.Series(turns)
        r = pd.Series(rets).sub(2 * cost * turns, fill_value=0.0)
        return r, turns

    top_r, top_t = leg(lambda row: row > 1 - 1 / q)
    bot_r, _ = leg(lambda row: row <= 1 / q)
    ls = (top_r - bot_r).dropna()

    def metrics(r):
        ann = r.mean() * 252
        vol = r.std() * np.sqrt(252)
        cum = r.cumsum()
        mdd = (cum - cum.cummax()).min()
        return {"ann_ret": round(float(ann), 4), "sharpe": round(float(ann / vol), 2),
                "max_dd": round(float(mdd), 3)}

    return {"long_net": metrics(top_r.dropna()), "ls_net": metrics(ls),
            "avg_turnover_per_rebal": round(float(top_t.iloc[1:].mean()), 3) if len(top_t) > 1 else None,
            "n_rebalances": len(top_t), "ls_series": ls}


def _rand_panel(t=120, n=40, seed=11):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=t)
    codes = [f"sh.6{i:05d}" for i in range(n)]
    factor = pd.DataFrame(rng.normal(size=(t, n)), index=dates, columns=codes)
    fwd1 = pd.DataFrame(rng.normal(0, 0.02, size=(t, n)), index=dates, columns=codes)
    return factor, fwd1


def test_backcompat_exact():
    # 多种子×多参数扫: 逐位一致对集合迭代顺序敏感(mean 求和顺序),
    # 单一种子/单一规模会漏检 —— 实测 held=set(target) 拷贝就曾只在
    # 部分种子上产生末位比特漂移。
    for seed, t, n in ((11, 120, 40), (1, 150, 60), (2026, 90, 55)):
        for q, rd, c in ((5, 5, 15.0), (3, 10, 30.0), (10, 1, 0.0)):
            factor, fwd1 = _rand_panel(t=t, n=n, seed=seed)
            new = layered_backtest(factor, fwd1, q=q, rebal_days=rd, cost_bps=c)
            ref = _layered_backtest_v01(factor, fwd1, q=q, rebal_days=rd, cost_bps=c)
            for key in ("long_net", "ls_net", "avg_turnover_per_rebal", "n_rebalances"):
                assert new[key] == ref[key], \
                    f"向后兼容破裂 seed={seed} q={q} rd={rd} c={c}: {key} {new[key]} != {ref[key]}"
            pd.testing.assert_series_equal(new["ls_series"], ref["ls_series"], check_exact=True)
    print("backcompat_exact OK (无掩码路径与 v0.1 逐位一致, 3种子×3参数)")


# ---------- 3/4. 一字板场景: 买不进没进仓 / 卖不出滞留 ----------

def _four_stock_panel(fwd_vals, factor_rows):
    idx = pd.bdate_range("2024-01-01", periods=15)
    codes = ["sh.600001", "sh.600002", "sh.600003", "sh.600004"]  # A B C D
    factor = pd.DataFrame(factor_rows, index=idx, columns=codes)
    fwd1 = pd.DataFrame(np.tile(fwd_vals, (15, 1)), index=idx, columns=codes)
    return idx, codes, factor, fwd1


def test_cannot_buy_skipped():
    # 因子恒定: top={A,B} bottom={C,D}; fwd1: A=.02 B=.01 C=D=0 → 多头收益即持仓指纹
    # 掩码按执行日取值: idx[0] 信号的买入发生在 idx[1] 收盘 → 封 idx[1]
    idx, codes, factor, fwd1 = _four_stock_panel(
        fwd_vals=[0.02, 0.01, 0.0, 0.0],
        factor_rows=np.tile([4.0, 3.0, 2.0, 1.0], (15, 1)))
    can_buy = pd.DataFrame(True, index=idx, columns=codes)
    can_buy.loc[idx[1], "sh.600001"] = False   # 首个调仓日的执行日 A 一字涨停买不进

    res = layered_backtest(factor, fwd1, q=2, rebal_days=5, cost_bps=0, can_buy=can_buy)
    # 第一期只持 B(0.01, 若 A 混进来会是 0.015); 第二个调仓日(执行日 idx[6] 可买) → 0.015
    expect = [0.01] * 5 + [0.015] * 10
    assert np.allclose(res["long_series"].values, expect), \
        f"买不进的票进了持仓? {res['long_series'].values}"
    # 空头腿 {C,D} 收益恒 0 → ls 应等于多头腿
    assert np.allclose(res["ls_series"].values, expect)

    # 对照: 不传掩码时 A 第一天就该在仓里
    res0 = layered_backtest(factor, fwd1, q=2, rebal_days=5, cost_bps=0)
    assert np.allclose(res0["long_series"].values, [0.015] * 15)
    print("cannot_buy_skipped OK")


def test_execution_day_alignment():
    # 30问#3 要杀的核心场景: 信号日 T 正常、执行日 T+1 一字涨停 → 必须拦住;
    # 反向: T 一字、T+1 打开 → 不得多余排除。按信号日取值的语义两个都答错,
    # 这里把执行日语义钉死成回归锚。
    idx, codes, factor, fwd1 = _four_stock_panel(
        fwd_vals=[0.02, 0.01, 0.0, 0.0],
        factor_rows=np.tile([4.0, 3.0, 2.0, 1.0], (15, 1)))

    # (a) 只封执行日 idx[1](信号日 idx[0] 正常): A 必须没被买进
    m = pd.DataFrame(True, index=idx, columns=codes)
    m.loc[idx[1], "sh.600001"] = False
    res = layered_backtest(factor, fwd1, q=2, rebal_days=5, cost_bps=0, can_buy=m)
    assert np.allclose(res["long_series"].iloc[:5], [0.01] * 5), \
        "执行日一字涨停的票被买入 —— 一天错位漏网回归"

    # (b) 只封信号日 idx[0](执行日 idx[1] 已打开): A 应正常买入
    m2 = pd.DataFrame(True, index=idx, columns=codes)
    m2.loc[idx[0], "sh.600001"] = False
    res2 = layered_backtest(factor, fwd1, q=2, rebal_days=5, cost_bps=0, can_buy=m2)
    assert np.allclose(res2["long_series"].values, [0.015] * 15), \
        "信号日一字但执行日打开的票被多余排除"
    print("execution_day_alignment OK")


def test_cannot_sell_forced_hold():
    # 前 5 天 top={A,B}; 第 2 个调仓日(信号 idx[5], 执行 idx[6])起因子翻转 →
    # top={C,D}; A 在执行日 idx[6..8] 一字跌停卖不出 → 信号日 idx[8](执行日
    # idx[9] 可卖)才卖得掉, 滞留期收益计在信号日 idx[5..7]
    rows = np.vstack([np.tile([4.0, 3.0, 2.0, 1.0], (5, 1)),
                      np.tile([1.0, 2.0, 3.0, 4.0], (10, 1))])
    idx, codes, factor, fwd1 = _four_stock_panel(
        fwd_vals=[0.01, 0.02, 0.03, 0.05], factor_rows=rows)
    can_sell = pd.DataFrame(True, index=idx, columns=codes)
    can_sell.loc[idx[6]:idx[8], "sh.600001"] = False

    res = layered_backtest(factor, fwd1, q=2, rebal_days=5, cost_bps=0, can_sell=can_sell)
    top = [0.015] * 5 + [(0.01 + 0.03 + 0.05) / 3] * 3 + [0.04] * 7   # 滞留期 {A,C,D}
    bot = [0.04] * 5 + [0.015] * 10
    assert np.allclose(res["long_series"].values, top), \
        f"卖不出的票没滞留在持仓? {res['long_series'].values}"
    assert np.allclose(res["ls_series"].values, np.array(top) - np.array(bot))

    # 补卖成本计在补卖的信号日(idx[8]): 卖出 A 占持仓 {A,C,D} 的 1/3
    res_c = layered_backtest(factor, fwd1, q=2, rebal_days=5, cost_bps=100, can_sell=can_sell)
    delta = res["ls_series"] - res_c["ls_series"]
    assert abs(delta.loc[idx[8]] - 0.01 / 3) < 1e-12, f"补卖成本错日/错额: {delta.loc[idx[8]]}"
    assert abs(delta.loc[idx[6]]) < 1e-15 and abs(delta.loc[idx[7]]) < 1e-15, "无交易日不应计成本"
    print("cannot_sell_forced_hold OK")


def test_pending_sell_end_of_data():
    # 末行的执行日越过数据末端 → 无从判定 → 按"掩码只做减法"默认可成交:
    # A 从执行日 idx[6] 起一路封死到末端, 滞留到最后一个信号日 idx[14]
    # (执行日越界)才按默认可成交卖出
    rows = np.vstack([np.tile([4.0, 3.0, 2.0, 1.0], (5, 1)),
                      np.tile([1.0, 2.0, 3.0, 4.0], (10, 1))])
    idx, codes, factor, fwd1 = _four_stock_panel(
        fwd_vals=[0.01, 0.02, 0.03, 0.05], factor_rows=rows)
    can_sell = pd.DataFrame(True, index=idx, columns=codes)
    can_sell.loc[idx[6]:, "sh.600001"] = False

    res = layered_backtest(factor, fwd1, q=2, rebal_days=5, cost_bps=0, can_sell=can_sell)
    top = [0.015] * 5 + [(0.01 + 0.03 + 0.05) / 3] * 9 + [0.04]
    assert np.allclose(res["long_series"].values, top), \
        f"末行执行日越界应默认可成交: {res['long_series'].values}"
    print("pending_sell_end_of_data OK")


# ---------- 5. 冲击项与成本单调性 ----------

def test_impact_and_cost_monotonic():
    factor, fwd1 = _rand_panel(seed=23)

    # (a) 加和等价: impact 与 cost_bps 同乘同一成交比例 → cost 20+impact 30 == cost 50
    r1 = layered_backtest(factor, fwd1, cost_bps=20, impact_bps_extra=30)
    r2 = layered_backtest(factor, fwd1, cost_bps=50)
    assert r1["ls_net"] == r2["ls_net"] and r1["long_net"] == r2["long_net"]
    pd.testing.assert_series_equal(r1["ls_series"], r2["ls_series"], check_exact=True)

    # (b) 冲击单调: 作弊因子(高换手) impact 越大净收益越低。
    # 注意断言在 long_net: v0.1 口径 ls=top_net-bot_net 里空头腿成本符号为正,
    # ls 对成本不保证单调(模块 docstring 已披露), 单调性只对多头腿良定义。
    cheat = fwd1.copy()
    anns = [layered_backtest(cheat, fwd1, cost_bps=15, impact_bps_extra=x)["long_net"]["ann_ret"]
            for x in (0, 50, 200)]
    assert anns[0] > anns[1] > anns[2], f"impact 应单调侵蚀多头净收益: {anns}"

    # (c) 掩码路径下成本/冲击单调性保持(全 True 掩码, 只换成本参数)
    all_true = pd.DataFrame(True, index=factor.index, columns=factor.columns)
    lo = layered_backtest(cheat, fwd1, cost_bps=0, can_buy=all_true, can_sell=all_true)
    hi = layered_backtest(cheat, fwd1, cost_bps=50, can_buy=all_true, can_sell=all_true)
    im = layered_backtest(cheat, fwd1, cost_bps=50, impact_bps_extra=100,
                          can_buy=all_true, can_sell=all_true)
    assert lo["long_net"]["ann_ret"] > hi["long_net"]["ann_ret"] > im["long_net"]["ann_ret"], \
        "掩码路径成本单调性破裂"
    print("impact_and_cost_monotonic OK")


if __name__ == "__main__":
    test_tradability_masks()
    test_backcompat_exact()
    test_cannot_buy_skipped()
    test_execution_day_alignment()
    test_cannot_sell_forced_hold()
    test_pending_sell_end_of_data()
    test_impact_and_cost_monotonic()
    print("ALL GREEN (m5_bt)")
