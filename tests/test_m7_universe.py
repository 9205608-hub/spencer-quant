"""M7 PIT 宇宙构造器断言 (全合成数据, 不碰网络)。

跑法: python tests/test_m7_universe.py

覆盖任务规定五连 + 附加:
  1. 用未来名单会被抓出 (FUT: 一直有行情但3月末才进名单 → 之前任何一天都不许在;
     NEW: 5月才上市的股票, 之前任何一天都不许在)
  2. 退市股退出 (DEL: 行情消失当日就出宇宙, 名单滞后退出也兜得住)
  3. 预热期屏蔽 (WRM: 名单先接纳, 上市天数不足照样挡, 首次入选日精确到天)
  4. top_n 截断的 PIT 性 (流动性王座换位序列: 只有"过去60日"口径才产生)
  5. 全 NaN 日不崩 (该行全 False, 输出无 NaN、全 bool)
附加: ST 剔除开关生效 / 停牌日剔除 / store 有名单没有的股恒 False /
     名单里 store 没有的幽灵股不崩不出现。

反影子覆盖设计(对抗性复核后加固): close.notna() 退市闸与 is_trading 闸
是两道独立的闸, fixture 若恒有 is_trading == close.notna(), 删掉实现里的
`& close.notna()` 测试照样全绿(变异测试实测)。因此:
  a) 主 fixture 里 DEL 退市后 is_trading 故意保持 1(模拟"交易状态旗标滞后/
     脏数据"), 使 require_trading=True 口径下 DEL 的退出只能由 close 闸完成;
  b) 另设 require_trading=False 专测: 关掉 is_trading 闸, DEL 仍必须当日退出;
  c) top_n 场景的 is_trading 恒为 1, 全 NaN 日的排除同样只能由 close 闸完成。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spencer.data.universe import build_pit_universe


class FakeStore:
    def __init__(self, frames):
        self.frames = frames

    def load(self, field):
        return self.frames[field]

    def end_date(self):
        return self.frames["close"].index[-1]


def month_ends(dates: pd.DatetimeIndex) -> pd.Series:
    """每个月的最后一个面板日 (index=Period, values=Timestamp)。"""
    return pd.Series(dates, index=dates).groupby(dates.to_period("M")).max()


# ---------------------------------------------------------------- 主场景
# 130 个工作日 (2024-01-02 ~ 2024-07-01), min_list_days=30。
# 角色表:
#   AAA  全程在市在名单        → 首次入选 = 预热期满那天
#   FUT  全程有行情, 3月末才进名单 → 3月末前任何一天在 = 用了未来名单
#   DEL  4-10 退市(行情消失), 名单4月末才移除 → 行情闸必须当日拿掉
#   WRM  2-15 上市, 2月末进名单   → 名单先到, 预热闸挡到第30个有效日
#   NEW  5-06 上市, 5月末进名单   → "t+1月才上市"的未来股
#   STX  3-05~3-20 戴 ST         → 窗口内出宇宙
#   SUS  2-20~2-27 停牌(价格仍在) → 停牌日出宇宙
#   ZZZ  有行情但从不在名单       → 恒 False
#   GHOST 名单里有、store 没有    → 不崩、不出现在输出列

def build_main_case():
    dates = pd.bdate_range("2024-01-02", periods=130)
    codes = ["AAA", "FUT", "DEL", "WRM", "NEW", "STX", "SUS", "ZZZ"]
    i = dates.searchsorted
    idx = {
        "wrm_start": i(pd.Timestamp("2024-02-15")),
        "new_start": i(pd.Timestamp("2024-05-06")),
        "del_stop": i(pd.Timestamp("2024-04-10")),
        "st0": i(pd.Timestamp("2024-03-05")), "st1": i(pd.Timestamp("2024-03-21")),
        "sus0": i(pd.Timestamp("2024-02-20")), "sus1": i(pd.Timestamp("2024-02-28")),
    }

    close = pd.DataFrame(10.0, index=dates, columns=codes)
    close.iloc[:idx["wrm_start"], codes.index("WRM")] = np.nan
    close.iloc[:idx["new_start"], codes.index("NEW")] = np.nan
    close.iloc[idx["del_stop"]:, codes.index("DEL")] = np.nan

    amount = close * 1e6
    is_st = pd.DataFrame(0.0, index=dates, columns=codes)
    is_st.iloc[idx["st0"]:idx["st1"], codes.index("STX")] = 1.0
    is_trading = close.notna().astype(float)
    is_trading.iloc[idx["sus0"]:idx["sus1"], codes.index("SUS")] = 0.0  # 停牌但价格仍落盘
    # DEL 退市后 is_trading 故意置 1(脏数据/旗标滞后): 若 is_trading 与
    # close.notna() 完全同步, close 闸会被 is_trading 闸影子覆盖, DEL 断言
    # 就测不到"行情 NaN 当日拿掉"这道双保险本身。
    is_trading.iloc[idx["del_stop"]:, codes.index("DEL")] = 1.0

    me = month_ends(dates)

    def roster(period):
        m = period.month
        r = ["AAA", "STX", "SUS"] + (["DEL"] if m <= 3 else [])
        if m >= 2:
            r.append("WRM")
        if m >= 3:
            r += ["FUT", "GHOST"]
        if m >= 5:
            r.append("NEW")
        return r

    rows = [(d, c) for p, d in me.items() for c in roster(p)]
    membership = pd.DataFrame(rows, columns=["date", "code"])
    store = FakeStore({"close": close, "amount": amount,
                       "is_st": is_st, "is_trading": is_trading})
    return dates, me, close, membership, store, idx


def test_main_semantics():
    dates, me, close, membership, store, idx = build_main_case()
    res = build_pit_universe(membership, store, min_list_days=30,
                             exclude_st=True, top_n_liquidity=None)
    jan_end, feb_end, mar_end, may_end = me.iloc[0], me.iloc[1], me.iloc[2], me.iloc[4]

    # -- 结构: 形状锚定 close, 全 bool 无 NaN, 幽灵股不出现 --
    assert res.index.equals(close.index) and list(res.columns) == list(close.columns)
    assert (res.dtypes == bool).all(), "输出必须全 bool(NaN 不算成员)"
    assert "GHOST" not in res.columns, "store 里没有的名单股不得出现"
    assert not res["ZZZ"].any(), "从不在名单的股票必须恒 False"

    # -- 第一期快照 + 预热期之前: 全体不在 --
    assert res.loc[dates < dates[29]].to_numpy().sum() == 0

    # -- AAA: 首次入选 = max(第一期快照, 第30个有效日) --
    assert res["AAA"].any() and res["AAA"].idxmax() == max(jan_end, dates[29])
    assert res.loc[dates[60], "AAA"] and res.loc[dates[128], "AAA"]

    # -- 测试1a FUT: 3月末快照前任何一天在 = 用了未来名单 --
    assert res.loc[dates < mar_end, "FUT"].sum() == 0, "FUT 提前入选 → 未来名单泄漏"
    assert res.loc[dates >= mar_end, "FUT"].all(), "FUT 3月末起应全程在"

    # -- 测试1b NEW: t+1月才上市的股票, 之前任何一天必须不在;
    #    首次入选 = max(5月末快照, 上市后第30个有效日) --
    exp_new = max(may_end, dates[idx["new_start"] + 30 - 1])
    assert res["NEW"].any() and res["NEW"].idxmax() == exp_new
    assert res.loc[dates < exp_new, "NEW"].sum() == 0
    assert res.loc[dates >= exp_new, "NEW"].all()

    # -- 测试2 DEL: 退市(行情消失)当日起必须不在, 哪怕名单要到4月末才移除 --
    d_stop = dates[idx["del_stop"]]
    assert res.loc[dates >= d_stop, "DEL"].sum() == 0, "退市股行情消失后仍在宇宙"
    assert res.loc[(dates >= dates[29]) & (dates < d_stop), "DEL"].all(), \
        "退市前(过完预热)应全程在"

    # -- 测试3 WRM: 2月末名单已接纳, 但预热闸挡到上市后第30个有效日 --
    exp_wrm = max(feb_end, dates[idx["wrm_start"] + 30 - 1])
    assert res["WRM"].any() and res["WRM"].idxmax() == exp_wrm, \
        f"WRM 首次入选应为 {exp_wrm.date()}, got {res['WRM'].idxmax().date()}"
    assert exp_wrm > feb_end, "构造前提: 名单接纳早于预热期满(否则测不到预热闸)"
    assert res.loc[dates < exp_wrm, "WRM"].sum() == 0

    # -- 附加 STX: ST 窗口内出宇宙, 摘帽回归 --
    st_days = dates[idx["st0"]:idx["st1"]]
    assert not res.loc[st_days, "STX"].any(), "ST 期间必须不在"
    assert res.loc[dates[idx["st0"] - 1], "STX"] and res.loc[dates[idx["st1"]], "STX"]

    # -- 附加 SUS: 停牌日(价格仍落盘)出宇宙, 复牌回归 --
    sus_days = dates[idx["sus0"]:idx["sus1"]]
    assert not res.loc[sus_days, "SUS"].any(), "停牌日必须不在"
    assert res.loc[dates[idx["sus0"] - 1], "SUS"] and res.loc[dates[idx["sus1"]], "SUS"]

    # -- 附加: exclude_st=False 时 ST 窗口不剔除(参数真的在起作用) --
    res_keep_st = build_pit_universe(membership, store, min_list_days=30,
                                     exclude_st=False, top_n_liquidity=None)
    assert res_keep_st.loc[st_days, "STX"].all()
    print("main_semantics OK (asof/退市/预热/ST/停牌)")


def test_delist_close_gate_without_trading_filter():
    """钉死 close.notna() 退市闸独立于 is_trading 闸 (require_trading=False 口径)。

    关掉 is_trading 闸后, "行情消失 = 当日出宇宙"只可能由 close 闸完成 ——
    这是任务书测试要点② "store 里行情消失也要处理干净(NaN 不算成员)" 的
    独立断言; 删掉实现里的 `& close.notna()` 本测试必红。
    """
    dates, me, close, membership, store, idx = build_main_case()
    res = build_pit_universe(membership, store, min_list_days=30,
                             exclude_st=True, top_n_liquidity=None,
                             require_trading=False)

    # DEL: 退市(行情消失)当日起必须不在 —— 此口径下唯一能拦它的就是 close 闸
    d_stop = dates[idx["del_stop"]]
    assert res.loc[dates >= d_stop, "DEL"].sum() == 0, \
        "require_trading=False 下退市股仍在宇宙 → close.notna() 退市闸失守"
    assert res.loc[(dates >= dates[29]) & (dates < d_stop), "DEL"].all(), \
        "退市前(过完预热)应全程在"

    # NEW: 上市前 close 全 NaN, 没有 is_trading 闸也必须不在
    assert res.loc[dates < dates[idx["new_start"]], "NEW"].sum() == 0

    # 差分对照: SUS 停牌日价格仍落盘, 关掉停牌闸后应回到宇宙
    # (证明上面 DEL 的排除不是 require_trading 参数没生效的假象)
    sus_days = dates[idx["sus0"]:idx["sus1"]]
    assert res.loc[sus_days, "SUS"].all(), \
        "require_trading=False 应保留停牌日(价格仍在) —— 参数未生效?"

    # 输出仍全 bool 无 NaN
    assert (res.dtypes == bool).all()
    print("delist_close_gate_without_trading_filter OK")


# ---------------------------------------------------------- top_n 的 PIT 性
# 两只股票流动性王座换位: X 前80日成交额100、后段1; Y 相反。
# 过去60日中位数 → 90日王座属 X、115日属 Y。
#   若用全样本统计(偷看未来): X 永远赢 → 115日断言抓出;
#   若用未来窗口: 90日就该 Y 赢 → 90日断言抓出。
# 同一面板里塞一个全 NaN 日(测试5)。

def test_topn_pit_and_all_nan_day():
    dates = pd.bdate_range("2024-01-02", periods=130)
    codes = ["XLIQ", "YLIQ"]
    close = pd.DataFrame(10.0, index=dates, columns=codes)
    amount = pd.DataFrame(1.0, index=dates, columns=codes)
    amount.iloc[:80, 0] = 100.0   # X 前段活络
    amount.iloc[80:, 1] = 100.0   # Y 后段活络
    nan_day = 100
    close.iloc[nan_day] = np.nan  # 测试5: 全 NaN 日
    amount.iloc[nan_day] = np.nan
    # is_trading 恒 1(与 close.notna() 解耦): 全 NaN 日的排除只能由
    # close 闸完成, 否则该断言被 is_trading 闸影子覆盖(见文件头说明 c)。
    is_trading = pd.DataFrame(1.0, index=dates, columns=codes)
    is_st = pd.DataFrame(0.0, index=dates, columns=codes)

    me = month_ends(dates)
    membership = pd.DataFrame([(d, c) for d in me.values for c in codes],
                              columns=["date", "code"])
    store = FakeStore({"close": close, "amount": amount,
                       "is_st": is_st, "is_trading": is_trading})
    res = build_pit_universe(membership, store, min_list_days=1,
                             exclude_st=True, top_n_liquidity=1)

    # 第一期快照前全体不在
    assert not res.iloc[:21].to_numpy().any()
    # 流动性窗口没凑够(min_periods)之前: 已入名单也不进 top_n 宇宙
    assert not res.loc[dates[30]].any(), "流动性证据不足期应为空"
    # 测试4 PIT 关键断言(双向):
    assert res.loc[dates[90], "XLIQ"] and not res.loc[dates[90], "YLIQ"], \
        "90日王座应属 X —— 若选了 Y 说明流动性统计偷看了未来"
    assert res.loc[dates[115], "YLIQ"] and not res.loc[dates[115], "XLIQ"], \
        "115日王座应属 Y —— 若还是 X 说明用了全样本(非滚动)统计"
    # top_1 从不超员; 窗口凑够后除全NaN日外每天恰好1只
    assert (res.sum(axis=1) <= 1).all()
    mid = res.iloc[45:129].drop(index=dates[nan_day])
    assert (mid.sum(axis=1) == 1).all()

    # 测试5: 全 NaN 日不崩 + 该行全 False + 输出全 bool 无 NaN
    assert not res.loc[dates[nan_day]].any(), "全 NaN 日应全体不在"
    assert (res.dtypes == bool).all()
    print("topn_pit + all_nan_day OK")


# --------------------------------------------------- 名单为空期与退化输入

def test_degenerate_inputs():
    dates = pd.bdate_range("2024-01-02", periods=40)
    codes = ["AAA", "BBB"]
    close = pd.DataFrame(10.0, index=dates, columns=codes)
    frames = {"close": close, "amount": close * 1e6,
              "is_st": pd.DataFrame(0.0, index=dates, columns=codes),
              "is_trading": pd.DataFrame(1.0, index=dates, columns=codes)}
    store = FakeStore(frames)

    # 名单里只有幽灵股(store 全没有) → 不崩, 全 False
    ghost_only = pd.DataFrame({"date": [dates[20]], "code": ["GHOST"]})
    res = build_pit_universe(ghost_only, store, min_list_days=1,
                             top_n_liquidity=None)
    assert res.shape == close.shape and not res.to_numpy().any()

    # 快照日不是面板交易日(月末落在周末)也能 asof 生效
    weekend = pd.DataFrame({"date": [pd.Timestamp("2024-01-27")],  # 周六
                            "code": ["AAA"]})
    res2 = build_pit_universe(weekend, store, min_list_days=1,
                              top_n_liquidity=None)
    first_after = dates[dates.searchsorted(pd.Timestamp("2024-01-27"))]
    assert res2["AAA"].idxmax() == first_after
    assert not res2.loc[dates < first_after, "AAA"].any()
    assert not res2["BBB"].any()
    print("degenerate_inputs OK")


if __name__ == "__main__":
    test_main_semantics()
    test_delist_close_gate_without_trading_filter()
    test_topn_pit_and_all_nan_day()
    test_degenerate_inputs()
    print("ALL GREEN (m7_universe)")
