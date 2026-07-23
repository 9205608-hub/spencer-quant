"""带成本的分层回测 (M5, 近似口径 —— 每条假设都写在这里, 不藏)。

口径假设:
- 每 rebal_days 个交易日按因子值重排一次, 顶层做多/底层做空, 组内等权;
- 日收益用 fwd1 (T 日信号 → T+1 建仓 → T+2 结算) 近似, 不建模日内成交价;
- 涨跌停可成交性(30问#3 的补齐): 可选注入 tradability_masks() 产出的
  can_buy/can_sell 宽表 —— 调仓日想买而不可买的票放弃(仓位自然摊给可买的,
  因为组内等权), 想卖而不可卖的票被迫持有到下一个可卖日再卖出并计成本。
  不注入掩码时, 行为与 v0.1 完全一致(向后兼容, 有回归测试锚);
- 掩码按「执行日」取值, 不按信号日: fwd1 口径下 T 日信号在 T+1 收盘建/平仓
  (fwd1[T]=adj[T+2]/adj[T+1]-1), 成交与否由 T+1 当日是否一字板决定 —— 若按
  T 日取值, 会在过滤器的核心目标场景(T 日正常、T+1 一字涨停买不进)上漏网
  一天, 又把 T 日一字、T+1 打开的票多余排除。这不是给信号喂未来(选股仍只
  用 ≤T 信息), 是执行模拟与执行日对表(qlib 回测在执行步做涨跌停过滤同理)。
  执行日取"对齐后日期轴的下一行"—— 若因子日期轴是完整交易日历的真子集,
  这一步是近似; 末行没有下一行(执行日越过数据末端)无从判定, 按"掩码只做
  减法"惯例默认可成交;
- 成本 = 单边费率 × 成交比例, 单边费率 = (cost_bps + impact_bps_extra)/1e4。
  impact_bps_extra 是"额外冲击项 = impact_bps_extra × 换手"的线性近似:
  数学上等价于把单边成本抬高同样的 bp 数, 单独立参是为了把"佣金+印花+点差
  地板"(cost_bps) 与"冲击预算"(impact) 分开报数、分开做敏感性。真实市场
  冲击随参与率非线性增长(如 Almgren-Chriss 平方根/线性临时冲击), 这里是
  研究级线性近似, 不是容量模型, 不回答"能装多少钱"的问题;
- 成本计账口径披露: 无掩码路径沿用 v0.1 的「2×单边×换手」(首次建仓也按
  2× 收, 偏保守); 掩码路径按"实际发生的买+卖笔数"计账(首次建仓只收买入
  单边)。两条路径成本口径不完全相同 —— 对比数字别跨路径并排;
- 换手计账口径同样分叉: 无掩码路径 = 1 - |旧∩新|/|新|(v0.1 原样, 单边、按
  目标组数); 掩码路径 = (买笔+卖笔)/(2×持仓数)(双边平均、按实际持仓数),
  且滞留票的补卖只计成本、不计换手 —— avg_turnover_per_rebal 在掩码路径下
  系统性略低报, 跨路径不可并排;
- 掩码路径调仓日边缘行为: 若换仓后持仓将为空(目标组全员买不进、旧仓全部
  可卖出), 整次调仓静默跳过 —— 维持旧持仓、不计成本不计换手。宁持旧仓不
  清仓: 空仓期的组合日收益无定义; "目标组整组一字涨停"在真实宽截面下极
  罕见, 但这个分支存在, 要知道;
- 空头腿是纸面组合(A股不可便捷裸卖空), 掩码对两腿按同一"进/出"机制施加,
  空头腿的可成交性只是对称近似;
- 已知口径怪癖(v0.1 起继承, 为向后兼容不改): 两腿各自算"净多头组合"再相减,
  ls = top_net - bot_net, 于是空头腿的成本在 ls 里符号为正 —— 提高成本参数
  不保证降低 ls 净值(取决于哪腿换手更高)。成本单调性只对多头腿(long_net)
  良定义, 判成本敏感性看 long_net, 别看 ls_net;
- 结果是"研究级净收益", 不是"实盘级": 它回答的是排序问题
  (这个因子扣掉合理成本后还剩不剩肉), 不回答容量与冲击成本问题。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 创业板(sz.300)注册制改革(2020-08-24 首个交易日)后涨跌幅放宽到 20%,
# 科创板(sh.688) 同为 20%。判定阈值取 19.8%(留 0.2% 余量对冲最小报价单位
# 的取整误差, 与主板 9.8% / ST 4.8% 的留余量逻辑一致)。
# 已知近似: 科创板实际自 2019-07-22 开板起即 20%, 本实现按统一的 2020-08
# 边界切换 —— 2019-07~2020-08 间科创板 |开盘涨幅|>9.8% 的一字板会被误判为
# 涨跌停(过滤偏保守); 本框架宇宙(a800)中该窗口科创板成分极少, 影响可忽略,
# 但要知道这个近似存在。
GROWTH_BOARD_PREFIXES = ("sz.300", "sh.688")
GROWTH_BOARD_LIMIT = 0.198
GROWTH_BOARD_SINCE = "2020-08-24"


def tradability_masks(store, limit: float = 0.098,
                      limit_st: float = 0.048) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从日线判「一字涨停不可买 / 一字跌停不可卖」, 返回 (can_buy, can_sell)。

    这是 A 股日频回测的可成交性通识(qlib 回测层 limit_threshold 同思路):
    涨跌停价上排队买/卖大概率成交不了, 其中"一字板"(open==high==low 且
    开盘价相对昨收越限)是全天封死的极端情形 —— 只过滤一字板是保守下界:
    盘中打开过的涨停板仍视为可成交(实际部分成交, 这里不建模排队)。

    判定规则(为什么这么定):
    - 一字板: open==high==low 且 |open/prev_close - 1| ≥ 阈值。阈值用 9.8%
      而非 10%, 因为涨跌停价按最小报价单位取整后, 算出的涨幅常略低于名义
      限幅(如昨收 10.00 → 涨停 11.00 → 恰 10%, 但昨收 9.87 → 涨停 10.86
      → 10.03%, 而 9.53 → 10.48 → 9.97%), 留 0.2% 余量防漏判;
    - ST 股(is_st==1)阈值 4.8%(名义限幅 5%);
    - 创业板 sz.300 / 科创板 sh.688 在 2020-08 之后阈值 19.8%(名义 20%),
      按代码前缀区分; 该覆盖在 ST 之后施加 —— 注册制创业板/科创板的 ST 股
      同样适用 20% 限幅, 无 5% 特例;
    - prev_close 用不复权 close.shift(1)。已知近似: 除权除息日交易所的涨跌
      停基准是除权参考价而非昨收盘价, 这里未做除权调整, 除权日的判定可能
      漏判/误判 —— 日频研究级近似, 影响限于除权当日;
    - 缺数据(NaN)一律视为可成交: 掩码只回答涨跌停问题, 只做减法; 停牌/未
      上市的过滤走 is_trading, 在使用端另行处理(与 30问#3 的分工一致)。

    返回两张 date×code 的 bool 宽表: can_buy = 非一字涨停, can_sell = 非一字
    跌停。
    """
    o = store.load("open")
    h = store.load("high")
    l = store.load("low")
    c = store.load("close")
    is_st = store.load("is_st")

    prev_close = c.shift(1)
    ret_open = o / prev_close - 1.0
    one_line = (o == h) & (o == l) & prev_close.notna()

    thresh = pd.DataFrame(limit, index=o.index, columns=o.columns)
    thresh = thresh.mask(is_st.reindex_like(thresh) == 1, limit_st)
    growth_cols = [s for s in o.columns if str(s).startswith(GROWTH_BOARD_PREFIXES)]
    if growth_cols:
        after = thresh.index >= pd.Timestamp(GROWTH_BOARD_SINCE)
        thresh.loc[after, growth_cols] = GROWTH_BOARD_LIMIT

    limit_up_one_line = one_line & (ret_open >= thresh)
    limit_dn_one_line = one_line & (ret_open <= -thresh)
    can_buy = ~limit_up_one_line
    can_sell = ~limit_dn_one_line
    return can_buy, can_sell


def _tradable(mask: pd.DataFrame | None, dt, code) -> bool:
    """掩码查询: 无掩码/执行日越过数据末端(dt=None)/缺日期/缺代码/NaN 一律
    视为可成交(掩码只做减法, 见模块 docstring)。"""
    if mask is None or dt is None:
        return True
    if dt not in mask.index or code not in mask.columns:
        return True
    v = mask.at[dt, code]
    return True if pd.isna(v) else bool(v)


def layered_backtest(factor: pd.DataFrame, fwd1: pd.DataFrame,
                     q: int = 5, rebal_days: int = 5,
                     cost_bps: float = 15.0,
                     can_buy: pd.DataFrame | None = None,
                     can_sell: pd.DataFrame | None = None,
                     impact_bps_extra: float = 0.0) -> dict:
    """带成本分层多空回测。新参数全部可选, 不传时与 v0.1 逐位一致。

    can_buy / can_sell: tradability_masks() 产出的 bool 宽表(True=可成交)。
      掩码按执行日取值 = 信号日在日期轴上的下一行(fwd1 口径的 T+1, 交易
      实际发生的那天; 为什么见模块 docstring): 想买(在目标组但未持有)而
      执行日不可买 → 放弃, 仓位因组内等权自然摊给其余成员; 想卖(持有但
      不在目标组)而执行日不可卖 → 被迫继续持有(其收益继续计入组合), 此后
      每天盯, 首个执行日可卖的日子卖出并计成本(记在该信号日行)。
    impact_bps_extra: 额外冲击项(bp), 按"impact_bps_extra × 换手"线性计入
      成本 —— 与 cost_bps 同乘同一个成交比例, 等价于抬高单边成本。这是
      研究级线性近似, 不是容量/冲击模型(见模块 docstring)。
    返回 dict 含 long_series(净多头日收益, 便于逐日核对)与 v0.1 全部键。
    """
    rate = (cost_bps + impact_bps_extra) / 1e4
    use_masks = can_buy is not None or can_sell is not None
    idx = factor.index.intersection(fwd1.index)
    factor, fwd1 = factor.loc[idx], fwd1.loc[idx]

    pct = factor.rank(axis=1, pct=True)
    rebal_dates = idx[::rebal_days]

    def leg(select_mask_at) -> tuple[pd.Series, pd.Series]:
        rets, turns, costs = {}, {}, {}
        held: set = set()
        pending_sell: set = set()   # 想卖没卖掉、被迫滞留的票(仅掩码路径)
        next_i = 0
        for i, dt in enumerate(idx):
            # 执行日 = 日期轴下一行: fwd1[T] 覆盖 T+1 收盘→T+2 收盘, 交易发生
            # 在 T+1 收盘, 可成交性按执行日取值(按信号日取值会在"T 正常、T+1
            # 一字板"的核心场景漏网一天)。末行无下一行 → None → 默认可成交。
            exec_dt = idx[i + 1] if i + 1 < len(idx) else None
            # ① 滞留票补卖: 每天盯, 执行日可卖即卖, 按占持仓比例计成本
            if pending_sell:
                sellable = {s for s in pending_sell if _tradable(can_sell, exec_dt, s)}
                if sellable:
                    costs[dt] = costs.get(dt, 0.0) + rate * len(sellable) / len(held)
                    held -= sellable
                    pending_sell -= sellable
            # ② 调仓日
            if next_i < len(rebal_dates) and dt == rebal_dates[next_i]:
                row = pct.loc[dt].dropna()
                target = set(row[select_mask_at(row)].index)
                if target:
                    if not use_masks:
                        # 无掩码路径: 与 v0.1 完全一致(回归测试锚, 勿改口径)。
                        # held 直接绑定 target 而非 set(target) 拷贝: 重建集合
                        # 会改变迭代顺序 → mean() 求和顺序变 → 末位比特漂移,
                        # 逐位一致就不成立了(实测踩过)。
                        turns[dt] = 1.0 if not held else 1 - len(held & target) / len(target)
                        costs[dt] = 2 * rate * turns[dt]    # 卖旧+买新
                        held = target
                    else:
                        buys = {s for s in target - held if _tradable(can_buy, exec_dt, s)}
                        want_out = held - target
                        sells = {s for s in want_out if _tradable(can_sell, exec_dt, s)}
                        forced = want_out - sells
                        new_held = (held & target) | buys | forced
                        # new_held 为空(目标组全员买不进且旧仓全部可卖出)时
                        # 整次调仓跳过: 宁持旧仓不清仓, 已在模块 docstring 披露。
                        if new_held:
                            n_traded = len(buys) + len(sells)
                            # 报表口径: 双边平均换手(买笔+卖笔)/(2×持仓数)
                            turns[dt] = n_traded / (2 * len(new_held))
                            costs[dt] = costs.get(dt, 0.0) + rate * n_traded / len(new_held)
                            held = new_held
                            pending_sell = forced
                next_i += 1
            if held:
                rets[dt] = fwd1.loc[dt, list(held)].mean()
        turns = pd.Series(turns, dtype=float)
        r = pd.Series(rets, dtype=float).sub(pd.Series(costs, dtype=float), fill_value=0.0)
        return r, turns

    top_r, top_t = leg(lambda row: row > 1 - 1 / q)
    bot_r, bot_t = leg(lambda row: row <= 1 / q)
    ls = (top_r - bot_r).dropna()

    def metrics(r: pd.Series) -> dict:
        ann = r.mean() * 252
        vol = r.std() * np.sqrt(252)
        cum = r.cumsum()
        mdd = (cum - cum.cummax()).min()
        return {"ann_ret": round(float(ann), 4), "sharpe": round(float(ann / vol), 2),
                "max_dd": round(float(mdd), 3)}

    return {
        "long_net": metrics(top_r.dropna()),
        "ls_net": metrics(ls),
        "avg_turnover_per_rebal": round(float(top_t.iloc[1:].mean()), 3) if len(top_t) > 1 else None,
        "n_rebalances": len(top_t),
        "ls_series": ls,
        "long_series": top_r,
    }
