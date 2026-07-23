"""PIT 宇宙构造器 (M7 核心)：历史时点在市名单 + 规则过滤 → bool 宽表 date×code。

解决的问题（30问 #6）：用今天的名单跑五年回测 = 幸存者偏差（survivorship
bias，回测方法论通识，AFML 亦反复强调数据必须 point-in-time）——退市的
股票被静默抹掉，IC/收益系统性虚高。本模块把"某天谁在研究宇宙里"变成一个
逐日可查、只依赖当日及以前信息的布尔矩阵。

方法论出处（全公开）：
- PIT / 幸存者偏差纪律：AFML（Advances in Financial Machine Learning）
  与通用回测教科书；
- 动态宇宙思想：qlib 的 instruments 机制（成分随时间进出，按日生效）；
- asof 语义：pandas merge_asof —— 永远取"最近的过去"快照，绝不向后看；
- 低流动性剔除：学术实证研究的常见样本构造惯例（如 Amihud 2002 一类
  研究先剔除微流动性股票再做检验）。

输入契约：
- membership: 长表 DataFrame[date, code]，date = 月末采样日，每行表示
  "该月末该股在市"（含后来退市的股票，见 examples/fetch_pit.py）；
- store: WideStore（或同接口对象），需有 close / amount / is_st /
  is_trading 四张 date×code 宽表；
- 输出: bool 宽表，index/columns 与 store 的 close 完全一致，True = 该日
  在研究宇宙内。下游用法: factor_df.where(universe_mask)。

已知近似（明示，不藏）：
1. 上市天数 = 该股在 store 内累计有效行情日（close 非 NaN 的累计数）。
   面板起点之前的历史被截断，所以面板头部 min_list_days 天内所有股票都会
   被预热闸挡住 —— 研究读数窗口应从"面板起点 + 预热期"之后开始。
2. 名单按月末采样：月中新上市的股票要等下一个月末快照才可能入选（保守方
   向，宁可晚进不提前）；月中退市的股票名单侧最多滞后一个月退出，但行情
   NaN 闸当日就把它拿掉（双保险，见下）。
3. 流动性口径 = 过去60日成交额中位数（中位数抗单日脉冲），目标是"可投资
   池"筛选，不是流动性因子本身。
"""
from __future__ import annotations

import pandas as pd

from .store import WideStore

# 流动性统计窗口(交易日)与窗口内最少有效成交日。
# 为什么 min_periods=40（窗口的2/3）：近60日里有效成交日不足40天的股票，
# 要么长期停牌要么刚复牌，其流动性估计不可信 —— 视为"流动性证据不足"，
# top_n 模式下直接剔除。这顺带把长期停牌股挡在宇宙外。
LIQ_WINDOW = 60
LIQ_MIN_PERIODS = 40


def build_pit_universe(
    membership: pd.DataFrame,
    store: WideStore,
    min_list_days: int = 120,
    exclude_st: bool = True,
    top_n_liquidity: int | None = 1500,
    require_trading: bool = True,
) -> pd.DataFrame:
    """构造逐日 PIT 研究宇宙，返回 bool 宽表（date×code，形状 == close）。

    四道闸，每道只用 t 日及以前的信息：

    1. 成员资格（asof）：t 日的名单 = 采样日 ≤ t 的最近一期月末名单。
       为什么允许 "=t"：月末快照描述的是该日实际在市的名单，t 日收盘即可
       观测；因子在 T 日、T+1 才建仓（见 eval/panel.forward_returns），
       不越权。为什么第一期快照之前全体 False：没有可用的历史名单就宁可
       空着，绝不用未来名单回填（backfill 是幸存者偏差的另一种写法）。
    2. 预热期：累计有效行情日 >= min_list_days 才放行（30问 #7：上市初期
       的换手/波动是另一个物种）。cumsum 只累加过去，PIT 安全。
    3. 可交易性：close 为 NaN（退市/无行情）当日不算成员 —— 这是退市股的
       第二道保险，名单还没来得及退出时行情闸先生效；exclude_st 剔除当日
       ST（is_st 当日可观测）；require_trading 剔除停牌日（30问 #3：停牌
       价格 ffill 后当有效样本会稀释 IC）。
    4. 流动性 top-N：过去60日成交额中位数的**当日截面**排名 ≤ N 才留下。
       排名只在已过前三道闸的池内做 —— top_N 的语义是"可投资池里最活络
       的 N 只"，不是全市场绝对名次。滚动统计只用 t 及以前的成交额，排名
       只用 t 日截面，PIT 安全。method='first' 按列序破平票，保证恰好 N
       只且可复现。

    参数
    ----
    membership: 长表 [date, code]，月末在市名单（含后来退市股）。
    min_list_days: 上市预热天数（按 store 内有效行情日计）。
    exclude_st: 是否剔除当日 ST。
    top_n_liquidity: 流动性截断名额；None = 不做流动性筛选。
    require_trading: 是否剔除停牌日（is_trading != 1）。
    """
    assert {"date", "code"}.issubset(membership.columns), \
        "membership 需要 date/code 两列"
    assert len(membership) > 0, "membership 为空"

    close = store.load("close")
    days, codes = close.index, close.columns

    # ---- 闸1: 成员资格 asof —— 距 t 最近的过去月末名单 ----
    mem = membership[["date", "code"]].copy()
    mem["date"] = pd.to_datetime(mem["date"]).dt.normalize()
    # 名单里有、store 里没有行情的代码（抓取失败等）不可能入选：直接丢弃,
    # 输出形状锚定 close, 保证 mask 可与任何因子宽表直接对齐。
    mem = mem[mem["code"].isin(codes)]
    snap = (pd.crosstab(mem["date"], mem["code"]) > 0) if len(mem) else \
        pd.DataFrame(index=pd.DatetimeIndex([]), columns=[])
    snap = snap.reindex(columns=codes, fill_value=False)
    # union + ffill = merge_asof(≤t) 的矩阵写法: 快照只向未来生效, 绝不回填。
    # 快照日不要求恰好是交易日(周末月末也能对齐)。
    union_idx = snap.index.union(days)
    member = (snap.astype("float64").reindex(union_idx).ffill()
              .reindex(days).fillna(0.0).astype(bool))

    # ---- 闸2: 预热期 —— store 内累计有效行情日 ----
    valid_days = close.notna().cumsum()
    seasoned = valid_days >= min_list_days

    # ---- 闸3: 当日可交易性 ----
    elig = member & seasoned & close.notna()
    if exclude_st:
        # eq(1) 对 NaN 返回 False → 缺数据不误杀(缺行情已被 close 闸挡掉)
        elig &= ~store.load("is_st").reindex_like(close).eq(1)
    if require_trading:
        # 这里 NaN → eq(1)=False → 剔除: 没有交易状态记录的日子不进宇宙
        elig &= store.load("is_trading").reindex_like(close).eq(1)

    # ---- 闸4: 流动性 top-N（当日截面排名, 只用 t 及以前数据）----
    if top_n_liquidity is not None:
        amt = store.load("amount").reindex_like(close)
        amt = amt.where(amt > 0)  # 停牌/零成交日不参与中位数, 交给 min_periods 兜底
        liq = amt.rolling(LIQ_WINDOW, min_periods=LIQ_MIN_PERIODS).median()
        rank = liq.where(elig).rank(axis=1, ascending=False, method="first")
        # rank 为 NaN(流动性证据不足) → le=False → 剔除
        elig &= rank.le(top_n_liquidity)

    return elig.astype(bool)
