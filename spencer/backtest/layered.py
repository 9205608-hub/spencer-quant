"""带成本的分层回测 (M5, 近似口径 —— 每条假设都写在这里, 不藏)。

口径假设:
- 每 rebal_days 个交易日按因子值重排一次, 顶层做多/底层做空, 组内等权;
- 日收益用 fwd1 (T 日信号 → T+1 建仓 → T+2 结算) 近似, 不建模日内成交价
  与涨跌停可成交性(30问#3 欠账);
- 成本 = cost_bps(单边) × 名义换手(按成员进出比例近似), 调仓日一次性扣,
  多空两腿分别计费;
- 结果是"研究级净收益", 不是"实盘级": 它回答的是排序问题
  (这个因子扣掉合理成本后还剩不剩肉), 不回答容量与冲击成本问题。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def layered_backtest(factor: pd.DataFrame, fwd1: pd.DataFrame,
                     q: int = 5, rebal_days: int = 5,
                     cost_bps: float = 15.0) -> dict:
    cost = cost_bps / 1e4
    idx = factor.index.intersection(fwd1.index)
    factor, fwd1 = factor.loc[idx], fwd1.loc[idx]

    pct = factor.rank(axis=1, pct=True)
    rebal_dates = idx[::rebal_days]

    def leg(select_mask_at) -> tuple[pd.Series, pd.Series]:
        rets, turns = {}, {}
        members: set = set()
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
        r = pd.Series(rets).sub(2 * cost * turns, fill_value=0.0)   # 卖旧+买新
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
    }
