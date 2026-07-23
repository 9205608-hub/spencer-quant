"""因子入库验证契约 (admission checklist) —— 因子进入正式因子池前的统一门禁。

概念是行业通行做法(统一的入库前验证关卡, qlib/alphalens 生态同类思想),
关卡选择与实现全部原创。为什么要有这道门: 没有统一契约时, 每个因子按
作者自己的口径"自证优秀", 口径漂移 + 选择性汇报会让因子池慢慢烂掉;
契约的意义 = 所有因子过同一把尺子, 尺子本身版本化。

七关(fail 任一关 → fail; 仅预警关亮灯 → warn):
1. 末端对齐(硬): 因子末端 == 数据末端 —— 上线第一天就没值的因子不收。
2. 覆盖率(硬): 近一年日均覆盖 ≥ min_coverage × 可交易截面 —— 覆盖过窄的
   因子 IC 再高也撑不起组合。
3. 方向(硬): 全样本 IC > 0 —— 契约要求送检前已 orient(值大=看多)。
4. 逐年一致性(硬): 正 IC 年份占比 ≥ min_pos_year_ratio —— 单年驱动=假信号。
5. 显著性(硬): Newey-West t ≥ min_t_nw(重叠窗口修正后的显著性)。
6. 换手预警(软): 秩自相关 < min_rank_autocorr 亮灯 —— 快因子不是不能收,
   是要带着"成本吃得起吗"的问题进入下一关(成本后回测)。
7. 在库查重(软): 与在库因子的截面秩相关时序均值 |ρ| > max_pool_corr 亮灯
   —— 高冗余因子挤占风险预算, 合成时权重白给。

元信息联动: 若注册时给了 valid_from, 起始日之前的因子值不参与检查
(数据源有效起始日概念 —— 财报类因子在首个披露季前无意义)。
"""
from __future__ import annotations

import pandas as pd

from .base import get_meta
from ..eval.panel import ic_series, newey_west_tstat, rank_autocorr, yearly_table

DEFAULTS = dict(min_coverage=0.40, min_pos_year_ratio=0.60,
                min_t_nw=3.0, min_rank_autocorr=0.50, max_pool_corr=0.70)


def admission_check(name: str, factor_df: pd.DataFrame, store, fwd: pd.DataFrame,
                    pool: dict[str, pd.DataFrame] | None = None,
                    horizon: int = 5, min_names: int = 30, **overrides) -> dict:
    """返回 {"verdict": pass/warn/fail, "checks": {关名: {读数, 判定}}, ...}"""
    th = {**DEFAULTS, **overrides}
    meta = get_meta(name)
    if meta.get("valid_from"):
        factor_df = factor_df.loc[pd.Timestamp(meta["valid_from"]):]

    checks: dict[str, dict] = {}

    data_end = store.end_date()
    checks["end_aligned"] = {"value": str(factor_df.index[-1].date()),
                             "ok": factor_df.index[-1] == data_end, "hard": True}

    cov = (factor_df.notna().sum(axis=1).tail(252)
           / store.load("close").notna().sum(axis=1).tail(252)).mean()
    checks["coverage_1y"] = {"value": round(float(cov), 3),
                             "ok": cov >= th["min_coverage"], "hard": True}

    ic = ic_series(factor_df, fwd, min_names)
    checks["direction"] = {"value": round(float(ic.mean()), 4),
                           "ok": ic.mean() > 0, "hard": True}

    yr = yearly_table(ic)
    ratio = float(yr["positive"].mean()) if len(yr) else 0.0
    checks["yearly_consistency"] = {"value": round(ratio, 2),
                                    "ok": ratio >= th["min_pos_year_ratio"], "hard": True}

    t_nw = newey_west_tstat(ic, lag=horizon)
    checks["significance_nw"] = {"value": round(float(t_nw), 2),
                                 "ok": t_nw >= th["min_t_nw"], "hard": True}

    ac = rank_autocorr(factor_df)
    checks["turnover_flag"] = {"value": ac, "ok": ac >= th["min_rank_autocorr"],
                               "hard": False}

    worst = None
    if pool:
        for pname, pdf in pool.items():
            # 每 5 日抽一天算截面秩相关, 取时序均值 —— 全日历逐日算是 O(慢),
            # 抽样对"冗余与否"这个粗判据足够
            sub = factor_df.iloc[::5]
            rho = sub.rank(axis=1).corrwith(pdf.reindex(sub.index).rank(axis=1),
                                            axis=1).mean()
            if pd.notna(rho) and (worst is None or abs(rho) > abs(worst[1])):
                worst = (pname, float(rho))
    checks["pool_redundancy"] = {
        "value": None if worst is None else {worst[0]: round(worst[1], 3)},
        "ok": worst is None or abs(worst[1]) <= th["max_pool_corr"], "hard": False}

    hard_fail = any(not c["ok"] for c in checks.values() if c["hard"])
    soft_flag = any(not c["ok"] for c in checks.values() if not c["hard"])
    verdict = "fail" if hard_fail else ("warn" if soft_flag else "pass")
    return {"name": name, "verdict": verdict, "checks": checks,
            "meta": meta, "thresholds": th}


def format_report(result: dict) -> str:
    lines = [f"== 入库验证: {result['name']} → {result['verdict'].upper()} =="]
    for k, c in result["checks"].items():
        mark = "✓" if c["ok"] else ("✗" if c["hard"] else "⚠")
        lines.append(f"  {mark} {k}: {c['value']}")
    if result["meta"]:
        lines.append(f"  登记: {result['meta']}")
    return "\n".join(lines)
