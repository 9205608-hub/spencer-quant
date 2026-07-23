"""PIT 终跑: 全市场(含退市)宇宙 + 全模块升级后的十年终版读数。

与此前所有 run 的本质区别: 宇宙第一次是 PIT 的 —— 成员资格来自历史月末
在市名单(含后来退市的股票), 幸存者偏差消除。同时启用本轮全部升级:
BTOP 六风格中性化 / Newey-West t / IC 衰减曲线 / 一字板可成交过滤 /
DSR+PBO 统计纪律。model_gb 已证伪结案(output/falsify_model_gb.md), 不再上桌。

分段容错: 单段失败记入报告继续跑, 全部读数进台账。
产出: output/PIT终跑报告.md + panel_*_pit.png + summary_pit.csv。
"""
from __future__ import annotations

import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spencer.config import load_config
from spencer.data.fetch import NUMERIC_COLS
from spencer.data.store import WideStore, build_wide_store
from spencer.data.universe import build_pit_universe
from spencer.factor import zoo  # noqa: F401
from spencer.factor.base import compute, list_factors
from spencer.factor.ops import orient, winsorize_mad
from spencer.risk.style import build_styles, load_industry
from spencer.risk.fundamental import build_value_style
from spencer.risk.neutral import residualize
from spencer.eval.panel import (forward_return_1d, forward_returns,
                                horizon_profile, ic_decay_plot, run_panel)
from spencer.strategy.composite import equal_weight
from spencer.strategy.portfolio import export_targets, topn_buffer_weights, turnover_series
from spencer.backtest.layered import layered_backtest, tradability_masks
from spencer.discipline.ledger import log_run, trial_count
from spencer.discipline.stats import dsr_from_ledger, pbo_cscv

CFG = load_config()
OUT = CFG["output_dir"]
LEDGER = CFG["root"] / "research_ledger.csv"
REPORT: list[str] = [f"# Spencer 框架 PIT 终跑报告 ({datetime.now():%Y-%m-%d %H:%M})", ""]
FAILURES: list[str] = []


def add(line: str = ""):
    REPORT.append(line)


def guard(title):
    def deco(fn):
        def wrapped(*a, **k):
            t0 = time.time()
            try:
                r = fn(*a, **k)
                print(f"[ok] {title} ({time.time() - t0:.0f}s)", flush=True)
                return r
            except Exception:
                tb = traceback.format_exc()
                print(f"[FAIL] {title}\n{tb}", flush=True)
                FAILURES.append(f"{title}: {tb.splitlines()[-1]}")
                return None
        return wrapped
    return deco


@guard("合并 pit parts → 长表")
def build_long():
    raw_dir = CFG["data_dir"] / "raw"
    out = raw_dir / "daily_long_pit.parquet"
    if out.exists():
        print("[long] 已存在, 跳过合并")
        return out
    frames = []
    for p in sorted((raw_dir / "pit_parts").glob("*.parquet")):
        df = pd.read_parquet(p)
        if len(df) and "date" in df.columns:
            frames.append(df)
    long_df = pd.concat(frames, ignore_index=True)
    # 类型转换与口径统一(与 fetch.fetch_daily 同一套规则, parts 里存的是原始字符串)
    for col in NUMERIC_COLS + ["adj_close"]:
        long_df[col] = pd.to_numeric(long_df[col], errors="coerce")
    long_df["date"] = pd.to_datetime(long_df["date"])
    long_df["turn"] = (long_df["turn"] / 100.0).fillna(0.0).clip(0.0, 1.0)
    long_df.loc[long_df["tradestatus"] != "1", "turn"] = 0.0
    long_df["is_st"] = (long_df["isST"] == "1").astype("int8")
    long_df["is_trading"] = (long_df["tradestatus"] == "1").astype("int8")
    long_df = long_df.drop(columns=["isST", "tradestatus"])
    long_df.to_parquet(out, index=False)
    print(f"[long] rows={len(long_df):,} codes={long_df['code'].nunique()}")
    return out


def main():
    long_path = build_long()
    store = build_wide_store(long_path, CFG["data_dir"] / "wide_pit")
    close = store.load("close")

    mem = pd.read_parquet(CFG["data_dir"] / "raw" / "pit_membership.parquet")
    up = CFG["universe_pit"]
    universe = build_pit_universe(
        mem, store, min_list_days=up["min_list_days"],
        exclude_st=up["exclude_st"], top_n_liquidity=up["top_n_liquidity"])
    add("## 数据与宇宙")
    add(f"- 全市场(含退市): {close.shape[1]} 只 × {close.shape[0]} 交易日, "
        f"{close.index[0].date()} → {close.index[-1].date()}")
    add(f"- PIT 宇宙: 月末在市名单 asof + 预热{up['min_list_days']}日 + 剔ST + "
        f"流动性前{up['top_n_liquidity']}; 日均成员 {int(universe.sum(axis=1).mean())} 只")
    add("- 幸存者偏差: **已消除**(名单含后来退市股); 行业仍为快照(30问#31)")
    add()

    industry = load_industry(CFG["data_dir"] / "raw" / "industry.csv", close.columns)
    print("[risk] 六风格(含BTOP)...", flush=True)
    styles = build_styles(store)
    vb = build_value_style(store, CFG["root"] / CFG["fundamentals_cache"])
    n_styles = 5
    if vb is not None:
        styles["value_btop"] = vb
        n_styles = 6
    add(f"- 风格集: {n_styles} 风格({'含' if n_styles == 6 else '缺'} BTOP) + 行业哑变量")
    add()

    h = CFG["label"]["horizon"]
    uni = universe & (store.load("is_trading") == 1) & (store.load("is_st") == 0)
    fwd = forward_returns(store, h).where(uni)
    fwd1 = forward_return_1d(store)
    print("[risk] 标签残差化(宇宙内截面)...", flush=True)
    fwd_neut = residualize(fwd, styles, industry)

    # ---- 单因子三档 ----
    rows, ls_matrix = [], {}
    for name in list_factors():
        stage = guard(f"因子 {name} 三档")(lambda n=name: one_factor(
            n, store, uni, styles, industry, fwd, fwd1, fwd_neut, h))
        r = stage()
        if r:
            rows.extend(r["rows"])
            if r.get("ls") is not None:
                ls_matrix[name] = r["ls"]

    add("## 单因子三档读数(PIT 宇宙)")
    summary = pd.DataFrame(rows)
    add("```")
    add(summary.to_string(index=False))
    add("```")
    summary.to_csv(OUT / "summary_pit.csv", index=False)
    add()

    # ---- 合成 + 组合(可成交过滤) ----
    comp_stage = guard("合成+组合")(lambda: composite_and_portfolio(
        store, uni, styles, industry, fwd, fwd1, fwd_neut, h))
    comp = comp_stage()

    # ---- 统计纪律 ----
    disc = guard("DSR+PBO")(lambda: discipline_block(comp, ls_matrix))
    disc()

    add("## 诚实声明")
    add(f"- 台账累计 N = {trial_count(LEDGER)}; 本报告全部读数已入台账")
    add("- 行业为快照口径; 成本为研究级近似; model_gb 已证伪结案不再上桌")
    if FAILURES:
        add("## 失败段")
        for f in FAILURES:
            add(f"- {f}")
    (OUT / "PIT终跑报告.md").write_text("\n".join(REPORT), encoding="utf-8")
    print(f"\nPIT_FINAL_DONE 报告: {OUT / 'PIT终跑报告.md'}")


def one_factor(name, store, uni, styles, industry, fwd, fwd1, fwd_neut, h):
    base = winsorize_mad(compute(name, store, CFG["data_dir"] / "factors_pit").where(uni))
    modes = {
        "raw": (base, fwd),
        "size_neut": (residualize(base, {"size": styles["size"]}), fwd),
        "full_neut": (residualize(base, styles, industry), fwd_neut),
    }
    rows, ls = [], None
    for mode, (f, label) in modes.items():
        f, sign = orient(f, label)
        res = run_panel(f"{name}@{mode}_pit", f, store, horizon=h,
                        q=CFG["eval"]["quantiles"], min_names=CFG["eval"]["min_names"],
                        outdir=OUT, fwd=label, fwd1=fwd1)
        res["orient_sign"] = sign
        if mode == "full_neut":
            prof = horizon_profile(f, store, horizons=(1, 5, 10, 20),
                                   min_names=CFG["eval"]["min_names"])
            ic_decay_plot(prof, f"{name}_pit", OUT / f"decay_{name}_pit.png")
            bt = layered_backtest(f, fwd1, q=CFG["eval"]["quantiles"], rebal_days=h,
                                  cost_bps=CFG["eval"]["cost_bps"])
            res["ls_net_ann"] = bt["ls_net"]["ann_ret"]
            res["ls_net_sharpe"] = bt["ls_net"]["sharpe"]
            ls = bt["ls_series"]
        log_run(LEDGER, res, params={"mode": mode, "universe": "pit_full_market"})
        rows.append({k: res.get(k) for k in
                     ("factor", "ic_mean", "ic_ir_daily", "t_stat_conservative",
                      "t_stat_nw", "yearly_all_positive", "rank_autocorr_5d",
                      "ls_net_ann", "ls_net_sharpe")})
    return {"rows": rows, "ls": ls}


def composite_and_portfolio(store, uni, styles, industry, fwd, fwd1, fwd_neut, h):
    oriented = {}
    for name in list_factors():
        base = winsorize_mad(compute(name, store, CFG["data_dir"] / "factors_pit").where(uni))
        f = residualize(base, styles, industry)
        f, _ = orient(f, fwd_neut)
        oriented[name] = f
    comp = equal_weight(oriented)
    comp_o, sign = orient(comp, fwd_neut)
    res = run_panel("comp_eq_pit", comp_o, store, horizon=h,
                    q=CFG["eval"]["quantiles"], min_names=CFG["eval"]["min_names"],
                    outdir=OUT, fwd=fwd_neut, fwd1=fwd1)
    prof = horizon_profile(comp_o, store, horizons=(1, 5, 10, 20))
    ic_decay_plot(prof, "comp_eq_pit", OUT / "decay_comp_eq_pit.png")

    can_buy, can_sell = tradability_masks(store)
    bt = layered_backtest(comp_o, fwd1, q=CFG["eval"]["quantiles"], rebal_days=h,
                          cost_bps=CFG["eval"]["cost_bps"],
                          can_buy=can_buy, can_sell=can_sell)
    res["ls_net_ann"], res["ls_net_sharpe"] = bt["ls_net"]["ann_ret"], bt["ls_net"]["sharpe"]
    log_run(LEDGER, res, params={"mode": "comp_eq", "universe": "pit_full_market",
                                 "masks": True})

    weights = topn_buffer_weights(comp_o, top_n=50, buffer_n=80, rebal_days=h)
    port_ret = (weights.shift(0) * fwd1).sum(axis=1)          # 权重日=信号日, fwd1 自带 T+1 建仓
    to = turnover_series(weights)
    cost_daily = to * (CFG["eval"]["cost_bps"] / 1e4)
    long_net = (port_ret - cost_daily).dropna()
    bench = fwd1.where(uni).mean(axis=1).reindex(long_net.index)
    excess = (long_net - bench).dropna()

    def ann(r):
        return round(float(r.mean() * 252), 4), round(float(r.mean() / r.std() * np.sqrt(252)), 2)

    la, ls_ = ann(long_net); ba, bs_ = ann(bench.dropna()); ea, es = ann(excess)
    add("## 合成与组合(PIT 宇宙, 一字板过滤, 成本后)")
    add(f"- comp_eq_pit: IC {res['ic_mean']}, NW-t {res.get('t_stat_nw')}, "
        f"多空净年化 {res['ls_net_ann']:.1%}, sharpe {res['ls_net_sharpe']}")
    add(f"- top50/buffer80 多头净: 年化 {la:.1%} (sharpe {ls_}) vs 宇宙等权基准 "
        f"{ba:.1%} (sharpe {bs_}) → 超额 {ea:.1%} (sharpe {es})")
    add(f"- 年化单边换手 {to.mean() * 252:.1f}x")
    tbl = export_targets(weights, weights.index[-1], OUT / "target_holdings_pit.csv")
    add(f"- 执行接口: {weights.index[-1].date()} 目标持仓 {len(tbl)} 只 → target_holdings_pit.csv")
    add()
    return {"long_net": long_net, "ls_net_sharpe_daily":
            float(bt["ls_series"].mean() / bt["ls_series"].std()) if len(bt["ls_series"]) else None,
            "ls_series": bt["ls_series"]}


def discipline_block(comp, ls_matrix):
    add("## 统计纪律(对最好读数动刀)")
    if comp and comp.get("ls_series") is not None and len(comp["ls_series"]):
        s = comp["ls_series"].dropna()
        sr_daily = float(s.mean() / s.std())
        d = dsr_from_ledger(sr_daily, T=len(s), ledger_path=LEDGER)
        add(f"- comp_eq 多空净 日频Sharpe {sr_daily:.4f} (T={len(s)}), "
            f"台账 N={trial_count(LEDGER)} 校正后 DSR: {d}")
    if len(ls_matrix) >= 3:
        m = pd.DataFrame(ls_matrix).dropna(how="all")
        p = pbo_cscv(m, n_splits=16)
        add(f"- 7因子多空收益矩阵 PBO(CSCV): {p}")
    add()


if __name__ == "__main__":
    main()
