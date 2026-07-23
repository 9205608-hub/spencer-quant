"""Spencer 框架端到端: 数据 → 因子 → 三档中性化读数 → 成本后回测 → 台账。

三档读数(工业界通行的"剂量曲线"读法, 全部同面板函数同参数):
  @raw        原始因子 vs 原始前瞻收益
  @size_neut  市值中性因子 vs 原始前瞻收益
  @full_neut  行业+五风格中性因子 vs 行业+五风格中性收益(纯alpha读数)
三档 IC 的落差 = 该因子有多少收益其实是风格搭便车。

用法:
  python examples/quickstart.py --limit 80    # 冒烟
  python examples/quickstart.py               # 全量 a800
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spencer.config import load_config
from spencer.data.fetch import fetch_to_parquet, fetch_industry, BaostockSession
from spencer.data.store import build_wide_store
from spencer.factor import zoo  # noqa: F401  注册因子
from spencer.factor.base import compute, list_factors
from spencer.factor.ops import orient, winsorize_mad
from spencer.risk.style import build_styles, load_industry
from spencer.risk.neutral import residualize
from spencer.eval.panel import forward_return_1d, forward_returns, run_panel
from spencer.backtest.layered import layered_backtest
from spencer.discipline.ledger import log_run, trial_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()

    # ---- 数据 ----
    long_path = fetch_to_parquet(cfg, limit=args.limit)
    ind_csv = cfg["data_dir"] / "raw" / "industry.csv"
    if not ind_csv.exists():
        with BaostockSession():
            fetch_industry(ind_csv)
    store = build_wide_store(long_path, cfg["data_dir"] / "wide")

    tradable = (store.load("is_trading") == 1) & (store.load("is_st") == 0)
    industry = load_industry(ind_csv, store.load("close").columns)
    print("[risk] 构建五风格 (size/beta/momentum/volatility/liquidity)...")
    styles = build_styles(store)

    h = cfg["label"]["horizon"]
    fwd = forward_returns(store, h)
    fwd1 = forward_return_1d(store)
    print("[risk] 标签中性化 (行业+五风格残差收益)...")
    fwd_neut = residualize(fwd, styles, industry)

    ledger = cfg["root"] / "research_ledger.csv"
    rows = []
    for name in list_factors():
        base = winsorize_mad(compute(name, store, cfg["data_dir"] / "factors").where(tradable))
        modes = {
            "raw":       (base, fwd),
            "size_neut": (residualize(base, {"size": styles["size"]}), fwd),
            "full_neut": (residualize(base, styles, industry), fwd_neut),
        }
        for mode, (f, label) in modes.items():
            f, sign = orient(f, label)
            res = run_panel(f"{name}@{mode}", f, store, horizon=h,
                            q=cfg["eval"]["quantiles"], min_names=cfg["eval"]["min_names"],
                            outdir=cfg["output_dir"], fwd=label, fwd1=fwd1)
            res["orient_sign"] = sign
            if mode == "full_neut":                      # 交易发生在原始收益空间
                bt = layered_backtest(f, fwd1, q=cfg["eval"]["quantiles"],
                                      rebal_days=h, cost_bps=cfg["eval"]["cost_bps"])
                res["ls_net_ann"] = bt["ls_net"]["ann_ret"]
                res["ls_net_sharpe"] = bt["ls_net"]["sharpe"]
                res["turnover_per_rebal"] = bt["avg_turnover_per_rebal"]
                print(f"  [bt] 多空净年化 {bt['ls_net']['ann_ret']:.1%} "
                      f"sharpe {bt['ls_net']['sharpe']} 回撤 {bt['ls_net']['max_dd']:.1%} "
                      f"每次调仓换手 {bt['avg_turnover_per_rebal']:.0%}")
            log_run(ledger, res, params={"mode": mode, "universe": cfg["universe"],
                                         "limit": args.limit, "orient_sign": sign})
            rows.append({k: res.get(k) for k in
                         ("factor", "ic_mean", "ic_ir_daily", "t_stat_conservative",
                          "yearly_all_positive", "rank_autocorr_5d",
                          "ls_net_ann", "ls_net_sharpe")})

    print("\n======= 三档读数总表 =======")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n台账累计 N = {trial_count(ledger)}   面板图在 {cfg['output_dir']}/")


if __name__ == "__main__":
    main()
