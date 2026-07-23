"""调仓频率×缓冲区扫参 (v1.0 后首个优化项) + 噪声对照首次咬真数据。

背景: PIT 终跑里 top50 组合年化单边换手 15.4x, 是多头超额的主要漏点。
本脚本在同一信号(comp_eq_pit)上扫 调仓周期×缓冲区宽度, 找换手-alpha
的性价比拐点。同口径原则: 六格全部同信号/同成本/同宇宙/同窗口, 只动
两个待扫参数。

另: 用 discipline.noise 对 comp 信号做 20 臂噪声对照(经验 p 值) ——
静态阈值之外的动态尺子, 首次在全市场真数据上运行。

产出: output/rebal_sweep.csv + 控制台表格; 全部读数入台账。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spencer.config import load_config
from spencer.data.store import WideStore
from spencer.data.universe import build_pit_universe
from spencer.eval.panel import forward_return_1d, forward_returns
from spencer.factor import zoo  # noqa: F401
from spencer.factor.base import compute, list_factors
from spencer.factor.ops import orient, winsorize_mad
from spencer.risk.style import build_styles, load_industry
from spencer.risk.fundamental import build_value_style
from spencer.risk.neutral import residualize
from spencer.strategy.composite import equal_weight
from spencer.strategy.portfolio import topn_buffer_weights, turnover_series
from spencer.discipline.ledger import log_run, trial_count
from spencer.discipline.noise import noise_control

cfg = load_config()
store = WideStore(cfg["data_dir"] / "wide_pit")
mem = pd.read_parquet(cfg["data_dir"] / "raw" / "pit_membership.parquet")
up = cfg["universe_pit"]
uni_raw = build_pit_universe(mem, store, min_list_days=up["min_list_days"],
                             exclude_st=up["exclude_st"],
                             top_n_liquidity=up["top_n_liquidity"])
uni = uni_raw & (store.load("is_trading") == 1) & (store.load("is_st") == 0)
industry = load_industry(cfg["data_dir"] / "raw" / "industry.csv",
                         store.load("close").columns)
print("[sweep] 六风格...", flush=True)
styles = build_styles(store)
vb = build_value_style(store, cfg["root"] / cfg["fundamentals_cache"])
if vb is not None:
    styles["value_btop"] = vb

h = cfg["label"]["horizon"]
fwd = forward_returns(store, h).where(uni)
fwd1 = forward_return_1d(store)
print("[sweep] 标签残差化...", flush=True)
fwd_neut = residualize(fwd, styles, industry)

print("[sweep] 重建 comp_eq 信号...", flush=True)
oriented = {}
for name in list_factors():
    base = winsorize_mad(compute(name, store, cfg["data_dir"] / "factors_pit").where(uni))
    f = residualize(base, styles, industry)
    f, _ = orient(f, fwd_neut)
    oriented[name] = f
comp, _ = orient(equal_weight(oriented), fwd_neut)

print("[sweep] 噪声对照 20 臂(动态尺子)...", flush=True)
nc = noise_control(comp, fwd_neut, n_arms=20, seed=11)
print(f"  comp_eq |IC| {nc['real_abs_ic']} vs 噪声臂 max {nc['noise_abs_ic_max']} "
      f"→ 经验 p = {nc['p_value']}")

ledger = cfg["root"] / "research_ledger.csv"
cost = cfg["eval"]["cost_bps"] / 1e4
bench = fwd1.where(uni).mean(axis=1)

rows = []
for rebal in (5, 10, 20):
    for buffer_n in (80, 120):
        w = topn_buffer_weights(comp, top_n=50, buffer_n=buffer_n, rebal_days=rebal)
        port = (w * fwd1).sum(axis=1)
        to = turnover_series(w)
        net = (port - to * cost).dropna()
        ex = (net - bench.reindex(net.index)).dropna()
        r = {
            "rebal_days": rebal, "buffer_n": buffer_n,
            "net_ann": round(float(net.mean() * 252), 4),
            "net_sharpe": round(float(net.mean() / net.std() * np.sqrt(252)), 2),
            "excess_ann": round(float(ex.mean() * 252), 4),
            "excess_sharpe": round(float(ex.mean() / ex.std() * np.sqrt(252)), 2),
            "turnover_ann": round(float(to.mean() * 252), 1),
        }
        rows.append(r)
        log_run(ledger, {"factor": f"comp_eq_port_r{rebal}b{buffer_n}",
                         "ic_mean": r["excess_ann"]},
                params={"sweep": "rebal_buffer", **r},
                note="多头超额年化记入ic_mean列(组合读数, 口径见rebal_sweep)")

tbl = pd.DataFrame(rows)
tbl.to_csv(cfg["output_dir"] / "rebal_sweep.csv", index=False)
print("\n== 调仓频率×缓冲区 扫参(top50, 成本单边15bp, PIT宇宙) ==")
print(tbl.to_string(index=False))
print(f"\n台账 N = {trial_count(ledger)}")
print("SWEEP_DONE")
