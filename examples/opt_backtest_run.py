"""优化器逐期回测·真数据双臂: 事后扣费 vs 成本进目标函数(τ=费率)。

对照设计(同口径: 同信号/同 Σ/同约束/同费率, 只动 τ):
  臂A τ=0      : 优化器不知道成本, 事后扣费 —— 传统做法;
  臂B τ=0.0015 : 单边费率写进目标函数的 L1 项 —— 优化器自己权衡"这次换
                 仓赚的 α 够不够付成本"。
参照系: rebal_sweep 的 topn r20/b80(超额 5.8%/sharpe 0.68/换手 6.8x) 与
宇宙等权基准。产出: output/opt_backtest.csv + 净值图 + 台账。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples._pit_common import build_context
from spencer.backtest.opt_bt import optimizer_backtest
from spencer.discipline.ledger import log_run, trial_count

ctx = build_context(need_covariance=True)
cfg = ctx["cfg"]
pf = cfg["portfolio"]
bench = ctx["fwd1"].where(ctx["uni"]).mean(axis=1)

rows, curves = [], {}
for tag, tau in (("A_cost_after", 0.0), ("B_cost_in_objective", 0.0015)):
    print(f"[arm {tag}] 求解 ~{2562 // pf['rebal_days']} 期...", flush=True)
    r = optimizer_backtest(ctx["comp"], ctx["fwd1"], ctx["uni"],
                           ctx["styles"], ctx["style_ret"], ctx["resid"],
                           rebal_days=pf["rebal_days"], lam=20.0, tau=tau,
                           cap=0.02, bound=0.10,
                           cost_bps=cfg["eval"]["cost_bps"])
    ex = (r["net_series"] - bench.reindex(r["net_series"].index)).dropna()
    row = {
        "arm": tag, "tau": tau,
        "net_ann": r["net_ann"], "net_sharpe": r["net_sharpe"],
        "excess_ann": round(float(ex.mean() * 252), 4),
        "excess_sharpe": round(float(ex.mean() / ex.std() * np.sqrt(252)), 2),
        "turnover_ann": r["turnover_ann_oneside"],
        "n_rebal": r["n_rebalances"], "kkt_median": r["kkt_median"],
    }
    rows.append(row)
    curves[tag] = ex.cumsum()
    log_run(cfg["root"] / "research_ledger.csv",
            {"factor": f"opt_bt_{tag}", "ic_mean": row["excess_ann"]},
            params=row, note="优化器逐期回测, 超额年化记入ic_mean列")

tbl = pd.DataFrame(rows)
tbl.to_csv(cfg["output_dir"] / "opt_backtest.csv", index=False)
print("\n== 优化器逐期回测(PIT宇宙, λ=20, cap2%, 暴露带0.10, 成本15bp) ==")
print(tbl.to_string(index=False))
print("\n参照: topn r20/b80 超额 5.8%/sharpe 0.68/换手 6.8x (rebal_sweep)")

fig, ax = plt.subplots(figsize=(9, 5))
for tag, c in curves.items():
    c.plot(ax=ax, lw=1.2, label=f"{tag} (excess cum)")
ax.axhline(0, color="gray", lw=0.6)
ax.legend()
ax.set_title("optimizer backtest: excess vs universe equal-weight (net)")
fig.tight_layout()
fig.savefig(cfg["output_dir"] / "opt_backtest_excess.png", dpi=120)
print(f"\n台账 N = {trial_count(cfg['root'] / 'research_ledger.csv')}")
print("OPT_BT_DONE")
