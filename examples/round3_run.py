"""第三轮优化: α 校准(滚动实测IC) + 主动度对齐的 λ 前沿 + 2026 回撤归因。

三个问题:
1. α 用滚动实测 IC(PIT 移位, 负值截0)替代统一先验 0.02, 优化器是否更聪明;
2. 把主动度放开(λ 从紧到松, cap 4%, 带 0.30), 优化器组合在"主动度-超额IR"
   前沿上与 topn r20/b80 是否可比 —— 上一轮的教训: 主动度不对齐不许并排;
3. comp_eq 2026 年回撤(终跑月度热力图 4-6 月连红)到底是谁在亏: 哪个成分
   因子死了 / 哪个风格今年在絞殺殘差空間。

产出: output/round3_报告.md + frontier csv + 台账。
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples._pit_common import build_context
from spencer.backtest.opt_bt import optimizer_backtest
from spencer.eval.panel import ic_series
from spencer.discipline.ledger import log_run, trial_count

ctx = build_context(need_covariance=True)
cfg = ctx["cfg"]
h = ctx["h"]
bench = ctx["fwd1"].where(ctx["uni"]).mean(axis=1)
ledger = cfg["root"] / "research_ledger.csv"
R: list[str] = [f"# 第三轮报告 ({datetime.now():%Y-%m-%d %H:%M})", ""]

# ---- ① α 校准: 滚动实测 IC, PIT 移位(标签完成日之后才可用), 负值截 0 ----
ic_raw = ic_series(ctx["comp"], ctx["fwd_neut"]).reindex(ctx["comp"].index)
ic_roll = ic_raw.shift(h + 2).rolling(252, min_periods=120).mean().clip(lower=0.0)
R.append("## ① α 校准 + ② 主动度前沿(τ=0.0015 固定, cap 4%, 暴露带 0.30)")
R.append(f"- 滚动实测 IC(可用段): 均值 {ic_roll.mean():.4f}, "
         f"最低 {ic_roll.min():.4f}(负值已截0), 最高 {ic_roll.max():.4f}")

rows, star = [], None
for lam in (0.5, 2.0, 8.0):
    print(f"[frontier] λ={lam} ...", flush=True)
    r = optimizer_backtest(ctx["comp"], ctx["fwd1"], ctx["uni"],
                           ctx["styles"], ctx["style_ret"], ctx["resid"],
                           rebal_days=cfg["portfolio"]["rebal_days"],
                           lam=lam, tau=0.0015, cap=0.04, bound=0.30,
                           ic_prior=ic_roll, cost_bps=cfg["eval"]["cost_bps"])
    ex = (r["net_series"] - bench.reindex(r["net_series"].index)).dropna()
    te = float(ex.std() * np.sqrt(252))
    row = {"lam": lam, "active_share": r["active_share_mean"],
           "excess_ann": round(float(ex.mean() * 252), 4),
           "excess_ir": round(float(ex.mean() / ex.std() * np.sqrt(252)), 2),
           "te_ann": round(te, 4), "turnover_ann": r["turnover_ann_oneside"],
           "kkt_median": r["kkt_median"]}
    rows.append(row)
    log_run(ledger, {"factor": f"opt_frontier_lam{lam}", "ic_mean": row["excess_ann"]},
            params=row, note="第三轮前沿臂, 超额年化记入ic_mean列")

tbl = pd.DataFrame(rows)
tbl.to_csv(cfg["output_dir"] / "round3_frontier.csv", index=False)
R.append("```")
R.append(tbl.to_string(index=False))
R.append("```")
R.append("- 参照(口径注记): topn r20/b80 超额 5.8%/IR 0.68, 主动份额≈0.96 "
         "(50只集中 vs ~1400只等权)。只有主动份额同量级的臂才可与之并排。")
R.append("")

# ---- ③ 2026 回撤归因 ----
R.append("## ③ 2026 回撤归因")
y = "2026"
fac_rows = []
for name, f in ctx["oriented"].items():
    s = ic_series(f, ctx["fwd_neut"])
    s26 = s.loc[y]
    fac_rows.append({"factor": name,
                     "ic_2026": round(float(s26.mean()), 4),
                     "ic_2026_h1_by_month": " ".join(
                         f"{m}:{v:+.3f}" for m, v in
                         s26.groupby(s26.index.month).mean().items()),
                     "ic_full": round(float(s.mean()), 4)})
comp26 = ic_series(ctx["comp"], ctx["fwd_neut"]).loc[y]
R.append(f"- comp_eq 2026 逐月 IC: " + " ".join(
    f"{m}:{v:+.3f}" for m, v in comp26.groupby(comp26.index.month).mean().items()))
R.append("```")
R.append(pd.DataFrame(fac_rows).to_string(index=False))
R.append("```")
sty26 = ctx["style_ret"].loc[y].cumsum().iloc[-1]
R.append("- 2026 风格因子累计收益(截面回归口径): "
         + "  ".join(f"{k} {v:+.3f}" for k, v in sty26.items()))
R.append("")
R.append(f"- 台账 N = {trial_count(ledger)}")

(cfg["output_dir"] / "round3_报告.md").write_text("\n".join(R), encoding="utf-8")
print("\n".join(R))
print("ROUND3_DONE")
