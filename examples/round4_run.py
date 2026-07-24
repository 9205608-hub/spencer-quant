"""第四轮: ① τ 真前沿(单变量) + ② 动量崩盘防护证伪实验(预注册首用)。

① τ 扫描: 上轮发现 λ 在 τ 主导区失效, 本轮只动 τ∈{0, 5, 15, 50}bp,
   其余参数全部冻结(λ=2/cap4%/带0.30/α=滚动实测IC/rebal20d/成本15bp)。
   预期看到真前沿: τ→0 主动份额解锁、换手上升, 净超额存在内部峰。

② 预注册证伪: icir_weight 自适应降权能否在 2026-04~06 动量崩盘中保护
   合成信号? 研究员预测: 不能(252日窗+h+2移位的惯性跟不上三个月的崩盘
   +鞭打)。判据在计算之前用 preregister 冻结 —— 让系统裁决预测。
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
from spencer.strategy.composite import icir_weight
from spencer.factor.ops import orient
from spencer.discipline.ledger import log_run, trial_count
from spencer.discipline.preregister import create_ticket, evaluate

ctx = build_context(need_covariance=True)
cfg = ctx["cfg"]
h = ctx["h"]
ledger = cfg["root"] / "research_ledger.csv"
R: list[str] = [f"# 第四轮报告 ({datetime.now():%Y-%m-%d %H:%M})", ""]

# ================= ② 预注册在先: 动量崩盘防护证伪 =================
R.append("## ② icir 自适应加权的崩盘防护(预注册证伪实验)")
tickets = cfg["root"] / "tickets"
ticket_name = "icir_crash_protection"
existing = list(tickets.glob(f"*_{ticket_name}.json")) if tickets.exists() else []
if existing:
    tk = existing[0]
    R.append(f"- 工单已存在(判据冻结不重开): {tk.name}")
else:
    tk = create_ticket(
        tickets, ticket_name,
        hypothesis=("icir_weight(252日窗, h+2移位)不能在2026-04~06动量崩盘中"
                    "显著保护合成信号 —— 窗口惯性跟不上三个月崩盘+鞭打"),
        criteria={          # 「保护成立」必须同时满足的三关(数值≥阈值/布尔==True)
            "crash_ic_improve": 0.02,     # 崩盘窗(2026-04~06)平均IC至少改善+0.02
            "july_keep_ok": True,         # 7月鞭打反弹至少保住等权臂的一半
            "full_not_worse": True,       # 全样本IC相对等权臂回撤不超过0.002
        },
        trial_budget=1, stoploss="一次性判定, 不追加调参不换窗口",
        notes="研究员预测: 过不了闸(即保护不成立)。判据先于计算冻结。")
    R.append(f"- 工单已冻结: {tk.name}")

print("[round4] icir 合成...", flush=True)
comp_icir = icir_weight(ctx["oriented"], ctx["fwd_neut"], horizon=h)
comp_icir, _ = orient(comp_icir, ctx["fwd_neut"])

ic_eq = ic_series(ctx["comp"], ctx["fwd_neut"])
ic_ad = ic_series(comp_icir, ctx["fwd_neut"])
crash = slice("2026-04-01", "2026-06-30")
res2 = {
    "ic_full_eq": round(float(ic_eq.mean()), 4),
    "ic_full_icir": round(float(ic_ad.mean()), 4),
    "ic_crash_eq": round(float(ic_eq.loc[crash].mean()), 4),
    "ic_crash_icir": round(float(ic_ad.loc[crash].mean()), 4),
    "ic_jul_eq": round(float(ic_eq.loc["2026-07"].mean()), 4),
    "ic_jul_icir": round(float(ic_ad.loc["2026-07"].mean()), 4),
}
res2["crash_ic_improve"] = round(res2["ic_crash_icir"] - res2["ic_crash_eq"], 4)
res2["july_keep_ok"] = bool(res2["ic_jul_icir"] >= 0.5 * res2["ic_jul_eq"])
res2["full_not_worse"] = bool(res2["ic_full_icir"] >= res2["ic_full_eq"] - 0.002)
log_run(ledger, {"factor": ticket_name, "ic_mean": res2["crash_ic_improve"]},
        params=res2, note="崩盘防护证伪, crash_ic_improve记入ic_mean列")
verdict = evaluate(tk, res2, ledger_path=ledger, factor=ticket_name)
R.append(f"- 读数: {res2}")
R.append(f"- **工单裁决: {verdict['verdict'].upper()}** "
         f"(逐关: {{k: c['ok'] for k, c in verdict['checks'].items()}})"
         if verdict["verdict"] == "tampered" else
         f"- **工单裁决: {verdict['verdict'].upper()}** "
         f"(逐关: { {k: c['ok'] for k, c in verdict['checks'].items()} })")
R.append("")

# ================= ① τ 真前沿(单变量) =================
R.append("## ① τ 前沿(唯一自由参数=τ; λ=2/cap4%/带0.30/α=滚动实测IC 全冻结)")
ic_roll = (ic_series(ctx["comp"], ctx["fwd_neut"]).reindex(ctx["comp"].index)
           .shift(h + 2).rolling(252, min_periods=120).mean().clip(lower=0.0))
bench = ctx["fwd1"].where(ctx["uni"]).mean(axis=1)
rows = []
for tau_bp in (0, 5, 15, 50):
    tau = tau_bp / 1e4
    print(f"[tau前沿] τ={tau_bp}bp ...", flush=True)
    r = optimizer_backtest(ctx["comp"], ctx["fwd1"], ctx["uni"],
                           ctx["styles"], ctx["style_ret"], ctx["resid"],
                           rebal_days=cfg["portfolio"]["rebal_days"],
                           lam=2.0, tau=tau, cap=0.04, bound=0.30,
                           ic_prior=ic_roll, cost_bps=cfg["eval"]["cost_bps"])
    ex = (r["net_series"] - bench.reindex(r["net_series"].index)).dropna()
    row = {"tau_bp": tau_bp, "active_share": r["active_share_mean"],
           "turnover_ann": r["turnover_ann_oneside"],
           "excess_ann": round(float(ex.mean() * 252), 4),
           "excess_ir": round(float(ex.mean() / ex.std() * np.sqrt(252)), 2),
           "kkt_median": r["kkt_median"]}
    rows.append(row)
    log_run(ledger, {"factor": f"opt_tau{tau_bp}bp", "ic_mean": row["excess_ann"]},
            params=row, note="τ前沿臂, 超额年化记入ic_mean列")
tbl = pd.DataFrame(rows)
tbl.to_csv(cfg["output_dir"] / "round4_tau_frontier.csv", index=False)
R.append("```")
R.append(tbl.to_string(index=False))
R.append("```")
R.append(f"- 台账 N = {trial_count(ledger)}")

(cfg["output_dir"] / "round4_报告.md").write_text("\n".join(R), encoding="utf-8")
print("\n".join(R))
print("ROUND4_DONE")
