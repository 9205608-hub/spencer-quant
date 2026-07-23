"""夜班总控: 长历史全量 → 7因子×3档 → 合成 → 组合 → 模型因子 → 晨报。

设计原则:
- 每个阶段独立 try/except, 单阶段失败不拖垮整晚, 失败原因写进晨报;
- 全部读数进台账(N 如实增长);
- 晨报 = output/夜班报告.md, 人一早直接读。
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spencer.config import load_config
from spencer.data.fetch import fetch_to_parquet
from spencer.data.store import build_wide_store
from spencer.factor import zoo  # noqa: F401
from spencer.factor.base import compute, list_factors
from spencer.factor.ops import orient, winsorize_mad
from spencer.risk.style import build_styles, load_industry
from spencer.risk.neutral import residualize
from spencer.eval.panel import forward_return_1d, forward_returns, run_panel
from spencer.backtest.layered import layered_backtest
from spencer.strategy.composite import equal_weight, icir_weight
from spencer.strategy.portfolio import topn_buffer_weights, turnover_series, export_targets
from spencer.discipline.ledger import log_run, trial_count

REPORT: list[str] = []
FAILURES: list[str] = []


def section(title: str):
    print(f"\n########## {title} ##########")
    REPORT.append(f"\n## {title}\n")


def add(text: str):
    REPORT.append(text)


def guard(title, fn):
    try:
        return fn()
    except Exception:
        tb = traceback.format_exc()
        print(f"[FAIL] {title}\n{tb}")
        FAILURES.append(f"### {title}\n```\n{tb[-1500:]}\n```")
        return None


def main():
    cfg = load_config()
    out = cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    ledger = cfg["root"] / "research_ledger.csv"
    h = cfg["label"]["horizon"]
    q = cfg["eval"]["quantiles"]

    # ---------- 数据 ----------
    section("数据")
    long_path = fetch_to_parquet(cfg)
    store = build_wide_store(long_path, cfg["data_dir"] / "wide")
    close = store.load("close")
    tradable = (store.load("is_trading") == 1) & (store.load("is_st") == 0)
    industry = load_industry(cfg["data_dir"] / "raw" / "industry.csv", close.columns)
    add(f"- 宇宙 {cfg['universe']}: {close.shape[1]} 只 × {close.shape[0]} 交易日, "
        f"{close.index[0].date()} → {close.index[-1].date()}(末端)\n"
        f"- 已知欠账: 成分与行业均为快照口径(幸存者偏差, 30问#6/#31), "
        f"读数只作框架验证, 不作 alpha 宣言\n")

    styles = build_styles(store)
    fwd = forward_returns(store, h)
    fwd1 = forward_return_1d(store)
    print("[risk] 标签中性化...")
    fwd_neut = residualize(fwd, styles, industry)

    # ---------- 因子 × 三档 ----------
    section("单因子三档读数")
    rows, fn_factors, feat_raw = [], {}, {}

    def eval_factor(name):
        base = winsorize_mad(compute(name, store, cfg["data_dir"] / "factors").where(tradable))
        feat_raw[name] = base
        modes = {
            "raw": (base, fwd),
            "size_neut": (residualize(base, {"size": styles["size"]}), fwd),
            "full_neut": (residualize(base, styles, industry), fwd_neut),
        }
        for mode, (f, label) in modes.items():
            f, sign = orient(f, label)
            res = run_panel(f"{name}@{mode}", f, store, horizon=h, q=q,
                            min_names=cfg["eval"]["min_names"], outdir=out,
                            fwd=label, fwd1=fwd1)
            res["orient_sign"] = sign
            if mode == "full_neut":
                bt = layered_backtest(f, fwd1, q=q, rebal_days=h,
                                      cost_bps=cfg["eval"]["cost_bps"])
                res["ls_net_ann"], res["ls_net_sharpe"] = bt["ls_net"]["ann_ret"], bt["ls_net"]["sharpe"]
                res["turnover_per_rebal"] = bt["avg_turnover_per_rebal"]
                fn_factors[name] = f
            log_run(ledger, res, params={"mode": mode, "run": "night_2016",
                                         "universe": cfg["universe"], "orient_sign": sign})
            rows.append(res)

    for name in list_factors():
        print(f"== {name} ==")
        guard(f"factor:{name}", lambda n=name: eval_factor(n))

    summary = pd.DataFrame(rows)[[
        "factor", "ic_mean", "ic_ir_daily", "t_stat_conservative",
        "yearly_all_positive", "rank_autocorr_5d", "ls_net_ann", "ls_net_sharpe"]]
    summary.to_csv(out / "summary_factors.csv", index=False)
    add("```\n" + summary.to_string(index=False) + "\n```\n")

    # ---------- 合成 ----------
    section("多因子合成 (M9)")

    def do_composites():
        comps = {
            "comp_eq": equal_weight(fn_factors),
            "comp_icir": icir_weight(fn_factors, fwd_neut, horizon=h),
        }
        lines = []
        for cname, sig in comps.items():
            sig, sign = orient(sig, fwd_neut)
            res = run_panel(cname, sig, store, horizon=h, q=q,
                            min_names=cfg["eval"]["min_names"], outdir=out,
                            fwd=fwd_neut, fwd1=fwd1)
            bt = layered_backtest(sig, fwd1, q=q, rebal_days=h,
                                  cost_bps=cfg["eval"]["cost_bps"])
            res.update(ls_net_ann=bt["ls_net"]["ann_ret"], ls_net_sharpe=bt["ls_net"]["sharpe"])
            log_run(ledger, res, params={"mode": "composite", "run": "night_2016"})
            lines.append(f"- **{cname}**: IC {res['ic_mean']}, ICIR日频 {res['ic_ir_daily']}, "
                         f"保守t {res['t_stat_conservative']}, 多空净年化 {bt['ls_net']['ann_ret']:.1%}, "
                         f"sharpe {bt['ls_net']['sharpe']}")
        add("\n".join(lines) + "\n")
        return comps

    comps = guard("composites", do_composites) or {}

    # ---------- 组合 (执行接口) ----------
    section("组合构造与执行接口 (M9)")

    def do_portfolio():
        sig = comps.get("comp_icir")
        if sig is None or sig.notna().sum().sum() == 0:
            sig = comps["comp_eq"]
        w = topn_buffer_weights(sig.where(tradable), top_n=50, buffer_n=80, rebal_days=h)
        cost = cfg["eval"]["cost_bps"] / 1e4
        gross = (w * fwd1).sum(axis=1)
        fee = w.diff().abs().sum(axis=1) * cost
        net = (gross - fee)[w.sum(axis=1) > 0.99]
        bench = fwd1.where(tradable).mean(axis=1).reindex(net.index)
        excess = net - bench

        def m(r):
            ann = r.mean() * 252
            sh = ann / (r.std() * np.sqrt(252))
            cum = r.cumsum()
            return f"年化 {ann:.1%}, sharpe {sh:.2f}, 最大回撤 {(cum-cum.cummax()).min():.1%}"

        to_ann = float(turnover_series(w).mean() * 252)
        last_dt = net.index[-1]
        tbl = export_targets(w, last_dt, out / "target_holdings_latest.csv")
        add(f"- 组合: comp_icir top50/buffer80, {h}日调仓, 成本单边 {cfg['eval']['cost_bps']}bps\n"
            f"- 多头净: {m(net)}\n- 等权基准: {m(bench.dropna())}\n- 超额: {m(excess.dropna())}\n"
            f"- 年化单边换手 {to_ann:.1f}x\n"
            f"- 执行接口示例: {last_dt.date()} 目标持仓 {len(tbl)} 只 → target_holdings_latest.csv\n")
        pd.DataFrame({"net": net, "bench": bench, "excess": excess}).cumsum().to_csv(
            out / "portfolio_curves.csv")

    guard("portfolio", do_portfolio)

    # ---------- 模型因子 ----------
    section("模型因子 (M8 初版)")

    def do_model():
        from spencer.model.ml_factor import walk_forward_model_factor
        features = {**feat_raw, **styles}
        mf = walk_forward_model_factor(features, fwd_neut, train_years=3,
                                       embargo_days=h + 5)
        mf = mf.where(tradable)
        mf, sign = orient(mf, fwd_neut)
        res = run_panel("model_gb", mf, store, horizon=h, q=q,
                        min_names=cfg["eval"]["min_names"], outdir=out,
                        fwd=fwd_neut, fwd1=fwd1)
        bt = layered_backtest(mf, fwd1, q=q, rebal_days=h, cost_bps=cfg["eval"]["cost_bps"])
        log_run(ledger, {**res, "ls_net_ann": bt["ls_net"]["ann_ret"]},
                params={"mode": "model_gb", "run": "night_2016"})
        add(f"- model_gb(默认超参, 3年滚动, embargo {h+5}d): IC {res['ic_mean']}, "
            f"ICIR日频 {res['ic_ir_daily']}, 保守t {res['t_stat_conservative']}, "
            f"多空净年化 {bt['ls_net']['ann_ret']:.1%}, sharpe {bt['ls_net']['sharpe']}\n")

    guard("model_gb", do_model)

    # ---------- 晨报 ----------
    section("台账与诚实声明")
    add(f"- 台账累计试验数 N = {trial_count(ledger)}(含昨日短窗 run 与今晚全部)\n"
        f"- 快照宇宙 = 幸存者偏差, 所有正读数系统性偏乐观; 行业为快照; "
        f"orient 为样本内决策(每条已披露符号)\n"
        f"- 跨框架数字永不并排(见 docs/与工业级框架的能力对照.md)\n")
    if FAILURES:
        REPORT.append("\n## 阶段失败记录\n" + "\n".join(FAILURES))

    stamp = pd.Timestamp.today().strftime("%Y-%m-%d %H:%M")
    header = (f"# Spencer 框架夜班报告 ({stamp})\n\n"
              f"长历史 a800 全量: 7 因子 × 3 档 + 合成 + 组合 + 模型因子。"
              f"面板图 output/panel_*.png, 汇总 summary_factors.csv。\n")
    (out / "夜班报告.md").write_text(header + "".join(REPORT), encoding="utf-8")
    print(f"\nNIGHT_RUN_DONE  报告: {out/'夜班报告.md'}")


if __name__ == "__main__":
    main()
