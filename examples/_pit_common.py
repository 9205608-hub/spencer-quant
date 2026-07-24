"""PIT 实验的公共装配层: store/宇宙/风格/标签/comp信号 一次构建, 多脚本复用。

rebal_sweep 与 opt_backtest_run 共用这份装配, 保证"同口径对比"不是口号:
两边的宇宙/风格/信号来自同一段代码。协方差输入(因子收益/残差)带磁盘缓存
(data/covariance/), 首次构建 ~5 分钟, 之后秒读。
"""
from __future__ import annotations

import sys
from pathlib import Path

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
from spencer.risk.covariance import style_factor_returns
from spencer.strategy.composite import equal_weight


def build_context(need_covariance: bool = False) -> dict:
    cfg = load_config()
    store = WideStore(cfg["data_dir"] / "wide_pit")
    mem = pd.read_parquet(cfg["data_dir"] / "raw" / "pit_membership.parquet")
    up = cfg["universe_pit"]
    uni = build_pit_universe(mem, store, min_list_days=up["min_list_days"],
                             exclude_st=up["exclude_st"],
                             top_n_liquidity=up["top_n_liquidity"])
    uni = uni & (store.load("is_trading") == 1) & (store.load("is_st") == 0)
    industry = load_industry(cfg["data_dir"] / "raw" / "industry.csv",
                             store.load("close").columns)
    print("[ctx] 六风格...", flush=True)
    styles = build_styles(store)
    vb = build_value_style(store, cfg["root"] / cfg["fundamentals_cache"])
    if vb is not None:
        styles["value_btop"] = vb

    h = cfg["label"]["horizon"]
    fwd = forward_returns(store, h).where(uni)
    fwd1 = forward_return_1d(store)
    print("[ctx] 标签残差化...", flush=True)
    fwd_neut = residualize(fwd, styles, industry)

    print("[ctx] comp_eq 信号...", flush=True)
    oriented = {}
    for name in list_factors():
        base = winsorize_mad(compute(name, store, cfg["data_dir"] / "factors_pit").where(uni))
        f = residualize(base, styles, industry)
        f, _ = orient(f, fwd_neut)
        oriented[name] = f
    comp, _ = orient(equal_weight(oriented), fwd_neut)

    ctx = dict(cfg=cfg, store=store, uni=uni, industry=industry, styles=styles,
               fwd=fwd, fwd1=fwd1, fwd_neut=fwd_neut, comp=comp, h=h,
               oriented=oriented)

    if need_covariance:
        cov_dir = cfg["data_dir"] / "covariance"
        cov_dir.mkdir(parents=True, exist_ok=True)
        sr_p, rs_p = cov_dir / "style_ret.parquet", cov_dir / "resid.parquet"
        if sr_p.exists() and rs_p.exists():
            style_ret, resid = pd.read_parquet(sr_p), pd.read_parquet(rs_p)
            print(f"[ctx] 协方差输入命中缓存 ({len(style_ret)} 日)")
        else:
            print("[ctx] 因子收益全窗截面回归(首次 ~5min)...", flush=True)
            style_ret, resid = style_factor_returns(fwd1.where(uni), styles, industry)
            style_ret.to_parquet(sr_p)
            resid.to_parquet(rs_p)
        ctx["style_ret"], ctx["resid"] = style_ret, resid
    return ctx
