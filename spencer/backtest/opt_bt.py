"""优化器驱动的逐期回测: 信号 → (α, Σ, 约束) → FISTA 求解 → 净值曲线。

与 layered/topn_buffer 的本质区别: 组合不再是"排序取前N", 而是逐期解
max α'w − λ(w−w0)'Σ(w−w0) − τ‖w−w_prev‖₁, 风险与换手直接进目标函数。

口径(全部明示):
- α 用 Grinold 规则(公开): α_i = IC_prior × spec_i × z_i, 其中 z=信号截面
  zscore, spec=特异波动率, IC_prior 是先验IC(默认0.02, 参数不是拟合值);
- w0 = 当期宇宙等权(主动风险相对它度量); w_prev = 上期目标权重按当前
  股票集 reindex(缺失→0)。已知近似: 不建模期间漂移(drift), 换手略被低估;
- 权重在信号日 t 定, 收益按 fwd1[t] 记(T+1建仓口径, 与全仓一致);
- 成本 = 单边费率 × Σ|Δw|(买卖各付单边), 调仓日一次性扣;
- τ 是优化器内的换手惩罚(事前), cost_bps 是事后计账 —— 两者独立设置,
  τ=费率 即"成本进目标函数"的经济学设定。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..risk.covariance import sigma_at
from ..strategy.optimizer import solve


def optimizer_backtest(signal: pd.DataFrame, fwd1: pd.DataFrame,
                       universe: pd.DataFrame,
                       styles: dict, style_ret: pd.DataFrame, resid: pd.DataFrame,
                       rebal_days: int = 20, lam: float = 20.0, tau: float = 0.0,
                       cap: float = 0.02, bound: float = 0.10,
                       ic_prior: float = 0.02, cost_bps: float = 15.0,
                       max_iter: int = 6000, warmup_obs: int = 150,
                       min_members: int = 100, verbose_every: int = 10) -> dict:
    rate = cost_bps / 1e4
    idx = signal.index.intersection(fwd1.index)
    rebal_dates = [dt for dt in idx[::rebal_days]]

    weights = pd.DataFrame(0.0, index=idx, columns=signal.columns)
    prev_w: pd.Series | None = None
    costs, kkts, solved = {}, [], 0

    for ri, dt in enumerate(rebal_dates):
        try:
            B, F, spec, codes = sigma_at(dt, styles, style_ret, resid,
                                         min_obs=warmup_obs)
        except ValueError:
            continue                                     # 协方差暖机期, 跳过
        members = universe.loc[dt]
        sig = signal.loc[dt]
        keep = [c for c in codes if bool(members.get(c, False))
                and pd.notna(sig.get(c))]
        if len(keep) < min_members:
            continue
        pos = [codes.index(c) for c in keep]
        B_k, spec_k = B[pos], spec[pos]

        z = sig[keep]
        z = ((z - z.mean()) / z.std()).to_numpy()
        alpha = ic_prior * spec_k * z                    # Grinold: α = IC·σ·z
        w0 = np.full(len(keep), 1.0 / len(keep))
        wp = (prev_w.reindex(keep).fillna(0.0).to_numpy()
              if prev_w is not None else w0)

        r = solve(alpha, (B_k, F, spec_k), w0, wp, lam=lam, tau=tau,
                  cap=cap, B=B_k, bound=bound, max_iter=max_iter)
        w = pd.Series(np.asarray(r["w"], float), index=keep)
        kkts.append(r["kkt_residual"])
        solved += 1

        prev_full = prev_w.reindex(w.index).fillna(0.0) if prev_w is not None \
            else pd.Series(0.0, index=w.index)
        dropped = (prev_w.drop(index=[c for c in prev_w.index if c in w.index]).abs().sum()
                   if prev_w is not None else 0.0)
        trade = float((w - prev_full).abs().sum() + dropped)
        costs[dt] = rate * trade

        span = idx[(idx >= dt)]
        nxt = rebal_dates[ri + 1] if ri + 1 < len(rebal_dates) else None
        span = span[span < nxt] if nxt is not None else span
        weights.loc[span, w.index] = w.values
        prev_w = w
        if verbose_every and solved % verbose_every == 0:
            print(f"  [opt_bt λ={lam} τ={tau}] {solved}/{len(rebal_dates)} "
                  f"{dt.date()} n={len(keep)} kkt={r['kkt_residual']:.1e}", flush=True)

    port = (weights * fwd1).sum(axis=1)
    cost_s = pd.Series(costs, dtype=float).reindex(idx).fillna(0.0)
    net = (port - cost_s).loc[weights.abs().sum(axis=1) > 0].dropna()
    to_ann = float(pd.Series(costs).sum() / rate / max(len(net) / 252, 1e-9)) / 2

    def ann(r):
        return round(float(r.mean() * 252), 4), \
            round(float(r.mean() / r.std() * np.sqrt(252)), 2)

    net_ann, net_sh = ann(net)
    return {"net_series": net, "weights": weights,
            "net_ann": net_ann, "net_sharpe": net_sh,
            "turnover_ann_oneside": round(to_ann, 1),
            "n_rebalances": solved,
            "kkt_median": round(float(np.median(kkts)), 8) if kkts else None}
