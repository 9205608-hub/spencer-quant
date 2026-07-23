"""M9 收尾接线 demo: 风险模型(covariance) × 优化器(FISTA) 在真实 PIT 数据上闭环。

演示三件事(这是机制演示, 不是策略——alpha 的量纲是示意性的, 明示):
1. Σ = B F B' + diag(spec²) 从真实数据估出来并直接喂 optimizer.solve;
2. λ 从松到紧, 组合从"追 alpha"滑向"贴基准"(风险厌恶旋钮的实相);
3. λ→∞ 恒等式: w 收敛回 w0 —— 优化器正确性的现场检验。

前置: data/wide_pit 与 data/raw/pit_membership.parquet 已就位(pit_final_run 产物)。
产出: output/optimizer_demo.txt
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
from spencer.eval.panel import forward_return_1d
from spencer.risk.style import build_styles, load_industry
from spencer.risk.covariance import sigma_at, style_factor_returns
from spencer.strategy.optimizer import solve

cfg = load_config()
store = WideStore(cfg["data_dir"] / "wide_pit")
mem = pd.read_parquet(cfg["data_dir"] / "raw" / "pit_membership.parquet")
up = cfg["universe_pit"]
uni = build_pit_universe(mem, store, min_list_days=up["min_list_days"],
                         exclude_st=up["exclude_st"],
                         top_n_liquidity=up["top_n_liquidity"])
industry = load_industry(cfg["data_dir"] / "raw" / "industry.csv",
                         store.load("close").columns)
print("[demo] 五风格(demo 省略 BTOP, 明示)...")
styles = build_styles(store)

fwd1 = forward_return_1d(store).where(uni)
window = fwd1.index[-520:]
print("[demo] 因子收益: 近520日截面回归...")
style_ret, resid = style_factor_returns(fwd1.loc[window], styles, industry)

asof = store.load("close").index[-1]
B, F, spec, codes = sigma_at(asof, styles, style_ret, resid, min_obs=100)
in_uni = uni.loc[asof].reindex(codes).fillna(False).to_numpy()
B, spec = B[in_uni], spec[in_uni]
codes = [c for c, k in zip(codes, in_uni) if k]
n = len(codes)
print(f"[demo] asof {asof.date()}: n={n}, k={B.shape[1]}, "
      f"日频特异波动率中位 {np.median(spec):.4f}")

# 示意性 alpha: 三个 PIT 幸存因子(rev_5/chip_age/turn_surge)缓存值的截面
# 排名等权, zscore 后乘 5bp 日频量纲 —— 量纲是演示用的, 不构成收益预测
ranks = []
for fac in ("rev_5", "chip_age", "turn_surge_5_60"):
    df = pd.read_parquet(cfg["data_dir"] / "factors_pit" / f"{fac}.parquet")
    ranks.append(df.loc[asof].rank(pct=True))
sig = sum(ranks).reindex(codes)
sig = (sig - sig.mean()) / sig.std()
alpha = (sig.fillna(0.0) * 5e-4).to_numpy()

w0 = np.full(n, 1.0 / n)
lines = [f"# optimizer demo  asof {asof.date()}  n={n}",
         f"约束: 单纯形 + cap 2% + 风格暴露带 |B'(w-w0)|≤0.10, τ=0", ""]
for lam in (5.0, 50.0, 1e9):
    r = solve(alpha, (B, F, spec), w0, w0, lam=lam, tau=0.0, cap=0.02,
              B=B, bound=0.10, max_iter=20000)
    w = np.asarray(r["w"], float)
    active = w - w0
    expo = np.abs(r.get("style_exposure", B.T @ active)).max()
    lines.append(
        f"λ={lam:g}: alpha'w={float(alpha @ w):.2e}  主动风险={float(active @ (B @ (F @ (B.T @ active)) + spec**2 * active))**0.5:.2e}  "
        f"max|w-w0|={np.abs(active).max():.2e}  最大风格暴露={expo:.3f}  "
        f"持仓>1bp {int((w > 1e-4).sum())} 只  KKT={r['kkt_residual']:.1e}  iter={r['n_iter']}")
identity_ok = np.abs(active).max() < 1e-5
lines.append("")
lines.append(f"λ→∞ 恒等式(w→w0): {'通过' if identity_ok else '未通过'}")
out = cfg["output_dir"] / "optimizer_demo.txt"
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
assert identity_ok, "λ→∞ 恒等式未通过"
print(f"\nOPT_DEMO_DONE → {out}")
