"""M9 组合优化器断言: 恒等式 / SLSQP 对照 / 约束满足 / KKT / 暴露带单调。

跑法: python3 tests/test_m9_opt.py  (合成数据, 不碰网络)
依赖: numpy, pandas; SLSQP 对照组用 scipy(仅测试依赖, 优化器本体零 scipy)。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spencer.strategy.optimizer import prox_capped_simplex_l1, solve

from scipy.optimize import minimize


# ---------- 合成问题 ----------

def make_problem(n, seed):
    """良态稠密协方差 + 可行的 w0/w_prev。量纲对齐日频: Σ ~ 1e-2 量级。"""
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(n, 2 * n)) / np.sqrt(2 * n)
    Sigma = 0.01 * (C @ C.T) + 0.005 * np.eye(n)
    alpha = rng.normal(0.02, 0.02, n)
    w0 = np.full(n, 1.0 / n)
    cap = np.full(n, 3.0 / n)
    w_prev = prox_capped_simplex_l1(
        w0 + rng.normal(0, 0.3 / n, n), w0, 0.0, cap)
    return alpha, Sigma, w0, w_prev, cap


def slsqp_qp(alpha, Sigma, w0, lam, cap):
    """SLSQP 参照解: 同目标(τ=0, 光滑), 同约束(盒 + 和=1)。"""
    n = len(alpha)

    def f(w):
        d = w - w0
        return -(alpha @ w - lam * d @ Sigma @ d)

    def jac(w):
        d = w - w0
        return -(alpha - 2.0 * lam * (Sigma @ d))

    res = minimize(f, np.full(n, 1.0 / n), jac=jac, method="SLSQP",
                   bounds=[(0.0, c) for c in cap],
                   constraints=[{"type": "eq",
                                 "fun": lambda w: w.sum() - 1.0,
                                 "jac": lambda w: np.ones(n)}],
                   options={"maxiter": 1000, "ftol": 1e-14})
    assert res.success, res.message
    return res.x, -res.fun


# ---------- ① 恒等式: λ→∞ ⇒ w→w0 ----------

def test_identity_lambda_huge():
    alpha, Sigma, w0, w_prev, cap = make_problem(60, seed=0)
    for tau in (0.0, 1e-3):
        res = solve(alpha, Sigma, w0, w_prev, lam=1e9, tau=tau, cap=cap)
        dmax = np.max(np.abs(res["w"] - w0))
        assert dmax < 1e-6, f"λ=1e9 应回到 w0, max|Δ|={dmax:.2e} (tau={tau})"
    print("identity(λ=1e9 → w=w0) OK")


# ---------- ② τ=0 小规模与 SLSQP 同目标值 ----------

def test_matches_slsqp():
    for seed in (1, 2, 3):
        alpha, Sigma, w0, w_prev, cap = make_problem(8, seed=seed)
        lam = 5.0
        res = solve(alpha, Sigma, w0, w_prev, lam=lam, tau=0.0, cap=cap)
        w_ref, obj_ref = slsqp_qp(alpha, Sigma, w0, lam, cap)
        rel = abs(res["objective"] - obj_ref) / max(abs(obj_ref), 1e-8)
        assert rel < 1e-4, f"与SLSQP目标值相对差 {rel:.2e} (seed={seed})"
        assert res["objective"] >= obj_ref - 1e-8, "FISTA 目标值不应劣于参照解"
    print("slsqp_match(3 seeds, rel<1e-4) OK")


# ---------- ③④ 约束满足 + KKT 残差 ----------

def test_constraints_and_kkt():
    rng = np.random.default_rng(4)
    alpha, Sigma, w0, w_prev, cap = make_problem(120, seed=4)
    cap = np.full(120, 2.0 / 120) + rng.uniform(0, 2.0 / 120, 120)  # 非均匀上限
    B = rng.normal(size=(120, 2))
    B /= np.linalg.norm(B, axis=0)                                  # 列标准化
    res = solve(alpha, Sigma, w0, w_prev, lam=5.0, tau=2e-3, cap=cap,
                B=B, bound=0.05)
    w = res["w"]
    assert np.all(w >= -1e-10), f"w 下界违反 {w.min():.2e}"
    assert np.all(w <= cap + 1e-10), f"cap 违反 {(w - cap).max():.2e}"
    assert abs(w.sum() - 1.0) < 1e-8, f"权重和 {w.sum():.12f} ≠ 1"
    assert res["kkt_residual"] < 1e-5, f"KKT 残差 {res['kkt_residual']:.2e}"
    assert res["converged"]
    print(f"constraints+kkt OK (kkt={res['kkt_residual']:.1e}, "
          f"n_iter={res['n_iter']})")


# ---------- ⑤ 暴露带罚项生效: bound 收紧 → 暴露单调变小 ----------

def test_style_band_monotone():
    n = 80
    rng = np.random.default_rng(5)
    _, Sigma, w0, w_prev, _ = make_problem(n, seed=5)
    cap = np.full(n, 5.0 / n)
    b_raw = rng.normal(size=n)
    z = (b_raw - b_raw.mean()) / b_raw.std()
    B = (z / np.linalg.norm(z)).reshape(-1, 1)     # 单位范数列
    alpha = 0.04 * z + 0.01 * rng.normal(size=n)   # 信号故意压在风格上

    base = solve(alpha, Sigma, w0, w_prev, lam=2.0, tau=0.0, cap=cap)
    expo_free = abs(float((B.T @ (np.asarray(base["w"]) - w0))[0]))

    bounds = [0.5 * expo_free, 0.25 * expo_free, 0.1 * expo_free,
              0.05 * expo_free, 0.02 * expo_free]
    expos = []
    for b in bounds:
        r = solve(alpha, Sigma, w0, w_prev, lam=2.0, tau=0.0, cap=cap,
                  B=B, bound=b)
        expos.append(abs(float(r["style_exposure"][0])))
    for e_loose, e_tight in zip(expos, expos[1:]):
        assert e_tight <= e_loose + 1e-8, f"bound收紧暴露反增: {expos}"
    assert expos[-1] < 0.5 * expo_free, \
        f"最紧带暴露 {expos[-1]:.4f} 未显著低于无带 {expo_free:.4f}"
    print(f"style_band OK (无带 {expo_free:.3f} → 阶梯 "
          + " ".join(f"{e:.3f}" for e in expos) + ")")


# ---------- prox 精确性: 对照 SLSQP 解同一个子问题 ----------

def test_prox_exact():
    n = 6
    rng = np.random.default_rng(6)
    v = rng.normal(0, 0.5, n)
    p = prox_capped_simplex_l1(rng.normal(0, 0.3, n),
                               np.zeros(n), 0.0, np.full(n, 0.4))
    cap = np.full(n, 0.4)
    for t in (0.0, 0.05):        # t=0 即纯截断单纯形投影
        w = prox_capped_simplex_l1(v, p, t, cap)

        # SLSQP 参照: min ½‖w−v‖² + t·Σu, u ≥ |w−p| (拆变量化掉绝对值)
        def f(zz):
            return 0.5 * ((zz[:n] - v) ** 2).sum() + t * zz[n:].sum()

        res = minimize(
            f, np.concatenate([np.full(n, 1.0 / n), np.ones(n)]),
            method="SLSQP",
            bounds=[(0.0, c) for c in cap] + [(0.0, None)] * n,
            constraints=[
                {"type": "eq", "fun": lambda zz: zz[:n].sum() - 1.0},
                {"type": "ineq", "fun": lambda zz: zz[n:] - (zz[:n] - p)},
                {"type": "ineq", "fun": lambda zz: zz[n:] + (zz[:n] - p)},
            ],
            options={"maxiter": 2000, "ftol": 1e-14})
        assert res.success, res.message
        dmax = np.max(np.abs(w - res.x[:n]))
        assert dmax < 5e-6, f"prox 与 SLSQP 差 {dmax:.2e} (t={t})"
        assert abs(w.sum() - 1.0) < 1e-10 and np.all(w >= 0) \
            and np.all(w <= cap + 1e-12)
    print("prox_exact OK")


# ---------- 低秩表示与稠密同解 ----------

def test_lowrank_equals_dense():
    n, k = 50, 3
    rng = np.random.default_rng(7)
    alpha, _, w0, w_prev, cap = make_problem(n, seed=7)
    Br = rng.normal(size=(n, k)) / np.sqrt(n)
    F = np.diag(rng.uniform(0.5, 2.0, k))
    spec = rng.uniform(0.05, 0.15, n)
    dense = Br @ F @ Br.T + np.diag(spec ** 2)

    r1 = solve(alpha, dense, w0, w_prev, lam=5.0, tau=1e-3, cap=cap)
    r2 = solve(alpha, (Br, F, spec), w0, w_prev, lam=5.0, tau=1e-3, cap=cap)
    r3 = solve(alpha, {"B": Br, "F": F, "spec": spec}, w0, w_prev,
               lam=5.0, tau=1e-3, cap=cap)
    d12 = np.max(np.abs(r1["w"] - r2["w"]))
    d23 = np.max(np.abs(r2["w"] - r3["w"]))
    assert d12 < 1e-6 and d23 < 1e-12, f"低秩/稠密不同解 {d12:.2e} {d23:.2e}"
    print(f"lowrank==dense OK (max|Δw|={d12:.1e})")


# ---------- L1 换手项: τ 增大换手单调降, τ→∞ 不动 ----------

def test_turnover_monotone_tau():
    alpha, Sigma, w0, w_prev, cap = make_problem(60, seed=8)
    tos = []
    for tau in (0.0, 1e-3, 5e-3, 2e-2):
        r = solve(alpha, Sigma, w0, w_prev, lam=5.0, tau=tau, cap=cap)
        tos.append(float(np.abs(np.asarray(r["w"]) - w_prev).sum()))
    for hi_, lo_ in zip(tos, tos[1:]):
        assert lo_ <= hi_ + 1e-8, f"τ增大换手反增: {tos}"
    r = solve(alpha, Sigma, w0, w_prev, lam=5.0, tau=1e6, cap=cap)
    assert np.max(np.abs(np.asarray(r["w"]) - w_prev)) < 1e-8, \
        "τ→∞ 时应精确停在 w_prev(L1 不交易区)"
    print(f"turnover(τ单调 {' → '.join(f'{t:.3f}' for t in tos)}; τ=1e6不动) OK")


if __name__ == "__main__":
    test_identity_lambda_huge()
    test_matches_slsqp()
    test_constraints_and_kkt()
    test_style_band_monotone()
    test_prox_exact()
    test_lowrank_equals_dense()
    test_turnover_monotone_tau()
    print("ALL GREEN (m9_opt)")
