"""组合优化器 (M9 后半): FISTA 求解带换手惩罚与风格暴露带的均值-方差问题。

问题(最大化):
    α'w − λ·(w−w0)'Σ(w−w0) − τ·‖w−w_prev‖₁
    s.t.  0 ≤ w ≤ cap,  Σw = 1          —— 硬约束: 含个股上限的截断单纯形
          |B'(w−w0)| ≤ bound            —— 软约束: 二次罚项实现的风格暴露带

角色: composite.py 把多因子合成单一信号, portfolio.py 是规则式持仓
(top-N 缓冲区), 这里是优化式持仓 —— 把信号(α)、相对基准 w0 的主动风险
(Σ)、换手成本(τ·L1)与风格敞口(B/bound)放进同一个目标里权衡。

方法出处(全部公开文献, 不含任何机构私有实现):
- FISTA 加速近端梯度: Beck & Teboulle (2009, SIAM J. Imaging Sciences)。
  为什么选它: 目标 = 光滑二次 + 非光滑(L1 + 约束示性函数), 天然是
  "梯度 + 近端算子"的复合结构; FISTA 有 O(1/k²) 收敛率且每步只要
  一次 Σ·v 乘法, 规模到全 A 股截面也撑得住, 不需要引入 QP 求解器依赖。
- 自适应重启: O'Donoghue & Candès (2015, Foundations of Computational
  Mathematics) 的梯度重启 —— 动量方向与本步位移背离时把动量清零,
  强凸问题上实测接近线性收敛。
- 截断单纯形投影: Duchi, Shalev-Shwartz, Singer & Chandra (2008, ICML)
  单纯形投影思想的推广。加上限 cap 与 L1 中心后, 排序式闭式水位的
  断点结构变复杂(每个坐标最多 4 个断点), 改用二分法找水位 ν:
  每次迭代 O(n) 向量化, 80 次二分把水位压到机器精度量级。
- L1 换手项二选一的取舍: 用**精确近端算子**, 不用平滑近似(Huber)。
  为什么: L1 在 w=w_prev 处的角点正是"不交易区"的来源 —— 边际alpha
  抵不过 τ 的票精确不动; 平滑近似抹掉角点后每期都会产生一堆无意义的
  微小调仓, 恰好毁掉引入 τ 的目的。近端算子的代价只是 prox 里多一次
  软阈值, 而二分水位反正要做, 所以精确解几乎不增加成本。
- 风格暴露带用二次罚项(Nocedal & Wright, Numerical Optimization,
  ch.17 罚函数法)而非硬约束。为什么: 截断单纯形 ∩ 暴露带的联合投影
  没有便宜解(交集投影 ≠ 依次投影), 罚项把带约束挪进光滑部分,
  prox 保持便宜。**已知近似, 明示不藏**: 罚项是软约束, 最优解可能
  轻微越界, 越界量 ≈ 边际alpha / (2·penalty_weight), 调大
  penalty_weight 可压到任意小(代价是 Lipschitz 常数变大、步长变小)。

工程口径:
- 步长 η = 1/L, L 取 2λ·λmax(Σ) + 2μ·λmax(B'B) 的**上界**而非估计值:
  低估 L 会直接发散, 保守一点只是慢一点。
- KKT 残差 = 近端梯度映射 G(w) = (w − prox(w − η∇f(w)))/η 的无穷范数
  (Beck, First-Order Methods in Optimization, 2017)。凸问题下 G(w)=0
  当且仅当 w 全局最优, 所以它是可交付的最优性证据, 不是"迭代不动了"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------- Σ 的两种表示统一成 matvec ----------

def _cov_operator(Sigma, n: int):
    """把协方差统一成 (matvec 闭包, λmax 上界)。

    支持两种表示:
    - 稠密 (n,n) ndarray;
    - 因子结构 Σ = B_r·F·B_r' + diag(spec²): 传 (B_r, F, spec) 三元组或
      {"B": …, "F": …, "spec": …} 字典。这是 Barra 型风险模型的标准存法
      (B_r=风险因子暴露 n×k, F=因子协方差 k×k, spec=特异波动率 n),
      matvec 按 B_r(F(B_r'v)) + spec²∘v 结合律算, O(nk) 不落地 n×n。

    λmax 口径: 稠密用 eigvalsh 精确取(对称化后); 因子结构用 Weyl 不等式
    λmax(BFB' + D) ≤ λmax(BFB') + max(spec²), 其中 λmax(BFB') 通过
    F = V·diag(e)·V' 分解转成 k×k 矩阵 (S'S, S = B_r·V·√e) 精确算。
    上界可能略保守(步长略小), 但绝不冒发散的险。
    """
    if isinstance(Sigma, np.ndarray) and Sigma.ndim == 2:
        S = 0.5 * (np.asarray(Sigma, float) + np.asarray(Sigma, float).T)
        if S.shape != (n, n):
            raise ValueError(f"稠密 Sigma 形状应为 ({n},{n}), got {S.shape}")
        lam_max = float(np.linalg.eigvalsh(S)[-1])
        return (lambda v: S @ v), max(lam_max, 0.0)

    if isinstance(Sigma, dict):
        Br, F, spec = Sigma["B"], Sigma["F"], Sigma["spec"]
    else:
        Br, F, spec = Sigma
    Br = np.asarray(Br, float)
    F = 0.5 * (np.asarray(F, float) + np.asarray(F, float).T)
    spec2 = np.asarray(spec, float) ** 2
    if Br.shape[0] != n or spec2.shape != (n,):
        raise ValueError("低秩 Sigma 维度不齐: B_r 应 n×k, spec 应长 n")

    e, V = np.linalg.eigh(F)
    S_half = Br @ (V * np.sqrt(np.clip(e, 0.0, None)))
    lam_bfb = float(np.linalg.eigvalsh(S_half.T @ S_half)[-1]) if S_half.size else 0.0
    lam_max = max(lam_bfb, 0.0) + float(spec2.max())

    def matvec(v):
        return Br @ (F @ (Br.T @ v)) + spec2 * v

    return matvec, lam_max


# ---------- 精确近端算子: L1 + 截断单纯形 ----------

def prox_capped_simplex_l1(v: np.ndarray, center: np.ndarray, thresh: float,
                           cap: np.ndarray, iters: int = 80) -> np.ndarray:
    """精确求解  argmin_w  ½‖w−v‖² + thresh·‖w−center‖₁
                s.t.  0 ≤ w ≤ cap,  Σw = 1。

    结构(为什么可以精确解): 对 Σw=1 做拉格朗日对偶, 水位 ν 给定时逐坐标
    有闭式解 —— 先向 center 软阈值收缩(L1 的近端), 再截断到 [0, cap]
    (一维凸函数在区间上的最小点 = 无约束最小点的裁剪)。Σw(ν) 关于 ν
    连续且单调不增, 二分找 Σw(ν)=1 的水位即可。

    thresh=0 时退化为经典的截断单纯形投影(Duchi et al. 2008 的推广);
    cap=∞ 且 thresh=0 时就是原版单纯形投影。

    为什么二分而不是排序闭式: cap 与 center 让每个坐标有至多 4 个断点,
    排序法的代码复杂度换不来多少速度; 二分 80 次区间缩 2⁻⁸⁰,
    水位误差在机器精度量级, 每次 O(n) 向量化, 稳且便宜。
    """
    lo = float(np.min(v - thresh - cap)) - 1.0   # 此处 Σw = Σcap ≥ 1
    hi = float(np.max(v + thresh)) + 1.0         # 此处 Σw = 0 ≤ 1

    def w_of(nu):
        z = v - nu
        w = np.where(z > center + thresh, z - thresh,
                     np.where(z < center - thresh, z + thresh, center))
        return np.clip(w, 0.0, cap)

    for _ in range(iters):
        nu = 0.5 * (lo + hi)
        if w_of(nu).sum() > 1.0:
            lo = nu
        else:
            hi = nu
    return w_of(0.5 * (lo + hi))


# ---------- 主入口 ----------

def solve(alpha, Sigma, w0, w_prev, lam: float, tau: float, cap,
          B=None, bound=None, max_iter: int = 2000, *,
          tol: float = 1e-7, penalty_weight: float = 100.0) -> dict:
    """max α'w − λ(w−w0)'Σ(w−w0) − τ‖w−w_prev‖₁, s.t. 截断单纯形 + 暴露带。

    参数
    ----
    alpha    : 预期收益/信号 (n,)。pd.Series 则返回的 w 带同一 index。
    Sigma    : 协方差 —— 稠密 (n,n), 或 (B_r, F, spec) / dict 因子结构。
    w0       : 基准权重 (n,)。风险与暴露带都相对它度量(主动风险口径)。
    w_prev   : 上期持仓 (n,), 换手惩罚的中心。
    lam, tau : 风险厌恶与换手惩罚系数 (≥0)。
    cap      : 个股上限, 标量或 (n,)。要求 Σcap ≥ 1, 否则可行域为空。
    B, bound : 风格暴露 (n,k) 与带宽(标量或 (k,)); 罚项为
               penalty_weight·Σ_k max(|b_k'(w−w0)|−bound_k, 0)²。
               B 列尺度直接进 Lipschitz 常数(∝λmax(B'B)), 建议先标准化。
    tol      : KKT 残差(近端梯度映射 ∞ 范数)的收敛线。
    penalty_weight : 暴露带罚项权重 μ。越大带越硬、步长越小。

    返回 dict: w(权重), kkt_residual, n_iter, objective(真实目标值,
    不含罚项), converged; 给了 B 时另附 style_exposure=B'(w−w0) 与
    style_penalty(罚项值, 用于判断带是否被顶到)。

    对齐口径: 所有向量按位置对齐, 不做 reindex 魔法 —— 调用方负责把
    alpha/w0/w_prev/cap 摆到同一套代码顺序上(静默重排比报错更危险)。
    """
    idx = alpha.index if isinstance(alpha, pd.Series) else None
    a = np.asarray(alpha, float).ravel()
    n = a.size
    w0_ = np.asarray(w0, float).ravel()
    wp_ = np.asarray(w_prev, float).ravel()
    cap_ = np.broadcast_to(np.asarray(cap, float), (n,)).astype(float)

    if not (w0_.size == wp_.size == n):
        raise ValueError("alpha/w0/w_prev 长度不一致")
    for name, arr in (("alpha", a), ("w0", w0_), ("w_prev", wp_), ("cap", cap_)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} 含 NaN/Inf —— 优化器不猜缺失值, 上游先清")
    if lam < 0 or tau < 0:
        raise ValueError("lam/tau 必须 ≥ 0")
    if np.any(cap_ <= 0):
        raise ValueError("cap 必须全为正")
    if cap_.sum() < 1.0 - 1e-12:
        raise ValueError(f"Σcap = {cap_.sum():.4f} < 1, 可行域为空")

    matvec, lam_sigma = _cov_operator(Sigma, n)

    if B is not None:
        Bm = np.asarray(B, float)
        if Bm.ndim == 1:
            Bm = Bm.reshape(-1, 1)
        if Bm.shape[0] != n:
            raise ValueError(f"B 应为 (n,k), got {Bm.shape}")
        k = Bm.shape[1]
        if bound is None:
            raise ValueError("给了 B 必须给 bound")
        bnd = np.broadcast_to(np.asarray(bound, float), (k,)).astype(float)
        if np.any(bnd < 0):
            raise ValueError("bound 必须 ≥ 0")
        lam_bb = float(np.linalg.eigvalsh(Bm.T @ Bm)[-1])
    else:
        Bm, bnd, lam_bb = None, None, 0.0

    # Lipschitz 上界: ∇²f ⪯ 2λΣ + 2μBB' (罚项 max(|u|−b,0)² 是 C¹,
    # 导数分段线性、斜率 ≤ 2, 故其 Hessian 上界为 2μBB')。
    L = 2.0 * lam * lam_sigma + 2.0 * penalty_weight * lam_bb
    L = max(L, 1e-8)          # λ=0 且无 B 的纯线性退化情形兜底
    eta = 1.0 / L

    def grad(w):
        d = w - w0_
        g = 2.0 * lam * matvec(d) - a
        if Bm is not None:
            u = Bm.T @ d
            excess = np.sign(u) * np.maximum(np.abs(u) - bnd, 0.0)
            g += 2.0 * penalty_weight * (Bm @ excess)
        return g

    def prox(v):
        return prox_capped_simplex_l1(v, wp_, eta * tau, cap_)

    def kkt_at(w):
        """近端梯度映射的 ∞ 范数 —— 0 ⟺ 全局最优(凸)。"""
        return float(np.max(np.abs(w - prox(w - eta * grad(w)))) / eta)

    # 初始化: w0 投影进可行域(纯投影, 不带 L1 收缩)。基准通常离最优不远,
    # 且 λ 很大时最优点 → P(w0), 这个初始化让恒等式情形一步到位。
    x = prox_capped_simplex_l1(w0_, wp_, 0.0, cap_)
    x_prev = x.copy()
    y = x.copy()
    tk = 1.0
    kkt = np.inf
    n_iter = 0

    for it in range(1, max_iter + 1):
        n_iter = it
        y_in = y
        x = prox(y_in - eta * grad(y_in))

        # 梯度式自适应重启 (O'Donoghue & Candès 2015)
        if float(np.dot(y_in - x, x - x_prev)) > 0.0:
            tk = 1.0
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
        y = x + ((tk - 1.0) / t_next) * (x - x_prev)
        tk = t_next

        # 便宜筛(y 处的梯度映射已在手) → 通过才花一次 grad+prox 精确验证
        if float(np.max(np.abs(y_in - x))) / eta < tol:
            kkt = kkt_at(x)
            if kkt < tol:
                break
        x_prev = x

    # 收敛则 kkt 即最终 x 的残差; 否则(含便宜筛在中途算过旧残差的情形)
    # 必须在最终 x 上重算 —— 交付的残差永远对应交付的权重。
    if not (np.isfinite(kkt) and kkt < tol):
        kkt = kkt_at(x)

    d = x - w0_
    risk = float(lam * np.dot(d, matvec(d)))
    l1 = float(tau * np.abs(x - wp_).sum())
    obj = float(np.dot(a, x)) - risk - l1

    out = {
        "w": pd.Series(x, index=idx) if idx is not None else x,
        "kkt_residual": kkt,
        "n_iter": n_iter,
        "objective": obj,
        "converged": bool(kkt < tol),
    }
    if Bm is not None:
        u = Bm.T @ d
        out["style_exposure"] = u
        out["style_penalty"] = float(
            penalty_weight * np.sum(np.maximum(np.abs(u) - bnd, 0.0) ** 2))
    return out
