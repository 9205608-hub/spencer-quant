"""交叉验证: spencer.discipline.stats vs alpha-court court/{dsr,pbo,sharpe}.

跑法:
    python3 -m pytest tests/test_cross_validation_court.py -q
    ALPHA_COURT_PATH=/nonexistent python3 -m pytest tests/test_cross_validation_court.py -q
        # 仓不可达时应全部 skip, 零 fail

前提: 环境变量 ALPHA_COURT_PATH 指向 alpha-court 仓根目录;
缺省 /Users/spensir/Desktop/alpha-court。路径不存在或 court 不可 import
时本模块整体 skip —— 公开仓用户没有姊妹仓, 不许硬红。

映射与边界的文字结论见 docs/交叉验证-court.md。
两侧实现全部冻结: 本文件只读对比, 不改 spencer/ 也不改 court/。

import 陷阱: court/__init__.py 把函数 dsr 重导出, `import court.dsr as x`
会把 x 绑到函数而非模块。一律 importlib.import_module("court.dsr")。
"""
from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spencer.discipline.stats import (
    deflated_sharpe,
    expected_max_sharpe,
    pbo_cscv,
    probabilistic_sharpe,
)

_DEFAULT_COURT = "/Users/spensir/Desktop/alpha-court"
_COURT_ROOT = Path(os.environ.get("ALPHA_COURT_PATH", _DEFAULT_COURT))


def _load_court():
    """加载 court.dsr / court.pbo / court.sharpe; 失败返回 (None, reason)。"""
    if not _COURT_ROOT.is_dir():
        return None, f"ALPHA_COURT_PATH 不存在: {_COURT_ROOT}"
    # 插到最前, 避免同名第三方包抢 import
    sys.path.insert(0, str(_COURT_ROOT))
    try:
        dsr_mod = importlib.import_module("court.dsr")
        pbo_mod = importlib.import_module("court.pbo")
        sharpe_mod = importlib.import_module("court.sharpe")
    except Exception as exc:  # noqa: BLE001 — 优雅降级, 任何导入失败都 skip
        return None, f"court 不可 import: {exc}"
    # 必须是模块: 若误用 `import court.dsr` 会绑到函数, 没有 expected_max_sr
    if not hasattr(dsr_mod, "expected_max_sr") or not hasattr(dsr_mod, "dsr"):
        return None, f"court.dsr 不是模块 (got {type(dsr_mod)!r})"
    return (dsr_mod, pbo_mod, sharpe_mod), None


_COURT, _COURT_SKIP = _load_court()
# 用 skipif 而不是 allow_module_level skip: 后者在收集期直接放弃整模块,
# pytest 退出码 5 (no tests collected)。skipif 让 7 条测试都被收集后 skip,
# 退出码 0 —— 公开仓用户看到的是「全部 skip, 零 fail」。
court_dsr = court_pbo = court_sharpe = None
if _COURT is not None:
    court_dsr, court_pbo, court_sharpe = _COURT

pytestmark = pytest.mark.skipif(_COURT is None, reason=_COURT_SKIP or "")

# ---- 容差 (H1/H2 解析式; H3 全枚举计数) ----
_SR0_ATOL = 1e-9
_DSR_ATOL = 1e-9
_PBO_ATOL = 1e-12

# ---- PBO 三张固定种子矩阵 ----
_PBO_T, _PBO_N, _PBO_S = 256, 8, 8
_PBO_SIGMA = 0.01
_PBO_NCOMB = math.comb(_PBO_S, _PBO_S // 2)  # C(8,4)=70
_PBO_DELTA = 1.0 / _PBO_NCOMB                # 分辨率 1/70
# 带宽论证见文档 §PBO 方向带宽: 独立 Bernoulli(0.5) 的 SE≈0.060≈4.2δ
_NOISE_BAND = 4 * _PBO_DELTA                 # |φ-0.5| ≤ 4/70
_SKILL_MAX = 8 * _PBO_DELTA                  # 技能「显著低」φ ≤ 8/70
_TRAP_MIN = 0.5 + 14 * _PBO_DELTA            # 陷阱「显著高」φ ≥ 49/70 = 0.7

_NOISE_SEED = 17
_SKILL_SEED = 9
_TRAP_SEED = 18


# =====================================================================
# H1  SR0
# =====================================================================

def test_h1_sr0_grid():
    """H1: spencer.expected_max_sharpe(N, T, V) == court.expected_max_sr(0, √V, N)。

    文献同一式 (Bailey & López de Prado 2014 Eq.1): √V · [(1-γ)Φ⁻¹(1-1/N)
    + γ Φ⁻¹(1-1/(N e))]。spencer 吃方差 V, court 吃标准差 σ=√V。
    """
    max_d = 0.0
    n_pts = 0
    for n_trials in (2, 10, 100, 1000):
        for T in (50, 252, 1000):
            for v_scale in (1.0, 0.5):
                var_trials = v_scale / T
                sp = expected_max_sharpe(n_trials, T, var_trials=var_trials)
                ct = court_dsr.expected_max_sr(
                    0.0, math.sqrt(var_trials), n_trials
                )
                d = abs(sp - ct)
                max_d = max(max_d, d)
                n_pts += 1
                assert d <= _SR0_ATOL, (
                    f"H1 fail N={n_trials} T={T} V={var_trials}: "
                    f"spencer={sp!r} court={ct!r} |Δ|={d}"
                )
                # 公开导出之外, 也经 DsrResult.sr_star 钉死同一字段
                star = court_dsr.dsr(
                    0.05, T, 0.0, 3.0, math.sqrt(var_trials), n_trials
                ).sr_star
                assert abs(sp - star) <= _SR0_ATOL
    print(f"h1_sr0_grid OK ({n_pts} pts, max|Δ|={max_d:.3e})")


# =====================================================================
# H2  DSR / 峰度 / Bessel
# =====================================================================

def test_h2_dsr_grid():
    """H2: spencer.deflated_sharpe(...)['dsr'] == court.dsr.dsr(...).dsr。

    峰度两边都是原始峰度(正态=3), 不换算; court 的 σ 仍是 √V。
    """
    max_d = 0.0
    n_pts = 0
    for sr in (0.0, 0.05, 0.1, 0.2):
        for n_trials in (2, 10, 100):
            for T in (50, 252):
                for skew, kurt in ((0.0, 3.0), (-0.5, 5.0)):
                    var_trials = 1.0 / T
                    sp = deflated_sharpe(
                        sr, n_trials, T, skew, kurt, var_trials=var_trials
                    )
                    ct = court_dsr.dsr(
                        sr, T, skew, kurt, math.sqrt(var_trials), n_trials
                    )
                    d = abs(sp["dsr"] - ct.dsr)
                    max_d = max(max_d, d)
                    n_pts += 1
                    assert d <= _DSR_ATOL, (
                        f"H2 fail sr={sr} N={n_trials} T={T} "
                        f"skew={skew} kurt={kurt}: "
                        f"spencer={sp['dsr']!r} court={ct.dsr!r} |Δ|={d}"
                    )
    print(f"h2_dsr_grid OK ({n_pts} pts, max|Δ|={max_d:.3e})")


def test_conventions_kurtosis_raw_and_bessel_tm1():
    """钉死两个容易翻车的约定: 峰度=原始(正态=3); PSR 分子用 √(T-1) 不是 √T。"""
    # 1) court.series_moments 的 kurt_hat 是原始峰度
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20000)
    moments = court_sharpe.series_moments(x)
    assert 2.8 < moments.kurt_hat < 3.2, (
        f"court kurt_hat 应贴近原始峰度 3, got {moments.kurt_hat}"
    )

    # 2) 同一组 (sr, T, skew, kurt) 上, 两边 PSR 与手写 √(T-1) 式逐位一致,
    #    与手写 √T 式差 ~1e-3, 排除「court 用了 √T」的可能
    sr, T, skew, kurt = 0.1, 50, -0.5, 5.0
    sp = probabilistic_sharpe(sr, 0.0, T, skew, kurt)
    ct = court_sharpe.psr(sr, 0.0, T, skew, kurt)
    vf = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    phi = NormalDist()
    psr_tm1 = phi.cdf(sr * math.sqrt(T - 1.0) / math.sqrt(vf))
    psr_t = phi.cdf(sr * math.sqrt(float(T)) / math.sqrt(vf))
    assert abs(sp - ct) <= 1e-15
    assert abs(sp - psr_tm1) <= 1e-15
    assert abs(sp - psr_t) > 1e-3, (
        f"若两边实现的是 √T 式, 本断言会翻; 实测 |Δ√T|={abs(sp - psr_t):.3e}"
    )
    print(
        f"conventions OK (kurt_hat={moments.kurt_hat:.4f}, "
        f"PSR={sp:.6f} matches √(T-1), |Δ√T|={abs(sp - psr_t):.3e})"
    )


# =====================================================================
# H3  PBO
# =====================================================================

def _noise_matrix(seed: int = _NOISE_SEED) -> np.ndarray:
    """纯噪声: i.i.d. N(0, σ), T=256 × N=8。"""
    return np.random.default_rng(seed).normal(
        0.0, _PBO_SIGMA, size=(_PBO_T, _PBO_N)
    )


def _skill_matrix(seed: int = _SKILL_SEED) -> np.ndarray:
    """单列植入真实技能 μ = 0.1 σ (每期 Sharpe = 0.1)。"""
    M = np.random.default_rng(seed).normal(
        0.0, _PBO_SIGMA, size=(_PBO_T, _PBO_N)
    )
    M[:, 0] += 0.1 * _PBO_SIGMA
    return M


def _trap_matrix(seed: int = _TRAP_SEED) -> np.ndarray:
    """强选择陷阱: 各列同分布, 将第 0 列的收益按大小重排 ——
    最大的 T/2 个放进偶数块, 最小的 T/2 个放进奇数块。

    边际分布(取值多重集)不变, 但 CSCV 在「偶数块 ∈ IS」时会选中第 0 列,
    对应 OOS(奇数块) 它系统性垫底 → PBO 应显著高于 0.5。
    """
    rng = np.random.default_rng(seed)
    M = rng.normal(0.0, _PBO_SIGMA, size=(_PBO_T, _PBO_N))
    block = _PBO_T // _PBO_S  # 32
    col = M[:, 0].copy()
    order = np.argsort(col)
    even_idx = np.concatenate(
        [np.arange(s * block, (s + 1) * block) for s in range(0, _PBO_S, 2)]
    )
    odd_idx = np.concatenate(
        [np.arange(s * block, (s + 1) * block) for s in range(1, _PBO_S, 2)]
    )
    clustered = np.empty_like(col)
    clustered[even_idx] = col[order[_PBO_T // 2:]]  # 大的一半 → 偶数块
    clustered[odd_idx] = col[order[: _PBO_T // 2]]  # 小的一半 → 奇数块
    M[:, 0] = clustered
    return M


def _both_pbo(M: np.ndarray):
    """同一矩阵喂两边, 返回 (spencer_dict, court_PboResult)。"""
    sp = pbo_cscv(M, n_splits=_PBO_S)
    ct = court_pbo.pbo_cscv(M, _PBO_S, court_sharpe.sharpe_ratio)
    return sp, ct


def test_h3_pbo_three_matrices():
    """H3: 三张固定种子矩阵上两边 φ/PBO 逐位(≤1e-12)一致, 且 C(8,4)=70。"""
    cases = (
        ("noise", _noise_matrix()),
        ("skill", _skill_matrix()),
        ("trap", _trap_matrix()),
    )
    for name, M in cases:
        assert M.shape == (_PBO_T, _PBO_N)
        assert not np.isnan(M).any()
        sp, ct = _both_pbo(M)
        d = abs(sp["pbo"] - ct.phi)
        assert d <= _PBO_ATOL, (
            f"H3 {name}: spencer pbo={sp['pbo']!r} court phi={ct.phi!r} |Δ|={d}"
        )
        assert sp["n_combinations"] == ct.n_combinations == _PBO_NCOMB
        assert sp["n_periods_used"] == _PBO_T
        print(f"h3_{name} match OK (φ={sp['pbo']:.6f}, |Δ|={d:.3e})")


def test_h3_pbo_directional():
    """PBO 方向符合各矩阵预期。带宽按 C(8,4)=70 的分辨率 1/70 写死。"""
    noise_phi = pbo_cscv(_noise_matrix(), n_splits=_PBO_S)["pbo"]
    skill_phi = pbo_cscv(_skill_matrix(), n_splits=_PBO_S)["pbo"]
    trap_phi = pbo_cscv(_trap_matrix(), n_splits=_PBO_S)["pbo"]

    assert abs(noise_phi - 0.5) <= _NOISE_BAND, (
        f"噪声 PBO 应在 0.5±{ _NOISE_BAND:.4f} 内, got {noise_phi}"
    )
    assert skill_phi <= _SKILL_MAX, (
        f"技能列 PBO 应 ≤ {_SKILL_MAX:.4f}, got {skill_phi}"
    )
    assert trap_phi >= _TRAP_MIN, (
        f"陷阱 PBO 应 ≥ {_TRAP_MIN:.4f}, got {trap_phi}"
    )
    assert skill_phi < noise_phi < trap_phi
    print(
        f"h3_directional OK (noise={noise_phi:.4f}, "
        f"skill={skill_phi:.4f}, trap={trap_phi:.4f})"
    )


# =====================================================================
# 边界行为 (钉死差异, 不是要两边一致)
# =====================================================================

def test_pbo_boundary_tail_and_nan():
    """边界: spencer 丢不能整除的尾部; court 要求 T % S == 0。
    NaN: spencer AssertionError, court ValueError。两边都不静默吃进去。
    """
    rng = np.random.default_rng(1)
    M_tail = rng.normal(0.0, _PBO_SIGMA, size=(260, _PBO_N))  # 260 % 8 ≠ 0
    sp = pbo_cscv(M_tail, n_splits=_PBO_S)
    assert sp["n_periods_used"] == 256, "spencer 应丢弃末 4 行"

    with pytest.raises(ValueError, match="divisible"):
        court_pbo.pbo_cscv(M_tail, _PBO_S, court_sharpe.sharpe_ratio)

    M_nan = rng.normal(0.0, _PBO_SIGMA, size=(_PBO_T, _PBO_N))
    M_nan[0, 0] = np.nan
    with pytest.raises(AssertionError, match="NaN"):
        pbo_cscv(M_nan, n_splits=_PBO_S)
    with pytest.raises(ValueError, match="finite"):
        court_pbo.pbo_cscv(M_nan, _PBO_S, court_sharpe.sharpe_ratio)
    print("pbo_boundary OK (tail drop vs require-divisible; both reject NaN)")


def test_dsr_boundary_var_factor_nonpositive():
    """越出公式适用域(var_factor≤0): spencer 返回 nan, court 抛 ValueError。"""
    sp = probabilistic_sharpe(2.0, 0.0, 500, skew=5.0, kurt=3.0)
    assert math.isnan(sp)
    with pytest.raises(ValueError, match="var_factor"):
        court_sharpe.psr(2.0, 0.0, 500, 5.0, 3.0)
    print("dsr_boundary var_factor OK (nan vs ValueError)")
