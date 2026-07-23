"""v1.0 收尾三件套测试: 协方差(含PIT钉子) / 入库契约 / 因子元信息。

跑法: python3 tests/test_v1_finish.py  (纯合成数据, 不碰网络)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spencer.risk.covariance import sigma_at, style_factor_returns


class FakeStore:
    def __init__(self, frames):
        self.frames = frames

    def load(self, field):
        return self.frames[field]

    def end_date(self):
        return self.frames["close"].index[-1]


def synth_world(t=400, n=120, seed=11):
    """两风格世界: fwd1 = B @ f + eps, F 已知(f1 波动是 f2 的 3 倍)。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=t)
    codes = [f"s{i:03d}" for i in range(n)]
    b1 = rng.normal(0, 1, n)
    b2 = rng.normal(0, 1, n)
    f1 = rng.normal(0, 0.003, t)
    f2 = rng.normal(0, 0.001, t)
    eps = rng.normal(0, 0.005, (t, n))
    fwd1 = pd.DataFrame(np.outer(f1, b1) + np.outer(f2, b2) + eps,
                        index=dates, columns=codes)
    styles = {
        "sty1": pd.DataFrame(np.tile(b1, (t, 1)), index=dates, columns=codes),
        "sty2": pd.DataFrame(np.tile(b2, (t, 1)), index=dates, columns=codes),
    }
    return dates, codes, fwd1, styles


def test_covariance_recovers_structure():
    dates, codes, fwd1, styles = synth_world()
    sr, resid = style_factor_returns(fwd1, styles, industry=None, min_names=30)
    assert len(sr) > 350, "因子收益序列覆盖不足"
    # 真世界里 sty1 因子收益波动是 sty2 的 3 倍 → 估计的方差比应远大于 1
    ratio = sr["sty1"].var() / sr["sty2"].var()
    assert ratio > 4, f"F 结构未恢复: var比 {ratio:.1f} (真值≈9)"
    # 残差应大致失去与风格的相关(回归吃掉了)
    corr = resid.iloc[50].dropna().corr(styles["sty1"].iloc[50])
    assert abs(corr) < 0.1, f"残差仍含风格: {corr}"
    print("covariance_structure OK")


def test_sigma_at_is_pit():
    dates, codes, fwd1, styles = synth_world()
    sr, resid = style_factor_returns(fwd1, styles, industry=None, min_names=30)
    asof = dates[300]
    B1, F1, s1, c1 = sigma_at(asof, styles, sr, resid, min_obs=100)
    # 篡改 asof 之后的"未来"因子收益与残差 → sigma_at(asof) 必须逐位不变
    sr2, resid2 = sr.copy(), resid.copy()
    sr2.loc[dates[301]:] = 99.0
    resid2.loc[dates[301]:] = 99.0
    B2, F2, s2, c2 = sigma_at(asof, styles, sr2, resid2, min_obs=100)
    assert np.allclose(F1, F2) and np.allclose(s1, s2) and c1 == c2, \
        "sigma_at 吃到了未来信息"
    # F 数值护栏: PSD
    assert np.linalg.eigvalsh(F1).min() >= -1e-15
    print("sigma_at_pit OK")


def test_admission_contract():
    from spencer.factor.verify import admission_check, format_report
    rng = np.random.default_rng(5)
    t, n = 800, 100
    dates = pd.bdate_range("2022-01-03", periods=t)
    codes = [f"s{i:03d}" for i in range(n)]
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (t, n)), axis=0)),
                      index=dates, columns=codes)
    store = FakeStore({"adj_close": px, "close": px})
    from spencer.eval.panel import forward_returns
    fwd = forward_returns(store, 5)

    cheat = fwd * 0.5 + pd.DataFrame(rng.normal(0, fwd.std().mean(), fwd.shape),
                                     index=dates, columns=codes)
    cheat.iloc[-6:] = 0.0            # 补齐末端(fwd 末端天然 NaN)
    res = admission_check("cheat", cheat, store, fwd)
    assert res["checks"]["significance_nw"]["ok"], "强信号应过显著性关"
    assert res["checks"]["direction"]["ok"]

    noise = pd.DataFrame(rng.normal(size=(t, n)), index=dates, columns=codes)
    res2 = admission_check("noise", noise, store, fwd)
    assert res2["verdict"] == "fail", "噪声因子必须被拒"
    assert not res2["checks"]["significance_nw"]["ok"]

    res3 = admission_check("cheat2", cheat, store, fwd, pool={"cheat": cheat})
    assert not res3["checks"]["pool_redundancy"]["ok"], "自身查重必须亮灯"
    assert res3["verdict"] in ("warn", "fail")
    print(format_report(res2).splitlines()[0])
    print("admission_contract OK")


def test_factor_meta():
    from spencer.factor.base import factor, get_meta, _REGISTRY, _META
    @factor("_meta_probe", author="spencer", tags=["test"], valid_from="2020-01-01")
    def _probe(store):
        return store.load("close")
    m = get_meta("_meta_probe")
    assert m["author"] == "spencer" and m["tags"] == ["test"]
    m["author"] = "hacked"
    assert get_meta("_meta_probe")["author"] == "spencer", "get_meta 应返回拷贝"
    assert get_meta("不存在") == {}
    _REGISTRY.pop("_meta_probe"), _META.pop("_meta_probe")
    print("factor_meta OK")


if __name__ == "__main__":
    test_covariance_recovers_structure()
    test_sigma_at_is_pit()
    test_admission_contract()
    test_factor_meta()
    print("ALL GREEN (v1_finish)")
