"""纪律层二期测试: 预注册工单(冻结/篡改/预算) + 噪声对照(置换检验)。

跑法: python3 tests/test_discipline2.py  (纯合成, 不碰网络)
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spencer.discipline.preregister import create_ticket, evaluate, load_ticket
from spencer.discipline.noise import noise_control, permute_within_rows
from spencer.discipline.ledger import log_run


def test_preregister():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tk = create_ticket(td / "tickets", "probe", "假设: 测试因子有效",
                           criteria={"ic_mean": 0.02, "yearly_all_positive": True},
                           trial_budget=2, stoploss="IC连续两窗<0.01即停")
        # 判据全过 + 预算内 → pass
        r = evaluate(tk, {"ic_mean": 0.03, "yearly_all_positive": True})
        assert r["verdict"] == "pass", r
        # 任一判据不过 → fail
        r = evaluate(tk, {"ic_mean": 0.01, "yearly_all_positive": True})
        assert r["verdict"] == "fail"
        # 预算超支 → 最好也只能 warn
        ledger = td / "ledger.csv"
        for i in range(3):
            log_run(ledger, {"factor": "probe", "ic_mean": 0.03})
        r = evaluate(tk, {"ic_mean": 0.03, "yearly_all_positive": True},
                     ledger_path=ledger, factor="probe")
        assert r["verdict"] == "warn" and r["over_budget"], r
        # 篡改工单(改判据) → tampered
        raw = tk.read_text(encoding="utf-8").replace("0.02", "0.001")
        tk.write_text(raw, encoding="utf-8")
        r = evaluate(tk, {"ic_mean": 0.03, "yearly_all_positive": True})
        assert r["verdict"] == "tampered"
        # 同名工单不允许覆盖(判据冻结)
        try:
            create_ticket(td / "tickets", "probe", "x", {"ic_mean": 0}, 1, "s")
            assert False, "应拒绝覆盖"
        except FileExistsError:
            pass
    print("preregister OK")


def test_noise_control():
    rng = np.random.default_rng(3)
    t, n = 300, 80
    dates = pd.bdate_range("2024-01-01", periods=t)
    codes = [f"s{i:03d}" for i in range(n)]
    fwd = pd.DataFrame(rng.normal(0, 0.02, (t, n)), index=dates, columns=codes)
    # 保形状: 挖掉 10% 缺失
    hole = rng.random((t, n)) < 0.1
    real = (fwd * 0.6 + rng.normal(0, 0.02, (t, n))).mask(hole)   # 强信号
    noise = pd.DataFrame(rng.normal(size=(t, n)), index=dates,
                         columns=codes).mask(hole)

    # 置换保形状: NaN 位不动, 行边际分布不变
    p1 = permute_within_rows(real, np.random.default_rng(0))
    assert p1.isna().equals(real.isna()), "置换动了缺失结构"
    assert np.allclose(np.sort(p1.iloc[5].dropna()), np.sort(real.iloc[5].dropna()))

    r_real = noise_control(real, fwd, n_arms=19, seed=1)
    r_noise = noise_control(noise, fwd, n_arms=19, seed=1)
    assert r_real["p_value"] == 1 / 20, f"强信号应赢所有分身: {r_real}"
    assert r_noise["p_value"] > 0.2, f"噪声因子不应显著: {r_noise}"
    print(f"noise_control OK (real p={r_real['p_value']}, noise p={r_noise['p_value']})")


if __name__ == "__main__":
    test_preregister()
    test_noise_control()
    print("ALL GREEN (discipline2)")
