"""M2 因子缓存指纹化断言(30问#14)。

跑法: python tests/test_m2_cache.py  (零依赖, 不碰网络, 合成数据)

覆盖的失效场景:
① 同函数二次调用 → 命中缓存, 不重算;
② 换名的同逻辑函数(源码不同) → 代码指纹各自独立, 互不串缓存;
③ 数据形状变化(末端全部不变)四连: 起点后移 / 列数变 / 宇宙换血(列数
   恰好不变, 只换成分) / 中间交易日删除(起止与列全不变, 只少一行);
④ 末端判据独立兜底: 截尾缓存 + sidecar 输出签名也被伪造成截尾后的样子
   (模拟指纹链路自身出 bug), 唯一防线 = 末端判据, 必须仍然重算;
⑤ 缓存文件头部被截(sidecar 匹配、末端仍对齐)→ 输出签名失配重算 ——
   与④对称: 只查末端护尾不护头, 头部残缺必须靠输出签名抓;
附加: 同名因子源码被替换(模拟改代码后重启 session) → 代码指纹失配重算。

判定"是否重算"用调用计数器(CALLS), 不靠打印 —— 计数是硬证据。
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spencer.factor import base
from spencer.factor.base import compute, factor

# 每个因子函数被真正执行的次数 —— 缓存命中 = 计数不涨
CALLS = {"mom": 0, "mom_alias": 0, "shape": 0, "swap": 0}


class FakeStore:
    """与 tests/test_core.py 同款最小 store 契约: load + end_date。"""

    def __init__(self, frames):
        self.frames = frames

    def load(self, field):
        return self.frames[field]

    def end_date(self):
        return self.frames["close"].index[-1]


def make_store(end="2024-12-31", periods=120, n=40, seed=7):
    """合成一个可指定 起点(由 end+periods 反推)/列数 的假 store。

    用 bdate_range(end=..., periods=...) 固定末端 —— 测试③④要的正是
    "末端相同、起点/内容不同"的对照组。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=end, periods=periods)
    codes = [f"s{i:03d}" for i in range(n)]
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (periods, n)), axis=0)),
                      index=dates, columns=codes)
    return FakeStore({"close": px})


@factor("m2c_mom")
def m2c_mom(store):
    CALLS["mom"] += 1
    px = store.load("close")
    return px.pct_change(5, fill_method=None)


@factor("m2c_mom_alias")
def m2c_mom_alias(store):
    CALLS["mom_alias"] += 1
    prices = store.load("close")  # 同逻辑、不同源码(变量名/注释不同)
    return prices.pct_change(5, fill_method=None)


@factor("m2c_shape")
def m2c_shape(store):
    CALLS["shape"] += 1
    px = store.load("close")
    return px.pct_change(fill_method=None)


def m2c_swap_v1(store):
    CALLS["swap"] += 1
    px = store.load("close")
    return px.pct_change(5, fill_method=None)


def m2c_swap_v2(store):
    CALLS["swap"] += 1
    px = store.load("close")
    return px.pct_change(10, fill_method=None)  # 逻辑真的变了


def test_hit_on_second_call():
    """① 同函数二次调用命中缓存: 计数不涨, 值逐格一致。"""
    store = make_store()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        df1 = compute("m2c_mom", store, d)
        assert CALLS["mom"] == 1
        df2 = compute("m2c_mom", store, d)
        assert CALLS["mom"] == 1, "二次调用应命中缓存, 不得重算"
        # check_freq=False: parquet 往返会丢 DatetimeIndex.freq 元数据, 值不受影响
        pd.testing.assert_frame_equal(df1, df2, check_freq=False)
        assert (d / "m2c_mom.fingerprint.json").exists(), "sidecar 指纹文件应落盘"
    print("hit_on_second_call OK")


def test_renamed_same_logic_independent():
    """② 换名同逻辑函数各自独立: 代码指纹不同, 缓存互不干扰。"""
    store = make_store()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        compute("m2c_mom", store, d)
        compute("m2c_mom_alias", store, d)
        n_mom, n_alias = CALLS["mom"], CALLS["mom_alias"]

        fp_a = json.loads((d / "m2c_mom.fingerprint.json").read_text(encoding="utf-8"))
        fp_b = json.loads((d / "m2c_mom_alias.fingerprint.json").read_text(encoding="utf-8"))
        assert fp_a["code"] != fp_b["code"], "源码不同 → 代码指纹必须不同"
        assert fp_a["data"] == fp_b["data"], "同一 store → 数据指纹应相同"

        # 各自二次调用都命中各自的缓存
        compute("m2c_mom", store, d)
        compute("m2c_mom_alias", store, d)
        assert CALLS["mom"] == n_mom and CALLS["mom_alias"] == n_alias, "各自应命中各自缓存"
    print("renamed_same_logic_independent OK")


def test_data_shape_change_invalidates():
    """③ 数据形状变化(末端全部不变)四连 → 数据形状指纹失配, 缓存作废。

    后两组是对抗性复核实锤的盲区: 只存列数时, 宇宙换血(指数调仓换成分、
    列数恰好不变)与中间日删除(起止/列全不变)都会让旧值冒充新值。
    """
    store_a = make_store(periods=120, n=40)
    store_b = make_store(periods=80, n=40)   # 起点后移, 末端相同
    store_c = make_store(periods=120, n=30)  # 起点/末端同 a, 列数不同
    px_a = store_a.load("close")
    # 宇宙换血: 起止/行数/列数全同 a, 只把一个成分换名(s039→s999)
    store_d = FakeStore({"close": px_a.rename(columns={"s039": "s999"})})
    # 中间交易日删除: 起止/列全同 a, 行数少 1
    store_e = FakeStore({"close": px_a.drop(px_a.index[60])})
    ends = {s.end_date() for s in (store_a, store_b, store_c, store_d, store_e)}
    assert len(ends) == 1, "对照组必须同末端"

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        compute("m2c_shape", store_a, d)
        n0 = CALLS["shape"]
        # 旧判据(只看末端)会在这里静默返回 store_a 的缓存 —— 现在必须重算
        compute("m2c_shape", store_b, d)
        assert CALLS["shape"] == n0 + 1, "数据起点变化必须触发重算"
        compute("m2c_shape", store_c, d)
        assert CALLS["shape"] == n0 + 2, "列数(宇宙)变化必须触发重算"
        df_d = compute("m2c_shape", store_d, d)
        assert CALLS["shape"] == n0 + 3, "宇宙换血(列数不变)必须触发重算"
        assert "s999" in df_d.columns and "s039" not in df_d.columns, \
            "重算结果必须是新宇宙的列, 不得返回旧宇宙"
        df_e = compute("m2c_shape", store_e, d)
        assert CALLS["shape"] == n0 + 4, "中间交易日删除(起止/列不变)必须触发重算"
        assert len(df_e) == len(px_a) - 1, "重算结果行数必须跟随新数据"
        # 回到 store_e 再算一次 → 指纹已是 e 的, 应命中
        compute("m2c_shape", store_e, d)
        assert CALLS["shape"] == n0 + 4, "同 store 重复调用应命中缓存"
    print("data_shape_change_invalidates OK")


def test_stale_end_still_invalidates():
    """④ 末端判据独立兜底: 截尾 + 连 sidecar 输出签名都伪造成截尾后的样子。

    两步走:
    a) 只截尾、sidecar 原样 → 输出签名失配抓住(常规路径);
    b) 截尾并把 sidecar 的 out 改写成截尾后数据的签名(模拟指纹链路自身
       出 bug 或 sidecar 被一并篡改) → 指纹全"匹配", 唯一防线是末端判据。
       末端判据不依赖指纹实现的正确性, 这一步验证它真的独立存在。
    """
    store = make_store()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        compute("m2c_mom", store, d)
        n0 = CALLS["mom"]
        cache = d / "m2c_mom.parquet"
        sidecar = d / "m2c_mom.fingerprint.json"

        # a) 截尾, sidecar 保持原样 → 输出签名失配重算
        full = pd.read_parquet(cache)
        full.iloc[:-3].to_parquet(cache)
        df = compute("m2c_mom", store, d)
        assert CALLS["mom"] == n0 + 1, "截尾缓存必须触发重算(sidecar 原样)"
        assert df.index[-1] == store.end_date(), "重算后因子末端必须回到数据末端"

        # b) 截尾 + 伪造 sidecar 输出签名 → 只剩末端判据能抓
        truncated = pd.read_parquet(cache).iloc[:-3]
        truncated.to_parquet(cache)
        fp = json.loads(sidecar.read_text(encoding="utf-8"))
        fp["out"] = base._panel_signature(truncated)  # 签名与残缺文件"自洽"
        sidecar.write_text(json.dumps(fp, ensure_ascii=False), encoding="utf-8")
        df = compute("m2c_mom", store, d)
        assert CALLS["mom"] == n0 + 2, "指纹全匹配但末端不齐, 末端判据必须独立兜底"
        assert df.index[-1] == store.end_date(), "重算后因子末端必须回到数据末端"
    print("stale_end_still_invalidates OK")


def test_head_truncation_invalidates():
    """⑤ 缓存文件头部被截(sidecar 匹配、末端仍对齐) → 输出签名失配重算。

    与④对称: 旧实现只查末端, 护尾不护头 —— 砍头不砍尾的残缺缓存末端
    仍与数据末端对齐, 会被当新鲜返回(对抗性复核实锤)。输出签名含
    起始日期与行数, 必须看出来。
    """
    store = make_store()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        compute("m2c_mom", store, d)
        n0 = CALLS["mom"]
        cache = d / "m2c_mom.parquet"
        cached = pd.read_parquet(cache)
        cached.iloc[3:].to_parquet(cache)  # 砍头不砍尾: 末端仍对齐

        df = compute("m2c_mom", store, d)
        assert CALLS["mom"] == n0 + 1, "头部残缺必须触发重算(末端对齐也不放行)"
        assert len(df) == len(store.load("close")), "重算结果必须是完整行集合"
        assert df.index[0] == store.load("close").index[0], "重算后起点必须回到数据起点"
    print("head_truncation_invalidates OK")


def test_source_change_same_name_invalidates():
    """附加: 同名因子源码被替换 → 代码指纹失配重算。

    直接替换 _REGISTRY 里的函数对象, 模拟"改了因子代码后重启 session":
    注册名没变、缓存末端没变, 旧判据完全看不出来, 指纹判据必须看出来。
    """
    store = make_store()
    base._REGISTRY["m2c_swap"] = m2c_swap_v1
    try:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            compute("m2c_swap", store, d)
            n0 = CALLS["swap"]
            base._REGISTRY["m2c_swap"] = m2c_swap_v2  # "改代码"
            df = compute("m2c_swap", store, d)
            assert CALLS["swap"] == n0 + 1, "源码变化必须触发重算"
            expect = store.load("close").pct_change(10, fill_method=None)
            pd.testing.assert_frame_equal(df, expect)
            # 新源码的缓存再次调用应命中
            compute("m2c_swap", store, d)
            assert CALLS["swap"] == n0 + 1, "新源码缓存应命中"
    finally:
        del base._REGISTRY["m2c_swap"]
    print("source_change_same_name_invalidates OK")


if __name__ == "__main__":
    test_hit_on_second_call()
    test_renamed_same_logic_independent()
    test_data_shape_change_invalidates()
    test_stale_end_still_invalidates()
    test_head_truncation_invalidates()
    test_source_change_same_name_invalidates()
    print("ALL GREEN (m2_cache)")
