"""M6 VALUE 风格断言: 财报 PIT 闸门 + 拉取降级 + BTOP 组装。

跑法: python3 tests/test_m6_value.py  (纯合成数据, 不碰网络 —— 网络路径
用注入的假 fetcher 覆盖, 这正是 fetch_fundamentals 留 fetcher 参数的目的)
"""
import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spencer.risk.fundamental import (_normalize_yjbb, _to_bs_code,
                                      build_value_style, fetch_fundamentals,
                                      pit_ffill)

DATES = pd.bdate_range("2024-01-01", periods=40)   # 工作日索引, 周末不在盘中
CODES = ["s001", "s002", "s003"]


def ev(rows):
    return pd.DataFrame(rows, columns=["code", "report_date", "pub_date", "value"])


def test_pub_gate():
    """披露日当日取不到, 次日起取到; 披露前全 NaN; 周末披露落到下一交易日。"""
    pub = DATES[5]
    w = pit_ffill(ev([("s001", "2023-12-31", pub, 7.5)]), DATES, CODES)

    assert w.loc[:pub, "s001"].isna().all(), "披露日及以前不可用"
    assert (w.loc[DATES[6]:, "s001"] == 7.5).all(), "次日起生效并 ffill 持续"
    assert w["s002"].isna().all() and w["s003"].isna().all(), "无事件的 code 应全 NaN"

    # 周六披露(2024-01-13): 生效日 = 下一交易日周一 2024-01-15
    sat = pd.Timestamp("2024-01-13")
    assert sat not in DATES and sat.dayofweek == 5
    w2 = pit_ffill(ev([("s001", "2023-12-31", sat, 3.0)]), DATES, CODES)
    mon = pd.Timestamp("2024-01-15")
    assert np.isnan(w2.loc[DATES[DATES < mon][-1], "s001"]), "周末披露, 周五仍不可用"
    assert w2.loc[mon, "s001"] == 3.0, "周末披露应从下一交易日生效"
    print("pub_gate OK")


def test_latest_report_period_wins():
    """同 code 多次披露: 生效值 = 已披露事件中报告期最新者。"""
    w = pit_ffill(ev([
        ("s001", "2023-12-31", DATES[5], 1.0),    # 年报
        ("s001", "2024-03-31", DATES[20], 2.0),   # 一季报, 报告期更新
    ]), DATES, CODES)
    assert (w.loc[DATES[6]:DATES[20], "s001"] == 1.0).all(), "新报告期披露前用旧值"
    assert (w.loc[DATES[21]:, "s001"] == 2.0).all(), "新报告期次日起接管"
    print("latest_report_period_wins OK")


def test_out_of_order_disclosure():
    """乱序披露: 晚披露的旧报告期不覆盖已生效的新报告期;
    但同一报告期的更正公告(晚披露)应该覆盖。"""
    # 一季报先出(t=5), 年报审计拖延后出(t=20, 报告期更旧)
    w = pit_ffill(ev([
        ("s001", "2024-03-31", DATES[5], 2.0),
        ("s001", "2023-12-31", DATES[20], 9.9),   # 旧报告期晚披露
    ]), DATES, CODES)
    assert (w.loc[DATES[6]:, "s001"] == 2.0).all(), "旧报告期不得覆盖新报告期"

    # 同一报告期的更正公告: 披露靠后者为准
    w2 = pit_ffill(ev([
        ("s001", "2024-03-31", DATES[5], 2.0),
        ("s001", "2024-03-31", DATES[20], 2.5),   # 更正
    ]), DATES, CODES)
    assert (w2.loc[DATES[6]:DATES[20], "s001"] == 2.0).all()
    assert (w2.loc[DATES[21]:, "s001"] == 2.5).all(), "同报告期更正公告应覆盖"

    # 年报+一季报同日披露(A股常见): 同一生效日, 报告期新者胜出
    w3 = pit_ffill(ev([
        ("s001", "2023-12-31", DATES[5], 1.0),
        ("s001", "2024-03-31", DATES[5], 2.0),
    ]), DATES, CODES)
    assert (w3.loc[DATES[6]:, "s001"] == 2.0).all(), "同日双披露应取报告期新者"
    print("out_of_order_disclosure OK")


def test_edges_and_fallback():
    """边界: 窗前披露从首日可用 / 末日披露丢弃 / 不认识的 code 忽略 /
    缺 report_date 列退化为披露顺序覆盖。"""
    w = pit_ffill(ev([
        ("s001", "2023-09-30", DATES[0] - pd.Timedelta(days=30), 5.0),  # 窗前披露
        ("s002", "2023-12-31", DATES[-1], 6.0),                         # 末日披露, 无次日
        ("zzz9", "2023-12-31", DATES[3], 8.0),                          # 不在 codes 里
    ]), DATES, CODES)
    assert (w["s001"] == 5.0).all(), "窗口开始前已披露 → 首日即可用"
    assert w["s002"].isna().all(), "最后一天披露, 窗口内永不生效"
    assert w["s003"].isna().all()

    # 无 report_date 列: 退化为"后披露者生效"
    nak = pd.DataFrame({"code": ["s001", "s001"],
                        "pub_date": [DATES[5], DATES[20]],
                        "value": [1.0, 2.0]})
    w2 = pit_ffill(nak, DATES, CODES)
    assert (w2.loc[DATES[6]:DATES[20], "s001"] == 1.0).all()
    assert (w2.loc[DATES[21]:, "s001"] == 2.0).all()

    # 空事件表: 全 NaN 不报错
    w3 = pit_ffill(ev([]), DATES, CODES)
    assert w3.isna().all().all() and w3.shape == (len(DATES), len(CODES))
    print("edges_and_fallback OK")


# ---------------------------------------------------------- fetch 层(离线)

def fake_yjbb_raw(rows):
    """按 akshare stock_yjbb_em 实测列名造原始帧(含无关列, 验证解析只取所需)。"""
    df = pd.DataFrame(rows, columns=["股票代码", "每股净资产", "最新公告日期"])
    df.insert(0, "序号", range(1, len(df) + 1))
    df["净利润-净利润"] = 1.0
    return df


def test_normalize_and_code_map():
    raw = fake_yjbb_raw([
        ("600000", 20.5, "2024-04-30"),
        ("000001", 18.0, "2024-04-29"),
        ("871234", np.nan, "2024-04-28"),   # 每股净资产缺失 → 剔除
        ("300750", 12.0, None),             # 公告日期缺失 → 剔除
    ])
    out = _normalize_yjbb(raw, "20240331")
    assert list(out.columns) == ["code", "report_date", "pub_date", "bps"]
    assert len(out) == 2 and set(out["code"]) == {"600000", "000001"}
    assert (out["report_date"] == pd.Timestamp("20240331")).all()

    assert _to_bs_code("600000") == "sh.600000"
    assert _to_bs_code("000001") == "sz.000001"
    assert _to_bs_code("300750") == "sz.300750"
    assert _to_bs_code("871234") is None, "北交所应丢弃"
    print("normalize_and_code_map OK")


def test_fetch_cache_and_degrade():
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "fund.parquet"

        def good(quarter):
            if quarter == "20260331":
                return fake_yjbb_raw([("600000", 20.5, "2026-04-25")])
            return fake_yjbb_raw([])          # 其余报告期: 空截面

        def dead(quarter):
            raise ConnectionError("网络挂了")

        # 首拉: 落缓存
        out1 = fetch_fundamentals(cache, start_period="20260331",
                                  fetcher=good, pause=0)
        assert cache.exists() and len(out1) == 1
        assert out1.loc[0, "code"] == "600000" and out1.loc[0, "bps"] == 20.5

        # 网络全挂: 优雅降级, 内容与缓存一致
        out2 = fetch_fundamentals(cache, start_period="20260331",
                                  fetcher=dead, pause=0)
        pd.testing.assert_frame_equal(out1, out2)

        # 重拉窗口内数据更新(披露滚动补齐): 新行进来, 旧行仍在
        def good2(quarter):
            if quarter == "20260331":
                return fake_yjbb_raw([("600000", 20.5, "2026-04-25"),
                                      ("000001", 18.0, "2026-04-28")])
            return fake_yjbb_raw([])
        out3 = fetch_fundamentals(cache, start_period="20260331",
                                  fetcher=good2, pause=0)
        assert set(out3["code"]) == {"600000", "000001"}

        # 无缓存 + 网络全挂 = 必须炸, 不许静默给空表
        try:
            fetch_fundamentals(Path(tmp) / "nocache.parquet",
                               start_period="20260331", fetcher=dead, pause=0)
            raise AssertionError("无缓存且全失败时应抛 RuntimeError")
        except RuntimeError:
            pass
    print("fetch_cache_and_degrade OK")


def test_empty_frame_keeps_cache():
    """限流打嗝回归: fetcher 成功返回空帧(不抛异常)不得删掉该报告期缓存行。

    对抗性复核发现的静默数据丢失: 空帧曾被算作'拉成功'进 ok_quarters →
    keep 过滤把该报告期缓存整期删光且无新行顶替, 日志还显示'失败 0'。
    修后契约: ①缓存旧行保留(返回值与磁盘缓存都在) ②打印醒目警告
    ③恢复后非空新帧照常替换。"""
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "fund.parquet"

        def good(quarter):
            if quarter == "20260331":
                return fake_yjbb_raw([("600000", 20.5, "2026-04-25")])
            if quarter == "20260630":
                return fake_yjbb_raw([("000001", 18.0, "2026-07-10")])
            return fake_yjbb_raw([])

        out1 = fetch_fundamentals(cache, start_period="20260331",
                                  fetcher=good, pause=0)
        assert {"20260331", "20260630"} <= set(
            out1["report_date"].dt.strftime("%Y%m%d"))

        # 二拉: 20260331 返回空帧(限流打嗝, 不抛异常), 20260630 正常
        def hiccup(quarter):
            if quarter == "20260630":
                return fake_yjbb_raw([("000001", 18.0, "2026-07-10")])
            return fake_yjbb_raw([])          # 20260331: 空帧

        buf = io.StringIO()
        with redirect_stdout(buf):
            out2 = fetch_fundamentals(cache, start_period="20260331",
                                      fetcher=hiccup, pause=0)
        q1 = out2[out2["report_date"] == pd.Timestamp("20260331")]
        assert len(q1) == 1 and q1.iloc[0]["code"] == "600000", \
            "空帧不得删缓存: 20260331 的缓存行必须还在"
        # 磁盘缓存本身也必须保住, 不只是本次返回值
        disk = pd.read_parquet(cache)
        assert (pd.to_datetime(disk["report_date"])
                == pd.Timestamp("20260331")).sum() == 1, "磁盘缓存整期消失"
        log = buf.getvalue()
        assert "警告" in log and "20260331" in log, "空帧保缓存必须醒目警告, 不许静默"

        # 三拉恢复: 非空新帧(更正公告)照常替换, 保守逻辑不阻塞正常更新
        def recovered(quarter):
            if quarter == "20260331":
                return fake_yjbb_raw([("600000", 21.0, "2026-05-05")])
            if quarter == "20260630":
                return fake_yjbb_raw([("000001", 18.0, "2026-07-10")])
            return fake_yjbb_raw([])
        out3 = fetch_fundamentals(cache, start_period="20260331",
                                  fetcher=recovered, pause=0)
        q1b = out3[out3["report_date"] == pd.Timestamp("20260331")]
        assert len(q1b) == 1 and q1b.iloc[0]["bps"] == 21.0, \
            "恢复后非空新帧应正常替换缓存"
    print("empty_frame_keeps_cache OK")


# ---------------------------------------------------------- BTOP 组装(离线)

class FakeStore:
    def __init__(self, frames):
        self.frames = frames

    def load(self, field):
        return self.frames[field]


def test_build_value_style():
    dates = pd.bdate_range("2026-04-01", periods=30)
    codes = ["sh.600000", "sz.000001"]
    close = pd.DataFrame(10.0, index=dates, columns=codes)
    close.loc[dates[25], "sh.600000"] = np.nan          # 模拟一天停牌
    store = FakeStore({"close": close})

    pub = dates[10]

    def fetcher(quarter):
        if quarter == "20260331":
            return fake_yjbb_raw([("600000", 20.0, pub.strftime("%Y-%m-%d")),
                                  ("000001", 5.0, pub.strftime("%Y-%m-%d")),
                                  ("871234", 9.0, pub.strftime("%Y-%m-%d"))])
        return fake_yjbb_raw([])

    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "fund.parquet"
        btop = build_value_style(store, cache, start_period="20260331",
                                 fetcher=fetcher)

        assert btop.shape == close.shape and list(btop.columns) == codes
        assert btop.loc[:pub, "sh.600000"].isna().all(), "披露日及以前 BTOP 不可用"
        assert abs(btop.loc[dates[11], "sh.600000"] - 20.0 / 10.0) < 1e-12
        assert abs(btop.loc[dates[11], "sz.000001"] - 5.0 / 10.0) < 1e-12
        assert np.isnan(btop.loc[dates[25], "sh.600000"]), "停牌日 close NaN → BTOP NaN"
        assert not np.isnan(btop.loc[dates[26], "sh.600000"]), "复牌恢复读数"

        # 网络挂掉也能出面板(读缓存) —— BTOP 数值一致
        def dead(quarter):
            raise ConnectionError("网络挂了")
        btop2 = build_value_style(store, cache, start_period="20260331",
                                  fetcher=dead)
        pd.testing.assert_frame_equal(btop, btop2)
    print("build_value_style OK")


if __name__ == "__main__":
    test_pub_gate()
    test_latest_report_period_wins()
    test_out_of_order_disclosure()
    test_edges_and_fallback()
    test_normalize_and_code_map()
    test_fetch_cache_and_degrade()
    test_empty_frame_keeps_cache()
    test_build_value_style()
    print("ALL GREEN (m6_value)")
