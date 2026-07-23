"""财报基本面的 PIT(point-in-time) 管道 + VALUE(BTOP) 风格 —— 补 30问#33 欠账。

方法出处(全部公开):
- BTOP(账面市值比)定义来自 MSCI Barra CNE5 公开白皮书: 普通股账面价值 /
  当前市值, 逐股约分为 每股净资产 / 收盘价。
- PIT 对齐思想来自 qlib 的 PIT 数据库设计与 AFML 的数据纪律: 财报数值属于
  报告期(report_date), 但市场在披露日(pub_date)才知道它。任何 T 日截面只
  允许使用 T 日之前已公开披露的数值 —— 否则就是用未来财报改写历史截面,
  这是财报类因子最常见的前视偏差来源。

三个入口(职责分离, 为什么拆三层: PIT 逻辑必须能用纯合成数据离线测试,
不能和网络拉取、也不能和具体数据源的列名耦在一起):
- pit_ffill:           纯函数核心, 事件长表 → date×code PIT 宽表;
- fetch_fundamentals:  akshare 业绩报表按报告期批量拉取 + parquet 缓存,
                       网络失败优雅降级读缓存;
- build_value_style:   BTOP 宽表, 形状与 style.build_styles 的其他风格一致,
                       由集成方并入风格 dict(本模块不改 style.py)。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- pit_ffill


def pit_ffill(events: pd.DataFrame, dates: pd.DatetimeIndex, codes) -> pd.DataFrame:
    """事件长表 → date×code PIT 宽表(每格 = 该日可用的最新已披露值)。

    events 列契约: code / pub_date / value, 可选 report_date(报告期)。

    三条规则(每条都是一个前视/污染死法的对策):
    1. **次日生效**: 生效日 = dates 中第一个**严格大于** pub_date 的日期,
       披露日当日不可用。为什么: 财报多在盘后/晚间公告, 披露日收盘前市场
       未必看得到; 统一次日生效是保守口径 —— 宁可晚一天用到, 绝不提前一天
       偷看。副产品: 周末/节假日披露自动落到下一交易日。
    2. **报告期新者优先**: 生效值 = 已披露事件中 report_date 最大者。晚披露
       的旧报告期(如年报审计延迟晚于一季报)不覆盖已生效的新报告期 ——
       财报信息以报告期新旧论优先级, 不以披露先后论。同一报告期的重复披露
       (更正公告)以披露靠后者为准。
    3. **无报告期退化**: events 缺 report_date 列时按 report_date ≡ pub_date
       处理, 退化为"后披露者生效"的普通 PIT ffill。

    实现: 逐 code 按(pub_date, report_date)稳定排序后, 只保留
    report_date == 其运行最大值 的事件(旧报告期在更新报告期披露前仍有效,
    披露后被丢弃), 落到生效位再整列 ffill。生效日在 dates 末端之后的事件
    自然丢弃; 早于 dates[0] 披露的事件从 dates[0] 起即可用。
    """
    dates = pd.DatetimeIndex(dates)
    codes = pd.Index(codes)
    arr = np.full((len(dates), len(codes)), np.nan)

    if events is not None and len(events):
        ev = events.copy()
        ev["pub_date"] = pd.to_datetime(ev["pub_date"])
        if "report_date" in ev.columns:
            ev["report_date"] = pd.to_datetime(ev["report_date"]).fillna(ev["pub_date"])
        else:
            ev["report_date"] = ev["pub_date"]
        ev = ev[ev["value"].notna() & ev["pub_date"].notna() & ev["code"].isin(codes)]

        pos = {c: i for i, c in enumerate(codes)}
        for code, g in ev.groupby("code", sort=False):
            # mergesort 稳定排序: 同 pub_date 同 report_date 时保输入顺序可复现
            g = g.sort_values(["pub_date", "report_date"], kind="mergesort")
            g = g[g["report_date"] == g["report_date"].cummax()]
            idx = dates.searchsorted(g["pub_date"].to_numpy(), side="right")
            j = pos[code]
            for i, v in zip(idx, g["value"].to_numpy()):
                if i < len(dates):
                    arr[i, j] = v  # 同一生效日多条(如年报+一季报同日披露)时, 报告期新者覆盖

    return pd.DataFrame(arr, index=dates, columns=codes).ffill()


# ---------------------------------------------- fetch: akshare 业绩报表 → 缓存

# akshare stock_yjbb_em(date='YYYY0331') 实调返回的列名契约(2026-07 实测):
#   股票代码(6位字符串) / 每股净资产(float) / 最新公告日期('YYYY-MM-DD')
# 列名变更会在 _normalize_yjbb 抛 KeyError, 被按"该报告期拉取失败"处理并警告。
_YJBB_CODE = "股票代码"
_YJBB_BPS = "每股净资产"
_YJBB_PUB = "最新公告日期"

_CACHE_COLS = ["code", "report_date", "pub_date", "bps"]


def _canon(df: pd.DataFrame) -> pd.DataFrame:
    """统一事件表 dtype。为什么: parquet 往返会把日期降为 ms/s 精度,
    与内存里新拉的 ns 精度混拼后 dtype 漂移, 下游比较/断言随机翻车。"""
    df = df[_CACHE_COLS].copy()
    df["code"] = df["code"].astype(str)
    df["report_date"] = pd.to_datetime(df["report_date"]).astype("datetime64[ns]")
    df["pub_date"] = pd.to_datetime(df["pub_date"]).astype("datetime64[ns]")
    df["bps"] = df["bps"].astype(float)
    return df


def _akshare_fetch_quarter(quarter: str) -> pd.DataFrame:
    """真网络调用。延迟导入 akshare: 离线跑 pit_ffill/测试不需要装它。"""
    import akshare as ak

    return ak.stock_yjbb_em(date=quarter)


def _quarter_ends(start_period: str = "20151231") -> list[str]:
    """start_period 起到今天为止已到期的全部报告期(YYYYMMDD, 季度末)。"""
    qs = pd.date_range(pd.Timestamp(start_period), pd.Timestamp.today(), freq="QE")
    return [d.strftime("%Y%m%d") for d in qs]


def _normalize_yjbb(raw: pd.DataFrame, quarter: str) -> pd.DataFrame:
    """akshare 原始列 → 规范事件行: code/report_date/pub_date/bps。

    只做类型转换与列名规范, 不修数据(30问#8: 过滤在使用端做)。
    仅剔除无法构成事件的行: 每股净资产缺失(如北交所部分标的)或公告日期
    缺失。负 BPS(资不抵债)保留 —— 它是真实信息, 去不去极值是下游的事。
    """
    df = raw[[_YJBB_CODE, _YJBB_BPS, _YJBB_PUB]].copy()
    df.columns = ["code", "bps", "pub_date"]
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["bps"] = pd.to_numeric(df["bps"], errors="coerce")
    df["pub_date"] = pd.to_datetime(df["pub_date"], errors="coerce")
    df["report_date"] = pd.Timestamp(quarter)
    df = df.dropna(subset=["bps", "pub_date"])
    return df[_CACHE_COLS]


def fetch_fundamentals(cache_path,
                       start_period: str = "20151231",
                       refetch_recent: int = 3,
                       fetcher=None,
                       pause: float = 0.5) -> pd.DataFrame:
    """按报告期批量拉业绩报表(东财口径, akshare stock_yjbb_em) → 事件长表。

    返回列: code(6位) / report_date / pub_date / bps, 同步落 parquet 缓存。

    设计决定与已知近似(明示不藏):
    - 为什么用业绩报表 bulk 接口: 单次调用返回全市场一个报告期的截面(含
      每股净资产与公告日期), 2015Q4 至今 ~43 个报告期 = ~43 次调用;
      逐股接口要 5000+ 次, 且多数不带公告日期。
    - ★已知近似1: '最新公告日期'是**最新**公告日, 不是首次披露日。发生
      更正/重述时 (值, 日期) 对被最新版本替换。方向上安全: 数值绝不会早于
      任一真实公告日可见(无前视), 代价是可见时点可能晚于首披(保守口径,
      与 pit_ffill 的次日生效同向)。逐版本真 PIT 需要公告快照库, 免费数据
      源不提供。
    - ★已知近似2: 接口是"今天视角"的快照。缓存一经落盘即冻结当时视角,
      已缓存的旧报告期不再刷新(删缓存文件才全量重拉) —— 冻结反而更接近
      PIT: 之后发生的重述不会悄悄改写已落盘的历史。
    - 增量策略: 已缓存报告期跳过; 但最近 refetch_recent 个报告期总是重拉,
      因为披露是滚动的(年报截止次年4月底), 季中拉过的报告期会缺后来才
      披露的公司。
    - 降级策略: 单个报告期拉取失败 → 打印警告并沿用该报告期的缓存行;
      拉取"成功"但归一化后为空帧、且缓存原有该期数据 → 同样视同失败,
      保留缓存旧行并醒目警告(东财限流/打嗝常返回空帧而不抛异常, 空帧
      不能有资格删缓存); 全部失败且缓存为空才抛错。为什么不静默吞:
      缓存陈旧必须让人看见。
    - fetcher 参数 = 依赖注入口: 测试用合成函数替换真网络, 保证测试离线。
    """
    cache_path = Path(cache_path)
    cached = _canon(pd.read_parquet(cache_path)) if cache_path.exists() else None

    quarters = _quarter_ends(start_period)
    have = set() if cached is None else set(cached["report_date"].dt.strftime("%Y%m%d"))
    todo = sorted(set(q for q in quarters if q not in have) | set(quarters[-refetch_recent:]))

    fetcher = fetcher or _akshare_fetch_quarter
    fetched, failed = {}, []
    for q in todo:
        try:
            fetched[q] = _normalize_yjbb(fetcher(q), q)
            if pause > 0:
                time.sleep(pause)  # 对免费接口客气一点, 拉满约多花20秒
        except Exception as e:  # noqa: BLE001 —— 单报告期失败不该炸整条管道
            failed.append(q)
            print(f"[fundamental] 报告期 {q} 拉取失败({type(e).__name__}: {e}), "
                  f"降级沿用缓存")

    # 逐报告期决定新拉结果有没有资格替换缓存。为什么不能"拉成功即替换":
    # 东财接口限流/打嗝时会**成功返回空帧而不抛异常**, 空帧若也算成功,
    # 该报告期的缓存行会被下面的 keep 过滤整期删光且无新行顶替 —— 最新
    # 一季(信息量最大的一季)BTOP 静默回退, 日志还显示"失败 0"一切正常,
    # 违反本模块自己的"缓存陈旧必须让人看见"原则。对策: 新拉为空而缓存
    # 原有该期数据 → 视同拉取失败, 保留缓存旧行 + 醒目警告; 只有非空
    # 新帧才可替换。已知代价(明示): 若上游真把某报告期整期撤下, 本地会
    # 一直沿用缓存旧行 —— 有意的保守选择, 与"缓存冻结更接近PIT"同向。
    cached_qs = (set() if cached is None
                 else set(cached["report_date"].dt.strftime("%Y%m%d")))
    replace_qs, kept_qs = [], []
    for q, f in fetched.items():
        if len(f) == 0 and q in cached_qs:
            kept_qs.append(q)
            print(f"[fundamental] ★警告: 报告期 {q} 新拉为空但缓存原有该期数据"
                  f"(疑似接口限流/打嗝), 保留缓存旧行不替换")
        else:
            replace_qs.append(q)

    parts = [_canon(fetched[q]) for q in replace_qs if len(fetched[q])]
    if cached is not None:
        # 只替换真拿到非空新帧的报告期; 失败/空帧的报告期保留缓存旧行
        keep = cached[~cached["report_date"].dt.strftime("%Y%m%d").isin(replace_qs)]
        parts.insert(0, keep)
    # 空帧不进 concat: 既避开 pandas 空帧拼接的 dtype FutureWarning,
    # 也保证 out 的 dtype 恒由 _canon 决定
    parts = [p for p in parts if len(p)]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=_CACHE_COLS)

    if out.empty:
        raise RuntimeError(
            f"fetch_fundamentals: 网络全部失败且无可用缓存 ({cache_path})。"
            f"失败报告期: {failed}")

    out = (out.sort_values(["code", "pub_date", "report_date"], kind="mergesort")
              .reset_index(drop=True))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path, index=False)
    print(f"[fundamental] 事件 {len(out)} 行, 报告期 {out['report_date'].nunique()} 期 "
          f"(新拉 {len(replace_qs)}, 失败 {len(failed)}, 空帧保缓存 {len(kept_qs)}), "
          f"缓存 {cache_path}")
    return out


# ------------------------------------------------------- BTOP 风格宽表


def _to_bs_code(code6: str):
    """6位代码 → 本仓库宽表列名口径(baostock 'sh./sz.' 前缀)。

    6开头=沪市, 0/3开头=深市; 其余(北交所4/8、老三板等)返回 None 丢弃 ——
    宽表宇宙(a800)不含它们, 强行映射只会制造永远匹配不上的死列。
    """
    if code6.startswith("6"):
        return f"sh.{code6}"
    if code6.startswith(("0", "3")):
        return f"sz.{code6}"
    return None


def build_value_style(store, cache_path,
                      start_period: str = "20151231",
                      fetcher=None) -> pd.DataFrame:
    """BTOP = PIT 每股净资产 / 不复权收盘价, date×code 宽表。

    返回值形状与 style.build_styles 的其他风格一致, 由集成方接线:
        styles["value_btop"] = build_value_style(store, cache_path)
    方向: 值大 = 账面便宜 = 价值暴露高, 与 Barra CNE5 BTOP 同向。

    为什么分母用不复权 close: BPS 是"每股"口径, 必须除以同一股本口径的
    真实价格; 后复权价含累计复权因子, 除出来的不是账面市值比(README
    硬规矩: 价格位置类计算用不复权真实价)。
    ★已知近似3: 送转/拆股改变股本后, 上一份财报的 BPS 仍是旧股本口径,
    与新价格错位, 到下一份财报才刷新。工业级做法用 总净资产/总市值 绕开,
    但免费口径缺可靠的 PIT 总股本时序 —— 明示不藏。
    停牌日 close 为 NaN → BTOP 该格自然为 NaN, 不会用停牌前旧价假装有读数。
    """
    close = store.load("close")
    close = close.where(close > 1e-12)  # 防御 0 价脏数据除法, 实测当前数据无 0

    ev = fetch_fundamentals(cache_path, start_period=start_period, fetcher=fetcher)
    ev = ev.rename(columns={"bps": "value"})
    ev["code"] = ev["code"].map(_to_bs_code)
    ev = ev.dropna(subset=["code"])

    bps = pit_ffill(ev, close.index, close.columns)
    return bps / close
