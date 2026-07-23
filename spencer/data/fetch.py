"""数据拉取层：baostock 日线 → 长表 parquet。

口径说明（每一条都是前视偏差相关的决定，不是随手选的）：
- 同时拉「不复权」与「后复权」两套价格：收益计算用后复权（锚定上市日，
  只用历史信息向前推，PIT 安全）；筹码/价格位置类因子用不复权真实价。
  前复权从今天往回改写历史，是研究场景的前视偏差源，本项目不使用。
- turn(换手率) = 成交量/流通股本，baostock 返回百分数，这里统一转小数并
  clip 到 [0,1]，停牌日置 0。
- tradestatus/isST 一并落盘，宇宙过滤在下游做，原始数据不做删改。
"""
from __future__ import annotations

import time
from pathlib import Path

import baostock as bs
import pandas as pd

RAW_FIELDS = "date,code,open,high,low,close,volume,amount,turn,tradestatus,isST"
ADJ_FIELDS = "date,code,close"

NUMERIC_COLS = ["open", "high", "low", "close", "volume", "amount", "turn"]


class BaostockSession:
    """with 语义的登录会话，避免忘记 logout。"""

    def __enter__(self):
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"baostock login failed: {lg.error_msg}")
        return self

    def __exit__(self, *exc):
        bs.logout()
        return False


def _rs_to_df(rs) -> pd.DataFrame:
    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def get_trade_dates(start: str, end: str) -> pd.DatetimeIndex:
    rs = bs.query_trade_dates(start_date=start, end_date=end)
    df = _rs_to_df(rs)
    days = df.loc[df["is_trading_day"] == "1", "calendar_date"]
    return pd.DatetimeIndex(pd.to_datetime(days))


def get_universe(name: str) -> list[str]:
    """指数成分股（注意：baostock 只给当前快照，PIT 化留待 M7，README 有说明）。

    a800 = 沪深300 ∪ 中证500，接近中大盘研究宇宙的常用起点。
    """
    if name == "a800":
        return sorted(set(get_universe("hs300")) | set(get_universe("zz500")))
    query = {"hs300": bs.query_hs300_stocks, "zz500": bs.query_zz500_stocks}[name]
    df = _rs_to_df(query())
    return sorted(df["code"].tolist())


def fetch_industry(out_csv: Path) -> pd.DataFrame:
    """全市场行业分类快照(baostock, 申万口径), 一次一张表。

    快照非 PIT —— 历史时点的行业归属会被今天的分类改写, 与宇宙快照同源的
    已知欠账(30问 #6/#31)。
    """
    df = _rs_to_df(bs.query_stock_industry())
    df.loc[df["industry"] == "", "industry"] = "未分类"
    df[["code", "industry"]].to_csv(out_csv, index=False)
    return df


def fetch_daily(codes: list[str], start: str, end: str, verbose_every: int = 20) -> pd.DataFrame:
    """逐只拉取日线（raw + 后复权 close），返回合并长表。"""
    frames = []
    t0 = time.time()
    for i, code in enumerate(codes, 1):
        raw = _rs_to_df(
            bs.query_history_k_data_plus(
                code, RAW_FIELDS, start_date=start, end_date=end,
                frequency="d", adjustflag="3",
            )
        )
        adj = _rs_to_df(
            bs.query_history_k_data_plus(
                code, ADJ_FIELDS, start_date=start, end_date=end,
                frequency="d", adjustflag="1",
            )
        )
        if raw.empty:
            continue
        adj = adj.rename(columns={"close": "adj_close"})
        df = raw.merge(adj, on=["date", "code"], how="left")
        frames.append(df)
        if i % verbose_every == 0:
            print(f"  fetch {i}/{len(codes)}  ({time.time() - t0:.0f}s)")
    long_df = pd.concat(frames, ignore_index=True)

    for col in NUMERIC_COLS + ["adj_close"]:
        long_df[col] = pd.to_numeric(long_df[col], errors="coerce")
    long_df["date"] = pd.to_datetime(long_df["date"])
    # 换手率: 百分数→小数, 非交易日/停牌置0, 异常值clip
    long_df["turn"] = (long_df["turn"] / 100.0).fillna(0.0).clip(0.0, 1.0)
    long_df.loc[long_df["tradestatus"] != "1", "turn"] = 0.0
    long_df["is_st"] = (long_df["isST"] == "1").astype("int8")
    long_df["is_trading"] = (long_df["tradestatus"] == "1").astype("int8")
    return long_df.drop(columns=["isST", "tradestatus"])


def fetch_to_parquet(cfg: dict, limit: int | None = None) -> Path:
    """入口：按 config 拉数据落盘，返回长表路径。已存在则跳过（增量留待后续）。"""
    raw_dir = cfg["data_dir"] / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / "daily_long.parquet"
    if out.exists():
        print(f"[fetch] 已存在 {out}, 跳过拉取 (删除该文件可强制重拉)")
        return out

    end = cfg["end"] or pd.Timestamp.today().strftime("%Y-%m-%d")
    with BaostockSession():
        codes = get_universe(cfg["universe"])
        if limit:
            codes = codes[:limit]
        print(f"[fetch] universe={cfg['universe']} n={len(codes)} {cfg['start']}→{end}")
        long_df = fetch_daily(codes, cfg["start"], end)
    long_df.to_parquet(out, index=False)
    print(f"[fetch] 落盘 {out}  rows={len(long_df):,}")
    return out
