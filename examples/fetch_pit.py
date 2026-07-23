"""PIT 宇宙数据抓取 (M7 数据基础): 历史时点股票名单(含退市) + 全市场日线。

关键机制: baostock query_all_stock(day=某历史交易日) 返回**该日实际在市**的
证券名单 —— 用历史日期查询就能拿到后来退市的股票, 这是消除幸存者偏差的
关键。名单按月末采样(成分变化以月为尺度足够)。

可断点续跑: 每只股票一个 parquet part(data/raw/pit_parts/), 已存在即跳过;
类型转换与合并留给下游 build 步骤, parts 保持原始字符串。
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import baostock as bs

from spencer.config import load_config
from spencer.data.fetch import ADJ_FIELDS, RAW_FIELDS, BaostockSession, _rs_to_df


def main():
    cfg = load_config()
    raw_dir = cfg["data_dir"] / "raw"
    parts = raw_dir / "pit_parts"
    parts.mkdir(parents=True, exist_ok=True)
    member_path = raw_dir / "pit_membership.parquet"
    end = pd.Timestamp.today().strftime("%Y-%m-%d")

    with BaostockSession():
        # ---- 1. 月末时点在市名单(含后来退市的股票) ----
        if member_path.exists():
            mem = pd.read_parquet(member_path)
            print(f"[pit] 名单已存在: {mem['date'].nunique()} 期, 跳过")
        else:
            td = _rs_to_df(bs.query_trade_dates(start_date=cfg["start"], end_date=end))
            days = pd.to_datetime(td.loc[td["is_trading_day"] == "1", "calendar_date"])
            month_ends = days.groupby([days.dt.year, days.dt.month]).max().tolist()
            rows = []
            for d in month_ends:
                day = d.strftime("%Y-%m-%d")
                df = _rs_to_df(bs.query_all_stock(day=day))
                codes = df.loc[df["code"].str.startswith(("sh.6", "sz.0", "sz.3")), "code"]
                rows.append(pd.DataFrame({"date": day, "code": codes}))
                print(f"[pit] {day}: {len(codes)} 只在市", flush=True)
            mem = pd.concat(rows, ignore_index=True)
            mem["date"] = pd.to_datetime(mem["date"])
            mem.to_parquet(member_path, index=False)

        union = sorted(mem["code"].unique())
        print(f"[pit] 期数 {mem['date'].nunique()}, 股票并集 {len(union)} 只(含退市)", flush=True)

        # ---- 2. 全市场日线, 断点续跑 ----
        done = fail = 0
        t0 = time.time()
        for i, code in enumerate(union, 1):
            p = parts / f"{code}.parquet"
            if p.exists():
                continue
            try:
                raw = _rs_to_df(bs.query_history_k_data_plus(
                    code, RAW_FIELDS, start_date=cfg["start"], end_date=end,
                    frequency="d", adjustflag="3"))
                adj = _rs_to_df(bs.query_history_k_data_plus(
                    code, ADJ_FIELDS, start_date=cfg["start"], end_date=end,
                    frequency="d", adjustflag="1")).rename(columns={"close": "adj_close"})
                if raw.empty:
                    pd.DataFrame({"date": pd.Series(dtype=str),
                                  "code": pd.Series(dtype=str)}).to_parquet(p, index=False)
                else:
                    raw.merge(adj, on=["date", "code"], how="left").to_parquet(p, index=False)
                done += 1
            except Exception:
                fail += 1
                traceback.print_exc()
            if i % 200 == 0:
                el = time.time() - t0
                eta = el / max(done, 1) * (len(union) - i) / 60
                print(f"[pit] {i}/{len(union)} 新增{done} 失败{fail} "
                      f"{el / 60:.0f}min 预计剩余{eta:.0f}min", flush=True)

    print(f"PIT_FETCH_DONE 新增{done} 失败{fail} parts={len(list(parts.glob('*.parquet')))}")


if __name__ == "__main__":
    main()
