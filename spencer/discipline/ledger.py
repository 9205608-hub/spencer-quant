"""实验台账: append-only。

每跑一次评估就落一行。台账的意义不是记录成功, 而是记录「一共试了多少次」——
试验次数 N 是多重检验校正(DSR/PBO, 见 alpha-court)的输入。删台账 = 骗自己。
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

COLUMNS = ["ts", "factor", "params", "ic_mean", "ic_ir_daily",
           "t_stat_conservative", "yearly_all_positive", "note"]


def log_run(ledger_path: Path, result: dict, params: dict | None = None, note: str = "") -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    new = not ledger_path.exists()
    with open(ledger_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        w.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            result.get("factor"),
            json.dumps(params or {}, ensure_ascii=False, sort_keys=True),
            result.get("ic_mean"),
            result.get("ic_ir_daily"),
            result.get("t_stat_conservative"),
            result.get("yearly_all_positive"),
            note,
        ])


def trial_count(ledger_path: Path, factor: str | None = None) -> int:
    """当前累计试验次数 N —— 报告任何'发现'时必须一起报的数字。"""
    if not ledger_path.exists():
        return 0
    with open(ledger_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if factor:
        rows = [r for r in rows if r["factor"] == factor]
    return len(rows)
