"""预注册工单: 动手之前把判据写死。

概念出处(全公开): 科学界的 registered reports + AFML 的多重检验纪律;
也是姊妹项目 alpha-court 的核心理念, 此处为 Spencer 原生实现。

为什么(30问#24): 先看结果再定及格线, 判据永远刚好让结果及格。
机制三件套:
1. create_ticket() 开工前落盘 JSON, 同时写内容哈希 sidecar —— 判据冻结;
2. evaluate() 收工时只按工单里冻结的判据打分, 先验哈希 —— 内容被改动过
   的工单直接判 tampered, 不进入打分;
3. 试验预算: 台账实际 N 超过工单预算 → 强制降级(最好也只能 warn) ——
   预算不是装饰, 超了就要在结论上付代价(多重检验的赢家诅咒随 N 涨)。

判据格式(criteria): {结果键: 阈值}。数值键按 result[键] >= 阈值 判;
布尔阈值按 result[键] == 阈值 判。结果键与 eval.panel.run_panel 的返回
对齐(如 ic_mean / t_stat_nw / yearly_all_positive)。
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from .ledger import trial_count


def _canonical(obj: dict) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")


def create_ticket(tickets_dir: Path | str, name: str, hypothesis: str,
                  criteria: dict, trial_budget: int, stoploss: str,
                  notes: str = "") -> Path:
    tickets_dir = Path(tickets_dir)
    tickets_dir.mkdir(parents=True, exist_ok=True)
    ticket = {
        "name": name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hypothesis": hypothesis,
        "criteria": criteria,
        "trial_budget": int(trial_budget),
        "stoploss": stoploss,
        "notes": notes,
    }
    path = tickets_dir / f"{datetime.now():%Y%m%d}_{name}.json"
    if path.exists():
        raise FileExistsError(f"工单已存在, 不允许覆盖(判据冻结): {path}")
    path.write_text(json.dumps(ticket, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    (path.with_suffix(".sha256")).write_text(
        hashlib.sha256(_canonical(ticket)).hexdigest(), encoding="utf-8")
    return path


def load_ticket(path: Path | str) -> dict:
    """读工单并验哈希。返回 dict 带 _tampered 标记(True=内容被改过)。"""
    path = Path(path)
    ticket = json.loads(path.read_text(encoding="utf-8"))
    sha_path = path.with_suffix(".sha256")
    ok = (sha_path.exists()
          and sha_path.read_text(encoding="utf-8").strip()
          == hashlib.sha256(_canonical(ticket)).hexdigest())
    ticket["_tampered"] = not ok
    return ticket


def evaluate(ticket_path: Path | str, result: dict,
             ledger_path: Path | str | None = None,
             factor: str | None = None) -> dict:
    """按冻结判据给结果打分。verdict ∈ pass / warn / fail / tampered。

    - tampered: 工单哈希不符 —— 判据被事后改动, 结果作废;
    - fail: 任一判据不满足;
    - warn: 判据全过但台账 N 超预算(赢家诅咒风险, 结论降级留档);
    - pass: 判据全过且预算内。
    """
    ticket = load_ticket(ticket_path)
    if ticket["_tampered"]:
        return {"verdict": "tampered", "ticket": ticket["name"],
                "detail": "工单内容与冻结哈希不符, 判据可能被事后改动"}

    checks = {}
    for key, th in ticket["criteria"].items():
        val = result.get(key)
        if isinstance(th, bool):
            ok = (val == th)
        else:
            ok = (val is not None) and (val >= th)
        checks[key] = {"value": val, "threshold": th, "ok": ok}

    n_used = trial_count(Path(ledger_path), factor) if ledger_path else None
    over_budget = (n_used is not None) and (n_used > ticket["trial_budget"])

    if not all(c["ok"] for c in checks.values()):
        verdict = "fail"
    elif over_budget:
        verdict = "warn"
    else:
        verdict = "pass"
    return {"verdict": verdict, "ticket": ticket["name"], "checks": checks,
            "n_used": n_used, "trial_budget": ticket["trial_budget"],
            "over_budget": over_budget, "stoploss": ticket["stoploss"]}
