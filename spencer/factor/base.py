"""因子注册表 + 缓存(指纹化)。

两条硬规矩：
1. 因子函数只接收 store，输出 date×code 宽表 —— 输入输出契约唯一。
2. 缓存新鲜度 = 「指纹匹配」+「末端对齐」两关都过, 任一不过一律重算,
   杜绝"数据更新了/代码改了, 因子还是旧的"这类静默腐烂(30问#14 的回答)。

指纹 = 三段, 存 sidecar json(`<name>.fingerprint.json`):
- 代码指纹(code): 因子函数源码的 sha256(inspect.getsource)。为什么: 缓存的
  本质是"同样的计算不重跑", 而"同样的计算"必须包含代码本身 —— 改了因子
  逻辑但末端没变, 旧判据(只看末端)会把旧逻辑的值当新鲜缓存返回。
- 数据形状指纹(data): close 面板(与 store.end_date() 同锚点)的形状签名 =
  起始日期 + 末端日期 + 行数 + 列名哈希。为什么每一项都不能省:
  起点变(窗口平移/扩样本)→ 递推/滚动类因子每个值都变; 列名哈希而非列数
  → 宇宙换血(指数调仓换成分、列数恰好不变)时列数看不出来, 旧宇宙的因子
  矩阵会冒充新宇宙返回; 行数 → 中间交易日被补插/删除时起止日期不变,
  只有行数看得出来。
- 输出签名(out): 写入时因子输出宽表自身的形状签名(同上四要素)。为什么:
  命中时用它校验缓存 parquet 未被外部改动 —— 只查"末端对齐"护尾不护头,
  头部被截的残缺缓存末端仍对齐, 会被当新鲜返回。不拿 store 形状当参照,
  因为契约只要求因子末端==数据末端, 输出的行/列集合不必等于输入面板
  (如递推类因子内部 warmup 丢头部行), 唯一可靠参照是写入时的输出本身。

为什么用 sidecar json 而不是把指纹编进文件名: 文件名方案会在每次改代码后
留下一堆孤儿缓存文件; sidecar 方案一因子恒一缓存+一指纹, 目录不会膨胀。
写入顺序 parquet 在前、sidecar 在后 —— sidecar 是提交标记, 中途崩溃只会
留下"无/旧指纹的 parquet", 下次判定为过期重算, 不会把半成品当新鲜缓存。

方法出处: 函数源码哈希入缓存键是 joblib.Memory 的公开做法; "数据指纹决定
缓存有效性"是 qlib 表达式缓存的公开设计思想。

已知近似(明示不藏):
- 代码指纹只覆盖因子函数本体。因子调用的工具函数或全局常数改了, 指纹
  不变 → 不会失效。全依赖图哈希要 AST 级追踪, v0.1 不做, 先堵最大的坑。
- 数据形状指纹看 起止日期+行数+列名, 不看值本身。行列集合不变、某格的值
  被原地修订时指纹不变 → 不会失效。全值哈希每次命中都要读全量数据, 缓存
  就失去意义; 落盘层"原始值不删改"(30问#8)是这条近似成立的前提。
- 形状签名对列名取哈希, 顺带把"列序变化"也判为过期 —— 多算不漏算,
  错的方向是安全的(多余重算, 不会旧值冒充新值)。
- 旧版缓存没有 sidecar、或 sidecar 缺任一指纹段(如缺输出签名的两段旧格式)
  → 一律视为过期, 升级后首次会全量重算一次。
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from ..data.store import WideStore

_REGISTRY: dict[str, Callable[[WideStore], pd.DataFrame]] = {}
_META: dict[str, dict] = {}


def factor(name: str, **meta):
    """注册装饰器: @factor("mom_20_5") 或带登记元信息:

        @factor("ep_ttm", author="spencer", tags=["fundamental"],
                valid_from="2016-06-30", data_deps=["业绩报表"])

    元信息是行业通行的"因子登记"概念(作者/标签/数据依赖/有效起始日),
    供入库验证契约(verify.admission_check)与面板判读消费。
    刻意不参与缓存指纹 —— 改备注不应触发因子重算。
    """
    def deco(fn):
        if name in _REGISTRY:
            raise KeyError(f"因子重名: {name}")
        _REGISTRY[name] = fn
        _META[name] = dict(meta)
        fn.factor_name = name
        return fn
    return deco


def get_meta(name: str) -> dict:
    """因子登记元信息(拷贝, 防外部原地改注册表)。未登记的键返回空 dict。"""
    return dict(_META.get(name, {}))


def list_factors() -> list[str]:
    return sorted(_REGISTRY)


def _code_fingerprint(fn: Callable) -> str:
    """因子函数源码哈希。

    inspect.getsource 拿不到源码时(exec/交互式定义)退回字节码+常量+符号名
    的哈希 —— 比源码哈希粗(丢注释/格式), 但仍能捕捉逻辑变化, 且绝不因
    拿不到源码而放弃指纹(放弃 = 退化回旧的只看末端判据)。
    """
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        code = fn.__code__
        src = repr((code.co_code, code.co_consts, code.co_names))
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]


def _panel_signature(df: pd.DataFrame) -> str:
    """宽表形状签名: 起始日期~末端日期~行数~列名哈希。

    数据形状指纹与输出签名共用同一定义 —— 一个签名函数, 两处使用,
    避免两套"形状"口径各自漂移。列名取 sha256 前 16 位而非直存列表:
    a800 宇宙列名全文 ~7KB, 哈希后 sidecar 保持人眼可读的一行。
    """
    start = pd.Timestamp(df.index[0]).date()
    end = pd.Timestamp(df.index[-1]).date()
    cols = hashlib.sha256(",".join(map(str, df.columns)).encode("utf-8")).hexdigest()[:16]
    return f"{start}~{end}~{df.shape[0]}~{cols}"


def _data_fingerprint(store: WideStore) -> str:
    """数据形状指纹 = close 面板的形状签名。

    锚点字段用 close, 与 store.end_date() 同源 —— 末端锚点唯一(30问#4),
    形状指纹不另立锚点。
    """
    return _panel_signature(store.load("close"))


def _fingerprint(fn: Callable, store: WideStore) -> dict[str, str]:
    return {"code": _code_fingerprint(fn), "data": _data_fingerprint(store)}


def _read_sidecar(path: Path) -> dict | None:
    """读 sidecar 指纹; 缺失/损坏一律返回 None(= 视为过期, 不抛错)。"""
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    # 合法 JSON 但不是 dict(如被写成列表)同样算损坏 —— 返回 None 走重算,
    # 不让 .get() 在命中路径上抛 AttributeError
    return obj if isinstance(obj, dict) else None


def compute(name: str, store: WideStore, cache_dir: Path, use_cache: bool = True) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{name}.parquet"
    sidecar = cache_dir / f"{name}.fingerprint.json"
    data_end = store.end_date()
    fp = _fingerprint(_REGISTRY[name], store)

    if use_cache and cache.exists():
        old = _read_sidecar(sidecar)
        if old is None:
            print(f"[factor] {name} 缓存无指纹(旧版缓存或写入中断), 重算")
        elif old.get("code") != fp["code"]:
            print(f"[factor] {name} 因子源码指纹变化, 重算")
        elif old.get("data") != fp["data"]:
            print(f"[factor] {name} 数据形状指纹 {old.get('data')} != {fp['data']}, 重算")
        elif not old.get("out"):
            print(f"[factor] {name} sidecar 缺输出签名(旧格式), 重算")
        else:
            df = pd.read_parquet(cache)
            df.index = pd.to_datetime(df.index)
            if len(df) == 0 or _panel_signature(df) != old["out"]:
                # 缓存文件与写入时的输出签名不符 = parquet 被外部改动
                # (截头/截尾/换列都在这里被抓), sidecar 再新鲜也不算数
                print(f"[factor] {name} 缓存文件与写入时输出签名不符(被外部改动), 重算")
            elif df.index[-1] != data_end:
                # 末端判据: 逻辑上已被"数据指纹+输出签名"蕴含, 仍独立保留 ——
                # 它不依赖指纹实现的正确性, 是30问#4/#10的末端锚点最后一道闸
                print(f"[factor] {name} 缓存末端 {df.index[-1].date()} != 数据末端 {data_end.date()}, 重算")
            else:
                return df

    df = _REGISTRY[name](store)
    assert df.index[-1] == data_end, (
        f"{name}: 因子末端 {df.index[-1].date()} != 数据末端 {data_end.date()} —— "
        f"因子必须铺满到数据末端"
    )
    df.to_parquet(cache)
    # sidecar 最后写 = 提交标记: 崩在中间只会留下"无指纹缓存", 下次自动重算。
    # out = 落盘那一刻输出的形状签名, 命中时校验 parquet 未被外部改动
    record = dict(fp, out=_panel_signature(df))
    sidecar.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return df
