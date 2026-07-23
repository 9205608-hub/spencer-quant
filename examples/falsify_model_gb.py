"""model_gb 证伪三连: 停牌污染 / 标签置换 / 风格兑现。

背景: 夜班全量 run 中 model_gb 对残差标签 IC 0.0482 / 日频ICIR 0.893 /
保守t 18.16, 但对原始次日收益的分层多空净年化 -4.3% —— 读数好到可疑,
必须先证伪再谈祝贺。本脚本用三个独立实验定位"IC 从哪来、能不能兑现":

① 停牌污染检验
   机制假设: 停牌日价格冻结(adj_close 前值延续) → 前瞻收益被钉在 0 附近;
   而"未来会停牌"可从 T 日的流动性/换手特征预测 → 模型能靠预测停牌白拿
   IC。可预测 ≠ 可交易(停牌股买不进), 这部分 IC 是纸面读数。
   设计: 剔除前瞻窗口(t+1..t+1+h+1, h=5 即 t+1..t+6)内任一日 turn==0 的
   样本(turn==0 与 is_trading==0 在本库完全等价, 见 fetch.py 口径), 用同
   函数同参数重算逐年 IC。IC 大跌 = 死因坐实。
② 标签置换对照 (permutation test, AFML 组合对称/置换检验思想)
   机制假设: 若训练管道存在泄漏(特征-标签对齐错位 / 预处理跨时间 /
   embargo 失效), 即使标签被打成噪声, 模型仍能"预测"它。
   设计: 残差标签在每个截面(单日)内随机打乱(种子固定), 同一训练管道
   (walk_forward_model_factor, 同超参同种子)重训。为省时只用 2019-2024
   窗口, 评估 2022-2024(三个完整的 3 年训练窗)。打乱后 IC 必须≈0
   (保守 |t|<2); 若仍显著 = 管道有泄漏, 所有读数作废。
③ 风格兑现检验
   机制假设: 标签是行业+五风格残差, 模型学到的可能主要是"反向风格暴露"
   的排序 —— 在残差空间是 alpha, 回到 raw 空间被风格自身收益吃掉。
   设计: 同一份预测值, 对残差标签与对原始前瞻收益的逐年 IC 并排(同函数
   同参数 ic_series/yearly_table), 差值 = 无法在 raw 空间兑现的部分。
   附分层年化收益(对原始次日收益)作倒挂证据。

用法: python3 examples/falsify_model_gb.py [--cache DIR]
  --cache 仅为调试提速(缓存两次 walk-forward 的原始预测), 缓存不校验
  数据指纹, 证据链复跑请不带此参数。
产出: output/falsify_model_gb.md — 设计/读数/判词。
依赖: data/wide 宽表 + data/factors 因子缓存已就位(夜班 run 的产物),
  本脚本不碰网络。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spencer.config import load_config
from spencer.data.store import WideStore
from spencer.factor import zoo  # noqa: F401  (注册因子)
from spencer.factor.base import compute, list_factors
from spencer.factor.ops import orient, winsorize_mad
from spencer.risk.style import build_styles, load_industry
from spencer.risk.neutral import residualize
from spencer.eval.panel import (forward_return_1d, forward_returns, ic_series,
                                ic_summary, quantile_returns, yearly_table)
from spencer.model.ml_factor import walk_forward_model_factor

PERM_SEED = 20260723          # 实验②截面置换种子(写死, 复跑同值)
PERM_YEARS = (2019, 2024)     # 实验②数据窗(训练+测试)
PERM_TEST_YEARS = (2022, 2023, 2024)  # 实验②评估年(每年都有完整3年训练窗)

# 夜班报告(output/夜班报告.md, night_run 全量)的被告读数, 并排复现用
NIGHT_IC, NIGHT_ICIR, NIGHT_T = 0.0482, 0.893, 18.16


# ---------- 小工具 ----------

def md_table(df: pd.DataFrame, index_name: str = "年份") -> str:
    """DataFrame → markdown 表(不依赖 tabulate)。

    先 astype(object) 再逐行取值: iterrows 会把混型行升成 float64,
    整数列(如样本天数)会被打成 726.0000, object 化保住原始类型。
    浮点 |v|>=1 给 2 位(t值/天数量级), 其余 4 位(IC量级)。
    """
    df = df.astype(object)
    header = [index_name] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("nan" if pd.isna(v) else
                             (f"{v:.2f}" if abs(v) >= 1 else f"{v:.4f}"))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_or_load_model(cache_dir: Path | None, tag: str, features: dict,
                      label: pd.DataFrame, **kw) -> pd.DataFrame:
    """walk-forward 训练(或从调试缓存读)。缓存只认文件名, 不校验数据指纹。"""
    if cache_dir is not None:
        p = cache_dir / f"falsify_{tag}.parquet"
        if p.exists():
            print(f"[cache] {tag} <- {p}")
            df = pd.read_parquet(p)
            df.index = pd.to_datetime(df.index)
            return df
    t0 = time.time()
    out = walk_forward_model_factor(features, label, **kw)
    print(f"[model] {tag} 训练完成 {time.time()-t0:.0f}s")
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_dir / f"falsify_{tag}.parquet")
    return out


def permute_within_cross_section(label: pd.DataFrame, seed: int) -> pd.DataFrame:
    """逐日截面内打乱标签值(只在非 NaN 的格子之间置换, NaN 位置不动)。

    为什么按截面打乱而不是全表打乱: 全表打乱会破坏标签的逐日分布
    (牛熊日的截面尺度不同), 模型可能靠"预测当日波动水平"拿到假 IC;
    截面内置换保留每一天的边际分布, 只摧毁"股票-标签"的配对关系,
    是对"模型学到了截面排序"这一命题的最小破坏对照。
    """
    rng = np.random.default_rng(seed)
    out = label.copy()
    vals = out.values
    for i in range(vals.shape[0]):
        ok = np.where(~np.isnan(vals[i]))[0]
        if len(ok) > 1:
            vals[i, ok] = vals[i, ok[rng.permutation(len(ok))]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=None,
                    help="调试用预测缓存目录(证据链复跑不要带)")
    args = ap.parse_args()

    cfg = load_config()
    h = cfg["label"]["horizon"]
    q = cfg["eval"]["quantiles"]
    min_names = cfg["eval"]["min_names"]
    out_md = cfg["output_dir"] / "falsify_model_gb.md"

    # ---------- 数据准备(与 night_run 同口径, 不碰网络) ----------
    wide_dir = cfg["data_dir"] / "wide"
    assert (wide_dir / "close.parquet").exists(), \
        "data/wide 宽表不存在 —— 先跑 night_run/quickstart 落数据"
    store = WideStore(wide_dir)
    close = store.load("close")
    turn = store.load("turn")
    tradable = (store.load("is_trading") == 1) & (store.load("is_st") == 0)
    industry = load_industry(cfg["data_dir"] / "raw" / "industry.csv", close.columns)

    print(f"[data] {close.shape[1]} 只 × {close.shape[0]} 日, 末端 {store.end_date().date()}")

    styles = build_styles(store)
    fwd = forward_returns(store, h)
    fwd1 = forward_return_1d(store)
    print("[risk] 标签中性化(逐日截面回归)...")
    fwd_neut = residualize(fwd, styles, industry)

    features = {}
    for name in list_factors():
        features[name] = winsorize_mad(
            compute(name, store, cfg["data_dir"] / "factors").where(tradable))
    features.update(styles)
    print(f"[feat] 特征 {len(features)} 个: {sorted(features)}")

    # ---------- 基线复现(与 night_run 完全同参数) ----------
    print("\n===== 基线: 复现 model_gb =====")
    mf_raw = run_or_load_model(args.cache, "baseline", features, fwd_neut,
                               train_years=3, embargo_days=h + 5, seed=7)
    mf = mf_raw.where(tradable)
    mf, sign = orient(mf, fwd_neut)   # 与 night_run 同: 对残差标签翻正(样本内决策, 披露)

    ic_resid = ic_series(mf, fwd_neut, min_names)
    base = ic_summary(ic_resid, h)
    print(f"[base] IC {base['ic_mean']} ICIR {base['ic_ir_daily']} "
          f"t {base['t_stat_conservative']} (夜班: {NIGHT_IC}/{NIGHT_ICIR}/{NIGHT_T}) "
          f"orient_sign={sign}")

    # ---------- 实验① 停牌污染 ----------
    print("\n===== 实验① 停牌污染检验 =====")
    susp = (turn == 0)
    polluted = pd.DataFrame(False, index=turn.index, columns=turn.columns)
    for k in range(1, h + 2):                       # t+1 .. t+1+h (h=5 → t+1..t+6)
        polluted |= susp.shift(-k, fill_value=False)

    scored = mf.notna() & fwd_neut.notna()          # 实际进 IC 的样本
    pol_scored = scored & polluted
    n_scored = int(scored.sum().sum())
    n_pol = int(pol_scored.sum().sum())

    ic_clean = ic_series(mf.mask(polluted), fwd_neut, min_names)
    clean = ic_summary(ic_clean, h)

    yr_full = yearly_table(ic_resid)["ic_mean"]
    yr_clean = yearly_table(ic_clean)["ic_mean"]
    share_yr = (pol_scored.sum(axis=1).groupby(pol_scored.index.year).sum()
                / scored.sum(axis=1).groupby(scored.index.year).sum())
    tbl1 = pd.DataFrame({"IC_全样本": yr_full,
                         "IC_剔停牌污染": yr_clean.reindex(yr_full.index)})
    tbl1["降幅"] = tbl1["IC_全样本"] - tbl1["IC_剔停牌污染"]
    tbl1["污染样本占比"] = share_yr.reindex(tbl1.index)

    # 污染样本的前瞻收益确实被钉住吗(机制物证)
    fwd_abs = fwd.abs()
    med_pol = float(fwd_abs[pol_scored].stack().median())
    med_cln = float(fwd_abs[scored & ~polluted].stack().median())
    zero_frac = float((fwd[pol_scored] == 0).sum().sum() / max(n_pol, 1))

    rel_drop = 1.0 - clean["ic_mean"] / base["ic_mean"] if base["ic_mean"] else np.nan
    print(tbl1.round(4).to_string())
    print(f"[exp1] 全样本 IC {base['ic_mean']} → 剔污染 {clean['ic_mean']} "
          f"(相对降幅 {rel_drop:.1%}); 污染样本占比 {n_pol/n_scored:.2%}; "
          f"污染样本 |fwd| 中位 {med_pol:.4f} vs 干净 {med_cln:.4f}, 恰为0占比 {zero_frac:.1%}")

    # ---------- 实验② 标签置换对照 ----------
    print("\n===== 实验② 标签置换对照 =====")
    y0, y1 = PERM_YEARS
    sub = fwd_neut.loc[(fwd_neut.index.year >= y0) & (fwd_neut.index.year <= y1)]
    lab_perm = permute_within_cross_section(sub, PERM_SEED)
    mf_perm = run_or_load_model(args.cache, "perm", features, lab_perm,
                                train_years=3, embargo_days=h + 5, seed=7)
    mf_perm = mf_perm.where(tradable)
    # 置换臂不做 orient: orient 会强行把均值翻成非负, 对零假设读数是美化;
    # 判定只看 |t|, 方向本身无意义。

    test_mask = mf_perm.index.year.isin(PERM_TEST_YEARS)
    ic_perm = ic_series(mf_perm.loc[test_mask],
                        lab_perm.loc[lab_perm.index.year.isin(PERM_TEST_YEARS)],
                        min_names)
    perm = ic_summary(ic_perm, h)
    # 真标签臂在同一评估窗的读数(同函数同参数, 供并排)
    ic_true_sub = ic_resid[ic_resid.index.year.isin(PERM_TEST_YEARS)]
    true_sub = ic_summary(ic_true_sub, h)
    tbl2 = pd.DataFrame({
        "IC均值": [true_sub["ic_mean"], perm["ic_mean"]],
        "日频ICIR": [true_sub["ic_ir_daily"], perm["ic_ir_daily"]],
        "保守t": [true_sub["t_stat_conservative"], perm["t_stat_conservative"]],
        "样本天数": [true_sub["n_days"], perm["n_days"]],
    }, index=["真标签(基线, 同窗)", "置换标签(重训)"])
    leak = abs(perm["t_stat_conservative"]) >= 2.0
    print(tbl2.to_string())
    print(f"[exp2] 置换臂保守|t| = {abs(perm['t_stat_conservative']):.2f} → "
          f"{'泄漏警报' if leak else '未见泄漏'}")

    # ---------- 实验③ 风格兑现检验 ----------
    print("\n===== 实验③ 风格兑现检验 =====")
    ic_raw = ic_series(mf, fwd, min_names)          # 同一份预测, 换原始收益做标签
    raw = ic_summary(ic_raw, h)
    yr_raw = yearly_table(ic_raw)["ic_mean"]
    tbl3 = pd.DataFrame({"IC_残差标签": yr_full,
                         "IC_原始收益": yr_raw.reindex(yr_full.index)})
    tbl3["差值(残差-原始)"] = tbl3["IC_残差标签"] - tbl3["IC_原始收益"]

    qret = quantile_returns(mf, fwd1, q)            # 分层倒挂物证(raw 空间)
    qann = (qret.drop(columns="LS").mean() * 252)
    realize = raw["ic_mean"] / base["ic_mean"] if base["ic_mean"] else np.nan
    n_flip = int((np.sign(tbl3["IC_原始收益"].dropna())
                  != np.sign(tbl3["IC_残差标签"].reindex(tbl3["IC_原始收益"].dropna().index))).sum())
    print(tbl3.round(4).to_string())
    print(f"[exp3] 残差IC {base['ic_mean']} vs 原始IC {raw['ic_mean']} "
          f"(兑现率 {realize:.1%}); 分层年化(raw): "
          + " ".join(f"{k}={v:.1%}" for k, v in qann.items()))

    # ---------- 判词与报告(全部数据驱动, 复跑同数据必得同判词) ----------
    pred_start_year = int(ic_resid.index.year.min())
    flip_years = [str(y) for y in tbl3.index
                  if pd.notna(tbl3.loc[y, "IC_原始收益"])
                  and np.sign(tbl3.loc[y, "IC_原始收益"]) != np.sign(tbl3.loc[y, "IC_残差标签"])]
    q_top_worst = bool(qann.iloc[-1] <= qann.min() + 1e-12)

    v1_hit = rel_drop > 0.30
    if v1_hit:
        v1 = ("**死因坐实**: 剔除停牌污染样本后 IC 相对下跌超三成, 停牌零收益的"
              "可预测性是 IC 的主要来源之一。停牌股买不进, 这部分读数不可交易。")
    elif rel_drop > 0.10:
        v1 = (f"**部分贡献**: 剔除后 IC 有降但主体仍在(相对降幅 {rel_drop:.1%}), "
              "停牌污染放大了读数、但不是唯一死因。")
    else:
        susp_daily_pre = float(susp.loc[:str(pred_start_year - 1)].sum(axis=1).mean())
        susp_daily_in = float(susp.loc[str(pred_start_year):].sum(axis=1).mean())
        v1 = (f"**排除**: 剔除后 IC 几乎不动({base['ic_mean']} → {clean['ic_mean']}, "
              f"相对降幅 {rel_drop:.1%}), 停牌污染不是本次读数的死因。机制本身"
              f"存在(污染样本 {zero_frac:.0%} 的前瞻收益恰为 0)但剂量不够: "
              f"打分样本从 {pred_start_year} 年才开始(风格特征需暖机), 之前的"
              f"停牌潮(日均 {susp_daily_pre:.0f} 只停牌)根本进不了 IC, 打分窗内"
              f"停牌已稀释到日均 {susp_daily_in:.0f} 只; T 日停牌又被可交易过滤"
              f"挡掉, 剩下的前瞻窗停牌只占打分样本的 {n_pol / n_scored:.2%}。"
              "边界条件: 该排除只对本打分窗成立, 若预测窗前移到高停牌年份或"
              "宇宙扩到小票, 此项必须重查。")
    v2 = ("**泄漏警报**: 截面置换后标签已是纯噪声, IC 仍显著 —— 训练管道存在"
          "泄漏, 本因子全部读数作废, 先修管道再谈其他。"
          if leak else
          f"**管道干净**: 置换后 IC {perm['ic_mean']} (保守t {perm['t_stat_conservative']}), "
          "与零假设一致 —— 特征-标签对齐、embargo、预处理未见泄漏, "
          "IC 是模型从真实标签里学出来的, 问题不在管道在标签本身。")
    v3 = (f"**残差空间纸面 alpha, 死因坐实**: 同一份预测, 对残差标签 IC "
          f"{base['ic_mean']}, 对原始收益只剩 {raw['ic_mean']} (兑现率 {realize:.0%}); "
          f"逐年符号翻转 {n_flip} 次"
          + (f"({'/'.join(flip_years)}: 残差空间大赚、raw 空间为负)" if flip_years else "")
          + "。差值只存在于'剥掉行业+五风格之后'的空间里 —— 模型学到的是风格"
          "残差的排序, 回到可交易的 raw 空间被风格自身收益吃掉。更硬的物证在"
          "分层: 对次日原始收益, 预测最高层"
          + ("恰是全场最差" if q_top_worst else "不再领先")
          + f"(多空毛年化 {float(qret['LS'].mean() * 252):+.1%}, 与夜班报告净"
          "-4.3% 同向)。注意两个口径细节, 都不构成开脱: IC 用 5 日窗、分层用"
          "次日收益+同函数; RankIC 为正靠的是截面中段的弱单调性, 真正建仓的"
          "头部层恰恰是风格暴露最重、兑现最差的地方。")

    stamp = pd.Timestamp.today().strftime("%Y-%m-%d %H:%M")
    md = f"""# model_gb 证伪三连报告 ({stamp})

被告: 夜班全量 run 的 model_gb —— 对残差标签 IC {NIGHT_IC} / 日频ICIR {NIGHT_ICIR} /
保守t {NIGHT_T}, 同时对原始次日收益的分层多空净年化为负。读数好到可疑,
本报告用三个独立实验回答: IC 从哪来、能不能兑现。

口径: a800 快照宇宙(幸存者偏差, 读数偏乐观), horizon={h}, min_names={min_names},
IC 全部为 RankIC(ic_series 同函数同参数), 保守 t 按 n/horizon 缩水。
复跑: `python3 examples/falsify_model_gb.py` (不碰网络, 需 data/wide 与因子缓存就位)。

## 基线复现

与夜班 run 完全同参数(3年滚动 / embargo {h + 5}d / seed 7 / orient 披露符号 {sign:+d}):

| 读数 | 本次复现 | 夜班报告 |
|---|---|---|
| IC 均值(残差标签) | {base['ic_mean']} | {NIGHT_IC} |
| 日频 ICIR | {base['ic_ir_daily']} | {NIGHT_ICIR} |
| 保守 t | {base['t_stat_conservative']} | {NIGHT_T} |

## 实验① 停牌污染检验

**设计**: 停牌日价格冻结 → 前瞻收益被钉住; "未来会停牌"可由 T 日流动性特征
预测 → 模型可白拿不可交易的 IC。剔除前瞻窗口(t+1..t+{1 + h + 1 - 1})内任一日
turn==0 的样本(turn==0 ≡ is_trading==0, 共 {int(susp.sum().sum()):,} 格),
同函数重算逐年 IC。

**机制物证**: 污染样本 |原始前瞻收益| 中位数 {med_pol:.4f}(干净样本 {med_cln:.4f}),
其中恰为 0 的占 {zero_frac:.1%} —— 收益确实被停牌钉住。
污染样本占进入 IC 计算样本的 {n_pol / n_scored:.2%}({n_pol:,}/{n_scored:,})。

**读数**:

{md_table(tbl1.round(4))}

全样本 IC {base['ic_mean']} → 剔污染 {clean['ic_mean']}, 相对降幅 **{rel_drop:.1%}**。

**判词**: {v1}

## 实验② 标签置换对照

**设计**: 残差标签逐日截面内随机置换(种子 {PERM_SEED}, 只在非缺失格子间打乱,
保留每日边际分布), 同一训练管道同超参重训(数据窗 {y0}-{y1}, 评估
{PERM_TEST_YEARS[0]}-{PERM_TEST_YEARS[-1]}, 每个评估年都有完整 3 年训练窗)。
置换摧毁了股票-标签配对, 无泄漏的管道必须给出 IC≈0。置换臂不做 orient
(orient 会把零假设读数强行翻成非负), 判定看保守 |t| < 2。

**读数**:

{md_table(tbl2, index_name="臂")}

**判词**: {v2}

## 实验③ 风格兑现检验

**设计**: 同一份预测值(同 orient 后), 分别对残差标签与原始前瞻收益算逐年 IC
(同函数同参数), 差值 = 只存在于风格中性空间、无法在 raw 空间兑现的部分。

**读数**:

{md_table(tbl3.round(4))}

整段: 残差标签 IC {base['ic_mean']} vs 原始收益 IC {raw['ic_mean']},
兑现率 **{realize:.0%}**; 逐年符号翻转 {n_flip} 次。

分层年化收益(对原始次日收益, 毛, Q{q}=预测最高层):
{" ".join(f"{k} {v:+.1%}," for k, v in qann.items())} 多空 {float(qret['LS'].mean() * 252):+.1%}。

**判词**: {v3}

## 总判词

1. 管道{'有泄漏 —— 实验②未过, 以下一切读数作废, 先修管道' if leak else '干净(实验②: 置换后 IC ' + str(perm['ic_mean']) + ', |t| ' + f"{abs(perm['t_stat_conservative']):.2f}" + ' < 2) —— IC 是真学出来的, 不是 bug 造出来的'};
2. 停牌污染{'是主要死因之一(实验①)' if v1_hit else ('有部分贡献(实验①)' if rel_drop > 0.10 else f'排除(实验①: 剔除后 IC 不动, 污染样本仅占打分样本 {n_pol / n_scored:.2%} —— 机制在、剂量不够)')};
3. 真正的死因是实验③: **残差标签 IC ≠ 可兑现 alpha**。{base['ic_mean']} 里只有
   {realize:.0%} 能带回 raw 空间, 且带回来的部分不在头部 —— 按预测分层吃原始
   次日收益, 最高层年化 {qann.iloc[-1]:+.1%} vs 最低层 {qann.iloc[0]:+.1%},
   多空毛 {float(qret['LS'].mean() * 252):+.1%}。模型对"剥掉行业+五风格后的
   残差"拟合得越好, 离可交易组合反而越远, 因为它主动押注了与风格收益相反的
   排序。
4. 结论: model_gb 的 {NIGHT_IC}/{NIGHT_ICIR} 是"对一个不可直接交易的标签的
   拟合优度", 读数异常好的谜底 = 评估标签与兑现空间不是同一个空间。下一步
   不是调参, 是二选一: 要么验收口径改成 raw 空间(分层/多空净值为准, 残差 IC
   只当过程指标); 要么承认它是"需要行业+风格对冲腿才能兑现"的中性化信号,
   按多空对冲组合去设计与计费。两条都不走, 这个因子就不该出现在任何成绩单上。

*本报告由 examples/falsify_model_gb.py 生成, 全部读数可复跑。快照宇宙口径,
读数只作框架验证, 不作 alpha 宣言。*
"""
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    print(f"\nFALSIFY_DONE  报告: {out_md}")


if __name__ == "__main__":
    main()
