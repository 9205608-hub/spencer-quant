# spencer ↔ alpha-court 统计实现交叉验证

测试所在: `tests/test_cross_validation_court.py`
运行前提: 环境变量 `ALPHA_COURT_PATH` 指向 alpha-court 仓根目录
(缺省 `/Users/spensir/Desktop/alpha-court`)。路径不存在或 `court` 不可
import 时该测试模块整体 `pytest.skip`, 不硬红。

```
python3 -m pytest tests/test_cross_validation_court.py -q
ALPHA_COURT_PATH=/nonexistent python3 -m pytest tests/test_cross_validation_court.py -q
```

两侧实现全部冻结, 本文件只记录对照结论。spencer 源码与 alpha-court 仓
零改动。分歧本身是信息: 解释得了的写在这里, 解释不了的会让测试失败。

import 陷阱: `court/__init__.py` 把函数 `dsr` 重导出, Python ≥ 3.7 下
`import court.dsr as x` 会把 `x` 绑到**函数**而非模块。测试一律
`importlib.import_module("court.dsr")`(`pbo` / `sharpe` 同理)。

---

## 1. 对照范围

| 量 | spencer | court | 文献 |
|---|---|---|---|
| 期望最大夏普 SR0 | `expected_max_sharpe(N, T, var_trials=V)` | `expected_max_sr(0.0, √V, N)` | Bailey & López de Prado 2014 Eq. (1) / App. A.1 |
| 缩水夏普 DSR | `deflated_sharpe(...)["dsr"]` | `dsr(...).dsr` | 2014 Eq. (2) = PSR(SR0) |
| 概率夏普 PSR | `probabilistic_sharpe` | `sharpe.psr` | Bailey & López de Prado 2012 Eq. (11) |
| 过拟合概率 PBO | `pbo_cscv(...)["pbo"]` | `pbo_cscv(...).phi` | Bailey, Borwein, López de Prado & Zhu 2017 Alg. 2.3 |

两边零代码依赖, 各自从同一批公开文献重写。

---

## 2. 参数映射表 (实测钉死)

### 2.1 三个容易翻车的约定

| 约定 | spencer | court | 实测结论 | 映射 |
|---|---|---|---|---|
| σ vs σ² | `var_trials` 是跨试验 Sharpe **方差** V, 公式里自己开方 | `sr_trials_std` 是跨试验 Sharpe **标准差** σ | 同一式; 必须 `σ = √V` | `court_std = sqrt(spencer_V)` |
| 峰度 | 原始峰度, 正态 = 3 (docstring 明示; pandas `.kurt()` 要 +3) | `kurt_hat` 原始峰度, 正态 = 3 (`scipy.stats.kurtosis(..., fisher=False)`; `docs/research/dsr.md` §2.a) | **两边同为原始峰度, 不换算** | 原样传递 |
| PSR 自由度 | 分子 `√(T-1)` (Bessel, 2012 Eq. 11) | 分子 `√(n_obs-1)` (同式) | **两边都是 √(T-1), 不是 √T** | 原样传递 |

H1 网格 (N ∈ {2,10,100,1000} × T ∈ {50,252,1000} × V ∈ {1/T, 0.5/T})
在正确映射 `σ=√V` 下 max\|ΔSR0\| = 3.9×10⁻¹⁶; 故意不换算 (把 V 当 σ
传给 court) 时 max\|Δ\| = 0.40, 断言 1e-9 立刻红 —— 见 receipt 红先记录。

H2 网格 (sr ∈ {0, 0.05, 0.1, 0.2} × N ∈ {2,10,100} × T ∈ {50,252} ×
(skew, kurt) ∈ {(0,3), (-0.5,5)}) 在「原始峰度 + σ=√V」下
max\|Δdsr\| = 1.1×10⁻¹⁵。故意把超额峰度 `kurt-3` 传给 court 时,
非正态格子 max\|Δdsr\| ≈ 3.6×10⁻³, 同样破 1e-9。

PSR 手写对照 (sr=0.1, T=50, skew=-0.5, kurt=5):

- spencer = court = Φ[√(T-1) 式] = 0.7517161419963828 (Δ = 0)
- Φ[√T 式] = 0.7538965036720922, 与实现差 2.2×10⁻³

court 公开导出 `expected_max_sr`; H1 同时用该函数和 `DsrResult.sr_star`
对照, 字段名就是 `sr_star`。

### 2.2 完整调用对照

```
# SR0
spencer.expected_max_sharpe(N, T, var_trials=V)
    == court.dsr.expected_max_sr(0.0, sqrt(V), N)
    == court.dsr.dsr(sr, T, skew, kurt, sqrt(V), N).sr_star

# DSR
spencer.deflated_sharpe(sr, N, T, skew, kurt, var_trials=V)["dsr"]
    == court.dsr.dsr(sr, T, skew, kurt, sqrt(V), N).dsr

# PSR (基准 0, 用于分解)
spencer.probabilistic_sharpe(sr, 0.0, T, skew, kurt)
    == court.sharpe.psr(sr, 0.0, T, skew, kurt)

# PBO  (T % n_splits == 0, 无 NaN)
spencer.pbo_cscv(M, n_splits=S)["pbo"]
    == court.pbo.pbo_cscv(M, S, court.sharpe.sharpe_ratio).phi
```

两边实现的都是文献原式, 不是各自发明的变体。数值差来自
`statistics.NormalDist.inv_cdf` vs `scipy.stats.norm.isf` 的尾部量化,
本票网格内 ≤ 1e-15; N=10⁶ 时分位数差仍在 10⁻¹¹ 量级。

---

## 3. 假设裁决

| 假设 | 裁决 | 说明 |
|---|---|---|
| H1 (SR0, σ↔σ²) | **成立** | 换算 `σ=√V` 后逐格 \|Δ\|≤1e-9; 不换算则红 |
| H2 (DSR) | **成立** | 峰度不换算、σ=√V 后逐格 \|Δdsr\|≤1e-9 |
| H3 (PBO φ) | **成立** | 三张种子矩阵上 φ 逐位相同 (Δ=0, 远小于 1e-12) |

没有需要推翻后改写映射的格子。

---

## 4. 边界行为差异清单

这些差异**解释得了**, 不构成 H1–H3 失败。测试
`test_pbo_boundary_tail_and_nan` / `test_dsr_boundary_var_factor_nonpositive`
把它们钉成断言 (断言的是「两边行为不同」, 不是「两边数值相同」)。

| # | 场景 | spencer | court | 谁更贴近文献 / 备注 |
|---|---|---|---|---|
| B1 | T 不能被 n_splits 整除 | 丢弃尾部不足整块的行 (`M[:L*S]`), 继续算 | `ValueError: T must be divisible` | 论文 Alg. 2.3 要求块等长; court 硬约束, spencer 静默截断。截断后若再喂 court, φ 一致 |
| B2 | 矩阵含 NaN | `AssertionError` (中文消息) | `ValueError: finite entries` | 两边都不静默传播; 异常类型不同 |
| B3 | 块内 Sharpe 的 ddof | `ddof=1` (预计算 sum/sum² 后用 T_half-1) | `sharpe_ratio` → `np.std(..., ddof=1)` | **一致**, Bessel σ |
| B4 | 评价指标 | 写死每期 Sharpe | `metric` 必传, 本票传 `sharpe_ratio` | court 按论文「metric-agnostic」做成插件 |
| B5 | OOS 秩的平局 | `(oos_sr ≤ star).sum() / (N+1)` = 平局时给冠军该组**最高秩** | `rankdata(..., method="average")` = **中位秩** | 论文假设唯一最优; 无平局时两式恒等。本票三张连续高斯矩阵几乎不可能平局, φ 逐位相同。有平局时 φ 可能差 1/C(S,S/2) |
| B6 | IS 冠军平局 | `np.argmax` = 最小下标 | 同 (`np.argmax`) | 一致 |
| B7 | var_factor ≤ 0 (极端偏度/峰度) | PSR/DSR 返回 `nan` | `ValueError` | 文献未规定越出适用域的返回; spencer 让调用方面对 nan, court fail-closed |
| B8 | N=1 的 SR0 | 返回 0.0 (无选择效应) | `expected_max_sr` 返回 `sr_trials_mean` (零假设下也是 0) | 零假设下一致; court 还允许非零 mean |
| B9 | N ∈ (1, ~1.29) 的 EVT 下溢 | spencer `n_trials` 是 int, 走不到这段 | `max_z` 钳到 ≥ 0, 避免 hurdle 低于 mean | court 为 `N̂=1+(M-1)(1-ρ̂)` 的浮点带做的防护; 本票整数 N≥2 网格碰不到 |
| B10 | 返回形状 | `dict` (`dsr`/`sr0`/`psr_vs_zero`/`n_trials`/`T`; PBO 另有 `oos_rank_mean`/`lambda_mean`) | `DsrResult(dsr, sr_star, z, var_factor)` / `PboResult(phi, logits, n_combinations, n_lambda_negative)` | 字段名不同, 本票只比数值 |

---

## 5. 任一侧的疑似问题

本票网格里**没有**「同一公式、同一输入、解释不了的数值分歧」。
下面几条是对照时看到的、值得各自维护者知道的点, **不是本票要修的 bug**
(两侧都冻结)。

1. **PBO 平局秩 (B5)** —— 唯一一处算法分叉。spencer 的 `≤` 计数在平局时把
   冠军抬到该组最高秩; court 用 midrank。论文 Alg. 2.3(f) 写
   `ω̄ = r̄_{n*} / (N+1)` 且假设唯一最优, 两种打破平局都是实现者补丁。
   复现: 构造 OOS Sharpe 含精确平局的矩阵, 比较 `omega` 与 `rankdata`。
   对 φ = #{λ<0}/C(S,S/2), 只有平局落在中位数附近时才会让 φ 差一个计数。

2. **spencer 静默丢尾部 (B1)** —— 调用方传入 T=260, S=8 时用了 256 行
   且不警告。court 直接拒。论文要求整除; spencer 的截断是便利而非另一条
   公式。不是算错, 但是静默。

3. **越出适用域 (B7)** —— 同一组极端 (SR=2, skew=5, kurt=3) 让
   `1 - 5·2 + (3-1)/4 · 4 = -7 < 0`。spencer 给 nan, court 抛错。
   都比「硬算复数再取实部」安全; 只是 API 契约不同。

4. **分位数实现** —— spencer 用标准库 `NormalDist.inv_cdf(1-1/N)`,
   court 用 `norm.isf(1/N)` 避开 `1-1/N` 的浮点对消 (`dsr.md` §5.5)。
   N≤1000 网格内差 ≤ 5×10⁻¹⁵; 只有 N ~ 10¹⁵ 量级才会让 spencer 的
   `1-1/N` 先糊掉。本票范围无需改任一侧。

没有发现任一侧把超额峰度当成原始峰度、或把 √T 写成 √(T-1) 的实现错误。
两边 docstring / `docs/research/dsr.md` 与代码一致。

---

## 6. PBO 三张矩阵与方向带宽

T=256, N=8, n_splits=8, C(8,4)=70。σ = 0.01。种子写死。

| 矩阵 | 种子 | 构造 | 两边 φ | 期望方向 |
|---|---|---|---|---|
| 纯噪声 | 17 | i.i.d. N(0, σ) | 35/70 = 0.5 | ≈ 0.5 |
| 真技能 | 9 | 同上, 第 0 列 += 0.1σ (每期 SR=0.1) | 0/70 = 0.0 | 显著低 |
| 选择陷阱 | 18 | 各列先 i.i.d. N(0, σ); 第 0 列按值重排, 最大 128 个进偶数块、最小 128 个进奇数块 (边际分布不变) | 65/70 ≈ 0.9286 | 显著高 |

### 带宽 (按分辨率 δ = 1/70 写死)

PBO 只能取 k/70。若 70 个组合是独立 Bernoulli(0.5),
SE = √(0.25/70) ≈ 0.0598 ≈ 4.18δ, 双边 95% 半宽 ≈ 8.2δ。
CSCV 组合共享块, 实际方差**更大**(探测约 200 个噪声种子, φ 跨度大约
0.13–0.93), 所以单次实现的「≈0.5」不能拿独立 Bernoulli 的 95% 当硬栏。

本票钉的是**固定种子的方向**, 栏宽取分辨率的整数倍:

- 噪声: \|φ − 1/2\| ≤ **4δ** = 4/70 ≈ 0.0571 (约 1 个独立 SE; 种子 17 恰为 35/70)
- 技能: φ ≤ **8δ** = 8/70 ≈ 0.1143 (「显著低于 0.5」; 种子 9 实现为 0)
- 陷阱: φ ≥ 1/2 + **14δ** = 49/70 = 0.7 (「显著高于 0.5」; 种子 18 实现为 65/70)

μ=0.1σ 在 (T=256, S=8, N=8) 下并不是「几乎必然」低 PBO —— 半样本只有
128 期, 噪声冠军的 E[max SR] 与 0.1 同量级, 约 40% 种子的技能矩阵
φ 仍可 >0.2。种子 9 是该 μ 下 φ=0 且第 0 列全样本 SR 也是冠军的一颗,
用来钉方向, 不是宣称「0.1σ 无条件打败选择偏差」。

---

## 7. 红先 (断言有判别力)

故意用错误映射跑同一网格, 1e-9 断言失败 (完整命令与退出码见 worker
receipt 的 `self_test`):

- σ/σ² 不换算: `court.expected_max_sr(0, V, N)` 对比
  `expected_max_sharpe(N, T, V)` → max\|Δ\| ≈ 0.40
- 峰度不换算(把超额峰度 kurt−3 喂给期望原始峰度的 court) →
  非正态格子 max\|Δdsr\| ≈ 3.6×10⁻³

正确映射下同一断言全绿。容差 1e-9 / 1e-12 不是空转。
