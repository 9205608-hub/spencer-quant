# Spencer 框架 PIT 终跑报告 (2026-07-23 12:09)

## 数据与宇宙
- 全市场(含退市): 5791 只 × 2562 交易日, 2016-01-04 → 2026-07-22
- PIT 宇宙: 月末在市名单 asof + 预热120日 + 剔ST + 流动性前1500; 日均成员 1430 只
- 幸存者偏差: **已消除**(名单含后来退市股); 行业仍为快照(30问#31)

- 风格集: 6 风格(含 BTOP) + 行业哑变量

## 单因子三档读数(PIT 宇宙)
```
                       factor  ic_mean  ic_ir_daily  t_stat_conservative  t_stat_nw  yearly_all_positive  rank_autocorr_5d  ls_net_ann  ls_net_sharpe
            amihud_20@raw_pit   0.0586        0.332                 7.32       8.54                 True            0.9868         NaN            NaN
      amihud_20@size_neut_pit   0.0315        0.431                 9.51      10.91                 True            0.9390         NaN            NaN
      amihud_20@full_neut_pit   0.0008        0.013                 0.28       0.34                False            0.8615      0.0336           0.54
             chip_age@raw_pit   0.0838        0.345                 7.63       8.98                 True            0.9977         NaN            NaN
       chip_age@size_neut_pit   0.0189        0.125                 2.77       3.22                False            0.9916         NaN            NaN
       chip_age@full_neut_pit   0.0145        0.232                 4.94       5.77                False            0.9666      0.0488           0.66
             mom_20_5@raw_pit   0.0251        0.139                 3.08       3.73                False            0.6916         NaN            NaN
       mom_20_5@size_neut_pit   0.0245        0.152                 3.36       4.10                False            0.6898         NaN            NaN
       mom_20_5@full_neut_pit   0.0161        0.152                 3.25       3.86                False            0.6676      0.0502           0.48
           px_pos_250@raw_pit   0.0002        0.001                 0.03       0.03                False            0.9292         NaN            NaN
     px_pos_250@size_neut_pit   0.0080        0.046                 1.01       1.17                False            0.9311         NaN            NaN
     px_pos_250@full_neut_pit   0.0052        0.046                 0.97       1.12                False            0.8951     -0.0349          -0.30
                rev_5@raw_pit   0.0158        0.094                 2.08       2.84                False           -0.0198         NaN            NaN
          rev_5@size_neut_pit   0.0183        0.126                 2.78       3.83                False           -0.0276         NaN            NaN
          rev_5@full_neut_pit   0.0291        0.303                 6.44       8.47                False           -0.0320      0.1569           1.51
      turn_surge_5_60@raw_pit   0.0093        0.065                 1.44       1.75                False            0.6439         NaN            NaN
turn_surge_5_60@size_neut_pit   0.0197        0.157                 3.46       4.25                False            0.6311         NaN            NaN
turn_surge_5_60@full_neut_pit   0.0187        0.226                 4.82       5.95                False            0.5864      0.0579           0.68
               vol_20@raw_pit   0.0848        0.389                 8.59      10.14                 True            0.9406         NaN            NaN
         vol_20@size_neut_pit   0.0363        0.235                 5.19       6.14                 True            0.8759         NaN            NaN
         vol_20@full_neut_pit   0.0094        0.120                 2.56       3.10                False            0.7531     -0.0098          -0.12
```

## 合成与组合(PIT 宇宙, 一字板过滤, 成本后)
- comp_eq_pit: IC 0.0263, NW-t 5.96, 多空净年化 8.2%, sharpe 0.75
- top50/buffer80 多头净: 年化 6.1% (sharpe 0.25) vs 宇宙等权基准 1.0% (sharpe 0.04) → 超额 5.4% (sharpe 0.6)
- 年化单边换手 15.4x
- 执行接口: 2026-07-22 目标持仓 50 只 → target_holdings_pit.csv

## 统计纪律(对最好读数动刀)
- comp_eq 多空净 日频Sharpe 0.0474 (T=2271), 台账 N=60 校正后 DSR: {'dsr': 0.4658564788231254, 'sr0': 0.04921492385498662, 'psr_vs_zero': 0.988021389748095, 'n_trials': 60, 'T': 2271, 'n_trials_from_ledger': 60}
- 7因子多空收益矩阵 PBO(CSCV): {'pbo': 0.06713286713286713, 'n_combinations': 12870, 'n_strategies': 7, 'n_periods_used': 2256, 'n_splits': 16, 'oos_rank_mean': 0.8188519813519813, 'lambda_mean': 1.6468380213091716}

## 诚实声明
- 台账累计 N = 60; 本报告全部读数已入台账
- 行业为快照口径; 成本为研究级近似; model_gb 已证伪结案不再上桌