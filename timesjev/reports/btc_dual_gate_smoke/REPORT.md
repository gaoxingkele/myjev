# Crypto dual-gate 对照实验

## 设定
- 数据: Binance 日K，15 个 USDT 现货对，约 1600 根/币
- 标签: 20 日清晰涨跌（UP: 最大涨幅≥12% 且收盘≥+6% 且回撤≥-10%；DOWN 对称），剔除 chop
- 方法: 与 A 股 tabular 里程碑同构（多尺度手工特征 / 双专家+gate / 双闸门弃权）
- 切分: 单 BTC purged holdout；多币种 GroupShuffle by symbol

## 结果（生产条：selective ≥80% @ coverage ≥30%）

| 设定 | full_acc | selective | coverage | 达标 |
|------|----------|-----------|----------|------|
| BTC only purged | 48.3% | 39.1% | 76.9% | 否 |
| 15币 LogReg pool | 55.5% | 57.2% | 33.1% | 否 |
| 15币 HGB pool | 57.9% | 65.9% | 23.7% | 否 |
| HGB ultra 子集 | 56.2% | 65.8% | 30.0% | 否（acc<80%）|

## 结论
**FAIL**：同样 dual-gate 方法未能迁移到比特币/加密池并达到 A 股里程碑条。
校准集上 HGB 能摸到 ~80%@~22% 覆盖，但测试集掉到 ~66%，不具备晋升条件。
