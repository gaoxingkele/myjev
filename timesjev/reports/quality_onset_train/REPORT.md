# 优质起涨点训练成效报告

## 设定
- 正样本: Q2 ∪ Q1_20（+50% 且不破起点 / 连续上行）
- 负样本: NEG_break（4× 下采样）
- soft 消融: 加入 R20 作 medium 正样本加大训练量
- 模型: 双专家 HGB + dual-gate（与既有里程碑同构）

## 结果（5 seed 平均）

- **A股 hard Q**: full=0.758 auc=0.6143244362485083 sel=0.8280007923094601 cov=0.860 meet=5/5
- **A股 soft+R20**: full=0.610 auc=0.6359346846006744 sel=0.7749949325161628 cov=0.036 meet=0/5
- **融合 A+美股+BTC**: full=0.778 auc=0.6305791099757023 sel=0.8382220071832542 cov=0.880 meet=5/5
- **融合 A+美股**: full=0.781 auc=0.6253226557225989 sel=0.8388775569170293 cov=0.881 meet=5/5

## 样本规模
- A股 hard: {'n_pos_src': 1460, 'n_neg_kept': 5840, 'skipped': 0, 'n_final': 7300}
- A股 soft: {'n_pos_src': 1460, 'n_neg_kept': 5840, 'skipped': 0, 'n_final': 10220}
- Fuse: {'ashare': {'n_pos_src': 1460, 'n_neg_kept': 5840, 'skipped': 0, 'n_final': 7300}, 'us_pos': 17, 'us_neg': 828, 'btc_pos': 0, 'btc_neg': 30, 'n_total': 8175, 'pos_total': 1477}

## 结论
- 当前最优配置倾向 **fuse_all**（meet折数优先）；生产条仍以 selective≥80%@≥30% 为准。
- Q 标签与 R20/S20 无硬冲突（见 quality_onset_ashare/REPORT.md）；训练以 Q 为主、R20 扩量。
