# timesjev-onset-q-v1.1

相对 v1：正样本去重、难负样本、截面特征、更严阈值。

## Holdout（thr=0.55）
- selective=0.8074639525021204 coverage=0.8681885125184094 meets=True
- natural thr=0.5: sel=0.8036758563074352 cov=0.8814432989690721

## Multi-seed
{
  "mean_full": 0.7528105169967431,
  "mean_auc": 0.5710790143851275,
  "mean_sel": 0.8158111124484254,
  "mean_cov": 0.8375623793441849,
  "mean_thr": 0.558,
  "n_meet": 5,
  "n_folds": 5,
  "feat_dim": 42
}

推理: `python scripts/quality_onset_infer.py --bundle models/timesjev-onset-q-v1.1/bundle_full.joblib --code 000001`
（v1.1 特征维=42，infer 脚本需用 v11 特征构建；见 quality_onset_live_week.py）
