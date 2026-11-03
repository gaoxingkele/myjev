# timesjev-onset-q-v1

TimesJev 自有固化权重：A股 **优质起涨点** dual-gate（Q vs NEG）。

## 指标（分组 holdout，阈值在校准集锁定）

- full_acc: 0.7542372881355932
- full_auc: 0.5998693834900731
- selective_acc: 0.8401510383889238
- coverage: 0.8416313559322034
- threshold: 0.5
- meets ≥80%@≥30%: True

## 文件

- `bundle_full.joblib`：**推荐推理**（阈值锁定后全样本重训）
- `bundle_trca.joblib`：仅 train+calib 重训（更保守）
- `manifest.json`：标签、切分、特征约定

## 推理示例

```python
import joblib, numpy as np
from pathlib import Path
# handcrafted: from timesjev scripts
bundle = joblib.load("bundle_full.joblib")
ea, eb, gate, thr = bundle["expert_a"], bundle["expert_b"], bundle["gate"], bundle["threshold"]
x = np.asarray(feats, float).reshape(1, -1)  # dim=30, left=60 bars
Xa, Xb = x[:, :15], x[:, 15:]
p_gate = gate.predict_proba(np.hstack([ea.predict_proba(Xa)[:, 1:], eb.predict_proba(Xb)[:, 1:]]))[0]
p_vote = np.mean([ea.predict_proba(Xa)[0], eb.predict_proba(Xb)[0]], axis=0)
pred_g, pred_v = int(p_gate.argmax()), int(p_vote.argmax())
conf_g, conf_v = float(p_gate.max()), float(p_vote.max())
select = pred_g == pred_v and conf_g >= thr and conf_v >= thr
# select & pred_g==1 → 优质起涨出手
```

或直接: `python scripts/quality_onset_infer.py --code 000001 --bundle models/timesjev-onset-q-v1/bundle_full.joblib`
