#!/usr/bin/env python3
"""固化 TimesJev 自有版本权重：A股优质起涨点 dual-gate（Q vs NEG）。

版本目录（默认同写两处）:
  D:/aicoding/TimesJev/models/timesjev-onset-q-v1/
  F:/aicoding/myjev/models/timesjev-onset-q-v1/

产物:
  bundle.joblib     - expert_a / expert_b / gate / threshold / meta
  manifest.json     - 版本、标签、切分指标、特征约定
  README.md         - 用法
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from quality_onset_train_pipeline import (  # noqa: E402
    ASHARE,
    LEFT,
    NEG_RATIO,
    SEED,
    TARGET_ACC,
    TARGET_COV,
    apply_dual,
    build_xy_from_onset,
    dual_search,
    handcrafted,
    make_clf,
)

VERSION = "timesjev-onset-q-v1"
OUT_DIRS = [
    ROOT / "models" / VERSION,
    Path(r"F:/aicoding/myjev/models") / VERSION,
]


def pack_proba(ea, eb, gate, Xa, Xb):
    pa = gate.predict_proba(np.hstack([ea.predict_proba(Xa)[:, 1:], eb.predict_proba(Xb)[:, 1:]]))
    pb = np.mean([ea.predict_proba(Xa), eb.predict_proba(Xb)], axis=0)
    return pa, pb


def fit_bundle(Xa, Xb, y):
    ea, eb, gate = make_clf(), make_clf(), make_clf()
    ea.fit(Xa, y)
    eb.fit(Xb, y)
    gate.fit(np.hstack([ea.predict_proba(Xa)[:, 1:], eb.predict_proba(Xb)[:, 1:]]), y)
    return ea, eb, gate


def main():
    onset = pd.read_parquet(ASHARE / "onset_candidates.parquet")
    cache = ASHARE / "bar_cache"
    X, y, codes, dates, tiers, meta = build_xy_from_onset(onset, cache, mode="hard")
    assert len(X) > 100, "not enough samples"
    Xa, Xb = X[:, :15], X[:, 15:]

    # 诚实 holdout：按股票分组
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    trval, te = next(gss1.split(X, y, codes))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 4)
    tri, cai = next(gss2.split(X[trval], y[trval], codes[trval]))
    tr, ca = trval[tri], trval[cai]

    ea0, eb0, gate0 = fit_bundle(Xa[tr], Xb[tr], y[tr])
    hit80 = dual_search(y[ca], *pack_proba(ea0, eb0, gate0, Xa[ca], Xb[ca]), 0.80, 20)
    hit75 = dual_search(y[ca], *pack_proba(ea0, eb0, gate0, Xa[ca], Xb[ca]), 0.75, 20)
    thr = (hit80 or hit75 or {"threshold": 0.60})["threshold"]
    holdout = apply_dual(y[te], *pack_proba(ea0, eb0, gate0, Xa[te], Xb[te]), thr)

    # 生产权重：用 train+calib 重训（不看 test），阈值锁定
    trca = np.concatenate([tr, ca])
    ea, eb, gate = fit_bundle(Xa[trca], Xb[trca], y[trca])

    # 再在全量（含 test）上可选最终重训 —— 对「落地生产」更常见；holdout 指标仍用上面冻结报告
    ea_full, eb_full, gate_full = fit_bundle(Xa, Xb, y)

    feature_spec = {
        "left_bars": LEFT,
        "scales": [20, 40, 60],
        "per_scale": [
            "ret1",
            "ret5",
            "ret10",
            "pos_in_range",
            "ret_std",
            "vol_ratio5",
            "path_max",
            "path_min",
            "mean_ret5",
            "close_vs_mean",
        ],
        "dim": int(X.shape[1]),
        "expert_a": "feats[0:15]",
        "expert_b": "feats[15:]",
        "builder": "quality_onset_train_pipeline.handcrafted",
    }

    manifest = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task": "A-share quality onset: Q1_20∪Q2 vs NEG_break",
        "labels": {
            "positive": "Q2 (60d +50% no-break) ∪ Q1_20 (soft consecutive +50%)",
            "negative": "NEG_break (early break + weak thrust)",
            "neg_ratio": NEG_RATIO,
        },
        "sample_meta": meta,
        "n_total": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "n_codes": int(len(set(codes.tolist()))),
        "split": {
            "seed": SEED,
            "n_train": int(len(tr)),
            "n_calib": int(len(ca)),
            "n_test": int(len(te)),
            "test_codes": sorted(set(codes[te].tolist())),
        },
        "threshold": float(thr),
        "calib_dual80": hit80,
        "calib_dual75": hit75,
        "holdout_test": holdout,
        "meets_prod_bar": bool(holdout.get("meets")),
        "feature_spec": feature_spec,
        "files": {
            "bundle_holdout_refit": "bundle_trca.joblib",
            "bundle_full": "bundle_full.joblib",
            "recommend": "bundle_full.joblib",
        },
        "inference": {
            "select_if": "gate_pred == vote_pred AND gate_conf >= thr AND vote_conf >= thr",
            "class": {0: "NEG / not quality onset", 1: "quality onset (Q)"},
        },
    }

    payload_trca = {
        "version": VERSION,
        "expert_a": ea,
        "expert_b": eb,
        "gate": gate,
        "threshold": float(thr),
        "feature_spec": feature_spec,
        "handcrafted_note": "import handcrafted from quality_onset_train_pipeline or duplicate",
        "fit_on": "train+calib",
        "holdout_test": holdout,
    }
    payload_full = {
        "version": VERSION,
        "expert_a": ea_full,
        "expert_b": eb_full,
        "gate": gate_full,
        "threshold": float(thr),
        "feature_spec": feature_spec,
        "fit_on": "all_samples_after_locking_threshold",
        "holdout_test": holdout,
    }

    readme = f"""# {VERSION}

TimesJev 自有固化权重：A股 **优质起涨点** dual-gate（Q vs NEG）。

## 指标（分组 holdout，阈值在校准集锁定）

- full_acc: {holdout.get('full_acc')}
- full_auc: {holdout.get('full_auc')}
- selective_acc: {holdout.get('acc')}
- coverage: {holdout.get('coverage')}
- threshold: {thr}
- meets ≥80%@≥30%: {holdout.get('meets')}

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

或直接: `python scripts/quality_onset_infer.py --code 000001 --bundle models/{VERSION}/bundle_full.joblib`
"""

    for out in OUT_DIRS:
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload_trca, out / "bundle_trca.joblib", compress=3)
        joblib.dump(payload_full, out / "bundle_full.joblib", compress=3)
        # convenience alias
        shutil.copyfile(out / "bundle_full.joblib", out / "bundle.joblib")
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "README.md").write_text(readme, encoding="utf-8")
        print("wrote", out)

    print(json.dumps({"holdout": holdout, "threshold": thr, "meta": meta}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
