#!/usr/bin/env python3
"""Decider-2B 上涨/下跌定版流水线（P0–P4）。

阶段:
  splits   - 固化 Group 切分 + 表格双闸门 baseline
  samples  - 同切分写出 TimesJev Sample jsonl + selection
  train_a  - freeze_lm value-encoder 对照
  train_b  - 解冻小 LoRA + value-encoder
  eval     - 校准集搜双闸门阈值，测试集报告 selective ≥80%@≥30%
  promote  - 达标晋升 prod，否则 archive 并写停用说明

用法:
  python scripts/updown_prod_pipeline.py --stage all
  python scripts/updown_prod_pipeline.py --stage splits
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from timesjev.schema import Sample
from timesjev.train import sft as sft_mod

DECIDER = Path(r"F:/aicoding/myjev/models/decider-2b")
PROD_DIR = Path(r"F:/aicoding/myjev/models/decider-2b-updown-prod")
CACHE = ROOT / "runs" / "smoke_updown_multimodal_v2" / "events_cache.pkl"
WORK = ROOT / "runs" / "updown_prod"
SEL_KEY = "updown-prod-mm-L100"
SEED = 7
LEFT_N = 100
TARGET_ACC = 0.80
TARGET_COV = 0.30
MIN_N = 20


@dataclass
class Event:
    code: str
    date: str
    label: int
    left_n: int
    close: list[float]
    vol: list[float]
    macd_h: list[float]
    rsi: list[float]
    ma5: list[float]
    ma20: list[float]


def load_events() -> list[Event]:
    raw = pickle.loads(CACHE.read_bytes())
    rows = raw[100] if 100 in raw else raw["100"]
    return [Event(**e) if isinstance(e, dict) else e for e in rows]


def _tail(a, n):
    x = np.asarray(a, float)
    return x[-n:] if len(x) >= n else x


def handcrafted(ev: Event) -> list[float]:
    feats = []
    for n in (20, 50, 100):
        c = _tail(ev.close, n)
        v = _tail(ev.vol, n)
        m = _tail(ev.macd_h, n)
        r = _tail(ev.rsi, n)
        ma5 = _tail(ev.ma5, n)
        ma20 = _tail(ev.ma20, n)

        def ret(k):
            if len(c) <= k or c[-k - 1] == 0:
                return 0.0
            return float(c[-1] / c[-k - 1] - 1)

        rng = float(c.max() - c.min()) + 1e-9
        rets = np.diff(c) / np.clip(c[:-1], 1e-9, None)
        feats += [
            ret(1),
            ret(min(5, len(c) - 1)),
            float((c[-1] - c.min()) / rng),
            float(np.std(rets)) if len(rets) else 0.0,
            float(np.mean(v[-min(5, len(v)) :]) / (np.median(v) + 1e-9)),
            float(ma5[-1] / (ma20[-1] + 1e-9) - 1) if len(ma5) and len(ma20) else 0.0,
            float(r[-1]) / 100.0,
            float(m[-1]),
            float(c.max() / c[0] - 1),
            float(c.min() / c[0] - 1),
        ]
    return feats


def write_selection():
    names = []
    for ch in ("X", "Y", "Z"):
        names += [f"{ch} std", f"{ch} range", f"{ch} slope", f"{ch} acf lag1", f"{ch} zero-crossing rate"]
    names += [
        "X-Y xcorr lag0", "X-Y best corr", "X-Z xcorr lag0", "X-Z best corr",
        "Y-Z xcorr lag0", "Y-Z best corr",
        "X high_fluctuation", "Y high_fluctuation", "X periodicity", "Z periodicity",
    ]
    path = ROOT / "configs" / "selections" / f"{SEL_KEY}.json"
    path.write_text(json.dumps({"names": names, "method": "updown prod"}, indent=2), encoding="utf-8")
    return SEL_KEY


def to_sample(ev: Event) -> Sample:
    inter = []
    for a, b, c in zip(ev.close, ev.vol, ev.macd_h):
        inter.extend([a, b, c])
    c0, c1 = ev.close[0], ev.close[-1]
    ctx = (
        f"Stock {ev.code} multimodal L={ev.left_n}. "
        f"Close {c0:.4g}->{c1:.4g} ({(c1 / c0 - 1) * 100:.2f}%). "
        f"RSI={ev.rsi[-1]:.1f} MA5/20={ev.ma5[-1]:.4g}/{ev.ma20[-1]:.4g}."
    )
    return Sample(
        id=f"L{ev.left_n}/{ev.code}/{ev.date}/{ev.label}",
        task="tsqa_classification",
        split="train",
        context=ctx,
        question="Will the next ~20 sessions be a clear UP move (not chop), or a clear DOWN move?",
        options={"A": "Yes — clear UP", "B": "No — clear DOWN"},
        answer="A" if ev.label == 1 else "B",
        series=[inter],
        domain="stock",
        source="updown_prod",
        raw_problem="updown",
    )


def make_clf():
    return CalibratedClassifierCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.7)),
        method="sigmoid",
        cv=3,
    )


def dual_search(y, p_a, p_b, target=0.80, min_n=MIN_N):
    best = None
    for thr in np.linspace(0.50, 0.90, 41):
        pa, pb = p_a.argmax(1), p_b.argmax(1)
        m = (pa == pb) & (p_a.max(1) >= thr) & (p_b.max(1) >= thr)
        n = int(m.sum())
        if n < min_n:
            continue
        acc = float(accuracy_score(y[m], pa[m]))
        if acc >= target:
            cov = float(m.mean())
            if best is None or cov > best["coverage"]:
                best = {"threshold": float(thr), "acc": acc, "n": n, "coverage": cov, "target": target}
    return best


def stage_splits():
    WORK.mkdir(parents=True, exist_ok=True)
    events = load_events()
    assert all(e.left_n == LEFT_N for e in events)
    y = np.asarray([e.label for e in events])
    codes = np.asarray([e.code for e in events])
    X = np.asarray([handcrafted(e) for e in events], float)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    trval, te = next(gss1.split(X, y, codes))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 4)
    tr_i, ca_i = next(gss2.split(X[trval], y[trval], codes[trval]))
    tr, ca = trval[tr_i], trval[ca_i]

    # tabular experts: price-ish halves
    X_a, X_b = X[:, :15], X[:, 15:]
    ea, eb = make_clf(), make_clf()
    ea.fit(X_a[tr], y[tr])
    eb.fit(X_b[tr], y[tr])
    meta_tr = np.hstack([ea.predict_proba(X_a[tr])[:, 1:], eb.predict_proba(X_b[tr])[:, 1:]])
    gate = make_clf()
    gate.fit(meta_tr, y[tr])

    def pack(idx):
        pa = gate.predict_proba(np.hstack([ea.predict_proba(X_a[idx])[:, 1:], eb.predict_proba(X_b[idx])[:, 1:]]))
        pb = np.mean([ea.predict_proba(X_a[idx]), eb.predict_proba(X_b[idx])], axis=0)
        return pa, pb

    p_gate_ca, p_vote_ca = pack(ca)
    p_gate_te, p_vote_te = pack(te)
    # calibrate thr on calib, report on test
    thr_pack = dual_search(y[ca], p_gate_ca, p_vote_ca, TARGET_ACC, MIN_N)
    thr = thr_pack["threshold"] if thr_pack else 0.60
    pa, pb = p_gate_te.argmax(1), p_vote_te.argmax(1)
    m = (pa == pb) & (p_gate_te.max(1) >= thr) & (p_vote_te.max(1) >= thr)
    baseline = {
        "full_acc_gate": float(accuracy_score(y[te], p_gate_te.argmax(1))),
        "full_auc_gate": float(roc_auc_score(y[te], p_gate_te[:, 1])),
        "calib_dual80": thr_pack,
        "test_dual": {
            "threshold": thr,
            "acc": float(accuracy_score(y[te][m], pa[m])) if m.any() else None,
            "n": int(m.sum()),
            "coverage": float(m.mean()),
            "meets_prod_bar": bool(m.sum() >= MIN_N and accuracy_score(y[te][m], pa[m]) >= TARGET_ACC and m.mean() >= TARGET_COV)
            if m.any()
            else False,
        },
        "n_train": int(len(tr)),
        "n_calib": int(len(ca)),
        "n_test": int(len(te)),
        "n_train_codes": int(len(set(codes[tr]))),
        "n_calib_codes": int(len(set(codes[ca]))),
        "n_test_codes": int(len(set(codes[te]))),
        "code_leak_train_test": int(len(set(codes[tr]) & set(codes[te]))),
    }

    ids = [f"L{e.left_n}/{e.code}/{e.date}/{e.label}" for e in events]
    manifest = {
        "seed": SEED,
        "left_n": LEFT_N,
        "target_acc": TARGET_ACC,
        "target_cov": TARGET_COV,
        "ids": ids,
        "labels": y.tolist(),
        "codes": codes.tolist(),
        "dates": [e.date for e in events],
        "train_idx": tr.tolist(),
        "calib_idx": ca.tolist(),
        "test_idx": te.tolist(),
        "tabular_baseline": baseline,
    }
    (WORK / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (WORK / "baseline.json").write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[splits]", json.dumps(baseline, ensure_ascii=False, indent=2))
    return manifest


def stage_samples():
    events = load_events()
    manifest = json.loads((WORK / "split_manifest.json").read_text(encoding="utf-8"))
    write_selection()
    samples = [to_sample(e) for e in events]
    by_id = {s.id: s for s in samples}
    # ensure ids match
    for i, sid in enumerate(manifest["ids"]):
        assert sid in by_id, sid
        samples[i] = by_id[sid]
        samples[i].split = (
            "train" if i in set(manifest["train_idx"]) else ("calib" if i in set(manifest["calib_idx"]) else "test")
        )

    def dump(name, idxs):
        path = WORK / f"samples_{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for i in idxs:
                s = samples[i]
                f.write(json.dumps(s.__dict__, ensure_ascii=False) + "\n")
        return path

    paths = {
        "train": str(dump("train", manifest["train_idx"])),
        "calib": str(dump("calib", manifest["calib_idx"])),
        "test": str(dump("test", manifest["test_idx"])),
        "selection": SEL_KEY,
        "n": {k: len(manifest[f"{k}_idx"]) for k in ("train", "calib", "test")},
    }
    (WORK / "samples_index.json").write_text(json.dumps(paths, indent=2), encoding="utf-8")
    # also pickle samples list for fast FT monkeypatch
    with open(WORK / "samples_all.pkl", "wb") as f:
        pickle.dump(samples, f)
    print("[samples]", paths)
    return samples, manifest


def _patch_loaders(train_ids, eval_ids, all_by_id):
    def load_samples(data_cfg: dict):
        split = data_cfg.get("split", "train")
        ids = train_ids if split == "train" else eval_ids
        # allow calib via split name
        if split == "calib":
            ids = eval_ids
        return [all_by_id[i] for i in ids]

    import timesjev.data as data_mod
    import timesjev.train.sft as sft_file
    from timesjev.eval import runner as runner_mod

    data_mod.load_samples = load_samples
    sft_file.load_samples = load_samples
    runner_mod.load_samples = load_samples
    return load_samples


def stage_train(tag: str, freeze_lm: bool, epochs: int = 4, n_samples: int = 400, lr: float = 2e-4):
    assert DECIDER.is_dir(), DECIDER
    if not (WORK / "samples_all.pkl").is_file() or not (WORK / "split_manifest.json").is_file():
        stage_samples()
    samples = pickle.loads((WORK / "samples_all.pkl").read_bytes())
    manifest = json.loads((WORK / "split_manifest.json").read_text(encoding="utf-8"))

    write_selection()
    all_by_id = {s.id: s for s in samples}
    train_ids = [manifest["ids"][i] for i in manifest["train_idx"]]
    # monitor on calib during train
    calib_ids = [manifest["ids"][i] for i in manifest["calib_idx"]]
    _patch_loaders(set(train_ids), set(calib_ids), all_by_id)

    state_kind = f"selv_noarr:{SEL_KEY}"
    cfg = {
        "name": f"updown-prod-{tag}",
        "model": {
            "backend": "decider",
            "checkpoint": str(DECIDER).replace("\\", "/"),
            "revision": None,
            "temperature": None,
            "use_graphs": False,
        },
        "data": {"source": "stock", "split": "train", "state": state_kind},
        "train": {
            "method": "sft",
            "sampling": "balanced_random",
            "n_samples": min(n_samples, len(train_ids)),
            "positive_prefix": "Yes",
            "seed": SEED,
            "state": state_kind,
            "freeze_lm": freeze_lm,
            "batch_like": "raw",
            "max_ctx_tokens": 4096,
            "monitor": "test-updown",
            "lora": {"r": 8, "alpha": 16, "dropout": 0.05 if not freeze_lm else 0.0, "target_modules": None},
            "optim": {
                "lr": lr if freeze_lm else 5e-5,
                "epochs": epochs,
                "max_tokens_per_batch": 8192,
                "warmup_ratio": 0.05,
                "weight_decay": 0.0,
            },
            "ts_input": {"value_types": True, "value_block": f"select:{SEL_KEY}", "lr": 1.0e-3},
        },
        "eval": [{"suffix": "test-updown", "data": {"source": "stock", "split": "test", "state": state_kind}}],
    }
    # monitor uses split=test in eval entry but we patched eval_ids=calib
    cfg["eval"][0]["data"]["split"] = "test"

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ckpt = sft_mod.train_sft(cfg)
    meta = {
        "tag": tag,
        "freeze_lm": freeze_lm,
        "checkpoint": str(ckpt),
        "train_seconds": round(time.time() - t0, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0,
        "state": state_kind,
        "n_train_used": cfg["train"]["n_samples"],
        "epochs": epochs,
    }
    # read last monitor
    logp = Path(ckpt).parent / "train.log"
    monitor_acc = None
    if logp.is_file():
        for line in logp.read_text(encoding="utf-8").splitlines():
            if "test-updown accuracy" in line:
                try:
                    monitor_acc = float(line.strip().split()[-1]) / 100.0
                except Exception:
                    pass
    meta["monitor_acc_calib"] = monitor_acc
    (WORK / f"train_{tag}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[train]", meta)
    return meta


def _predict_decider_ts(ckpt: str, state_kind: str, samples: list[Sample]):
    from timesjev.models import build_model

    model = build_model(
        {
            "backend": "decider_ts",
            "checkpoint": ckpt,
            "state": state_kind,
            "revision": None,
            "temperature": None,
            "use_graphs": False,
        }
    )
    return model.predict(samples)


def stage_eval(tag: str):
    meta = json.loads((WORK / f"train_{tag}.json").read_text(encoding="utf-8"))
    manifest = json.loads((WORK / "split_manifest.json").read_text(encoding="utf-8"))
    samples = pickle.loads((WORK / "samples_all.pkl").read_bytes())
    by_id = {s.id: s for s in samples}
    state = meta["state"]
    ckpt = meta["checkpoint"]

    calib = [by_id[manifest["ids"][i]] for i in manifest["calib_idx"]]
    test = [by_id[manifest["ids"][i]] for i in manifest["test_idx"]]
    # also need a second vote: tabular on same events
    events = load_events()
    X = np.asarray([handcrafted(e) for e in events], float)
    y = np.asarray([e.label for e in events])
    tr = np.asarray(manifest["train_idx"])
    ca = np.asarray(manifest["calib_idx"])
    te = np.asarray(manifest["test_idx"])
    X_a, X_b = X[:, :15], X[:, 15:]
    ea, eb = make_clf(), make_clf()
    ea.fit(X_a[tr], y[tr])
    eb.fit(X_b[tr], y[tr])
    meta_tr = np.hstack([ea.predict_proba(X_a[tr])[:, 1:], eb.predict_proba(X_b[tr])[:, 1:]])
    gate = make_clf()
    gate.fit(meta_tr, y[tr])
    p_tab_ca = gate.predict_proba(np.hstack([ea.predict_proba(X_a[ca])[:, 1:], eb.predict_proba(X_b[ca])[:, 1:]]))
    p_tab_te = gate.predict_proba(np.hstack([ea.predict_proba(X_a[te])[:, 1:], eb.predict_proba(X_b[te])[:, 1:]]))

    print(f"[eval] predicting calib n={len(calib)} ...", flush=True)
    preds_ca = _predict_decider_ts(ckpt, state, calib)
    print(f"[eval] predicting test n={len(test)} ...", flush=True)
    preds_te = _predict_decider_ts(ckpt, state, test)

    def to_proba(preds, samples_):
        # columns [P(DOWN), P(UP)] aligned to label 0/1
        P = np.zeros((len(samples_), 2), float)
        ytrue = np.asarray([1 if s.answer == "A" else 0 for s in samples_])
        for i, (p, s) in enumerate(zip(preds, samples_)):
            probs = p.probs or {}
            # A=UP, B=DOWN
            pu = float(probs.get("A", 0.5))
            pd_ = float(probs.get("B", 1 - pu))
            # normalize
            z = pu + pd_ + 1e-9
            P[i, 1] = pu / z
            P[i, 0] = pd_ / z
        return ytrue, P

    y_ca, p_dec_ca = to_proba(preds_ca, calib)
    y_te, p_dec_te = to_proba(preds_te, test)

    # dual: Decider ∧ tabular gate
    thr_pack = dual_search(y_ca, p_dec_ca, p_tab_ca, TARGET_ACC, MIN_N)
    thr85 = dual_search(y_ca, p_dec_ca, p_tab_ca, 0.85, max(12, MIN_N // 2))
    thr = thr_pack["threshold"] if thr_pack else 0.65
    pa, pb = p_dec_te.argmax(1), p_tab_te.argmax(1)
    m = (pa == pb) & (p_dec_te.max(1) >= thr) & (p_tab_te.max(1) >= thr)
    full_acc = float(accuracy_score(y_te, p_dec_te.argmax(1)))
    sel = {
        "threshold": thr,
        "acc": float(accuracy_score(y_te[m], pa[m])) if m.any() else None,
        "n": int(m.sum()),
        "coverage": float(m.mean()),
    }
    meets = bool(
        sel["n"] >= MIN_N and sel["acc"] is not None and sel["acc"] >= TARGET_ACC and sel["coverage"] >= TARGET_COV
    )
    report = {
        "tag": tag,
        "checkpoint": ckpt,
        "full_acc_decider": full_acc,
        "full_auc_decider": float(roc_auc_score(y_te, p_dec_te[:, 1])),
        "calib_dual80": thr_pack,
        "calib_dual85": thr85,
        "test_dual_decider_x_tabular": sel,
        "meets_prod_bar": meets,
        "bar": {"acc": TARGET_ACC, "coverage": TARGET_COV, "min_n": MIN_N},
        "n_calib": len(calib),
        "n_test": len(test),
    }
    (WORK / f"eval_{tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[eval]", json.dumps(report, ensure_ascii=False, indent=2))
    return report


def stage_promote():
    baseline = json.loads((WORK / "baseline.json").read_text(encoding="utf-8"))
    evals = {}
    for tag in ("A_freeze", "B_lora"):
        p = WORK / f"eval_{tag}.json"
        if p.is_file():
            evals[tag] = json.loads(p.read_text(encoding="utf-8"))

    # pick best meeting bar; else best selective acc with coverage>=0.2
    candidates = []
    for tag, r in evals.items():
        sel = r.get("test_dual_decider_x_tabular") or {}
        candidates.append(
            {
                "tag": tag,
                "meets": r.get("meets_prod_bar"),
                "acc": sel.get("acc") or 0,
                "coverage": sel.get("coverage") or 0,
                "n": sel.get("n") or 0,
                "full_acc": r.get("full_acc_decider"),
                "checkpoint": r.get("checkpoint"),
                "threshold": sel.get("threshold"),
                "report": r,
            }
        )
    winners = [c for c in candidates if c["meets"]]
    if winners:
        win = max(winners, key=lambda c: (c["coverage"], c["acc"]))
        decision = "promote"
    else:
        win = max(candidates, key=lambda c: (c["acc"], c["coverage"])) if candidates else None
        decision = "archive"

    out = {
        "decision": decision,
        "winner": win,
        "tabular_baseline": baseline,
        "evals": evals,
        "prod_dir": str(PROD_DIR),
        "criteria": {"acc>=": TARGET_ACC, "coverage>=": TARGET_COV},
    }

    if decision == "promote" and win:
        if PROD_DIR.exists():
            shutil.rmtree(PROD_DIR)
        PROD_DIR.mkdir(parents=True)
        src = Path(win["checkpoint"])
        # copy checkpoint tree
        shutil.copytree(src, PROD_DIR / "checkpoint")
        # also adapter sibling if present
        adapter = src.parent / "adapter"
        if adapter.is_dir():
            shutil.copytree(adapter, PROD_DIR / "adapter")
        promo = f"""# Decider-2B Up/Down Production Promotion

**Decision:** PROMOTE (`{win['tag']}`)
**Date:** {time.strftime('%Y-%m-%d %H:%M')}
**Checkpoint:** `{win['checkpoint']}`

## Criteria
- Selective dual-gate (Decider ∧ tabular): acc ≥ {TARGET_ACC}, coverage ≥ {TARGET_COV}

## Result
- threshold: {win['threshold']}
- selective acc: {win['acc']}
- coverage: {win['coverage']}
- n: {win['n']}
- full Decider acc: {win['full_acc']}

## Split
- See `D:/aicoding/TimesJev/runs/updown_prod/split_manifest.json` (seed={SEED})
- code_leak train∩test must be 0

## Inference
- backend: `decider_ts`
- state: `selv_noarr:{SEL_KEY}`
- dual-gate with tabular vote, threshold={win['threshold']}

## Tabular baseline (same split)
```json
{json.dumps(baseline, ensure_ascii=False, indent=2)}
```
"""
        (PROD_DIR / "PROMOTION.md").write_text(promo, encoding="utf-8")
        (PROD_DIR / "promotion.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        (WORK / "PROMOTION.md").write_text(promo, encoding="utf-8")
        print("[promote] PROMOTED ->", PROD_DIR)
    else:
        archive = WORK / "ARCHIVE_NO_PROMOTE.md"
        reason = (
            f"No Decider run met selective bar (acc≥{TARGET_ACC}, cov≥{TARGET_COV}).\n"
            f"Best candidate: {win}\n"
            f"Keep tabular dual-gate as production path (milestone v5).\n"
            f"Decider checkpoints remain experimental under runs/updown_prod/.\n"
        )
        archive.write_text("# ARCHIVE — Decider not promoted\n\n" + reason, encoding="utf-8")
        (WORK / "promotion.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[promote] ARCHIVE (not promoted)\n", reason)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        default="all",
        choices=["splits", "samples", "train_a", "train_b", "eval_a", "eval_b", "promote", "all"],
    )
    args = ap.parse_args()
    stage = args.stage
    if stage in ("splits", "all"):
        stage_splits()
    if stage in ("samples", "all"):
        stage_samples()
    if stage in ("train_a", "all"):
        stage_train("A_freeze", freeze_lm=True, epochs=4, n_samples=400, lr=2e-4)
    if stage in ("train_b", "all"):
        stage_train("B_lora", freeze_lm=False, epochs=3, n_samples=400, lr=5e-5)
    if stage in ("eval_a", "all"):
        stage_eval("A_freeze")
    if stage in ("eval_b", "all"):
        stage_eval("B_lora")
    if stage in ("promote", "all"):
        stage_promote()


if __name__ == "__main__":
    main()
