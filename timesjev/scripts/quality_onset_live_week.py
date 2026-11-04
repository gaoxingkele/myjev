#!/usr/bin/env python3
"""A股优质起涨点 live 周扫账本（timesjev-onset-q-v1.1）。

对 bar_cache 宇宙：用最近 LEFT 根构建价量+截面特征，dual-gate 推理。
只把 select∧pred=Q 记为出手；其余弃权。写入 runs/quality_onset_live/。
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from quality_onset_train_pipeline import LEFT, handcrafted  # noqa: E402
from quality_onset_v11_pipeline import cs_ranks, path_stats_left  # noqa: E402

BUNDLE = ROOT / "models" / "timesjev-onset-q-v1.1" / "bundle_full.joblib"
BAR_CACHE = ROOT / "runs" / "quality_onset_ashare" / "bar_cache"
OUT = ROOT / "runs" / "quality_onset_live"


def load_all_bars(cache: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for fp in sorted(cache.glob("*.parquet")):
        code = fp.name.split("_")[0]
        try:
            df = pd.read_parquet(fp).sort_values("date").drop_duplicates("date").reset_index(drop=True)
        except Exception:
            continue
        if len(df) >= LEFT:
            out[code] = df
    return out


def latest_cs_map(bars: dict[str, pd.DataFrame]) -> tuple[str, dict[str, dict]]:
    """用各股最后交易日；取出现最多的日期作为 asof，截面只在该日有数据的股票上算。"""
    from collections import Counter

    lasts = []
    for code, df in bars.items():
        lasts.append(str(df["date"].iloc[-1]))
    asof = Counter(lasts).most_common(1)[0][0]
    stats = {}
    for code, df in bars.items():
        # align to asof if present, else skip for CS
        hit = df.index[df["date"].astype(str) == asof]
        if len(hit) == 0:
            continue
        i = int(hit[0])
        st = path_stats_left(df, i)
        if st:
            stats[code] = st
    return asof, stats


def predict_row(bundle, feats: list[float]) -> dict:
    x = np.asarray(feats, float).reshape(1, -1)
    dim = bundle["feature_spec"]["dim"]
    if x.shape[1] != dim:
        raise ValueError(f"feat dim {x.shape[1]} != {dim}")
    ea, eb, gate, thr = bundle["expert_a"], bundle["expert_b"], bundle["gate"], float(bundle["threshold"])
    Xa, Xb = x[:, :15], x[:, 15:]
    p_gate = gate.predict_proba(np.hstack([ea.predict_proba(Xa)[:, 1:], eb.predict_proba(Xb)[:, 1:]]))[0]
    p_vote = np.mean([ea.predict_proba(Xa)[0], eb.predict_proba(Xb)[0]], axis=0)
    pred_g, pred_v = int(p_gate.argmax()), int(p_vote.argmax())
    conf_g, conf_v = float(p_gate.max()), float(p_vote.max())
    select = pred_g == pred_v and conf_g >= thr and conf_v >= thr
    return {
        "pred": pred_g,
        "p_up": float(p_gate[1]),
        "conf_gate": conf_g,
        "conf_vote": conf_v,
        "select": select,
        "action": "ENTER_WATCH" if (select and pred_g == 1) else "ABSTAIN",
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(BUNDLE)
    bars = load_all_bars(BAR_CACHE)
    asof, cs_stats = latest_cs_map(bars)
    print(f"asof={asof} universe={len(bars)} cs_n={len(cs_stats)} thr={bundle['threshold']}", flush=True)

    rows = []
    for code, df in bars.items():
        if code not in cs_stats:
            continue
        i = int(df.index[df["date"].astype(str) == asof][0])
        sl = slice(i - LEFT + 1, i + 1)
        closes = df["close"].to_numpy(float)[sl]
        vols = df["vol"].to_numpy(float)[sl] if "vol" in df.columns else np.ones(LEFT)
        feats = handcrafted(closes, vols) + cs_ranks(cs_stats, code)
        try:
            pred = predict_row(bundle, feats)
        except Exception as e:
            print("skip", code, e, flush=True)
            continue
        st = cs_stats[code]
        rows.append(
            {
                "code": code,
                "asof": asof,
                "close": float(df.loc[i, "close"]),
                "ret5": st["ret5"],
                "ret20": st["ret20"],
                **pred,
            }
        )

    df = pd.DataFrame(rows).sort_values(["action", "p_up"], ascending=[True, False])
    stamp = asof.replace("-", "")
    csv_path = OUT / f"live_onset_{stamp}.csv"
    json_path = OUT / f"live_onset_{stamp}.json"
    md_path = OUT / f"live_onset_{stamp}.md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    enter = df[df["action"] == "ENTER_WATCH"]
    abstain = df[df["action"] == "ABSTAIN"]
    soft = df[(df["pred"] == 1) & (df["p_up"] >= 0.65)].sort_values("p_up", ascending=False)
    report = {
        "version": bundle.get("version"),
        "asof": asof,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_scanned": int(len(df)),
        "n_enter": int(len(enter)),
        "n_abstain": int(len(abstain)),
        "n_soft_watch": int(len(soft)),
        "threshold": float(bundle["threshold"]),
        "enter_codes": enter["code"].tolist(),
        "soft_watch": soft.head(15)[["code", "p_up", "conf_gate", "conf_vote", "select"]].to_dict(orient="records"),
        "top_abstain_by_p": abstain.head(10)[["code", "p_up", "conf_gate"]].to_dict(orient="records"),
        "note": "ENTER_WATCH=dual-gate通过; soft_watch=仅gate偏Q供观察，不计入正式出手",
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# Live 优质起涨扫描 {asof}",
        "",
        f"- 模型: `{bundle.get('version')}` thr={bundle['threshold']}",
        f"- 扫描 {report['n_scanned']} | 出手 {report['n_enter']} | 弃权 {report['n_abstain']}",
        "",
        "## ENTER_WATCH",
        "",
    ]
    if len(enter) == 0:
        lines.append("（无）")
    else:
        lines.append("| code | close | ret5 | ret20 | p_up | conf |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, r in enter.iterrows():
            lines.append(
                f"| {r['code']} | {r['close']:.3f} | {r['ret5']*100:.1f}% | {r['ret20']*100:.1f}% | {r['p_up']:.3f} | {r['conf_gate']:.3f} |"
            )
    lines += ["", "## soft_watch（非正式出手）", ""]
    if len(soft) == 0:
        lines.append("（无）")
    else:
        lines.append("| code | p_up | conf_gate | conf_vote | select |")
        lines.append("|---|---:|---:|---:|---|")
        for _, r in soft.head(15).iterrows():
            lines.append(
                f"| {r['code']} | {r['p_up']:.3f} | {r['conf_gate']:.3f} | {r['conf_vote']:.3f} | {r['select']} |"
            )
    lines += ["", "## 说明", "", "- 仅 select∧Q 正式出手；soft_watch 只观察。", "- 后续跟踪 hit_25/hit_50 写入 ledger.csv。", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    # append ledger
    ledger = OUT / "ledger.csv"
    led_rows = []
    for _, r in enter.iterrows():
        led_rows.append(
            {
                "asof": asof,
                "code": r["code"],
                "action": "ENTER_WATCH",
                "close": r["close"],
                "p_up": r["p_up"],
                "version": bundle.get("version"),
                "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "hit_25": "",
                "hit_50": "",
                "notes": "",
            }
        )
    led_df = pd.DataFrame(led_rows)
    if ledger.exists():
        old = pd.read_csv(ledger)
        # avoid dup same asof+code
        key = set(zip(old["asof"].astype(str), old["code"].astype(str)))
        led_df = led_df[~led_df.apply(lambda r: (str(r["asof"]), str(r["code"])) in key, axis=1)]
        if len(led_df):
            pd.concat([old, led_df], ignore_index=True).to_csv(ledger, index=False, encoding="utf-8-sig")
    elif len(led_df):
        led_df.to_csv(ledger, index=False, encoding="utf-8-sig")
    else:
        # ensure ledger exists
        pd.DataFrame(
            columns=["asof", "code", "action", "close", "p_up", "version", "recorded_at", "hit_25", "hit_50", "notes"]
        ).to_csv(ledger, index=False, encoding="utf-8-sig")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("saved", csv_path, md_path, ledger)


if __name__ == "__main__":
    main()
