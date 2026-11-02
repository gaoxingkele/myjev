# Milestone: Up/Down Dual-Gate Selective Predictor (v5)

**Date:** 2026-11-01  
**Status:** usable for *selective* deployment（只出手高置信；不保证全市场全覆盖）  
**Code:** `scripts/smoke_updown_evolve_v5.py` · `scripts/smoke_updown_paper_v4.py`  
**Report:** `runs/smoke_updown_evolve_v5/evolve_report.json`

## What is frozen

- **Task:** clear UP vs clear DOWN over ~20 sessions; chop excluded from binary train
- **Features:** multi-scale handcrafted + TimesJev multi-channel stats + same-day cross-section ranks
- **Model:** modality experts → stacking gate + soft vote
- **Decision rule (production):** **dual-gate** = gate ∧ vote agree, then confidence threshold
- **Honesty protocol:** Group-purged CV (no same-stock leak)

## Verified numbers (purged group, 4/4 folds)

| Target | Mean acc | Mean coverage |
|--------|----------|---------------|
| ≥80%   | ~80.9%   | ~41%          |
| ≥85%   | ~86.0%   | ~27%          |

Production holdout (unseen tickers): dual ≥80% ≈ 80.5% @ 55% coverage; ≥85% ≈ 85.1% @ 39%.

## Known limits

- Walk-forward (pure time shift) is weaker → regime shifts hurt; needs regime/news/moneyflow
- Not a full-coverage oracle; abstain is first-class
- Charts aligned for VLM later; numeric path is what is production-ready now

## Next after this milestone

1. Live weekly scan on latest close (as-of 2026-10-30)
2. Multimodal: regime type + news/policy + moneyflow
