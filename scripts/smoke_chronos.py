"""Zero-shot Chronos-Bolt tiny smoke test from local weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parents[1] / "models" / "chronos-bolt-tiny"),
    )
    args = ap.parse_args()

    from chronos import BaseChronosPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = BaseChronosPipeline.from_pretrained(args.model, device_map=device)
    context = torch.tensor(
        [100, 101.2, 100.8, 102.5, 103.1, 102.7, 104.0, 105.2, 104.8, 106.1],
        dtype=torch.float32,
    )
    forecast = pipe.predict(context, prediction_length=5)
    # Chronos-Bolt returns quantiles: [num_series, num_quantiles, horizon]
    arr = forecast[0].detach().cpu().float()
    mid = arr.shape[0] // 2
    point = arr[mid].tolist()
    out = {
        "model": args.model,
        "device": device,
        "shape": list(arr.shape),
        "point_forecast_mid_quantile": [round(x, 4) for x in point],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
