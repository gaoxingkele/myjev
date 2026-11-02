"""Zero-shot stock-style System One request against a local Kev server."""
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8008")
    args = ap.parse_args()

    payload = {
        "model": "kev-latest",
        "state": {
            "ticker": "TEST",
            "timeframe": "1d",
            "summary": (
                "Close rose 8 of last 10 sessions. Price sits above 20-day and 60-day MA. "
                "Volume on up days is higher than on down days. RSI(14)=62. "
                "Nearest resistance is 3.2% above; nearest support is 4.1% below."
            ),
            "recent_closes": [100, 101.2, 100.8, 102.5, 103.1, 102.7, 104.0, 105.2, 104.8, 106.1],
        },
        "questions": {
            "trend": {
                "type": "choice",
                "instructions": "Judge the near-term price trend from the state only.",
                "criteria": {
                    "up": "Higher highs/lows or sustained upward drift is more likely",
                    "range": "Sideways oscillation around current level is more likely",
                    "down": "Lower highs/lows or sustained downward drift is more likely",
                },
            },
            "breakout_valid": {
                "type": "noul",
                "instructions": "Is a bullish continuation / breakout setup currently more plausible than a fakeout?",
            },
            "conviction": {
                "type": "score",
                "instructions": "How strong is the directional evidence in the state?",
                "criteria": ["very_weak", "weak", "moderate", "strong", "very_strong"],
            },
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/systemone",
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    ms = (time.perf_counter() - t0) * 1000

    out = {
        "client_latency_ms": round(ms, 1),
        "server_latency_ms": body.get("latency_ms"),
        "model": body.get("model"),
        "answers": body.get("answers"),
        "usage": body.get("usage"),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
