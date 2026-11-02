#!/usr/bin/env python3
"""免 Key 财经新闻聚合（替代 Tushare cctv_news 权限卡点）。

实测可用（2026-11-01）:
  - 财联社电报 CLS: https://www.cls.cn/api/cache?...name=telegraph
  - 新浪 7x24:      https://zhibo.sina.com.cn/api/zhibo/feed?...
  - 东财个股搜索:   https://search-api-web.eastmoney.com/search/jsonp
  - 央视新闻联播页: https://tv.cctv.com/lm/xwlb/ （HTML，可再抽文字稿）

另可选（需 akshare / 库）:
  - ak.stock_news_em / stock_info_global_em / stock_info_global_cls / ths / futu
  - OpenStockData / a-stock-data 封装层
  - 巨潮公告 cninfo（事件驱动，非情绪快讯）

输出兼容 live_week_predict_mm.load_policy_sentiment 的 JSON。
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

OUT = Path(r"D:/aicoding/stockagent-analysis/output/policy_heat")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"}


def _get(url: str, headers: dict | None = None, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "ignore")


def fetch_cls(limit: int = 40) -> list[dict]:
    url = "https://www.cls.cn/api/cache?app=CailianpressWeb&name=telegraph&os=web&sv=8.7.9"
    raw = _get(url, {"Referer": "https://www.cls.cn/telegraph"})
    data = json.loads(raw).get("data") or {}
    roll = data.get("roll_data") or data.get("telegraph") or data.get("roll") or []
    rows = []
    for it in roll[:limit]:
        title = (it.get("title") or it.get("content") or it.get("brief") or "").strip()
        if not title:
            continue
        ctime = it.get("ctime")
        if isinstance(ctime, (int, float)):
            ts = datetime.fromtimestamp(ctime).strftime("%Y-%m-%d %H:%M")
        else:
            ts = str(ctime or "")
        rows.append({"time": ts, "title": title[:200], "content": (it.get("content") or title)[:400], "source": "cls"})
    return rows


def fetch_sina(limit: int = 40) -> list[dict]:
    url = f"https://zhibo.sina.com.cn/api/zhibo/feed?page=1&page_size={limit}&zhibo_id=152&tag_id=0&type=0"
    raw = _get(url, {"Referer": "https://finance.sina.com.cn/7x24/"})
    feed = ((json.loads(raw).get("result") or {}).get("data") or {}).get("feed") or {}
    rows = []
    for it in (feed.get("list") or [])[:limit]:
        text = re.sub(r"<[^>]+>", "", it.get("rich_text") or it.get("content") or "").strip()
        if not text:
            continue
        rows.append(
            {
                "time": str(it.get("create_time") or ""),
                "title": text[:120],
                "content": text[:400],
                "source": "sina",
            }
        )
    return rows


def fetch_eastmoney_stock(symbol: str, limit: int = 10) -> list[dict]:
    param = json.dumps(
        {
            "uid": "",
            "keyword": symbol,
            "type": ["cmsArticleWebOld"],
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "param": {
                "cmsArticleWebOld": {
                    "searchScope": "default",
                    "sort": "default",
                    "pageIndex": 1,
                    "pageSize": limit,
                    "preTag": "",
                    "postTag": "",
                }
            },
        },
        ensure_ascii=False,
    )
    url = f"https://search-api-web.eastmoney.com/search/jsonp?cb=jQuery&param={urllib.parse.quote(param)}"
    raw = _get(url, {"Referer": "https://so.eastmoney.com/"})
    m = re.search(r"jQuery\((.*)\)\s*$", raw, re.S)
    data = json.loads(m.group(1) if m else raw)
    cms = (data.get("result") or {}).get("cmsArticleWebOld")
    if isinstance(cms, dict):
        lst = cms.get("list") or []
    elif isinstance(cms, list):
        lst = cms
    else:
        lst = []
    rows = []
    for it in lst[:limit]:
        title = re.sub(r"<[^>]+>", "", it.get("title") or "")
        content = re.sub(r"<[^>]+>", "", it.get("content") or "")
        rows.append(
            {
                "time": str(it.get("date") or ""),
                "title": title[:160],
                "content": content[:400],
                "url": it.get("url") or "",
                "source": it.get("mediaName") or "eastmoney",
            }
        )
    return rows


def keyword_sentiment(texts: list[str]) -> dict:
    blob = "\n".join(texts)
    bull_kw = ["增长", "利好", "回升", "扩大", "稳增长", "降准", "降息", "支持", "突破", "加仓", "买入", "上涨", "创新高"]
    bear_kw = ["风险", "下滑", "制裁", "收紧", "处罚", "暴跌", "战争", "下行", "减持", "亏损", "爆雷", "退市", "下跌"]
    b = sum(blob.count(k) for k in bull_kw)
    e = sum(blob.count(k) for k in bear_kw)
    if b > e + 3:
        sent = "bullish"
    elif e > b + 3:
        sent = "bearish"
    else:
        sent = "neutral"
    return {
        "market_sentiment": sent,
        "bull_hits": b,
        "bear_hits": e,
        "heat_score": int(min(100, 40 + 4 * abs(b - e))),
    }


def build_policy_heat(asof: str | None = None, stock_symbols: list[str] | None = None) -> dict:
    asof = asof or datetime.now().strftime("%Y%m%d")
    items: list[dict] = []
    errors: list[str] = []
    for name, fn in (("cls", fetch_cls), ("sina", fetch_sina)):
        try:
            rows = fn(40)
            items.extend(rows)
        except Exception as ex:
            errors.append(f"{name}:{type(ex).__name__}:{ex}")
    stock_news = {}
    for sym in stock_symbols or []:
        try:
            stock_news[sym] = fetch_eastmoney_stock(sym, 8)
        except Exception as ex:
            errors.append(f"em:{sym}:{type(ex).__name__}")
    texts = [f"{x.get('title','')} {x.get('content','')}" for x in items]
    for rows in stock_news.values():
        texts.extend(f"{x.get('title','')} {x.get('content','')}" for x in rows)
    sent = keyword_sentiment(texts)
    payload = {
        "trade_date": asof,
        "market_sentiment": sent["market_sentiment"],
        "macro_signal": f"free-news cls+sina bull={sent['bull_hits']} bear={sent['bear_hits']}",
        "themes": [
            {
                "topic": "market_flash_aggregate",
                "heat_score": sent["heat_score"],
                "benefit_sectors": [],
                "key_phrases": [],
            }
        ],
        "source": "cls+sina(+eastmoney_stock)",
        "status": "ok_free_sources",
        "n_flash": len(items),
        "n_stock_news": sum(len(v) for v in stock_news.values()),
        "errors": errors,
        "flash_sample": items[:12],
        "stock_news": stock_news,
    }
    return payload


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=datetime.now().strftime("%Y%m%d"))
    ap.add_argument("--stocks", default="600519,300750,000001", help="comma codes for per-stock EM news")
    args = ap.parse_args()
    stocks = [s.strip() for s in args.stocks.split(",") if s.strip()]
    payload = build_policy_heat(args.asof, stocks)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"daily_policy_heat_{args.asof}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "saved": str(path),
                "status": payload["status"],
                "sentiment": payload["market_sentiment"],
                "n_flash": payload["n_flash"],
                "n_stock_news": payload["n_stock_news"],
                "errors": payload["errors"],
                "sample": [x["title"][:50] for x in payload["flash_sample"][:5]],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
