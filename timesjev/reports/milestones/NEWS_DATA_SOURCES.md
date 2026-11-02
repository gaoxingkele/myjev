# A 股 / 财经新闻数据源清单（替代 Tushare `cctv_news`）

> 结论：卡点不是「网上没有新闻」，而是我们绑死了 Tushare 高权限接口。  
> **免 Key 公开源已经够用**；2026-11-01 实测 CLS + 新浪 7x24 可拉。

## 推荐优先级

| 优先级 | 源 | 获取方式 | Key | 用途 | 实测 |
|--------|----|----------|-----|------|------|
| P0 | 财联社电报 | `cls.cn/api/cache?name=telegraph` 或 `ak.stock_info_global_cls` | 否 | 全市场快讯/情绪 | OK |
| P0 | 新浪 7x24 | `zhibo.sina.com.cn/api/zhibo/feed` 或 `ak.stock_info_global_sina` | 否 | 全市场快讯 | OK |
| P0 | 东财个股新闻 | `search-api-web.eastmoney.com` 或 `ak.stock_news_em` | 否 | 个股新闻 | 可用（解析注意 list/dict） |
| P1 | 东财全球快讯 | `ak.stock_info_global_em` / np-weblist | 否 | 7x24 | 端点偶发空 |
| P1 | 同花顺 / 富途快讯 | `ak.stock_info_global_ths` / `futu` | 否 | 备用快讯 | 未测 |
| P1 | 央视新闻联播 | `tv.cctv.com/lm/xwlb/` HTML / a-stock-data 封装 | 否 | 政策面（替代 cctv_news） | 页面 OK |
| P2 | 巨潮公告 | cninfo | 否 | 公告事件（非情绪） | 库内已有能力 |
| P2 | OpenStockData / a-stock-data | pip/git 封装上述源 | 否 | 统一客户端 | 社区维护 |
| P3 | Perplexity sonar | 现有 `news_search.py` | 要 Key | 增强检索 | 可选 |
| P3 | Tushare cctv_news | pro API | 要权限 | 联播全文 | **当前无权限** |

## 直接可用的公开 URL

```text
# 财联社电报
GET https://www.cls.cn/api/cache?app=CailianpressWeb&name=telegraph&os=web&sv=8.7.9
Referer: https://www.cls.cn/telegraph

# 新浪 7x24
GET https://zhibo.sina.com.cn/api/zhibo/feed?page=1&page_size=30&zhibo_id=152&tag_id=0&type=0
Referer: https://finance.sina.com.cn/7x24/

# 东财个股新闻搜索（JSONP）
GET https://search-api-web.eastmoney.com/search/jsonp?cb=jQuery&param=<urlencoded JSON>
```

## 本仓库接入

- 聚合脚本：`scripts/mm_news_free_sources.py`
- 输出：`stockagent-analysis/output/policy_heat/daily_policy_heat_<YYYYMMDD>.json`
- 消费方：`scripts/live_week_predict_mm.py`（`load_policy_sentiment`）

## 参考项目

- [akshare 新闻](https://akshare.akfamily.xyz/)：`stock_news_em` / `stock_info_global_*`
- [OpenStockData](https://github.com/Litton-Lei/OpenStockData)：东财/财联社/新闻层封装
- [a-stock-data](https://github.com/simonlin1212/a-stock-data)：含新闻联播文字稿、华尔街见闻等
- [ashare-news-fetcher source_apis](https://github.com/daymade/claude-code-skills/blob/main/daymade-financial/ashare-news-fetcher/references/source_apis.md)
