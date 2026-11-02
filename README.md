# myjev

本地 JEV 族（Kev）+ Chronos 零样本试验台。权重与缓存放在 **F 盘**。

## 目录

| 路径 | 内容 |
|------|------|
| `F:\aicoding\myjev\models\` | 模型本地副本（Kev adapter、Qwen 基座、Chronos） |
| `F:\aicoding\myjev\hf_cache\` | Hugging Face 缓存（`kev.serve` 从此加载） |
| `F:\aicoding\myjev\repos\kev\` | [jaredpalmer/kev](https://github.com/jaredpalmer/kev) 源码 + `.venv` |
| `F:\aicoding\myjev\scripts\` | 启动 / 烟测脚本 |

## 已下载（3090 友好小模型）

- `kev-0.8b` + `Qwen3.5-0.8B-Base`
- `kev-4b` + `Qwen3.5-4B-Base`（约 8.8GB 权重）
- `chronos-bolt-tiny`、`chronos-2-small`

## 启动 Kev-0.8B

**不要用 `uv run`**（会把 torch 同步回 CPU 版）。用 venv 里的 CUDA torch：

```powershell
powershell -File F:\aicoding\myjev\scripts\serve_kev08.ps1
```

或：

```bash
export HF_HUB_CACHE=/f/aicoding/myjev/hf_cache
export HF_HOME=/f/aicoding/myjev/hf_home
cd /f/aicoding/myjev/repos/kev
.venv/Scripts/python.exe -m kev.serve --run jaredpalmer/kev-0.8b --port 8008 --device cuda
```

若 CUDA 丢失，重新装：

```bash
cd /f/aicoding/myjev/repos/kev
uv pip install --python .venv/Scripts/python.exe "torch==2.8.0+cu128" --index-url https://download.pytorch.org/whl/cu128
```

## 烟测

```bash
# 判定 API（需服务已启动）
python F:/aicoding/myjev/scripts/smoke_stock_decide.py

# Chronos（独立，不依赖 Kev 服务）
python F:/aicoding/myjev/scripts/smoke_chronos.py
```

## 备注

- Kev 需要 Python 3.12/3.13；本环境用 uv 安装的 3.13 + torch 2.8.0+cu128。
- 后续对比：停 0.8B 后改 `--run jaredpalmer/kev-4b --port 8009`。
- 下一步：Chronos 预测摘要写入 `state`，再让 Kev 做 Choice/Noul。

## TimesJev dual-gate 实验（已同步）

代码与摘要报告在 `timesjev/`（不含大权重与 `runs` 缓存）：

- `timesjev/scripts/`：A股 dual-gate、BTC/美股扩池、新闻与 live-week 脚本
- `timesjev/configs/selections/`：特征选择配置
- `timesjev/reports/`：BTC / 美股扩池 / 里程碑文字报告

本地完整 TimesJev 工程仍在 `D:\aicoding\TimesJev`；模型权重不入库。
