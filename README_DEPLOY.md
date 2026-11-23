# A股情绪作战室 · 每日自动更新部署手册

本目录已整合实测最靠谱的 **第一梯队核心指标**（`s_MB10` 资金风格、`Health_MO` 波段择时、`ACSI` 十年大周期）与 **第二梯队逆向护栏**（`STSI` 冰点/高潮 + 大面率），单次增量更新仅需 **~4 秒**（通过腾讯批量行情接口拉取全 A 5,200+ 只股票与 9 大宽基指数）。

## 一、快速手动更新

```bash
# 完整更新（含 ACSI 融资/QVIX/破净率/股债利差非阻塞刷新）
.venv/bin/python daily_update.py

# 极速更新（仅全 A 盘口 + 指数，~3.5 秒）
.venv/bin/python daily_update.py --fast

# 更新并推送到飞书 / 钉钉 / 企业微信 Webhook
WEBHOOK_URL="https://open.feishu.cn/open-apis/bot/v2/hook/..." .venv/bin/python daily_update.py --notify
```

- **盘中运行（09:25–14:59 北京时间）**：仅生成「盘中实时预览」，**不会污染**历史收盘 CSV。
- **收盘后运行（≥ 15:00 北京时间）**：自动将当日确认收盘数据写入 `data/*.csv` 与 `data/daily_state.json`，并同步刷新 `results_live/index.html` 与 `docs/index.html`。

## 二、三种自动化落地方式

### 1. GitHub Actions + GitHub Pages（零成本云端推荐）
已内置 `.github/workflows/daily_sentiment_update.yml`（每个交易日 **15:15 北京时间**自动触发）：
1. 将本目录推送到你的 GitHub 仓库；
2. 仓库 `Settings → Pages` 中将 Source 设为 `Deploy from a branch`，分支选 `main`、目录选 `/docs`；
3. （可选）在 `Settings → Secrets → Actions` 添加 `WEBHOOK_URL` 即可每日收盘自动推群；
4. 之后每个交易日 15:16 打开你的 GitHub Pages 链接即可在手机/电脑查看最新看板。

### 2. 本机 / VPS / NAS 定时任务 + 本地一键刷新看板
```bash
# 安装工作日 15:15 crontab 定时任务
bash setup_cron.sh

# 启动带“立即拉取最新”按钮的本地 HTTP 看板
.venv/bin/python serve_dashboard.py --host 0.0.0.0 --port 8765
```

### 3. 遗漏多日后的状态自愈
如果机器停机多日导致 `data/daily_state.json` 落后，先运行 `bootstrap_state.py` 重建最新收盘日的连板状态即可无缝接续。
