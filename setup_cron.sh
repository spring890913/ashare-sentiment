#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

CRON_TAG="# ASHARE_SENTIMENT_DAILY"
# Run Mon-Fri at 15:15 (set system TZ to Asia/Shanghai or adjust hour accordingly)
CRON_CMD="15 15 * * 1-5 cd ${PROJECT_DIR} && TZ=Asia/Shanghai ${PYTHON_BIN} ${PROJECT_DIR}/daily_update.py --notify >> ${LOG_DIR}/daily_update.log 2>&1 ${CRON_TAG}"

(crontab -l 2>/dev/null | grep -v "${CRON_TAG}" || true; echo "${CRON_CMD}") | crontab -
echo "[OK] Installed weekday 15:15 cron job:"
crontab -l | grep "${CRON_TAG}"
echo "Logs will be written to: ${LOG_DIR}/daily_update.log"
