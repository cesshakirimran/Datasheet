#!/usr/bin/env bash
# Full experiment with the real models in config.yaml (Ollama must be running).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m harag check
python -m harag ingest
python -m harag calibrate
python -m harag evaluate
pip freeze > results/environment.txt
command -v ollama >/dev/null && ollama list > results/ollama_models.txt || true
echo "Report: results/test/report.md"
