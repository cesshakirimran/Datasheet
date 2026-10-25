#!/usr/bin/env bash
# End-to-end smoke test without any models: ingest -> ask -> calibrate -> evaluate.
# Numbers produced by this profile are NOT research results (the demo LLM injects errors on purpose).
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=config.offline.yaml
python -m harag --config $CFG ingest
python -m harag --config $CFG ask "What is the hotel limit per night in London?" --compare
python -m harag --config $CFG calibrate
python -m harag --config $CFG evaluate
echo "Report: results_offline/test/report.md"
