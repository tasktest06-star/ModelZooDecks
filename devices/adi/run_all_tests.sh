#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

pip install -r requirements.txt -q

echo "=== Running ADI per-device inference tests ==="
pytest max78002/tests/ max32690/tests/ adsp_sc835/tests/ -v --tb=short
echo "=== All ADI device inference tests passed ==="
