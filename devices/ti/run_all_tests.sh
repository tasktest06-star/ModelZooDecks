#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
pip install -r requirements.txt -q
echo "=== Running TI per-device inference tests ==="
pytest am62a/tests/ am67a/tests/ am68a/tests/ am69a/tests/ tda4vm/tests/ -v --tb=short
echo "=== All TI device inference tests passed ==="
