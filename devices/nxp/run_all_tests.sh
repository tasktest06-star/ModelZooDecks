#!/bin/bash
set -e
echo "=== NXP eIQ Per-Device Inference Tests ==="
cd "$(dirname "$0")/../.."
python -m pytest devices/nxp/ -v --tb=short --cov=devices/nxp --cov-report=term-missing 2>&1
