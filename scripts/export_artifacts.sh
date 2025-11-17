#!/usr/bin/env bash
set -euo pipefail

dest="export_$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$dest"
cp -r artifacts "$dest/" 2>/dev/null || true
cp -r configs "$dest/"
cp README.md MODEL_CARD.md "$dest/" 2>/dev/null || true
echo "Exported to $dest"
