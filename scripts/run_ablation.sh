#!/usr/bin/env bash
set -euo pipefail

variants=(structural semantic hybrid)
out="artifacts/ablation_$(date -u +%Y%m%d-%H%M%S)"
mkdir -p "$out"

echo "| Variant | ROC-AUC | PR-AUC | F1 | Precision@k |" > "$out/table.md"
echo "|---|---:|---:|---:|---:|" >> "$out/table.md"

for v in "${variants[@]}"; do
  uv run python -m graph_lp.train --config configs/full.yaml --variant "$v" >/dev/null 2>&1 || true
  last=$(ls -td artifacts/*/ | head -1)
  m="${last%/}/metrics.json"

  roc=$(uv run python - "$m" <<'PY'
import json,sys
with open(sys.argv[1]) as f:
    j=json.load(f)
print(j['test']['roc_auc'])
PY
)
  pr=$(uv run python - "$m" <<'PY'
import json,sys
with open(sys.argv[1]) as f:
    j=json.load(f)
print(j['test']['pr_auc'])
PY
)
  f1=$(uv run python - "$m" <<'PY'
import json,sys
with open(sys.argv[1]) as f:
    j=json.load(f)
print(j['test']['f1'])
PY
)
  pk=$(uv run python - "$m" <<'PY'
import json,sys
with open(sys.argv[1]) as f:
    j=json.load(f)
print(j['test']['precision_at_k'])
PY
)
  echo "| $v | $roc | $pr | $f1 | $pk |" >> "$out/table.md"
done

cat "$out/table.md"
