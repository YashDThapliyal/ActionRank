#!/bin/zsh
set -o pipefail
cd /Users/yash/Documents/ActionRank
export ACTIONRANK_CONFIG=configs/unmonitored_holdout.yaml
.venv/bin/python eval.py --systems baseline_sft,tier2 --offset 503 --all-remaining 2>&1 | tee results/06-unmonitored-holdout/eval.log || { echo "EVAL FAILED rc=$?" >> results/06-unmonitored-holdout/eval.log; exit 1; }
.venv/bin/python scripts/analyse_predictions.py > results/06-unmonitored-holdout/analysis_body.md
.venv/bin/python scripts/paired_test.py results/06-unmonitored-holdout/predictions_tier2.jsonl results/06-unmonitored-holdout/predictions_baseline_sft.jsonl --margin 3 | tee results/06-unmonitored-holdout/paired.txt
echo "RUN COMPLETE" >> results/06-unmonitored-holdout/eval.log
