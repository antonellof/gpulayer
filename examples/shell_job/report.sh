#!/usr/bin/env bash
# Any executable can be a GpuLayer job: here, plain shell tools.
# The input file is staged into $GPULAYER_INPUT_DIR; anything written to
# $GPULAYER_OUTPUT_DIR comes back as an output artifact.
set -euo pipefail
in="$GPULAYER_INPUT_DIR/sales.csv"
out="$GPULAYER_OUTPUT_DIR/totals.csv"
echo "running on $(hostname), accelerator=$GPULAYER_ACCELERATOR"
awk -F, 'NR > 1 { total[$1] += $3 } END { for (r in total) printf "%s,%.2f\n", r, total[r] }' "$in" \
  | sort > "$out"
cat "$out"
