#!/bin/bash
# Submits one job per model, then a summary that starts when they have all ended. Run on the login node, from the
# repository root. Rerunning continues unfinished work: finished questions are skipped.
#   bash accuracy_tables/hpc/submit_all.sh test
#   bash accuracy_tables/hpc/submit_all.sh all qwen3 phi
set -euo pipefail
V=${1:?usage: submit_all.sh <test|all> [models...]}
shift
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(qwen2.5 qwen3 internvl gemma phi smolvlm)
fi
IDS=()
for M in "${MODELS[@]}"; do
    ID=$(sbatch --parsable --job-name="acc-$V-$M" accuracy_tables/hpc/run_model.SBATCH "$M" "$V")
    ID=${ID%%;*}
    echo "$M: job $ID"
    IDS+=("$ID")
done
AFTER=$(IFS=:; echo "${IDS[*]}")
echo "summary: job $(sbatch --parsable --job-name="acc-$V-summary" --dependency="afterany:$AFTER" accuracy_tables/hpc/summarize.SBATCH "$V")"
