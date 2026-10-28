#!/bin/bash
# Simulation phase of the VHA comparison, sharded across processes.
#
# Cases are independent, so they are split across concurrent processes. Only
# PatientsWithPersonality needs the vLLM meta model, and one 14B engine fills the
# GPU, so that arm runs alone in a single process while the API-only arms are
# sharded. Arms and cases are read from the config, so already-simulated runs are
# skipped by run_patient_comparison.py's own resume check. LLM-judged metrics are
# deferred to run_posthoc_eval.py.
#
# Usage: CONFIG=<experiment config> scripts/run_vha_sharded.sh [n_api_shards] [--api-only] [--missing-only]
#   Shards write to <output_dir>_pwp and <output_dir>_api<i>; merge them with merge_result_shards.py.
#   --api-only     skips the PatientsWithPersonality process, for when the GPU is busy.
#   --missing-only drops arms already covering every configured case in MAIN_DIR.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON=.venv/bin/python
CONFIG=${CONFIG:?set CONFIG to an experiment config name, e.g. patient_comparison_vha}

MAIN_DIR=$($PYTHON -c "from omegaconf import OmegaConf; print(OmegaConf.load('configs/experiment/$CONFIG.yaml').output_dir)")
N_SHARDS=4
API_ONLY=0
MISSING_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --api-only) API_ONLY=1 ;;
    --missing-only) MISSING_ONLY=1 ;;
    *) N_SHARDS=$arg ;;
  esac
done

mapfile -t CASES < <($PYTHON -c "
from omegaconf import OmegaConf
for c in OmegaConf.load('configs/experiment/$CONFIG.yaml').conversations:
    print(c)
")

read_arms() {
  $PYTHON -c "
import sys, yaml
from pathlib import Path
import pandas as pd
from omegaconf import OmegaConf
from patient_simulator.misc.utils import expand_param_grid, patient_name_from_config

keep, missing_only = '$1', '$MISSING_ONLY' == '1'
cfg = OmegaConf.load('configs/experiment/$CONFIG.yaml')
cases = set(cfg.conversations)

done = {}
conv_csv = Path('$MAIN_DIR') / 'all_conversations.csv'
if missing_only and conv_csv.exists():
    df = pd.read_csv(conv_csv)
    done = df.groupby('patient_name')['conversation_name'].agg(set).to_dict()

for arm in cfg.patient_configs:
    is_pwp = arm.patient_type == 'PatientsWithPersonality'
    if (keep == 'pwp') != is_pwp:
        continue
    arm = OmegaConf.to_container(arm, resolve=True)
    if missing_only:
        names = [
            patient_name_from_config(arm['patient_type'], combo)
            for combo in expand_param_grid(arm.get('params') or {})
        ]
        if all(cases <= done.get(n, set()) for n in names):
            print(f'skipping complete arm: {names}', file=sys.stderr)
            continue
    print(yaml.safe_dump(arm, default_flow_style=True, width=10**6).strip())
"
}

mapfile -t PWP_ARMS < <(read_arms pwp)
mapfile -t API_ARMS < <(read_arms api)
API_JOINED=$(IFS=,; echo "${API_ARMS[*]}")

echo "${#CASES[@]} cases | ${#API_ARMS[@]} API arms over $N_SHARDS shards | ${#PWP_ARMS[@]} PWP arms"
mkdir -p logs

if [[ $API_ONLY -eq 0 && ${#PWP_ARMS[@]} -gt 0 ]]; then
  PWP_JOINED=$(IFS=,; echo "${PWP_ARMS[*]}")
  nohup $PYTHON scripts/run_patient_comparison.py --config-name "$CONFIG" \
    output_dir="${MAIN_DIR}_pwp" \
    "patient_configs=[$PWP_JOINED]" \
    > "logs/${CONFIG}_pwp.log" 2>&1 &
  echo "  pwp -> pid $! (${#CASES[@]} cases, logs/${CONFIG}_pwp.log)"
fi

for ((i = 0; i < N_SHARDS; i++)); do
  shard=()
  for ((j = i; j < ${#CASES[@]}; j += N_SHARDS)); do
    shard+=("${CASES[$j]}")
  done
  joined=$(IFS=,; echo "${shard[*]}")
  nohup $PYTHON scripts/run_patient_comparison.py --config-name "$CONFIG" \
    output_dir="${MAIN_DIR}_api$i" \
    "conversations=[$joined]" \
    "patient_configs=[$API_JOINED]" \
    > "logs/${CONFIG}_api$i.log" 2>&1 &
  echo "  api$i -> pid $! (${#shard[@]} cases, logs/${CONFIG}_api$i.log)"
done

wait
echo "All simulation shards finished."
