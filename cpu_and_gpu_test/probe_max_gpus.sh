# file: probe_max_gpus.sh
#!/usr/bin/env bash
set -euo pipefail

PART=${1:-gpu}
CPUS_PER_GPU=${2:-4}   # modest rule of thumb
MEM=${3:-16G}
TIME=${4:-00:05:00}

#this code was only used to test the max gpus of the lab you can ignore it

# Try 1..8 because nodes show up to 8x RTX 5000 Ada
CANDIDATES=(1 2 3 4 5 6 7 8)

echo "Probing max GPUs on partition=$PART (mem=$MEM, time=$TIME)"
MAX_OK=0
for G in "${CANDIDATES[@]}"; do
  CPUS=$((G * CPUS_PER_GPU))
  echo -n "  gpus=$G (cpus-per-task=$CPUS) ... "
  OUT=$(sbatch --parsable \
    --partition="$PART" \
    --gres=gpu:"$G" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --no-requeue \
    --wrap 'nvidia-smi >/dev/null 2>&1; echo OK on $(hostname)')
  RC=$?
  if [[ $RC -eq 0 ]]; then
    JOBID="$OUT"
    echo "ACCEPTED (job $JOBID)"
    scancel "$JOBID" >/dev/null 2>&1 || true
    MAX_OK=$G
  else
    echo "REJECTED"
  fi
done

if [[ $MAX_OK -gt 0 ]]; then
  echo "==> Max GPUs accepted: $MAX_OK"
else
  echo "No GPU counts accepted on $PART (check quotas or partition name)."
fi
