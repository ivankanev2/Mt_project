# file: probe_max_cpus.sh
#!/usr/bin/env bash
set -euo pipefail

PART=${1:-ws-ia}
MEM=${2:-4G}
TIME=${3:-00:02:00}

#this code was only used to test the max cpus of the lab you can ignore it

# Try a sensible ladder; adjust if you want finer steps
CANDIDATES=(1 2 4 8 12 16 20 24 28 32 36 40 48 56 64)

echo "Probing max CPUs on partition=$PART (mem=$MEM, time=$TIME)"
MAX_OK=0
for N in "${CANDIDATES[@]}"; do
  echo -n "  cpus-per-task=$N ... "
  OUT=$(sbatch --parsable \
    --partition="$PART" \
    --cpus-per-task="$N" \
    --mem="$MEM" \
    --time="$TIME" \
    --no-requeue \
    --wrap 'echo OK on $(hostname)')
  RC=$?
  if [[ $RC -eq 0 ]]; then
    JOBID="$OUT"
    echo "ACCEPTED (job $JOBID)"
    # cancel immediately so we don't actually run
    scancel "$JOBID" >/dev/null 2>&1 || true
    MAX_OK=$N
  else
    echo "REJECTED"
  fi
done

if [[ $MAX_OK -gt 0 ]]; then
  echo "==> Max cpus-per-task accepted: $MAX_OK"
else
  echo "No CPU counts accepted on $PART (check quotas or partition name)."
fi
