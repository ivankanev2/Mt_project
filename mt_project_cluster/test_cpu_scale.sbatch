#!/bin/bash
#SBATCH --job-name=cpu_scale
#SBATCH --partition=ws-ia
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/cpu_scale_%j.out

module purge 2>/dev/null || true

echo "Node: $(hostname)"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Start: $(date)"

MAX=${SLURM_CPUS_PER_TASK}
for CPUS in 2 4 8 16 24 32 48 64; do
  if (( CPUS > MAX )); then
    echo "---- Skipping cpus-per-task=${CPUS} (exceeds allocation ${MAX}) ----"
    continue
  fi
  echo "---- Running cpus-per-task=${CPUS} ----"
  srun --exclusive -N1 -n1 --cpus-per-task=${CPUS} --cpu-bind=none bash -lc '
    echo "  Step CPUs: $SLURM_CPUS_PER_TASK"
    python3 - <<PY
import multiprocessing as mp, time, os
N = int(os.environ.get("SLURM_CPUS_PER_TASK","1"))
def work(_):
    s = 0
    for i in range(2_000_000):
        s += i*i % 97
    return s
t0=time.time()
with mp.Pool(processes=N) as p:
    out = sum(p.map(work, range(N)))
dt=time.time()-t0
print(f"  Python mp throughput with {N} procs -> {out} in {dt:.2f}s")
PY
  '
done

echo "End: $(date)"
