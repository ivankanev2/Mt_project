#!/bin/bash
# ====== user-configurable ======
PART=gpu            # <- change this to whatever sinfo showed (e.g., gpu)
CPUS=8              # try 8, then 16, 32, 64...
MEM=16G
TIME=00:05:00
# ===============================

#SBATCH -p ${PART}
#SBATCH -c ${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH -J cpu_test
#SBATCH -o logs/%x_%j.out

# If submitting to a GPU partition, explicitly request ZERO GPUs
#SBATCH --gres=gpu:0

set -euo pipefail
echo "Node: $(hostname)"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK}"

python - <<'PY'
import multiprocessing, time, os
n = int(os.environ.get("SLURM_CPUS_PER_TASK","1"))
print("Python will use", n, "CPUs")
def work(_):
    s = 0
    for i in range(4_000_000):
        s += i*i
    return s
start = time.time()
from multiprocessing import Pool
with Pool(n) as p:
    p.map(work, range(n))
print("Done in", round(time.time()-start,2), "seconds")
PY
