#!/bin/bash
# Environment inspection: GPU, driver, toolkit, profilers, counter permissions.
#SBATCH --job-name=gie_env
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -u
source "${GPU_REPO:-$HOME/cuda-inference}/scripts/env.sh"
echo "== host =="; hostname; date -Is; uname -a; grep PRETTY /etc/os-release
echo "== slurm =="; echo "JOB=${SLURM_JOB_ID:-} NODE=${SLURMD_NODENAME:-} PART=${SLURM_JOB_PARTITION:-} GPUS=${CUDA_VISIBLE_DEVICES:-none}"
echo "== nvidia-smi =="; nvidia-smi
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,clocks.max.sm,clocks.max.mem --format=csv
echo "== toolchain =="; gcc --version | head -1; cmake --version | head -1; nvcc --version | tail -2
which ncu nsys compute-sanitizer cuobjdump
echo "== perf counter permission =="; grep -i RmProfilingAdminOnly /proc/driver/nvidia/params 2>/dev/null
echo "== modules =="; module list 2>&1
echo "== python =="; "$GPU_PYENV/bin/python" -c "import sys, numpy, pandas, matplotlib; print(sys.version.split()[0], numpy.__version__, pandas.__version__, matplotlib.__version__)" 2>&1
echo "== cudnn/cutlass =="; ls "$CUDNN_ROOT/include/cudnn_version.h" "$CUTLASS_DIR/include/cutlass/version.h" 2>&1
"$GPU_BUILD_DIR/gie_info" --measure-ceilings 2>&1 || true
