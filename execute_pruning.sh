#!/bin/bash
#MSUB -o /work/ws/nemo/fr_ds567-lth_ws-0/nemo_logs/prune_28.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#MSUB -e /work/ws/nemo/fr_ds567-lth_ws-0/nemo_logs/prune_28.err # STDERR  (the folder log has to be created prior to running or this won't work)
#MSUB -l nodes=1:ppn=1:gpus=1
#MSUB -l walltime=5:00:00
#MSUB -q gpu
#MSUB -l pmem=6GB
#MSUB -m bea
#MSUB -M dipti.sengupta@students.uni-freiburg.de

set -e
set -o pipefail


cd $(ws_find lth_ws)/LTH_Master
echo "load modules conda, create env, then load mpi"
module load tools/conda/latest
conda config --prepend envs_dirs $( ws_find lth_ws )/conda/envs
conda config --prepend pkgs_dirs $( ws_find lth_ws )/conda/pkgs
conda config --show envs_dirs
conda config --show pkgs_dirs

# conda create --name lth_env python=3.8 -y
conda activate lth_env

echo "Other libs and check"
python3 --version
conda install -y numpy matplotlib pytorch tensorboard torchvision pandas tqdm
pip install torchsummary
conda install -y tensorflow

python3 -c "import torch; print(torch.__version__)"

python3 src/run_prune_net.py --epochs 66 --pruning-levels 28
echo "Run time"

echo "clean up env"
conda deactivate