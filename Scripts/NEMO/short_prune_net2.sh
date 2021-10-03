#!/bin/bash -l
#MSUB -o /work/ws/nemo/fr_ds567-lth_ws-0/nemo_logs/IMP_net2.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#MSUB -e /work/ws/nemo/fr_ds567-lth_ws-0/nemo_logs/IMP_net2.err # STDERR  (the folder log has to be created prior to running or this won't work)
#MSUB -l nodes=1:ppn=1
#MSUB -l walltime=49:59:00
#MSUB -m bea
#MSUB -M dipti.sengupta@students.uni-freiburg.de

cd $(ws_find lth_ws)/LTH_Master
module load tools/conda/latest
conda config --prepend envs_dirs $( ws_find lth_ws )/conda/envs
conda config --prepend pkgs_dirs $( ws_find lth_ws )/conda/pkgs
conda config --show envs_dirs
conda config --show pkgs_dirs

conda activate lth_env
conda install -y numpy matplotlib pytorch tensorboard torchvision pandas
conda install -c conda-forge -y pytorch-model-summary pytorch-lightning
python3 -c "import torch; print(torch.__version__)"

python3 -m src.vanilla_pytorch.shortcut_pruning --model Net2 --batch-size 60 --epochs 30 --lr 2e-4 --dataset cifar10 --name Net2_IMP1

conda deactivate