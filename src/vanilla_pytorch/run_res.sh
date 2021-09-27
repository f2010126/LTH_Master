#!/bin/bash -l
#MSUB -o /work/ws/nemo/fr_ds567-lth_ws-0/nemo_logs/run_res.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#MSUB -e /work/ws/nemo/fr_ds567-lth_ws-0/nemo_logs/run_res.err # STDERR  (the folder log has to be created prior to running or this won't work)
#MSUB -q gpu
#MSUB -l nodes=1:ppn=8:gpus=1
#MSUB -l walltime=59:59:00
#MSUB -l pmem=5000mb
#MSUB -l naccesspolicy=singlejob
#MSUB -m bea
#MSUB -M dipti.sengupta@students.uni-freiburg.de

cd $(ws_find lth_ws)/LTH_Master
module load tools/conda/latest
conda config --prepend envs_dirs $( ws_find lth_ws )/conda/envs
conda config --prepend pkgs_dirs $( ws_find lth_ws )/conda/pkgs
conda config --show envs_dirs
conda config --show pkgs_dirs

cd src
conda activate lth_env
conda install -y numpy matplotlib pytorch tensorboard torchvision pandas
conda install -c conda-forge -y pytorch-model-summary
python3 -c "import torch; print(torch.__version__)"

python3 -m run_pruning_experiment --model $MODEL --batch-size $BATCH --epochs $EPOCH --lr $LR --pruning-levels 25 --dataset cifar10 --name $NAME

conda deactivate