#!/bin/bash -l
#MSUB -o /work/ws/nemo/fr_ds567-lth_ws-0/nemo_logs/short_res.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#MSUB -e /work/ws/nemo/fr_ds567-lth_ws-0/nemo_logs/short_res.err # STDERR  (the folder log has to be created prior to running or this won't work)
#MSUB -l nodes=1:ppn=1
#MSUB -l walltime=49:59:00
#MSUB -m bea
#MSUB -M dipti.sengupta@students.uni-freiburg.de

cd $(ws_find lth_ws)
# python3 -m venv lth_env
source lth_env/bin/activate
pip list
cd LTH_Master

#pip install numpy matplotlib torch tensorboard torchvision pandas
#pip install pytorch-model-summary pytorch-lightning
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"

python3 -m src.vanilla_pytorch.shortcut_pruning --model Resnets --batch-size 512 --epochs 30 --lr 0.01 --dataset cifar10 --name short_res

conda deactivate