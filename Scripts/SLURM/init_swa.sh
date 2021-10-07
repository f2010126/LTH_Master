#!/bin/bash -l
#SBATCH -o /work/dlclarge1/dsengupt-lth_ws/nemo_logs/init_swa.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e /work/dlclarge1/dsengupt-lth_ws/nemo_logs/init_swa.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J Init_SWA
#SBATCH -N 1
#SBATCH -t 19:59:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de

cd $(ws_find lth_ws)
# python3 -m venv lth_env
source lth_env/bin/activate
pip list
cd LTH_Master

python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"

echo "Starting Init SWA for Net2  Run 1"
python3 -m src.vanilla_pytorch.init_swa --model Net2 --batch-size 60 --epochs 30 --lr 2e-4 --dataset cifar10 --name InitSWA_Net2_Run1

echo "Starting Init SWA for Resnet Run 1"
python3 -m src.vanilla_pytorch.init_swa --model Resnets --batch-size 512 --epochs 30 --lr 0.01 --dataset cifar10 --name InitSWA_ResNet_Run1

echo "Starting Init SWA for Net2 Run 2"
python3 -m src.vanilla_pytorch.init_swa --model Net2 --batch-size 60 --epochs 30 --lr 2e-4 --dataset cifar10 --name InitSWA_Net2_Run2

echo "Starting Init SWA for Net2 Run 3"
python3 -m src.vanilla_pytorch.init_swa --model Net2 --batch-size 60 --epochs 30 --lr 2e-4 --dataset cifar10 --name InitSWA_Net2_Run3

echo "Starting Init SWA for Resnet Run 2"
python3 -m src.vanilla_pytorch.init_swa --model Resnets --batch-size 512 --epochs 30 --lr 0.01 --dataset cifar10 --name InitSWA_ResNet_Run2

echo "Starting Init SWA for Resnet Run 3"
python3 -m src.vanilla_pytorch.init_swa --model Resnets --batch-size 512 --epochs 30 --lr 0.01 --dataset cifar10 --name InitSWA_ResNet_Run3


deactivate
