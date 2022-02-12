#!/bin/bash -l
#SBATCH -o /work/dlclarge1/dsengupt-lth_ws/slurm_logs/lightning_wandb.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e /work/dlclarge1/dsengupt-lth_ws/slurm_logs/lightning_wandb.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J Lightning_WandB
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH --mem=0
#SBATCH -q dlc-dsengupt
#SBATCH -t 9:59:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de

cd $(ws_find lth_ws)
# python3 -m venv lth_env
source lth_env/bin/activate
# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

pip list
cd LTH_Master
# pick config file here
srun python3 -m src.Lightning_WandB.lightning_wandb_basic --wand_exp_name 94BaseLineTorch --trial Baseline7 --epochs 30 --seed 7 --model torch_resnet

deactivate