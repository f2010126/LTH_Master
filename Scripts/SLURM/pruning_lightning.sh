#!/bin/bash -l
#SBATCH -o /work/dlclarge2/dsengupt-lth_ws/slurm_logs/lth.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e /work/dlclarge2/dsengupt-lth_ws/slurm_logs/lth.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J RepExp4_3
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH -t 10:59:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dipti.sengupta@students.uni-freiburg.de


while getopts c: flag
do
    case "${flag}" in
        c) config_name=${OPTARG};;
    esac
done
echo "# Config file: $config_name";

cd $(ws_find lth_ws)
# python3 -m venv lth_env
source lth_env/bin/activate
pip list
cd LTH_Master
# pick config file here
python3 -m src.Lightning_WandB.iterative_pruning --config_file_name $config_name

deactivate