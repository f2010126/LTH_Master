cd $(ws_find lth_ws)/LTH_Master

ml devel/conda
conda config --prepend envs_dirs $( ws_find conda )/conda/envs
conda config --prepend pkgs_dirs $( ws_find conda )/conda/pkgs
conda config --show envs_dirs
conda config --show pkgs_dirs


cd src
conda activate lth_env
