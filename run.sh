#!/bin/bash

#SBATCH -p dgx2
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

module load miniconda3
__conda_setup="$('/lustre/opt/cascadelake/linux-centos7-x86_64/gcc-4.8.5/miniconda3-4.8.2-5yczksexambgeule63z3smwiwrbokjtu/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
conda activate sound_event_detect

python run.py train_evaluate configs/baseline.yaml data/eval/feature.csv data/eval/label.csv 

