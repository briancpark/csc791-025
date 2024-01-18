#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p rome
#SBATCH -t 24:00:00

nvidia-smi
source ~/.bashrc
conda activate csc791
cd /home/bcpark/csc791-025/proj3



python3 proj3.py hpo TPE no-vgg regular
python3 proj3.py hpo Evolution no-vgg regular
python3 proj3.py hpo Hyperband no-vgg regular
python3 proj3.py hpo TPE no-vgg advanced
python3 proj3.py hpo Evolution no-vgg advanced
python3 proj3.py hpo Hyperband no-vgg advanced

python3 proj3.py hpo TPE vgg regular
python3 proj3.py hpo Evolution vgg regular
python3 proj3.py hpo Hyperband vgg regular
python3 proj3.py hpo TPE vgg advanced
python3 proj3.py hpo Evolution vgg advanced
python3 proj3.py hpo Hyperband vgg advanced