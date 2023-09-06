#!/bin/bash
#SBATCH --job-name=training
#SBATCH --output=/home/%u/out/training.%j.out
#SBATCH --error=/home/%u/err/training.%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --qos=gpu
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
ml python
pip3 install --user transformers
pip3 install --user --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
python3 /farmshare/learning/scripts/scripts/transfer_learning/training.py
