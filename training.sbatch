#!/usr/bin/bash
#SBATCH --job-name=training
#SBATCH --output=/home/%u/out/training.%j.out
#SBATCH --error=/home/%u/err/training.%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB

pip3 install transformers
pip3 install --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
python3 training.py
