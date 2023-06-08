#!/usr/bin/bash
#SBATCH --job-name=training
#SBATCH --output=/home/users/%u/out/training.%j.out
#SBATCH --error=/home/users/%u/err/training.%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=12GB
module load python/3.9.0
#pip3 install git+https://github.com/huggingface/transformers
pip3 install transformers
#no pytorch support for 3.10 yet...use this instead of nightly if it becomes supported
#pip3 install torch torchvision torchaudio
pip3 install --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
python3 training.py