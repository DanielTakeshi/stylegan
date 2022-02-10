#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --exclude=compute-0-[7,9,11,13,17,19,21,23,25,27],compute-1-[9]
#SBATCH --cpus-per-task=10
#SBATCH --time=240:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH -o ./log.txt
#SBATCH -e ./log_e.txt
#SBATCH --mail-type=END,FAIL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=dseita@andrew.cmu.edu   # Where to send mail
set -x
set -u
set -e

time  \
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate stylegan
    python pretrained_example.py
