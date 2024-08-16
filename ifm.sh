#!/bin/bash
#SBATCH --job-name=IFM_foldingdiff
#SBATCH --mail-user=sizhuang@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --gpus=a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32gb
#SBATCH --time=10:00:00
#SBATCH --output=/home/sh2748/Job_Logs/IFM_protein/foldingdiff%J.log

date;hostname;pwd
module load miniconda
conda activate foldingdiff

cd /home/sh2748/foldingdiff

python lightning_try.py \
    data.batch_size=4 \
    experiment.lr=0.000005 \
    experiment.beta=0.01 \
    experiment.weight_decay=0
