#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --job-name train-finance-2
#SBATCH --output train-fiannce-300-2.out
#SBATCH --error train.err

# Run python file.

# Load our conda environment
module load anaconda/2024.02-py311
source activate SentimentAnalysis

# Run the test code
python3 ~/sentiment_analysis/train.py
