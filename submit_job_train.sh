#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --job-name train-lr25
#SBATCH --output train-lr25.out
#SBATCH --error train.err

# Run python file.

# Load our conda environment
module load anaconda/2024.02-py311
source activate SentimentAnalysis

# Run the test code
python3 ~/sentiment_analysis/stock_prediction/train.py
