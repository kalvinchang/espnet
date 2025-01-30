#!/bin/bash
#SBATCH --job-name=fine_tune_owsm       # Job name
#SBATCH --output=/ocean/projects/cis210027p/eyeo1/workspace/POWSM/logs/fine_tune_owsm_%j.log # Output log file with job ID
#SBATCH --error=/ocean/projects/cis210027p/eyeo1/workspace/POWSM/logs/fine_tune_owsm_%j.err  # Error log file with job ID
#SBATCH --partition=GPU-small    
#SBATCH --gres=gpu:v100-32:1                
#SBATCH --time=8:00:00
#SBATCH --cpus-per-gpu 5
#SBATCH --mem-per-gpu 64375M

/ocean/projects/cis210027p/eyeo1/anaconda3/envs/espnet/bin/python tune_owsm_ej.py

curl -X POST -H 'Content-type: application/json' --data '{"text":"owsm finetuning DONE"}' https://hooks.slack.com/services/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