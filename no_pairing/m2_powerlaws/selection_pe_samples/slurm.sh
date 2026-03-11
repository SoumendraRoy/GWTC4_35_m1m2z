#!/bin/bash
#SBATCH --job-name=sel
#SBATCH --output=sel.out
#SBATCH --error=sel.err
#SBATCH -p cca
#SBATCH --time=5-00:10:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=50
#SBATCH --mem-per-cpu=500M 
python Trim_Sel_PE.py