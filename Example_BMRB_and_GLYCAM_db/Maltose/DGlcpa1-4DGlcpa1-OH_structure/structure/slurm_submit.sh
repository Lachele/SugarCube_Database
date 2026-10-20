#!/bin/bash
#SBATCH --chdir=/website/userdata/sequence/cb/Builds/c59adee4-08b5-4704-9236-8208a82006e5/New_Builds/structure
#SBATCH --error=slurm_%x-%A.err
#SBATCH --get-user-env
#SBATCH --job-name=Glycan-c59adee4-
#SBATCH --nodes=1
#SBATCH --output=slurm_%x-%A.out
#SBATCH --partition=amber
#SBATCH --tasks-per-node=4

/website/userdata/sequence/cb/Builds/c59adee4-08b5-4704-9236-8208a82006e5/New_Builds/structure/Minimize.bash
