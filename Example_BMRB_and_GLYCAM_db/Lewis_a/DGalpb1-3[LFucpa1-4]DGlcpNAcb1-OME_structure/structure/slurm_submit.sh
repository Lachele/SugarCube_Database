#!/bin/bash
#SBATCH --chdir=/website/userdata/sequence/cb/Builds/003c3607-34af-45d3-8ef4-61a3fb438303/New_Builds/structure
#SBATCH --error=slurm_%x-%A.err
#SBATCH --get-user-env
#SBATCH --job-name=Glycan-003c3607-
#SBATCH --nodes=1
#SBATCH --output=slurm_%x-%A.out
#SBATCH --partition=amber
#SBATCH --tasks-per-node=4

/website/userdata/sequence/cb/Builds/003c3607-34af-45d3-8ef4-61a3fb438303/New_Builds/structure/Minimize.bash
