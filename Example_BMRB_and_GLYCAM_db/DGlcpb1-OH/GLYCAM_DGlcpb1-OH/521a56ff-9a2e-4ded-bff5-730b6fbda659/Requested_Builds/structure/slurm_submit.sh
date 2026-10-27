#!/bin/bash
#SBATCH --chdir=/website/userdata/sequence/cb/Builds/62437030-5565-40e8-a66d-451dc5643a0a/New_Builds/structure
#SBATCH --error=slurm_%x-%A.err
#SBATCH --get-user-env
#SBATCH --job-name=Glycan-62437030-
#SBATCH --nodes=1
#SBATCH --output=slurm_%x-%A.out
#SBATCH --partition=amber
#SBATCH --tasks-per-node=4

/website/userdata/sequence/cb/Builds/62437030-5565-40e8-a66d-451dc5643a0a/New_Builds/structure/Minimize.bash
