#!/bin/bash
#SBATCH --chdir=/website/userdata/sequence/cb/Builds/76e2c72c-5097-44c9-a435-154446df2500/New_Builds/structure
#SBATCH --error=slurm_%x-%A.err
#SBATCH --get-user-env
#SBATCH --job-name=Glycan-76e2c72c-
#SBATCH --nodes=1
#SBATCH --output=slurm_%x-%A.out
#SBATCH --partition=amber
#SBATCH --tasks-per-node=4

/website/userdata/sequence/cb/Builds/76e2c72c-5097-44c9-a435-154446df2500/New_Builds/structure/Minimize.bash
