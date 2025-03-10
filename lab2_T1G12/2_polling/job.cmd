#!/bin/bash
#SBATCH --job-name=ex2
#SBATCH -p ippd-cpu
#SBATCH --output=out_polling_%j.out
#SBATCH --error=out_polling_%j.err
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --time=00:00:30

module load GCC/10.2.0
module load OpenMPI/4.1.2-GCC-10.2.0-with-slurm

make >> make.out || exit 1      # Exit if make fails

srun count
