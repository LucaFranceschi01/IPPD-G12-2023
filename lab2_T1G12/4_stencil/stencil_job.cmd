#!/bin/bash
#SBATCH --job-name=ex1
#SBATCH -p ippd-cpu
#SBATCH --output=out_stencil_%j.out
#SBATCH --error=out_stencil_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --tasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=00:00:30

module load GCC/10.2.0
module load OpenMPI/4.1.2-GCC-10.2.0-with-slurm

# make all >> make.out || exit 1      # Exit if make fails

srun stencil 4096 10 1000 8 8
