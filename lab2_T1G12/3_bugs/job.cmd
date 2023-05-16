#!/bin/bash
#SBATCH --job-name=ex1
#SBATCH -p ippd-cpu
#SBATCH --output=out_bugs_%j.out
#SBATCH --error=out_bugs_%j.err
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --time=00:00:30

module load GCC/10.2.0
module load OpenMPI

make bug3 >> make.out || exit 1      # Exit if make fails

mpirun -np 2 bug3
