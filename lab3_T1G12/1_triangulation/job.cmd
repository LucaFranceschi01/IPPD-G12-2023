#!/bin/bash
#SBATCH --job-name=ex1
#SBATCH -p ippd-cpu
#SBATCH --output=out_triangulation_%j.out
#SBATCH --error=out_triangulation_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=1
#SBATCH --threads-per-core=1
#SBATCH --time=00:00:30

module load GCC/10.2.0

make >> make.out || exit 1      # Exit if make fails

srun delaunay
