#!/bin/bash
#SBATCH --job-name=ex1
#SBATCH -p ippd-cpu
#SBATCH --output=out_sendrecv_%j.out
#SBATCH --error=out_sendrecv_%j.err
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=1
#SBATCH --threads-per-core=1
#SBATCH --time=00:00:30

module load GCC/10.2.0
module load OpenMPI/4.1.2-GCC-10.2.0-with-slurm

make >> make.out || exit 1      # Exit if make fails

srun sendrecvp
