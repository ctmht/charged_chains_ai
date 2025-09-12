#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=10:00
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=256mb

module load LAMMPS/23Jun2022-foss-2021b-kokkos
srun lmp -in in.single_chain > out.dat
