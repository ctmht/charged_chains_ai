{% extends "base_script.sh" %}
{% block header %}
#!/bin/bash
#SBATCH --job-name="{{ id }}"
#SBATCH --output=slurm_outs/slurm-job-%j.out
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

module load LAMMPS/23Jun2022-foss-2021b-kokkos
module unload SciPy-bundle/2021.10-foss-2021b
module load Anaconda3/2023.03-1
conda activate myenv

{% endblock %}