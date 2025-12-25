{% extends "base_script.sh" %}
{% block header %}
#!/bin/bash
#SBATCH --job-name="{{ id }}"
#SBATCH --output=slurm_outs/slurm-run_simulation_job-%j.out
#SBATCH --time=00:125:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=omni36

source /etc/profile.d/modules.sh

module --ignore-cache load LAMMPS/23Jun2022-foss-2021b-kokkos
module unload SciPy-bundle/2021.10-foss-2021b
module --ignore-cache load Anaconda3/2023.03-1
conda activate myenv

{% endblock %}