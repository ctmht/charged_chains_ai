{% extends "base_script.sh" %}
{% block header %}
#!/bin/bash
#SBATCH --job-name="{{ id }}"
#SBATCH --time=00:10:00
#SBATCH --partition=regularshort
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

module load LAMMPS/23Jun2022-foss-2021b-kokkos
module unload SciPy-bundle/2021.10-foss-2021b

{% endblock %}
