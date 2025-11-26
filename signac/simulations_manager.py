from typing import Literal
import shutil
import os

from flow import FlowProject
import pandas as pd
import signac


class SimulationsManager(FlowProject):
    """ Signac flow for management of LAMMPS simulations """
    
    def __init__(
        self,
        path: str | None = None
    ):
        """
        Initialize an object for general LAMMPS simulation management
        """
        super().__init__(
            path,
            environment = None,
            entrypoint = None
        )
    
    
    def create_jobs_simulations(
        self,
        taskname: Literal["autocorr", "full"],
        dataframe_path: str
    ) -> None:
        """
        Creates statepoints and then jobs based on the information found in the dataframes
        """
        df: pd.DataFrame = pd.read_pickle(dataframe_path)
        
        for idx, row in df.iterrows():
            statepoint = row.to_dict()
            
            statepoint["idx"] = idx
            statepoint["taskname"] = taskname
            
            # TODO: statepoint["expected_outputs"]?
            
            job = self.open_job(statepoint)
            
            if job not in self:
                # Initialize job
                job.init()
                
                # Copy task-specific template folder into the job's workspace
                jpp = os.path.abspath(job.project.path)
                template_fld = os.path.join(jpp, f"simulation_prototype_{taskname}")
                job_dest_fld = job.path
                
                for f in os.listdir(template_fld):
                    if os.path.isfile(os.path.join(template_fld, f)):
                        template_filepath = os.path.join(template_fld, f)
                        target_filepath = os.path.join(job_dest_fld, f)
                        
                        shutil.copyfile(template_filepath, target_filepath)
    
    
    def create_jobs_mltraining(
        self,
        dataframe_path: str
    ) -> None:
        """
        Creates statepoints and then jobs based on the information found in the dataframes
        """
        df: pd.DataFrame = pd.read_pickle(dataframe_path)
        
        for idx, row in df.iterrows():
            statepoint = row.to_dict()
            
            statepoint["idx"] = idx
            statepoint["taskname"] = "ml_training"




##############################
# Labels for simulations
##############################

@SimulationsManager.label
def polymer_created(job):
    return job.isfile("1_polymer.mol")

@SimulationsManager.label
def environment_created(job):
    return job.isfile("3_created.data")

@SimulationsManager.label
def polymer_assembled(job):
    return job.isfile("7_assembled.data")

@SimulationsManager.label
def simulation_postprocessed(job):
    return job.isfile("9_processed.pkl")


##############################
# Operations for simulations
##############################

@SimulationsManager.post(polymer_created)
@SimulationsManager.operation(cmd = True, with_job = True)
def create_polymer(
    job: signac.job.Job
):
    """
    Create a polymer with the sequence given by this job
    """
    # Find molecule file creation script
    jpp = os.path.abspath(job.project.path)
    create_mol_fname = f"simulation_prototype_{job.sp.taskname}/0_create_molecule.py"
    python_script_location = os.path.join(jpp, create_mol_fname)
    
    # Set up shell command to run script
    sequence = job.sp.sequence
    command_to_run = f"python {python_script_location} {sequence}"
    
    print(f"\nrunning 0_create_molecule.py on job id {job.id} to generate a chain with sequence {sequence}\n")

    return command_to_run


@SimulationsManager.pre(polymer_created)
@SimulationsManager.post(environment_created)
@SimulationsManager.operation(cmd = True, with_job = True)
def create_lammps_simulation(
    job: signac.job.Job
):
    """
    Run LAMMPS script to create the environment
    """
    # Find LAMMPS creation script and run
    jpp = os.path.abspath(job.project.path)
    create_env_fname = f"simulation_prototype_{job.sp.taskname}/2_LAMMPS_creation.in"
    lammps_script_location = os.path.join(jpp, create_env_fname)
    
    # Set up shell command to run script
    command_to_run = f"srun lmp -screen out.lammps -in {lammps_script_location}"
    
    return command_to_run


@SimulationsManager.pre(environment_created)
@SimulationsManager.post(polymer_assembled)
@SimulationsManager.operation(cmd = True, with_job = True)
def run_simulation(
    job: signac.job.Job
):
    """
    Run LAMMPS script to simulate polymer assembly
    """
    # Find (energy minimization and) running script
    jpp = os.path.abspath(job.project.path)
    create_env_fname = f"simulation_prototype_{job.sp.taskname}/4_LAMMPS_mnr.in"
    lammps_script_location = os.path.join(jpp, create_env_fname)
    
    # Set up shell command to run script
    command_to_run = f"srun lmp -screen out.lammps -in {lammps_script_location}"
    
    return command_to_run


@SimulationsManager.pre(polymer_assembled)
@SimulationsManager.post(simulation_postprocessed)
@SimulationsManager.operation(cmd = True, with_job = True)
def run_postprocessing(
    job: signac.job.Job
):
    """
    Run Python postprocessing script
    """
    # Find postprocessing script
    jpp = os.path.abspath(job.project.path)
    postprocess_fname = f"simulation_prototype_{job.sp.taskname}/8_postprocessing.py"
    python_script_location = os.path.join(jpp, postprocess_fname)
    
    # Set up shell command to run script
    command_to_run = f"python {python_script_location}"
    print(f"\nrunning 8_postprocessing.py on job id {job.id} to postprocess results\n")

    return command_to_run


##############################
# Labels for ML training
##############################


##############################
# Operations for ML training
##############################





if __name__=="__main__":
    simman = SimulationsManager('signac')
    simman.main()
    