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
        
        # Reorganize workspace for easier task separation
        self.config['workspace_dir'] = "workspace/{sp.taskname}/{id}"
    
    
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
                template_fld = os.path.join(jpp, f"simulation_prototype")
                job_dest_fld = job.path
                
                for f in os.listdir(template_fld):
                    if os.path.isfile(os.path.join(template_fld, f)):
                        # Only copy the task-specific LAMMPS script:
                        # 4_LAMMPS_mnr_autocorr.in or 4_LAMMPS_mnr_full.in
                        if f[0] == '4' and taskname not in f:
                            continue
                        
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
    # Find Python script for molecule file creation
    jpp = os.path.abspath(job.path)
    python_script_location = os.path.join(jpp, "0_create_molecule.py")
    
    # Set up shell command to run script
    sequence = job.sp.sequence
    run_py_script = f"python {python_script_location} {sequence}"
    command_to_run = run_py_script
    
    print(f"signac job {job.id[:7]}..: Running '0_create_molecule.py' to generate a "
          f"chain with sequence '{sequence}'")
    
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
    # Find LAMMPS script for creating the environment from the molecule file
    jpp = os.path.abspath(job.path)
    lammps_script_location = os.path.join(jpp, "2_LAMMPS_creation.in")
    
    # Set up shell command to run script
    run_creation = f"lmp -in {lammps_script_location}"
    lammps_flags = "-log 6_a_log.lammps"
    command_to_run = run_creation + ' ' + lammps_flags
    
    sequence = job.sp.sequence
    print(f"signac job {job.id[:7]}..: Running '2_LAMMPS_creation' to create the LAMMPS "
          f"environment for sequence '{sequence}'")
    
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
    # Find LAMMPS script for running minimization procedure and the simulation
    jpp = os.path.abspath(job.path)
    lammps_script_location = os.path.join(jpp, f"4_LAMMPS_mnr_{job.sp.taskname}.in")
    
    # Set up shell command to run script
    run_simulation = f"lmp -in {lammps_script_location}"
    lammps_flags = "-log 6_a_log.lammps"
    command_to_run = run_simulation + ' ' + lammps_flags
    
    sequence = job.sp.sequence
    print(f"signac job {job.id[:7]}..: Running '4_LAMMPS_mnr_{job.sp.taskname}' to run the "
          f"simulation for sequence '{sequence}'")
    
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
    # Find Python script for postprocessing
    jpp = os.path.abspath(job.path)
    postprocess_fname = f"8_postprocessing.py"
    python_script_location = os.path.join(jpp, postprocess_fname)
    
    # Set up shell command to run script
    run_py_script = f"python {python_script_location} {job.sp.taskname}"
    command_to_run = run_py_script
    
    print(f"signac job {job.id[:7]}..: Running `8_postprocessing.py {job.sp.taskname}' "
          f"to postprocess results")
    
    return command_to_run


##############################
# Labels for ML training
##############################


##############################
# Operations for ML training
##############################





if __name__=="__main__":
    usedpath = os.path.dirname(os.path.abspath(__file__))
    print("Directory absolute path:", usedpath)
    simman = SimulationsManager(usedpath)
    simman.main()
    