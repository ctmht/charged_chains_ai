from typing import Literal
import shutil
import json
import os

from flow import FlowProject
import pandas as pd
import signac

from mltraining_scripts.training import load_configs


class SimulationsManager(FlowProject):
    """ Signac flow for management of LAMMPS simulations """

    def __init__(
        self,
        path: str | None = None,
        job_limit: int = 20_000
    ):
        """
        Initialize an object for general LAMMPS simulation management
        """
        self.job_limit = job_limit
        
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
        if len(self) >= self.job_limit:
            return

        df: pd.DataFrame = pd.read_pickle(dataframe_path)
        
        for idx, row in df.iterrows():
            if row["used"] == 1:
                continue
            
            statepoint = {
                "sequence": row["sequence"],
                "taskname": taskname
            }
            
            job = self.open_job(statepoint)
            
            if job in self:
                continue
            
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
            
            df.loc[idx]["used"] = 1
            
            if len(self) % 50 == 0:
                print(f"Created job {job.id} for sequence {job.sp.sequence}, count {len(self)}")

            if len(self) == self.job_limit:
                break
        
        df.to_pickle(dataframe_path)
    

    def create_jobs_mltraining(
        self,
        config_filepath: str
    ) -> None:
        """
        Creates statepoints and then jobs based on the information found in the dataframes
        """
        config_path = os.path.abspath(config_filepath)
        configs = load_configs(path = config_path)
        
        for config in configs:
            statepoint = {
                'taskname': 'ml_training',
                'config': config
            }
            
            job = self.open_job(statepoint)
            
            job_config_json = os.path.abspath(os.path.join(job.path, 'config.json'))
            json.dump(config, open(job_config_json, 'w'))
            
            if job in self:
                continue
            
            # Initialize job
            job.init()
        




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
    # Find Python script for running a training configuration
    jpp = os.path.abspath(job.project.path)
    postprocess_fname = f"training.py"
    python_script_location = os.path.join(jpp, postprocess_fname)
    
    # Set up shell command to run script
    run_py_script = f"python {python_script_location} {job.sp.taskname} {job.path}"
    command_to_run = run_py_script
    
    print(f"signac job {job.id[:7]}..: Running `8_postprocessing.py {job.sp.taskname} {job.path}' "
          f"to postprocess results")
    
    return command_to_run


##############################
# Labels for ML training
##############################
@SimulationsManager.label
def training_completed(job):
    return job.isfile("final.json")

##############################
# Operations for ML training
##############################
@SimulationsManager.post(training_completed)
@SimulationsManager.operation(cmd = True, with_job = True)
def train_transformer_config(
    job: signac.job.Job
):
    """
    Run Python training script
    """
    # Find Python script for running a training configuration
    jpp = os.path.abspath(job.project.path)
    training_fname = f"training.py"
    python_script_location = os.path.join(jpp, training_fname)
    
    job_config_json = os.path.abspath(os.path.join(job.path, 'config.json'))
    job_final_json = os.path.abspath(os.path.join(job.path, 'final.json'))
    
    # Set up shell command to run script
    run_py_script = f"python {python_script_location} {job_config_json} {job_final_json}"
    command_to_run = run_py_script
    
    print(f"signac job {job.id[:7]}..: Running `training.py {job_config_json} {job_final_json}' "
          f"to train model")
    
    return command_to_run




if __name__=="__main__":
    usedpath = os.path.dirname(os.path.abspath(__file__))
    print("Directory absolute path:", usedpath)
    simman = SimulationsManager(usedpath)
    simman.main()
    