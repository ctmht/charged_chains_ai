import os

from simulations_manager import SimulationsManager


DATA_FOLDER = os.path.abspath("./data/")
print(DATA_FOLDER)

SIGNAC_FOLDER = os.path.join(os.getcwd(), "signac")
print(SIGNAC_FOLDER)

simman = SimulationsManager(path = SIGNAC_FOLDER)
simman.init_project(path = SIGNAC_FOLDER)

# for job in simman:
#       job.remove()

#for taskname in ["autocorr", "full"]:
        # dfpath = os.path.join(DATA_FOLDER, f"{taskname}_dataframe.pkl")
        # simman.create_jobs_simulations(taskname, dfpath)

CONFIG_FOLDER = os.path.abspath(os.path.join('signac', 'mltraining_scripts', 'configs'))
print(CONFIG_FOLDER)

for fname in os.listdir(CONFIG_FOLDER):
    if 'config' in fname and 'test' not in fname:
        configpath = os.path.join(CONFIG_FOLDER, fname)
        simman.create_jobs_mltraining(configpath)