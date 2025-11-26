import os

from simulations_manager import SimulationsManager


DATA_FOLDER = os.path.abspath("./data/")
print(DATA_FOLDER)

SIGNAC_FOLDER = os.path.join(os.getcwd(), "signac")
print(SIGNAC_FOLDER)

simman = SimulationsManager(path = SIGNAC_FOLDER)
simman.init_project(path = SIGNAC_FOLDER)

for taskname in ["autocorr", "full"]:
	dfpath = os.path.join(DATA_FOLDER, f"{taskname}_dataframe.pkl")
	simman.create_jobs(taskname, dfpath)
	# simman.create_jobs_mltraining(dfpath)