import sys
import os

import pandas as pd

from simulations_manager import SimulationsManager


REPLACE = False
try:
    if sys.argv[1] == 1:
        REPLACE = True
        print(f"Results of this run will replace old ones ({REPLACE=})")
except:
    pass


DATA_FOLDER = os.path.abspath("./data/")
print(DATA_FOLDER)

SIGNAC_FOLDER = os.path.join(os.getcwd(), "signac")
print(SIGNAC_FOLDER)

simman = SimulationsManager(path = SIGNAC_FOLDER)
simman.init_project(path = SIGNAC_FOLDER)

# Dictionary to collect results by task
results_by_task = {}


for job in simman:
	# Check if job is completed (has the postprocessed file)
    if job.isfile("9_processed.pkl"):
        taskname = job.sp.taskname
        
		# Load the job's results
        job_results_path = job.fn("9_processed.pkl")
        job_df = pd.read_pickle(job_results_path)
        
        # Add job metadata
        job_df['job_id'] = job.id
        job_df['statepoint'] = str(job.sp)
        
        # Append to task-specific results
        if taskname not in results_by_task:
            results_by_task[taskname] = []
        results_by_task[taskname].append(job_df)


# Append to existing centralized dataframes for each task
for taskname, job_dfs in results_by_task.items():
    if job_dfs:  # Only if there are new completed jobs
        combined_df = pd.concat(job_dfs, ignore_index=True)
        output_path = os.path.join(DATA_FOLDER, f"{taskname}_results.pkl")
        
        # Load existing results if they exist
        if os.path.exists(output_path) and not REPLACE:
            existing_df = pd.read_pickle(output_path)
            # Combine existing and new results, removing duplicates by job_id
            final_df = pd.concat([existing_df, combined_df], ignore_index=True)
            # Remove duplicates - keep the newest entry if same job appears again
            final_df = final_df.drop_duplicates(subset=['job_id'], keep='last')
        else:
            final_df = combined_df
        
        final_df.to_pickle(output_path)
        print(f"Updated {taskname} results: {len(combined_df)} new jobs, "
              f"total {len(final_df)} jobs in {output_path}")