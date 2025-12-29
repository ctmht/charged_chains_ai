import argparse
import sys
import os

import pandas as pd
import numpy as np
import h5py

from simulations_manager import SimulationsManager

def parse_arguments():
    """Parse command line arguments for the data collection script."""
    parser = argparse.ArgumentParser(
        description = 'Collect signac simulation results into HDF5 dataset.',
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
Examples:
  %(prog)s                    # Append new jobs for all tasknames
  %(prog)s -r -t full         # Replace dataset with only 'full' task jobs
  %(prog)s -t autocorr        # Append only 'autocorr' task jobs
        """
    )
    
    parser.add_argument('-r', '--replace', 
                       action = 'store_true',
                       help = 'replace old dataset (sets REPLACE=True)')
    
    parser.add_argument('-t', '--taskname', 
                       choices = ['autocorr', 'full'],
                       help = "filter jobs by taskname (sets task=taskname)")
    
    return parser.parse_args()


def old_collect_autocorr():
    # Dictionary to collect results by task
    results = {}


    for job in simman.find_jobs({'taskname': 'autocorr'}):
        # Check if job is completed (has the postprocessed file)
        if not job.isfile("9_processed.pkl"):
            continue
        
        # Load the job's results
        job_results_path = job.fn("9_processed.pkl")
        job_df = pd.read_pickle(job_results_path)
        
        # Add job metadata
        job_df['job_id'] = job.id
        job_df['statepoint'] = str(job.sp)
        
        # Append to task-specific results
        if taskname not in results:
            results[taskname] = []
        results[taskname].append(job_df)


    # Append to existing centralized dataframes for each task
    for taskname, job_dfs in results.items():
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


# Define the data structure and types
data_structure = {
    'indexers/job_id': ('S100', ()),
    'indexers/sequence': ('S100', ()),
    'sequence_descriptors/counts/count_aliphatic_A': ('int64', ()),
    'sequence_descriptors/counts/count_anion_C': ('int64', ()),
    'sequence_descriptors/blockinesses/blockiness_A': ('float64', ()),
    'sequence_descriptors/blockinesses/blockiness_B': ('float64', ()),
    'sequence_descriptors/blockinesses/blockiness_C': ('float64', ()),
    'sequence_descriptors/blockinesses/blockiness_D': ('float64', ()),
    'shape_descriptors/gyr_tensor_oevals/gyr_tensor_oevals_perfm': ('float64', (3,)),
    'shape_descriptors/gyr_tensor_oevals/gyr_tensor_oevals_perfv': ('float64', (3, 3)),
}

# Add nbc_perbm datasets
for bead in ['A', 'B', 'C', 'D']:
    data_structure[f'shape_descriptors/nbc_perbm/nbc_perbm_{bead}'] = ('float64', ())

# Add statistics datasets
for stat in ['radgyr2', 'aspher', 'acylin', 'relsha']:
    for mode in ['perfm', 'perfs', 'propm', 'props']:
        dataset_name = f'shape_descriptors/gyr_tensor_oevals/{stat}/{stat}_{mode}'
        data_structure[dataset_name] = ('float64', ())


def ensure_h5_structure(h5f):
    """
    Create groups and datasets if they don't exist
    """
    for path, (dtype, shape) in data_structure.items():
        if path not in h5f:
            # Create parent groups
            parts = path.split('/')
            for i in range(1, len(parts)):
                group_path = '/'.join(parts[:i])
                if group_path and group_path not in h5f:
                    h5f.create_group(group_path)
            
            # Create dataset with resizable dimension
            full_shape = (0,) + shape
            maxshape = (None,) + shape
            h5f.create_dataset(path, shape=full_shape, maxshape=maxshape, dtype=dtype)



def job_exists_in_h5(h5f, job_id):
    """
    Check if a job ID already exists in the HDF5 file
    """
    if 'indexers/job_id' not in h5f or h5f['indexers/job_id'].shape[0] == 0:
        return False
    
    existing_ids = h5f['indexers/job_id'][:]
    # Convert bytes to string for comparison
    if isinstance(existing_ids[0], bytes):
        existing_ids = [id.decode('utf-8') for id in existing_ids]
    
    return job_id in existing_ids



def append_job_to_h5(h5f, job_id, job_data):
    """
    Append a single job's data to the HDF5 file
    """
    # Get current size
    current_size = h5f['indexers/job_id'].shape[0]
    
    # Resize all datasets
    for path in data_structure.keys():
        dataset = h5f[path]
        new_shape = list(dataset.shape)
        new_shape[0] = current_size + 1
        dataset.resize(new_shape)
    
    # Fill data for each dataset
    for path, (dtype, shape) in data_structure.items():
        dataset = h5f[path]
        
        # Extract the last part of the path as the column name
        col_name = path.split('/')[-1]
        
        if col_name == 'job_id':
            dataset[current_size] = np.bytes_(job_id).astype('S100')
        elif col_name in job_data:
            value = job_data[col_name]
            if isinstance(value, np.ndarray):
                dataset[current_size] = value
            else:
                dataset[current_size] = value
        else:
            # For grouped datasets, extract from nested dict
            parts = path.split('/')
            if len(parts) == 4 and parts[-2] in ['counts', 'blockinesses', 'nbc_perbm']:
                # e.g., sequence_descriptors/counts/count_aliphatic_A
                group_type = parts[-2]
                field_name = parts[-1]
                
                if group_type == 'counts' and field_name in job_data:
                    dataset[current_size] = job_data[field_name]
                elif group_type == 'blockinesses' and field_name in job_data:
                    dataset[current_size] = job_data[field_name]
                elif group_type == 'nbc_perbm' and field_name in job_data:
                    dataset[current_size] = job_data[field_name]
            
            elif len(parts) == 5 and parts[-3] == 'gyr_tensor_oevals':
                # e.g., shape_descriptors/gyr_tensor_oevals/radgyr2/radgyr2_perfm
                stat_name = parts[-2]
                mode_name = parts[-1]
                col_key = f"{stat_name}_{mode_name.split('_')[-1]}"
                if col_key in job_data:
                    dataset[current_size] = job_data[col_key]



if __name__ == '__main__':
    args = parse_arguments()
    REPLACE = args.replace
    TASK = args.taskname
    
    print(f"Running collector with {REPLACE=} and {TASK=}")

    DATA_FOLDER = os.path.abspath("./data/")
    print(DATA_FOLDER)

    SIGNAC_FOLDER = os.path.join(os.getcwd(), "signac")
    print(SIGNAC_FOLDER)


    simman = SimulationsManager(path = SIGNAC_FOLDER)
    simman.init_project(path = SIGNAC_FOLDER)
    
    if TASK not in ['autocorr', 'full']:
        print(f"Taskname {TASK=} not supported, no action will be taken."
               "Use option -h for example usage")
    elif TASK == 'autocorr':
        old_collect_autocorr()
    elif TASK == 'full':
        output_h5_path = os.path.join(DATA_FOLDER, f"{TASK}_results.h5")
        
        # Main processing
        with h5py.File(output_h5_path, 'a' if not REPLACE else 'w') as h5f:
            # Ensure the structure exists
            ensure_h5_structure(h5f)
            
            # Track processed jobs
            processed_count = 0
            skipped_count = 0
            
            for job in simman.find_jobs({'taskname': 'full'}):
                # Check if job is completed and matches task filter
                if not job.isfile("9_processed.pkl"):
                    continue
                    
                taskname = job.sp.taskname
                if TASK is not None and TASK != taskname:
                    continue
                
                # Check if job already exists in HDF5
                if not REPLACE and job_exists_in_h5(h5f, job.id):
                    print(f"Job {job.id[:8]}... already exists, skipping")
                    skipped_count += 1
                    continue
                
                # Load job results
                job_df = pd.read_pickle(job.fn("9_processed.pkl"))
                job_data = job_df.iloc[0].to_dict()
                
                # Add job metadata
                job_data['job_id'] = job.id
                
                # Append to HDF5
                append_job_to_h5(h5f, job.id, job_data)
                processed_count += 1
                
                # Print progress every 100 jobs
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} jobs...")
            
            # Final summary
            total_jobs = h5f['indexers/job_id'].shape[0]
            print(f"\nCollection complete:")
            print(f"  Processed: {processed_count} new jobs")
            print(f"  Skipped: {skipped_count} existing jobs")
            print(f"  Total in HDF5: {total_jobs} jobs")
            print(f"  Saved to: {output_h5_path}")