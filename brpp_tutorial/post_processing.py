
import freud  # Used for advanced analysis of molecular dynamics and other simulations
import numpy as np  # Fundamental package for scientific computing in Python
import MDAnalysis as mda  # Used to manipulate and analyse molecular dynamics trajectories


def main():
    # File paths for the topology and trajectory files
    topology = 'final.data'  # The topology file containing structural information of the system
    trajectory_lin = 'traj.lin'  # The linear trajectory file from MD simulation
    trajectory_log = 'traj.log'  # The logarithmic trajectory file from MD simulation

    # Time step size used in the MD simulations (in reduced units)
    DT_INTEGRATION = 0.005

    # Creating Universe objects for linear and logarithmic trajectories with MDAnalysis
    # These objects represent the entire system and allow for analysis of the data
    u_log = mda.Universe(topology, trajectory_log, format='LAMMPSDUMP', dt=DT_INTEGRATION)
    u_lin = mda.Universe(topology, trajectory_lin, format='LAMMPSDUMP', dt=DT_INTEGRATION)

    # Calculating the number of frames and beads (particles) in the simulations
    N_FRAMES_LOG = u_log.trajectory.n_frames  # Total number of frames in the logarithmic trajectory
    N_FRAMES_LIN = u_lin.trajectory.n_frames  # Total number of frames in the linear trajectory
    N_BEADS = u_lin.atoms.n_atoms  # Total number of beads (atoms) in the system

    # Initialise an array to store the radius of gyration for each frame
    radius_of_gyration = np.zeros(N_FRAMES_LIN)
    end_to_end_distance = np.zeros(N_FRAMES_LIN)
    # Initialise an array to store the simulation times for each frame
    times_lin = np.zeros(N_FRAMES_LIN)
    # Iterate over each frame in the linear trajectory
    for i, _ in enumerate(u_lin.trajectory):
        # Compute and store the radius of gyration for the current frame
        radius_of_gyration[i] = u_lin.atoms.radius_of_gyration()
        # Store the simulation time for the current frame
        times_lin[i] = u_lin.trajectory.ts.data['time']
        # End-to-end distance calculation
        end_to_end_distance[i] = np.linalg.norm(u_lin.atoms.positions[0] - u_lin.atoms.positions[-1])

    # Initialise an array to store the positions of beads for each frame
    bead_positions = np.zeros((N_FRAMES_LOG, N_BEADS, 3))
    # Initialise an array to store the simulation times for each frame
    times_log = np.zeros(N_FRAMES_LOG)
    # Iterate over each frame in the logarithmic trajectory to read the positions and times
    for i, _ in enumerate(u_log.trajectory):
        # Copy and store the positions of atoms for the current frame
        bead_positions[i] = u_log.atoms.positions.copy()
        # Store the simulation time for the current frame
        times_log[i] = u_log.trajectory.ts.data['time']

    # Define a cubic box based on the dimensions of the simulation box
    box = freud.Box.cube(u_log.dimensions[0])
    # Compute the mean squared displacement (MSD) for the logged positions
    mean_squared_displacement, _ = compute_msd(box, bead_positions)

    # Save the radius of gyration to a text file
    np.savetxt('rad_gyr.txt', radius_of_gyration)
    # Save the end-to-end distance to a text file
    np.savetxt('end_to_end_distance.txt', end_to_end_distance)
    # Save the mean squared displacement to a text file
    np.savetxt('msd.txt', mean_squared_displacement)
    # Save the times corresponding to each frame to a text file
    np.savetxt('times_lin.txt', times_lin)
    np.savetxt('times_log.txt', times_log)


def compute_msd(box, positions, mode='direct', plane='full'):
    """
    Computes the mean squared displacement (MSD) for a set of positions within a simulation box.

    Parameters
    ----------
    box : list
        The dimensions of the simulation box as a freud box item ([Lx, Ly, Lz]). This defines the boundary conditions.
    positions : np.ndarray
        A numpy array of particle positions with the shape (trajectory_length, number_of_particles, 3), representing the x, y, z coordinates over time.
    mode : str, optional
        The mode of MSD calculation. 'direct' computes MSD for the entire trajectory, while 'window' computes it within a moving window for time-averaging. Default is 'direct'.
    plane : str, optional
        Specifies the plane ('full', 'yz', 'xy', 'xz') over which the MSD is calculated. 'full' calculates the 3D MSD, while the other options calculate 2D MSD in specified planes. Default is 'full'.

    Returns
    -------
    MSD : np.ndarray
        A numpy array containing the computed mean squared displacement for each timestep in the trajectory.
    """
    msd = freud.msd.MSD(box=box, mode=mode)
    tmp = positions.copy()  # don't modify the positions 
    if plane == 'full':

        msd.compute(tmp)
    elif plane == 'yz':

        tmp[:,:,0] = 0.0
        msd.compute(tmp)
    elif plane == 'xz':

        tmp[:,:,1] = 0.0
        msd.compute(tmp)
    elif plane == 'xy':

        tmp[:,:,2] = 0.0
        msd.compute(tmp)
    else:
        raise ValueError("Use one of the options: 'full', 'xy', 'yz', 'xz' ")
    
    MSD = msd.msd
    MSD_per_particle = msd.particle_msd

    return MSD, MSD_per_particle

if __name__ == '__main__':
    main()
