### EXAMPLE SCRIPT FOR THE SDCC ####
### This script calculates the relaxation time of a spherical magnetite grain 60nm in size ###
### This may take several minutes to run, depending on the speed of your computer ###

from sdcc.simulation import relaxation_time, parallelized_mono_dispersion
import pickle
import os
import matplotlib.pyplot as plt
from sdcc.plotting import plot_relax_experiment
import numpy as np

# Parallelized scripts MUST be run inside an "if __name__=='__main__'" statement
if __name__ == "__main__":
    # Path to installation particles file, replace with yours
    particle_path = "../particles/thermal/"

    # Path to a magnetite particle with elongation 1.00 along the hard axis
    particle = particle_path + "TM00_PRO_1.00_OBL_1.00_hard.gel"

    with open(particle, "rb") as f:
        thermParticle = pickle.load(f)

    # Set up treatment routine
    B_dir = [1, 0, 0]  # Field direction - along x here
    B_str = 40  # Field strength - 40uT
    relax_routine = relaxation_time(thermParticle, B_dir, B_str)

    # Set up particle size
    size = 60  # size in nm

    # get number of available slurm cores using this:

    # Run simulation for mono-dispersion of grains
    starting_prob = np.full(8, 1 / 8)  # Starting probability of being in each state
    moments, probabilities = parallelized_mono_dispersion(
        starting_prob, size, relax_routine, thermParticle, n_dirs=30, ctx="spawn"
    )
    # Plot Results
    plot_relax_experiment(moments, relax_routine)
    plt.savefig("relax_test.png")
    plt.close()
