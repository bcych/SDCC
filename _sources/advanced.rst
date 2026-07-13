Advanced Settings
=============
This module goes over some advanced settings that can help with the sdcc.

Parallelization
------------
Parallelization can be called through the `sdcc.simulation.parallelized_mono_dispersion` and `sdcc.simulation.parallelized_hyst_mono_dispersion` functions. Parallelization in python is complicated and often depends on your exact setup. It will often not work inside a jupyter notebook. For surefire parallelization, use the `ctx = "spawn"` argument and contain your function inside an `if __name__ == "__main__":` block. Here is an example script tgat goes over the relaxation time result in the previous example.

.. code-block:: python3
   
   from sdcc.particles import load_particle
   from sdcc.simulation import parallelized_mono_dispersion
   from sdcc.treatment import relaxation_time

   # Read in GEL file for particle
   thermParticle = load_particle('../particles/thermal/TM00_PRO_2.50_OBL_1.00_hard.gel')

   # Set up treatment routine
   B_dir = [1,0,0] #Field direction - along x here
   B_str = 40 #Field strength - 40uT
   relax_routine = relaxation_time(thermParticle, B_dir, B_str)

   # Set up particle size
   size = 21 #size in nm 
   
   with __name__ == "__main__":
       starting_prob = [1/2, 1/2] 
       moments, probabilities = parallelized_mono_dispersion(starting_prob, 
                                                            size, 
                                                            relax_routine, 
                                                            thermParticle,
                                                            n_dirs = 30,
                                                            ctx = "spawn")

The number of cpus can additionally be set using the `cpu_count` argument. By default, the parallelized functions run using the maximum number of available jobs. Setting `cpu_count` can be useful when, for example, setting up jobs on a slurm HPC queue system. Set the `--cpus-per-task` flag to an integer value in your slurm job, then set `cpu_count = int(os.environ["SLURM_CPUS_PER_TASK"])`.

Initializing Mono-Dispersions
------------
Models can be initialized using a vector of probabilities for the starting probabilities of being in each state. This vector must be of length n (the number of states, which can be determined by the length of e.g. `EnergyLandscape.get_params(T)["min_e"]`). However, a mono_dispersion may contain a set of m particles in different orientations. If you want to use the output of a previous model as the starting input to a new model, then you need to have a separate vector for each particle orientation. To do this, you can simply pass an m x n array to the relevant `mono_dispersion` function, and it will assign the correct vector to each particle in the dispersion.

Custom Treatment Routines
------------
All treatment steps in the sdcc inherit from the `sdcc.treatment.TreatmentStep` class. This class needs to be initialized with four arrays: ts (times), Ts (temperatures), field_dirs (Field Directions) and field_str (Field Strengths). There are several examples of custom treatment step classes already implemented into the sdcc (e.g. `sdcc.treatment.CoolingStep` and `sdcc.treatment.HeatingStep`) as well as full experimental protocols (`sdcc.treatment.thellier_experiment`). For more accurate results, ensure sure there are small changes in temperature / field between each time step. Large jumps in time are completely acceptable as these are integrated over to high floating point precision by the kinetic model in the SDCC. 

Custom Particles (New in v1.0)
------------
The SDCC has various particle types that inherit from the `sdcc.particles.EnergyLandscape` base class. There are several examples in the `sdcc.particles` moduleshowing how to construct an `EnergyLandscape` type class for your input data. The functions in the `sdcc.simulation` module are now no longer restricted to single domain particles - this is shown in the non single-domain `sdcc.particles.SMELT` class.

Converting From Pre v1.0 Particle Files
------------
Version 1.0.0 contains `particles.OldGEL` and `particles.OldHEL` classes that allow users to convert particles from versions 0.4.3 and earlier to the new sdcc format. Please update any previously generated particles you were using when running the updated version of the code. The SDCC is fully backwards compatible once these are converted.
