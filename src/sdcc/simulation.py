import numpy as np
import mpmath as mp
import pickle
import multiprocessing as mpc
import warnings
from sdcc.barriers import GrainEnergyLandscape, GEL, HEL, find_all_barriers
from sdcc.energy import angle2xyz, dir_to_rot_mat, get_material_parms
from sdcc.utils import fib_sphere

mp.prec = 100
mp.mp.prec = 100
from sdcc.treatment import relaxation_time


def change_minima(p_old, old_theta_list, old_phi_list, new_theta_list, new_phi_list):
    """
    DEPRECATED

    Calculates the set of energy minima at one temperature for a grain
    which correspond to the set of energy minima at another temperature.
    """
    old_len = len(old_theta_list)  # number of old minima
    new_len = len(new_theta_list)  # number of new minima

    # Calculate cosine difference between old and new minima
    cos_dist = np.empty((old_len, new_len))
    for i in range(old_len):
        for j in range(new_len):
            xyz_old = angle2xyz(old_theta_list[i], old_phi_list[i])
            xyz_new = angle2xyz(new_theta_list[j], new_phi_list[j])
            cos_dist[i, j] = np.dot(xyz_old, xyz_new)

    # Assign new p vector
    p_new = np.zeros(new_len)
    for j in range(len(p_new)):
        p_new[j] += np.sum(p_old[np.where(cos_dist[:, j] == np.amax(cos_dist[:, j]))])
    p_new /= sum(p_new)
    return p_new


def Q_matrix(params: dict, d, field_dir=np.array([1, 0, 0]), field_str=0.0):
    """
    Constructs a Q matrix of rates of transition between LEM states.

    Inputs
    ------
    params: dictionary
    Dictionary output from a barriers.GEL object - contains states and
    energy barriers.

    d: float
    Equivalent volume spherical diameter of grain (nm).

    field_dir: numpy array
    Unit vector with external field direction.

    field_str: numpy array
    Strength of field (uT)

    Returns
    -------
    Q: numpy array
    Array of transition rates.
    """
    theta_list = params["min_dir"][:, 0]
    phi_list = params["min_dir"][:, 1]
    theta_mat = params["bar_dir"][:, :, 0]
    phi_mat = params["bar_dir"][:, :, 1]
    energy_densities = params["bar_e"]
    T = params["T"]
    Ms = params["Ms"]

    V = 4 / 3 * np.pi * ((d / 2 * 1e-9) ** 3)
    kb = 1.380649e-23
    tau_0 = 1e-9
    tt, pp = np.meshgrid(theta_list, phi_list)
    pp = pp.T
    xyz = angle2xyz(tt, pp)
    xyz *= Ms * V
    xyz_T = angle2xyz(theta_mat, phi_mat)
    xyz_T *= Ms * V
    xyz = xyz_T - xyz

    field_dir = field_dir * field_str * 1e-6

    # Needs to be this shape for dot producting?
    field_mat = np.empty((3, len(theta_list), len(theta_list)))
    for i in range(len(theta_list)):
        for j in range(len(theta_list)):
            field_mat[:, i, j] = field_dir
    field_mat = np.array([field_mat[0].T, field_mat[1].T, field_mat[2].T])

    zeeman_energy = np.sum(xyz * field_mat, axis=0)
    zeeman_energy = zeeman_energy.at[np.isinf(phi_mat)].set(0.0)
    logQ = -(energy_densities.T * V - zeeman_energy) / (kb * (273 + T))

    logQ = np.array(logQ)
    logQ[np.isnan(logQ)] = -mp.inf
    logQ[np.isinf(logQ) & (logQ > 0)] = -mp.inf
    precise_exp = np.vectorize(mp.exp)
    Q = precise_exp(logQ)
    Q /= mp.mpmathify(tau_0)

    # print(Q)
    for i in range(len(theta_list)):
        Q[i, i] = 0.0
        Q[i, i] = -mp.fsum(Q[:, i])
    return Q


def Q_matrix_legacy(
    theta_list,
    phi_list,
    theta_mat,
    phi_mat,
    energy_densities,
    T,
    d,
    Ms,
    field_dir=np.array([1, 0, 0]),
    field_str=0.0,
):
    """
    DEPRECATED

    Legacy version of Q_matrix function.
    """
    V = 4 / 3 * np.pi * ((d / 2 * 1e-9) ** 3)
    kb = 1.380649e-23
    tau_0 = 1e-9
    tt, pp = np.meshgrid(theta_list, phi_list)
    xyz = angle2xyz(tt, pp)
    xyz *= Ms * V
    xyz_T = angle2xyz(theta_mat, phi_mat)
    xyz_T *= Ms * V
    xyz = xyz_T - xyz

    field_dir = field_dir * field_str * 1e-6

    field_mat = np.empty((3, len(theta_list), len(theta_list)))
    for i in range(len(theta_list)):
        for j in range(len(theta_list)):
            field_mat[:, i, j] = field_dir
    field_mat = np.array([field_mat[0].T, field_mat[1].T, field_mat[2].T])

    zeeman_energy = np.sum(xyz * field_mat, axis=0)
    logQ = -(energy_densities.T * V - zeeman_energy) / (kb * (273 + T))

    logQ = np.array(logQ)
    logQ[np.isnan(logQ)] = -mp.inf
    logQ[np.isinf(logQ)] = -mp.inf
    precise_exp = np.vectorize(mp.exp)
    Q = precise_exp(logQ)
    Q /= mp.mpmathify(tau_0)

    # print(Q)
    for i in range(len(theta_list)):
        Q[i, i] = -np.sum(Q[:, i])
    return Q


def _update_p_vector(p_vec, Q, dt):
    """
    Given an initial state vector, a Q matrix and a time, calculates a
    new state vector. This is very slow due to the high floating point
    precision which is required and could probably benefit from a C++
    implementation. Additionally - this is very susceptible to floating
    point errors even with the high precision when dt gets large. Using
    mpmath's Pade approximations is slower than Taylor series and
    doesn't seem to help much. If there's an algorithm that improves
    this it would be extremely helpful as we're dealing with some large
    numbers (age of Solar System) here.

    Inputs
    ------
    p_vec: numpy array
    Vector of relative proportions of grains in each state.

    Q: numpy array
    Rate matrix of transition times between states.

    dt: float
    Amount of time spent in these field conditions/temperature.

    Returns
    -------
    p_vec_new: numpy array
    New state vector after treatment applied.
    """
    # To get the new state vector, we use a matrix exponential.
    dp_dt = mp.expm(mp.matrix(Q) * mp.mpmathify(dt))
    # The old p vector is matrix multiplied with this matrix
    p_vec_new = dp_dt * mp.matrix(p_vec)
    # Convert back to numpy array
    p_vec_new = np.array(p_vec_new, dtype="float64")
    # Sometimes due to floating point errors the sum of the vector isn't
    # Exactly 1 - this will blow up if we don't renormalize.
    p_vec_new /= sum(p_vec_new)
    return p_vec_new


def thermal_treatment_legacy(
    start_t,
    start_p,
    Ts,
    ts,
    d,
    energy_landscape: GrainEnergyLandscape,
    field_strs,
    field_dirs,
    eq=False,
):
    """
    DEPRECATED

    Legacy version of thermal_treatment
    """
    old_T = Ts[0]
    old_min_energies = energy_landscape.min_energies[energy_landscape.Ts == old_T][0]
    old_thetas = energy_landscape.theta_lists[energy_landscape.Ts == old_T][0]
    old_phis = energy_landscape.phi_lists[energy_landscape.Ts == old_T][0]
    old_saddle_thetas = energy_landscape.theta_mats[energy_landscape.Ts == old_T][0]
    old_saddle_phis = energy_landscape.phi_mats[energy_landscape.Ts == old_T][0]
    old_mats = energy_landscape.energy_mats[energy_landscape.Ts == old_T][0]
    rot_mat, k1, k2, Ms = get_material_parms(
        energy_landscape.TMx, energy_landscape.alignment, old_T
    )

    if eq:
        old_p = eq_ps(
            old_thetas,
            old_phis,
            old_min_energies,
            field_strs[0],
            field_dirs[0],
            old_T,
            d,
            Ms,
        )

    else:

        Q = Q_matrix(
            old_thetas,
            old_phis,
            old_saddle_thetas,
            old_saddle_phis,
            old_mats,
            old_T,
            d,
            Ms,
            field_dir=field_dirs[0],
            field_str=field_strs[0],
        )
        old_p = _update_p_vector(start_p, Q, ts[0] - start_t)
    ps = [old_p]
    theta_lists = [old_thetas]
    phi_lists = [old_phis]
    for i in range(1, len(Ts)):
        T = Ts[i]
        dt = ts[i] - ts[i - 1]
        new_thetas = energy_landscape.theta_lists[energy_landscape.Ts == Ts[i]][0]
        new_phis = energy_landscape.phi_lists[energy_landscape.Ts == Ts[i]][0]
        new_min_energies = energy_landscape.min_energies[energy_landscape.Ts == Ts[i]][
            0
        ]
        new_saddle_thetas = energy_landscape.theta_mats[energy_landscape.Ts == Ts[i]][0]
        new_saddle_phis = energy_landscape.phi_mats[energy_landscape.Ts == Ts[i]][0]
        new_mats = energy_landscape.energy_mats[energy_landscape.Ts == Ts[i]][0]

        old_p = change_minima(old_p, old_thetas, old_phis, new_thetas, new_phis)

        rot_mat, k1, k2, Ms = get_material_parms(
            energy_landscape.TMx, energy_landscape.alignment, T
        )

        if eq:
            new_p = eq_ps(
                new_thetas,
                new_phis,
                new_min_energies,
                field_strs[i],
                field_dirs[i],
                T,
                d,
                Ms,
            )

        else:
            Q = Q_matrix(
                new_thetas,
                new_phis,
                new_saddle_thetas,
                new_saddle_phis,
                new_mats,
                T,
                d,
                Ms,
                field_dir=field_dirs[i],
                field_str=field_strs[i],
            )
            new_p = _update_p_vector(old_p, Q, dt)

        ps.append(new_p)

        old_thetas = new_thetas
        old_phis = new_phis
        old_mats = new_mats
        old_p = new_p
        theta_lists.append(new_thetas)
        phi_lists.append(new_phis)

    return (ps, theta_lists, phi_lists)


def thermal_treatment(
    start_t, start_p, Ts, ts, d, energy_landscape: GEL, field_strs, field_dirs, eq=False
):
    """
    Function for calculating the probability of different LEM states in
    a grain during a thermal experiment.

    Inputs
    ------
    start_t: float
    Time at which this experiment step starts

    start_p: numpy array
    Initial state vector

    Ts: numpy array
    Set of temperatures at the times corresponding to ts.

    ts: numpy array
    Time steps at which we calculate the state.

    d: float
    Equivalent volume spherical diameter of grain (nm).

    energy_landscape: barriers.GEL object
    Object describing energy barriers for a particular grain geometry.

    field_strs: numpy array
    Array of field strengths at each time step.

    field_dirs: numpy array
    Array of field directions at each time step.

    eq: bool
    If True, ignore time steps and run magnetization to equilibrium.

    Returns
    -------
    ps: numpy array
    Array of state vectors at each time step

    theta_lists: numpy array
    Magnetization directions at each time step

    phi_lists: numpy array
    Magnetization magnitudes at each time step
    """

    # Get the starting temperature
    old_T = Ts[0]

    # Get the energy barriers and LEM states at this temperature.
    params = energy_landscape.get_params(old_T)
    # If doing equilibrium, run this state to equilibrium.
    if eq:
        old_p = eq_ps(params, field_strs[0], field_dirs[0], d)

    # Otherwise, calculate the Q matrix as normal, calculate the new state
    # vector.

    else:
        Q = Q_matrix(params, d, field_dir=field_dirs[0], field_str=field_strs[0])
        old_p = _update_p_vector(start_p, Q, ts[0] - start_t)  # New state vector
    # Create list of state vectors, place first one in there.
    ps = [old_p]
    # Create list of LEM state directions - put initial ones in there.
    theta_lists = [params["min_dir"][:, 0]]
    phi_lists = [params["min_dir"][:, 1]]

    # Loop through time steps
    for i in range(1, len(ts)):
        # Get time, temperature, LEM states and barriers at each temperature
        T = Ts[i]
        dt = ts[i] - ts[i - 1]
        params = energy_landscape.get_params(T)

        # Again if equilibrium run for infinite time
        if eq:
            new_p = eq_ps(params, field_strs[i], field_dirs[i], d)

        # Otherwise calculate Q matrix and new state vector
        else:
            Q = Q_matrix(params, d, field_dir=field_dirs[i], field_str=field_strs[i])
            new_p = _update_p_vector(ps[-1], Q, dt)

        # Add state vector to list of state vectors
        ps.append(new_p)

        # Do the same for state magnetization directions at this temp.
        theta_list = params["min_dir"][:, 0]
        phi_list = params["min_dir"][:, 1]
        theta_lists.append(theta_list)
        phi_lists.append(phi_list)

    return (ps, theta_lists, phi_lists)


def get_avg_vectors(ps, theta_lists, phi_lists, Ts, rot_mat, energy_landscape, d):
    """
    Obtains the average magnetization vectors for a grain during a
    thermal experiment, given the probabilities and magnetization
    directions of said state.

    Inputs
    ------
    ps: numpy array
    Array of state vectors at each time step.

    theta_lists,phi_lists: numpy arrays
    Arrays of magnetization directions at each time step

    Ts: numpy array
    Array of temperatures at each time step

    rot_mat: numpy array
    Rotation matrix applied to field direction - the inverse of this is
    applied to the states

    energy_landscape: barriers.GEL object
    Object describing energy barriers and LEM states for a particular
    grain geometry. Here it's used to get Ms.

    Returns
    -------
    vs: numpy array
    Array of average magnetization directions at each time step
    """
    vs = []

    # Slightly complicated - the rotation matrix is applied to the field
    # direction, however in most experiments, the grains are rotated,
    # not the field. We correct for this by applying the inverse of the
    # rotation matrix to the grain magnetization states at the end.
    inv_rot = np.linalg.inv(rot_mat)
    # SOMETHING IS WRONG WITH THIS - SHOULD BE DIFFERENT SOMEHOW

    for i in range(len(ps)):
        # Volume of grain
        V = 4 / 3 * np.pi * ((d / 2 * 1e-9) ** 3)

        # Temperature at time Tces
        T = Ts[i]

        # Get Ms material parameter
        Ms = energy_landscape.get_params(T)["Ms"]

        # Get directions associated with states
        theta_list = theta_lists[i]
        phi_list = phi_lists[i]

        # For SD, magnitude is V * Ms
        vecs = angle2xyz(theta_list, phi_list) * ps[i] * Ms * V

        v = np.sum(vecs, axis=1)  # Grain direction
        vs.append(inv_rot @ v)  # Rotate back into constant field direction
    return np.array(vs)


def grain_vectors(
    start_t,
    start_p,
    Ts,
    ts,
    d,
    energy_landscape: GEL,
    grain_dir,
    field_strs,
    field_dirs,
    eq=False,
):
    """
    Gets the state vectors and average magnetization vectors at each
    time step in a thermal treatment for a single direction in a
    mono-dispersion of grains. This calculation is performed for a
    single treatment step - i.e. a single heating or cooling.
    See treatment.TreatmentStep for a full description of this.

        Inputs
    ------
    start_t: float
    Time at which this experiment step starts

    start_p: numpy array
    Initial state vector

    Ts: numpy array
    Set of temperatures at the times corresponding to ts.

    ts: numpy array
    Time steps at which we calculate the state.

    d: float
    Equivalent volume spherical diameter of grain (nm).

    energy_landscape: barriers.GEL object
    Object describing energy barriers for a particular grain geometry.

    grain_dir: numpy array
    Direction associated with this grain.

    field_strs: numpy array
    Array of field strengths at each time step.

    field_dirs: numpy array
    Array of field directions at each time step.

    eq: bool
    If True, ignore time steps and run magnetization to equilibrium.

    Returns
    -------
    vs: numpy array
    Array of average magnetization vectors at each time step

    ps: numpy array
    Array of state vectors at each time step
    """
    # Convert our field directions to a rotation matrix, so that we can
    # rotate back to grain coordinates later.
    grain_dirstr = grain_dir.astype(str)
    grain_dirstr = " ".join(grain_dirstr)
    ref_dir = np.array([1, 0, 0])
    ref_dirstr = ref_dir.astype(str)
    ref_dirstr = " ".join(ref_dirstr)
    rot_mat = dir_to_rot_mat(ref_dirstr, grain_dirstr)

    # Get the field directions rotated according to this matrix.
    rot_field_dirs = []
    for f in field_dirs:
        rot_dir = rot_mat @ f
        rot_field_dirs.append(rot_dir)

    # Get the state vectors at each time step.
    ps, theta_lists, phi_lists = thermal_treatment(
        start_t, start_p, Ts, ts, d, energy_landscape, field_strs, rot_field_dirs, eq=eq
    )

    # Get the average magnetization vectors at each time step
    vs = get_avg_vectors(ps, theta_lists, phi_lists, Ts, rot_mat, energy_landscape, d)
    return (vs, ps)


def mono_direction(grain_dir, start_p, d, steps, energy_landscape: GEL, eq=[False]):
    """
    Gets the state vectors and average magnetization vectors at each
    time step in a thermal treatment for a single direction in a
    mono-dispersion of grains. This calculation is performed for a
    set of treatment steps - see treatment.TreatmentStep for more details.

    Inputs
    ------
    grain_dir: numpy array
    Direction of this grain in the mono dispersion.

    start_p: numpy array
    Initial state vector of grain.

    d: float
    Equivalent volume spherical diameter of grain (nm).

    steps: list of treatment.TreatmentStep objects
    Set of steps that describe a thermal experiment.

    energy_landscape: barriers.GEL object
    Object describing LEM states and energy barriers as a function of
    temperature

    eq: bool
    If True, ignore time steps and run magnetization to equilibrium.

    Returns
    -------
    vs: lists
    List of arrays of average magnetization vectors at each time step,
    in each treatment step.

    ps: list
    List of arrays of state vectors at each time step, in each treatment
    step.
    """
    # Gets the v and p arrays associated with each step.
    v_step = []
    p_step = []

    # Steps are progressed through linearly
    new_start_p = start_p
    new_start_t = 0
    j = 0
    for step in steps:
        # Get temperatures and times associated with each timestep
        ts = step.ts
        Ts = step.Ts
        # Get fields associated with with each timestep
        field_strs = step.field_strs
        field_dirs = step.field_dirs
        # Get the vectors for each time step
        v, p = grain_vectors(
            new_start_t,
            new_start_p,
            Ts,
            ts,
            d,
            energy_landscape,
            grain_dir,
            field_strs,
            field_dirs,
            eq=eq[j],
        )
        j += 1

        # Our new start vectors are whatever is left over after the
        # last step. One step follows immediately from another in
        # Our model.

        new_start_p = p[-1]
        new_start_t = ts[-1]

        # Add results for each thermal step to the lists.
        v_step.append(v)
        p_step.append(p)
    return (v_step, p_step)


def mono_dispersion(start_p, d, steps, energy_landscape: GEL, n_dirs=50, eq=False):
    """
    Gets the state vectors and average magnetization vectors at each
    time step in a thermal treatment for all directions in a
    mono-dispersion of grains. This calculation is performed for a
    set of treatment steps - see treatment.TreatmentStep for more details.
    N.B. - Recommend using parallelized_mono_dispersion instead of this,
    it's a lot faster.

    Inputs
    ------
    start_p: numpy array
    Initial state vector of grain.

    d: float
    Equivalent volume spherical diameter of grain (nm).

    steps: list of treatment.TreatmentStep objects
    Set of steps that describe a thermal experiment.

    energy_landscape: barriers.GEL object
    Object describing LEM states and energy barriers as a function of
    temperature

    n_dirs: int
    Number of Fibonacci sphere directions to use for mono-dispersion

    eq: bool
    If True, ignore time steps and run magnetization to equilibrium.

    Returns
    -------
    vs: numpy array
    List of arrays of average magnetization vectors at each time step,
    in each treatment step, for each mono-dispersion direction.

    ps: numpy array
    List of arrays of state vectors at each time step, in each treatment
    step, for each mono-dispersion direction.
    """
    if d > energy_landscape.d_min:
        warnings.warn(
            "WARNING: This particle may be too large to be single domain, results may be innaccurate"
        )

    dirs = fib_sphere(n_dirs)
    vs = []
    ps = []
    i = 0

    if isinstance(eq, bool):
        eq = np.full(len(steps), eq)
    else:
        pass

    for grain_dir in dirs:
        i += 1
        print("Working on grain {i} of {n}".format(i=i, n=n_dirs), end="\r")
        v_step = []
        p_step = []
        new_start_p = start_p
        new_start_t = 0
        j = 0
        for step in steps:
            ts = step.ts
            Ts = step.Ts
            field_strs = step.field_strs
            field_dirs = step.field_dirs
            v, p = grain_vectors(
                new_start_t,
                new_start_p,
                Ts,
                ts,
                d,
                energy_landscape,
                grain_dir,
                field_strs,
                field_dirs,
                eq=eq[j],
            )
            j += 1
            new_start_p = p[-1]
            new_start_t = ts[-1]
            v_step.append(v)
            p_step.append(p)

        vs.append(v_step)
        ps.append(p_step)
    vs = np.array(vs, dtype="object")
    ps = np.array(ps, dtype="object")
    vs = np.sum(vs, axis=0)
    return (vs, ps)


def parallelized_mono_dispersion(
    start_p, d, steps, energy_landscape: GEL, n_dirs=50, eq=False
):
    """
    Gets the state vectors and average magnetization vectors at each
    time step in a thermal treatment for all directions in a
    mono-dispersion of grains. This calculation is performed for a
    set of treatment steps - see treatment.TreatmentStep for more details.

    Inputs
    ------
    start_p: numpy array
    Initial state vector of grain.

    d: float
    Equivalent volume spherical diameter of grain (nm).

    steps: list of treatment.TreatmentStep objects
    Set of steps that describe a thermal experiment.

    energy_landscape: barriers.GEL object
    Object describing LEM states and energy barriers as a function of
    temperature

    n_dirs: int
    Number of Fibonacci sphere directions to use for mono-dispersion

    eq: bool
    If True, ignore time steps and run magnetization to equilibrium.

    Returns
    -------
    vs: numpy array
    List of arrays of average magnetization vectors at each time step,
    in each treatment step, for each mono-dispersion direction.

    ps: numpy array
    List of arrays of state vectors at each time step, in each treatment
    step, for each mono-dispersion direction.
    """
    if d > energy_landscape.d_min:
        warnings.warn(
            "WARNING: This particle may be too large to be single domain, results may be innaccurate"
        )

    dirs = fib_sphere(n_dirs)

    if isinstance(eq, bool):
        eq = np.full(len(steps), eq)
    else:
        pass

    pool = mpc.Pool(mpc.cpu_count())
    objs = np.array(
        [
            pool.apply_async(
                mono_direction,
                args=(grain_dir, start_p, d, steps, energy_landscape, eq),
            )
            for grain_dir in dirs
        ]
    )
    vps = np.array([obj.get() for obj in objs], dtype="object")
    pool.close()
    vs = vps[:, 0]
    ps = vps[:, 1]
    vs = np.sum(vs, axis=0)
    return (vs, ps)


class SDCCResult:
    """
    Class to store results from a set of grain results - able to be
    dumped to file.
    """

    def __init__(self, sizes, thermal_steps, vs, ps):
        self.sizes = sizes
        self.thermal_steps = thermal_steps
        self.vs = vs
        self.ps = ps

    def to_file(self, fname):
        with open(fname, "wb") as f:
            pickle.dump(self, f, -1)


def eq_ps(params, field_str, field_dir, d):
    """
    Get the probabilities of each state in a grain under a specific set
    of conditions after an infinite amount of time.

    Inputs
    ------
    params: dictionary
    Dictionary output from a barriers.GEL object - contains states and
    energy barriers.

    field_str: float
    Field strength (uT)

    field_dirs: numpy array
    Unit vector of field direction

    T: float
    Temperature (degrees C)

    d: float
    Equivalent volume spherical diameter of grain (nm).

    Ms: float
    Saturation magnetization of grain (A/m)

    Return
    ------
    ps: numpy array
    Equilibrium state vector.
    """
    # This works very similarly to Q matrix, except we just use the
    # Relative energies of the states instead of the barriers approach!
    T = params["T"]
    Ms = params["Ms"]
    theta_list = params["min_dir"][:, 0]
    phi_list = params["min_dir"][:, 1]
    min_energies = params["min_e"]

    V = 4 / 3 * np.pi * ((d / 2 * 1e-9) ** 3)
    kb = 1.380649e-23
    min_energies = np.array(min_energies)
    xyz = angle2xyz(theta_list, phi_list)
    for i in range(len(theta_list)):
        zeeman_energy = np.dot(xyz[:, i], field_dir * field_str * 1e-6) * Ms
        min_energies[i] -= zeeman_energy

    precise_exp = np.vectorize(mp.exp)
    e_ratio = precise_exp(-(min_energies * V) / (kb * (273 + T)))
    ps = e_ratio / sum(e_ratio)
    return np.array(ps, dtype="float64")


def eq_ps_legacy(theta_list, phi_list, min_energies, field_str, field_dir, T, d, Ms):
    """
    DEPRECATED
    Legacy version of eq_ps (no params).
    """
    V = 4 / 3 * np.pi * ((d / 2 * 1e-9) ** 3)
    kb = 1.380649e-23
    min_energies = np.array(min_energies)
    xyz = angle2xyz(theta_list, phi_list)
    for i in range(len(theta_list)):
        zeeman_energy = np.dot(xyz[:, i], field_dir * field_str * 1e-6) * Ms
        min_energies[i] -= zeeman_energy

    precise_exp = np.vectorize(mp.exp)
    e_ratio = precise_exp(-(min_energies * V) / (kb * (273 + T)))
    ps = e_ratio / sum(e_ratio)
    return np.array(ps, dtype="float64")


def calc_relax_time(start_p, d, relax_routine, energy_landscape, ts):
    """
    Function for calculating the relaxation time of a mono-dispersion
    of grains. The relaxation time is calculated as the time it takes
    for the magnetization to decay to 1/e.

    Inputs
    ------
    start_p: numpy array
    Starting state vector

    relax_routine: list of treatment.TreatmentStep objects.
    Steps describing a relaxation time treatment (cooling infield,
    followed by hold at room temperature infield).

    energy_landscape: barriers.GEL object
    Object describing LEM states and energy barriers as a function of
    temperature.

    ts: numpy array
    Array of time steps to check relaxation time at
    N.B. this should be roughly the same as
    relax_routine[1].ts - relax_routine[0].ts[-1]
    """
    # Run a parallelized mono dispersion
    vs, ps = mono_dispersion(
        start_p,
        d,
        relax_routine,
        energy_landscape,
        n_dirs=30,
        eq=np.array([True, False, False]),
    )
    # Calculate magnitude of vector
    mags = np.linalg.norm(vs[2], axis=1)
    # Calculate TRMs
    TRM = np.linalg.norm(vs[1][-1])
    # Get relaxation time (M = TRM/e)
    if mags[-1] <= (TRM / np.e**2):
        relax_time = ts[mags <= (TRM / np.e**2)][0]
    else:
        relax_time = ts[-1]
    return relax_time


def relax_time_crit_size(relax_routine, energy_landscape, init_size=[5], size_incr=10):
    """
    Finds the critical SP size of a grain.

    Inputs
    ------
    relax_routine: list of treatment.TreatmentStep objects.
    Steps describing a relaxation time treatment (cooling infield,
    followed by hold at room temperature infield).

    energy_landscape: barriers.GEL object
    Object describing LEM states and energy barriers as a function of
    temperature.

    init_size: list of ints
    Initial grain sizes to try.

    size_incr: int
    Amount to increment size by in certain situations.

    Returns
    -------
    d: int
    Critical SD size in nm.
    """
    n_states = len(energy_landscape.get_params(energy_landscape.T_max)["min_e"])
    start_p = np.full(n_states, 1 / n_states)
    ts = relax_routine[2].ts - relax_routine[1].ts[-1]

    relax_times = []
    ds = []

    # Run through all the possible relaxation times
    # From energy barriers calculated
    for d in np.ceil(init_size).astype(int):
        if d != np.ceil(init_size[0]).astype(int):
            if (min(relax_times) < 100) & (max(relax_times) >= 100):
                pass
            else:
                print(f"Current Size {d} nm                 ")
                relax_time = calc_relax_time(
                    start_p, d, relax_routine, energy_landscape, ts
                )
                print("Relaxation time %1.1e" % relax_time)
                relax_times.append(relax_time)
                ds.append(d)
        else:
            print(f"Current Size {d} nm                 ")
            relax_time = calc_relax_time(
                start_p, d, relax_routine, energy_landscape, ts
            )
            print("Relaxation time %1.1e" % relax_time)
            relax_times.append(relax_time)
            ds.append(d)

    # If relaxation times don't span
    # the necessary range, step up until they do

    # What stopping condition do we use for stepping?
    if np.amax(relax_times) < 100:
        statefun = lambda r: np.amax(r) < 100
        state = statefun(relax_times)
    elif np.amin(relax_times) >= 100:
        statefun = lambda r: np.amin(r) >= 100
        state = statefun(relax_times)
    else:
        state = False

    # What direction do we step in?
    if relax_time < 100:
        sign = 1
    else:
        sign = -1

    # Step upwards.
    while state:
        d += sign * int(size_incr)
        if d <= 0:
            d = 1
        print(f"Current Size {d} nm                 ")
        relax_time = calc_relax_time(start_p, d, relax_routine, energy_landscape, ts)
        print("Relaxation time %1.1e" % relax_time)
        relax_times.append(relax_time)
        ds.append(d)
        state = statefun(relax_times)

    # Now we use a bisection method
    # to find the critical size,
    # followed by a root-finding method
    # when close enough to resolve the
    # relaxation time
    continuing = True
    while continuing:
        d_sorted = np.array(np.sort(ds))
        r_sorted = np.array(relax_times)[np.argsort(ds)]

        d_min = d_sorted[r_sorted < 100][-1]
        d_max = d_sorted[r_sorted >= 100][0]
        r_min = r_sorted[r_sorted < 100][-1]
        r_max = r_sorted[r_sorted >= 100][0]

        if r_max == ts[-1] or r_min == ts[0]:
            d = int(np.ceil((d_min + d_max) / 2))

        else:
            d = int(np.ceil(np.interp(2, np.log10(r_sorted), d_sorted)))

        print(f"Current Size {d} nm                ")
        if d in ds and (len(ds) > 2):
            continuing = False
        else:
            relax_time = calc_relax_time(
                start_p, d, relax_routine, energy_landscape, ts
            )
            print("Relaxation time %1.1e" % relax_time)
            relax_times.append(relax_time)
            ds.append(d)

    return d


def critical_size(K):
    """
    Calculates the critical size (nm) given an energy barrier simply
    using the Neel relaxation time equation and nothing else.
    """
    tau_0 = 1e-9
    t = 100
    kb = 1.380649e-23
    V = np.log(t / tau_0) * kb * 293 / K
    r = (V * 3 / (4 * np.pi)) ** (1 / 3)
    d = 2 * r
    return d * 1e9


def full_crit_size(TMx, PRO, OBL, alignment):
    """
    Calculate critical SD size of a grain, taking some shortcuts by
    using Neel relaxation time in cases when there should be only one
    energy barrier.

    Inputs
    ------
    TMx: float
    Titanomagnetite composition % (0 - 100)

    PRO: float
    Prolateness ratio (major axis/intermediate axis)

    OBL: float
    Oblateness ratio (intermediate axis/minor axis)

    alignment: string
    Either `easy` or `hard`. Specifies the magnetocrystalline direction
    that should be aligned with the x direction, which for our
    ellipsoids is the major (shape easy) axis.

    Returns
    -------
    d: int
    Critical SD size in nm.
    """
    theta_list, phi_list, min_energy_list, theta_mat, phi_mat, barriers = (
        find_all_barriers(TMx, alignment, PRO, OBL)
    )

    if PRO == 1.00 and OBL == 1.00:
        do_full = False
    elif len(theta_list) == 2:
        do_full = False
    else:
        do_full = True
    if do_full:
        Energy = GEL(TMx, alignment, PRO, OBL)
        relax_routine = relaxation_time(Energy, np.array([1, 0, 0]), 40)
        relax_routine = relaxation_time(Energy, np.array([1, 0, 0]), 40)
        # If the grain is unfeasibly large, we might not reach equilibrium
        # And so have zero magnetization
        # In these cases, a "pre-hold" where we force the grain to equilibrium
        # At max temperature.
        pre_hold = [HoldStep(0, Energy.T_max, 40, np.array([1, 0, 0]), hold_steps=2)]
        for step in pre_hold:
            step.ts -= 1801
        relax_routine = pre_hold + relax_routine

        barrierslist = []
        for barrier in np.unique(np.floor(barriers[~np.isinf(barriers)] / 1000) * 1000):
            barrierslist.append(
                np.mean(barriers[(barriers >= barrier) & (barriers < barrier + 1000)])
            )
        potential_ds = critical_size(np.array(barrierslist))
        d = relax_time_crit_size(relax_routine, Energy, init_size=potential_ds)
        return d
    else:
        d = critical_size(np.array(barriers)[0, 1])
        return d


def hyst_treatment(start_t, start_p, Bs, ts, d, energy_landscape: HEL, eq=False):
    """
    Function for calculating the probability of different LEM states in
    a grain during a hysteresis experiment.

    Inputs
    ------
    start_t: float
    Time at which this experiment step starts

    start_p: numpy array
    Initial state vector

    Bs: numpy array
    Set of field strengths at the times corresponding to ts.

    ts: numpy array
    Time steps at which we calculate the state.

    d: float
    Equivalent volume spherical diameter of grain (nm).

    energy_landscape: barriers.HEL object
    Object describing energy barriers for a particular grain geometry.

    eq: bool
    If True, ignore time steps and run magnetization to equilibrium.

    Returns
    -------
    ps: numpy array
    Array of state vectors at each time step

    theta_lists: numpy array
    Magnetization directions at each time step

    phi_lists: numpy array
    Magnetization magnitudes at each time step
    """

    # Get the starting temperature
    old_B = Bs[0]

    # Get the energy barriers and LEM states at this temperature.
    params = energy_landscape.get_params(old_B)
    # If doing equilibrium, run this state to equilibrium.
    if eq:
        old_p = eq_ps(params, 0, [1, 0, 0], d)

    # Otherwise, calculate the Q matrix as normal, calculate the new state
    # vector.

    else:
        Q = Q_matrix(params, d, field_dir=np.array([1, 0, 0]), field_str=0)
        old_p = _update_p_vector(start_p, Q, ts[0] - start_t)  # New state vector
    # Create list of state vectors, place first one in there.
    ps = [old_p]
    # Create list of LEM state directions - put initial ones in there.
    theta_lists = [params["min_dir"][:, 0]]
    phi_lists = [params["min_dir"][:, 1]]

    # Loop through time steps
    for i in range(1, len(ts)):
        # Get time, temperature, LEM states and barriers at each temperature
        B = Bs[i]
        dt = ts[i] - ts[i - 1]
        params = energy_landscape.get_params(B)

        # Again if equilibrium run for infinite time
        if eq:
            new_p = eq_ps(params, 0, [1, 0, 0], d)

        # Otherwise calculate Q matrix and new state vector
        else:
            Q = Q_matrix(params, d, field_dir=np.array([1, 0, 0]), field_str=0)
            new_p = _update_p_vector(ps[-1], Q, dt)

        # Add state vector to list of state vectors
        ps.append(new_p)

        # Do the same for state magnetization directions at this temp.
        theta_list = params["min_dir"][:, 0]
        phi_list = params["min_dir"][:, 1]
        theta_lists.append(theta_list)
        phi_lists.append(phi_list)

    return (ps, theta_lists, phi_lists)


def grain_hyst_vectors(
    start_t,
    start_p,
    Bs,
    ts,
    d,
    energy_landscape: HEL,
    field_dir,
    eq=False,
):
    """
    Gets the state vectors and average magnetization vectors at each
    time step in a hysteresis experiment for a single direction in a
    mono-dispersion of grains. This calculation is performed for a
    single treatment step - i.e. a single heating or cooling.
    See treatment.TreatmentStep for a full description of this.

        Inputs
    ------
    start_t: float
    Time at which this experiment step starts

    start_p: numpy array
    Initial state vector

    Bs: numpy array
    Set of field strengths at the times corresponding to ts.

    ts: numpy array
    Time steps at which we calculate the state.

    d: float
    Equivalent volume spherical diameter of grain (nm).

    energy_landscape: barriers.HEL object
    Object describing energy barriers for a particular grain geometry.

    grain_dir: numpy array
    Direction associated with this grain.

    field_dir: numpy array
    Direction of field relative to grain - will be rotated to 1,0,0.

    eq: bool
    If True, ignore time steps and run magnetization to equilibrium.

    Returns
    -------
    vs: numpy array
    Array of average magnetization vectors at each time step

    ps: numpy array
    Array of state vectors at each time step
    """
    # Convert our field directions to a rotation matrix, so that we can
    # rotate back to grain coordinates later.
    field_dirstr = field_dir.astype(str)
    field_dirstr = " ".join(field_dirstr)
    ref_dir = np.array([1, 0, 0])
    ref_dirstr = ref_dir.astype(str)
    ref_dirstr = " ".join(ref_dirstr)
    rot_mat = dir_to_rot_mat(ref_dirstr, field_dirstr)

    # Get the field directions rotated according to this matrix.

    # Get the state vectors at each time step.
    ps, theta_lists, phi_lists = hyst_treatment(
        start_t, start_p, Bs, ts, d, energy_landscape, eq=eq
    )

    # Get the average magnetization vectors at each time step
    vs = get_avg_vectors(ps, theta_lists, phi_lists, Bs, rot_mat, energy_landscape, d)
    return (vs, ps)


def mono_hyst_direction(start_p, d, steps, energy_landscape: HEL, eq=[False]):
    """
    Gets the state vectors and average magnetization vectors at each
    time step in a thermal treatment for a single direction in a
    mono-dispersion of grains. This calculation is performed for a
    set of treatment steps - see treatment.TreatmentStep for more details.

    Inputs
    ------
    grain_dir: numpy array
    Direction of this grain in the mono dispersion.

    start_p: numpy array
    Initial state vector of grain.

    d: float
    Equivalent volume spherical diameter of grain (nm).

    steps: list of treatment.TreatmentStep objects
    Set of steps that describe a thermal experiment.

    energy_landscape: barriers.GEL object
    Object describing LEM states and energy barriers as a function of
    temperature

    eq: bool
    If True, ignore time steps and run magnetization to equilibrium.

    Returns
    -------
    vs: lists
    List of arrays of average magnetization vectors at each time step,
    in each treatment step.

    ps: list
    List of arrays of state vectors at each time step, in each treatment
    step.
    """
    # Gets the v and p arrays associated with each step.
    v_step = []
    p_step = []

    # Steps are progressed through linearly
    new_start_p = start_p
    new_start_t = 0
    j = 0
    field_dir = np.array(energy_landscape.B_dir)
    for step in steps:
        # Get temperatures and times associated with each timestep
        ts = step.ts
        Bs = step.field_strs / 1e6
        # Get the vectors for each time step
        j += 1
        # Our new start vectors are whatever is left over after the
        # last step. One step follows immediately from another in
        # Our model.
        v, p = grain_hyst_vectors(
            new_start_t, new_start_p, Bs, ts, d, energy_landscape, field_dir, eq=eq
        )
        new_start_p = p[-1]
        new_start_t = ts[-1]

        # Add results for each thermal step to the lists.
        v_step.append(v)
        p_step.append(p)
    return (v_step, p_step)


def hyst_mono_dispersion(d, steps, energy_landscape, eq=False):
    """
    Gets the state vectors and average magnetization vectors at each
    time step in a high-field treatment for all directions in a
    mono-dispersion of grains. This calculation is performed for a
    set of treatment steps - see treatment.TreatmentStep for more details.

    Inputs
    ------
    start_p: numpy array
    Initial state vector of grain.

    d: float
    Equivalent volume spherical diameter of grain (nm).

    steps: list of treatment.TreatmentStep objects
    Set of steps that describe a hysteresis experiment.

    energy_landscape: barriers.HELs object
    Object describing LEM states and energy barriers as a function of
    fields.

    n_dirs: int
    Number of Fibonacci sphere directions to use for mono-dispersion

    eq: bool
    If True, ignore time steps and run magnetization to equilibrium.

    Returns
    -------
    vs: numpy array
    List of arrays of average magnetization vectors at each time step,
    in each treatment step, for each mono-dispersion direction.

    ps: numpy array
    List of arrays of state vectors at each time step, in each treatment
    step, for each mono-dispersion direction.
    """
    vs = []
    ps = []
    if d > energy_landscape.HEL_list[0].d_min:
        warnings.warn(
            "WARNING: This particle may be too large to be single domain, results may be innaccurate"
        )

    for hel in energy_landscape.HEL_list:
        min_e = hel.get_params(0)["min_e"]
        start_p = np.zeros(len(min_e))
        start_p[~np.isinf(min_e)] = 1 / len(start_p[~np.isinf(min_e)])
        v, p = mono_hyst_direction(start_p, d, steps, hel, eq=eq)
        v = np.array(v, dtype="object")
        vs.append(v)
        ps.append(p)
    vs = sum(vs)
    return (vs, ps)
