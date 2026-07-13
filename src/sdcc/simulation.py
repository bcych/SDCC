import numpy as np
import mpmath as mp
import pickle
import multiprocessing as mpc
import warnings

from sdcc.particles import EnergyLandscape
from sdcc.energy import angle2xyz
from sdcc.utils import fib_hypersphere

from jax import jit, config
from jax.scipy.linalg import expm

# Set high precision (handled for different
# versions of mpmath)
mp_active = False
try:
    mp.prec = 100
    mp_active = True
except AttributeError:
    pass

try:
    mp.mp.prec = 100
    mp_active = True
except AttributeError:
    pass

if not mp_active:
    raise AttributeError(
        "Could not initialize high floating point precision, please install a working version of mpmath"
    )

config.update("jax_enable_x64", True)


def Q_matrix(params: dict, d, field_dir=np.array([1, 0, 0]), field_str=0.0):
    """
    Constructs a Q matrix of rates of transition between LEM states.

    Parameters
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
    M = params["min_m"]
    bar_M = params["bar_m"]

    V = 4 / 3 * np.pi * ((d / 2 * 1e-9) ** 3)
    kb = 1.380649e-23
    tau_0 = 1e-9
    tt, pp = np.meshgrid(theta_list, phi_list)
    pp = pp.T
    xyz = angle2xyz(tt, pp)
    MM, mmT = np.meshgrid(M, M)
    xyz *= MM * V
    xyz_T = angle2xyz(theta_mat, phi_mat)
    xyz_T *= bar_M * V
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
    zeeman_energy = zeeman_energy.at[np.isnan(zeeman_energy)].set(0.0)
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

    Parameters
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


@jit
def _update_p_vector_fast(p_vec, Q, dt):
    """
    Given an initial state vector, a Q matrix and a time, calculates a
    new state vector. This is a faster implementation but can have
    precision errors which rapidly blow up. Improvements to jax expm
    may make this viable one day.

    Parameters
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
    dp_dt = expm(Q * dt, max_squarings=256)
    # The old p vector is matrix multiplied with this matrix
    p_vec_new = dp_dt @ p_vec
    # Sometimes due to floating point errors the sum of the vector isn't
    # Exactly 1 - this will blow up if we don't renormalize.
    p_vec_new /= sum(p_vec_new)
    return p_vec_new


def thermal_treatment(
    start_t,
    start_p,
    Ts,
    ts,
    d,
    energy_landscape: EnergyLandscape,
    field_strs,
    field_dirs,
    eq=False,
):
    """
    Function for calculating the probability of different LEM states in
    a grain during a thermal experiment.

    Parameters
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
        if ts[0] == start_t:
            old_p = start_p
        else:
            old_p = _update_p_vector(start_p, Q, ts[0] - start_t)  # New state vector
    # Create list of state vectors, place first one in there.
    if len(old_p.shape) > 1:
        old_p = old_p[:, 0]
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
        if len(new_p.shape) > 1:
            new_p = new_p[:, 0]

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

    Parameters
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

    for i in range(len(ps)):
        # Volume of grain
        V = 4 / 3 * np.pi * ((d / 2 * 1e-9) ** 3)

        # Temperature at time Tces
        T = Ts[i]

        # Get Ms material parameter
        M = energy_landscape.get_params(T)["min_m"]

        # Get directions associated with states
        theta_list = theta_lists[i]
        phi_list = phi_lists[i]
        # For SD, magnitude is V * Ms
        vecs = angle2xyz(theta_list, phi_list) * ps[i] * M * V
        vecs = np.nan_to_num(vecs, nan=0)
        v = np.sum(vecs, axis=1)  # Grain direction
        vs.append(inv_rot @ v)  # Rotate back into constant field direction
    return np.array(vs)


def grain_vectors(
    start_t,
    start_p,
    Ts,
    ts,
    d,
    energy_landscape: EnergyLandscape,
    rot_mat,
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

    Parameters
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

    rot_mat: 3x3 matrix
        Orientation of grain

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


def mono_direction(
    rot_mat, start_p, d, steps, energy_landscape: EnergyLandscape, eq=[False]
):
    """
    Gets the state vectors and average magnetization vectors at each
    time step in a thermal treatment for a single direction in a
    mono-dispersion of grains. This calculation is performed for a
    set of treatment steps - see treatment.TreatmentStep for more details.

    Parameters
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

    rot_mat: numpy array
        Direction of this grain in the mono dispersion.

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
            rot_mat,
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


def mono_dispersion(
    start_p, d, steps, energy_landscape: EnergyLandscape, n_dirs=50, eq=False
):
    """
    Gets the state vectors and average magnetization vectors at each
    time step in a thermal treatment for all directions in a
    mono-dispersion of grains. This calculation is performed for a
    set of treatment steps - see treatment.TreatmentStep for more details.
    N.B. - Recommend using parallelized_mono_dispersion instead of this,
    it's a lot faster.

    Parameters
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
    if d > energy_landscape.info_dict["Maximum Size"]:
        warnings.warn(
            "WARNING: This particle may be too large to be single domain, results may be innaccurate"
        )

    rot_mats = fib_hypersphere(n_dirs)
    vs = []
    ps = []
    i = 0

    if len(np.array(start_p).shape) == 1:
        start_p = np.repeat(np.array([start_p]), n_dirs, axis=0)
    else:
        start_p = np.array(start_p)

    if isinstance(eq, bool):
        eq = np.full(len(steps), eq)
    else:
        pass

    for rot_mat in rot_mats:
        i += 1
        print("Working on grain {i} of {n}".format(i=i, n=n_dirs), end="\r")
        v_step = []
        p_step = []
        new_start_p = start_p[i - 1]
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
                rot_mat,
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
    start_p,
    d,
    steps,
    energy_landscape: EnergyLandscape,
    n_dirs=50,
    eq=False,
    cpu_count=0,
    ctx=None,
):
    """
    Gets the state vectors and average magnetization vectors at each
    time step in a thermal treatment for all directions in a
    mono-dispersion of grains. This calculation is performed for a
    set of treatment steps - see treatment.TreatmentStep for more details.

    Parameters
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

    cpu_count: int
        Number of cores to parallelize with, if 0 chooses available cores

    ctx: None or str
        context used for multiprocessing. If working with None, then don't
        change this. Otherwise, set to "spawn" and call this function in a
        python script inside an `if __name__ == "__main__"` statement. Note:
        if set to spawn, multiprocessing will not work in a jupyter notebook

    Returns
    -------
    vs: numpy array
        List of arrays of average magnetization vectors at each time step,
        in each treatment step, for each mono-dispersion direction.

    ps: numpy array
        List of arrays of state vectors at each time step, in each treatment
        step, for each mono-dispersion direction.
    """
    if d > energy_landscape.info_dict["Maximum Size"]:
        warnings.warn(
            "WARNING: This particle may be too large to be single domain, results may be innaccurate"
        )

    rot_mats = fib_hypersphere(n_dirs)

    if len(np.array(start_p).shape) == 1:
        start_p = np.repeat(np.array([start_p]), n_dirs, axis=0)
    else:
        start_p = np.array(start_p)

    if isinstance(eq, bool):
        eq = np.full(len(steps), eq)
    else:
        pass
    if type(ctx) != type(None):
        context = mpc.get_context(ctx)
    else:
        context = mpc
    if cpu_count == 0:
        cpu_count = context.cpu_count()
    pool = context.Pool(cpu_count)
    objs = np.array(
        [
            pool.apply_async(
                mono_direction,
                args=(rot_mats[i], start_p[i], d, steps, energy_landscape, eq),
            )
            for i in range(n_dirs)
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

    Parameters
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
    M = params["min_m"]
    theta_list = params["min_dir"][:, 0]
    phi_list = params["min_dir"][:, 1]
    min_energies = params["min_e"]

    V = 4 / 3 * np.pi * ((d / 2 * 1e-9) ** 3)
    kb = 1.380649e-23
    min_energies = np.array(min_energies)
    xyz = angle2xyz(theta_list, phi_list)
    for i in range(len(theta_list)):
        zeeman_energy = np.dot(xyz[:, i], field_dir * field_str * 1e-6) * M[i]
        if np.isnan(zeeman_energy):
            zeeman_energy = 0
        min_energies[i] -= zeeman_energy

    precise_exp = np.vectorize(mp.exp)
    e_ratio = precise_exp(-(min_energies * V) / (kb * (273 + T)))
    ps = e_ratio / sum(e_ratio)
    return np.array(ps, dtype="float64")


def hyst_treatment(
    start_t, start_p, Bs, ts, d, energy_landscape: EnergyLandscape, eq=False
):
    """
    Function for calculating the probability of different LEM states in
    a grain during a hysteresis experiment.

    Parameters
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
    energy_landscape: EnergyLandscape,
    rot_mat,
    eq=False,
):
    """
    Gets the state vectors and average magnetization vectors at each
    time step in a hysteresis experiment for a single direction in a
    mono-dispersion of grains. This calculation is performed for a
    single treatment step - i.e. a single heating or cooling.
    See treatment.TreatmentStep for a full description of this.

        Parameters
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

    rot_mat: 3x3 matrix
        Orientation associated with this grain.

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

    # Get the field directions rotated according to this matrix.

    # Get the state vectors at each time step.
    ps, theta_lists, phi_lists = hyst_treatment(
        start_t, start_p, Bs, ts, d, energy_landscape, eq=eq
    )

    # Get the average magnetization vectors at each time step
    vs = get_avg_vectors(ps, theta_lists, phi_lists, Bs, rot_mat, energy_landscape, d)
    return (vs, ps)


def mono_hyst_direction(
    start_p, d, steps, energy_landscape: EnergyLandscape, eq=[False]
):
    """
    Gets the state vectors and average magnetization vectors at each
    time step in a thermal treatment for a single direction in a
    mono-dispersion of grains. This calculation is performed for a
    set of treatment steps - see treatment.TreatmentStep for more details.

    Parameters
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
    rot_mat = np.array(energy_landscape.info_dict["Field Rotation Matrix"])
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
            new_start_t, new_start_p, Bs, ts, d, energy_landscape, rot_mat, eq=eq
        )
        new_start_p = p[-1]
        new_start_t = ts[-1]

        # Add results for each thermal step to the lists.
        v_step.append(v)
        p_step.append(p)
    return (v_step, p_step)


def hyst_mono_dispersion(start_p, d, steps, energy_landscape, eq=False):
    """
    Gets the state vectors and average magnetization vectors at each
    time step in a high-field treatment for all directions in a
    mono-dispersion of grains. This calculation is performed for a
    set of treatment steps - see treatment.TreatmentStep for more details.

    Parameters
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
    if d > energy_landscape.HEL_list[0].info_dict["Maximum Size"]:
        warnings.warn(
            "WARNING: This particle may be too large to be single domain, results may be innaccurate"
        )

    n_dirs = len(energy_landscape.HEL_list)
    if len(np.array(start_p).shape) == 1:
        start_p = np.repeat(np.array([start_p]), n_dirs, axis=0)
    else:
        start_p = np.array(start_p)

    i = 1
    for hel in energy_landscape.HEL_list:
        print(
            "Working on grain {i} of {n}".format(i=i, n=len(energy_landscape.HEL_list)),
            end="\r",
        )
        v, p = mono_hyst_direction(start_p[i - 1], d, steps, hel, eq=eq)
        v = np.array(v, dtype="object")
        vs.append(v)
        ps.append(p)
        i += 1
    vs = sum(vs)
    return (vs, ps)


def parallelized_hyst_mono_dispersion(
    start_p, d, steps, energy_landscape, eq=False, cpu_count=0, ctx=None
):
    """
    Gets the state vectors and average magnetization vectors at each
    time step in a hysteresis treatment for all directions in a
    mono-dispersion of grains. This calculation is performed for a
    set of treatment steps - see treatment.TreatmentStep for more details.

    Parameters
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

    cpu_count: int
        Number of cores to parallelize with, if 0 chooses available cores

    ctx: None or str
        context used for multiprocessing. If working with None, then don't
        change this. Otherwise, set to "spawn" and call this function in a
        python script inside an `if __name__ == "__main__"` statement. Note:
        if set to spawn, multiprocessing will not work in a jupyter notebook

    Returns
    -------
    vs: numpy array
        List of arrays of average magnetization vectors at each time step,
        in each treatment step, for each mono-dispersion direction.

    ps: numpy array
        List of arrays of state vectors at each time step, in each treatment
        step, for each mono-dispersion direction.
    """
    if d > energy_landscape.HEL_list[0].info_dict["Maximum Size"]:
        warnings.warn(
            "WARNING: This particle may be too large to be single domain, results may be innaccurate"
        )

    if type(ctx) != type(None):
        context = mpc.get_context(ctx)
    else:
        context = mpc
    if cpu_count == 0:
        cpu_count = context.cpu_count()
    n_dirs = len(energy_landscape.HEL_list)
    if len(np.array(start_p).shape) == 1:
        start_p = np.repeat(np.array([start_p]), n_dirs, axis=0)
    else:
        start_p = np.array(start_p)
    pool = context.Pool(cpu_count)

    objs = np.array(
        [
            pool.apply_async(
                mono_hyst_direction,
                args=(start_p[i], d, steps, energy_landscape.HEL_list[i], eq),
            )
            for i in range(n_dirs)
        ]
    )
    vps = np.array([obj.get() for obj in objs], dtype="object")
    pool.close()
    vs = vps[:, 0]
    ps = vps[:, 1]
    vs = np.sum(vs, axis=0)
    return (vs, ps)


def result_to_file(
    energyLandscape,
    size,
    routine,
    moments,
    probabilities,
    file_ext: str,
    directory="./",
):
    """
    Saves an SDCC result to file, along with information about the result

    Parameters
    ----------
    energyLandscape: GEL, HEL or HELs object
        The object that describes the particle energies

    size: float
        The size of the particle (in nm)

    routine: list
    The routine of treatment steps

    moments: list
        The moments output by the SDCC

    probabilities: list
        The probabilities of each state in each particle at each
        time-step.

    file_ext: str
        The file extension - should relate to the experiment type.

    Returns
    -------
    None
    """
    info_dict = energyLandscape.info_dict
    result = {
        "particle": info_dict,
        "routine": routine,
        "result": {"moments": moments, "probs": probabilities},
    }

    fname = (
        directory
        + f"{info_dict['Shape Class']}_{info_dict['Material']}_PRO_{info_dict['Prolateness']:1.2f}_OBL_{info_dict['Oblateness']:1.2f}_{size:3.1f}nm."
        + file_ext
    )
    with open(fname, "wb") as f:
        pickle.dump(result, f)
        f.close()
    print("Saved file to " + fname)
    return None
