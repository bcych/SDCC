import numpy as np
import matplotlib.pyplot as plt
from sdcc.treatment import CoolingStep
from sdcc.particles import energy_result, load_particle
from scipy.optimize import minimize


def analyze_hyst_data(vs, steps, d, hels, plot=False, ax=None):
    init_ramp = vs[0][:, 0]
    back_ramp = vs[1][:, 0]
    forward_ramp = vs[2][:, 0]
    Mrs = back_ramp[steps[1].field_strs == 0][0]
    Mrs2 = np.abs(back_ramp[steps[2].field_strs == 0][0])
    Mrs = (Mrs2 + Mrs) / 2
    Msat = hels[0].Ms(steps[0].Ts[0])

    n = len(hels.HEL_list)
    v = 4 / 3 * np.pi * (d / 1e9 / 2) ** 3

    Ms = n * v * Msat
    Bs = steps[2].field_strs / 1e3
    Bc = np.interp(0, forward_ramp, Bs)
    Bc2 = np.abs(np.interp(0, np.flip(back_ramp), Bs))
    Bc = (Bc2 + Bc) / 2

    if plot == True:
        if ax == None:
            fig, ax = plt.subplots()
        ax.axhline(0, color="grey", lw=1)
        ax.axvline(0, color="grey", lw=1)
        ax.plot(steps[0].field_strs / 1e3, init_ramp, "k")
        ax.plot(steps[1].field_strs / 1e3, back_ramp, "r")
        ax.set_ylabel(r"Moment (Am$^2$)")
        ax.set_xlabel("B (mT)")
        ax2 = ax.twinx()
        ax2.axhline(0.5, color="grey", ls="--", lw=1, zorder=0)
        ax2.axhline(-0.5, color="grey", ls="--", lw=1, zorder=0)
        ax2.plot(steps[2].field_strs / 1e3, forward_ramp / Ms, "r")
        ax2.set_ylabel("M/Ms")
        ax2.set_ylim(np.array(ax.get_ylim()) / Ms)
        ax2.text(min(steps[1].field_strs / 1e3), 0.9, "Mrs/Ms: %1.3f" % (Mrs / Ms))
        ax2.text(min(steps[1].field_strs / 1e3), 0.8, "Bc: %2.1f" % (Bc) + " mT")
    return (Mrs, Ms, Bc)


def process_thellier_data(vs_list, routine, weights):
    """
    Processes data from a Thellier experiment to create Arai and
    Zijderveld plot data.

    Inputs
    ------
    vs_list: list
    Set of moment vectors at every time step
    from an experiment run with the SDCC.

    routine: list
    Routine of treatment steps used for the experiment.

    weights: list or array
    Relative contribution of each grain to the result -
    a higher weight would mean relatively more of those
    grains.

    Returns
    -------
    Zs_list: array
    Array of moment vectors for Zijderveld plot data

    Is_list: array
    In field step moment vectors (vector subtracted from
    zero field vectors)

    Zs_mag, Is_mag: array
    Magnitudes of above vectors, normalized by the TRM
    magnitude (Arai plot data).
    """
    steps = []
    for step in routine:
        if step == routine[0]:
            steps.append("TRM")
        elif type(step) == CoolingStep:
            if step.field_strs[0] > 0:
                steps.append("I")
            else:
                steps.append("Z")
        else:
            steps.append("N")
    steps = np.array(steps)

    n = len(steps[steps == "Z"]) + 1

    Zs_tot = np.empty((n, 3, len(vs_list)))
    Is_tot = np.empty((n, 3, len(vs_list)))

    for i, vs in enumerate(vs_list):
        vs = np.array(vs, dtype="object")
        TRM = vs[steps == "TRM"][0][-1]
        Zcools = vs[steps == "Z"]
        Icools = vs[steps == "I"]

        Zs = [TRM]
        for Z in Zcools:
            Zs.append(Z[-1])

        Is = [TRM]
        for I in Icools:
            Is.append(I[-1])

        Zs_tot[:, :, i] = Zs
        Is_tot[:, :, i] = Is

    Zs_sum = np.flip(Zs_tot, axis=2) * weights
    Zs_sum = np.sum(Zs_sum, axis=2)

    Is_sum = np.flip(Is_tot, axis=2) * weights
    Is_sum = np.sum(Is_sum, axis=2)

    Is_sum -= Zs_sum

    Zs_mag = np.linalg.norm(Zs_sum, axis=1)
    Is_mag = np.linalg.norm(Is_sum, axis=1)

    return (Zs_sum, Is_sum, Zs_mag, Is_mag)


def uniaxial_relaxation_time(d, T, K):
    """
    Calculates the Neel relaxation time for an
    energy barrier at a particular temperature

    Parameters
    ------
    d: float
        Equivalent spherical volume diameter of
        particle

    T: float
        Temperature

    K: float
        Energy density of energy barrier

    Returns
    -------
    t: float
        Relaxation time
    """
    tau_0 = 1e-9
    kb = 1.380649e-23
    V = 4 / 3 * np.pi * (d / 2 * 1e-9) ** 3
    t = tau_0 * np.exp(K * V / (kb * (T + 273)))
    return t


def uniaxial_critical_size(K, T, t=100):
    """
    Calculates the critical size of a particle
    assuming only a uniaxial transition time

    Parameters
    ------
    K: float
        Energy barrier (J/m^3)

    T: float
        Temperature energy barrier is calculated at

    t: float
        Desired timescale to target

    Returns
    -------
    d: float
        Grain equivalent sphere volume diameter (nm)
    """
    tau_0 = 1e-9
    kb = 1.380649e-23
    V = np.log(t / tau_0) * kb * (T + 273) / K
    r = (V * 3 / (4 * np.pi)) ** (1 / 3)
    d = 2 * r
    return d * 1e9


def blocking_temperature(gel, d, i, j, block_t=100.0):
    """
    Calculates the blocking temperature associated
    with an energy barrier. Involves a minimization
    to obtain the correct time.

    Parameters
    ------
    gel: GEL object
        Energy landscape of particle to be considered

    d: float
        Size of particle (nm)

    i: int
        index of first state in barrier

    j: int
        index of second state in barrier

    block_t: float
        Timescale at which the barrier is considered
        'blocked'

    Returns
    -------
    T: float
        Blocking temperature
    """

    def loss_func(testT):
        """
        Single parameter loss function.
        """
        t, c, k = gel.bar_energy[i, j]
        K = energy_result(t, c, k, testT)
        obs_t = uniaxial_relaxation_time(d, testT, K)
        loss = (np.log(obs_t) - np.log(block_t)) ** 2
        return loss

    Ts = np.arange(gel.T_min, gel.T_max)
    loss = np.empty(Ts.shape)
    for k, T in enumerate(Ts):
        loss[k] = loss_func(T)
    T_start = Ts[loss == min(loss)][0]
    T_block = minimize(loss_func, T_start, method="Nelder-Mead").x[0]
    return T_block


def get_critical_sizes(energyLandscape, Blocking_Ts):
    sizes = []
    for T in Blocking_Ts:
        K = energyLandscape.get_params(T)["bar_e"][0, 1]
        size = uniaxial_critical_size(K, T)
        sizes.append(size)
    return sizes


def load_result(fname):
    return load_particle(fname)
