import numpy as np
import matplotlib.pyplot as plt
from sdcc.treatment import CoolingStep
from sdcc.barriers import uniaxial_critical_size


def analyze_hyst_data(vs, steps, d, hels, plot=False, ax=None):
    init_ramp = vs[0][:, 0]
    back_ramp = vs[1][:, 0]
    forward_ramp = vs[2][:, 0]
    Mrs = back_ramp[steps[1].field_strs == 0][0]
    Mrs2 = np.abs(back_ramp[steps[2].field_strs == 0][0])
    Mrs = (Mrs2 + Mrs) / 2
    Msat = hels.HEL_list[0].get_params(0)["Ms"]

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


def get_critical_sizes(energyLandscape, Blocking_Ts):
    sizes = []
    for T in Blocking_Ts:
        K = energyLandscape.get_params(T)["bar_e"][0, 1]
        size = uniaxial_critical_size(K, T)
        sizes.append(size)
    return sizes
