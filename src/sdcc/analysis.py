import numpy as np
import matplotlib.pyplot as plt


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
