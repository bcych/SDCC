import numpy as np
import matplotlib.pyplot as plt
from sdcc.energy import (
    demag_factors,
    get_material_parms,
    energy_surface,
    angle2xyz,
    energy_xyz,
    xyz2angle,
)
from sdcc.treatment import TreatmentStep
from jax import jit, grad
import jax.numpy as jnp
from jax import config
from sdcc.barriers import uniaxial_relaxation_time
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
from itertools import combinations

config.update("jax_enable_x64", True)
from sdcc.utils import oneoneones, oneonezeros, onezerozeros


def dimap(D, I):
    """
    Function to map directions  to x,y pairs in equal area projection.
    This function is reproduced from the PmagPy Project
    (https://github.com/PmagPy/Pmagpy).

    Copyright (c) 2023, PmagPy contributors
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
    notice,this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

    * Neither the name of PmagPy nor the names of its contributors may
    be used to endorse or promote products derived from this software
    without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
    COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
    ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.


    Parameters
    ----------
        D : list or array of declinations (as float)
        I : list or array or inclinations (as float)

    Returns
    -------
        XY : x, y values of directions for equal area projection [x,y]
    """
    try:
        D = float(D)
        I = float(I)
    except TypeError:  # is an array
        return dimap_V(D, I)
    # DEFINE FUNCTION VARIABLES
    # initialize equal area projection x,y
    XY = [0.0, 0.0]

    # GET CARTESIAN COMPONENTS OF INPUT DIRECTION
    X = angle2xyz(D, I)
    X = np.array(X)

    # CHECK IF Z = 1 AND ABORT
    if X[2] == 1.0:
        return XY  # return [0,0]

    # TAKE THE ABSOLUTE VALUE OF Z
    if X[2] < 0:
        # this only works on lower hemisphere projections
        X[2] = -X[2]

    # CALCULATE THE X,Y COORDINATES FOR THE EQUAL AREA PROJECTION
    # from Collinson 1983
    R = np.sqrt(1.0 - X[2]) / np.sqrt(X[0] ** 2 + X[1] ** 2)
    XY[1], XY[0] = X[0] * R, X[1] * R

    # RETURN XY[X,Y]
    return XY


def dimap_V(D, I):
    """
    Maps declinations and inclinations into equal area projections.
    This function is a modified version of the dimap function from
    pmagpy (https://github.com/PmagPy/PmagPy).

    Copyright (c) 2023, PmagPy contributors
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
    notice,this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

    * Neither the name of PmagPy nor the names of its contributors may
    be used to endorse or promote products derived from this software
    without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
    COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
    ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

    Parameters
    ----------
        D, I : numpy arrays

    Returns
    -------
        XY : array of equal area projections

    Examples
    --------
    >>> dimap_V([35,60,20],[70,80,-10])
    array([[0.140856382055789, 0.20116376126988 ],
       [0.106743548942519, 0.061628416716219],
       [0.310909633795401, 0.85421719834377 ]])
    """
    # GET CARTESIAN COMPONENTS OF INPUT DIRECTION
    X = angle2xyz(D, I)
    X = np.array(X)
    # CALCULATE THE X,Y COORDINATES FOR THE EQUAL AREA PROJECTION
    # from Collinson 1983
    R = np.sqrt(1.0 - abs(X[2])) / (np.sqrt(X[0] ** 2 + X[1] ** 2))
    XY = np.array([X[1] * R, X[0] * R]).transpose()

    # RETURN XY[X,Y]
    return XY


def plot_net(ax=None):
    """
    Draws circle and tick marks for equal area projection. Adapted
    from pmagpy (https://github.com/PmagPy/PmagPy).

    Copyright (c) 2023, PmagPy contributors
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
    notice,this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

    * Neither the name of PmagPy nor the names of its contributors may
    be used to endorse or promote products derived from this software
    without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
    COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
    ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

    Parameters
    ------
    ax: matplotlib axis or None
    axis to draw net onto (creates new axis if None)

    Returns
    -------
        None
    """
    if ax == None:
        fig, ax = plt.subplots()
        plt.clf()

    ax.axis("off")
    Dcirc = np.radians(np.arange(0, 361.0))
    Icirc = np.zeros(361, "f")
    Xcirc, Ycirc = [], []
    for k in range(361):
        XY = dimap(Dcirc[k], Icirc[k])
        Xcirc.append(XY[0])
        Ycirc.append(XY[1])
    ax.plot(Xcirc, Ycirc, "k")

    # put on the tick marks
    Xsym, Ysym = [], []
    for I in np.radians(range(10, 100, 10)):
        XY = dimap(0.0, I)
        Xsym.append(XY[0])
        Ysym.append(XY[1])
    ax.plot(Xsym, Ysym, "k+")
    Xsym, Ysym = [], []
    for I in np.radians(range(10, 90, 10)):
        XY = dimap(np.pi / 2, I)
        Xsym.append(XY[0])
        Ysym.append(XY[1])
    ax.plot(Xsym, Ysym, "k+")
    Xsym, Ysym = [], []
    for I in np.radians(range(10, 90, 10)):
        XY = dimap(np.pi, I)
        Xsym.append(XY[0])
        Ysym.append(XY[1])
    ax.plot(Xsym, Ysym, "k+")
    Xsym, Ysym = [], []
    for I in np.radians(range(10, 90, 10)):
        XY = dimap(3 * np.pi / 2, I)
        Xsym.append(XY[0])
        Ysym.append(XY[1])
    ax.plot(Xsym, Ysym, "k+")
    for D in np.radians(range(0, 360, 10)):
        Xtick, Ytick = [], []
        for I in np.radians(range(4)):
            XY = dimap(D, I)
            Xtick.append(XY[0])
            Ytick.append(XY[1])
        ax.plot(Xtick, Ytick, "k")

    ax.axis("equal")
    ax.axis((-1.05, 1.05, -1.05, 1.05))


def plot_cubeocta(ax, rot_mat=np.eye(3), kind="cube", proj="upper"):
    """
    Plots the edges of a cube or an octahedron on the equal area plot,
    along the edges of the magnetocrystalline directions

    Parameters
    ----------
    ax: matplotlib axis
    axis to be plotted to

    rot_mat: 3x3 numpy array
    rotation matrix for magnetocrystalline directions

    kind: string
    "cube" for cube edges, "octa" for octahedron edges

    proj: string
    "upper" for upper hemisphere, "lower" for lower hemisphere
    """
    if kind == "cube":
        combos = np.array(list(combinations(oneoneones, 2)))
        marker = "w--"
    elif kind == "octa":
        combos = np.array(list(combinations(onezerozeros, 2)))
        marker = "r"
    else:
        raise ValueError("kind must be cube or octa")
    combo_sums = np.sum(combos, axis=1)
    combo_sums = np.transpose(combo_sums.T / np.linalg.norm(combo_sums, axis=1))
    idxs = []
    combo_filter = (
        (np.isclose(oneonezeros[:, None], combo_sums)).any(axis=0).all(axis=1)
    )
    combos = combos[combo_filter]
    for combo in combos:
        lerp = np.linspace(combo[0], combo[1], 100)
        lerp = lerp.T / np.linalg.norm(lerp, axis=1)
        Xs = []
        Ys = []
        for l in lerp.T:
            D, I = xyz2angle(rot_mat @ l)
            if proj == "upper":
                if I >= 0:
                    X, Y = dimap(D, I)
                    Xs.append(X)
                    Ys.append(-Y)
            elif proj == "lower":
                if I <= 0:
                    X, Y = dimap(D, I)
                    Xs.append(X)
                    Ys.append(-Y)
            else:
                raise ValueError("proj must be upper or lower")

        ax.plot(Xs, Ys, marker, zorder=1.5, lw=1)


def plot_energy_surface(
    TMx,
    alignment,
    PRO,
    OBL,
    T=20,
    ext_field=jnp.array([0, 0, 0]),
    levels=10,
    n_points=100,
    projection="equirectangular",
    cubic_dirs=False,
    ax=None,
):
    """
    Plots an energy density surface for a single domain grain.

    Parameters
    ----------
    TMx: float
        Titanomagnetite composition (0 - 100)

    alignment: str
        Alignment of magnetocrystalline and shape axis. Either 'hard' or
        'easy' magnetocrystalline always aligned with shape easy.

    PRO: float
        Prolateness of ellipsoid (major / intermediate axis)

    OBL: float
        Oblateness of ellipsoid (intermediae / minor axis)

    T: float
        Temperature (degrees C)

    ext_field: numpy array
        Array of field_theta,field_phi,field_str where theta and phi are in
        radians and str is in Tesla.

    levels: int
        Number of contour levels

    n_points: int
        Number of grid points to evaluate the energy surface at.

    projection: string
        Either "equirectangular" or "stereonet".

    Returns
    -------
    None
    """
    # Get material parameters
    LMN = demag_factors(PRO, OBL)
    rot_mat, k1, k2, Ms = get_material_parms(TMx, alignment, T)

    # Get the energy surface
    thetas, phis, energies = energy_surface(
        k1, k2, rot_mat, Ms, LMN, ext_field=ext_field, n_points=n_points
    )

    energies -= np.amin(energies)

    # If equidimensional, plot an equidimensional plot
    if "equi" in projection.lower():
        if type(ax) == type(None):
            fig, ax = plt.subplots()
        ax.contour(
            np.degrees(thetas),
            np.degrees(phis),
            energies,
            levels=levels,
            cmap="viridis",
            antialiased=True,
        )
        contourf = ax.contourf(
            np.degrees(thetas),
            np.degrees(phis),
            energies,
            levels=levels,
            cmap="viridis",
            antialiased=True,
            linewidths=0.2,
        )
        plt.colorbar(contourf, label="$\Delta$ Energy Density (Jm$^{-3}$)", ax=ax)
        ax.set_xlabel(r"$\theta$", fontsize=14)
        ax.set_ylabel("$\phi$", fontsize=14)

    # If stereonet, plot on two stereonets (upper, lower)
    elif "stereo" in projection.lower():
        if type(ax) == type(None):
            fig = plt.figure(figsize=(9, 4))
            grid = fig.add_gridspec(100, 20)
            ax1 = fig.add_subplot(grid[:, :9])
            ax2 = fig.add_subplot(grid[:, 9:18])
            ax3 = fig.add_subplot(grid[5:95, 18:])
            ax = [ax1, ax2, ax3]
        plot_net(ax[0])
        if cubic_dirs:
            plot_cubeocta(ax[0], rot_mat, "octa", "upper")
            plot_cubeocta(ax[0], rot_mat, "cube", "upper")
        vmin = np.amin(energies)
        vmax = np.amax(energies)
        xs, ys = dimap(thetas[phis >= 0].flatten(), phis[phis >= 0].flatten()).T
        upper = ax[0].tricontourf(
            xs,
            ys,
            energies[phis >= 0].flatten(),
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            antialiased=True,
        )
        plot_net(ax[1])
        if cubic_dirs:
            plot_cubeocta(ax[1], rot_mat, "octa", "lower")
            plot_cubeocta(ax[1], rot_mat, "cube", "lower")
        xs, ys = dimap(thetas[phis <= 0].flatten(), phis[phis <= 0].flatten()).T
        lower = ax[1].tricontourf(
            xs,
            ys,
            energies[phis <= 0].flatten(),
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            antialiased=True,
        )
        ax[2].axis("off")
        bounds = MaxNLocator(levels + 1).tick_values(vmin, vmax)
        norm = BoundaryNorm(bounds, 256)
        mappable = ScalarMappable(norm=norm, cmap="viridis")
        plt.colorbar(mappable, ax=ax[2], label="$\Delta$ Energy Density (Jm$^{-3}$)")
    # Otherwise raise an error
    else:
        raise KeyError("Unknown projection type: " + projection)
    # fig.suptitle("SD Energy Surface TM" + str(TMx).zfill(2) + " AR %1.2f" % (PRO / OBL))
    plt.tight_layout()


@jit
def update(xyz, k1, k2, rot_mat, LMN, Ms, ext_field, lr):
    """
    Update for gradient descent function
    """
    gradient = grad(energy_xyz)(xyz, k1, k2, rot_mat, LMN, Ms, ext_field)
    delta_xyz = -lr * gradient
    return xyz + delta_xyz


@jit
def _descent_update(i, j, energies):
    """
    Computes the minimum energy in a 10x10 grid around energies[i,j],
    goes to that energy, like a fast gradient descent except not using
    gradients.
    """
    i_temps = []
    j_temps = []
    energy_list = []
    for i_temp in jnp.arange(-10, 11) + i:
        for j_temp in jnp.arange(-10, 11) + j:
            i_temps.append(jnp.clip(i_temp, 0, 1000))
            j_temps.append(j_temp % 1000)
            energies_new = energies[jnp.clip(i_temp, 0, 1000), j_temp % 1000]
            energies_orig = energies[i, j]
            energy_list.append(energies_new)
    i_temps = jnp.array(i_temps)
    j_temps = jnp.array(j_temps)
    energy_list = jnp.array(energy_list)
    i_new = i_temps[jnp.where(energy_list == jnp.min(energy_list), size=1)][0]
    j_new = j_temps[jnp.where(energy_list == jnp.min(energy_list), size=1)][0]
    return (i_new, j_new, energies[i_new, j_new])


def gradient_descent(
    max_iterations,
    threshold,
    xyz_init,
    k1,
    k2,
    rot_mat,
    LMN,
    Ms,
    ext_field=np.array([0, 0, 0]),
    learning_rate=1e-4,
):
    """
    Slow gradient descent function, more accurate than fast one
    """
    xyz = xyz_init
    xyz_history = xyz
    e_history = energy_xyz(xyz, k1, k2, rot_mat, LMN, Ms, ext_field)
    delta_xyz = jnp.zeros(xyz.shape)
    i = 0
    diff = 1.0e10

    while i < max_iterations and diff > threshold:
        xyz = update(xyz, k1, k2, rot_mat, LMN, Ms, ext_field, learning_rate)
        xyz_history = jnp.vstack((xyz_history, xyz))
        e_history = jnp.vstack(
            (e_history, energy_xyz(xyz, k1, k2, rot_mat, LMN, Ms, ext_field))
        )
        i += 1
        diff = jnp.absolute(e_history[-1] - e_history[-2])

    return xyz_history, e_history


def fast_path_to_min(energies, i, j):
    """
    Fast 'gradient descent' like function for getting energy barrier
    paths.
    """
    different = True
    i_history = [i]
    j_history = [j]
    e_history = [energies[i, j]]
    while different:
        i, j, e = _descent_update(i, j, energies)
        if e in e_history:
            different = False
        i_history.append(i)
        j_history.append(j)
        e_history.append(e)
    return (i_history, j_history)


def plot_energy_path(
    TM,
    alignment,
    PRO,
    OBL,
    mask,
    T=20,
    ext_field=np.array([0, 0, 0]),
    n_perturbations=10,
    n_saddles=5,
    projection="equirectangular",
    method="fast",
    **kwargs,
):
    """
    Plots the path for an energy surface by first finding energy
    barriers, then doing a gradient descent from nearby regions.
    Efficiency scales with n_perturbations * n_saddles. It's not a fast
    function either way.

    Parameters
    ------
    TMx: float
        Titanomagnetite composition (0 - 100)

    alignment: str
        Alignment of magnetocrystalline and shape axis. Either 'hard' or
        'easy' magnetocrystalline always aligned with shape easy.

    PRO: float
        Prolateness of ellipsoid (major / intermediate axis)

    OBL: float
        Oblateness of ellipsoid (intermediae / minor axis)

    mask: numpy array
        An array of True/False wherever the energy barriers are -> this
        could probably be taken out and the function could just calculate
        where they are.

    T: float
        Temperature (degrees C)

    ext_field: numpy array
        Array of field_theta,field_phi,field_str where theta and phi are in
        radians and str is in Tesla.

    n_perturbations: int
        number of perturbations to run from each energy barrier

    n_saddles: int
        number of barriers to find paths from.

    projection: str
        Either "equirectangular" or "stereonet".

    method: str
        Either fast or slow

    kwargs:
        arguments to pass to plot_energy_surface function

    Returns
    -------
    None
    """
    # First plot the energy surface background.
    plot_energy_surface(
        TM,
        alignment,
        PRO,
        OBL,
        T=T,
        ext_field=ext_field,
        projection=projection,
        **kwargs,
    )

    # Get material parameters
    rot_mat, k1, k2, Ms = get_material_parms(TM, alignment, T)
    LMN = demag_factors(PRO, OBL)

    # Get energy surface
    thetas, phis, energies = energy_surface(
        k1, k2, rot_mat, Ms, LMN, ext_field, n_points=1001
    )

    # If we have a mask which is at (phi = 90 or -90, set location to closest
    # saddlepoint range - boundary condition stuff
    mask_2 = mask[
        ((-1) & (energies[-2, :] == np.amin(energies[-2, :])))
        | ((0) & (energies[1, :] == np.amin(energies[1, :]))),
        :,
    ]
    mask_2 = mask_2.at[1:-1].set(True)
    mask = mask_2 & mask

    saddle_thetas = thetas[mask]
    saddle_phis = phis[mask]
    cs = np.random.choice(
        len(saddle_thetas), min(len(saddle_thetas), n_saddles), replace=False
    )
    descent_thetas = []
    descent_phis = []

    # If we're using the slow, (gradient descent) method to find the path
    # then use that routine
    if (
        "slow" in method.lower()
        or "gradient" in method.lower()
        or "descent" in method.lower()
    ):
        learning_rate = (
            1 / (np.amax(np.linalg.norm(np.gradient(energies), axis=0))) * 1e-3
        )
        for i in cs:
            for j in range(n_perturbations):
                start_theta = saddle_thetas[i]
                start_phi = saddle_phis[i]
                xyz = angle2xyz(start_theta, start_phi)
                xyz += np.random.normal(0, 5e-3, xyz.shape)
                xyz /= np.linalg.norm(xyz)
                result = gradient_descent(
                    10000,
                    1e-5,
                    xyz,
                    k1,
                    k2,
                    rot_mat,
                    LMN,
                    Ms,
                    ext_field=ext_field,
                    learning_rate=learning_rate,
                )
                descent_theta = [np.degrees(saddle_thetas[i])]
                descent_phi = [np.degrees(saddle_phis[i])]
                for r in result[0]:
                    theta, phi = np.degrees(xyz2angle(r / np.linalg.norm(r)))
                    theta = theta % 360
                    descent_theta.append(theta)
                    descent_phi.append(phi)
                descent_theta = np.array(descent_theta)
                descent_phi = np.array(descent_phi)
                descent_thetas.append(descent_theta)
                descent_phis.append(descent_phi)

    # If we're using the fast, (grid search) method to find the path,
    # then use that routine.
    elif "fast" in method.lower():
        jjs, iis = np.meshgrid(np.arange(1001), np.arange(1001))
        iis = iis[mask]
        jjs = jjs[mask]
        for c in cs:
            for ishift in range(-1, 2):
                for jshift in range(-1, 2):
                    i = np.clip(iis[c] + ishift, 0, 1000)
                    j = (jjs[c] + jshift) % 1000
                    i_hist, j_hist = fast_path_to_min(energies, i, j)
                    descent_thetas.append(np.degrees(thetas[i_hist, j_hist]))
                    descent_phis.append(np.degrees(phis[i_hist, j_hist]))

    else:
        raise KeyError(
            "method must contain one of the terms fast, slow, gradient, descent"
        )
    if "equi" in projection.lower():
        # Loop through runs
        for l in range(len(descent_thetas)):
            descent_theta = descent_thetas[l]
            descent_phi = descent_phis[l]

            # For equirectangular maps, each array must be split wjere
            # It crosses the 0/360 meridian to avoid drawing over
            # Center of map.
            splits = jnp.where(jnp.abs(jnp.diff(descent_theta)) >= 180)
            descent_theta = jnp.split(descent_theta, splits[0] + 1)
            descent_phi = jnp.split(descent_phi, splits[0] + 1)

            if len(splits) == 0:
                descent_theta = [descent_theta]
                descent_phi = [descent_phi]
            for k in range(len(descent_theta)):
                plt.plot(descent_theta[k], descent_phi[k], "r", alpha=1)

        plt.plot(np.degrees(saddle_thetas[cs]), np.degrees(saddle_phis[cs]), "w.")
    elif "stereo" in projection.lower():
        ax0 = plt.gcf().axes[0]
        ax1 = plt.gcf().axes[1]

        for l in range(len(descent_thetas)):
            descent_theta = np.radians(descent_thetas[l])
            descent_phi = np.radians(descent_phis[l])

            # Try except is here to catch scalar vs vector dimap output.
            try:
                thetas_plus, phis_plus = dimap(
                    descent_theta[descent_phi >= 0], descent_phi[descent_phi >= 0]
                ).T
            except:
                if len(descent_theta) > 1:
                    thetas_plus, phis_plus = dimap(
                        descent_theta[descent_phi >= 0], descent_phi[descent_phi >= 0]
                    )
                elif len(descent_theta) == 1 and descent_phi >= 0:
                    thetas_plus, phis_plus = dimap(descent_theta, descent_phi)

            ax0.plot(thetas_plus, phis_plus, "r", alpha=1)

            # Plot lower hemisphere on separate access
            try:
                thetas_minus, phis_minus = dimap(
                    descent_theta[descent_phi <= 0], descent_phi[descent_phi <= 0]
                ).T
            except:
                if len(descent_theta) > 1:
                    thetas_minus, phis_minus = dimap(
                        descent_theta[descent_phi <= 0], descent_phi[descent_phi <= 0]
                    )
                elif len(descent_theta) == 1 and descent_phi <= 0:
                    thetas_minus, phis_minus = dimap(descent_theta, descent_phi)

            ax1.plot(thetas_minus, phis_minus, "r", alpha=1)

        try:
            saddle_x, saddle_y = dimap(saddle_thetas[cs], saddle_phis[cs]).T
        except:
            saddle_x, saddle_y = dimap(saddle_thetas[cs], saddle_phis[cs])

        if len(saddle_phis) > 1:
            ax0.plot(
                saddle_x[saddle_phis[cs] >= 0], saddle_y[saddle_phis[cs] >= 0], "w."
            )
            ax1.plot(
                saddle_x[saddle_phis[cs] <= 0], saddle_y[saddle_phis[cs] <= 0], "w."
            )
        elif len(saddle_phis) == 1 and saddle_phis >= 0:
            ax0.plot(saddle_x, saddle_y, "w.")
        else:
            ax1.plot(saddle_x, saddle_y, "w.")

    else:
        raise KeyError("Unknown projection type: " + projection)


def plot_minima(minima_thetas, minima_phis, projection="equirectangular", ax=None):
    """
    Plots a set of LEM states on an energy surface plot as numerals.

    Inputs
    ------
    minima_thetas,minima_phis: numpy arrays
    Locations of the minima on the surface.

    projection: str
    Either "equirectangular" or "stereonet".
    """
    if ax == None:
        ax = plt.gcf().get_axes()
    for i in range(len(minima_thetas)):
        if "equi" in projection.lower():
            plt.text(
                np.degrees(minima_thetas[i]),
                np.degrees(minima_phis[i]),
                i,
                color="w",
                va="center",
                ha="center",
            )
        elif "stereo" in projection.lower():
            if minima_phis[i] <= 0:
                theta, phi = dimap(minima_thetas[i], -minima_phis[i])

                ax[1].text(
                    theta,
                    phi,
                    i,
                    color="w",
                    va="center",
                    ha="center",
                    path_effects=[pe.withStroke(linewidth=1, foreground="black")],
                )
            if minima_phis[i] >= 0:
                theta, phi = dimap(minima_thetas[i], minima_phis[i])
                ax[0].text(
                    theta,
                    phi,
                    i,
                    color="w",
                    va="center",
                    ha="center",
                    path_effects=[pe.withStroke(linewidth=1, foreground="black")],
                )


def plot_barriers(barrier_thetas, barrier_phis, projection="equirectangular", ax=None):
    """
    Plots a set of barrier locations on an energy surface plot as
    numerals.

    Inputs
    ------
    barrier_thetas,barrier_phis: numpy arrays
    Locations of the barriers on the surface.

    projection: str
    Either "equirectangular" or "stereonet".

    ax: matplotlib axes
    Axes to plot on top of.
    """
    if ax == None:
        ax = plt.gcf().get_axes()
    for i in range(len(barrier_thetas)):
        for j in range(i, len(barrier_thetas)):
            if not np.isinf(barrier_thetas[i, j]):
                if "equi" in projection.lower():
                    ax[0].text(
                        np.degrees(barrier_thetas[i, j]),
                        np.degrees(barrier_phis[i, j]),
                        str(i) + "," + str(j),
                        color="tomato",
                        va="center",
                        ha="center",
                        path_effects=[pe.withStroke(linewidth=1, foreground="black")],
                    )
                elif "stereo" in projection.lower():
                    if barrier_phis[i, j] <= 0:
                        theta, phi = dimap(barrier_thetas[i, j], -barrier_phis[i, j])
                        ax[1].text(
                            theta,
                            phi,
                            str(i) + "," + str(j),
                            color="tomato",
                            va="center",
                            ha="center",
                            path_effects=[
                                pe.withStroke(linewidth=1, foreground="black")
                            ],
                        )
                    if barrier_phis[i, j] >= 0:
                        theta, phi = dimap(barrier_thetas[i, j], barrier_phis[i, j])
                        ax[0].text(
                            theta,
                            phi,
                            str(i) + "," + str(j),
                            color="tomato",
                            va="center",
                            ha="center",
                            path_effects=[
                                pe.withStroke(linewidth=1, foreground="black")
                            ],
                        )


def plot_routine(steps):
    """
    Plots a set of thermal steps for a particular experiment or routine

    Parameters
    ------
    steps: list of treatment.TreatmentStep objects

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    plt.xlabel("t (hrs)")
    plt.ylabel(r"T ($^\circ$C)")
    for step in steps:
        if step.step_type == "cooling":
            c = "b"
        elif step.step_type == "heating":
            c = "r"
        else:
            c = "purple"
        if step.field_strs[0] > 0:
            ls = "-"
        else:
            ls = "--"

        plt.plot(step.ts / 3600, step.Ts, color=c, linestyle=ls)


def energy_matrix_plot(barriers):
    """
    Plots the energy barriers between states as a matrix pair-plot.

    Parameters
    ------
    barriers: numpy array
        Matrix of energy barriers

    Returns
    -------
    None
    """
    plt.imshow(barriers, cmap="magma", vmin=0)
    ticks = []
    for lim in np.arange(0, np.amax(barriers[~np.isinf(barriers)]), 1000):
        relevant_barriers = barriers[(barriers >= lim) & (barriers <= lim + 1000)]
        if len(relevant_barriers) > 0:
            ticks.append(int(np.mean(relevant_barriers)))
    plt.colorbar(label="Energy Barrier (Jm$^{-3}$)", ticks=ticks)
    plt.xlabel("Minima Direction j")
    plt.ylabel("Minima Direction i")
    plt.xticks(np.arange(0, len(barriers)))
    plt.yticks(np.arange(0, len(barriers)))


def plot_pullaiah_curves(
    gel, ds, i, j, ax=None, plot_size=True, color="k", add_ticks=True, **kwargs
):
    """
    Plots Pullaiah curves for a particular energy barrier in a grain as a function
    of size.

    Parameters
    ------
    gel: sdcc.barriers.gel object
        Energy landscape that can calculate barriers as a function
        of temperature.

    ds: float or array of floats
        Equivalent sphere volume diameters of grains to plot.

    i: int
        Index of state we are switching from in energy barrier.

    j: int
        Index of state we are switching to in energy barrier.

    ax: None or matplotlib axis
        axis to plot data to. If no axis specified, creates
        a new figure.

    plot_size: bool
        If True, plots size next to pullaiah curve as text.

    color: string or array of floats
        Matplotlib color for plotting

    add_ticks:
        Whether to add y ticks to plot.

    Returns
    -------
    None
    """
    s_in_yr = 3.154e7
    ts = [1, 10, 100, 3600, 3600 * 24, 3600 * 24 * 30, s_in_yr]
    for k in range(1, 10):
        ts.append(s_in_yr * 10**k)
    ts.append(4.5 * 10**9 * s_in_yr)
    ts = np.log10(ts)  #
    if ax == None:
        fig, ax = plt.subplots()
    ilist = []
    jlist = []
    Ts = np.arange(gel.T_min, gel.T_max, 1)
    for T in Ts:
        barriers = gel.get_params(T)["bar_e"]
        LEMs = gel.get_params(T)["min_e"]
        if np.isinf(LEMs[i]):
            i = np.arange(len(barriers[i]))[barriers[i] == np.amin(barriers[i])][-1]
        if np.isinf(LEMs[j]):
            j = np.arange(len(barriers[j]))[barriers[j] == np.amin(barriers[j])][-1]
        ilist.append(i)
        jlist.append(j)

    for d in ds:
        rts = []

        for k, T in enumerate(Ts):
            barriers = gel.get_params(T)["bar_e"]
            barrier = max(barriers[ilist[k], jlist[k]], 0)
            relax = uniaxial_relaxation_time(d, T, barrier)
            rts.append(relax)

        ax.plot(Ts, np.log10(rts), color=color, lw=1, **kwargs)
        logrts = np.log10(rts)
        center_rts = logrts[(logrts >= min(ts)) & (logrts <= max(ts))]
        if len(center_rts > 0):
            minrt = min(center_rts)
            maxrt = max(center_rts)
            meanrt = (minrt + maxrt) / 2
            meandiff = (logrts - meanrt) ** 2
            meanT = Ts[(meandiff == np.amin(meandiff))][0]
            angle = np.degrees(
                np.arctan(
                    np.gradient(logrts / (max(ts) - min(ts)), 1.5 * Ts / 600)[
                        (meandiff == np.amin(meandiff))
                    ][0]
                )
            )
            if plot_size:
                ax.text(
                    meanT + 7,
                    meanrt + 0.25,
                    f"{d} nm",
                    rotation=angle,
                    ha="center",
                    va="center",
                    color=color,
                )
    if add_ticks:
        yticks = [
            "1s",
            "10s",
            "100s",
            "1h",
            "1d",
            "1m",
            "1y",
            "10y",
            "100y",
            "1000y",
            "1e4y",
            "1e5y",
            "1My",
            "10My",
            "100My",
            "1Gy",
            "4.5Gy",
        ]
        ax.set_yticks(ts, yticks)
        ax.set_xticks(np.arange(0, 650, 50))
        ax.set_xticks(np.arange(0, 610, 10), minor=True)
        ax.set_ylim(min(ts), max(ts))

        ax.set_xlabel(r"T ($^\circ$C)")
        ax.set_ylabel("Relaxation Time")

        ax.tick_params(
            axis="x",
            which="minor",
            bottom=True,
            top=True,
            labelbottom=True,
            labeltop=True,
        )
        ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
        ax.grid()


def plot_relax_experiment(vs, relax_routine, ax=None):
    """
    Plots a VRM decay experiment for a mono-dispersion of particles

    Parameters
    ------
    vs: list
        Result from mono-dispersion

    relax_routine: list
        Relaxation time routine

    ax: matplotlib axis
        Axis to plot to

    Returns
    -------
    None
    """

    if ax == None:
        fig, ax = plt.subplots()

    # The first set of
    TRM_vectors = vs[0]
    TRM_ts = relax_routine[0].ts
    relax_vectors = vs[1]
    relax_ts = relax_routine[1].ts

    TRM_mags = np.linalg.norm(TRM_vectors, axis=1)
    relax_mags = np.linalg.norm(relax_vectors, axis=1)

    ax.plot(TRM_ts, TRM_mags, "b", label="TRM acquisition")
    ax.plot(relax_ts, relax_mags, "k", label="TRM decay")
    ax.axvline(TRM_ts[-1], color="r")
    ax.text(
        TRM_ts[-1] * 2,
        0.9 * TRM_mags[-1],
        "Field switched off",
        rotation=90,
        color="r",
        ha="left",
        va="top",
    )
    interpx = relax_mags / TRM_mags[-1]
    interpy = np.log10(relax_ts)
    paleo_relax_time = 10 ** np.interp(1 / np.e**2, np.flip(interpx), np.flip(interpy))

    ax.axvline(paleo_relax_time, color="k")
    ax.text(
        paleo_relax_time * 2,
        TRM_mags[-1] / np.e**2,
        "Paleomagnetic Relaxation Time",
        rotation=90,
        ha="left",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Moment (A$m^2$)")
    ax2 = plt.twinx()
    ax2.set_ylim(0, 1.2)
    ax.set_ylim(0, 1.2 * TRM_mags[-1])
    ax2.set_ylabel("Moment/TRM")
    ax.semilogx()

    ax2.annotate(
        text="",
        xy=(TRM_ts[-1], 1.15),
        xytext=(ax2.get_xlim()[0], 1.15),
        arrowprops=dict(arrowstyle="<->"),
        color="b",
    )
    ax2.annotate(
        text="Acquisition",
        xy=(np.exp((np.log(TRM_ts[-1]) + np.log(ax2.get_xlim()[0])) / 2), 1.14),
        ha="center",
        va="top",
    )
    ax2.annotate(
        text="",
        xy=(TRM_ts[-1], 1.15),
        xytext=(paleo_relax_time, 1.15),
        arrowprops=dict(arrowstyle="<->"),
        color="k",
    )
    ax2.annotate(
        text="Relaxation",
        xy=(np.exp((np.log(TRM_ts[-1]) + np.log(paleo_relax_time)) / 2), 1.14),
        ha="center",
        va="top",
    )
    return None


def plot_arai(Zs_mag, Is_mag, B_anc, B_lab, temp_steps, ax=None):
    """
    Plots an Arai plot, and expected Arai plot line for a
    paleointensity experiment.

    Parameters
    ----------
    Zs_mag, Is_mag: length n arrays
        Zero-field and in field data for Arai plot

    B_anc, B_lab: floats
        Ancient and laboratory fields used in paleointensity simulation

    Returns
    -------
    None
    """
    if ax == None:
        fig, ax = plt.subplots()
    ax.plot(Is_mag, Zs_mag, "k")
    ax.plot(Is_mag, Zs_mag, "ro", markeredgecolor="k")
    for i, T in enumerate(temp_steps):
        if (T >= 300) & (T <= 560):
            plt.text(Is_mag[i] + 0.01, Zs_mag[i] + 0.01, str(T))
    ax.plot([0, B_lab / B_anc], [1, 0], "g", label="Ideal line")
    plt.ylabel("NRM/NRM0")
    plt.xlabel("pTRM/NRM0")
    return None
