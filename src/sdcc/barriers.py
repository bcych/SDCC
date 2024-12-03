### The following module calculates energy barriers between local energy minima
### In a single domain surface.

### IMPORT STATEMENTS ###
import numpy as np
from copy import deepcopy
from jax import jit
import jax.numpy as jnp
from sdcc.energy import (
    demag_factors,
    get_material_parms,
    energy_surface,
    calculate_anisotropies,
    angle2xyz,
    xyz2angle,
)
from jax import config
from skimage import measure
from itertools import combinations, product
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import pickle
from scipy.interpolate import splprep, splrep, BSpline
from sdcc.utils import fib_sphere, calc_d_min, fib_hypersphere
from scipy.optimize import minimize

config.update("jax_enable_x64", True)
### GENERIC FUNCTIONS ###
### These functions are used by multiple energy barrier calculation
### methods. They are handy for calculating the minimum energy barriers


def find_global_minimum(thetas, phis, energies, mask=None):
    """
    Finds the global minimum of a masked region of a theta-phi map

    Inputs
    ------
    thetas: numpy array
    grid of theta angles

    phis: numpy array
    grid of phi angles

    energies: numpy array
    grid of energies corresponding to theta, phi angles

    mask: numpy array
    grid of True or False indicating the location within the grid which
    should be evaluated to find the minimum

    Returns
    -------
    best_coords: numpy array
    Indices of minima

    best_theta: float
    Theta value of minima

    best_phi: float
    Phi value of minima

    best_energy: float
    Energy value of minima
    """
    if np.all(mask == None):
        mask = np.full(energies.shape, True)
    masked_thetas = thetas[mask]
    masked_phis = phis[mask]
    masked_energies = energies[mask]
    best_energy = np.amin(masked_energies)
    best_theta = masked_thetas[masked_energies == best_energy][0]
    best_phi = masked_phis[masked_energies == best_energy][0]
    best_coords = np.where((thetas == best_theta) & (phis == best_phi))
    return (best_coords, best_theta, best_phi, best_energy)


def wrap_labels(labels):
    """
    Wraps a set of labels assigned to an array so that the labels in
    the first and last columns are the same - this is useful as the
    arrays we use wrap at theta = 0 and 2*pi.

    Inputs
    ------
    Labels: array
    Array of integer labels for the "drainage basins" for each LEM

    Returns
    -------
    Labels: array
    Array of wrapped labels.
    """
    for l in jnp.unique(labels[:, 0]):
        locs = jnp.where(labels[:, 0] == l)
        wrapped_loc = jnp.unique(labels[locs, -1])
        for m in wrapped_loc:
            labels[labels == m] = l
    return labels


def segment_region(mask):
    """
        Splits an energy array into a segmented regions according to a mask
        Produces a set of labels for the array. Creates regions of
        "drainage basins" for the array.

        Inputs
        ------
        mask: numpy array
        mask of regions to segment
    
        Returns
        -------
        labels: numpy array
        array of drainage basin labels.
    """
    labels = measure.label(mask)
    labels = wrap_labels(labels)
    return labels


def get_minima(thetas, phis, energies, labels):
    """
    Gets the minimum energies for each label in an array with labelled
    regions.

    Inputs
    ------
    thetas: numpy array
    grid of theta angles

    phis: numpy array
    grid of phi angles

    energies: numpy array
    grid of energies corresponding to theta, phi angles

    labels: numpy array
    grid of integer labels for each drainage basin

    Returns
    -------
    theta_coords, phi_coords: arrays
    arrays of theta and phi coordinates of each minimum.
    """
    theta_coords = []
    phi_coords = []
    temp_energies = []

    for label in np.unique(labels):
        mask = labels == label
        temp_coords, temp_theta, temp_phi, temp_energy = find_global_minimum(
            thetas, phis, energies, mask
        )
        theta_coords.append(float(temp_theta))
        phi_coords.append(float(temp_phi))
        temp_energies.append(temp_energy)

    theta_coords = np.array(theta_coords)
    phi_coords = np.array(phi_coords)
    temp_energies = np.array(temp_energies)
    theta_coords = theta_coords[temp_energies != max(temp_energies)]
    phi_coords = phi_coords[temp_energies != max(temp_energies)]
    return (theta_coords, phi_coords)

### SKIMAGE WATERSHED IMPLEMENTATION ###
### These functions calculate ALL energy barriers for a specimen using the
### skimage implementation of the watershed algorithm. This is far faster and
### more efficient than the handwritten version and can quickly be used to
### compute all the energy barriers for a given SD energy surface. It's rather
### quick to do this.


def dijkstras(i, j, barriers):
    """
    A modified version of dijkstras algorithm used to "prune" energy
    barriers between states which already have a smaller barrier
    between them.

    Inputs
    ------
    i, j: ints
    Indices for barriers

    barriers: numpy array
    matrix of energy barriers

    returns:
    barriers: numpy array
    barrier array with steps pruned
    """
    visited = np.full(len(barriers), False)
    distances = barriers[i]
    distances[np.isinf(distances)] = np.inf
    visited[i] = True
    while visited[j] is False:
        temp_distances = np.copy(distances)
        temp_distances[visited] = np.inf

        k = np.where((temp_distances == np.min(temp_distances)))[0]
        if isinstance(k, float):
            k = k[0]
        new_distances = np.copy(barriers[k])
        new_distances[visited] = np.inf
        new_distances[np.isinf(new_distances)] = np.inf
        new_distances[(new_distances < distances[k])] = distances[k]
        distances = np.amin(np.array([distances, new_distances]), axis=0)
        visited[k] = True
    if distances[j] * 1.001 < barriers[i, j]:
        barriers[i, j] = np.inf
    return barriers


def prune_energies(barriers, thetas, phis):
    """
    Uses modified dijkstras algorithm to "prune" extraneous barriers

    Inputs
    ------
    barriers: numpy array
    Matrix of energy barriers

    thetas: numpy array
    Theta locations of barriers

    phis: numpy array
    Phi locations of barriers

    Returns
    -------
    temp_barriers: numpy array
    "Pruned" barrier array

    thetas: numpy array
    "Pruned" theta array

    phis: numpy array
    "Pruned" phi array
    """
    temp_barriers = barriers
    for i in range(barriers.shape[0]):
        for j in range(barriers.shape[1]):
            if not np.isinf(barriers[i, j]) and j != i:
                temp_barriers = dijkstras(i, j, barriers)
    thetas[np.isinf(temp_barriers)] = -np.inf
    phis[np.isinf(temp_barriers)] = -np.inf
    return (temp_barriers, thetas, phis)


def get_min_regions(energies, markers=None):
    """
    Partitions an SD energy surface into a set of "drainage basins"
    each of which contain an LEM state. This uses the watershed
    algorithm in scipy. We tile the maps so the connectivity
    constraints for this algorithm work. The minimum points along the
    drainage divides between two watersheds will later be used as the
    energy barriers.

    Inputs
    ------
    energies: numpy array
    SD energy surface

    markers: None or numpy array
    If supplied, uses "markers" as the starting minima to calculate the
    drainage divides

    Returns
    -------
    min_coords: numpy array
    indices of the minima in each region (should correspond to markers
    if this was supplied.

    labels: numpy array
    array of labelled drainage divides corresponding to the
    """

    # tile energies to preserve connectivity relationship
    tiled_energies = np.tile(energies, 2)

    # If using markers, tile those.
    if type(markers) != type(None):
        tiled_markers = np.tile(markers, 2)  # tile markers
        tiled_markers = tiled_markers.astype("int")

    else:
        tiled_markers = None

    # take watershed of image
    labels_old = watershed(tiled_energies, connectivity=1, markers=tiled_markers)

    # Take center part of map (this is done so that wrapping is not an issue)
    labels = labels_old[:, 500:1501]

    rolled_energies = np.roll(energies, 501, axis=1)  # align energies with label
    # map

    # Change things on the right edge to have the same label as things on left
    for i in np.unique(labels[:, -1]):
        # Adjust minima accordingly
        if len(labels[labels == i]) > 0:
            minimum = np.where(
                (labels == i)
                & (rolled_energies == np.amin(rolled_energies[labels == i]))
            )
        else:
            minimum = [[], []]
        # Special case for if minimum on the boundary
        if 1000 in minimum[1]:
            minindex = np.where(minimum[1] == 1000)
            j = labels[minimum[0][minindex], 0]
            if type(j) != int:
                j = j[0]
            labels[labels == j] = i

    # Change things on the left edge to have the same label as things on right
    for i in np.unique(labels[:, 0]):
        if len(labels[labels == i]) > 0:
            minimum = np.where(
                (labels == i)
                & (rolled_energies == np.amin(rolled_energies[labels == i]))
            )
        else:
            minimum = [[], []]
        if 0 in minimum[1]:
            minindex = np.where(minimum[1] == 0)
            j = labels[minimum[0][minindex], -1]
            if type(j) != int:
                j = j[0]
            labels[labels == j] = i

    # align labels with energies
    labels = np.roll(labels, -501, axis=1)

    # Create an array of minima coordinates.
    min_coords = []
    for i in np.unique(labels):
        where = np.where((labels == i) & (energies == np.amin(energies[labels == i])))
        if len(where[0]) > 1:
            where = (np.array([where[0][0]]), np.array([where[1][0]]))
        min_coords.append(where)
    return (min_coords, labels)


def construct_energy_mat_fast(thetas, phis, energies, labels):
    """
    Constructs an energy matrix from a set of labels and energies
    Does so in a rapid fashion by finding the minimum along an edge
    between two minima.

    Inputs
    ------
    thetas: numpy array
    grid of theta angles

    phis: numpy array
    grid of phi angles

    energies: numpy array
    grid of energies corresponding to theta, phi angles

    labels: numpy array
    grid of integer labels for each drainage basin

    Returns
    -------
    theta_mat: numpy array
    matrix of energy barrier thetas between state i and j

    phi_mat: numpy array
    matrix of energy barrier phis between state i and j

    energy_mat: numpy array
    matrix of energy barriers between state i and state j
    """
    labels_pad_v = np.pad(labels, ((1, 1), (0, 0)), mode="edge")
    labels_pad_h = np.pad(labels, ((0, 0), (1, 1)), mode="wrap")
    labels_pad = np.pad(labels_pad_v, ((0, 0), (1, 1)), mode="wrap")

    # Instead of looping through every neighbor for each pixel in the energy
    # matrix, simply calculate the difference from shifted pixels in the array

    pad_ul = labels_pad[:-2, :-2]
    pad_u = labels_pad_v[:-2, :]
    pad_ur = labels_pad[:-2, 2:]
    pad_l = labels_pad_h[:, :-2]
    pad_r = labels_pad_h[:, 2:]
    pad_bl = labels_pad[2:, :-2]
    pad_b = labels_pad_v[2:, :]
    pad_br = labels_pad[2:, 2:]

    # List the shifts and loop through them.
    shifts = [pad_ul, pad_u, pad_ur, pad_l, pad_r, pad_bl, pad_b, pad_br]

    theta_mat = np.full((len(np.unique(labels)), len(np.unique(labels))), -np.inf)
    phi_mat = np.full((len(np.unique(labels)), len(np.unique(labels))), -np.inf)
    energy_mat = np.full((len(np.unique(labels)), len(np.unique(labels))), -np.inf)

    # Loop through the combinations of i and j
    for i, j in combinations(range(len(np.unique(labels))), 2):
        l = np.unique(labels)[i]
        m = np.unique(labels)[j]
        edge_filter = np.full(labels.shape, False)

        # loop through the shifts and find the edges
        for shift in shifts:
            edge_filter = edge_filter | ((labels == l) & (shift == m))
            edge_filter = edge_filter | ((labels == m) & (shift == l))

        # Get the MINIMUM energy and its location along the edge
        # This is the energy barrier!
        if len(energies[edge_filter]) > 0:
            min_energy = np.amin(energies[edge_filter])
            energy_mat[i, j] = min_energy - np.amin(energies[labels == l])
            energy_mat[j, i] = min_energy - np.amin(energies[labels == m])
            theta_mat[i, j] = thetas[(energies == min_energy) & (edge_filter)][0]
            phi_mat[i, j] = phis[(energies == min_energy) & (edge_filter)][0]
            theta_mat[j, i] = theta_mat[i, j]
            phi_mat[j, i] = phi_mat[i, j]

    return (theta_mat, phi_mat, energy_mat)


def fix_minima(min_coords, energies, max_minima):
    """
    Reduces a set of minima on an energy surface to some nominal
    "maximum number" (usually set by the magnetocrystalline anisotropy)
    Trims out the lowest minima and makes sure things on the border
    wrap for good minima.

    Inputs
    ------
    min_coords: list
    List of indices of LEM values on our grid

    energies: numpy array
    SD energy surface as a function of direction

    max_minima: int
    Nominal maximum number of minima

    Returns
    -------
    new_markers: numpy array
    Corrected indices of energy minima.
    """

    # First work out the energies
    min_coords = np.array(min_coords).T[0].T
    min_energies = []
    for i in range(len(min_coords)):
        min_energy = energies[min_coords[i, 0], min_coords[i, 1]]
        min_energies.append(min_energy)
    min_energies = np.array(min_energies)

    # Keep removing the largest energies until you end up with
    # max_minima
    while len(min_coords) > max_minima:
        # First check how many LEMs we're dropping
        max_filter = min_energies == max(min_energies)
        max_len = len(min_energies[max_filter])

        # If that's too many, drop them one by one
        if len(min_coords) - max_len > max_minima:
            where = np.where(max_filter)[0][0]
            max_filter = range(len(min_coords)) == where

        # min filter is the inverse of max_filter
        min_filter = ~max_filter

        min_coords = min_coords[min_filter]
        min_energies = min_energies[min_filter]

    # Construct array of new markers for watershed.
    new_markers = np.zeros(energies.shape)
    i = 0
    for new_coord in min_coords:
        i += 1
        new_markers[new_coord[0], new_coord[1]] = i
        if new_coord[1] == 0:
            new_markers[new_coord[0], 1000] = i
        if new_coord[1] == 1000:
            new_markers[new_coord[0], 0] = i
    return new_markers


def merge_similar_minima(min_coords, thetas, phis, energies, tol):
    """
    Merges together close together LEM states assuming they're the same
    state. Takes the state with lowest energy as the "true" minimum.

    Inputs
    ------
    min_coords: list
    List of minima coordinate values.

    thetas,phis: numpy arrays
    Grid of theta,phi values on which energies are evaluated.

    energies: numpy array
    Energies at theta, phi locations.

    tol: float
    Angular distance in degrees, below which two states considered the
    same.

    Returns
    -------
    min_coords: list
    List of minima index values:

    markers: numpy array
    Locations of these minima in the energy matrix.
    """

    # Find your thetas phis, energies associated with your minima
    min_coords_new = np.array(min_coords).T[0]
    theta_list = thetas[min_coords_new[0], min_coords_new[1]]
    phi_list = phis[min_coords_new[0], min_coords_new[1]]
    energy_list = energies[min_coords_new[0], min_coords_new[1]]

    # Find angular differences
    xyz = angle2xyz(theta_list, phi_list)
    diffs = np.dot(xyz.T, xyz)
    cos_lim = np.cos(np.radians(tol))  # how small do we want to go
    dis, djs = np.where(diffs >= cos_lim)

    # Obviously the angular distance to same state is the same so remove
    # these.
    dis_u = dis[dis != djs]
    djs_u = djs[dis != djs]

    # Get the unique states that might need replacing
    udis = np.unique(dis_u)

    # Loop through and find groups of states
    complete = np.array([])
    groups = []
    for i in udis:
        # If we already have grouped this ignore it
        # N.B. what if there's two minima two degrees apart each, this
        # Would only count the two with closest indices - hopefully ok
        if i in complete:
            pass
        # Otherwise find everything that's got that index paired with it
        else:
            group = np.intersect1d(djs[dis == i], dis[djs == i])
            complete = np.append(complete, group)
            groups.append(group)

    dropped = np.array([])
    # Loop through groups and find which minima to ignore "drop"
    for group in groups:
        group_energies = energy_list[group]
        drop = group[group_energies != min(group_energies)]  # Keep maximum
        while len(drop) < len(group) - 1:  # If you removed too few
            # Kick another out!
            drop = np.append(drop, group[~np.isin(group, drop)][0])
        dropped = np.append(dropped, drop)

    # Delete the stuff in dropped from min_coords_new
    min_coords_new = min_coords_new[:, ~np.isin(range(len(theta_list)), dropped)]

    # Make the markers from our new min_coords
    new_markers = np.zeros(energies.shape)
    i = 0
    for new_coord in min_coords_new.T:
        i += 1
        new_markers[new_coord[0], new_coord[1]] = i
        # Loop condition
        if new_coord[1] == 0:
            new_markers[new_coord[0], 1000] = i
        if new_coord[1] == 1000:
            new_markers[new_coord[0], 0] = i

    # Remake min_coords in correct format!
    min_coords = []
    for i in np.unique(new_markers):
        if i != 0:
            where = np.where(
                (new_markers == i) & (energies == np.amin(energies[new_markers == i]))
            )
            if len(where[0]) > 1:
                where = (np.array([where[0][0]]), np.array([where[1][0]]))
            min_coords.append(where)
    return (min_coords, new_markers)


def find_all_barriers(
    TMx,
    alignment,
    PRO,
    OBL,
    T=20,
    ext_field=np.array([0, 0, 0]),
    prune=True,
    trim=True,
    tol=2.0,
    return_labels=False,
):
    """
    Finds all the minimum energy states in an SD grains and the
    barriers between them.

    Inputs
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

    T: float
    Temperature (degrees C)

    ext_field: numpy array
    Array of field_theta,field_phi,field_str where theta and phi are in
    radians and str is in Tesla.

    prune: boolean
    if True, uses dijkstras algorithm to prune extraneous barriers from
    the result.

    trim: boolean
    if True, removes minima with more barriers than the number of
    magnetocrystalline easy directions. Can cause problems if tol is
    set too low, because some close together minima may be chosen
    instead of distinct ones, removing legitimate minima.

    tol: float
    Angular distance in degrees between minima below which they're
    considered  "the same" state. Used to fix numerical errors where
    multiple minima are found in extremely flat regions of the
    energy surface.

    return_labels: boolean
    If True, returns the "labels" array. Default behavior is False.

    Returns
    -------
    theta_list, phi list: numpy arrays
    Arrays of the minimum energy directions in the SD energy surface

    min_energy_list: numpy array
    Minimum energy at the minimum

    theta_mat,phi_mat: numpy arrays
    Arrays of the directions associated with the energy barriers.

    barriers: numpy array
    Matrix of energy barriers between state i and state j.
    """

    # Set up parameters
    rot_mat, k1, k2, Ms = get_material_parms(TMx, alignment, T)
    LMN = demag_factors(PRO, OBL)
    easy_axis = calculate_anisotropies(TMx)[0]

    # Get a number of minima associated with the magnetocrystalline
    # directions (unused)
    if easy_axis == "1 1 1":
        n_minima = 8
    elif easy_axis == "1 0 0":
        n_minima = 6
    elif easy_axis == "1 1 0":
        n_minima = 12
    else:
        raise ValueError("Something wrong with anisotropy field")
    # Get the energy surface
    thetas, phis, energies = energy_surface(
        k1, k2, rot_mat, Ms, LMN, ext_field, n_points=1001
    )

    # Get the drainage divides of the LEM states
    theta_list = []
    phi_list = []
    min_energy_list = []
    min_coords, labels = get_min_regions(energies)

    # Run a second pass to eliminate similar minima
    min_coords_new, markers_new = merge_similar_minima(
        min_coords, thetas, phis, energies, tol
    )
    n_markers_old = len(np.unique(labels))
    n_markers_new = len(np.unique(markers_new)) - 1

    # Eliminate minima which have more than magnetocrystalline directions
    if len(min_coords_new) > n_minima and trim:
        markers_new = fix_minima(min_coords_new, energies, n_minima)

    if n_markers_old != n_markers_new or (len(min_coords_new) > n_minima and trim):
        min_coords, labels = get_min_regions(energies, markers=markers_new)

    # Find the global minimum for each LEM state
    # This might be superfluous as it's already done in get_min_regions
    # But oh well.
    for i in range(len(np.unique(labels))):
        where, theta, phi, energy = find_global_minimum(
            thetas, phis, energies, labels == np.unique(labels)[i]
        )
        theta_list.append(theta)
        phi_list.append(phi)
        min_energy_list.append(energy)

    # Get energy barriers associated with minimum regions.
    theta_mat, phi_mat, barriers = construct_energy_mat_fast(
        thetas, phis, energies, labels
    )

    # Prune states where it's easier to reach from another state always.
    if prune:
        barriers, theta_mat, phi_mat = prune_energies(barriers, theta_mat, phi_mat)

    # Convert to arrays
    theta_list = np.array(theta_list)
    phi_list = np.array(phi_list)
    min_energy_list = np.array(min_energy_list)

    # Check for weird erroneous states with entirely infinite barriers
    # to/from and remove.
    # This is yet another weird edge case that we have to catch
    if len(barriers) > 1:
        bad_filter = np.all(np.isinf(barriers), axis=1) | np.all(
            np.isinf(barriers), axis=0
        )

        theta_list = theta_list[~bad_filter]
        phi_list = phi_list[~bad_filter]
        min_energy_list = min_energy_list[~bad_filter]

        # Yuck! Must be a better way to do this.
        dels = np.arange(len(bad_filter))[bad_filter]
        theta_mat = np.delete(theta_mat, dels, axis=0)
        theta_mat = np.delete(theta_mat, dels, axis=1)
        phi_mat = np.delete(phi_mat, dels, axis=0)
        phi_mat = np.delete(phi_mat, dels, axis=1)
        barriers = np.delete(barriers, dels, axis=0)
        barriers = np.delete(barriers, dels, axis=1)
    fixed_labels = np.full(labels.shape, -1)
    if return_labels:
        for i in range(len(theta_list)):
            loc = np.where((thetas == theta_list[i]) & (phis == phi_list[i]))
            label = labels[loc][0]
            fixed_labels[labels == label] = i

        return (
            theta_list,
            phi_list,
            min_energy_list,
            theta_mat,
            phi_mat,
            barriers,
            fixed_labels,
        )
    return (theta_list, phi_list, min_energy_list, theta_mat, phi_mat, barriers)


def mat_to_mask(theta_mat, phi_mat):
    """
    Creates a mask for use in plot_energy_barriers

    Inputs
    ------
    theta_mat,phi_mat: numpy arrays
    Matrices of directions associated with energy barriers

    Returns
    -------
    mask: numpy array
    Array of bools showing where the energy minima are.
    """

    thetas = np.linspace(0, 2 * np.pi, 1001)
    phis = np.linspace(-np.pi / 2, np.pi / 2, 1001)
    thetas, phis = np.meshgrid(thetas, phis)
    theta_mat_new = np.copy(theta_mat)
    phi_mat_new = np.copy(phi_mat)

    # Only want to record barriers twice
    for i in range(len(theta_mat) - 1):
        for j in range(i + 1, len(theta_mat)):
            theta_mat_new[i, j] = -np.inf
            phi_mat_new[i, j] = -np.inf

    theta_list = theta_mat_new[~np.isinf(theta_mat)]
    phi_list = phi_mat_new[~np.isinf(phi_mat)]
    mask = np.full(thetas.shape, False)
    thetas = np.round(thetas, 4)
    phis = np.round(phis, 4)
    theta_list = np.round(theta_list, 4)
    phi_list = np.round(phi_list, 4)
    for theta, phi in zip(theta_list, phi_list):
        mask = mask | ((thetas == theta) & (phis == phi))
    return jnp.array(mask)


def delete_splits(new_indices, old_indices, high_energy_list):
    """
    Function that deletes "new" states. This is currently
    intended behaviour but will cause problems for e.g. low-T
    magnetite

    Inputs
    ------
    new_indices: numpy array
    Array of "new" set of indices of states

    old_indices: numpy array
    Array of "old" set of indices of states

    high_energy_list: numpy array
    Energies at higher temperature or field.
    """
    # Currently if we find a ``new'' state at a higher field or temperature
    # this is considered to be anomalous and is ignored. This might lead
    # to some undesired behavior - the other option is adding rows to everything
    # probably unnnecessary but worth checking if the warning occurs persistently.

    unique, counts = np.unique(new_indices, return_counts=True)
    if len(counts[counts > 1]) > 0:
        print("warning! split occurred!")
        multiple_states = unique[counts > 1]
        for i in multiple_states:
            bad_indices = old_indices[new_indices == i]
            bad_energies = high_energy_list[bad_indices]
            worst_states = bad_indices[bad_energies > min(bad_energies)]
            for state in worst_states:
                old_indices = np.delete(old_indices, state)
                new_indices = np.delete(new_indices, state)
    return (new_indices, old_indices)


def smart_sort(
    low_theta_list,
    low_phi_list,
    low_energy_list,
    low_theta_mat,
    low_phi_mat,
    low_barriers,
    low_labels,
    high_theta_list,
    high_phi_list,
    high_energy_list,
    high_theta_mat,
    high_phi_mat,
    high_barriers,
    high_labels,
):
    """
    Function for tracking LEM states or energy barriers as a function of temperature or field.
    Uses the final set of labels from the watershed algorithm to calculate what each state would
    minimize to.

    Inputs
    ------
    low_theta_list: length n numpy array
    Array of theta coordinates for lower temperature or field states

    low_phi_list: length n numpy array
    Ditto, for phi coordinates

    low_energy_list: length n numpy array
    Ditto for energies

    low_theta_mat: n x n array
    Ditto for energy barrier theta coordinate

    low_phi_mat: n x n array
    Ditto for energy barrier phi coordinate

    low_barriers: n x n array
    Ditto for energy barriers

    low_labels: 1001 x 1001 array
    Labels for low temperature/field energy barriers

    high_theta_list: length m numpy array
    Array of theta coordinates for lower temperature or field states

    high_phi_list: length m numpy array
    Ditto, for phi coordinates

    high_energy_list: length m numpy array
    Ditto for energies

    high_theta_mat: m x m array
    Ditto for energy barrier theta coordinate

    high_phi_mat: m x m array
    Ditto for energy barrier phi coordinate

    high_barriers: m x m array
    Ditto for energy barriers

    high_labels: 1001 x 1001 array
    Labels for high temperature/field energy barriers

    Returns
    -------

    thetas_new: length m numpy array
    Reindexed high_theta_list

    phis_new: length m numpy array
    Reindexed high_phi_list

    energy_list_new: length m numpy array
    Reindexed high_energy_list

    theta_mat_new: m x m array
    Reindexed high_theta_mat

    phi_mat_new: m x m array
    Reindexed high_phi_mat

    barriers_new: m x m array
    Reindexed high_barriers

    labels_new: 1001 x 1001 array
    Reindexed high_labels
    """

    thetas, phis, blank = energy_surface(
        0,
        0,
        np.eye(3),
        0,
        np.array([1 / 3, 1 / 3, 1 / 3]),
        np.array([0, 0, 0]),
        n_points=1001,
    )
    new_indices = []
    old_indices = []
    for i in range(len(high_theta_list)):
        where = np.where((thetas == high_theta_list[i]) & (phis == high_phi_list[i]))
        new_index = low_labels[where][0]
        new_indices.append(new_index)
        old_indices.append(i)

    new_indices = np.array(new_indices)
    old_indices = np.array(old_indices)

    # The "lost indices" are states which existed in the previous T/B, but have now disappeared
    lost_indices = np.setdiff1d(
        np.array(range(len(low_theta_list))), new_indices, assume_unique=True
    )
    # If the indices are already lost, they have inf theta, so ignore em
    lost_indices = lost_indices[~np.isinf(low_theta_list[lost_indices])]
    
    

    
    delete_splits(new_indices, old_indices, high_energy_list)
    
    #if len(lost_indices) == 0:
    #    cos_dist = np.full((len(high_theta_list), len(low_theta_list)), -np.inf)
    #    for i in range(len(high_theta_list)):
    #        for j in range(len(low_theta_list)):
    #            xyz_new = angle2xyz(high_theta_list[i], high_phi_list[i])
    #            xyz_old = angle2xyz(low_theta_list[j], low_phi_list[j])
    #            cos_dist[i, j] = np.dot(xyz_new, xyz_old)
    #            if np.isnan(cos_dist[i,j]):
    #                cos_dist[i,j] = np.nan
    #                
    #        old_j = new_indices[i]
    #        new_j = np.where(cos_dist[i]==np.amax(cos_dist[i]))[0]
    #        print(old_j,new_j)
    #        if type(new_j) == np.ndarray:
    #            if len(new_j) == 0:
    #                new_j = i
    #            else:
    #                new_j = new_j[0]
    #        if old_j != new_j:
    #            print('Warning: indexing methods disagree!','Watershed: ',old_j,'Nearest Neighbor: ',new_j)
    # Create new barriers with shape of old ones.
    barriers_new = deepcopy(low_barriers)
    thetas_new = deepcopy(low_theta_list)
    phis_new = deepcopy(low_phi_list)
    theta_mat_new = deepcopy(low_theta_mat)
    phi_mat_new = deepcopy(low_phi_mat)
    energy_list_new = deepcopy(low_energy_list)

    ### Dealing with LEM states ###

    # Reorder to fit old ordering
    for i in old_indices:
        j = new_indices[i]
        thetas_new[j] = high_theta_list[i]
        phis_new[j] = high_phi_list[i]
        energy_list_new[j] = high_energy_list[i]

    # Set "lost" states to inf
    for i in lost_indices:
        thetas_new[i] = np.inf
        phis_new[i] = np.inf
        energy_list_new[i] = np.inf

        barriers_new[i] = np.inf
        # No need for barriers between states
        # which no longer exist
        # We'll add the "exit states" for these
        # In later -> we just want them to
        # Disappear to the state that they'd
        # Minimize to.

    # Reorder barriers
    for i, j in combinations(old_indices, 2):
        k = new_indices[i]
        l = new_indices[j]
        barriers_new[k, l] = high_barriers[i, j]
        barriers_new[l, k] = high_barriers[j, i]
        theta_mat_new[k, l] = high_theta_mat[i, j]
        theta_mat_new[l, k] = high_theta_mat[j, i]
        phi_mat_new[k, l] = high_phi_mat[i, j]
        phi_mat_new[l, k] = high_phi_mat[j, i]

    # Set barriers for "lost" states to infinite
    for i, j in product(lost_indices, new_indices):
        barriers_new[j, i] = np.inf

    # Set outflow from "lost" states to 0s
    # Check which state "lost" states turned into
    for i in lost_indices:
        where = np.where((thetas == low_theta_list[i]) & (phis == low_phi_list[i]))
        j = new_indices[high_labels[where][0]]
        barriers_new[i, j] = 0

    # reindex new_labels
    labels_new = deepcopy(high_labels)
    for i in np.unique(high_labels):
        labels_new[high_labels == i] = new_indices[i]
    return (
        thetas_new,
        phis_new,
        energy_list_new,
        theta_mat_new,
        phi_mat_new,
        barriers_new,
        labels_new,
    )


def find_T_barriers(TMx, alignment, PRO, OBL, T_spacing=1):
    """
    Finds all LEM states and energy barriers for a grain composition
    and geometry at all temperatures. Runs from room temperature
    (20 C) to Tc. Todo: Write one of these for Hysteresis. This
    function could also easily be parallelized using multiprocessing.

    Inputs
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

    T_spacing: float
    Spacing of temperature steps (Degrees C)

    Returns
    -------
    theta_lists, phi lists: numpy arrays
    Arrays of the minimum energy directions in the SD energy surface

    min_energy_lists: numpy array
    Minimum energy at the minimum

    theta_mats,phi_mats: numpy arrays
    Arrays of the directions associated with the energy barriers.

    energy_mats: numpy array
    Matrix of energy barriers between state i and state j.

    Ts: numpy array
    List of temperatures.
    """
    Tc = (
        3.7237e02 * (TMx / 100) ** 3
        - 6.9152e02 * (TMx / 100) ** 2
        - 4.1385e02 * (TMx / 100) ** 1
        + 5.8000e02
    )

    theta_lists = []
    phi_lists = []
    min_energy_lists = []
    theta_mats = []
    phi_mats = []
    energy_mats = []
    Ts = np.arange(20, Tc, T_spacing)

    for T in Ts:
        print(
            "Calculating energy barriers at "
            + str(int(T)).zfill(3)
            + "°C, calculating up to "
            + str(int(Tc)).zfill(3)
            + "°C",
            end="\r",
        )
        (
            theta_list,
            phi_list,
            min_energy_list,
            theta_mat,
            phi_mat,
            energy_mat,
            labels,
        ) = find_all_barriers(TMx, alignment, PRO, OBL, T=T, return_labels=True)
        if len(theta_list) == 0:
            Ts = np.delete(Ts, Ts == T)
        else:
            if T > min(Ts):
                (
                    theta_list,
                    phi_list,
                    min_energy_list,
                    theta_mat,
                    phi_mat,
                    energy_mat,
                    labels,
                ) = smart_sort(
                    theta_lists[-1],
                    phi_lists[-1],
                    min_energy_lists[-1],
                    theta_mats[-1],
                    phi_mats[-1],
                    energy_mats[-1],
                    labels_old,
                    theta_list,
                    phi_list,
                    min_energy_list,
                    theta_mat,
                    phi_mat,
                    energy_mat,
                    labels,
                )

            theta_lists.append(theta_list)
            phi_lists.append(phi_list)
            energy_mats.append(energy_mat)
            min_energy_lists.append(min_energy_list)
            theta_mats.append(theta_mat)
            phi_mats.append(phi_mat)
            labels_old = deepcopy(labels)

    return (
        theta_lists,
        phi_lists,
        min_energy_lists,
        theta_mats,
        phi_mats,
        energy_mats,
        Ts,
    )


def get_antipode_order(theta_list, phi_list):
    """
    Calculates the nearest neighbor between a set of LEM states and the
    set of LEM states located in an antipodal orientation.

    Inputs
    ------
    theta_list,phi_list: numpy arrays
    Directions of minima on energy surface at first temperature.

    Returns
    -------
    antipode_indices: numpy array
    indices of numpy arrays
    """
    old_len = len(theta_list)  # number of old minima
    antipode_order = []
    antipode_thetas = (theta_list + np.pi) % (2 * np.pi)
    antipode_phis = -phi_list

    cos_dist = np.full((old_len, old_len), -np.inf)
    antipode_order = np.empty(old_len)
    for i in range(old_len):
        for j in range(old_len):
            xyz_orig = angle2xyz(theta_list[i], phi_list[i])
            xyz_ant = angle2xyz(antipode_thetas[j], antipode_phis[j])
            cos_dist[i, j] = np.dot(xyz_orig, xyz_ant)
    for j in range(old_len):
        n_neighbor = np.where(cos_dist[j, :] == np.amax(cos_dist[j, :]))[0]
        if type(n_neighbor) == np.ndarray:
            n_neighbor = n_neighbor[0]
        antipode_order[j] = n_neighbor
    antipode_order = antipode_order.astype(int)
    return antipode_order


def make_antipode_array(
    Bs, theta_lists, phi_lists, min_energy_lists, theta_mats, phi_mats, energy_mats
):
    """
    Takes a set of LEM states and energy barriers calculated a range of fields
    and calculates the barriers for a range of antipodal fields. This works
    because all other anisotropies are axisymmetric.

    Inputs
    ------
    Bs: list
    List of field strengths

    theta_lists: list
    List of state theta coordinates at each field

    phi_lists: list
    List of state phi coordinates at each field

    min_energy_lists: list
    List of state energies at each field

    theta_mats: list
    List of barrier theta coordinates at each field

    phi_mats: list
    List of barrier phi coordinates at each field

    energy_mats: list
    List of barrier energies at each field

    Returns
    -------
    Bs_new: list
    List of field strengths

    theta_lists_new: list
    List of state theta coordinates at each field

    phi_lists_new: list
    List of state phi coordinates at each field

    min_energy_lists_new: list
    List of state energies at each field

    theta_mats_new: list
    List of barrier theta coordinates at each field

    phi_mats_new: list
    List of barrier phi coordinates at each field

    energy_mats_new: list
    List of barrier energies at each field

    """
    theta_lists_new = deepcopy(theta_lists)
    phi_lists_new = deepcopy(phi_lists)
    min_energy_lists_new = deepcopy(min_energy_lists)
    theta_mats_new = deepcopy(theta_mats)
    phi_mats_new = deepcopy(phi_mats)
    energy_mats_new = deepcopy(energy_mats)

    antipode_order = get_antipode_order(theta_lists[0], phi_lists[0])
    for i in range(1, len(Bs)):
        theta_list = (theta_lists[i] + np.pi) % (2 * np.pi)
        theta_list = theta_list[antipode_order]
        theta_lists_new = [theta_list] + theta_lists_new
        phi_list = -phi_lists[i]
        phi_list = phi_list[antipode_order]
        phi_lists_new = [phi_list] + phi_lists_new
        theta_mat = (theta_mats[i] + np.pi) % (2 * np.pi)
        theta_mat[np.isnan(theta_mat)] = -np.inf
        theta_mat = theta_mat[:, antipode_order][antipode_order]
        theta_mats_new = [theta_mat] + theta_mats_new
        phi_mat = -phi_mats[i]
        phi_mat[np.isinf(phi_mat)] = -np.inf
        phi_mat = phi_mat[:, antipode_order][antipode_order]
        phi_mats_new = [phi_mat] + phi_mats_new
        min_energy_list = min_energy_lists[i]
        min_energy_list = min_energy_list[antipode_order]
        min_energy_lists_new = [min_energy_list] + min_energy_lists_new
        energy_mat = energy_mats[i]
        energy_mat = energy_mat[:, antipode_order][antipode_order]
        energy_mats_new = [energy_mat] + energy_mats_new
    Bs_new = np.append(-np.flip(Bs[1:]), Bs)

    return (
        Bs_new,
        theta_lists_new,
        phi_lists_new,
        min_energy_lists_new,
        theta_mats_new,
        phi_mats_new,
        energy_mats_new,
    )


def find_B_barriers(TMx, alignment, PRO, OBL, B_dir, B_max, B_spacing, T=20):
    """
    Finds all LEM states and energy barriers for a grain composition
    and geometry over a range of fields in a specfic direction. Runs
    from zero-field to B_max. This function could also easily be
    parallelized using multiprocessing.

    Inputs
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

    B_dir: length 3 array of floats
    Cartesian direction (unit vector) of field.

    B_max: float
    Maximum field (T).

    B_spacing: float
    Spacing of field steps (T)

    T: float
    Temperature (Degrees C) - must be between 0 and 80.

    Returns
    -------
    theta_lists, phi lists: numpy arrays
    Arrays of the minimum energy directions in the SD energy surface

    min_energy_lists: numpy array
    Minimum energy at the minimum

    theta_mats,phi_mats: numpy arrays
    Arrays of the directions associated with the energy barriers.

    energy_mats: numpy array
    Matrix of energy barriers between state i and state j.

    Ts: numpy array
    List of temperatures.
    """

    theta_lists = []
    phi_lists = []
    min_energy_lists = []
    theta_mats = []
    phi_mats = []
    energy_mats = []

    Bs = np.arange(0, B_max + B_spacing, B_spacing)
    ext_theta, ext_phi = np.degrees(xyz2angle(B_dir))

    for B in Bs:
        print(
            "Calculating energy barriers at "
            + str(int(B * 1e3)).zfill(4)
            + " mT, calculating up to "
            + str(B_max * 1e3).zfill(4)
            + " mT",
            end="\r",
        )

        ext_field = np.array([ext_theta, ext_phi, B])
        (
            theta_list,
            phi_list,
            min_energy_list,
            theta_mat,
            phi_mat,
            energy_mat,
            labels,
        ) = find_all_barriers(
            TMx, alignment, PRO, OBL, T=T, ext_field=ext_field, return_labels=True
        )
        if len(theta_list) == 0:
            Bs = np.delete(Bs, Bs == B)
        else:
            if B > min(Bs):
                (
                    theta_list,
                    phi_list,
                    min_energy_list,
                    theta_mat,
                    phi_mat,
                    energy_mat,
                    labels,
                ) = smart_sort(
                    theta_lists[-1],
                    phi_lists[-1],
                    min_energy_lists[-1],
                    theta_mats[-1],
                    phi_mats[-1],
                    energy_mats[-1],
                    labels_old,
                    theta_list,
                    phi_list,
                    min_energy_list,
                    theta_mat,
                    phi_mat,
                    energy_mat,
                    labels,
                )

            theta_lists.append(theta_list)
            phi_lists.append(phi_list)
            energy_mats.append(energy_mat)
            min_energy_lists.append(min_energy_list)
            theta_mats.append(theta_mat)
            phi_mats.append(phi_mat)
            labels_old = deepcopy(labels)

    Bs, theta_lists, phi_lists, min_energy_lists, theta_mats, phi_mats, energy_mats = (
        make_antipode_array(
            Bs,
            theta_lists,
            phi_lists,
            min_energy_lists,
            theta_mats,
            phi_mats,
            energy_mats,
        )
    )

    return (
        np.array(theta_lists),
        np.array(phi_lists),
        np.array(min_energy_lists),
        np.array(theta_mats),
        np.array(phi_mats),
        np.array(energy_mats),
        Bs,
    )


### STORING RESULTS ###
def energy_spline(x, y):
    """
    Creates a piecewise scalar B-Spline from a set of x,y data.

    Inputs
    ------
    x: numpy array
    Input x data for cubic spline (usually temperature)

    y: numpy array
    Input y data for cubic spline (usually energy or magnetization)

    Returns
    -------
    t: numpy array
    Cubic spline knot positions

    c: numpy array
    B-Spline coefficients

    k: numpy array
    Polynomial degrees
    """
    x_fin = x[~np.isinf(y)]
    y_fin = y[~np.isinf(y)]
    if len(x_fin) <= 1:
        t = np.array([min(x), max(x)])
        c = np.array([np.inf, np.inf])
        k = 0
    elif np.all(y_fin == 0):
        t = np.array([min(x_fin), max(x_fin)])
        c = np.array([0.0, 0.0])
        k = 0
    elif len(x_fin) <= 3:
        std = min(1, np.ptp(y_fin) / 100)
        w = np.full(len(x_fin), 1 / std)
        t, c, k = splrep(x_fin, y_fin, w=w, task=0, k=1)
    else:
        std = min(1, np.ptp(y_fin) / 100)
        w = np.full(len(x_fin), 1 / std)
        t, c, k = splrep(x_fin, y_fin, w=w, task=0)
    return (t, c, k)


def direction_spline(x, y):
    """
    Creates a piecewise 3D unit vector B-Spline from a set of x,y data.

    Inputs
    ------
    x: numpy array
    Input x data for cubic spline (usually temperature)

    y: numpy array of 3D unit vectors.
    Input y data for cubic spline (usually a direction)

    Returns
    -------
    t: numpy array
    Cubic spline knot positions

    c: numpy array
    B-Spline coefficients

    k: numpy array
    Polynomial degrees
    """
    x_fin = x[~np.any(np.isinf(y), axis=1)]
    y_fin = y[~np.any(np.isinf(y), axis=1)]

    if len(x_fin) <= 1:
        t = np.array([min(x), max(x)])
        c = np.array([[np.inf, np.inf], [np.inf, np.inf]])
        k = 0
    elif len(x_fin) <= 3:
        std = 0.001
        w = np.full(len(x_fin), 1 / std)
        tck, u = splprep(angle2xyz(y_fin[:, 0], y_fin[:, 1]), u=x_fin, w=w, task=0, k=1)
        t, c, k = tck
        c = np.array(c).T
        c_shape = len(t) - len(c)
        c = np.append(c, np.zeros((c_shape, 3)), axis=0)
    else:
        std = 0.001
        w = np.full(len(x_fin), 1 / std)
        tck, u = splprep(angle2xyz(y_fin[:, 0], y_fin[:, 1]), u=x_fin, w=w, task=0)
        t, c, k = tck
        c = np.array(c).T
        c_shape = len(t) - len(c)
        c = np.append(c, np.zeros((c_shape, 3)), axis=0)

    return (t, c, k)


def energy_result(t, c, k, T):
    """
    Calculates an energy from a temperature and a set of spline
    coefficients.

    Inputs
    ------
    t: numpy array
    Cubic spline knot positions

    c: numpy array
    B-Spline coefficients

    k: numpy array
    Polynomial degrees

    T: float
    Temperature (degrees C).

    Returns
    -------
    result: numpy array or float
    Energy at that temperature.
    """
    result = BSpline(t, c, k, extrapolate=False)(T)
    result = np.nan_to_num(result, nan=np.inf)
    return result


def direction_result(t, c, k, T):
    """
    Calculates a direction from a temperature and a set of spline
    coefficients.

    Inputs
    ------
    t: numpy array
    Cubic spline knot positions

    c: numpy array
    B-Spline coefficients

    k: numpy array
    Polynomial degrees

    T: float
    Temperature (degrees C).

    Returns
    -------
    direction: numpy array or float
    Direction at that temperature.
    """
    direction = BSpline(t, c, k, extrapolate=False)(T)
    direction = np.array(xyz2angle(direction.T)).T
    if len(direction.shape) > 1:
        direction[:, 0] = direction[:, 0] % (2 * np.pi)
    else:
        direction[0] = direction[0] % (2 * np.pi)
    return np.nan_to_num(direction, nan=-np.inf)


class GEL:
    """
    Class for storing energy barrier results at all temperatures for a
    given grain geometry and composition.

    Todo:
    1. This should inherit from some base class shared with Hysteresis.
    2. The run through of temperatures should be parallelized for much
    quicker object creation.

    Input Attributes
    ----------------
    TMx: float
    Titanomagnetite composition (0 - 100)

    alignment: str
    Alignment of magnetocrystalline and shape axis. Either 'hard' or
    'easy' magnetocrystalline always aligned with shape easy.

    PRO: float
    Prolateness of ellipsoid (major / intermediate axis)

    OBL: float
    Oblateness of ellipsoid (intermediae / minor axis)

    T_spacing: float
    Spacing of temperature steps (Degrees C)
    """

    def __init__(self, TMx, alignment, PRO, OBL, T_spacing=1):
        self.d_min = calc_d_min(TMx, alignment, PRO, OBL)

        theta_lists, phi_lists, min_energies, theta_mats, phi_mats, energy_mats, Ts = (
            find_T_barriers(TMx, alignment, PRO, OBL, T_spacing=T_spacing)
        )

        theta_lists = np.array(theta_lists)
        phi_lists = np.array(phi_lists)
        min_energies = np.array(min_energies)
        theta_mats = np.array(theta_mats)
        phi_mats = np.array(phi_mats)
        energy_mats = np.array(energy_mats)

        min_dir = np.empty(theta_lists.shape[1], dtype="object")
        min_energy = np.empty(theta_lists.shape[1], dtype="object")

        bar_dir = np.empty((theta_lists.shape[1], theta_lists.shape[1]), dtype="object")
        bar_energies = np.empty(
            (theta_lists.shape[1], theta_lists.shape[1]), dtype="object"
        )

        for i in range(theta_lists.shape[1]):
            dirs = np.array([theta_lists[:, i], phi_lists[:, i]]).T
            tck = direction_spline(Ts, dirs)
            min_dir[i] = tck
            min_energy[i] = energy_spline(Ts, min_energies[:, i])
            for j in range(theta_lists.shape[1]):
                dirs = np.array([theta_mats[:, i, j], phi_mats[:, i, j]]).T
                tck = direction_spline(Ts, dirs)
                bar_dir[i, j] = tck
                bar_energies[i, j] = energy_spline(Ts, energy_mats[:, i, j])

        self.min_dir = min_dir
        self.min_energy = min_energy
        self.bar_dir = bar_dir
        self.bar_energy = bar_energies
        self.TMx = TMx
        self.alignment = alignment
        self.PRO = PRO
        self.OBL = OBL
        self.T_max = max(Ts)
        self.T_min = min(Ts)

    def to_file(self, fname):
        """
        Saves GEL object to file.

        Inputs
        ------
        fname: string
        Filename

        Returns
        -------
        None
        """
        with open(fname, "wb") as f:
            pickle.dump(self, f)
        f.close()

    def __repr__(self):
        retstr = """Energy Landscape of TM{TMx} Grain with a prolateness of {PRO:.2f} and an oblateness of {OBL:.2f} elongated along the magnetocrystalline {alignment} axis.""".format(
            TMx=self.TMx, PRO=self.PRO, OBL=self.OBL, alignment=self.alignment
        )
        return retstr

    def get_params(self, T):
        """
        Gets directions and energies associated with LEM states and
        barriers for a grain as a function of temperature.

        Inputs
        ------
        T: int, float or numpy array
        Temperature(s) (degrees C)

        Returns
        -------
        params: dict
        Dictionary of arrays for directions and energies.
        """
        rot_mat, k1, k2, Ms = get_material_parms(self.TMx, self.alignment, T)
        min_dir = []
        min_e = []

        if isinstance(T, (float, int, np.float64, np.int64)):
            assert (T <= self.T_max) & (T >= self.T_min), (
                "T must be between " + str(self.T_min) + " and " + str(self.T_max)
            )
            bar_e = np.full(self.bar_energy.shape, np.inf)
            bar_dir = np.full((len(self.min_energy), len(self.min_energy), 2), np.inf)

        elif isinstance(T, np.ndarray):
            assert (np.amax(T) <= self.T_max) & (np.amin(T) >= self.T_min), (
                "T must be between " + str(self.T_min) + " and " + str(self.T_max)
            )
            bar_e = np.full(
                len(self.min_energy), len(self.min_energy), 2, T.shape, np.inf
            )
            bar_dir = np.full(
                (len(self.min_energy), len(self.min_energy), 2, T.shape), np.inf
            )

        else:
            print(T)
            raise TypeError("T should not be type " + str(type(T)))

        for i in range(len(self.min_energy)):
            t, c, k = self.min_energy[i]
            min_e.append(energy_result(t, c, k, T))
            t, c, k = self.min_dir[i]
            min_dir.append(direction_result(t, c, k, T))
            for j in range(len(self.min_energy)):
                t, c, k = self.bar_energy[i, j]
                bar_e[i, j] = energy_result(t, c, k, T)
                t, c, k = self.bar_dir[i, j]
                bar_dir[i, j] = direction_result(t, c, k, T)

        bar_e[bar_e > 1e308] = np.inf
        bar_e[bar_e < 0] = 0.0
        min_e = np.array(min_e)
        min_e[min_e > 1e308] = np.inf
        bar_dir[np.abs(bar_dir) > 1e308] = np.inf
        min_dir = np.array(min_dir)
        min_dir[np.abs(min_dir) > 1e308] = np.inf
        params = {
            "min_dir": np.array(min_dir),
            "min_e": min_e,
            "bar_dir": np.array(bar_dir),
            "bar_e": np.array(bar_e),
            "T": T,
            "Ms": Ms,
        }
        return params


class HEL:
    """
    Class for storing energy barrier results at all field strengths for a
    given grain geometry, composition and field direction.

    Todo:
    1. This should inherit from some base class shared with GEL.
    2. The run through of temperatures should be parallelized for much
    quicker object creation.

    Input Attributes
    ----------------
    TMx: float
    Titanomagnetite composition (0 - 100)

    alignment: str
    Alignment of magnetocrystalline and shape axis. Either 'hard' or
    'easy' magnetocrystalline always aligned with shape easy.

    PRO: float
    Prolateness of ellipsoid (major / intermediate axis)

    OBL: float
    Oblateness of ellipsoid (intermediae / minor axis)

    B_dir: length 3 array
    Unit vector of field direction

    B_max: float
    Maximum field (Tesla)

    B_spacing: float
    Spacing of field steps (Tesla)
    """

    def __init__(self, TMx, alignment, PRO, OBL, rot_mat, B_max, B_spacing=0.001, T=20):
        self.d_min = calc_d_min(TMx, alignment, PRO, OBL)
        B_dir = rot_mat @ np.array([1.,0.,0.])
        (
            theta_lists,
            phi_lists,
            min_energy_lists,
            theta_mats,
            phi_mats,
            energy_mats,
            Bs,
        ) = find_B_barriers(TMx, alignment, PRO, OBL, B_dir, B_max, B_spacing, T=T)
        split_point = int(len(Bs) / 2)

        first_theta_lists = theta_lists[split_point:]
        first_phi_lists = phi_lists[split_point:]
        first_energy_lists = min_energy_lists[split_point:]
        first_theta_mats = theta_mats[split_point:]
        first_phi_mats = phi_mats[split_point:]
        first_energy_mats = energy_mats[split_point:]

        second_theta_lists = theta_lists[: split_point + 1][::-1]
        second_phi_lists = phi_lists[: split_point + 1][::-1]
        second_energy_lists = min_energy_lists[: split_point + 1][::-1]
        second_theta_mats = theta_mats[: split_point + 1][::-1]
        second_phi_mats = phi_mats[: split_point + 1][::-1]
        second_energy_mats = energy_mats[: split_point + 1][::-1]

        pos_min_dir = np.empty(first_theta_lists.shape[1], dtype="object")
        pos_min_energy = np.empty(first_theta_lists.shape[1], dtype="object")
        pos_bar_dir = np.empty(
            (first_theta_lists.shape[1], first_theta_lists.shape[1]), dtype="object"
        )
        pos_bar_energies = np.empty(
            (first_theta_lists.shape[1], first_theta_lists.shape[1]), dtype="object"
        )

        pos_Bs = Bs[split_point:]
        for i in range(first_theta_lists.shape[1]):
            dirs = np.array([first_theta_lists[:, i], first_phi_lists[:, i]]).T
            tck = direction_spline(pos_Bs, dirs)
            pos_min_dir[i] = tck
            pos_min_energy[i] = energy_spline(pos_Bs, first_energy_lists[:, i])
            for j in range(first_theta_lists.shape[1]):
                dirs = np.array([first_theta_mats[:, i, j], first_phi_mats[:, i, j]]).T
                tck = direction_spline(pos_Bs, dirs)
                pos_bar_dir[i, j] = tck
                pos_bar_energies[i, j] = energy_spline(
                    pos_Bs, first_energy_mats[:, i, j]
                )

        neg_min_dir = np.empty(second_theta_lists.shape[1], dtype="object")
        neg_min_energy = np.empty(second_theta_lists.shape[1], dtype="object")

        neg_bar_dir = np.empty(
            (second_theta_lists.shape[1], second_theta_lists.shape[1]), dtype="object"
        )
        neg_bar_energies = np.empty(
            (second_theta_lists.shape[1], second_theta_lists.shape[1]), dtype="object"
        )

        neg_Bs = Bs[: split_point + 1]
        for i in range(second_theta_lists.shape[1]):
            dirs = np.array([second_theta_lists[::-1, i], second_phi_lists[::-1, i]]).T
            tck = direction_spline(neg_Bs, dirs)
            neg_min_dir[i] = tck
            neg_min_energy[i] = energy_spline(neg_Bs, second_energy_lists[::-1, i])
            for j in range(second_theta_lists.shape[1]):
                dirs = np.array(
                    [second_theta_mats[::-1, i, j], second_phi_mats[::-1, i, j]]
                ).T
                tck = direction_spline(neg_Bs, dirs)
                neg_bar_dir[i, j] = tck
                neg_bar_energies[i, j] = energy_spline(
                    neg_Bs, second_energy_mats[::-1, i, j]
                )

        self.min_dir = (neg_min_dir, pos_min_dir)
        self.min_energy = (neg_min_energy, pos_min_energy)
        self.bar_dir = (neg_bar_dir, pos_bar_dir)
        self.bar_energy = (neg_bar_energies, pos_bar_energies)

        self.TMx = TMx
        self.alignment = alignment
        self.PRO = PRO
        self.OBL = OBL

        self.B_max = B_max
        self.B_dir = B_dir
        self.rot_mat = rot_mat
        self.T = T

    def to_file(self, fname):
        """
        Saves GEL object to file.

        Inputs
        ------
        fname: string
        Filename

        Returns
        -------
        None
        """
        with open(fname, "wb") as f:
            pickle.dump(self, f)
        f.close()

    def __repr__(self):
        retstr = """Energy Landscape of TM{TMx} Grain with a prolateness of {PRO:.2f} and an oblateness of {OBL:.2f} elongated along the magnetocrystalline {alignment} axis.""".format(
            TMx=self.TMx, PRO=self.PRO, OBL=self.OBL, alignment=self.alignment
        )
        return retstr

    def get_params(self, B):
        """
        Gets directions and energies associated with LEM states and
        barriers for a grain as a function of field.

        Inputs
        ------
        B: int, float or numpy array
        External Field (T)

        Returns
        -------
        params: dict
        Dictionary of arrays for directions and energies.
        """
        rot_mat, k1, k2, Ms = get_material_parms(self.TMx, self.alignment, self.T)
        min_dir = []
        min_e = []

        if isinstance(B, (float, int, np.float64, np.int64)):
            assert (B <= self.B_max) & (B >= -self.B_max), (
                "B must be between " + str(-self.B_max) + " and " + str(self.B_max)
            )
            bar_e = np.full(self.bar_energy[0].shape, np.inf)
            bar_dir = np.full(
                (len(self.min_energy[0]), len(self.min_energy[0]), 2), np.inf
            )

        elif isinstance(B, np.ndarray):
            assert (np.amax(B) <= self.B_max) & (np.amin(B) >= -self.B_max), (
                "B must be between " + str(-self.B_max) + " and " + str(self.B_max)
            )
            bar_e = np.full(
                len(self.min_energy[0]), len(self.min_energy[0]), 2, B.shape, np.inf
            )
            bar_dir = np.full(
                (len(self.min_energy[0]), len(self.min_energy[0]), 2, B.shape), np.inf
            )

        else:
            print(B)
            raise TypeError("B should not be type " + str(type(B)))
        if B > 0:
            splindex = 1
        else:
            splindex = 0

        for i in range(len(self.min_energy[0])):
            t, c, k = self.min_energy[splindex][i]
            min_e.append(energy_result(t, c, k, B))
            t, c, k = self.min_dir[splindex][i]
            min_dir.append(direction_result(t, c, k, B))
            for j in range(len(self.min_energy[0])):
                t, c, k = self.bar_energy[splindex][i, j]
                bar_e[i, j] = energy_result(t, c, k, B)
                t, c, k = self.bar_dir[splindex][i, j]
                bar_dir[i, j] = direction_result(t, c, k, B)
        if B == 0:
            min_e2 = []
            bar_e2 = np.full(self.bar_energy[0].shape, np.inf)
            for i in range(len(self.min_energy[0])):
                t, c, k = self.min_energy[1][i]
                min_e2.append(energy_result(t, c, k, B))
                for j in range(len(self.min_energy[0])):
                    t, c, k = self.bar_energy[1][i, j]
                    bar_e2[i, j] = energy_result(t, c, k, B)
            min_e = np.mean([min_e, min_e2], axis=0)
            bar_e = np.mean([bar_e, bar_e2], axis=0)

        bar_e[bar_e > 1e308] = np.inf
        bar_e[bar_e < 0] = 0.0
        min_e = np.array(min_e)
        min_e[min_e > 1e308] = np.inf
        bar_dir[np.abs(bar_dir) > 1e308] = np.inf
        min_dir = np.array(min_dir)
        min_dir[np.abs(min_dir) > 1e308] = np.inf
        params = {
            "min_dir": np.array(min_dir),
            "min_e": min_e,
            "bar_dir": np.array(bar_dir),
            "bar_e": np.array(bar_e),
            "B": B,
            "Ms": Ms,
            "T": self.T,
        }
        return params


class HELs:
    def __init__(
        self,
        TMx,
        alignment,
        PRO,
        OBL,
        B_max,
        B_spacing=0.001,
        T=20,
        rot_mats=None,
        n_dirs=30,
    ):

        if type(rot_mats) == type(None):
            rot_mats = fib_hypersphere(n_dirs)
    
        HEL_list = []
        B_dirs = []
        for rot_mat in rot_mats:
            B_dirs.append(rot_mat@np.array([1,0,0]))
            HEL_list.append(HEL(TMx, alignment, PRO, OBL, rot_mat, B_max, B_spacing, T))
        self.HEL_list = HEL_list
        self.B_dirs = B_dirs
        self.rot_mats = rot_mats

    def __getitem__(self, index: int):
        return self.HEL_list[index]

    def to_file(self, fname):
        """
        Saves GEL object to file.

        Inputs
        ------
        fname: string
        Filename

        Returns
        -------
        None
        """
        with open(fname, "wb") as f:
            pickle.dump(self, f)
        f.close()


def uniaxial_relaxation_time(d, T, K):
    """
    Calculates the Neel relaxation time for an
    energy barrier at a particular temperature

    Inputs
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

    Inputs
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


def blocking_temperature(gel: GEL, d, i, j, block_t=100.0):
    """
    Calculates the blocking temperature associated
    with an energy barrier. Involves a minimization
    to obtain the correct time.

    Inputs
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
