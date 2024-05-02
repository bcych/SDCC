#######################################################################
### energy.py                                                       ###
### THIS MODULE CALCULATES SINGLE DOMAIN ENERGIES FOR               ###
### TITANOMAGNETITES AS A FUNCTION OF DIRECTION. IT COULD BENEFIT   ###
### FROM HOOKING INTO MATERIALS.PY FOR A MORE GENERAL APPLICATION   ###
#######################################################################

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import ellipkinc, ellipeinc
import jax.numpy as jnp
from jax import jit, vmap, config
from jax.tree_util import Partial

# IMPORTANT - Makes JAX calculations using double precision rather than
# single precision which is the default behaviour for some reason!
config.update("jax_enable_x64", True)


@jit
def angle2xyz(theta, phi):
    """
    Converts from coordinates on a sphere surface (theta, phi) radians
    to cartesian coordinates (x,y,z) as a unit vector

    Inputs
    ------
    theta: float
    horizontal angle (radians) between 0 and 2*pi

    phi: float
    vertical angle (radians) between -pi and pi

    Returns
    ------
    xyz: jax numpy array
    array of the x,y and z cartesian coordinates
    """
    x = jnp.cos(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.cos(phi)
    z = jnp.sin(phi)
    return jnp.array([x, y, z])


@jit
def xyz2angle(xyz):
    """
    Converts cartesian coordinates to coordinates on a sphere surface.
    Vector length is lost in conversion.

    Inputs
    ------
    x: float
    cartesian x coordinate

    y: float
    cartesian y coordinate

    z: float
    cartesian z coordinate

    Returns
    -------
    theta: float
    horizontal angle (radians) between 0 and 2*pi

    phi: float
    vertical angle (radians) between -pi and pi

    """
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    theta = jnp.arctan2(y, x)
    phi = jnp.arcsin(z)
    return (theta, phi)


def demag_factors(PRO, OBL):
    """
    Calculates the demagnetizing factors for a standard ellipsoid using
    the formulae of Osborn (1945). We assume that the major axis of
    the ellipsoid is always along x and the minor axis of the ellipsoid
    is always along z.

    Inputs
    ------
    PRO: float
    Prolateness ratio (major axis/intermediate axis)

    OBL: float
    Oblateness ratio (intermediate axis/minor axis)

    Returns
    -------
    LMN: numpy array
    Array of L, M, N, corresponding to the demagnetizing factors
    """
    a = PRO
    b = 1.0
    c = 1 / OBL

    # Prepare variables used in demag factor calculation
    otheta = np.arccos(c / a)
    ophi = np.arccos(b / a)
    k = np.sin(ophi) / np.sin(otheta)
    alpha = np.arcsin(k)

    # Elliptical integrals of the first and second kind are needed
    F = ellipkinc(otheta, k**2)
    E = ellipeinc(otheta, k**2)

    # More variables used
    first_num = (c / a) * (b / a)
    sin_3_otheta = np.sin(otheta) ** 3
    cos_2_alpha = np.cos(alpha) ** 2
    ttp = np.sin(otheta) * np.cos(otheta) / np.cos(ophi)
    tpt = np.sin(otheta) * np.cos(ophi) / np.cos(otheta)

    # Unfortunately the general equation used by Osborn is
    # numerically unstable when the ellipsoid isn't triaxial
    # so we have to use several cases.

    if a == b == c:  # Equidimensional (sphere)
        L = M = N = 1 / 3

    elif b == c:  # Prolate
        L = first_num / (sin_3_otheta * k**2) * (F - E)
        M = N = (1 - L) / 2

    elif a == b:  # Oblate
        N = first_num / (sin_3_otheta * cos_2_alpha) * (tpt - E)
        L = M = (1 - N) / 2

    else:  # Triaxial
        L = first_num / (sin_3_otheta * k**2) * (F - E)
        M = (
            first_num
            / (sin_3_otheta * cos_2_alpha * k**2)
            * (E - cos_2_alpha * F - k**2 * ttp)
        )
        N = first_num / (sin_3_otheta * cos_2_alpha) * (tpt - E)

    LMN = np.array([L, M, N])
    return LMN


@jit
def Ed(LMN, theta, phi, Ms):
    """
    Calculates the demagnetizing field energy for an ellipsoid as a
    function of theta, phi angles and saturation magnetization. Only
    applicable to single domain grains.

    Inputs
    ------
    LMN: numpy array
    Demagnetizing factors of the ellipsoid along x, y, z directions.

    theta: float
    horizontal angle (radians) between 0 and 2*pi

    phi: float
    vertical angle (radians) between -pi and pi

    Ms: float
    Saturation magnetization in A/m

    Returns
    -------
    Ed: float
    Demagnetizing field energy in J/m3
    """
    xyz = angle2xyz(theta, phi) * Ms
    N = jnp.eye(3) * LMN
    MN = jnp.dot(xyz, N)
    MNM = jnp.dot(MN, xyz.T)
    mu0 = 4 * jnp.pi * 1e-7
    return 0.5 * mu0 * MNM


@jit
def Ea(k1, k2, theta, phi, rot_mat):
    """
    Calculates the CUBIC magnetocrystalline anisotropy energy as a
    function of theta, phi angles. Anisotropy energy field can be
    rotated using a rotation matrix.

    Inputs
    ------
    k1,k2: floats
    Cubic magnetocrystalline anisotropy constants. If k2 is not
    calculated, leave as 0.

    theta,phi: floats
    Angles in spherical coordinates along which the energy is
    calculated (in radians).

    rot_mat: numpy array
    Rotation matrix to rotate anisotropy energy field by.

    Returns
    -------
    Ea: float
    Magnetocrystalline anisotropy energy.
    """
    xyz = angle2xyz(theta, phi)
    xyz = jnp.matmul(rot_mat, xyz)
    a1 = xyz[0]
    a2 = xyz[1]
    a3 = xyz[2]
    return (
        k1 * ((a1 * a2) ** 2 + (a2 * a3) ** 2 + (a1 * a3) ** 2)
        + k2 * (a1 * a2 * a3) ** 2
    )


@jit
def Ez(theta, phi, field_theta, field_phi, field_str, Ms):
    """
    Calculates the Zeeman energy as a function of theta, phi angles.

    Inputs
    ------
    theta, phi: floats
    Angles along which energy is evaluated (radians).

    field theta, field phi: floats
    Direction of field specified as an angle (radians).

    field str: float
    Strength of field (in Tesla).

    Ms: float
    Saturation magnetization of sample.

    Returns
    -------
    Ez: float
    Zeeman energy.
    """
    xyz = angle2xyz(theta, phi)
    field_xyz = angle2xyz(field_theta, field_phi)

    field_xyz *= field_str
    return -Ms * jnp.dot(xyz, field_xyz)


@jit
def energy_ang(angles, k1, k2, rot_mat, LMN, Ms, ext_field):
    """
    Calculates the total energy (Zeeman + Anisotropy + Demag) as a
    function of theta, phi angle. Requires material parameters and
    demagnetizing factors.

    Inputs
    ------
    angles: array
    Array of angles specified as [theta, phi]

    k1, k2: floats
    Magnetocrystalline anisotropy constant

    rot_mat: numpy array
    Rotation matrix for magnetocrystalline anisotropy field.

    LMN: numpy array
    Array of demagnetizing factors for Ellipsoid shape.

    Ms: float
    Saturation magnetization of material

    ext_field: numpy array
    Array of [theta, phi, B] for the external field, where theta and
    phi are expressed in DEGREES and B is in Tesla.

    Returns
    -------
    Etot: float
    Total energy as a function of theta, phi angle.
    """
    theta, phi = angles
    field_theta, field_phi, field_str = ext_field
    field_theta = jnp.radians(field_theta)
    field_phi = jnp.radians(field_phi)

    # Note that although "H" is used for these terms, they are energies.
    Ha = Ea(k1, k2, theta, phi, rot_mat)
    Hd = Ed(LMN, theta, phi, Ms)
    Hz = Ez(theta, phi, field_theta, field_phi, field_str, Ms)
    return Ha + Hd + Hz


@jit
def energy_xyz(xyz, k1, k2, rot_mat, LMN, Ms, ext_field):
    """
    Calculates the total energy (Zeeman + Anisotropy + Demag) as a
    function of x, y, z coordinate. Requires material parameters and
    demagnetizing factors.

    Inputs
    ------
    xyz: array
    Array of coordinates specified as [x,y,z]

    k1, k2: floats
    Magnetocrystalline anisotropy constant

    rot_mat: numpy array
    Rotation matrix for magnetocrystalline anisotropy field.

    LMN: numpy array
    Array of demagnetizing factors for Ellipsoid shape.

    Ms: float
    Saturation magnetization of material

    ext_field: numpy array
    Array of [theta, phi, B] for the external field, where theta and
    phi are expressed in DEGREES and B is in Tesla.

    Returns
    -------
    Etot: float
    Total energy as a function of theta, phi angle.
    """
    theta, phi = xyz2angle(xyz / jnp.linalg.norm(xyz))
    field_theta, field_phi, field_str = ext_field
    field_theta = jnp.radians(field_theta)
    field_phi = jnp.radians(field_phi)

    # Note that although "H" is used for these terms, they are energies.
    Ha = Ea(k1, k2, theta, phi, rot_mat)
    Hd = Ed(LMN, theta, phi, Ms)
    Hz = Ez(theta, phi, field_theta, field_phi, field_str, Ms)
    return jnp.nan_to_num(Ha + Hd, nan=jnp.inf)


def calculate_anisotropies(TMx):
    """
    Calculates the room temperature easy, intermediate and hard
    directions as a function of titanomagnetite composition. This
    function could probably be handled by materials.py in the future

    Inputs
    ------
    TMx: float
    Titanomagnetite composition (0 to 100).

    Returns
    -------
    sorted_axes: numpy array
    "Special Directions" array of [easy, intermediate, hard] axes.
    Direction is given as a space separated string e.g. '1 0 0' for
    compatibility with MERRILL script.
    """
    TMx /= 100
    Tc = 3.7237e02 * TMx**3 - 6.9152e02 * TMx**2 - 4.1385e02 * TMx**1 + 5.8000e02
    Tnorm = 20 / Tc
    K1 = (
        1e4
        * (-3.5725e01 * TMx**3 + 5.0920e01 * TMx**2 - 1.5257e01 * TMx**1 - 1.3579e00)
        * (1 - Tnorm) ** (-6.3643e00 * TMx**2 + 2.3779e00 * TMx**1 + 3.0318e00)
    )
    K2 = (
        1e4
        * (
            1.5308e02 * TMx**4
            - 2.2600e01 * TMx**3
            - 4.9734e01 * TMx**2
            + 1.5822e01 * TMx**1
            - 5.5522e-01
        )
        * (1 - Tnorm) ** 7.2652e00
    )

    oneoneone = K1 / 3 + K2 / 27
    oneonezero = K1 / 4
    onezerozero = 0

    axes_names = np.array(["1 1 1", "1 1 0", "1 0 0"])
    axes_values = np.array([oneoneone, oneonezero, onezerozero])
    sorted_axes = axes_names[np.argsort(axes_values)]
    return sorted_axes


def dir_to_rot_mat(x, x_prime):
    """
    Creates a rotation matrix that rotates from one crystallographic
    direction to another.

    Inputs
    ------
    x: string
    vector to be rotated from. Specified as a space separated string
    e.g. '1 0 0' for compatibility with MERRILL script.

    x_prime: string
    vector to be rotated to. Specified in the same way.

    Returns:
    rot_mat: numpy array
    Rotation matrix.
    """
    a = np.array(x.split(" ")).astype(float)
    b = np.array(x_prime.split(" ")).astype(float)

    # N.B. if these are crystollagraphic directions should they be
    # reciprocal?

    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    # Create an euler vector to turn into rotation matrix
    theta = np.arccos(np.dot(a, b))  # Angular difference
    v = np.cross(a, b)  # Pole
    euler_vector = v / np.linalg.norm(v) * theta

    # Use scipy rotation package
    rot = Rotation.from_rotvec(euler_vector)
    angles = rot.as_matrix()

    # Catch a bug that somehow makes angles nan, if so set to an
    # identity matrix (no rotation).
    if np.any(np.isnan(angles)):
        angles = np.identity(3)

    return angles


def get_material_parms(TMx, alignment, T):
    """
    Todo: Incorporate materials.py framework into material parameters
    Calculates the material parameters for titanomagnetites of different
    compositions.

    Inputs
    ------
    TMx: float
    Titanomagnetite composition % (0 - 100)

    alignment: string
    Either `easy` or `hard`. Specifies the magnetocrystalline direction
    that should be aligned with the x direction, which for our
    ellipsoids is the major (shape easy) axis.

    T: float
    Temperature (degrees C).

    Returns
    -------
    rot_mat:
    Rotation matrix associated with magnetocrystalline anisotropy
    energy.

    K1, K2: floats
    magnetocrystalline anisotropy constants.

    Ms: float
    Saturation magnetization.
    """
    if T < 0:
        raise ValueError("Error: Temperature should be greater than 0 degrees")
    anis = calculate_anisotropies(TMx)
    rot_to = "1 0 0"

    if alignment == "easy":
        rot_from = anis[0]
    elif alignment == "hard":
        rot_from = anis[2]

    rot_mat = dir_to_rot_mat(rot_from, rot_to)
    TMx /= 100
    Tc = 3.7237e02 * TMx**3 - 6.9152e02 * TMx**2 - 4.1385e02 * TMx**1 + 5.8000e02
    if T >= Tc:
        raise ValueError(
            "Error: Temperature should not exceed \
            Curie temperature (%1.0i"
            % Tc
            + "°C)"
        )
    Tnorm = T / Tc
    K1 = (
        1e4
        * (-3.5725e01 * TMx**3 + 5.0920e01 * TMx**2 - 1.5257e01 * TMx**1 - 1.3579e00)
        * (1 - Tnorm) ** (-6.3643e00 * TMx**2 + 2.3779e00 * TMx**1 + 3.0318e00)
    )
    K2 = (
        1e4
        * (
            1.5308e02 * TMx**4
            - 2.2600e01 * TMx**3
            - 4.9734e01 * TMx**2
            + 1.5822e01 * TMx**1
            - 5.5522e-01
        )
        * (1 - Tnorm) ** 7.2652e00
    )
    Ms = (
        -2.8106e05 * TMx**3 + 5.2850e05 * TMx**2 - 7.9381e05 * TMx**1 + 4.9537e05
    ) * (1 - Tnorm) ** 4.0025e-01
    return (rot_mat, K1, K2, Ms)


@Partial(jit, static_argnums=6)
def energy_surface(
    k1,
    k2,
    rot_mat,
    Ms,
    LMN,
    ext_field,
    n_points=100,
    bounds=jnp.array([[0, 2 * jnp.pi], [-jnp.pi / 2, jnp.pi / 2]]),
):
    """
    Calculates the total SD energy for theta, phi angles along an
    equirectangular grid. Note that this is not an area preserving
    grid which may have some numerical effects - perhaps a UV map
    would be better.

    N.B. This has a lot of arguments, maybe specify a dictionary?

    Inputs:
    ------
    k1, k2: floats
    Magnetocrystalline anisotropy constants

    rot_mat: numpy array
    Rotation matrix for magnetocrystalline anisotropy field.

    Ms: float
    Saturation magnetization.

    LMN: numpy array
    Demagnetizing factors of Ellipsoid along x, y, z directions.

    ext_field: numpy array
    Array of [theta, phi, B] for the external field, where theta and
    phi are expressed in DEGREES and B is in Tesla.

    n_points: int
    Number of points in theta, phi direction to calculate energy.

    bounds: numpy array
    Theta, phi bounds within which to calculate the energies. Specified
    as [[theta_min,theta_max],[phi_min,phi_max]]

    Returns:
    thetas: numpy array
    grid of theta angles

    phis: numpy array
    grid of phi angles

    energies: numpy array
    grid of energies associated with these thetas and phis.
    """
    thetas = jnp.linspace(bounds[0, 0], bounds[0, 1], n_points)
    phis = jnp.linspace(bounds[1, 0], bounds[1, 1], n_points)
    thetas, phis = jnp.meshgrid(thetas, phis)
    energy_temp = lambda theta, phi: energy_ang(
        [theta, phi], k1, k2, rot_mat, LMN, Ms, ext_field
    )
    energy_temp = vmap(energy_temp)
    energy_array = energy_temp(thetas.flatten(), phis.flatten())
    energies = jnp.reshape(energy_array, thetas.shape)
    return (thetas, phis, energies)
