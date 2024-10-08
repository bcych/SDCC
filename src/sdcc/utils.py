import numpy as np
import warnings
from scipy.interpolate import griddata


def fib_sphere(n=1000):
    """
    Algorithm for producing directions from a Fibonacci sphere

    Inputs
    ------
    n: int
    Number of directions

    Returns
    -------
    xyz: array
    Array of 3D cartesian directions
    """
    goldenRatio = (1 + 5**0.5) / 2
    i = np.arange(0, n)
    theta = 2 * np.pi * i / goldenRatio
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.array([x, y, z]).T


def smooth_surface(xx, yy, zs, grid_x, grid_y, kind="cubic"):
    """
    Interpolates critical size values on a grid to obtain
    a smoothed surface.

    Inputs
    ------
    xx: numpy array
    Meshgrid of TM compositions

    yy: numpy array
    Meshgrid of shapes

    zs: numpy array
    Critical sizes at (xx,yy)

    grid_x: float or array
    TMx value(s) to calculate smoothed surface for.

    grid_y: float or array
    Shapes to calculate smoothed surface for.

    kind: string
    What kind of smoothing to do (should always be 'cubic')

    Returns
    -------
    grid_z: numpy array
    Interpolated critical size(s)
    """
    xx = xx.flatten()
    yy = yy.flatten()
    zs_nan = np.zeros_like(zs.flatten())
    zs_nan[np.isnan(zs.flatten())] = 1
    xx = xx[~np.isnan(zs.flatten())]
    yy = yy[~np.isnan(zs.flatten())]
    zs = zs.flatten()[~np.isnan(zs.flatten())]

    grid_y = np.log(grid_y)

    xy = np.array([xx, np.log(yy)]).T
    grid_z = griddata(
        xy, zs, (grid_x, grid_y), method=kind, fill_value=np.nan, rescale=True
    )
    return grid_z


def calc_d_min(TMx, alignment, PRO, OBL):
    """
    Calculates the SD size limit (d_min) using data from Cych et al
    (Magnetic Domain States and Critical Sizes in the Titanomagnetite Series, 2024)

    Inputs
    ------
    TMx: float
    Titanomagnetite Composition

    alignment: str
    Magnetocrystalline axis aligned with shape easy axis, either 'easy','intermediate'
    or 'hard'

    PRO: float
    Prolateness (> 1)

    OBL: float
    Oblateness (> 1)

    Returns
    -------
    d_min: float
    Critical size below which the SV state does not exist.
    """

    if PRO > 1.0 and OBL > 1.0:
        warnings.warn(
            "Warning! The maximum SD size limit is not known for triaxial particles, proceed with caution!"
        )
        return np.nan
    elif PRO > 3 or OBL > 3:
        warnings.warn(
            "Warning! The maximum SD size limit is not known for this shape, proceed with caution!"
        )
    elif TMx > 60:
        warnings.warn(
            "Warning! The maximum SD size limit is not known for this shape, proceed with caution!"
        )
    elif TMx < 0:
        raise ValueError("Titanomagnetite compositon must be between 0 and 100")
    easy_d_min = np.array(
        [
            [65, 65, 65, 65, 75, 75, 65, 85, 95, 105, 145, 165, 205],
            [65, 65, 75, 75, 85, 85, 75, 95, 95, 115, 155, 185, 215],
            [75, 75, 75, 85, 85, 85, 75, 95, 105, 125, 165, 195, 225],
            [55, 75, 85, 85, 95, 95, 85, 105, 115, 135, 175, 215, 262.5],
            [55, 65, 85, 95, 105, 95, 85, 105, 125, 155, 185, 225, 262.5],
            [55, 75, 65, 95, 105, 105, 95, 115, 135, 175, 205, 235, 287.5],
            [65, 75, 65, 85, 115, 115, 95, 125, 155, 175, 215, 262.5, 312.5],
            [75, 95, 75, 105, 125, 115, 105, 125, 165, 195, 225, 262.5, 312.5],
            [75, 95, 105, 105, 135, 125, 115, 135, 175, 205, 245, 287.5, 337.5],
            [95, 105, 105, 135, 145, 135, 125, 145, 175, 225, 262.5, 312.5, 362.5],
            [125, 125, 135, 145, 155, 155, 135, 165, 185, 215, 287.5, 337.5, 412.5],
            [105, 95, 135, 145, 155, 165, 145, 185, 195, 215, 287.5, 337.5, 412.5],
            [115, 145, 155, 165, 175, 185, 155, 195, 215, 287.5, 337.5, 412.5, 487.5],
        ]
    )

    hard_d_min = np.array(
        [
            [55, 55, 65, 65, 55, 65, 65, 75, 75, 85, 105, 115, 135],
            [65, 55, 65, 75, 65, 65, 75, 75, 75, 85, 95, 115, 125],
            [65, 65, 75, 75, 65, 75, 75, 75, 75, 85, 95, 105, 115],
            [65, 65, 75, 75, 75, 75, 85, 75, 95, 95, 95, 105, 115],
            [75, 75, 85, 85, 75, 75, 85, 85, 105, 95, 95, 105, 115],
            [85, 85, 85, 85, 85, 85, 95, 95, 105, 95, 105, 115, 125],
            [85, 95, 95, 95, 95, 95, 95, 95, 115, 115, 125, 135, 135],
            [95, 95, 95, 95, 95, 95, 105, 105, 125, 125, 135, 145, 155],
            [105, 105, 105, 105, 95, 105, 115, 115, 125, 135, 145, 145, 155],
            [95, 115, 115, 115, 105, 105, 125, 125, 145, 145, 145, 155, 165],
            [85, 125, 125, 135, 125, 125, 135, 135, 155, 145, 155, 175, 195],
            [105, 105, 145, 145, 155, 155, 145, 145, 165, 185, 205, 235, 262.5],
            [135, 155, 165, 175, 175, 185, 155, 175, 165, 185, 205, 225, 245],
        ]
    )

    int_d_min = np.amin([easy_d_min, hard_d_min], axis=0)

    TMxs = np.arange(0, 65, 5)
    shapes = np.logspace(np.log10(1 / 3), np.log10(3), 13)
    shapes, TMxs = np.meshgrid(shapes, TMxs)

    if alignment.lower() == "intermediate":
        warnings.warn(
            "Warning! The maximum SD size limit is not known for MI-SE particles. Taking the minimum of ME-SE and MH-SE"
        )
        zz = int_d_min
    elif alignment.lower() == "easy":
        zz = easy_d_min
    elif alignment.lower() == "hard":
        zz = hard_d_min
    else:
        raise ValueError("alignment must be one of easy, intermediate, hard")

    d_min = smooth_surface(TMxs, shapes, zz, TMx, PRO / OBL)
    return np.round(float(d_min), 1)
