from ctypes import alignment

import numpy as np
from sdcc.energy import get_material_parms, angle2xyz
from sdcc.utils import fib_hypersphere, calc_d_min
from sdcc.barriers import find_T_barriers, find_B_barriers
from scipy.interpolate import splprep, splrep, BSpline
from scipy.optimize import minimize
import gc
from sdcc.energy import xyz2angle
import pickle


def energy_spline(x, y):
    """
    Creates a piecewise scalar B-Spline from a set of x,y data.

    Parameters
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

    Parameters
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

    Parameters
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

    Parameters
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
    direction = np.clip(direction, -1.0, 1.0)
    direction = np.array(xyz2angle(direction.T)).T
    if len(direction.shape) > 1:
        direction[:, 0] = direction[:, 0] % (2 * np.pi)
    else:
        direction[0] = direction[0] % (2 * np.pi)
    return np.nan_to_num(direction, nan=-np.inf)


class EnergyLandscape:
    """
    Generic EnergyLandscape class that other classes inherit from.
    """

    def __init__(
        self,
        min_dir,
        min_energy,
        min_Ms,
        bar_dir,
        bar_energy,
        bar_Ms,
        info_dict,
        switching_param,
        param,
        ext,
    ):

        self.min_dir = min_dir
        self.min_energy = min_energy
        self.min_Ms = min_Ms
        self.bar_dir = bar_dir
        self.bar_energy = bar_energy
        self.bar_Ms = bar_Ms
        self.info_dict = info_dict
        self.switching_param = switching_param
        self.ext = ext

        if switching_param == "T":
            self.Ts = param
            self.T_max = info_dict["Maximum Temperature"]
            self.T_min = info_dict["Minimum Temperature"]
            self.get_params = self.get_params_T

        elif switching_param == "B":
            self.Bs = param
            self.B_max = info_dict["Maximum Field"]
            self.B_min = info_dict["Minimum Field"]
            self.get_params = self.get_params_B

        else:
            raise ValueError("switching parameter must be T or B")

    def Ms(self, val):
        return self.Ms_func(val)

    def __repr__(self):
        retstr = """Energy Landscape:

Properties
----------
"""
        units_dict = {
            "Temperature": "C",
            "Minimum Temperature": "C",
            "Maximum Temperature": "C",
            "Size": "nm",
            "Minimum Size": "nm",
            "Maximum Size": "nm",
            "Minimum Field": "T",
            "Maximum Field": "T",
            "Field Spacing": "T",
        }
        for key in self.info_dict:
            if key in units_dict.keys():
                retstr += f"{key}: {self.info_dict[key]} {units_dict[key]}\n"
            else:
                retstr += f"{key}: {self.info_dict[key]}\n"
        return retstr

    def get_params_B(self, B):
        """
        Gets directions and energies associated with LEM states and
        barriers for a grain as a function of field.

        Parameters
        ------
        B: int, float or numpy array
            External Field (T)

        Returns
        -------
        params: dict
            Dictionary of arrays for directions and energies.
        """

        Ms = self.Ms(B)
        min_dir = []
        min_e = []
        min_m = []

        if isinstance(B, (float, int, np.float64, np.int64)):
            assert (B <= self.B_max) & (B >= -self.B_max), (
                "B must be between " + str(-self.B_max) + " and " + str(self.B_max)
            )
            bar_e = np.full(self.bar_energy[0].shape, np.inf)
            bar_dir = np.full(
                (len(self.min_energy[0]), len(self.min_energy[0]), 2), np.inf
            )
            bar_m = np.full(self.bar_energy[0].shape, np.inf)

        elif isinstance(B, np.ndarray):
            assert (np.amax(B) <= self.B_max) & (np.amin(B) >= -self.B_max), (
                "B must be between " + str(-self.B_max) + " and " + str(self.B_max)
            )
            bar_e = np.full(
                (len(self.min_energy[0]), len(self.min_energy[0]), 2, B.shape), np.inf
            )
            bar_dir = np.full(
                (len(self.min_energy[0]), len(self.min_energy[0]), 2, B.shape), np.inf
            )

            bar_m = np.full(
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
            t, c, k = self.min_Ms[splindex][i]
            min_m.append(energy_result(t, c, k, B))
            t, c, k = self.min_dir[splindex][i]
            min_dir.append(direction_result(t, c, k, B))
            for j in range(len(self.min_energy[0])):
                t, c, k = self.bar_energy[splindex][i, j]
                bar_e[i, j] = energy_result(t, c, k, B)
                t, c, k = self.bar_dir[splindex][i, j]
                bar_dir[i, j] = direction_result(t, c, k, B)
                t, c, k = self.bar_Ms[splindex][i, j]
                bar_m[i, j] = energy_result(t, c, k, B)
        if B == 0:
            min_e2 = []
            bar_e2 = np.full(self.bar_energy[0].shape, np.inf)
            min_m2 = []
            bar_m2 = np.full(self.bar_energy[0].shape, np.inf)
            for i in range(len(self.min_energy[0])):
                t, c, k = self.min_energy[1][i]
                min_e2.append(energy_result(t, c, k, B))
                t, c, k = self.min_Ms[1][i]
                min_m2.append(energy_result(t, c, k, B))
                for j in range(len(self.min_energy[0])):
                    t, c, k = self.bar_energy[1][i, j]
                    bar_e2[i, j] = energy_result(t, c, k, B)
                    t, c, k = self.bar_Ms[1][i, j]
                    bar_m2[i, j] = energy_result(t, c, k, B)
            min_e = np.mean([min_e, min_e2], axis=0)
            bar_e = np.mean([bar_e, bar_e2], axis=0)
            min_m = np.mean([min_m, min_m2], axis=0)
            bar_m = np.mean([bar_m, bar_m2], axis=0)

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
            "min_m": Ms * np.array(min_m),
            "bar_dir": np.array(bar_dir),
            "bar_e": np.array(bar_e),
            "bar_m": Ms * np.array(bar_m),
            "B": B,
            "T": self.info_dict["Temperature"],
        }

        return params

    def get_params_T(self, T):
        """
        Gets directions and energies associated with LEM states and
        barriers for a grain as a function of temperature.

        Parameters
        ------
        T: int, float or numpy array
            Temperature(s) (degrees C)

        Returns
        -------
        params: dict
            Dictionary of arrays for directions and energies.
        """
        Ms = self.Ms(T)
        min_dir = []
        min_e = []
        min_Ms = []

        if isinstance(T, (float, int, np.float64, np.int64)):
            assert (T <= self.T_max) & (T >= self.T_min), (
                "T must be between " + str(self.T_min) + " and " + str(self.T_max)
            )
            bar_e = np.full(self.bar_energy.shape, np.inf)
            bar_dir = np.full((len(self.min_energy), len(self.min_energy), 2), np.inf)
            bar_Ms = np.full(self.bar_energy.shape, np.inf)

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
            bar_Ms = np.full(
                len(self.min_energy), len(self.min_energy), 2, T.shape, np.inf
            )

        else:
            print(T)
            raise TypeError("T should not be type " + str(type(T)))

        for i in range(len(self.min_energy)):
            t, c, k = self.min_energy[i]
            min_e.append(energy_result(t, c, k, T))
            t, c, k = self.min_dir[i]
            min_dir.append(direction_result(t, c, k, T))
            t, c, k = self.min_Ms[i]
            min_Ms.append(energy_result(t, c, k, T))
            for j in range(len(self.min_energy)):
                t, c, k = self.bar_energy[i, j]
                bar_e[i, j] = energy_result(t, c, k, T)
                t, c, k = self.bar_dir[i, j]
                bar_dir[i, j] = direction_result(t, c, k, T)
                t, c, k = self.bar_Ms[i, j]
                bar_Ms[i, j] = energy_result(t, c, k, T)

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
            "min_m": Ms * np.array(min_Ms),
            "bar_dir": np.array(bar_dir),
            "bar_e": np.array(bar_e),
            "bar_m": Ms * np.array(bar_Ms),
            "T": T,
        }
        return params

    def to_file(self, fname):
        """
        Saves HELs object to file.

        Parameters
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


class OldGEL(EnergyLandscape):
    """ """

    def __init__(self, gel):

        self.TMx = gel.TMx
        self.alignment = gel.alignment

        Ts = np.arange(gel.T_min, gel.T_max + 1, 1)
        M_spoof = np.ones(len(Ts))
        M_spoof[1] -= 1e-12
        MMs = energy_spline(Ts, M_spoof)

        min_Ms = np.empty(gel.min_energy.shape, dtype="object")
        bar_Ms = np.empty(gel.bar_energy.shape, dtype="object")
        for i in range(len(min_Ms)):
            min_Ms[i] = MMs
            for j in range(len(min_Ms)):
                bar_Ms[i, j] = MMs

        info_dict = {
            "Material": f"TM{str(gel.TMx).zfill(2)}",
            "Shape Class": "Ellipsoid",
            "Long Axis Is": gel.alignment,
            "Prolateness": gel.PRO,
            "Oblateness": gel.OBL,
            "Minimum Size": 0,
            "Maximum Size": gel.d_min,
            "Type": "Thermal",
            "Minimum Temperature": gel.T_min,
            "Maximum Temperature": gel.T_max,
            "Generated By": "SDCC",
        }

        super().__init__(
            gel.min_dir,
            gel.min_energy,
            min_Ms,
            gel.bar_dir,
            gel.bar_energy,
            bar_Ms,
            info_dict,
            "T",
            Ts,
            "gel",
        )

    def Ms_func(self, T):
        a, b, c, Ms = get_material_parms(self.TMx, self.alignment, T)
        return Ms


class GEL(EnergyLandscape):
    """
    Class for storing energy barrier results at all temperatures for a
    given grain geometry and composition.

    Todo:
    1. This should inherit from some base class shared with Hysteresis.
    2. The run through of temperatures should be parallelized for much
    quicker object creation.

    Parameters
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

        M_spoof = np.ones(len(Ts))
        M_spoof[1] -= 1e-12
        MMs = energy_spline(Ts, M_spoof)

        min_Ms = np.empty(self.min_energy.shape, dtype="object")
        bar_Ms = np.empty(self.bar_energy.shape, dtype="object")
        for i in range(len(min_Ms)):
            min_Ms[i] = MMs
            for j in range(len(min_Ms)):
                bar_Ms[i, j] = MMs

        info_dict = {
            "Material": f"TM{str(self.TMx).zfill(2)}",
            "Shape Class": "Ellipsoid",
            "Long Axis Is": self.alignment,
            "Prolateness": self.PRO,
            "Oblateness": self.OBL,
            "Minimum Size": 0,
            "Maximum Size": self.d_min,
            "Type": "Thermal",
            "Minimum Temperature": self.T_min,
            "Maximum Temperature": self.T_max,
            "Generated By": "SDCC",
        }

        super().__init__(
            self.min_dir,
            self.min_energy,
            min_Ms,
            self.bar_dir,
            self.bar_energy,
            bar_Ms,
            info_dict,
            "T",
            Ts,
            "gel",
        )

    def Ms_func(self, T):
        a, b, c, Ms = get_material_parms(self.TMx, self.alignment, T)
        return Ms


class HEL(EnergyLandscape):
    """
    Class for storing energy barrier results at all field strengths for a
    given grain geometry, composition and field direction.

    Todo:
    1. This should inherit from some base class shared with GEL.
    2. The run through of temperatures should be parallelized for much
    quicker object creation.

    Parameters
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

    rot_mat: 3x3 matrix
        Rotation matrix that rotates field direction from [1,0,0].
        Note that although there are a number of matrices that will
        solve for a particular field direction, a single field vector
        cannot be used to describe the inverse transformation applied
        to particles when rotating all the fields into the same direction.

    B_max: float
        Maximum field (Tesla)

    B_spacing: float
        Spacing of field steps (Tesla)
    """

    def __init__(self, TMx, alignment, PRO, OBL, rot_mat, B_max, B_spacing=0.001, T=20):
        self.d_min = calc_d_min(TMx, alignment, PRO, OBL)
        B_dir = rot_mat @ np.array([1.0, 0.0, 0.0])
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
        del first_theta_lists
        del first_phi_lists
        del first_energy_lists
        del first_theta_mats
        del first_phi_mats
        del first_energy_mats

        del second_theta_lists
        del second_phi_lists
        del second_energy_lists
        del second_theta_mats
        del second_phi_mats
        del second_energy_mats

        del theta_lists
        del phi_lists
        del min_energy_lists
        del theta_mats
        del phi_mats
        del energy_mats

        gc.collect()
        gc.collect(generation=2)

        M_spoof = np.ones(len(Bs))
        M_spoof[1] -= 1e-12
        MMs = energy_spline(Bs, M_spoof)
        MMs2 = energy_spline(-Bs[::-1], M_spoof[::-1])

        min_Ms = np.empty(self.min_energy[0].shape, dtype="object")
        bar_Ms = np.empty(self.bar_energy[0].shape, dtype="object")
        min_Ms2 = np.empty(self.min_energy[0].shape, dtype="object")
        bar_Ms2 = np.empty(self.bar_energy[0].shape, dtype="object")

        for i in range(len(min_Ms)):
            min_Ms[i] = MMs
            min_Ms2[i] = MMs2
            for j in range(len(min_Ms)):
                bar_Ms[i, j] = MMs
                bar_Ms2[i, j] = MMs2

        min_Ms = [min_Ms2, min_Ms]
        bar_Ms = [bar_Ms2, bar_Ms]

        info_dict = {
            "Material": f"TM{str(TMx).zfill(2)}",
            "Shape Class": "Ellipsoid",
            "Long Axis Is": alignment,
            "Prolateness": PRO,
            "Oblateness": OBL,
            "Minimum Size": 0,
            "Maximum Size": self.d_min,
            "Type": "High-Field",
            "Minimum Field": 0,
            "Maximum Field": B_max,
            "Field Spacing": B_spacing,
            "Field Direction": B_dir,
            "Field Rotation Matrix": rot_mat,
            "Temperature": T,
            "Generated By": "SDCC",
        }

        super().__init__(
            self.min_dir,
            self.min_energy,
            min_Ms,
            self.bar_dir,
            self.bar_energy,
            bar_Ms,
            info_dict,
            "B",
            Bs,
            "hel",
        )

    def Ms_func(self, B):
        a, b, c, Ms = get_material_parms(self.TMx, self.alignment, self.T)
        return Ms


class OldHEL(EnergyLandscape):
    """ """

    def __init__(self, hel):

        B_spacing = 0.001  # Hard-coded but doesn't really matter for Ms
        Bs = np.arange(0, hel.B_max + B_spacing, B_spacing)
        M_spoof = np.ones(len(Bs))
        M_spoof[1] -= 1e-12
        MMs = energy_spline(Bs, M_spoof)
        MMs2 = energy_spline(-Bs[::-1], M_spoof[::-1])

        min_Ms = np.empty(hel.min_energy[0].shape, dtype="object")
        bar_Ms = np.empty(hel.bar_energy[0].shape, dtype="object")
        min_Ms2 = np.empty(hel.min_energy[0].shape, dtype="object")
        bar_Ms2 = np.empty(hel.bar_energy[0].shape, dtype="object")
        self.T = 20

        for i in range(len(min_Ms)):
            min_Ms[i] = MMs
            min_Ms2[i] = MMs2
            for j in range(len(min_Ms)):
                bar_Ms[i, j] = MMs
                bar_Ms2[i, j] = MMs2

        min_Ms = [min_Ms2, min_Ms]
        bar_Ms = [bar_Ms2, bar_Ms]
        self.TMx = hel.TMx
        self.alignment = hel.alignment

        info_dict = {
            "Material": f"TM{str(hel.TMx).zfill(2)}",
            "Shape Class": "Ellipsoid",
            "Long Axis Is": hel.alignment,
            "Prolateness": hel.PRO,
            "Oblateness": hel.OBL,
            "Minimum Size": 0,
            "Maximum Size": hel.d_min,
            "Type": "High-Field",
            "Minimum Field": 0,
            "Maximum Field": hel.B_max,
            "Field Direction": hel.B_dir,
            "Field Rotation Matrix": hel.rot_mat,
            "Temperature": hel.T,
            "Generated By": "SDCC",
        }

        super().__init__(
            hel.min_dir,
            hel.min_energy,
            min_Ms,
            hel.bar_dir,
            hel.bar_energy,
            bar_Ms,
            info_dict,
            "B",
            Bs,
            "hel",
        )

    def Ms_func(self, B):
        a, b, c, Ms = get_material_parms(self.TMx, self.alignment, 20)
        return Ms


class HELs:
    """
    Class for storing a variety of different HELs objects,
    with different field orientations.

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

    B_max: float
        Maximum field (Tesla)

    B_spacing: float
        Spacing of field steps (Te

    """

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
        n_dirs=50,
        HEL_list=None,
    ):

        if type(HEL_list) == type(None):
            if type(rot_mats) == type(None):
                rot_mats = fib_hypersphere(n_dirs)
            HEL_list = []
            B_dirs = []
            i = 1
            for rot_mat in rot_mats:
                B_dirs.append(rot_mat @ np.array([1, 0, 0]))
                print("\n", f"Working on orientation {i} of {n_dirs}")
                HEL_list.append(
                    HEL(TMx, alignment, PRO, OBL, rot_mat, B_max, B_spacing, T)
                )
                i += 1
        else:
            i = 1
            B_dirs = []
            rot_mats = []
            for hel in HEL_list:
                rot_mat = hel.info_dict["Field Rotation Matrix"]
                B_dirs.append(rot_mat @ np.array([1, 0, 0]))
                i += 1
                rot_mats.append(rot_mat)
            rot_mats = np.array(rot_mats)

        self.HEL_list = HEL_list
        self.B_dirs = B_dirs
        self.rot_mats = rot_mats
        self.info_dict = {
            "Material": self.HEL_list[0].info_dict["Material"],
            "Shape Class": self.HEL_list[0].info_dict["Shape Class"],
            "Long Axis Is": alignment,
            "Prolateness": PRO,
            "Oblateness": OBL,
            "Maximum Size": self.HEL_list[0].info_dict["Minimum Size"],
            "Maximum Size": self.HEL_list[0].info_dict["Minimum Size"],
            "Type": "High-Field Collection",
            "Maximum Field": self.HEL_list[0].info_dict["Minimum Field"],
            "Maximum Field": self.HEL_list[0].info_dict["Minimum Field"],
            "Number of Field Directions": n_dirs,
        }

    def __getitem__(self, index: int):
        return self.HEL_list[index]

    def to_file(self, fname):
        """
        Saves HELs object to file.

        Parameters
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
        retstr = """HELs Energy Landscape Collection:

Properties
----------
"""
        units_dict = {
            "Temperature": "C",
            "Minimum Temperature": "C",
            "Maximum Temperature": "C",
            "Size": "nm",
            "Minimum Size": "nm",
            "Maximum Size": "nm",
            "Minimum Field": "T",
            "Maximum Field": "T",
            "Field Spacing": "T",
        }
        for key in self.info_dict:
            if key in units_dict.keys():
                retstr += f"{key}: {self.info_dict[key]} {units_dict[key]}\n"
            else:
                retstr += f"{key}: {self.info_dict[key]}\n"
        return retstr


class DomainStructure:
    """
    Class describing a particular domain structure for a particle
    Allows for particles with multiple domain structures.

    Given a set of directions for a domain structure, calculates all possible
    LEMs using the symmetry operations possible for that particle's shape and
    magnetocrystalline anisotropy. It does this by rotating the grain using the
    symmetry operation matrices in the `symmetries` argument and checking them
    against the known directions within a given angle set using the `min_angle`
    argument. Any new LEMs found are assigned to the `alt_dirs` attribute. Any
    LEMs found to be the "same" are averaged together.

    Parameters
    -----
    idx: int
    Index of the domain structure

    symmetries: list of rotation matrices
    Rotation matrices that preserve the symmetry of the
    particle.

    dirs: n x m x 3 array
    List of n directions at m temperature of the LEM states associated with this
    domain structure.

    MMs: length m array
    M / Ms for the domain structure as a function of temperature.

    e_mins: length m array
    Energy of the domain structure as a function of temperature.

    min_angle: float
    Minimum angle (in degrees) at which LEM states are considered identical, tunable.


    """

    def __init__(self, idx, symmetries, dirs, MMs, e_mins, min_angle=2):
        self.idx = idx
        self.symmetries = symmetries
        self.dirs = dirs
        self.MMs = MMs  # M / Ms
        self.e_mins = e_mins
        self.alt_dirs, self.symmetries = self.calc_alt_dirs(min_angle=min_angle)

        # Add an extra step
        self.alt_dirs = np.append(self.alt_dirs, [self.alt_dirs[-1]], axis=0)
        self.e_mins = np.append(self.e_mins, [0], axis=0)
        self.MMs = np.append(self.MMs, [1], axis=0)

    def calc_alt_dirs(self, min_angle=2):
        """
        Calculates directions using symmetry of particle.

        Parameters
        -----
        min_angle: float
        Minimum angle within which directions are considered the "same" LEM.

        Returns
        -----
        alt_dirs_new: nx3 array of floats
        List of directions

        symmetries_new: nx3x3 array of floats
        List of symmetries associated with those directions
        """
        alts = []
        alt_dirs = []
        # Run through symmetry matrices and calculate new directions
        for R in self.symmetries:
            alts.append(R @ self.dirs[0])
            alt_dirs.append(R @ self.dirs.T)
        alts = np.array(alts)
        alt_dirs = np.array(alt_dirs)

        # Calculate dot products and check for those that are less than min_angle
        dot_prods = np.inner(alts, alts)
        same = dot_prods >= np.cos(np.radians(min_angle))

        # Create groups of consistent directions
        groups = []
        for i in range(len(alts)):
            exists = False
            for group in groups:
                if i in group:
                    exists = True
            if not exists:
                new_group = np.where(same[i] == True)[0]
                groups.append(new_group)

        # Average directions within groups
        alt_dirs_new = []
        new_symmetries = []
        for group in groups:
            alt_dirs_new.append(np.mean(alt_dirs[group], axis=0))
            new_symmetries.append(self.symmetries[group[0]])

        # Return new directions and symmetries
        new_symmetries = np.array(new_symmetries)
        alt_dirs_new = np.array(alt_dirs_new)
        alt_dirs_new = np.transpose(alt_dirs_new, axes=[2, 0, 1])
        alt_dirs_new /= np.linalg.norm(alt_dirs_new, axis=2, keepdims=True)
        return (alt_dirs_new, new_symmetries)


class EnergyTransition:
    """
    Class describing an energy transition / barrier / switching for
    a non single-domain particle.


    Parameters
    -----
    from_struct,to_struct: int
    DomainStructure index that switching occurs from and to.
    Can be the same structure (e.g. long vortex -> long vortex).

    symmetries: list of rotation matrices
    Rotation matrices that preserve the symmetry of the
    particle.

    e_array: m x n x n array
    Energy at the top of the barrier between from_struct and to_struct
    as a function of temperature.

    bdirs: m x n x n x 3 array
    Magnetization direction at the top of the barrier as a function
    of temperature.

    bMMs: m x n x n array
    M / Ms at the top of the barrier as a function of temperature .


    dir_array: m x x x 3 array
    List of n directions at m temperature of the LEM states associated with this
    domain structure.

    min_angle: float
    Minimum angle (in degrees) at which LEM states are considered identical, tunable.

    """

    def __init__(
        self,
        from_struct,
        to_struct,
        symmetries,
        b_dir_array,
        b_MMs_array,
        e_array,
        dir_array,
        min_angle=2,
    ):
        self.from_struct = from_struct
        self.to_struct = to_struct
        self.symmetries = symmetries
        self.es = e_array
        self.bdirs = b_dir_array
        self.bMMs = b_MMs_array
        self.dirs = dir_array
        self.i1, self.i2, self.connectivity = self.find_connectivity()
        self.idxs, self.alt_bdirs = self.find_alt_bdirs(min_angle=min_angle)
        self.alt_bdirs = np.append(self.alt_bdirs, [self.alt_bdirs[-1]], axis=0)
        self.es = np.append(self.es, np.array([[0, 0]]), axis=0)
        self.bMMs = np.append(self.bMMs, np.array([[1, 1]]), axis=0)

    def find_connectivity(self):
        dirs_1 = self.from_struct.alt_dirs[0]
        dirs_2 = self.to_struct.alt_dirs[0]

        d1 = self.dirs[0, 0]
        d2 = self.dirs[0, 1]
        dots1 = np.dot(dirs_1, d1)
        dots2 = np.dot(dirs_2, d2)

        i1 = np.where(dots1 == np.amax(dots1))[0][0]
        i2 = np.where(dots2 == np.amax(dots2))[0][0]
        dirs_dist = np.inner(dirs_1, dirs_2)
        connectivity = np.isclose(dirs_dist, dirs_dist[i1, i2])
        return (i1, i2, connectivity)

    def find_alt_bdirs(self, min_angle=2):
        dirs_1 = self.from_struct.alt_dirs
        dirs_2 = self.to_struct.alt_dirs
        d1 = dirs_1[0, self.i1]
        d2 = dirs_2[0, self.i2]
        coords = []
        alt_bdirs = []
        for R in self.symmetries:
            new_d1 = R @ d1
            new_d2 = R @ d2
            alt_bdirs.append([R @ bdir for bdir in self.bdirs[:, 0]])
            t1 = np.inner(dirs_1[0], new_d1)
            t2 = np.inner(dirs_2[0], new_d2)
            w1 = np.where(t1 == np.amax(t1))[0][0]
            w2 = np.where(t2 == np.amax(t2))[0][0]
            coords.append([w1, w2])
        coords = np.array(coords)
        alt_bdirs = np.array(alt_bdirs)
        dists = np.inner(alt_bdirs[:, 0], alt_bdirs[:, 0])
        same = dists >= np.cos(np.radians(min_angle))
        groups = []
        for i in range(len(alt_bdirs)):
            exists = False
            for group in groups:
                if i in group:
                    exists = True
            if not exists:
                new_group = np.where(same[i] == True)[0]
                groups.append(new_group)

        alt_bdirs_new = []
        for i in range(len(alt_bdirs)):
            for group in groups:
                if i in group:
                    alt_bdirs_new.append(np.mean(alt_bdirs[group], axis=0))

        alt_bdirs_new = np.transpose(np.array(alt_bdirs_new), axes=[1, 0, 2])
        alt_bdirs_new /= np.linalg.norm(alt_bdirs_new, axis=2, keepdims=True)
        return (coords, alt_bdirs_new)


class SMELT(EnergyLandscape):
    """
    Symmetric Micromagnetic Energy Landscape (Thermal)

    Class for storing energy barrier results at all temperatures output
    by MERRILL.

    Parameters
    ----------------
    material: str
        Name of particle material

    shape_class: str
        Class of particle shape, e.g. "Ellipsoid", "Cuboid",
        "Truncated Octahedron" etc.

    alignment: str
        Alignment of magnetocrystalline and shape axis. Either 'hard' or
        'easy' magnetocrystalline always aligned with shape easy.

    PRO: float
        Prolateness of ellipsoid (major / intermediate axis)

    OBL: float
        Oblateness of ellipsoid (intermediae / minor axis)

    d: float or int
        Size of particle (ESVD, nm).

    structs: list of DomainStructure objects
        Domain Structures present in the particle.

    ets: list of EnergyTransition objects
        List of energy transitions between domain structures / states .

    Ts: Array of floats
        Temperatures at which the LEMs and barriers were calculated.

    Ms_func: Function used to calculate saturation magnetization.
    """

    def __init__(
        self,
        material,
        materialName,
        shape_class,
        alignment,
        PRO,
        OBL,
        d,
        structs,
        ets,
        Ts,
    ):
        self.d_min = d + 1
        self.d = d
        self.Material = material
        self.materialName = materialName
        self.alignment = alignment
        self.PRO = PRO
        self.OBL = OBL
        self.Ts = Ts

        # Loop through structures, calculate total number of particles
        ktot = int(np.sum([s.alt_dirs.shape[1] for s in structs]))
        k = 0

        # Make arrays to store LEM info.
        min_energy = np.empty(ktot, dtype="object")
        min_MMs = np.empty(ktot, dtype="object")
        min_dir = np.empty(ktot, dtype="object")

        # Loop through structures and assign
        sdict = {}
        for struct in structs:
            coords = []
            # Loop through different directions of LEMs from a domain structure
            for i in range(struct.alt_dirs.shape[1]):
                # Get net mag direction of LEM states
                alts_temp = struct.alt_dirs[:, i]
                angs = np.array(xyz2angle(alts_temp.T)).T
                # Conversion from MERRILL directions for some reason
                angs[:, 0] = -(angs[:, 0] + np.pi) % (2 * np.pi)
                tck = direction_spline(Ts, angs)
                min_dir[k] = tck

                # Get energies of LEM states
                tck = energy_spline(
                    Ts[~np.isnan(struct.e_mins)],
                    struct.e_mins[~np.isnan(struct.e_mins)],
                )
                min_energy[k] = tck

                # Get net mag magnitude of LEM states
                tck = energy_spline(Ts, struct.MMs)
                min_MMs[k] = tck

                # Index mapping
                coords.append([i, k])
                k += 1
            coords = np.array(coords)
            sdict[struct.idx] = coords

        self.min_energy = min_energy
        self.min_Ms = min_MMs
        self.min_dir = min_dir

        # Do the same thing for the states at the top of the barrier
        bar_e = np.empty((ktot, ktot), dtype="object")
        bar_MMs = np.empty((ktot, ktot), dtype="object")
        bar_dirs = np.empty((ktot, ktot), dtype="object")

        for et in ets:
            # Set up connectivities states
            graph = et.connectivity
            coords = et.idxs

            # Which domain structure is this a barrier between?
            fs = et.from_struct.idx
            ts = et.to_struct.idx

            # Get energies and minimum energies
            es = et.es
            MMs = et.bMMs

            for n in range(len(coords)):
                # Get indices in convention set for LEMs
                i = sdict[fs][sdict[fs][:, 0] == coords[n, 0]][0, 1]
                j = sdict[ts][sdict[ts][:, 0] == coords[n, 1]][0, 1]

                # Set energies to values
                bar_e[i, j] = energy_spline(
                    Ts[~np.isnan(es[:, 0])], es[:, 0][~np.isnan(es[:, 0])]
                )

                # i, j is technically the same barrier as j, i but with a different
                # energy differential
                bar_e[j, i] = energy_spline(
                    Ts[~np.isnan(es[:, 1])], es[:, 1][~np.isnan(es[:, 1])]
                )

                # Do the same for moment magnitude.
                bar_MMs[i, j] = energy_spline(
                    Ts[~np.isnan(MMs[:, 0])], MMs[:, 0][~np.isnan(MMs[:, 0])]
                )
                bar_MMs[j, i] = energy_spline(
                    Ts[~np.isnan(MMs[:, 1])], MMs[:, 1][~np.isnan(MMs[:, 1])]
                )

                # Get all directions associated with this barrier change
                dirs = et.alt_bdirs[:, n]
                angs = np.array(xyz2angle(dirs.T)).T

                # Weird conversion from MERRILL again.
                angs[:, 0] = -(angs[:, 0] + np.pi) % (2 * np.pi)
                bar_dirs[i, j] = direction_spline(Ts, angs)
                bar_dirs[j, i] = direction_spline(Ts, angs)

        # Default values for vectors and scalars.
        default_vec_value = (
            np.array([min(Ts), max(Ts)]),
            np.array([[np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf]]),
            0,
        )
        default_sca_value = (
            np.array([min(Ts), max(Ts)]),
            np.array([np.inf, np.inf]),
            0,
        )

        # Assign default values when there's no barrier.
        for i in range(len(bar_e)):
            for j in range(len(bar_e)):
                if bar_e[i, j] == None:
                    bar_e[i, j] = default_sca_value
                    bar_MMs[i, j] = default_sca_value
                    bar_dirs[i, j] = default_vec_value

        self.bar_Ms = bar_MMs
        self.bar_energy = bar_e
        self.bar_dir = bar_dirs
        self.T_max = max(Ts)
        self.T_min = min(Ts)

        # Create info dictionary about particle.
        info_dict = {
            "Material": materialName,
            "Shape Class": shape_class,
            "Long Axis Is": self.alignment,
            "Prolateness": self.PRO,
            "Oblateness": self.OBL,
            "Minimum Size": self.d,
            "Maximum Size": self.d,
            "Type": "Thermal",
            "Minimum Temperature": self.T_min,
            "Maximum Temperature": self.T_max,
            "Generated By": "MERRILL (Symmetric Landscape)",
        }

        # Turn this into an EnergyLandscape object.
        super().__init__(
            self.min_dir,
            self.min_energy,
            self.min_Ms,
            self.bar_dir,
            self.bar_energy,
            self.bar_Ms,
            info_dict,
            "T",
            Ts,
            "smelt",
        )

    def Ms_func(self, T):
        return self.material.Ms(T)


class OldSMELT(EnergyLandscape):
    """
    LEGACY OBJECT FOR CONVERTING OLD FILES

    Simulated Micromagnetic Energy Landscape (Thermal)

    Class for storing energy barrier results at all temperatures output
    by MERRILL.

    Parameters
    ----------------
    smelt: SMELT object
    Legacy object converted into new format

    material: str
        Name of particle material

    shape_class: str
        Class of particle shape, e.g. "Ellipsoid", "Cuboid",
        "Truncated Octahedron" etc.

    Ms_func: Function used to calculate saturation magnetization.
    """

    def __init__(self, smelt, material, materialName, shape_class):
        # Create info dictionary about particle.
        info_dict = {
            "Material": materialName,
            "Shape Class": shape_class,
            "Long Axis Is": smelt.alignment,
            "Prolateness": smelt.PRO,
            "Oblateness": smelt.OBL,
            "Minimum Size": smelt.d,
            "Maximum Size": smelt.d,
            "Type": "Thermal",
            "Minimum Temperature": min(smelt.Ts),
            "Maximum Temperature": max(smelt.Ts),
            "Generated By": "MERRILL (Symmetric Landscape)",
        }
        self.material = material

        # Turn this into an EnergyLandscape object.
        super().__init__(
            smelt.min_dir,
            smelt.min_energy,
            smelt.min_MMs,
            smelt.bar_dir,
            smelt.bar_energy,
            smelt.bar_MMs,
            info_dict,
            "T",
            smelt.Ts,
            "smelt",
        )

    def Ms_func(self, T):
        return self.material.Ms(T)


class HELM(EnergyLandscape):
    """
    High-field Energy Landscape using Micromagnetics

    Coming Soon!
    """

    def __init__(self):
        return None


def load_particle(filename):
    with open(filename, "rb") as f:
        particle = pickle.load(f)
    return particle
