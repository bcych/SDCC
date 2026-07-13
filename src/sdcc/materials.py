import numpy as np


class Material:
    """
    Base class for calculating important quantities for MERRILL.
    Material classes can inherit from this class, and must have
    a "Tc" (Curie Temp) attribute and the following methods:

    Ms(T) - a function that calculates saturation magnetization
    as a function of temperature

    Aex(T) - a function that calculates the exchange constant
    as a function of temperature

    anisotropy_constants -  a function that calculates the
    anisotropy constants as a function of temperature and
    outputs them as a dictionary.
    """

    def __init__(self):
        self.mu = 4 * np.pi * 1e-07

    def Kd(self, T):
        return 0.5 * self.mu * self.Ms(T) ** 2

    def exch_len(self, T):
        return np.sqrt(self.Aex(T) / self.Kd(T))

    def min_exch_len(self, Tmin, Tmax):
        Ts = np.arange(Tmin, Tmax + 0.1, 0.1)
        exch_lens = self.exch_len(Ts)
        return min(exch_lens)

    def mesh_sizing(self, ESVD, Tmin, Tmax, availSizes):
        LambdaEx = self.min_exch_len(Tmin, Tmax)
        realMeshSize = min(LambdaEx * 0.85, ESVD * 1e-9 * 0.06666) * 1e9
        availSizes = np.sort(availSizes)
        closestAvailSize = availSizes[availSizes < realMeshSize][-1]
        return closestAvailSize


class TM(Material):
    """
    Material class for the titanomagnetite solid solution
    series.

    Parameters
    ------
    TMx: float
    Titanomagnetite composition (%, 0 - 60)
    """

    def __init__(self, TMx):
        self.TMx = TMx
        comp = self.TMx / 100
        self.Tc = (
            3.7237e02 * comp**3 - 6.9152e02 * comp**2 - 4.1385e02 * comp + 5.8000e02
        )
        self.anisform = "Cubic"
        super().__init__()

    def Ms(self, T):
        comp = self.TMx / 100
        val = (
            -2.8106e05 * comp**3 + 5.2850e05 * comp**2 - 7.9381e05 * comp**1 + 4.9537e05
        ) * (1 - T / self.Tc) ** 4.0025e-01
        return val

    def Aex(self, T):
        val = (
            1e-11
            * 1.3838e00
            * ((self.Tc + 273.15) / 853.15)
            * (1 - T / self.Tc) ** 6.7448e-01
        )
        return val

    def anisotropy_constants(self, T):
        comp = self.TMx / 100
        k1 = (
            1e4
            * (
                -3.5725e01 * comp**3
                + 5.0920e01 * comp**2
                - 1.5257e01 * comp**1
                - 1.3579e00
            )
            * (1 - T / self.Tc)
            ** (-6.3643e00 * comp**2 + 2.3779e00 * comp**1 + 3.0318e00)
        )
        k2 = (
            1e4
            * (
                1.5308e02 * comp**4
                - 2.2600e01 * comp**3
                - 4.9734e01 * comp**2
                + 1.5822e01 * comp**1
                - 5.5522e-01
            )
            * (1 - T / self.Tc) ** 7.2652e00
        )
        return {"k1": k1, "k2": k2}
