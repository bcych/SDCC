import numpy as np
from sdcc.barriers import GEL


def time2temp(t, t1, T0, T1, T_amb):
    """
    Converts a time to a temperature assuming Newtonian cooling.

    Inputs
    ------
    t: numpy array
    Array of time steps

    t1: float
    Characteristic cooling time (i.e. time to T1)

    T0: float
    Maximum temperature.

    T1: float
    Characteristic temperature at time T1

    T_amb: float
    Ambient temperature (usually 20C)

    Returns
    -------
    T: numpy array
    Array of temperatures at times ts.
    """
    T_range = T0 - T_amb
    t_rat = t / t1
    T_rat = (T1 - T_amb) / (T0 - T_amb)
    return T_amb + T_range * np.exp(t_rat * np.log(T_rat))


def temp2time(T, t1, T0, T1, T_amb):
    """
    Converts a temperature to a time assuming Newtonian cooling.

    Inputs
    ------
    T: numpy array
    Array of temperature steps

    t1: float
    Characteristic cooling time (i.e. time to T1)

    T0: float
    Maximum temperature.

    T1: float
    Characteristic temperature at time T1

    T_amb: float
    Ambient temperature (usually 20C)

    Returns
    -------
    t: numpy array
    Array of times at temperatures T.
    """
    frac_T = (T - T_amb) / (T0 - T_amb)
    T_rat = (T1 - T_amb) / (T0 - T_amb)
    return t1 * np.log(frac_T) / np.log(T_rat)


class TreatmentStep:
    """
    Class for a thermal treatment step.
    Instead of having a lot of cases, I think it would make more sense
    to make each type of treatment step a separate class that inherits
    from Step - more flexible to create new stuff this way.
    """

    def __init__(self, ts, Ts, field_strs, field_dirs):
        if len(ts) == len(Ts) == len(field_strs) == len(field_dirs):
            self.ts = ts
            self.Ts = Ts
            self.field_strs = field_strs
            self.field_dirs = field_dirs
            self.step_type = "custom"


class CoolingStep(TreatmentStep):
    """
    Treatment Step for cooling a specimen in a known field. Assumes
    Newtonian cooling.
    """

    def __init__(
        self,
        t_start,
        T_start,
        T_end,
        field_str,
        field_dir,
        char_time=1,
        max_temp=None,
        char_temp=None,
    ):
        if char_temp == None or max_temp == None:
            char_temp = T_start - 1
            max_temp = T_start
        Ts = np.arange(T_start, T_end - 1, -1, dtype="float64")
        Ts[-1] = Ts[-1] + 0.5
        self.Ts = Ts
        self.ts = temp2time(self.Ts, char_time, max_temp, char_temp, T_end)
        self.ts = self.ts - self.ts[0]
        self.field_strs = np.full(len(self.Ts), field_str)
        self.field_dirs = np.repeat(np.array([field_dir]), len(self.Ts), axis=0)
        self.step_type = "cooling"
        self.Ts = self.Ts.astype(int)
        self.ts += t_start

    def __repr__(self):
        return f"""Cooling from {self.Ts[0]} to {self.Ts[-1]}°C 
        in {self.field_strs[0]} μT field"""


class HeatingStep(TreatmentStep):
    """
    Treatment step for heating a specimen in a known field.
    Assumes linear heating ramp.
    """

    def __init__(self, t_start, T_start, T_end, field_str, field_dir, lin_rate=1 / 3):
        self.Ts = np.arange(T_start, T_end + 1)
        lin_time = (T_end - T_start) / lin_rate
        self.ts = np.linspace(0, lin_time, len(self.Ts))
        self.field_strs = np.full(len(self.Ts), field_str)
        self.field_dirs = np.repeat(np.array([field_dir]), len(self.Ts), axis=0)
        self.step_type = "heating"
        self.ts += t_start
        self.Ts = self.Ts.astype(int)

    def __repr__(self):
        return f"""Heating from {self.Ts[0]} to {self.Ts[-1]}°C
        in {self.field_strs[0]} μT field"""


class HoldStep(TreatmentStep):
    """
    Treatment step for holding a specimen in a field at constant
    temperature.
    """

    def __init__(
        self, t_start, T_start, field_str, field_dir, hold_steps=100, hold_time=1800.0
    ):
        self.Ts = np.full(hold_steps, T_start)
        self.ts = np.linspace(0, hold_time, len(self.Ts))
        self.field_strs = np.full(len(self.Ts), field_str)
        self.field_dirs = np.repeat(np.array([field_dir]), len(self.Ts), axis=0)
        self.step_type = "hold"
        self.ts += t_start
        self.Ts = self.Ts.astype(int)

    def __repr__(self):
        return f"""Hold at {self.Ts[0]}°C
                in {self.field_strs[0]} μT field"""


class VRMStep(TreatmentStep):
    """
    Treatment step for holding a specimen in a field at constant
    temperature with logarithmically spaced time steps (meant for VRM
    acquisitions).
    """

    def __init__(
        self, t_start, T_start, field_str, field_dir, hold_steps=100, hold_time=1800.0
    ):
        self.Ts = np.full(hold_steps, T_start)
        self.ts = np.logspace(-1, np.log10(hold_time), len(self.Ts))
        self.field_strs = np.full(len(self.Ts), field_str)
        self.field_dirs = np.repeat(np.array([field_dir]), len(self.Ts), axis=0)
        self.step_type = "VRM"
        self.ts += t_start
        self.Ts = self.Ts.astype(int)

    def __repr__(self):
        return f"""VRM acquisition at {self.Ts[0]}°C 
        in {self.field_strs[0]} μT field"""


class SuscStep(TreatmentStep):
    """
    Treatment step for susceptibility measurements. Field changes rather
    than temperature.
    """

    def __init__(
        self, t_start, t_dur, freq, field_peak, field_dir, n_points=100, T=20.0
    ):
        self.Ts = np.full(n_points, T)
        self.ts = np.linspace(0, t_dur, n_points)
        self.field_strs = field_peak * np.sin(self.Ts * freq)
        self.field_dirs = np.repeat(np.array([field_dir]), len(self.Ts), axis=0)
        self.step_type = "VRM"
        self.step_type = "Susc"
        self.ts += t_start
        self.Ts = self.Ts.astype(int)
        self.freq = freq

    def __repr__(self):
        return f"""Susceptibility experiment in a 
        {np.amax(self.field_strs)} μT {self.freq} Hz AC field"""


class HystBranch(TreatmentStep):
    def __init__(
        self, t_start, B_start, B_end, B_step=1000.0, t_step=1, T=20, B_dir=[1, 0, 0]
    ):
        if B_start > B_end:
            B_step = -B_step
        self.field_strs = np.arange(B_start, B_end + B_step, B_step)
        self.ts = np.arange(0.0, len(self.field_strs) * t_step, t_step)
        self.field_dirs = np.repeat(np.array([B_dir]), len(self.field_strs), axis=0)
        self.Ts = np.full(len(self.field_dirs), T)
        self.ts += t_start
        self.step_type = "hysteresis"

    def __repr__(self):
        return f"""Hysteresis Branch from {self.field_strs[0]/1e3} mT to 
{self.field_strs[-1]/1e3} mT in steps of {np.abs(self.field_strs[0]-self.field_strs[1])/1e3} mT"""


def coe_experiment(temp_steps, B_anc, B_lab, B_ancdir, B_labdir):
    """
    Creates a set of thermal treatment steps for an IZZI-Coe experiment

    Inputs
    ------
    temp_steps: numpy array
    Set of temperatures for the Coe experiment

    B_anc: float
    Ancient field strength (T)

    B_lab: float
    Lab field strength (T)

    B_ancdir: numpy array
    Unit vector ancient field direction

    B_labdir: numpy array
    Unit vector lab field direction

    Returns
    -------
    steps: list of treatment.ThermalStep objects
    Set of steps for coe experiment.
    """
    T_max = temp_steps[-1]
    T_min = temp_steps[0]
    steps = []
    TRM = CoolingStep(0, T_max, T_min, B_anc, B_ancdir)
    steps.append(TRM)
    for j in range(1, len(temp_steps)):
        ZjW = HeatingStep(
            steps[-1].ts[-1] + 1e-12,
            T_min,
            temp_steps[j],
            0,
            B_labdir,
        )
        steps.append(ZjW)
        ZjH = HoldStep(
            steps[-1].ts[-1] + 1e-12,
            temp_steps[j],
            0,
            B_labdir,
        )
        steps.append(ZjH)
        ZjC = CoolingStep(
            steps[-1].ts[-1] + 1e-12,
            temp_steps[j],
            T_min,
            0,
            B_labdir,
            max_temp=T_max,
            char_temp=T_max - 1,
        )
        steps.append(ZjC)
        IjW = HeatingStep(
            steps[-1].ts[-1] + 1e-12,
            T_min,
            temp_steps[j],
            B_lab,
            B_labdir,
        )
        steps.append(IjW)
        IjH = HoldStep(
            steps[-1].ts[-1] + 1e-12,
            temp_steps[j],
            B_lab,
            B_labdir,
        )
        steps.append(IjH)
        IjC = CoolingStep(
            steps[-1].ts[-1] + 1e-12,
            temp_steps[j],
            T_min,
            B_lab,
            B_labdir,
            max_temp=T_max,
            char_temp=T_max - 1,
        )
        steps.append(IjC)
    return steps


def relaxation_time(energy_landscape: GEL, B_dir, B):
    """
    Creates a set of thermal treatment steps for a relaxation time
    estimate.

    Inputs
    ------
    energy_landscape: barriers.GEL object
    Object describing energy barriers and LEM states for a particular
    grain geometry. Here it's used to get Ms.

    B_dir: numpy array
    Unit vector field direction

    B: float
    Field strength (uT)

    Returns
    -------
    steps: list of treatment.ThermalStep objects
    Set of steps for coe experiment.
    """
    T_max = energy_landscape.T_max
    T_min = energy_landscape.T_min
    TRM = CoolingStep(0, T_max, T_min, B, B_dir)
    V_Rel = VRMStep(
        TRM.ts[-1],
        T_min,
        0,
        B_dir,
        hold_time=1e17,
        hold_steps=361,
    )
    steps = [TRM, V_Rel]
    return steps


def dunlop_experiment(energy_landscape: GEL, B_dir, B, pTRM_min, pTRM_max, T_steps):
    T_max = energy_landscape.T_max
    T_min = energy_landscape.T_min
    TRM = CoolingStep(0, T_max, T_min, 0, B_dir)
    TRM.field_strs[(TRM.Ts <= pTRM_max) & (TRM.Ts >= pTRM_min)] = B
    steps = [TRM]
    for T in T_steps:
        heating = HeatingStep(steps[-1].ts[-1], T_min, T, 0, B_dir)
        steps.append(heating)
        hold = HoldStep(steps[-1].ts[-1], T, 0, B_dir)
        steps.append(hold)
        cooling = CoolingStep(steps[-1].ts[-1], T, T_min, 0, B_dir)
        steps.append(cooling)
    return steps


def hyst_loop(B_max, B_step=0.001, **kwargs):
    steps = []
    steps.append(HystBranch(0, 0, B_max * 1e6, B_step * 1e6, **kwargs))
    last_t = steps[-1].ts[-1]
    steps.append(
        HystBranch(last_t + 1e-12, B_max * 1e6, -B_max * 1e6, B_step * 1e6, **kwargs)
    )
    last_t = steps[-1].ts[-1]
    steps.append(
        HystBranch(last_t + 1e-12, -B_max * 1e6, B_max * 1e6, B_step * 1e6, **kwargs)
    )
    return steps


def overprint(TRM_mag, TRM_dir, oPrint_mag, oPrint_dir, oPrint_T, oPrint_t, gel):
    T_max = gel.T_max
    T_min = gel.T_min

    init_TRM = CoolingStep(0, T_max, T_min, TRM_mag, TRM_dir)
    oPrint_heat = HeatingStep(
        init_TRM.ts[-1] + 1e-12, T_min, oPrint_T, oPrint_mag, oPrint_dir
    )
    oPrint_hold = HoldStep(
        oPrint_heat.ts[-1] + 1e-12,
        oPrint_T,
        oPrint_mag,
        oPrint_dir,
        hold_steps=2,
        hold_time=oPrint_t,
    )
    oPrint_cool = CoolingStep(
        oPrint_hold.ts[-1] + 1e-12, oPrint_T, T_min, oPrint_mag, oPrint_dir
    )
    if oPrint_T == T_min:
        oPrint_cool.ts[0] = oPrint_hold.ts[-1] + 1e-12
        oPrint_cool.Ts[0] = T_min
    return [init_TRM, oPrint_heat, oPrint_hold, oPrint_cool]


def thermal_demag(Ts):
    NRM = HoldStep(0.0, 20.0, 0.0, np.array([1, 0, 0]), hold_steps=2, hold_time=100.0)
    steps = [NRM]
    for T in Ts:
        steps.append(
            HeatingStep(steps[-1].ts[-1] + 1e-12, 20.0, T, 0.0, np.array([1, 0, 0]))
        )
        steps.append(HoldStep(steps[-1].ts[-1] + 1e-12, T, 0.0, np.array([1, 0, 0])))
        steps.append(
            CoolingStep(
                steps[-1].ts[-1] + 1e-12,
                T,
                20.0,
                0.0,
                np.array([1, 0, 0]),
                max_temp=max(Ts),
            )
        )
    return steps
