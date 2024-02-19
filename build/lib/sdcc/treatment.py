import numpy as np
from sdcc.barriers import GrainEnergyLandscape,GEL

def time2temp(t,t1,T0,T1,T_amb):
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
    return(T_amb + T_range * np.exp(t_rat * np.log(T_rat)))

def temp2time(T,t1,T0,T1,T_amb):
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
    return (t1 * np.log(frac_T)/np.log(T_rat))

class ThermalStep:
    """
    Class for a thermal treatment step.
    Instead of having a lot of cases, I think it would make more sense
    to make each type of treatment step a separate class that inherits
    from Step - more flexible to create new stuff this way.
    """
    def __init__(self,t_start,T_start,T_end,field_str,field_dir,char_time=1,
    max_temp=None,char_temp=None,hold_steps=100,lin_rate=1/3,hold_time=1800,
    step_type='cooling'):

        self.step_type=step_type
        if char_temp==None or max_temp==None:
            char_temp=T_start-1
            max_temp=T_start
        if step_type.lower()=='cooling':
            Ts=np.arange(T_start,T_end-1,-1,dtype='float64')
            Ts[-1]=Ts[-1]+0.5
            self.Ts=Ts
            self.ts=temp2time(self.Ts,char_time,max_temp,char_temp,T_end)
            self.ts=self.ts-self.ts[0]
            
        elif step_type.lower()=='heating':
            self.Ts=np.arange(T_start,T_end+1)
            lin_time=(T_end-T_start)/lin_rate
            self.ts=np.linspace(0,lin_time,len(self.Ts))
            
        elif step_type.lower()=='vrm':
            self.Ts=np.full(hold_steps,T_start)
            self.ts=np.logspace(-1,np.log10(hold_time),len(self.Ts))
        
        elif step_type.lower()=='hold':
            self.Ts=np.full(hold_steps,T_start)
            self.ts=np.linspace(0,hold_time,len(self.Ts))
        
        else:
            raise KeyError(
                "step_type must be one of 'cooling','heating','hold'")
        
        self.Ts=self.Ts.astype(int)
        self.ts+=t_start
        self.field_strs=np.full(len(self.Ts),field_str)
        self.field_dirs=np.repeat(np.array([field_dir]),len(self.Ts),axis=0)
        
    def __repr__(self):
        if self.step_type.lower()=='cooling':
            return(f"""Cooling from {self.Ts[0]} to {self.Ts[-1]}°C
             in {self.field_strs[0]} μT field""")
        elif self.step_type.lower()=='heating':
            return(f"""Heating from {self.Ts[0]} to {self.Ts[-1]}°C
             in {self.field_strs[0]} μT field""")
        elif self.step_type.lower()=='hold':
            return(f"""Hold at {self.Ts[0]}°C
             in {self.field_strs[0]} μT field""")
        elif self.step_type.lower()=='vrm':
            return(f"""VRM acquisition at {self.Ts[0]}°C 
            in {self.field_strs[0]} μT field""")
        else:
            raise KeyError(
                "step_type must be one of 'cooling','heating' or 'hold'")

def coe_experiment(temp_steps,B_anc,B_lab,B_ancdir,B_labdir):
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
    T_max=temp_steps[-1]
    T_min=temp_steps[0]
    steps=[]
    TRM = ThermalStep(0,T_max,T_min,B_anc,B_ancdir,step_type='cooling')
    steps.append(TRM)
    for j in range(1,len(temp_steps)):
        ZjW = ThermalStep(steps[-1].ts[-1]+1e-12,T_min,temp_steps[j],0,B_labdir,
        step_type='heating')
        steps.append(ZjW)
        ZjH = ThermalStep(steps[-1].ts[-1]+1e-12,temp_steps[j],temp_steps[j],0,
        B_labdir,step_type='hold')
        steps.append(ZjH)
        ZjC = ThermalStep(steps[-1].ts[-1]+1e-12,temp_steps[j],T_min,0,B_labdir,
        max_temp=T_max,char_temp=T_max-1,step_type='cooling')
        steps.append(ZjC)
        IjW = ThermalStep(steps[-1].ts[-1]+1e-12,T_min,temp_steps[j],B_lab,
        B_labdir,step_type='heating')
        steps.append(IjW)
        IjH = ThermalStep(steps[-1].ts[-1]+1e-12,temp_steps[j],temp_steps[j],
        B_lab,B_labdir,step_type='hold')
        steps.append(IjH)
        IjC = ThermalStep(steps[-1].ts[-1]+1e-12,temp_steps[j],T_min,B_lab,
        B_labdir,max_temp=T_max,char_temp=T_max-1,step_type='cooling')
        steps.append(IjC)
    return(steps)

def relaxation_time(energy_landscape:GEL,B_dir,B):
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
    T_max=energy_landscape.T_max
    T_min=energy_landscape.T_min
    TRM=ThermalStep(0,T_max,T_min,B,B_dir,step_type='cooling')
    V_Rel=ThermalStep(TRM.ts[-1],T_min,T_min,0,B_dir,step_type='vrm',\
        hold_time=1e17,hold_steps=361)
    steps=[TRM,V_Rel]
    return(steps)

