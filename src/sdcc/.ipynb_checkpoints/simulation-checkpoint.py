import numpy as np
import mpmath as mp
import pickle
import multiprocessing as mpc
from sdcc.barriers import GrainEnergyLandscape,GEL,find_all_barriers
from sdcc.energy import angle2xyz,xyz2angle,dir_to_rot_mat,get_material_parms
mp.prec=100
from sdcc.treatment import relaxation_time

def change_minima(p_old,old_theta_list,old_phi_list,new_theta_list,
new_phi_list):
    """
    Calculates the set of energy minima at one temperature for a grain which
    correspond to the set of energy minima at another temperature.
    """
    old_len=len(old_theta_list) #number of old minima
    new_len=len(new_theta_list) #number of new minima
    
    #Calculate cosine difference between old and new minima
    cos_dist=np.empty((old_len,new_len))
    for i in range(old_len):
        for j in range(new_len):
            xyz_old=angle2xyz(old_theta_list[i],old_phi_list[i])
            xyz_new=angle2xyz(new_theta_list[j],new_phi_list[j])
            cos_dist[i,j]=np.dot(xyz_old,xyz_new)
    
    #Assign new p vector
    p_new=np.zeros(new_len)
    for j in range(len(p_new)):
        p_new[j]+=np.sum(p_old[np.where(cos_dist[:,j]==np.amax(cos_dist[:,j]))])
    p_new/=sum(p_new)
    return(p_new)

def Q_matrix(params:dict,d,field_dir=np.array([1,0,0]),field_str=0.):
    theta_list=params['min_dir'][:,0]
    phi_list=params['min_dir'][:,1]
    theta_mat=params['bar_dir'][:,:,0]
    phi_mat=params['bar_dir'][:,:,1]
    energy_densities=params['bar_e']
    T=params['T']
    Ms=params['Ms']
    
    V=4/3*np.pi*((d/2*1e-9)**3)
    kb=1.380649e-23
    tau_0=1e-9
    tt,pp=np.meshgrid(theta_list,phi_list)
    xyz=angle2xyz(tt,pp)
    xyz*=Ms*V
    xyz_T=angle2xyz(theta_mat,phi_mat)
    xyz_T*=Ms*V
    xyz=xyz_T-xyz

    
    field_dir=field_dir*field_str*1e-6
    
    field_mat=np.empty((3,len(theta_list),len(theta_list)))
    for i in range(len(theta_list)):
        for j in range(len(theta_list)):
            field_mat[:,i,j]=field_dir
    field_mat=np.array([field_mat[0].T,field_mat[1].T,field_mat[2].T])
    
    zeeman_energy=np.sum(xyz*field_mat,axis=0)
    zeeman_energy=zeeman_energy.at[np.isinf(phi_mat.T)].set(0.)
    logQ=-(energy_densities.T*V-zeeman_energy)/(kb*(273+T))

    logQ=np.array(logQ)
    logQ[np.isnan(logQ)]=-mp.inf
    logQ[np.isinf(logQ)&(logQ>0)]=-mp.inf
    precise_exp=np.vectorize(mp.exp)
    Q=precise_exp(logQ)
    Q/=mp.mpmathify(tau_0)
    
    #print(Q)
    for i in range(len(theta_list)):
        Q[i,i]=0.
        Q[i,i]=-mp.fsum(Q[:,i])
    return(Q)

def Q_matrix_legacy(theta_list,phi_list,theta_mat,phi_mat,energy_densities,T,d,Ms,
field_dir=np.array([1,0,0]),field_str=0.):
    V=4/3*np.pi*((d/2*1e-9)**3)
    kb=1.380649e-23
    tau_0=1e-9
    tt,pp=np.meshgrid(theta_list,phi_list)
    xyz=angle2xyz(tt,pp)
    xyz*=Ms*V
    xyz_T=angle2xyz(theta_mat,phi_mat)
    xyz_T*=Ms*V
    xyz=xyz_T-xyz

    
    field_dir=field_dir*field_str*1e-6
    
    field_mat=np.empty((3,len(theta_list),len(theta_list)))
    for i in range(len(theta_list)):
        for j in range(len(theta_list)):
            field_mat[:,i,j]=field_dir
    field_mat=np.array([field_mat[0].T,field_mat[1].T,field_mat[2].T])
    
    zeeman_energy=np.sum(xyz*field_mat,axis=0)
    logQ=-(energy_densities.T*V-zeeman_energy)/(kb*(273+T))
    
    logQ=np.array(logQ)
    logQ[np.isnan(logQ)]=-mp.inf
    logQ[np.isinf(logQ)]=-mp.inf
    precise_exp=np.vectorize(mp.exp)
    Q=precise_exp(logQ)
    Q/=mp.mpmathify(tau_0)
    
    #print(Q)
    for i in range(len(theta_list)):
        Q[i,i]=-np.sum(Q[:,i])
    return(Q)

def _update_p_vector(p_vec,Q,dt):
    dp_dt = mp.expm(mp.matrix(Q) * mp.mpmathify(dt))
    p_vec_new=dp_dt * mp.matrix(p_vec)
    p_vec_new=np.array(p_vec_new,dtype='float64')
    p_vec_new/=sum(p_vec_new)
    return(p_vec_new)

def thermal_treatment_legacy(start_t,start_p,Ts,ts,d,
energy_landscape:GrainEnergyLandscape,field_strs,field_dirs,eq=False):
    old_T=Ts[0]
    old_min_energies=\
        energy_landscape.min_energies[energy_landscape.Ts==old_T][0]
    old_thetas = energy_landscape.theta_lists[energy_landscape.Ts==old_T][0]
    old_phis = energy_landscape.phi_lists[energy_landscape.Ts==old_T][0]
    old_saddle_thetas=energy_landscape.theta_mats[energy_landscape.Ts==old_T][0]
    old_saddle_phis=energy_landscape.phi_mats[energy_landscape.Ts==old_T][0]
    old_mats = energy_landscape.energy_mats[energy_landscape.Ts==old_T][0]
    rot_mat,k1,k2,Ms=get_material_parms(energy_landscape.TMx,
    energy_landscape.alignment,old_T)
    
    if eq:
        old_p=eq_ps(old_thetas,old_phis,old_min_energies,field_strs[0],
        field_dirs[0],old_T,d,Ms)

    else:
        
        Q=Q_matrix(old_thetas,old_phis,old_saddle_thetas,old_saddle_phis,
        old_mats,old_T,d,Ms,field_dir=field_dirs[0],field_str=field_strs[0])
        old_p=_update_p_vector(start_p,Q,ts[0]-start_t)
    ps=[old_p]
    theta_lists=[old_thetas]
    phi_lists=[old_phis]
    for i in range(1,len(Ts)):
        T=Ts[i]
        dt=ts[i]-ts[i-1]
        new_thetas = energy_landscape.theta_lists[energy_landscape.Ts==Ts[i]][0]
        new_phis = energy_landscape.phi_lists[energy_landscape.Ts==Ts[i]][0]
        new_min_energies=\
            energy_landscape.min_energies[energy_landscape.Ts==Ts[i]][0]
        new_saddle_thetas=\
            energy_landscape.theta_mats[energy_landscape.Ts==Ts[i]][0]
        new_saddle_phis=energy_landscape.phi_mats[energy_landscape.Ts==Ts[i]][0]
        new_mats = energy_landscape.energy_mats[energy_landscape.Ts==Ts[i]][0]
        
        old_p = change_minima(old_p,old_thetas,old_phis,new_thetas,new_phis)
        
        rot_mat,k1,k2,Ms=get_material_parms(energy_landscape.TMx,
        energy_landscape.alignment,T)
        
        if eq:
            new_p=eq_ps(new_thetas,new_phis,new_min_energies,field_strs[i],
            field_dirs[i],T,d,Ms)

        else:
            Q = Q_matrix(new_thetas,new_phis,new_saddle_thetas,new_saddle_phis,
            new_mats,T,d,Ms,field_dir=field_dirs[i],field_str=field_strs[i])
            new_p=_update_p_vector(old_p,Q,dt)
        
        ps.append(new_p)
        
        
        old_thetas=new_thetas
        old_phis=new_phis
        old_mats=new_mats
        old_p=new_p
        theta_lists.append(new_thetas)
        phi_lists.append(new_phis)
        
    return(ps,theta_lists,phi_lists)

def thermal_treatment(start_t,start_p,Ts,ts,d,energy_landscape:GEL,field_strs,field_dirs,eq=False):
    old_T=Ts[0]
    params=energy_landscape.get_params(old_T)
    
    
    if eq:
        old_p=eq_ps(old_thetas,old_phis,old_min_energies,field_strs[0],
        field_dirs[0],old_T,d,Ms)

    else:
        Q=Q_matrix(params,d,field_dir=field_dirs[0],field_str=field_strs[0])
        old_p=_update_p_vector(start_p,Q,ts[0]-start_t)
    ps=[old_p]
    theta_lists=[params['min_dir'][:,0]]
    phi_lists=[params['min_dir'][:,1]]
    for i in range(1,len(Ts)):
        T=Ts[i]
        dt=ts[i]-ts[i-1]
        params=energy_landscape.get_params(T)
        
        if eq:
            new_p=eq_ps(barriers,field_strs[i],
            field_dirs[i],T,d,Ms)

        else:
            Q = Q_matrix(params,d,field_dir=field_dirs[i],field_str=field_strs[i])
            new_p=_update_p_vector(ps[-1],Q,dt)

        ps.append(new_p)
        theta_list=params['min_dir'][:,0]
        phi_list=params['min_dir'][:,1]
        theta_lists.append(theta_list)
        phi_lists.append(phi_list)
        
    return(ps,theta_lists,phi_lists)

def get_avg_vectors(ps,theta_lists,phi_lists,Ts,rot_mat,
energy_landscape:GEL,d):
    vs=[]
    inv_rot=np.linalg.inv(rot_mat)
    for i in range(len(ps)):
        V=4/3*np.pi*((d/2*1e-9)**3)
        T=Ts[i]
        rot_mat,k1,k2,Ms=get_material_parms(energy_landscape.TMx,
        energy_landscape.alignment,T)
        theta_list=theta_lists[i]
        phi_list=phi_lists[i]
        vecs=angle2xyz(theta_list,phi_list)*ps[i]*Ms*V
        v=np.sum(vecs,axis=1)
        vs.append(inv_rot@v)
    return(np.array(vs))

def grain_vectors(start_t,start_p,Ts,ts,d,
energy_landscape:GEL,grain_dir,field_strs,field_dirs,eq=False):
    grain_dirstr=grain_dir.astype(str)
    grain_dirstr=' '.join(grain_dirstr)
    ref_dir=np.array([1,0,0])
    ref_dirstr=ref_dir.astype(str)
    ref_dirstr=' '.join(ref_dirstr)
    rot_mat=dir_to_rot_mat(ref_dirstr,grain_dirstr)
    rot_field_dirs=[]
    for f in field_dirs:
        rot_dir=rot_mat@f
        rot_field_dirs.append(rot_dir)
    ps,theta_lists,phi_lists=thermal_treatment(start_t,start_p,Ts,ts,d,
    energy_landscape,field_strs,rot_field_dirs,eq=eq)
    vs=get_avg_vectors(ps,theta_lists,phi_lists,Ts,rot_mat,energy_landscape,d)
    return(vs,ps)

def mono_direction(grain_dir,start_p,d,steps,
energy_landscape:GEL,eq=False):
    v_step=[]
    p_step=[]
    new_start_p=start_p
    new_start_t=0
    for step in steps:
        ts=step.ts
        Ts=step.Ts
        field_strs=step.field_strs
        field_dirs=step.field_dirs
        v,p=grain_vectors(new_start_t,new_start_p,Ts,ts,d,energy_landscape,
        grain_dir,field_strs,field_dirs,eq=eq)
        new_start_p=p[-1]
        new_start_t=ts[-1]
        v_step.append(v)
        p_step.append(p)
    return(v_step,p_step)

def mono_dispersion(start_p,d,steps,energy_landscape:GEL,
n_dirs=50,eq=False):
    dirs=fib_sphere(n_dirs)
    vs=[]
    ps=[]
    i=0
    for grain_dir in dirs:
        i+=1
        print('Working on grain {i} of {n}'.format(i=i,n=n_dirs),end='\r')
        v_step=[]
        p_step=[]
        new_start_p=start_p
        new_start_t=0
        for step in steps:
            ts=step.ts
            Ts=step.Ts
            field_strs=step.field_strs
            field_dirs=step.field_dirs
            v,p=grain_vectors(new_start_t,new_start_p,Ts,ts,d,
            energy_landscape,grain_dir,field_strs,field_dirs,eq=eq)
            new_start_p=p[-1]
            new_start_t=ts[-1]
            v_step.append(v)
            p_step.append(p)
        
        vs.append(v_step)
        ps.append(p_step)
    vs=np.array(vs)
    ps=np.array(ps)
    vs=np.sum(vs,axis=0)

    return(vs,ps)

def parallelized_mono_dispersion(start_p,d,steps,
energy_landscape:GEL,n_dirs=50,eq=False):
    dirs=fib_sphere(n_dirs)
    pool=mpc.Pool(mpc.cpu_count())
    objs=np.array([pool.apply_async(mono_direction, 
    args=(grain_dir,start_p,d,steps,energy_landscape,eq)) for 
    grain_dir in dirs])
    vps=np.array([obj.get() for obj in objs],dtype='object')
    pool.close()
    vs=vps[:,0]
    ps=vps[:,1]
    vs=np.sum(vs,axis=0)
    return(vs,ps)
    
def fib_sphere(n=1000):
    goldenRatio = (1 + 5**0.5)/2
    i = np.arange(0, n)
    theta = 2 *np.pi * i / goldenRatio
    phi = np.arccos(1 - 2*(i+0.5)/n)
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi);
    return(np.array([x,y,z]).T)

class SDCCResult:
    def __init__(self,sizes,thermal_steps,vs,ps):
        self.sizes=sizes
        self.thermal_steps=thermal_steps
        self.vs=vs
        self.ps=ps
    def to_file(self,fname):     
        with open(fname, 'wb') as f:
            pickle.dump(self, f,-1)
            
def eq_ps(params,field_str,field_dir,T,d,Ms):
    phi_list=params['min_dir'][:,1]
    theta_mat=params['bar_dir'][:,:,0]
    phi_mat=params['bar_dir'][:,:,1]
    energy_densities=params['bar_e']
    
    V=4/3*np.pi*((d/2*1e-9)**3)
    kb=1.380649e-23
    min_energies=np.array(min_energies)
    xyz=angle2xyz(theta_list,phi_list)
    for i in range(len(theta_list)):
        zeeman_energy = np.dot(xyz[:,i],field_dir*field_str*1e-6) * Ms
        min_energies[i] -= zeeman_energy

    precise_exp=np.vectorize(mp.exp)
    e_ratio = precise_exp(-(min_energies * V) / (kb * (273 + T)))
    ps = e_ratio/sum(e_ratio)
    return(np.array(ps,dtype='float64'))


def eq_ps_legacy(theta_list,phi_list,min_energies,field_str,field_dir,T,d,Ms):
    V=4/3*np.pi*((d/2*1e-9)**3)
    kb=1.380649e-23
    min_energies=np.array(min_energies)
    xyz=angle2xyz(theta_list,phi_list)
    for i in range(len(theta_list)):
        zeeman_energy = np.dot(xyz[:,i],field_dir*field_str*1e-6) * Ms
        min_energies[i] -= zeeman_energy

    precise_exp=np.vectorize(mp.exp)
    e_ratio = precise_exp(-(min_energies * V) / (kb * (273 + T)))
    ps = e_ratio/sum(e_ratio)
    return(np.array(ps,dtype='float64'))

def calc_relax_time(start_p,d,relax_routine,energy_landscape,ts):
    #Run a parallelized mono dispersion
    vs,ps=parallelized_mono_dispersion(start_p,d,relax_routine,energy_landscape)
    #Calculate magnitude of vector
    mags=np.linalg.norm(vs[1],axis=1)
    #Calculate TRMs
    TRM=np.linalg.norm(vs[0][-1])
    #Get relaxation time (M = TRM/e)
    try:
        relax_time=ts[mags<=(TRM/np.e)][0]
    except:
        relax_time=ts[-1]
    return(relax_time)

def relax_time_crit_size(relax_routine,energy_landscape,init_size=[5],
size_incr=150):
    """
    Finds the critical size of relaxation.
    """
    n_states=len(energy_landscape.get_params(energy_landscape.T_max)['min_e'])
    start_p=np.full(n_states,1/n_states)
    ts=(relax_routine[1].ts-relax_routine[0].ts[-1])
    
    relax_times=[]
    ds=[]
    
    #Run through all the possible relaxation times
    #From energy barriers calculated
    for d in np.ceil(init_size).astype(int):
        if (d!=np.ceil(init_size[0]).astype(int)):
            if ((min(relax_times)<100)&(max(relax_times)>=100)):
                pass
            else:
                print(f'Current Size {d} nm                 ')
                relax_time=calc_relax_time(start_p,d,relax_routine,
                energy_landscape,ts)
                print('Relaxation time %1.1e'%relax_time)
                relax_times.append(relax_time)
                ds.append(d)
        else:
            print(f'Current Size {d} nm                 ')
            relax_time=calc_relax_time(start_p,d,relax_routine,
            energy_landscape,ts)
            print('Relaxation time %1.1e'%relax_time)
            relax_times.append(relax_time)
            ds.append(d)
    
    #If relaxation times don't span
    #the necessary range, step up until they do
    
    #What stopping condition do we use for stepping?
    if np.amax(relax_times)<100:
        statefun = lambda r: np.amax(r)<100
        state=statefun(relax_times)
    elif np.amin(relax_times)>=100:
        statefun = lambda r: np.amin(r)>=100
        state=statefun(relax_times)
    else:
        state=False
    
    #What direction do we step in?
    if relax_time<100:
        sign=1
    else:
        sign=-1

    #Step upwards.
    while state:
        d+=sign*int(size_incr)
        if d<=0:
            d=1
        print(f'Current Size {d} nm                 ')
        relax_time=calc_relax_time(start_p,d,relax_routine,energy_landscape,ts)
        print('Relaxation time %1.1e'%relax_time)
        relax_times.append(relax_time)
        ds.append(d)
        state=statefun(relax_times)
            
    
    #Now we use a bisection method
    #to find the critical size,
    #followed by a root-finding method
    #when close enough to resolve the 
    #relaxation time
    continuing=True
    while continuing:
        d_sorted=np.array(np.sort(ds))
        r_sorted=np.array(relax_times)[np.argsort(ds)]

        d_min=d_sorted[r_sorted<100][-1]
        d_max=d_sorted[r_sorted>=100][0]
        r_min=r_sorted[r_sorted<100][-1]
        r_max=r_sorted[r_sorted>=100][0]
        

        
        if r_max==ts[-1] or r_min==ts[0]:
            d = int(np.ceil((d_min+d_max)/2))
    
        else:
            d=int(np.ceil(np.interp(2,np.log10(r_sorted),d_sorted)))
        
        print(f'Current Size {d} nm                ')
        if d in ds and (len(ds)>2):
            continuing=False
        else:
            relax_time = calc_relax_time(start_p,d,relax_routine,
            energy_landscape,ts)
            print('Relaxation time %1.1e'%relax_time)
            relax_times.append(relax_time)
            ds.append(d)
            
    return(d)

def critical_size(K):
    tau_0=1e-9
    t=100
    kb=1.380649e-23
    V=np.log(t/tau_0)*kb*293/K
    r=(V*3/(4*np.pi))**(1/3)
    d=2*r
    return(d*1e9)

def full_crit_size(TMx,PRO,OBL,alignment):
    theta_list,phi_list,min_energy_list,theta_mat,phi_mat,barriers=find_all_barriers(TMx,alignment,PRO,OBL)
    if PRO==1.00 and OBL==1.00:
        do_full = False
    elif len(theta_list)==2:
        do_full = False
    else:
        do_full = True
    if do_full:
        Energy=GEL(TMx,alignment,PRO,OBL)
        relax_routine=relaxation_time(Energy,np.array([1,0,0]),40)
        barrierslist=[]
        for barrier in np.unique(
        np.floor(barriers[~np.isinf(barriers)]/1000)*1000):
            barrierslist.append(
                np.mean(barriers[(barriers>=barrier)&(barriers<barrier+1000)]))
        potential_ds=critical_size(np.array(barrierslist))
        d=relax_time_crit_size(relax_routine,Energy,init_size=potential_ds)
        return(d)
    else:
        d=critical_size(np.array(barriers)[0,1])
        return(d)
    
