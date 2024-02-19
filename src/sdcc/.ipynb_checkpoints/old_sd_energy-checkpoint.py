import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import jax.numpy as jnp
from jax import jit,vmap,grad,hessian,jacrev
from jax.tree_util import Partial
from jax.config import config
config.update("jax_enable_x64", True)
from skimage import measure
from pmagpy import pmag
from scipy.special import ellipkinc,ellipeinc
from itertools import combinations
from skimage.segmentation import watershed
from mpmath import mp,mpf
from skimage.feature import peak_local_max
import pickle
import multiprocessing as mpc
mp.prec=100

@jit
def angle2xyz(theta,phi):
    x=jnp.cos(theta)*jnp.cos(phi)
    y=jnp.sin(theta)*jnp.cos(phi)
    z=jnp.sin(phi)
    return jnp.array([x,y,z])

@jit
def xyz2angle(xyz):
    x=xyz[0]
    y=xyz[1]
    z=xyz[2]
    theta=jnp.arctan2(y,x)
    phi=jnp.arcsin(z)
    return(theta,phi)

def demag_factors(PRO,OBL):
    a=PRO
    b=1.
    c=1/OBL
    otheta=np.arccos(c/a)
    ophi=np.arccos(b/a)
    k=np.sin(ophi)/np.sin(otheta)
    alpha=np.arcsin(k)
    F=ellipkinc(otheta,k**2)
    E=ellipeinc(otheta,k**2)
    first_num=(c/a)*(b/a)
    sin_3_otheta=np.sin(otheta)**3
    cos_2_alpha=np.cos(alpha)**2
    ttp=np.sin(otheta)*np.cos(otheta)/np.cos(ophi)
    tpt=np.sin(otheta)*np.cos(ophi)/np.cos(otheta)
    if a==b==c:
        L=M=N=1/3
    elif b==c:
        L=first_num/(sin_3_otheta*k**2)*(F-E)
        M=N=(1-L)/2
    elif a==b:
        N=first_num/(sin_3_otheta*cos_2_alpha)*(tpt-E)
        L=M=(1-N)/2
    else:
        L=first_num/(sin_3_otheta*k**2)*(F-E)
        M=first_num/(sin_3_otheta*cos_2_alpha*k**2)*(E-cos_2_alpha*F-k**2*ttp)
        N=first_num/(sin_3_otheta*cos_2_alpha)*(tpt-E) 
    LMN=np.array([L,M,N])
    return(LMN)

@jit
def Ed(LMN,theta,phi,Ms):
    xyz=angle2xyz(theta,phi)*Ms
    N=jnp.eye(3)*LMN
    MN= jnp.dot(xyz,N)
    MNM=jnp.dot(MN,xyz.T)
    mu0=4*jnp.pi*1e-7
    return(0.5*mu0*MNM)

@jit
def Ea(k1,k2,theta,phi,rot_mat):
    xyz=angle2xyz(theta,phi)
    xyz=jnp.matmul(rot_mat,xyz)
    a1=xyz[0]
    a2=xyz[1]
    a3=xyz[2]
    return((k1*((a1*a2)**2+(a2*a3)**2+(a1*a3)**2)+k2*(a1*a2*a3)**2))

@jit
def energy_ang(angles,k1,k2,rot_mat,LMN,Ms):
    theta,phi=angles
    Ha=Ea(k1,k2,theta,phi,rot_mat)
    Hd=Ed(LMN,theta,phi,Ms)
    return(Ha+Hd)

@jit
def energy_xyz(xyz,k1,k2,rot_mat,LMN,Ms):
    theta,phi=xyz2angle(xyz/jnp.linalg.norm(xyz))
    Ha=Ea(k1,k2,theta,phi,rot_mat)
    Hd=Ed(LMN,theta,phi,Ms)
    return(jnp.nan_to_num(Ha+Hd,nan=jnp.inf))
    
def calculate_anisotropies(TMx):
    TMx/=100
    Tc = 3.7237e+02*TMx**3 - 6.9152e+02*TMx**2 - 4.1385e+02*TMx**1 + 5.8000e+02
    Tnorm = 20/Tc
    K1 = 1e4 * (-3.5725e+01*TMx**3 + 5.0920e+01*TMx**2 - 1.5257e+01*TMx**1 - 1.3579e+00) * (1-Tnorm)**(-6.3643e+00*TMx**2 + 2.3779e+00*TMx**1 + 3.0318e+00)
    K2 = 1e4 * (1.5308e+02*TMx**4 - 2.2600e+01*TMx**3 - 4.9734e+01*TMx**2 + 1.5822e+01*TMx**1 - 5.5522e-01) * (1-Tnorm)**7.2652e+00

    oneoneone=K1/3+K2/27
    oneonezero=K1/4
    onezerozero=0

    axes_names=np.array(['1 1 1', '1 1 0', '1 0 0'])
    axes_values=np.array([oneoneone,oneonezero,onezerozero])
    sorted_axes=axes_names[np.argsort(axes_values)]
    return(sorted_axes)

def dir_to_rot_mat(x,x_prime):
    a=np.array(x.split(' ')).astype(float)
    b=np.array(x_prime.split(' ')).astype(float)
    a/=np.linalg.norm(a)
    b/=np.linalg.norm(b)
    theta=np.arccos(np.dot(a,b))
    v=np.cross(a,b)
    euler_vector=v/np.linalg.norm(v)*theta
    rot=Rotation.from_rotvec(euler_vector)

    angles=rot.as_matrix()
    if np.any(np.isnan(angles)):
        angles=np.identity(3)
    return(angles)

def get_material_parms(TMx,alignment,T):
    if T<0:
        raise ValueError('Error: Temperature should be greater than 0 degrees')
    anis=calculate_anisotropies(TMx)
    rot_to='1 0 0'

    if alignment=='easy':
        rot_from=anis[0]
    elif alignment=='hard':
        rot_from=anis[2]


    rot_mat=dir_to_rot_mat(rot_from,rot_to)
    TMx/=100
    Tc = 3.7237e+02*TMx**3 - 6.9152e+02*TMx**2 - 4.1385e+02*TMx**1 + 5.8000e+02
    if T>=Tc:
        raise ValueError('Error: Temperature should not exceed Curie temperature (%1.0i'%Tc+'°C)')
    Tnorm = T/Tc
    K1 = 1e4 * (-3.5725e+01*TMx**3 + 5.0920e+01*TMx**2 - 1.5257e+01*TMx**1 - 1.3579e+00) * (1-Tnorm)**(-6.3643e+00*TMx**2 + 2.3779e+00*TMx**1 + 3.0318e+00)
    K2 = 1e4 * (1.5308e+02*TMx**4 - 2.2600e+01*TMx**3 - 4.9734e+01*TMx**2 + 1.5822e+01*TMx**1 - 5.5522e-01) * (1-Tnorm)**7.2652e+00
    Ms = (-2.8106e+05*TMx**3 + 5.2850e+05*TMx**2 - 7.9381e+05*TMx**1 + 4.9537e+05) * (1-Tnorm)**4.0025e-01
    return(rot_mat,K1,K2,Ms)

@Partial(jit,static_argnums=5)
def energy_surface(k1,k2,rot_mat,Ms,LMN,n_points=100,bounds=jnp.array([[0,2*jnp.pi],[-jnp.pi/2,jnp.pi/2]])):
    thetas=jnp.linspace(bounds[0,0],bounds[0,1],n_points)
    phis=jnp.linspace(bounds[1,0],bounds[1,1],n_points)
    thetas,phis=jnp.meshgrid(thetas,phis)
    energy_temp=lambda theta,phi: energy_ang([theta,phi],k1,k2,rot_mat,LMN,Ms)
    energy_temp=vmap(energy_temp)
    energy_array=energy_temp(thetas.flatten(),phis.flatten())
    energies=jnp.reshape(energy_array,thetas.shape)
    return(thetas,phis,energies)

def plot_net(ax=None):
    """
    Draws circle and tick marks for equal area projection.
    """
    if ax== None:
        fig,ax=plt.subplots()
        plt.clf()
    
    
    ax.axis("off")
    Dcirc = np.arange(0, 361.)
    Icirc = np.zeros(361, 'f')
    Xcirc, Ycirc = [], []
    for k in range(361):
        XY = pmag.dimap(Dcirc[k], Icirc[k])
        Xcirc.append(XY[0])
        Ycirc.append(XY[1])
    ax.plot(Xcirc, Ycirc, 'k')

    # put on the tick marks
    Xsym, Ysym = [], []
    for I in range(10, 100, 10):
        XY = pmag.dimap(0., I)
        Xsym.append(XY[0])
        Ysym.append(XY[1])
    ax.plot(Xsym, Ysym, 'k+')
    Xsym, Ysym = [], []
    for I in range(10, 90, 10):
        XY = pmag.dimap(90., I)
        Xsym.append(XY[0])
        Ysym.append(XY[1])
    ax.plot(Xsym, Ysym, 'k+')
    Xsym, Ysym = [], []
    for I in range(10, 90, 10):
        XY = pmag.dimap(180., I)
        Xsym.append(XY[0])
        Ysym.append(XY[1])
    ax.plot(Xsym, Ysym, 'k+')
    Xsym, Ysym = [], []
    for I in range(10, 90, 10):
        XY = pmag.dimap(270., I)
        Xsym.append(XY[0])
        Ysym.append(XY[1])
    ax.plot(Xsym, Ysym, 'k+')
    for D in range(0, 360, 10):
        Xtick, Ytick = [], []
        for I in range(4):
            XY = pmag.dimap(D, I)
            Xtick.append(XY[0])
            Ytick.append(XY[1])
        ax.plot(Xtick, Ytick, 'k')
    ax.axis("equal")
    ax.axis((-1.05, 1.05, -1.05, 1.05))

def plot_energy_surface(TMx,alignment,OBL,PRO,T=20,levels=10,n_points=100,projection='equirectangular'):
    LMN=demag_factors(PRO,OBL)
    rot_mat,k1,k2,Ms=get_material_parms(TMx,alignment,T)
    thetas,phis,energies=energy_surface(k1,k2,rot_mat,Ms,LMN,n_points=n_points)
    if 'equi' in projection.lower():
        fig=plt.figure()
        plt.contour(np.degrees(thetas),np.degrees(phis),energies,levels=levels,cmap='viridis',antialiased=True);
        plt.contourf(np.degrees(thetas),np.degrees(phis),energies,levels=levels,cmap='viridis',antialiased=True,linewidths=0.2);
        plt.colorbar(label='Energy Density (Jm$^{-3}$)')
        plt.xlabel(r'$\theta$',fontsize=14)
        plt.ylabel('$\phi$',fontsize=14)
        
    elif 'stereo' in projection.lower():
        fig,ax=plt.subplots(1,2,figsize=(9,4))
        plot_net(ax[0])
        vmin=np.amin(energies)
        vmax=np.amax(energies)
        xs,ys=pmag.dimap(np.degrees(thetas[phis>=0]).flatten(),np.degrees(phis[phis>=0]).flatten()).T
        upper=ax[0].tricontourf(xs,ys,energies[phis>=0].flatten(),vmin=vmin,vmax=vmax,levels=levels,antialiased=True)
        plot_net(ax[1])
        xs,ys=pmag.dimap(np.degrees(thetas[phis<=0]).flatten(),np.degrees(phis[phis<=0]).flatten()).T
        lower=ax[1].tricontourf(xs,ys,energies[phis<=0].flatten(),vmin=vmin,vmax=vmax,levels=levels,antialiased=True)
        cax = fig.add_axes([0.9, 0.05, 0.1, 0.9])
        cax.axis('Off')
        fig.colorbar(lower,ax=cax,label='Energy Density (Jm$^{-3}$)')
    
    else:
        raise KeyError('Unknown projection type: '+projection)
    fig.suptitle('SD Energy Surface TM'+str(TMx).zfill(2)+' AR %1.2f'%(PRO/OBL))    
    plt.tight_layout();
    
def find_global_minimum(thetas,phis,energies,mask=None):
    if np.all(mask==None):
        mask=np.full(energies.shape,True)
    masked_thetas=thetas[mask]
    masked_phis=phis[mask]
    masked_energies=energies[mask]
    best_energy=np.amin(masked_energies)
    best_theta=masked_thetas[masked_energies==best_energy][0]
    best_phi=masked_phis[masked_energies==best_energy][0]
    best_coords=np.where((thetas==best_theta)&(phis==best_phi))
    return (best_coords,best_theta,best_phi,best_energy)

def wrap_labels(labels):
    for l in jnp.unique(labels[:,0]):
        locs=jnp.where(labels[:,0] == l)
        wrapped_loc=jnp.unique(labels[locs,-1])
        for m in wrapped_loc:
            labels[labels==m] = l
    return(labels)


def segment_region(energies,mask):
    labels=measure.label(mask)
    labels=wrap_labels(labels)
    return(labels)

def get_minima(thetas,phis,energies,labels):
    theta_coords=[]
    phi_coords=[]
    temp_energies=[]

    for label in np.unique(labels):
        mask=labels==label
        temp_coords,temp_theta,temp_phi,temp_energy=find_global_minimum(thetas,phis,energies,mask)
        theta_coords.append(float(temp_theta))
        phi_coords.append(float(temp_phi))
        temp_energies.append(temp_energy)

    theta_coords=np.array(theta_coords)
    phi_coords=np.array(phi_coords)
    temp_energies=np.array(temp_energies)
    theta_coords=theta_coords[temp_energies!=max(temp_energies)]
    phi_coords=phi_coords[temp_energies!=max(temp_energies)]
    return(theta_coords,phi_coords)

def plot_contour_sweep(plot_dir,thetas,phis,energies,theta_coords,phi_coords,threshold,start_energy,final):
    if final:
        textcolor='r'
        contourcolor='r'
        idx=int(threshold-start_energy)+1
    else:
        textcolor='k'
        contourcolor='w'
        idx=int(threshold-start_energy)

    f=plt.figure()
    plt.contour(np.degrees(thetas),np.degrees(phis),energies,levels=10,cmap='viridis',antialiased=True);
    plt.contourf(np.degrees(thetas),np.degrees(phis),energies,levels=10,cmap='viridis',antialiased=True,linewidths=0.2);
    plt.colorbar(label='Energy Density (Jm$^{-3}$)')

    plt.xlabel(r'$\theta$',fontsize=14)
    plt.ylabel('$\phi$',fontsize=14)
    plt.plot(np.degrees(theta_coords),np.degrees(phi_coords),'r*',markersize=10,markeredgecolor='k')
    plt.contour(np.degrees(thetas),np.degrees(phis),energies,levels=[threshold],colors=contourcolor,linestyles='-');


    plt.annotate('Energy Barrier: '+str(int(threshold-start_energy))+' Jm$^{-3}$',(0.5,0.9),xycoords='axes fraction',color=textcolor)
    f.axes[1].axhline(threshold,color='w')
    plt.savefig(plot_dir+str(idx).zfill(4)+'.png',dpi=300)

    f.clear()
    f.clf()
    plt.close(f)


def contour_sweep(thetas,phis,energies,start_energy,incr,plot=False,plot_dir=None):

    #Get threshold value to compare to
    threshold=start_energy

    #Find everything that's below the threshold
    mask = energies <= threshold

    #Use segmentation algorithm to find number
    #Of minima
    labels = segment_region(energies,mask)

    #If number of minima differs, use segmentation
    old_len=len(np.unique(labels))
    
    #Get initial number of segmented parts
    theta_coords,phi_coords=get_minima(thetas,phis,energies,labels)
    new_len=old_len
    
    #Finished condition
    finished = False

    while not finished:
        #Sweep up by increasing threshold
        threshold+=incr

        #If we're outside of the range, we've somehow skipped our energy barrier, just end.
        if threshold<np.amin(energies):
            return(contour_sweep(thetas,phis,energies,start_energy,incr/10,plot=plot,plot_dir=plot_dir))

        #recalculate segmentation for new energy barriers
        mask = energies <= threshold
        labels = segment_region(energies,mask)
        new_len=len(np.unique(labels))

        #If things have changed, we need to update our minima
        if new_len!=old_len:
            theta_coords,phi_coords=get_minima(thetas,phis,energies,labels)

        #If plot, plot the contours
        if plot: 
            plot_contour_sweep(plot_dir,thetas,phis,energies,theta_coords,phi_coords,threshold,start_energy,final=False)
        #If we're sweeping up we're looking for a reduction in number of minima
        #If we're sweeping up we're looking for an increase.
        if np.sign(new_len-old_len)==-np.sign(incr):
            finished=True
        #Updar
        else:
            old_len=new_len
    if plot:
        plot_contour_sweep(plot_dir,thetas,phis,energies,theta_coords,phi_coords,threshold,start_energy,final=False)
    return(threshold)

def get_saddle_points(energies,saddle,incr):
    mask = energies <= saddle
    mask_last = energies <= saddle - incr
    masked_saddle = mask != mask_last
    return(masked_saddle)

@jit
def great_circle_dist(theta_a,theta_b,phi_a,phi_b):
    xyz_a=angle2xyz(theta_a,phi_a)
    xyz_b=angle2xyz(theta_b,phi_b)
    cos_alpha=jnp.dot(xyz_a,xyz_b)
    alpha=jnp.arccos(cos_alpha)
    return(jnp.degrees(alpha))

def check_min_dist(thetas,phis,energies,threshold,min_angle):
    #energies=np.pad(energies,((0,0),(20,20)),mode='wrap')
    #thetas=np.pad(thetas,((0,0),(20,20)),mode='wrap')
    #phis=np.pad(phis,((0,0),(20,20)),mode='wrap')
    mask = energies <= threshold
    labels = segment_region(energies,mask)
    theta_coords,phi_coords=get_minima(thetas,phis,energies,labels)

    for i,j in combinations(np.arange(len(theta_coords)),2):
        angle=great_circle_dist(theta_coords[i],theta_coords[j],phi_coords[i],phi_coords[j])
        if angle<min_angle:
            return(True)
        else:
            pass
    return(False)

def check_float_error(thetas,phis,energies,best_energy):
    floaterrors=np.append(0,np.logspace(-12,0,13))
    n_mins=[]
    for floaterror in np.append(0,np.logspace(-12,0,13)):
        mask = energies <= best_energy+floaterror
        labels = segment_region(energies,mask)

        n_min=len(np.unique(labels))
        n_mins.append(n_min)
    n_mins=np.array(n_mins)
    floaterror=floaterrors[n_mins==max(n_mins)][0]
    return(floaterror)


def find_energy_barrier(TM,alignment,PRO,OBL,n_sweeps,T=20,incr='auto',min_angle=0,plot=False,plot_dir=None):
    #Set up material parameters and demag factors, calculate energies
    rot_mat,k1,k2,Ms=get_material_parms(TM,alignment,T)
    LMN=demag_factors(PRO,OBL)
    thetas,phis,energies=energy_surface(k1,k2,rot_mat,Ms,LMN,n_points=1001)

    #Get global energy minimum
    best_coords,best_theta,best_phi,best_energy=find_global_minimum(thetas,phis,energies)

    #Find starting energy that contains all minima
    #Because we're on a grid, minima that should have same value might not
    #So we search over the next 1 Jm^-3 for the most minima.
    start_energy=best_energy+check_float_error(thetas,phis,energies,best_energy)

    #Automatic choice of increment for contour sweeping
    if incr=='auto':
        incr=(np.amax(energies)-np.amin(energies))/100

    #Calculate saddle point from first contour
    saddle=contour_sweep(thetas,phis,energies,start_energy,
                          incr,plot=plot,plot_dir=plot_dir)

    #Check that energy minima aren't really close to one another.
    min_check=check_min_dist(thetas,phis,energies,start_energy,min_angle)

    #If they are, keep sweeping up until you hit another energy barrier
    while min_check:
        saddle=contour_sweep(thetas,phis,energies,saddle,
                          incr,plot=plot,plot_dir=plot_dir)
        min_check=check_min_dist(thetas,phis,energies,saddle-incr,min_angle)

    #Now we've found our barrier, refine by iterating in to minimum value
    for i in range(n_sweeps-1):
        incr*=-0.1
        saddle=contour_sweep(thetas,phis,energies,saddle,
                              incr,plot=plot,plot_dir=plot_dir)

    #Get the closest points to the saddle points
    mask = get_saddle_points(energies,saddle,incr)

    #Calculate energy barrier
    barrier = saddle - best_energy
    return(barrier, energies, mask)


@jit
def update(xyz,k1,k2,rot_mat,LMN,Ms,lr):
    gradient=grad(energy_xyz)(xyz,k1,k2,rot_mat,LMN,Ms)
    delta_xyz = -lr*gradient
    return(xyz+delta_xyz)
    

def gradient_descent(max_iterations,threshold,xyz_init,k1,k2,rot_mat,LMN,Ms,learning_rate=1e-4):
    xyz = xyz_init
    xyz_history = xyz
    e_history = energy_xyz(xyz,k1,k2,rot_mat,LMN,Ms)
    delta_xyz = jnp.zeros(xyz.shape)
    i = 0
    diff = 1.0e10
    
    while  i<max_iterations and diff>threshold:
        xyz=update(xyz,k1,k2,rot_mat,LMN,Ms,learning_rate)
        xyz_history = jnp.vstack((xyz_history,xyz))
        e_history = jnp.vstack((e_history,energy_xyz(xyz,k1,k2,rot_mat,LMN,Ms)))
        i+=1
        diff = jnp.absolute(e_history[-1]-e_history[-2])

    return xyz_history,e_history

@jit
def _descent_update(i,j,energies):
    """Computes the minimum energy
    in a 10x10 grid around energies[i,j]"""
    i_temps=[]
    j_temps=[]
    energy_list=[]
    for i_temp in jnp.arange(-10,11)+i:
        for j_temp in jnp.arange(-10,11)+j:
            i_temps.append(jnp.clip(i_temp,0,1000))
            j_temps.append(j_temp%1000)
            energies_new=energies[jnp.clip(i_temp,0,1000),j_temp%1000]
            energies_orig=energies[i,j]
            energy_list.append(energies_new)
    i_temps=jnp.array(i_temps)
    j_temps=jnp.array(j_temps)
    energy_list=jnp.array(energy_list)
    i_new=i_temps[jnp.where(energy_list==jnp.min(energy_list),size=1)][0]
    j_new=j_temps[jnp.where(energy_list==jnp.min(energy_list),size=1)][0]
    return(i_new,j_new,energies[i_new,j_new])

def fast_path_to_min(energies,i,j):
    different=True
    i_history=[i]
    j_history=[j]
    e_history=[energies[i,j]]
    while different:
        i,j,e=_descent_update(i,j,energies)
        if e in e_history:
            different=False
        i_history.append(i)
        j_history.append(j)        
        e_history.append(e)
    return(i_history,j_history)

def plot_energy_barrier(TM,alignment,PRO,OBL,mask,T=20,n_perturbations=10,n_saddles=5,projection='equirectangular',method='fast',**kwargs):
    plot_energy_surface(TM,alignment,OBL,PRO,T=T,projection=projection,**kwargs)
    rot_mat,k1,k2,Ms=get_material_parms(TM,alignment,T)
    LMN=demag_factors(PRO,OBL)
    thetas,phis,energies=energy_surface(k1,k2,rot_mat,Ms,LMN,n_points=1001)    
    
    #If we have a mask which is at (phi = 90 or -90, set location to closest saddlepoint range
    mask_2=mask[((-1) & (energies[-2,:]==np.amin(energies[-2,:])))|((0) & (energies[1,:]==np.amin(energies[1,:]))),:]
    mask_2=mask_2.at[1:-1].set(True)
    mask=mask_2&mask
    
    saddle_thetas=thetas[mask]
    saddle_phis=phis[mask]
    cs=np.random.choice(len(saddle_thetas),min(len(saddle_thetas),n_saddles),replace=False)
    descent_thetas=[]
    descent_phis=[]
 
    #If we're using the slow, (gradient descent) method to find the path then use that routine
    if 'slow' in method.lower() or 'gradient' in method.lower() or 'descent' in method.lower():
        learning_rate=1/(np.amax(np.linalg.norm(np.gradient(energies),axis=0)))*1e-3
        for i in cs:
            for j in range(n_perturbations):
                start_theta=saddle_thetas[i]+np.random.normal(0,np.pi*2/1001)
                start_phi=saddle_phis[i]+np.random.normal(0,np.pi/2/1001)
                xyz=angle2xyz(start_theta,start_phi)
                result=gradient_descent(10000,1e-5,xyz,k1,k2,rot_mat,LMN,Ms,learning_rate=learning_rate)
                descent_theta=[np.degrees(saddle_thetas[i])]
                descent_phi=[np.degrees(saddle_phis[i])]
                for r in result[0]:
                    theta,phi=np.degrees(xyz2angle(r/np.linalg.norm(r)))
                    theta=theta%360
                    descent_theta.append(theta)
                    descent_phi.append(phi)
                descent_theta=np.array(descent_theta)
                descent_phi=np.array(descent_phi)
                descent_thetas.append(descent_theta)
                descent_phis.append(descent_phi)
    
    
    #If we're using the fast, (grid search) method to find the path, then use that routine.
    elif 'fast' in method.lower():
        jjs,iis=np.meshgrid(np.arange(1001),np.arange(1001))
        iis=iis[mask]
        jjs=jjs[mask]
        for c in cs:
            for ishift in range(-1,2):
                for jshift in range(-1,2):
                    i=np.clip(iis[c]+ishift,0,1000)
                    j=(jjs[c]+jshift)%1000
                    i_hist,j_hist=fast_path_to_min(energies,i,j)
                    descent_thetas.append(np.degrees(thetas[i_hist,j_hist]))
                    descent_phis.append(np.degrees(phis[i_hist,j_hist]))

    
    else:
        raise KeyError("method must contain one of the terms fast, slow, gradient, descent")
                    

    
    #Case statement to check for projection type
    
    if 'equi' in projection.lower():
        #Loop through runs
        for l in range(len(descent_thetas)):
            descent_theta = descent_thetas[l]
            descent_phi = descent_phis[l]
            
            #For equirectangular maps, each array must be split wjere
            #It crosses the 0/360 meridian to avoid drawing over
            #Center of map.
            splits=jnp.where(jnp.abs(jnp.diff(descent_theta))>=180)
            descent_theta=jnp.split(descent_theta,splits[0]+1)
            descent_phi=jnp.split(descent_phi,splits[0]+1)
            
            if len(splits)==0:
                descent_theta=[descent_theta]
                descent_phi=[descent_phi]
            for k in range(len(descent_theta)):
                plt.plot(descent_theta[k],descent_phi[k],'r',alpha=1)
        
        plt.plot(np.degrees(saddle_thetas[cs]),np.degrees(saddle_phis[cs]),'w.')
        
    

    elif 'stereo' in projection.lower():
        ax0=plt.gcf().axes[0]
        ax1=plt.gcf().axes[1]
        
        for l in range(len(descent_thetas)):
            descent_theta = descent_thetas[l]
            descent_phi = descent_phis[l]
            
            #Try except is here to catch scalar vs vector dimap output.
            try:
                thetas_plus,phis_plus=pmag.dimap(descent_theta[descent_phi>=0],descent_phi[descent_phi>=0]).T
            except:
                thetas_plus,phis_plus=pmag.dimap(descent_theta[descent_phi>=0],descent_phi[descent_phi>=0])
            
            ax0.plot(thetas_plus,phis_plus,'r',alpha=1)
            
            #Plot lower hemisphere on separate access
            try:
                thetas_minus,phis_minus=pmag.dimap(descent_theta[descent_phi<=0],descent_phi[descent_phi<=0]).T
            except:
                thetas_minus,phis_minus=pmag.dimap(descent_theta[descent_phi<=0],descent_phi[descent_phi<=0])

            ax1.plot(thetas_minus,phis_minus,'r',alpha=1)
        
        try:
            saddle_x,saddle_y=pmag.dimap(np.degrees(saddle_thetas[cs]),np.degrees(saddle_phis[cs])).T
        except:
            saddle_x,saddle_y=pmag.dimap(np.degrees(saddle_thetas[cs]),np.degrees(saddle_phis[cs]))

            
        ax0.plot(saddle_x[saddle_phis[cs]>=0],saddle_y[saddle_phis[cs]>=0],'w.')
        ax1.plot(saddle_x[saddle_phis[cs]<=0],saddle_y[saddle_phis[cs]<=0],'w.')

    else:
        raise KeyError('Unknown projection type: '+projection)

def critical_size(K):
    tau_0=1e-10
    t=100
    kb=1.380649e-23
    V=np.log(t/tau_0)*kb*293/K
    r=(V*3/(4*np.pi))**(1/3)
    d=2*r
    return(d*1e9)

def get_min_regions(energies,markers=None):
    """
    Finds regions separating the watershed o
    """

    tiled_energies=np.tile(energies,2) #tile energies to preserve connectivity relationship
    if type(markers)!=type(None):
        tiled_markers=np.tile(markers,2) #tile markers
        print(tiled_markers.shape)
    else:
        tiled_markers=None
    
    #take watershed of image    
    labels_old=watershed(tiled_energies,connectivity=2,markers=tiled_markers)
    #Take center part of map (wrapping not an issue)
    labels=labels_old[:,500:1501]

    rolled_energies=np.roll(energies,501,axis=1) #align energies with label map
    #Change things on the right edge to have the same label as things on the left edge
    for i in np.unique(labels[:,-1]):
        if len(labels[labels==i])>0:
            minimum=np.where((labels==i)&(rolled_energies==np.amin(rolled_energies[labels==i])))
        else:
            minimum=[[],[]]
        if 1000 in minimum[1]:
            minindex= np.where(minimum[1]==1000)
            j = labels[minimum[0][minindex],0]
            labels[labels==j]=i

    #Change things on the left edge to have the same label as things on the right edge
    for i in np.unique(labels[:,0]):
        if len(labels[labels==i])>0:
            minimum=np.where((labels==i)&(rolled_energies==np.amin(rolled_energies[labels==i])))
        else:
            minimum=[[],[]]
        if 0 in minimum[1]:
            minindex= np.where(minimum[1]==0)

            j = labels[minimum[0][minindex],-1]
            labels[labels==j]=i
    
    #align labels with energies
    labels=np.roll(labels,-501,axis=1)

    min_coords=[]
    for i in np.unique(labels):
        min_coords.append(np.where((labels==i)&(energies==np.amin(energies[labels==i]))))
    return(min_coords,labels)

def construct_energy_mat_fast(thetas,phis,energies,labels):
    labels_pad_v=np.pad(labels,((1,1),(0,0)),mode='edge')
    labels_pad_h=np.pad(labels,((0,0),(1,1)),mode='wrap')
    labels_pad=np.pad(labels_pad_v,((0,0),(1,1)),mode='wrap')

    pad_ul=labels_pad[:-2,:-2]
    pad_u=labels_pad_v[:-2,:]
    pad_ur=labels_pad[:-2,2:]
    pad_l=labels_pad_h[:,:-2]
    pad_r=labels_pad_h[:,2:]
    pad_bl=labels_pad[2:,:-2]
    pad_b=labels_pad_v[2:,:]
    pad_br=labels_pad[2:,2:]
    shifts=[pad_ul,pad_u,pad_ur,pad_l,pad_r,pad_bl,pad_b,pad_br]
    theta_list=[]
    phi_list=[]
    
    theta_mat=np.full((len(np.unique(labels)),len(np.unique(labels))),-np.inf)
    phi_mat=np.full((len(np.unique(labels)),len(np.unique(labels))),-np.inf)
    energy_mat=np.full((len(np.unique(labels)),len(np.unique(labels))),-np.inf)
    for i,j in combinations(range(len(np.unique(labels))),2):
        l=np.unique(labels)[i]
        m=np.unique(labels)[j]
        edge_filter=np.full(labels.shape,False)
        for shift in shifts:
            edge_filter=edge_filter|((labels==l)&(shift==m))
            edge_filter=edge_filter|((labels==m)&(shift==l))

        if len(energies[edge_filter])>0:
            min_energy=np.amin(energies[edge_filter])
            energy_mat[i,j]=min_energy-np.amin(energies[labels==l])
            energy_mat[j,i]=min_energy-np.amin(energies[labels==m])
            theta_mat[i,j]=thetas[(energies==min_energy)&(edge_filter)][0]
            phi_mat[i,j]=thetas[(energies==min_energy)&(edge_filter)][0]
            theta_mat[j,i]=theta_mat[i,j]
            phi_mat[j,i]=phi_mat[i,j]
    return(theta_mat,phi_mat,energy_mat)

def fix_minima(energies,max_minima):
    minima=np.array(peak_local_max(np.array(-energies[:,:-1]+np.amin(energies[:,:-1])),num_peaks=max_minima,exclude_border=False))
    markers=np.zeros(energies.shape)
    i=1
    for minimum in minima:
        markers[minimum[0],minimum[1]]=i
        if minimum[1]==0:
            markers[minimum[0],1000]=i
        if minimum[1]==1000:
            markers[minimum[0],0]=i
        i+=1
    return(markers)

def find_all_barriers(TMx,alignment,PRO,OBL,T=20):
    rot_mat,k1,k2,Ms=get_material_parms(TMx,alignment,T)
    LMN=demag_factors(PRO,OBL)
    easy_axis=calculate_anisotropies(TMx)[0]
    
    if easy_axis=='1 1 1':
        n_minima=8
    elif easy_axis=='1 0 0':
        n_minima=6
    elif easy_axis=='1 1 0':
        n_minima=12
    
    thetas,phis,energies=energy_surface(k1,k2,rot_mat,Ms,LMN,n_points=1001)
    theta_list=[]
    phi_list=[]
    min_energy_list=[]
    min_coords,labels=get_min_regions(energies)
    
    if len(min_coords)>n_minima:
        markers=fix_minima(energies,n_minima)
        min_coords,labels=get_min_regions(energies,markers=markers)
        
    for i in range(len(np.unique(labels))):
        where,theta,phi,energy=find_global_minimum(thetas,phis,energies,labels==np.unique(labels)[i])
        theta_list.append(theta)
        phi_list.append(phi)
        min_energy_list.append(energy)

    theta_mat,phi_mat,barriers=construct_energy_mat_fast(thetas,phis,energies,labels)
    
    return(np.array(theta_list),np.array(phi_list),np.array(min_energy_list),theta_mat,phi_mat,barriers)

def Q_matrix(theta_list,phi_list,theta_mat,phi_mat,energy_densities,T,d,Ms,field_dir=np.array([1,0,0]),field_str=0.):
    V=4/3*np.pi*((d/2*1e-9)**3)
    kb=1.380649e-23
    tau_0=1e-10
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
    #print(logQ)
    #plt.pcolormesh(logQ)
    #plt.colorbar()
    
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

def change_minima(p_old,old_theta_list,old_phi_list,new_theta_list,new_phi_list):
    old_len=len(old_theta_list)
    new_len=len(new_theta_list)
    
    cos_dist=np.empty((old_len,new_len))
    for i in range(old_len):
        for j in range(new_len):
            xyz_old=angle2xyz(old_theta_list[i],old_phi_list[i])
            xyz_new=angle2xyz(new_theta_list[j],new_phi_list[j])
            cos_dist[i,j]=np.dot(xyz_old,xyz_new)

    p_new=np.zeros(new_len)
    for j in range(len(p_new)):
        p_new[j]+=np.sum(p_old[np.where(cos_dist[:,j]==np.amax(cos_dist[:,j]))])
    p_new/=sum(p_new)
    return(p_new)

def fix_minima(energies,max_minima):
    minima=np.array(peak_local_max(np.array(-energies[:,:-1]+\
        np.amin(energies[:,:-1])),num_peaks=max_minima,exclude_border=False))
    markers=np.zeros(energies.shape)
    i=1
    for minimum in minima:
        markers[minimum[0],minimum[1]]=i
        if minimum[1]==0:
            markers[minimum[0],1000]=i
        i+=1
    return(markers)

def find_T_barriers(TMx,alignment,PRO,OBL,T_spacing=1):

    Tc = 3.7237e+02*(TMx/100)**3 - 6.9152e+02*(TMx/100)**2 \
        - 4.1385e+02*(TMx/100)**1 + 5.8000e+02
    
    theta_lists=[]
    phi_lists=[]
    min_energy_lists=[]
    theta_mats=[]
    phi_mats=[]
    energy_mats=[]
    Ts=np.arange(20,Tc,T_spacing)

    for T in Ts:
        print('Calculating energy barriers at '+str(int(T)).zfill(3)+\
            '°C, calculating up to '+str(int(Tc)).zfill(3)+'°C',end='\r')
        theta_list,phi_list,min_energy_list,theta_mat,phi_mat,energy_mat\
            =find_all_barriers(TMx,alignment,PRO,OBL,T=T)
        theta_lists.append(theta_list)
        phi_lists.append(phi_list)
        energy_mats.append(energy_mat)
        min_energy_lists.append(min_energy_list)
        theta_mats.append(theta_mat)
        phi_mats.append(phi_mat)
        
    return(theta_lists,phi_lists,min_energy_lists,theta_mats,phi_mats,\
        energy_mats,Ts)

class GrainEnergyLandscape():
    def __init__(self,TMx,alignment,PRO,OBL,T_spacing=1):
        theta_lists,phi_lists,min_energy_lists,theta_mats,phi_mats,energy_mats,\
            Ts=find_T_barriers(TMx,alignment,PRO,OBL,T_spacing=T_spacing)
        self.theta_lists=np.array(theta_lists)
        self.phi_lists=np.array(phi_lists)
        self.min_energies=np.array(min_energy_lists)
        self.theta_mats=np.array(theta_mats)
        self.phi_mats=np.array(phi_mats)
        self.energy_mats=np.array(energy_mats)
        self.Ts=Ts
        self.TMx=TMx
        self.alignment=alignment
        self.PRO=PRO
        self.OBL=OBL

    def to_file(self,fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)
    def __repr__(self):
        retstr="""Energy Landscape of TM{TMx} Grain with a prolateness of\
             {PRO} and an oblateness of {OBL} elongated along the\
                 magnetocrystalline {alignment} axis.""".format(
        TMx=self.TMx,PRO=self.PRO,OBL=self.OBL,alignment=self.alignment)
        return(retstr)

def thermal_treatment(start_t,start_p,Ts,ts,d,energy_landscape:GrainEnergyLandscape,field_strs,field_dirs,eq=False):
    old_T=Ts[0]
    old_min_energies=energy_landscape.min_energies[energy_landscape.Ts==old_T][0]
    old_thetas = energy_landscape.theta_lists[energy_landscape.Ts==old_T][0]
    old_phis = energy_landscape.phi_lists[energy_landscape.Ts==old_T][0]
    old_saddle_thetas=energy_landscape.theta_mats[energy_landscape.Ts==old_T][0]
    old_saddle_phis=energy_landscape.phi_mats[energy_landscape.Ts==old_T][0]
    old_mats = energy_landscape.energy_mats[energy_landscape.Ts==old_T][0]
    rot_mat,k1,k2,Ms=get_material_parms(energy_landscape.TMx,energy_landscape.alignment,old_T)
    
    if eq:
        old_p=eq_ps(old_thetas,old_phis,old_min_energies,field_strs[0],field_dirs[0],old_T,d,Ms)

    else:
        Q=Q_matrix(old_thetas,old_phis,old_saddle_thetas,old_saddle_phis,old_mats,old_T,d,Ms,field_dir=field_dirs[0],field_str=field_strs[0])
        old_p=_update_p_vector(start_p,Q,ts[0]-start_t)
    ps=[old_p]
    theta_lists=[old_thetas]
    phi_lists=[old_phis]
    for i in range(1,len(Ts)):
        T=Ts[i]
        dt=ts[i]-ts[i-1]
        new_thetas = energy_landscape.theta_lists[energy_landscape.Ts==Ts[i]][0]
        new_phis = energy_landscape.phi_lists[energy_landscape.Ts==Ts[i]][0]
        new_min_energies=energy_landscape.min_energies[energy_landscape.Ts==Ts[i]][0]
        new_saddle_thetas=energy_landscape.theta_mats[energy_landscape.Ts==Ts[i]][0]
        new_saddle_phis=energy_landscape.phi_mats[energy_landscape.Ts==Ts[i]][0]
        new_mats = energy_landscape.energy_mats[energy_landscape.Ts==Ts[i]][0]
        #print(old_p,old_thetas,old_phis,new_thetas,new_phis)
        
        old_p = change_minima(old_p,old_thetas,old_phis,new_thetas,new_phis)
        
        rot_mat,k1,k2,Ms=get_material_parms(energy_landscape.TMx,energy_landscape.alignment,T)
        
        if eq:
            new_p=eq_ps(new_thetas,new_phis,new_min_energies,field_strs[i],field_dirs[i],T,d,Ms)

        else:
            Q = Q_matrix(new_thetas,new_phis,new_saddle_thetas,new_saddle_phis,new_mats,T,d,Ms,field_dir=field_dirs[i],field_str=field_strs[i])
            new_p=_update_p_vector(old_p,Q,dt)
        
        ps.append(new_p)
        
        
        old_thetas=new_thetas
        old_phis=new_phis
        old_mats=new_mats
        old_p=new_p
        theta_lists.append(new_thetas)
        phi_lists.append(new_phis)
        
    return(ps,theta_lists,phi_lists)

def time2temp(t,t1,T0,T1,T_amb):
    T_range = T0 - T_amb
    t_rat = t / t1
    T_rat = (T1 - T_amb) / (T0 - T_amb)
    return(T_amb + T_range * np.exp(t_rat * np.log(T_rat)))

def temp2time(T,t1,T0,T1,T_amb):
    frac_T = (T - T_amb) / (T0 - T_amb)
    T_rat = (T1 - T_amb) / (T0 - T_amb)
    return (t1 * np.log(frac_T)/np.log(T_rat))

def get_avg_vectors(ps,theta_lists,phi_lists,Ts,rot_mat,energy_landscape:GrainEnergyLandscape,d):
    vs=[]
    inv_rot=np.linalg.inv(rot_mat)
    for i in range(len(ps)):
        V=4/3*np.pi*((d/2*1e-9)**3)
        T=Ts[i]
        rot_mat,k1,k2,Ms=get_material_parms(energy_landscape.TMx,energy_landscape.alignment,T)
        theta_list=theta_lists[i]
        phi_list=phi_lists[i]
        vecs=angle2xyz(theta_list,phi_list)*ps[i]*Ms*V
        v=np.sum(vecs,axis=1)
        vs.append(inv_rot@v)
    return(np.array(vs))

def grain_vectors(start_t,start_p,Ts,ts,d,energy_landscape:GrainEnergyLandscape,grain_dir,field_strs,field_dirs,eq=False):
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
    ps,theta_lists,phi_lists=thermal_treatment(start_t,start_p,Ts,ts,d,energy_landscape,field_strs,rot_field_dirs,eq=eq)
    vs=get_avg_vectors(ps,theta_lists,phi_lists,Ts,rot_mat,energy_landscape,d)
    return(vs,ps)

def mono_direction(grain_dir,start_p,d,steps,energy_landscape:GrainEnergyLandscape,eq=False):
    v_step=[]
    p_step=[]
    new_start_p=start_p
    new_start_t=0
    for step in steps:
        ts=step.ts
        Ts=step.Ts
        field_strs=step.field_strs
        field_dirs=step.field_dirs
        v,p=grain_vectors(new_start_t,new_start_p,Ts,ts,d,energy_landscape,grain_dir,field_strs,field_dirs,eq=eq)
        new_start_p=p[-1]
        new_start_t=ts[-1]
        v_step.append(v)
        p_step.append(p)
    return(v_step,p_step)

def parallelized_mono_dispersion(start_p,d,steps,energy_landscape:GrainEnergyLandscape,n_dirs=50,eq=False):
    dirs=fib_sphere(n_dirs)
    pool=mpc.Pool(mpc.cpu_count())
    objs=np.array([pool.apply_async(mono_direction, args=(grain_dir,start_p,d,steps,energy_landscape,eq)) for grain_dir in dirs])
    vps=np.array([obj.get() for obj in objs])
    pool.close()
    vs=vps[:,0]
    ps=vps[:,1]
    vs=np.sum(vs,axis=0)
    return(vs,ps)
    
def mono_dispersion(start_p,d,steps,energy_landscape:GrainEnergyLandscape,n_dirs=50,eq=False):
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
            v,p=grain_vectors(new_start_t,new_start_p,Ts,ts,d,energy_landscape,grain_dir,field_strs,field_dirs,eq=eq)
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

def eq_ps(theta_list,phi_list,min_energies,field_str,field_dir,T,d,Ms):
    V=4/3*np.pi*((d/2*1e-9)**3)
    kb=1.380649e-23
    min_energies=np.array(min_energies)
    xyz=angle2xyz(theta_list,phi_list)
    for i in range(len(theta_list)):
        zeeman_energy = np.dot(xyz[:,i],field_dir*field_str*1e-6) * Ms
        #print(zeeman_energy)
        min_energies[i] -= zeeman_energy

    precise_exp=np.vectorize(mp.exp)
    #print(-(min_energies * V) / (kb * (273 + T)))
    e_ratio = precise_exp(-(min_energies * V) / (kb * (273 + T)))
    #print(e_ratio)
    ps = e_ratio/sum(e_ratio)
    return(np.array(ps,dtype='float64'))

class ThermalStep:
    def __init__(self,t_start,T_start,T_end,field_str,field_dir,char_time=1,max_temp=None,char_temp=None,hold_steps=100,lin_rate=1/3,hold_time=1800,step_type='cooling'):
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
            raise KeyError("step_type must be one of 'cooling','heating', 'hold'")
        
        self.Ts=self.Ts.astype(int)
        self.ts+=t_start
        self.field_strs=np.full(len(self.Ts),field_str)
        self.field_dirs=np.repeat(np.array([field_dir]),len(self.Ts),axis=0)
        
    def __repr__(self):
        if self.step_type.lower()=='cooling':
            return(f'Cooling from {self.Ts[0]} to {self.Ts[-1]}°C in {self.field_strs[0]} μT field')
        elif self.step_type.lower()=='heating':
            return(f'Heating from {self.Ts[0]} to {self.Ts[-1]}°C in {self.field_strs[0]} μT field')
        elif self.step_type.lower()=='hold':
            return(f'Hold at {self.Ts[0]}°C in {self.field_strs[0]} μT field')
        elif self.step_type.lower()=='vrm':
            return(f'VRM acquisition at {self.Ts[0]}°C in {self.field_strs[0]} μT field')
        else:
            raise KeyError("step_type must be one of 'cooling','heating' or 'hold'")

def coe_experiment(temp_steps,B_anc,B_lab,B_ancdir,B_labdir):
    T_max=temp_steps[-1]
    T_min=temp_steps[0]
    steps=[]
    TRM = ThermalStep(0,T_max,T_min,B_anc,B_ancdir,step_type='cooling')
    steps.append(TRM)
    for j in range(1,len(temp_steps)):
        ZjW = ThermalStep(steps[-1].ts[-1]+1e-12,T_min,temp_steps[j],0,B_labdir,step_type='heating')
        steps.append(ZjW)
        ZjH = ThermalStep(steps[-1].ts[-1]+1e-12,temp_steps[j],temp_steps[j],0,B_labdir,step_type='hold')
        steps.append(ZjH)
        ZjC = ThermalStep(steps[-1].ts[-1]+1e-12,temp_steps[j],T_min,0,B_labdir,max_temp=T_max,char_temp=T_max-1,step_type='cooling')
        steps.append(ZjC)
        IjW = ThermalStep(steps[-1].ts[-1]+1e-12,T_min,temp_steps[j],B_lab,B_labdir,step_type='heating')
        steps.append(IjW)
        IjH = ThermalStep(steps[-1].ts[-1]+1e-12,temp_steps[j],temp_steps[j],B_lab,B_labdir,step_type='hold')
        steps.append(IjH)
        IjC = ThermalStep(steps[-1].ts[-1]+1e-12,temp_steps[j],T_min,B_lab,B_labdir,max_temp=T_max,char_temp=T_max-1,step_type='cooling')
        steps.append(IjC)
    return(steps)
def plot_routine(steps):
    fig,ax=plt.subplots(figsize=(12,4))
    plt.xlabel('t (hrs)')
    plt.ylabel('T ($^\circ$C)')
    for step in steps:
        if step.step_type=='cooling':
            c='b'
        elif step.step_type=='heating':
            c='r'
        else:
            c='purple'
        if step.field_strs[0]>0:
            ls='-'
        else:
            ls='--'
    
        plt.plot(step.ts/3600,step.Ts,color=c,linestyle=ls)

def relaxation_time(energy_landscape:GrainEnergyLandscape,B_dir,B):
    T_max=max(energy_landscape.Ts)
    T_min=min(energy_landscape.Ts)
    TRM=ThermalStep(0,T_max,T_min,B,B_dir,step_type='cooling')
    V_Rel=ThermalStep(TRM.ts[-1],T_min,T_min,0,B_dir,step_type='vrm',\
        hold_time=1e17,hold_steps=361)
    steps=[TRM,V_Rel]
    return(steps)

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

def relax_time_crit_size(relax_routine,energy_landscape,init_size=[5],size_incr=150):
    """
    Finds the critical size of relaxation.
    """
    n_states=len(energy_landscape.theta_lists[-1])
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
                relax_time=calc_relax_time(start_p,d,relax_routine,energy_landscape,ts)
                relax_times.append(relax_time)
                ds.append(d)
        else:
            print(f'Current Size {d} nm                 ')
            relax_time=calc_relax_time(start_p,d,relax_routine,energy_landscape,ts)
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
        print(f'Current Size {d} nm                 ')
        relax_time=calc_relax_time(start_p,d,relax_routine,energy_landscape,ts)
        relax_times.append(relax_times)
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
        
        print(f'Current Size {d} nm                 ')
        if d in ds:
            continuing=False
        else:
            relax_time = calc_relax_time(start_p,d,relax_routine,energy_landscape,ts)
            relax_times.append(relax_time)
            ds.append(d)
            
    return(d)

def full_crit_size(TMx,PRO,OBL,alignment):
    theta_list,phi_list,min_energy_list,theta_mat,phi_mat,barriers=find_all_barriers(TMx,alignment,PRO,OBL)
    if PRO==1.00 and OBL==1.00:
        do_full = False
    elif len(theta_list)==2:
        do_full = False
    else:
        do_full = True
    if do_full:
        Energy=GrainEnergyLandscape(TMx,alignment,PRO,OBL)
        relax_routine=relaxation_time(Energy,np.array([1,0,0]),40)
        barrierslist=[]
        for barrier in np.unique(np.floor(barriers[~np.isinf(barriers)]/1000)*1000):
            barrierslist.append(np.mean(barriers[(barriers>=barrier)&(barriers<barrier+1000)]))
        potential_ds=critical_size(np.array(barrierslist))
        d=relax_time_crit_size(relax_routine,Energy,init_size=potential_ds)
        return(d)
    else:
        return(np.nan)