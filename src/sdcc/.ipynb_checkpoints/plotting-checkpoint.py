import numpy as np
import matplotlib.pyplot as plt
from pmagpy import pmag
from sdcc.energy import demag_factors,get_material_parms,energy_surface,\
    angle2xyz,energy_xyz,xyz2angle
from sdcc.treatment import ThermalStep
from jax import jit,grad
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)


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

def plot_energy_surface(TMx,alignment,PRO,OBL,T=20,ext_field=jnp.array([0,0,0]),
levels=10,n_points=100,projection='equirectangular'):
    LMN=demag_factors(PRO,OBL)
    rot_mat,k1,k2,Ms=get_material_parms(TMx,alignment,T)
    thetas,phis,energies=energy_surface(k1,k2,rot_mat,Ms,LMN,ext_field=ext_field,n_points=n_points)
    if 'equi' in projection.lower():
        fig=plt.figure()
        plt.contour(np.degrees(thetas),np.degrees(phis),energies,
        levels=levels,cmap='viridis',antialiased=True);
        plt.contourf(np.degrees(thetas),np.degrees(phis),energies,
        levels=levels,cmap='viridis',antialiased=True,linewidths=0.2);
        plt.colorbar(label='Energy Density (Jm$^{-3}$)')
        plt.xlabel(r'$\theta$',fontsize=14)
        plt.ylabel('$\phi$',fontsize=14)
        
    elif 'stereo' in projection.lower():
        fig,ax=plt.subplots(1,2,figsize=(9,4))
        plot_net(ax[0])
        vmin=np.amin(energies)
        vmax=np.amax(energies)
        xs,ys=pmag.dimap(np.degrees(thetas[phis>=0]).flatten(),
        np.degrees(phis[phis>=0]).flatten()).T
        upper=ax[0].tricontourf(xs,ys,energies[phis>=0].flatten(),
        vmin=vmin,vmax=vmax,levels=levels,antialiased=True)
        plot_net(ax[1])
        xs,ys=pmag.dimap(np.degrees(thetas[phis<=0]).flatten(),
        np.degrees(phis[phis<=0]).flatten()).T
        lower=ax[1].tricontourf(xs,ys,energies[phis<=0].flatten(),
        vmin=vmin,vmax=vmax,levels=levels,antialiased=True)
        cax = fig.add_axes([0.9, 0.05, 0.1, 0.9])
        cax.axis('Off')
        fig.colorbar(lower,ax=cax,label='Energy Density (Jm$^{-3}$)')
    
    else:
        raise KeyError('Unknown projection type: '+projection)
    fig.suptitle('SD Energy Surface TM'+str(TMx).zfill(2)+' AR %1.2f'%(PRO/OBL))    
    plt.tight_layout();

@jit
def update(xyz,k1,k2,rot_mat,LMN,Ms,ext_field,lr):
    gradient=grad(energy_xyz)(xyz,k1,k2,rot_mat,LMN,Ms,ext_field)
    delta_xyz = -lr*gradient
    return(xyz+delta_xyz)

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

def gradient_descent(max_iterations,threshold,xyz_init,k1,k2,rot_mat,LMN,Ms,
ext_field=np.array([0,0,0]),learning_rate=1e-4):
    xyz = xyz_init
    xyz_history = xyz
    e_history = energy_xyz(xyz,k1,k2,rot_mat,LMN,Ms,ext_field)
    delta_xyz = jnp.zeros(xyz.shape)
    i = 0
    diff = 1.0e10
    
    while  i<max_iterations and diff>threshold:
        xyz=update(xyz,k1,k2,rot_mat,LMN,Ms,ext_field,learning_rate)
        xyz_history = jnp.vstack((xyz_history,xyz))
        e_history = jnp.vstack((e_history,energy_xyz(xyz,k1,k2,rot_mat,LMN,Ms,
        ext_field)))
        i+=1
        diff = jnp.absolute(e_history[-1]-e_history[-2])

    return xyz_history,e_history

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

def plot_energy_path(TM,alignment,PRO,OBL,mask,T=20,ext_field=np.array([0,0,0]),
n_perturbations=10,n_saddles=5,projection='equirectangular',method='fast',
**kwargs):
    plot_energy_surface(TM,alignment,PRO,OBL,T=T,ext_field=ext_field,
    projection=projection,**kwargs)
    rot_mat,k1,k2,Ms=get_material_parms(TM,alignment,T)
    LMN=demag_factors(PRO,OBL)
    thetas,phis,energies=energy_surface(k1,k2,rot_mat,Ms,LMN,ext_field,
    n_points=1001)    
    
    #If we have a mask which is at (phi = 90 or -90, set location to closest 
    # saddlepoint range
    mask_2=mask[((-1) & (energies[-2,:]==np.amin(energies[-2,:])))|((0)\
         & (energies[1,:]==np.amin(energies[1,:]))),:]
    mask_2=mask_2.at[1:-1].set(True)
    mask=mask_2&mask
    
    saddle_thetas=thetas[mask]
    saddle_phis=phis[mask]
    cs=np.random.choice(len(saddle_thetas),min(len(saddle_thetas),n_saddles),
    replace=False)
    descent_thetas=[]
    descent_phis=[]
 
    #If we're using the slow, (gradient descent) method to find the path 
    #then use that routine
    if 'slow' in method.lower() or 'gradient' in method.lower() \
        or 'descent' in method.lower():
        learning_rate=1/(np.amax(np.linalg.norm(
            np.gradient(energies),axis=0)))*1e-3
        for i in cs:
            for j in range(n_perturbations):
                start_theta=saddle_thetas[i]
                start_phi=saddle_phis[i]
                xyz=angle2xyz(start_theta,start_phi)
                xyz+=np.random.normal(0,5e-3,xyz.shape)
                xyz/=np.linalg.norm(xyz)
                result=gradient_descent(10000,1e-5,xyz,k1,k2,rot_mat,LMN,Ms,
                ext_field=ext_field,learning_rate=learning_rate)
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
    
    
    #If we're using the fast, (grid search) method to find the path,
    #then use that routine.
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
        raise KeyError(
        "method must contain one of the terms fast, slow, gradient, descent")
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
                thetas_plus,phis_plus=pmag.dimap(descent_theta[descent_phi>=0],
                descent_phi[descent_phi>=0]).T
            except:
                if len(descent_theta)>1:
                    thetas_plus,phis_plus\
                        =pmag.dimap(descent_theta[descent_phi>=0],
                        descent_phi[descent_phi>=0])
                elif len(descent_theta)==1 and descent_phi>=0:
                    thetas_plus,phis_plus\
                        =pmag.dimap(descent_theta,descent_phi)
            
            ax0.plot(thetas_plus,phis_plus,'r',alpha=1)
            
            #Plot lower hemisphere on separate access
            try:
                thetas_minus,phis_minus\
                    =pmag.dimap(descent_theta[descent_phi<=0],
                    descent_phi[descent_phi<=0]).T
            except:
                if len(descent_theta)>1:
                    thetas_minus,phis_minus\
                        =pmag.dimap(descent_theta[descent_phi<=0],
                        descent_phi[descent_phi<=0])
                elif len(descent_theta)==1 and descent_phi<=0:
                    thetas_minus,phis_minus\
                        =pmag.dimap(descent_theta,descent_phi)

            ax1.plot(thetas_minus,phis_minus,'r',alpha=1)
        
        try:
            saddle_x,saddle_y=pmag.dimap(np.degrees(saddle_thetas[cs]),
            np.degrees(saddle_phis[cs])).T
        except:
            saddle_x,saddle_y=pmag.dimap(np.degrees(saddle_thetas[cs]),
            np.degrees(saddle_phis[cs]))

        if len(saddle_phis)>1:   
            ax0.plot(saddle_x[saddle_phis[cs]>=0],saddle_y[saddle_phis[cs]>=0],'w.')
            ax1.plot(saddle_x[saddle_phis[cs]<=0],saddle_y[saddle_phis[cs]<=0],'w.')
        elif len(saddle_phis)==1 and saddle_phis>=0:
            ax0.plot(saddle_x,saddle_y,'w.')
        else:
            ax1.plot(saddle_x,saddle_y,'w.')

    else:
        raise KeyError('Unknown projection type: '+projection)

def plot_minima(minima_thetas,minima_phis,projection='equirectangular'):
    ax=plt.gcf().get_axes()
    for i in range(len(minima_thetas)):
        if 'equi' in projection.lower():
            plt.text(np.degrees(minima_thetas[i]),np.degrees(minima_phis[i]),i,
            color='w',va='center',ha='center')
        elif 'stereo' in projection.lower():
            if minima_phis[i]<=0:
                theta,phi=pmag.dimap(np.degrees(minima_thetas[i]),
                                     -np.degrees(minima_phis[i]))
            
                ax[1].text(theta,phi,i,color='w',va='center',ha='center')
            if minima_phis[i]>=0:
                theta,phi=pmag.dimap(np.degrees(minima_thetas[i]),
                                     np.degrees(minima_phis[i]))
                ax[0].text(theta,phi,i,color='w',va='center',ha='center')


def plot_barriers(barrier_thetas,barrier_phis,projection='equirectangular',ax=None):
    if ax==None:
        ax=plt.gcf().get_axes()
    for i in range(len(barrier_thetas)):
        for j in range(i,len(barrier_thetas)):
            if not np.isinf(barrier_thetas[i,j]):
                if 'equi' in projection.lower():
                    ax[0].text(np.degrees(barrier_thetas[i,j]),
                    np.degrees(barrier_phis[i,j]),str(i)+','+str(j),
                    color='r',va='center',ha='center')
                elif 'stereo' in projection.lower():
                    if barrier_phis[i,j]<=0:
                        theta,phi\
                            =pmag.dimap(np.degrees(barrier_thetas[i,j]),
                                         -np.degrees(barrier_phis[i,j]))
                        ax[1].text(theta,phi,str(i)+','+str(j),color='r',
                                   va='center',ha='center')
                    if barrier_phis[i,j]>=0:
                        theta,phi\
                            =pmag.dimap(np.degrees(barrier_thetas[i,j]),
                                         np.degrees(barrier_phis[i,j]))
                        ax[0].text(theta,phi,str(i)+','+str(j),color='r',
                                   va='center',ha='center')

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

def energy_matrix_plot(barriers):
    plt.imshow(barriers,cmap='magma',vmin=0)
    ticks=[]
    for lim in np.arange(0,np.amax(barriers[~np.isinf(barriers)]),1000):
        relevant_barriers=barriers[(barriers>=lim)&(barriers<=lim+1000)]
        if len(relevant_barriers)>0:
            ticks.append(int(np.mean(relevant_barriers)))
    plt.colorbar(label = 'Energy Barrier (Jm$^{-3}$)',ticks=ticks)
    plt.xlabel('Minima Direction j')
    plt.ylabel('Minima Direction i')
    plt.xticks(np.arange(0,len(barriers)))
    plt.yticks(np.arange(0,len(barriers)))