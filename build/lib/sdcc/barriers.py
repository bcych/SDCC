### The following module calculates energy barriers between local energy minima
### In a single domain surface. 

### IMPORT STATEMENTS ###
import numpy as np
from jax import jit
import jax.numpy as jnp
from sdcc.energy import demag_factors,get_material_parms,energy_surface,\
    calculate_anisotropies,angle2xyz,xyz2angle
from jax.config import config
config.update("jax_enable_x64", True)
from skimage import measure
from itertools import combinations
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import pickle
from scipy.interpolate import splprep,splrep,BSpline

### GENERIC FUNCTIONS ###
### These functions are used by multiple energy barrier calculation
### methods. They are handy for calculating the minimum energy barriers

def find_global_minimum(thetas,phis,energies,mask=None):
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
    for l in jnp.unique(labels[:,0]):
        locs=jnp.where(labels[:,0] == l)
        wrapped_loc=jnp.unique(labels[locs,-1])
        for m in wrapped_loc:
            labels[labels==m] = l
    return(labels)

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
    labels=measure.label(mask)
    labels=wrap_labels(labels)
    return(labels)

def get_minima(thetas,phis,energies,labels):
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
    theta_coords=[]
    phi_coords=[]
    temp_energies=[]

    for label in np.unique(labels):
        mask= labels==label
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

### HANDWRITTEN WATERSHED ALGORITHM ###
### These functions use a handwritten version of the watershed algorithm to
### calculate the MINIMUM energy barrier from the LOWEST energy state on the SD
### energy surface - this is not recommended as these do not calculate all 
### barriers for a grain - which can be non recommended for describing single
### domain behaviour. They are here for code preservation purposes.

def plot_contour_sweep(plot_dir,thetas,phis,energies,theta_coords,phi_coords,
threshold,start_energy,final):
    """
    Plots a contour sweep - Deprecated but no equivalent in new
    functions!
    """
    if final:
        textcolor='r'
        contourcolor='r'
        idx=int(threshold-start_energy)+1
    else:
        textcolor='k'
        contourcolor='w'
        idx=int(threshold-start_energy)

    plt.figure()
    plt.contour(np.degrees(thetas),np.degrees(phis),energies,levels=10,cmap='viridis',antialiased=True)
    plt.contourf(np.degrees(thetas),np.degrees(phis),energies,levels=10,cmap='viridis',antialiased=True,linewidths=0.2)
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
    """
    Sweeps a contour through the energies to find connected regions
    DEPRECATED
    """
    #Get threshold value to compare to
    threshold=start_energy

    #Find everything that's below the threshold
    mask = energies <= threshold

    #Use segmentation algorithm to find number
    #Of minima
    labels = segment_region(mask)

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
        labels = segment_region(mask)
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
    """
    DEPRECATED
    Finds the saddle points (i.e. barrier locations)
    between regions
    """
    mask = energies <= saddle
    mask_last = energies <= saddle - incr
    masked_saddle = mask != mask_last
    return(masked_saddle)

@jit
def great_circle_dist(theta_a,theta_b,phi_a,phi_b):
    """
    DEPRECATED
    Great circle distance function
    """
    xyz_a=angle2xyz(theta_a,phi_a)
    xyz_b=angle2xyz(theta_b,phi_b)
    cos_alpha=jnp.dot(xyz_a,xyz_b)
    alpha=jnp.arccos(cos_alpha)
    return(jnp.degrees(alpha))

def check_min_dist(thetas,phis,energies,threshold,min_angle):
    """
    DEPRECATED
    Checks the minimum distance between minima
    """
    #energies=np.pad(energies,((0,0),(20,20)),mode='wrap')
    #thetas=np.pad(thetas,((0,0),(20,20)),mode='wrap')
    #phis=np.pad(phis,((0,0),(20,20)),mode='wrap')
    mask = energies <= threshold
    labels = segment_region(mask)
    theta_coords,phi_coords=get_minima(thetas,phis,energies,labels)

    for i,j in combinations(np.arange(len(theta_coords)),2):
        angle=great_circle_dist(theta_coords[i],theta_coords[j],phi_coords[i],phi_coords[j])
        if angle<min_angle:
            return(True)
        else:
            pass
    return(False)

def check_float_error(energies,best_energy):
    """
    DEPRECATED
    test tolerances for floating point errors
    """
    floaterrors=np.append(0,np.logspace(-12,0,13))
    n_mins=[]
    for floaterror in np.append(0,np.logspace(-12,0,13)):
        mask = energies <= best_energy+floaterror
        labels = segment_region(mask)

        n_min=len(np.unique(labels))
        n_mins.append(n_min)
    n_mins=np.array(n_mins)
    floaterror=floaterrors[n_mins==max(n_mins)][0]
    return(floaterror)

def find_energy_barrier(TM,alignment,PRO,OBL,n_sweeps,
T=20,ext_field=np.array([0,0,0]),incr='auto',min_angle=0,plot=False,plot_dir=None):
    """
    DEPRECATED
    Manual watershed algorithm for finding energy barriers
    """

    #Set up material parameters and demag factors, calculate energies
    rot_mat,k1,k2,Ms=get_material_parms(TM,alignment,T)
    LMN=demag_factors(PRO,OBL)
    thetas,phis,energies=energy_surface(k1,k2,rot_mat,Ms,LMN,ext_field,n_points=1001)

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

### SKIMAGE WATERSHED IMPLEMENTATION ###
### These functions calculate ALL energy barriers for a specimen using the 
### skimage implementation of the watershed algorithm. This is far faster and
### more efficient than the handwritten version and can quickly be used to 
### compute all the energy barriers for a given SD energy surface. It's rather
### quick to do this. 

def dijkstras(i,j,barriers):
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
    visited=np.full(len(barriers),False)
    distances=barriers[i]
    distances[np.isinf(distances)]=np.inf
    visited[i]=True
    while visited[j] is False:
        temp_distances=np.copy(distances)
        temp_distances[visited==True]=np.inf

        k = np.where((temp_distances==np.min(temp_distances)))[0]
        if isinstance(k,float):
            k=k[0]
        new_distances = np.copy(barriers[k])
        new_distances[visited]=np.inf
        new_distances[np.isinf(new_distances)]=np.inf
        new_distances[(new_distances<distances[k])]=distances[k]
        distances=np.amin(np.array([distances,new_distances]),axis=0)
        visited[k]=True
    if distances[j]*1.001<barriers[i,j]:
        barriers[i,j]=np.inf
    return barriers

def prune_energies(barriers,thetas,phis):
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
    for i in range(barriers.shape[0]):
        for j in range(barriers.shape[1]):
            if not np.isinf(barriers[i,j]) and j!=i:
                temp_barriers=dijkstras(i,j,barriers)
    thetas[np.isinf(temp_barriers)]=-np.inf
    phis[np.isinf(temp_barriers)]=-np.inf
    return(temp_barriers,thetas,phis)

def get_min_regions(energies,markers=None):
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
    tiled_energies=np.tile(energies,2)

    #If using markers, tile those.
    if type(markers)!=type(None):
        tiled_markers=np.tile(markers,2) #tile markers

    else:
        tiled_markers=None

    #take watershed of image
    labels_old=watershed(tiled_energies,connectivity=2,markers=tiled_markers)

    #Take center part of map (this is done so that wrapping is not an issue)
    labels=labels_old[:,500:1501]

    rolled_energies=np.roll(energies,501,axis=1) #align energies with label
    #map

    #Change things on the right edge to have the same label as things on left
    for i in np.unique(labels[:,-1]):
        #Adjust minima accordingly
        if len(labels[labels==i])>0:
            minimum=np.where((labels==i)&(
                rolled_energies==np.amin(rolled_energies[labels==i])))
        else:
            minimum=[[],[]]
        #Special case for if minimum on the boundary
        if 1000 in minimum[1]:
            minindex= np.where(minimum[1]==1000)
            j = labels[minimum[0][minindex],0]
            labels[labels==j]=i

    #Change things on the left edge to have the same label as things on right
    for i in np.unique(labels[:,0]):
        if len(labels[labels==i])>0:
            minimum=np.where((labels==i)&(
                rolled_energies==np.amin(rolled_energies[labels==i])))
        else:
            minimum=[[],[]]
        if 0 in minimum[1]:
            minindex= np.where(minimum[1]==0)
            j = labels[minimum[0][minindex],-1]
            labels[labels==j]=i

    #align labels with energies
    labels=np.roll(labels,-501,axis=1)

    #Create an array of minima coordinates.
    min_coords=[]
    for i in np.unique(labels):
        where = np.where((labels==i)&(energies==np.amin(energies[labels==i])))
        if len(where[0])>1:
            where=(np.array([where[0][0]]),np.array([where[1][0]]))
        min_coords.append(where)
    return(min_coords,labels)

def construct_energy_mat_fast(thetas,phis,energies,labels):
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
    labels_pad_v=np.pad(labels,((1,1),(0,0)),mode='edge')
    labels_pad_h=np.pad(labels,((0,0),(1,1)),mode='wrap')
    labels_pad=np.pad(labels_pad_v,((0,0),(1,1)),mode='wrap')

    #Instead of looping through every neighbor for each pixel in the energy
    #matrix, simply calculate the difference from shifted pixels in the array

    pad_ul=labels_pad[:-2,:-2]
    pad_u=labels_pad_v[:-2,:]
    pad_ur=labels_pad[:-2,2:]
    pad_l=labels_pad_h[:,:-2]
    pad_r=labels_pad_h[:,2:]
    pad_bl=labels_pad[2:,:-2]
    pad_b=labels_pad_v[2:,:]
    pad_br=labels_pad[2:,2:]

    #List the shifts and loop through them.
    shifts=[pad_ul,pad_u,pad_ur,pad_l,pad_r,pad_bl,pad_b,pad_br]
    
    theta_mat=np.full((len(np.unique(labels)),len(np.unique(labels))),-np.inf)
    phi_mat=np.full((len(np.unique(labels)),len(np.unique(labels))),-np.inf)
    energy_mat=np.full((len(np.unique(labels)),len(np.unique(labels))),-np.inf)
    
    #Loop through the combinations of i and j
    for i,j in combinations(range(len(np.unique(labels))),2):
        l=np.unique(labels)[i]
        m=np.unique(labels)[j]
        edge_filter=np.full(labels.shape,False)

        #loop through the shifts and find the edges
        for shift in shifts:
            edge_filter=edge_filter|((labels==l)&(shift==m))
            edge_filter=edge_filter|((labels==m)&(shift==l))

        #Get the MINIMUM energy and its location along the edge
        #This is the energy barrier!
        if len(energies[edge_filter])>0:
            min_energy=np.amin(energies[edge_filter])
            energy_mat[i,j]=min_energy-np.amin(energies[labels==l])
            energy_mat[j,i]=min_energy-np.amin(energies[labels==m])
            theta_mat[i,j]=thetas[(energies==min_energy)&(edge_filter)][0]
            phi_mat[i,j]=phis[(energies==min_energy)&(edge_filter)][0]
            theta_mat[j,i]=theta_mat[i,j]
            phi_mat[j,i]=phi_mat[i,j]

    return(theta_mat,phi_mat,energy_mat)

def fix_minima(min_coords,energies,max_minima):
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

    #First work out the energies
    min_coords=np.array(min_coords).T[0].T
    min_energies=[]
    for i in range(len(min_coords)):
        min_energy = energies[min_coords[i,0],min_coords[i,1]]
        min_energies.append(min_energy)
    min_energies=np.array(min_energies)

    #Keep removing the largest energies until you end up with
    #max_minima
    while len(min_coords)>max_minima:
        #First check how many LEMs we're dropping
        max_filter = min_energies==max(min_energies)
        max_len = len(min_energies[max_filter])

        #If that's too many, drop them one by one
        if len(min_coords) - max_len > max_minima:
            where = np.where(max_filter)[0][0]
            max_filter = range(len(min_coords)) == where

        #min filter is the inverse of max_filter
        min_filter = ~max_filter

        min_coords = min_coords[min_filter]
        min_energies = min_energies[min_filter]

    #Construct array of new markers for watershed.
    new_markers=np.zeros(energies.shape)
    i=0
    for new_coord in min_coords:
        i+=1
        new_markers[new_coord[0],new_coord[1]]=i
        if new_coord[1]==0:
            new_markers[minimum[0],1000]=i
        if new_coord[1]==1000:
            new_markers[minimum[0],0]=i
    return(new_markers)

def merge_similar_minima(min_coords,thetas,phis,energies,tol):
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

    #Find your thetas phis, energies associated with your minima
    min_coords_new=np.array(min_coords).T[0]
    theta_list = thetas[min_coords_new[0],min_coords_new[1]]
    phi_list = phis[min_coords_new[0],min_coords_new[1]]
    energy_list = energies[min_coords_new[0],min_coords_new[1]]

    #Find angular differences
    xyz=angle2xyz(theta_list,phi_list)
    diffs=np.dot(xyz.T,xyz)
    cos_lim = np.cos(np.radians(tol)) #how small do we want to go
    dis,djs =np.where(diffs>=cos_lim)

    #Obviously the angular distance to same state is the same so remove
    #these.
    dis_u = dis[dis!=djs]
    djs_u = djs[dis!=djs]

    #Get the unique states that might need replacing
    udis = np.unique(dis_u)

    #Loop through and find groups of states
    complete = np.array([])
    groups = []
    for i in udis:
        #If we already have grouped this ignore it
        #N.B. what if there's two minima two degrees apart each, this
        #Would only count the two with closest indices - hopefully ok
        if i in complete:
            pass
        #Otherwise find everything that's got that index paired with it
        else:
            group=np.intersect1d(djs[dis == i],dis[djs == i])
            complete=np.append(complete,group)
            groups.append(group)

    dropped = np.array([])
    #Loop through groups and find which minima to ignore "drop"
    for group in groups:
        group_energies = energy_list[group]
        drop = group[group_energies!=min(group_energies)] #Keep maximum
        while len(drop) < len(group) - 1: #If you removed too few
            #Kick another out!
            drop = np.append(drop,group[~np.isin(group,drop)][0])
        dropped = np.append(dropped,drop)

    #Delete the stuff in dropped from min_coords_new
    min_coords_new = min_coords_new[:,~np.isin(range(len(theta_list)),dropped)]

    #Make the markers from our new min_coords
    new_markers=np.zeros(energies.shape)
    i=0
    for new_coord in min_coords_new.T:
        i+=1
        new_markers[new_coord[0],new_coord[1]]=i
        #Loop condition
        if new_coord[1]==0:
            new_markers[new_coord[0],1000]=i
        if new_coord[1]==1000:
            new_markers[new_coord[0],0]=i

    #Remake min_coords in correct format!
    min_coords=[]
    for i in np.unique(new_markers):
        if i!=0:
            where=np.where((new_markers==i)&(energies==np.amin(energies[new_markers==i])))
            if len(where[0])>1:
                where=(np.array([where[0][0]]),np.array([where[1][0]]))
            min_coords.append(where)
    return(min_coords,new_markers)



def find_all_barriers(TMx,alignment,PRO,OBL,T=20,ext_field=np.array([0,0,0]),
prune=True,trim=True,tol=2.):
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

    #Set up parameters
    rot_mat,k1,k2,Ms=get_material_parms(TMx,alignment,T)
    LMN=demag_factors(PRO,OBL)
    easy_axis=calculate_anisotropies(TMx)[0]

    #Get a number of minima associated with the magnetocrystalline
    #directions (unused)
    if easy_axis=='1 1 1':
        n_minima=8
    elif easy_axis=='1 0 0':
        n_minima=6
    elif easy_axis=='1 1 0':
        n_minima=12
    else:
        raise ValueError('Something wrong with anisotropy field')
    #Get the energy surface
    thetas,phis,energies=energy_surface(k1,k2,rot_mat,Ms,LMN,ext_field,n_points=1001)

    #Get the drainage divides of the LEM states
    theta_list=[]
    phi_list=[]
    min_energy_list=[]
    min_coords,labels=get_min_regions(energies)

    #Run a second pass to eliminate similar minima
    min_coords_new,markers_new=merge_similar_minima(min_coords,thetas,phis,energies,tol)
    n_markers_old = len(np.unique(labels))
    n_markers_new = len(np.unique(markers_new)) - 1

    #Eliminate minima which have more than magnetocrystalline directions
    if len(min_coords_new)>n_minima and trim:
        markers_new=fix_minima(min_coords_new,energies,n_minima)

    if n_markers_old != n_markers_new or (len(min_coords_new)>n_minima and trim):
        min_coords,labels=get_min_regions(energies,markers=markers_new)

    #Find the global minimum for each LEM state
    #This might be superfluous as it's already done in get_min_regions
    #But oh well.
    for i in range(len(np.unique(labels))):
        where,theta,phi,energy=find_global_minimum(thetas,phis,energies,
        labels==np.unique(labels)[i])
        theta_list.append(theta)
        phi_list.append(phi)
        min_energy_list.append(energy)

    #Get energy barriers associated with minimum regions.
    theta_mat,phi_mat,barriers=construct_energy_mat_fast(thetas,phis,energies,labels)

    #Prune states where it's easier to reach from another state always.
    if prune:
        barriers,theta_mat,phi_mat = prune_energies(barriers,theta_mat,phi_mat)

    #Convert to arrays
    theta_list=np.array(theta_list)
    phi_list=np.array(phi_list)
    min_energy_list=np.array(min_energy_list)

    #Check for weird erroneous states with entirely infinite barriers
    #to/from and remove.
    #This is yet another weird edge case that we have to catch
    bad_filter=(np.all(np.isinf(barriers),axis=1)|np.all(np.isinf(barriers),axis=0))

    theta_list=theta_list[~bad_filter]
    phi_list=phi_list[~bad_filter]
    min_energy_list=min_energy_list[~bad_filter]

    #Yuck! Must be a better way to do this.
    dels=np.arange(len(bad_filter))[bad_filter]
    theta_mat=np.delete(theta_mat,dels,axis=0)
    theta_mat=np.delete(theta_mat,dels,axis=1)
    phi_mat=np.delete(phi_mat,dels,axis=0)
    phi_mat=np.delete(phi_mat,dels,axis=1)
    barriers=np.delete(barriers,dels,axis=0)
    barriers=np.delete(barriers,dels,axis=1)

    return(theta_list,phi_list,min_energy_list,theta_mat,phi_mat,barriers)

def mat_to_mask(theta_mat,phi_mat):
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

    thetas = np.linspace(0,2*np.pi,1001)
    phis = np.linspace(-np.pi/2,np.pi/2,1001)
    thetas,phis = np.meshgrid(thetas,phis)
    theta_mat_new = np.copy(theta_mat)
    phi_mat_new = np.copy(phi_mat)
    
    #Only want to record barriers twice
    for i in range(len(theta_mat)-1):
        for j in range(i+1,len(theta_mat)):
            theta_mat_new[i,j]= -np.inf
            phi_mat_new[i,j]= -np.inf
    
    theta_list = theta_mat_new[~np.isinf(theta_mat)]
    phi_list = phi_mat_new[~np.isinf(phi_mat)]
    mask = np.full(thetas.shape,False)
    thetas = np.round(thetas,4)
    phis = np.round(phis,4)
    theta_list = np.round(theta_list,4)
    phi_list = np.round(phi_list,4)
    for theta,phi in zip(theta_list,phi_list):
        mask = mask | ((thetas == theta) & (phis == phi))
    return(jnp.array(mask))

def find_T_barriers(TMx,alignment,PRO,OBL,T_spacing=1):
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

### STORING RESULTS ###
class GrainEnergyLandscape():
    """
    DEPRECATED
    Class for storing energy barrier results at all temperatures for a
    given grain geometry and composition.
    """
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
    
def merge_barriers(tlists,plists,mlists,tmats,pmats,barriers,n,minstep,maxstep):
    """
    This function is disgusting. There must be a better way to write this
    surely? Tracks energy barriers across different temperatures and
    determines which barriers are the same when the number of barriers
    changes. I would like this to be less than 170 lines ideally.

    Inputs
    ------
    tlists, plists: numpy arrays
    Arrays of the minimum energy directions in the SD energy surface

    mlists: numpy array
    Minimum energy at these directions

    tmats,pmats: numpy arrays
    Arrays of the directions associated with the energy barriers.

    barriers: numpy array
    Matrices of energy barriers between state i and state j.

    n: int
    Index corresponding to temperature at which barriers change

    minstep: int
    Index correspondng to starting temperature (i.e. last
    dimension change)

    maxstep: int
    Index corresponding to final temperature (i.e. next
    dimension change)
    
    Returns
    -------
    tlists, plists: numpy arrays
    Consistemt length arrays of minimum energy directions in SD
    energy surface

    mlists: numpy array
    Consistent length arrays of minimum energies in SD energy
    surface

    tmats,pmats: numpy arrays
    Consistent size matrix of the directions associated with the
    energy barriers

    barriers: numpy array
    Consistent size matrix of energy barriers between states i and
    j.
    """
    
    #Make copies of tlists
    old_theta_list=tlists[n]
    new_theta_list=tlists[n+1]
    old_theta_list_orig=np.copy(tlists[n])
    new_theta_list_orig=np.copy(tlists[n+1])
    
    #same with plists
    old_phi_list=plists[n]
    new_phi_list=plists[n+1]
    old_phi_list_orig=np.copy(plists[n])
    new_phi_list_orig=np.copy(plists[n+1])
    
    old_m_list=mlists[n]
    new_m_list=mlists[n+1]
    
    old_barriers=barriers[n]
    new_barriers=barriers[n+1]
    
    old_theta_mat=tmats[n]
    new_theta_mat=tmats[n+1]
    old_phi_mat=pmats[n]
    new_phi_mat=pmats[n+1]
    
    #Get lengths of old and new barrier numbers
    old_len=len(old_theta_list)
    new_len=len(new_theta_list)

    #Get lengths of 
    cos_dist=np.empty((old_len,new_len))
    new_indices=np.empty((new_len))
    old_indices=np.empty((old_len))
    
    #Create an i  by j matrix of cosine distances
    #Between minima in the old and new array.
    for i in range(old_len):
        for j in range(new_len):
            xyz_old=angle2xyz(old_theta_list[i],old_phi_list[i])
            xyz_new=angle2xyz(new_theta_list[j],new_phi_list[j])
            cos_dist[i,j]=np.dot(xyz_old,xyz_new)
    

    #Correspondence between old and new indices using the cosine distance
    for i in range(old_len):
        old_index=np.where(cos_dist[i]==np.amax(cos_dist[i]))[0]
        if type(old_index)==np.ndarray:
            old_index=old_index[0]
        old_indices[i]=old_index
        

    for j in range(new_len):
        #Assign index to closest variable
        new_index=np.where(cos_dist[:,j]==np.amax(cos_dist[:,j]))[0]
        if type(new_index)==np.ndarray:
            new_index=new_index[0]
        #Clostest minima to new index becomes j
        new_indices[j] = new_index
    old_indices = old_indices.astype(np.int64)
    #Trim out new indices that are in the old array
    #i.e. closest to an index in the old array
    nnew_indices = np.setdiff1d(np.array(range(new_len)),old_indices)
    #Add new indices that in old_array multiple times
    #(i.e. we have a merge going on)
    unique,counts = np.unique(old_indices,return_counts=True)
    mergers = unique[counts>1]
    
    #Get a new set of indices
    nnew_indices = np.append(nnew_indices,mergers)
    
    #Combine everything together
    new_len = len(nnew_indices)
    new_indices = new_indices[nnew_indices]
    
    #Get the new sets of thetas and phis
    new_theta_list = new_theta_list[nnew_indices]
    new_phi_list = new_phi_list[nnew_indices]
    
    #Make the old arrays (thetas, phis and energies for barriers and for
    #minima) the same length.
    old_theta_list=np.append(old_theta_list,new_theta_list)
    new_theta_list=old_theta_list
    
    old_phi_list=np.append(old_phi_list,new_phi_list)
    new_phi_list=old_phi_list
    
    old_m_list=np.append(old_m_list,np.full(new_len,np.inf))
    new_m_list=np.append(np.full(old_len,np.inf),new_m_list)
    
    #Create a set of "new old" and "new new" barriers and minima
    nold_barriers=np.full((old_len+new_len,old_len+new_len),np.inf)
    nnew_barriers=np.full((old_len+new_len,old_len+new_len),np.inf)
    nold_thetas=np.full((old_len+new_len,old_len+new_len),-np.inf)
    nnew_thetas=np.full((old_len+new_len,old_len+new_len),-np.inf)
    nold_phis=np.full((old_len+new_len,old_len+new_len),-np.inf)
    nnew_phis=np.full((old_len+new_len,old_len+new_len),-np.inf)
    
    #Loop through the old set of barriers
    for i in range(old_len):
        for j in range(old_len):
            #Everything gets copied over from the old arrays
            nold_barriers[i,j]=np.copy(old_barriers[i,j])
            nnew_barriers[i,j]=np.copy(old_barriers[i,j])
            nold_thetas[i,j]=np.copy(old_theta_mat[i,j])
            nnew_thetas[i,j]=np.copy(old_theta_mat[i,j])
            nold_phis[i,j]=np.copy(old_phi_mat[i,j])
            nnew_phis[i,j]=np.copy(old_phi_mat[i,j])
    
    #For everything in the NEW set of barriers
    for i in range(old_len+new_len):
        for j in range(old_len,old_len+new_len):
            #If we're in the indices that are the new set of barriers
            if i>=old_len:
                #Copy everything over from the new set of barriers
                nold_barriers[i,j]=np.copy(new_barriers[i-old_len,j-old_len])
                nnew_barriers[i,j]=np.copy(new_barriers[i-old_len,j-old_len])
                nold_barriers[j,i]=np.copy(new_barriers[j-old_len,i-old_len])
                nnew_barriers[j,i]=np.copy(new_barriers[j-old_len,i-old_len])
                
                nold_thetas[i,j]=np.copy(new_theta_mat[i-old_len,j-old_len])
                nnew_thetas[i,j]=np.copy(new_theta_mat[i-old_len,j-old_len])
                nold_thetas[j,i]=np.copy(new_theta_mat[j-old_len,i-old_len])
                nnew_thetas[j,i]=np.copy(new_theta_mat[j-old_len,i-old_len])
                
                nold_phis[i,j]=np.copy(new_phi_mat[i-old_len,j-old_len])
                nnew_phis[i,j]=np.copy(new_phi_mat[i-old_len,j-old_len])
                nold_phis[j,i]=np.copy(new_phi_mat[j-old_len,i-old_len])
                nnew_phis[j,i]=np.copy(new_phi_mat[j-old_len,i-old_len])
            else:
                #For everything else, the barriers are 0 when that state
                #no longer exists, and you want to go to the nearest 
                #neighbor state - the barriers back will stay at 
                #infinity. The logic here is a little confusing, 
                #basically this is enforcing that the states are
                #immediately jumped out of at temperatures where
                #they don't exist.
                nold_barriers[j,int(new_indices[j-old_len])]=0
                nnew_barriers[i,int(old_indices[i]+old_len)]=0

    
    #Remake all the input variables with correct length arrays
    tlists[n]=old_theta_list
    tlists[n+1]=new_theta_list
    plists[n]=old_phi_list
    plists[n+1]=new_phi_list
    mlists[n]=old_m_list
    mlists[n+1]=new_m_list
    barriers[n]=nold_barriers
    barriers[n+1]=nnew_barriers
    tmats[n]=nold_thetas
    tmats[n+1]=nnew_thetas
    pmats[n]=nold_phis
    pmats[n+1]=nnew_phis
    
    #Propogate these new lengths to all the arrays at other temperatures
    #as well.
    #Change everything with the "old_barriers" states.
    for i in range(minstep,n):
        tlists[i]=np.append(tlists[i],new_theta_list_orig)
        plists[i]=np.append(plists[i],new_phi_list_orig)
        mlists[i]=np.append(mlists[i],np.full(new_len,np.inf))
        
        temp_barriers=np.full((old_len+new_len,old_len+new_len),np.inf)
        temp_barriers[:old_len,:old_len]=barriers[i]
        temp_barriers[old_len:]=nold_barriers[old_len:]
        temp_barriers[:,old_len:]=nold_barriers[:,old_len:]
        barriers[i]=temp_barriers
        
        temp_thetas=np.full((old_len+new_len,old_len+new_len),-np.inf)
        temp_thetas[:old_len,:old_len]=tmats[i]
        temp_thetas[old_len:]=nold_thetas[old_len:]
        temp_thetas[:,old_len:]=nold_thetas[:,old_len:]
        tmats[i]=temp_thetas
        
        temp_phis=np.full((old_len+new_len,old_len+new_len),-np.inf)
        temp_phis[:old_len,:old_len]=pmats[i]
        temp_phis[old_len:]=nold_phis[old_len:]
        temp_phis[:,old_len:]=nold_phis[:,old_len:]
        pmats[i]=temp_phis
        
    #Change everything with the "new_barriers" states.
    for i in range(n+2,maxstep):
        tlists[i]=np.append(old_theta_list_orig,tlists[i])
        plists[i]=np.append(old_phi_list_orig,plists[i])
        mlists[i]=np.append(np.full(old_len,np.inf),mlists[i])
        
        temp_barriers=np.full((old_len+new_len,old_len+new_len),np.inf)
        temp_barriers[old_len:,old_len:]=barriers[i]
        temp_barriers[:old_len]=nnew_barriers[:old_len]
        temp_barriers[:,:old_len]=nnew_barriers[:,:old_len]
        barriers[i]=temp_barriers
        
        temp_thetas=np.full((old_len+new_len,old_len+new_len),-np.inf)
        temp_thetas[old_len:,old_len:]=tmats[i]
        temp_thetas[:old_len]=nnew_thetas[:old_len]
        temp_thetas[:,:old_len]=nnew_thetas[:,:old_len]
        tmats[i]=temp_thetas
        
        temp_phis=np.full((old_len+new_len,old_len+new_len),-np.inf)
        temp_phis[old_len:,old_len:]=pmats[i]
        temp_phis[:old_len]=nnew_phis[:old_len]
        temp_phis[:,:old_len]=nnew_phis[:,:old_len]
        pmats[i]=temp_phis
    
    #Remake everything
    tlists=np.array([tlist for tlist in tlists])
    plists=np.array([plist for plist in plists])
    mlists=np.array([mlist for mlist in mlists])
    tmats=np.array([tmat for tmat in tmats])
    pmats=np.array([pmat for pmat in pmats])
    barriers=np.array([barrier for barrier in barriers])
    return(tlists,plists,mlists,tmats,pmats,barriers)


def test_merge(tlists,plists,mlists,tmats,pmats,barriers,n,minstep,maxstep):
    """
    DEPRECATED

    Old version of merge_barriers
    """
    old_theta_list=tlists[n]
    new_theta_list=tlists[n+1]
    old_theta_list_orig=np.copy(tlists[n])
    new_theta_list_orig=np.copy(tlists[n+1])
    
    old_phi_list=plists[n]
    new_phi_list=plists[n+1]
    old_phi_list_orig=np.copy(plists[n])
    new_phi_list_orig=np.copy(plists[n+1])
    
    old_m_list=mlists[n]
    new_m_list=mlists[n+1]
    old_m_list_orig=np.copy(mlists[n])
    new_m_list_orig=np.copy(mlists[n+1])
    
    old_barriers=barriers[n]
    new_barriers=barriers[n+1]
    
    old_theta_mat=tmats[n]
    new_theta_mat=tmats[n+1]
    old_phi_mat=pmats[n]
    new_phi_mat=pmats[n+1]
    
    old_len=len(old_theta_list)
    new_len=len(new_theta_list)
    cos_dist=np.empty((old_len,new_len))
    new_indices=np.empty((new_len))
    old_indices=np.empty((old_len))
    
    #Create an i  by j matrix of cosine distances
    #Between minima in the old and new array.
    for i in range(old_len):
        for j in range(new_len):
            xyz_old=angle2xyz(old_theta_list[i],old_phi_list[i])
            xyz_new=angle2xyz(new_theta_list[j],new_phi_list[j])
            cos_dist[i,j]=np.dot(xyz_old,xyz_new)
    
    #Minimum cosine distances are assigned to the new index in the new array
    
    #Go through old indices
    for i in range(old_len):
        #Find the corresponding "new index" in the old index list.
        old_index=np.where(cos_dist[i]==np.amax(cos_dist[i]))[0]
        if type(old_index)==np.ndarray:
            old_index=old_index[0]
        old_indices[i]=old_index
    
    
    #Propagate backwards first -> find new indices that correspond to old indices
    for j in range(new_len):
        #Assign index to closest variable
        new_index=np.where(cos_dist[:,j]==np.amax(cos_dist[:,j]))[0]
        if type(new_index)==np.ndarray:
            new_index=new_index[0]
        #Clostest minima to new index becomes j
        new_indices[j] = new_index
        

        
    
    
    old_theta_list=np.append(old_theta_list,new_theta_list)
    new_theta_list=old_theta_list
    
    old_phi_list=np.append(old_phi_list,new_phi_list)
    new_phi_list=old_phi_list
    
    old_m_list=np.append(old_m_list,np.full(new_len,np.inf))
    new_m_list=np.append(np.full(old_len,np.inf),new_m_list)
    
    nold_barriers=np.full((old_len+new_len,old_len+new_len),np.inf)
    nnew_barriers=np.full((old_len+new_len,old_len+new_len),np.inf)
    nold_thetas=np.full((old_len+new_len,old_len+new_len),-np.inf)
    nnew_thetas=np.full((old_len+new_len,old_len+new_len),-np.inf)
    nold_phis=np.full((old_len+new_len,old_len+new_len),-np.inf)
    nnew_phis=np.full((old_len+new_len,old_len+new_len),-np.inf)

    
    
    for i in range(old_len):
        for j in range(old_len):
            nold_barriers[i,j]=np.copy(old_barriers[i,j])
            nnew_barriers[i,j]=np.copy(old_barriers[i,j])
            nold_thetas[i,j]=np.copy(old_theta_mat[i,j])
            nnew_thetas[i,j]=np.copy(old_theta_mat[i,j])
            nold_phis[i,j]=np.copy(old_phi_mat[i,j])
            nnew_phis[i,j]=np.copy(old_phi_mat[i,j])
            
    for i in range(old_len+new_len):
        for j in range(old_len,old_len+new_len):
            if i>=old_len:
                nold_barriers[i,j]=np.copy(new_barriers[i-old_len,j-old_len])
                nnew_barriers[i,j]=np.copy(new_barriers[i-old_len,j-old_len])
                nold_barriers[j,i]=np.copy(new_barriers[j-old_len,i-old_len])
                nnew_barriers[j,i]=np.copy(new_barriers[j-old_len,i-old_len])
                
                nold_thetas[i,j]=np.copy(new_theta_mat[i-old_len,j-old_len])
                nnew_thetas[i,j]=np.copy(new_theta_mat[i-old_len,j-old_len])
                nold_thetas[j,i]=np.copy(new_theta_mat[j-old_len,i-old_len])
                nnew_thetas[j,i]=np.copy(new_theta_mat[j-old_len,i-old_len])
                
                nold_phis[i,j]=np.copy(new_phi_mat[i-old_len,j-old_len])
                nnew_phis[i,j]=np.copy(new_phi_mat[i-old_len,j-old_len])
                nold_phis[j,i]=np.copy(new_phi_mat[j-old_len,i-old_len])
                nnew_phis[j,i]=np.copy(new_phi_mat[j-old_len,i-old_len])
            else:
                nold_barriers[j,int(new_indices[j-old_len])]=0
                nnew_barriers[i,int(old_indices[i]+old_len)]=0

    

    tlists[n]=old_theta_list
    tlists[n+1]=new_theta_list
    plists[n]=old_phi_list
    plists[n+1]=new_phi_list
    mlists[n]=old_m_list
    mlists[n+1]=new_m_list
    barriers[n]=nold_barriers
    barriers[n+1]=nnew_barriers
    tmats[n]=nold_thetas
    tmats[n+1]=nnew_thetas
    pmats[n]=nold_phis
    pmats[n+1]=nnew_phis
    

    for i in range(minstep,n):
        tlists[i]=np.append(tlists[i],new_theta_list_orig)
        plists[i]=np.append(plists[i],new_phi_list_orig)
        mlists[i]=np.append(mlists[i],np.full(new_len,np.inf))
        
        temp_barriers=np.full((old_len+new_len,old_len+new_len),np.inf)
        temp_barriers[:old_len,:old_len]=barriers[i]
        temp_barriers[old_len:]=nold_barriers[old_len:]
        temp_barriers[:,old_len:]=nold_barriers[:,old_len:]
        barriers[i]=temp_barriers
        
        temp_thetas=np.full((old_len+new_len,old_len+new_len),-np.inf)
        temp_thetas[:old_len,:old_len]=tmats[i]
        temp_thetas[old_len:]=nold_thetas[old_len:]
        temp_thetas[:,old_len:]=nold_thetas[:,old_len:]
        tmats[i]=temp_thetas
        
        temp_phis=np.full((old_len+new_len,old_len+new_len),-np.inf)
        temp_phis[:old_len,:old_len]=pmats[i]
        temp_phis[old_len:]=nold_phis[old_len:]
        temp_phis[:,old_len:]=nold_phis[:,old_len:]
        pmats[i]=temp_phis
        
        
    for i in range(n+2,maxstep):
        tlists[i]=np.append(old_theta_list_orig,tlists[i])
        plists[i]=np.append(old_phi_list_orig,plists[i])
        mlists[i]=np.append(np.full(old_len,np.inf),mlists[i])
        
        temp_barriers=np.full((old_len+new_len,old_len+new_len),np.inf)
        temp_barriers[old_len:,old_len:]=barriers[i]
        temp_barriers[:old_len]=nnew_barriers[:old_len]
        temp_barriers[:,:old_len]=nnew_barriers[:,:old_len]
        barriers[i]=temp_barriers
        
        temp_thetas=np.full((old_len+new_len,old_len+new_len),-np.inf)
        temp_thetas[old_len:,old_len:]=tmats[i]
        temp_thetas[:old_len]=nnew_thetas[:old_len]
        temp_thetas[:,:old_len]=nnew_thetas[:,:old_len]
        tmats[i]=temp_thetas
        
        temp_phis=np.full((old_len+new_len,old_len+new_len),-np.inf)
        temp_phis[old_len:,old_len:]=pmats[i]
        temp_phis[:old_len]=nnew_phis[:old_len]
        temp_phis[:,:old_len]=nnew_phis[:,:old_len]
        pmats[i]=temp_phis
    
    
    tlists=np.array([tlist for tlist in tlists])
    plists=np.array([plist for plist in plists])
    mlists=np.array([mlist for mlist in mlists])
    tmats=np.array([tmat for tmat in tmats])
    pmats=np.array([pmat for pmat in pmats])
    barriers=np.array([barrier for barrier in barriers])
    return(tlists,plists,mlists,tmats,pmats,barriers)

def reorder(old_theta_list,old_phi_list,new_theta_list,new_phi_list,new_m_list,
            new_theta_mat,new_phi_mat,new_barriers):
    """
    Calculates the set of energy minima at one temperature for a grain
    whichcorrespond to the set of energy minima at another temperature
    using a nearest-neighbor algorithm.

    Inputs
    ------
    old_theta_list,old_phi_list: numpy arrays
    Directions of minima on energy surface at first temperature.

    new_theta_list,new_phi_list: numpy arrays
    Directions of minima on energy surface at second temperature
    
    new_m_list: numpy array
    Energy minima on energy surface at second temperature.

    new_theta_mat,new_phi_list: numpy array
    Directions associated with energy barriers at second temperatures

    new_barriers: numpy array
    Energy barriers at second temperature.

    Returns
    -------
    new_theta_list,new_phi_list: numpy arrays
    Directions of minima on energy surface at second temperature,
    reordered so that ordering is preserved from first temperature.
    
    new_m_list: numpy array
    Energy minima on energy surface at second temperature, reordered so 
    that ordering is preserved from first temperature.

    new_theta_mat,new_phi_list: numpy array
    Directions associated with energy barriers at second temperatures,
    reordered so that ordering is preserved from first temperature.

    new_barriers: numpy array
    Energy barriers at second temperature, reordered so that ordering is 
    preserved from first temperature.
    """
    old_len=len(old_theta_list) #number of old minima
    cos_dist=np.full((old_len,old_len),-np.inf)
    new_indices=np.empty(old_len)
    for i in range(old_len):
        for j in range(old_len):
            xyz_old=angle2xyz(old_theta_list[i],old_phi_list[i])
            xyz_new=angle2xyz(new_theta_list[j],new_phi_list[j])
            cos_dist[i,j]=np.dot(xyz_old,xyz_new)
            #cos_dist[j,i]=np.dot(xyz_old,xyz_new)
    for j in range(old_len):

        new_indices[j]=np.where(cos_dist[j,:]==np.amax(cos_dist[j,:]))[0]
    new_indices=new_indices.astype(int)
    return(new_theta_list[new_indices],new_phi_list[new_indices],new_m_list[new_indices],new_theta_mat[:,new_indices][new_indices],new_phi_mat[:,new_indices][new_indices],new_barriers[:,new_indices][new_indices])

def energy_spline(x,y):
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
    if len(x_fin)<=1:
        t=np.array([min(x),max(x)])
        c=np.array([np.inf,np.inf])
        k=0
    elif np.all(y_fin==0):
        t=np.array([min(x_fin),max(x_fin)])
        c=np.array([0.,0.])
        k=0
    elif len(x_fin)<=3:
        std=min(1,np.ptp(y_fin)/100)
        w=np.full(len(x_fin),1/std)
        t,c,k=splrep(x_fin,y_fin,w=w,task=0,k=1)
    else:
        std=min(1,np.ptp(y_fin)/100)
        w=np.full(len(x_fin),1/std)
        t,c,k=splrep(x_fin,y_fin,w=w,task=0)
    return(t,c,k)
    
def direction_spline(x,y):    
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
    x_fin=x[~np.any(np.isinf(y),axis=1)]
    y_fin=y[~np.any(np.isinf(y),axis=1)]

    if len(x_fin)<=1:
        t=np.array([min(x),max(x)])
        c=np.array([[np.inf,np.inf],[np.inf,np.inf]])
        k=0
    elif len(x_fin)<=3:
        std=0.001
        w=np.full(len(x_fin),1/std)
        tck,u=splprep(angle2xyz(y_fin[:,0],y_fin[:,1]),u=x_fin,w=w,task=0,k=1)
        t,c,k=tck
        c=np.array(c).T
        c_shape=len(t)-len(c)
        c=np.append(c,np.zeros((c_shape,3)),axis=0)
    else:
        std=0.001
        w=np.full(len(x_fin),1/std)
        tck,u=splprep(angle2xyz(y_fin[:,0],y_fin[:,1]),u=x_fin,w=w,task=0)
        t,c,k=tck
        c=np.array(c).T
        c_shape=len(t)-len(c)
        c=np.append(c,np.zeros((c_shape,3)),axis=0)
        
    return(t,c,k)

def energy_result(t,c,k,T):
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
    result=BSpline(t,c,k,extrapolate=False)(T)
    result=np.nan_to_num(result,nan=np.inf)
    return(result)

def direction_result(t,c,k,T):
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
    direction=BSpline(t,c,k,extrapolate=False)(T)
    direction=np.array(xyz2angle(direction.T)).T
    if len(direction.shape)>1:
        direction[:,0]=direction[:,0]%(2*np.pi)
    else:
        direction[0]=direction[0]%(2*np.pi)
    return(np.nan_to_num(direction,nan=-np.inf))

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
    def __init__(self,TMx,alignment,PRO,OBL,T_spacing=1):
        theta_lists,phi_lists,min_energies,theta_mats,phi_mats,energy_mats,\
            Ts=find_T_barriers(TMx,alignment,PRO,OBL,T_spacing=T_spacing)
        arrlens=np.array([len(thetas) for thetas in theta_lists])
        #Check for times where number of minima changes
        diffno=np.where(np.diff(arrlens)!=0)[0]
        
        #Go through each of these segments and reorder so everything is in a consistent order
        if len(diffno)>0:
            for i in range(0,int(diffno[0])):
                theta_lists[i+1],phi_lists[i+1],min_energies[i+1],theta_mats[i+1],phi_mats[i+1],energy_mats[i+1]\
                =reorder(theta_lists[i],phi_lists[i],
                         theta_lists[i+1],phi_lists[i+1],min_energies[i+1],theta_mats[i+1],phi_mats[i+1],energy_mats[i+1])

            for i in range(diffno[0]+1,len(theta_lists)-1):
                theta_lists[i+1],phi_lists[i+1],min_energies[i+1],theta_mats[i+1],phi_mats[i+1],energy_mats[i+1]\
                =reorder(theta_lists[i],phi_lists[i],
                         theta_lists[i+1],phi_lists[i+1],min_energies[i+1],theta_mats[i+1],phi_mats[i+1],energy_mats[i+1])
            
            
            #Now merge barriers that are 
            appended_diffno=np.append(diffno,len(theta_lists))
            for i in range(len(appended_diffno[:-1])):
                n=appended_diffno[i]
                maxstep=appended_diffno[i+1]
                theta_lists,phi_lists,min_energies,theta_mats,phi_mats,energy_mats\
                =merge_barriers(theta_lists,phi_lists,min_energies,theta_mats,phi_mats,energy_mats,n,0,maxstep)

        else:
            theta_lists=np.array(theta_lists)
            phi_lists=np.array(phi_lists)
            min_energies=np.array(min_energies)
            theta_mats=np.array(theta_mats)
            phi_mats=np.array(phi_mats)
            energy_mats=np.array(energy_mats)
        
        min_dir=np.empty(theta_lists.shape[1],dtype='object')
        min_energy=np.empty(theta_lists.shape[1],dtype='object')
        
        bar_dir=np.empty((theta_lists.shape[1],theta_lists.shape[1]),dtype='object')
        bar_energies=np.empty((theta_lists.shape[1],theta_lists.shape[1]),dtype='object')
        for i in range(theta_lists.shape[1]):
            dirs=np.array([theta_lists[:,i],phi_lists[:,i]]).T
            tck=direction_spline(Ts,dirs)
            min_dir[i]=tck
            min_energy[i]=energy_spline(Ts,min_energies[:,i])
            for j in range(theta_lists.shape[1]):
                dirs=np.array([theta_mats[:,i,j],phi_mats[:,i,j]]).T
                tck=direction_spline(Ts,dirs)
                bar_dir[i,j]=tck
                bar_energies[i,j]=energy_spline(Ts,energy_mats[:,i,j])
        
        self.min_dir=min_dir
        self.min_energy=min_energy
        self.bar_dir=bar_dir
        self.bar_energy=bar_energies
        self.TMx=TMx
        self.alignment=alignment
        self.PRO=PRO
        self.OBL=OBL
        self.T_max=max(Ts)
        self.T_min=min(Ts)

    def to_file(self,fname):
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
        with open(fname, 'wb') as f:
            pickle.dump(self, f)
        f.close()
            
    
    def __repr__(self):
        retstr="""Energy Landscape of TM{TMx} Grain with a prolateness of
        {PRO} and an oblateness of {OBL} elongated along the
        magnetocrystalline {alignment} axis.""".format(
        TMx=self.TMx,PRO=self.PRO,OBL=self.OBL,alignment=self.alignment)
        return(retstr)
    
    def get_params(self,T):
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
        rot_mat,k1,k2,Ms=get_material_parms(self.TMx,self.alignment,T)
        min_dir=[]
        min_e=[]

        if type(T)==float or  type(T)==int or type(T)==np.float64 or type(T)==np.int64:
            assert (T<=self.T_max)&(T>=self.T_min),\
            'T must be between '+str(self.T_min)+' and '+str(self.T_max)
            bar_e=np.full(self.bar_energy.shape,np.inf)
            bar_dir=np.full((len(self.min_energy),len(self.min_energy),2),np.inf)

        elif type(T)==np.ndarray:
            assert (np.amax(T)<=self.T_max)&(np.amin(T)>=self.T_min),\
            'T must be between '+str(self.T_min)+' and '+str(self.T_max)
            bar_e=np.full(len(self.min_energy),len(self.min_energy),2,T.shape,np.inf)
            bar_dir=np.full((len(self.min_energy),len(self.min_energy),2,T.shape),np.inf)
        
        else:
            print(T)
            raise TypeError('T should not be type '+str(type(T)))

        for i in range(len(self.min_energy)):
            t,c,k=self.min_energy[i]
            min_e.append(energy_result(t,c,k,T))
            t,c,k=self.min_dir[i]
            min_dir.append(direction_result(t,c,k,T))
            for j in range(len(self.min_energy)):
                t,c,k=self.bar_energy[i,j]
                bar_e[i,j]=energy_result(t,c,k,T)
                t,c,k=self.bar_dir[i,j]
                bar_dir[i,j]=direction_result(t,c,k,T)
                
        bar_e[bar_e>1e308]=np.inf
        bar_e[bar_e<0]=0.
        min_e=np.array(min_e)
        min_e[min_e>1e308]=np.inf
        bar_dir[np.abs(bar_dir)>1e308]=np.inf
        min_dir=np.array(min_dir)
        min_dir[np.abs(min_dir)>1e308]=np.inf
        params={'min_dir':np.array(min_dir),
                'min_e':min_e,
                'bar_dir':np.array(bar_dir),
                'bar_e':np.array(bar_e),
                'T':T,
                'Ms':Ms}
        return(params)
