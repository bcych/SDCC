import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import ellipkinc,ellipeinc
import jax.numpy as jnp
from jax import jit,vmap
from jax.tree_util import Partial
from jax.config import config
config.update("jax_enable_x64", True)

@jit
def angle2xyz(theta,phi):
    """
    Converts from coordinates on a sphere
    surface (theta, phi) radians to 
    cartesian coordinates (x,y,z)
    """
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
def Ez(theta,phi,field_theta,field_phi,field_str,Ms):
    xyz=angle2xyz(theta,phi)
    field_xyz=angle2xyz(field_theta,field_phi)
    
    field_xyz*=field_str
    return(Ms*jnp.dot(xyz,field_xyz))

@jit
def energy_ang(angles,k1,k2,rot_mat,LMN,Ms,ext_field):
    theta,phi = angles
    field_theta,field_phi,field_str = ext_field
    field_theta=jnp.radians(field_theta)
    field_phi=jnp.radians(field_phi)
    Ha=Ea(k1,k2,theta,phi,rot_mat)
    Hd=Ed(LMN,theta,phi,Ms)
    Hz=Ez(theta,phi,field_theta,field_phi,field_str,Ms)
    return(Ha+Hd+Hz)

@jit
def energy_xyz(xyz,k1,k2,rot_mat,LMN,Ms,ext_field):
    theta,phi=xyz2angle(xyz/jnp.linalg.norm(xyz))
    field_theta,field_phi,field_str=ext_field
    field_theta=jnp.radians(field_theta)
    field_phi=jnp.radians(field_phi)
    Ha=Ea(k1,k2,theta,phi,rot_mat)
    Hd=Ed(LMN,theta,phi,Ms)
    Hz=Ez(theta,phi,field_theta,field_phi,field_str,Ms)
    return(jnp.nan_to_num(Ha+Hd,nan=jnp.inf))
    
def calculate_anisotropies(TMx):
    TMx/=100
    Tc = 3.7237e+02*TMx**3 - 6.9152e+02*TMx**2 - 4.1385e+02*TMx**1 + 5.8000e+02
    Tnorm = 20/Tc
    K1 = 1e4 * (-3.5725e+01*TMx**3 + 5.0920e+01*TMx**2 
    - 1.5257e+01*TMx**1 - 1.3579e+00) * (1-Tnorm)**(-6.3643e+00*TMx**2 + 
    2.3779e+00*TMx**1 + 3.0318e+00)
    K2 = 1e4 * (1.5308e+02*TMx**4 - 2.2600e+01*TMx**3 - 
    4.9734e+01*TMx**2 + 1.5822e+01*TMx**1 - 5.5522e-01) * (1-Tnorm)**7.2652e+00

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
    """
    Todo: Incorporate materials.py framework into material parameters

    """
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
        raise ValueError('Error: Temperature should not exceed \
            Curie temperature (%1.0i'%Tc+'°C)')
    Tnorm = T/Tc
    K1 = 1e4 * (-3.5725e+01*TMx**3 + 5.0920e+01*TMx**2 
    - 1.5257e+01*TMx**1 - 1.3579e+00) * (1-Tnorm)**(-6.3643e+00*TMx**2 + 
    2.3779e+00*TMx**1 + 3.0318e+00)
    K2 = 1e4 * (1.5308e+02*TMx**4 - 2.2600e+01*TMx**3 - 
    4.9734e+01*TMx**2 + 1.5822e+01*TMx**1 - 5.5522e-01) * (1-Tnorm)**7.2652e+00
    Ms = (-2.8106e+05*TMx**3 + 5.2850e+05*TMx**2 - 
    7.9381e+05*TMx**1 + 4.9537e+05) * (1-Tnorm)**4.0025e-01
    return(rot_mat,K1,K2,Ms)

@Partial(jit,static_argnums=6)
def energy_surface(k1,k2,rot_mat,Ms,
LMN,ext_field,n_points=100,bounds=jnp.array([[0,2*jnp.pi],[-jnp.pi/2,jnp.pi/2]])):
    thetas=jnp.linspace(bounds[0,0],bounds[0,1],n_points)
    phis=jnp.linspace(bounds[1,0],bounds[1,1],n_points)
    thetas,phis=jnp.meshgrid(thetas,phis)
    energy_temp=lambda theta,phi: energy_ang([theta,phi],k1,k2,rot_mat,LMN,Ms,ext_field)
    energy_temp=vmap(energy_temp)
    energy_array=energy_temp(thetas.flatten(),phis.flatten())
    energies=jnp.reshape(energy_array,thetas.shape)
    return(thetas,phis,energies)

