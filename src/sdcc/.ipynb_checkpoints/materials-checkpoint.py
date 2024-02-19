import numpy as np
class Material:
    def __init__(self,materialDict,anisotropy):
        self.anisotropy = anisotropy;
        self.materialDict = materialDict;
        self.mu = 4 * np.pi * 1e-07;
        self.Tc = self.materialDict['Tc'];
        

    def Ms(self, T):

        return self.materialDict['Ms'](T,self.Tc)

    def Aex(self, T):
        return self.materialDict['Aex'](T,self.Tc)

    def Kd(self, T):
        return (0.5*self.mu*self.Ms(T)**2)

    def exch_len(self,T):
        return(np.sqrt(self.Aex(T)/self.Kd(T)))
    
    def min_exch_len(self,Tmin,Tmax):
        Ts=np.arange(Tmin,Tmax+0.1,0.1)
        exch_lens=self.exch_len(Ts)
        return(min(exch_lens))

    def mesh_sizing(self,ESVD,Tmin,Tmax,availSizes):
        LambdaEx=self.min_exch_len(Tmin,Tmax)
        realMeshSize=min(LambdaEx*0.85,ESVD*1e-9*0.06666)*1e9
        availSizes=np.sort(availSizes)
        closestAvailSize=availSizes[availSizes<realMeshSize][-1]
        return(closestAvailSize)
    
    def anisotropy_constants(self,T):
        if self.anisotropy is not None:
            ks = {}
            for k in self.anisotropy.anisotropyDict.keys():
                ks[k] = self.anisotropy.anisotropyDict[k](T,self.Tc)
            return(ks)
        else:
            return(None)


class SolidSolution:
    def __init__(self,solidSolutionDict,anisotropy=None):
        self.solidSolutionDict=solidSolutionDict
        self.anisotropy = anisotropy

    def composition(self,comp):
        comp/=100
        materialDict={'Tc':self.solidSolutionDict['Tc'](comp),
        'Ms':lambda T,Tc: self.solidSolutionDict['Ms'](comp,T,Tc),
        'Aex':lambda T,Tc: self.solidSolutionDict['Aex'](comp,T,Tc)}

        if self.anisotropy is not None:
            self.anisotropy.solidSolution = True
            new_anisotropy=self.anisotropy.make_functions(comp)
            return(Material(materialDict,new_anisotropy))
        else:
            return(Material(materialDict,new_anisotropy=None))

class Anisotropy:
    def __init__(self,anisotropyDict,solidSolution=False):
        self.solidSolution = solidSolution
        self.anisotropyDict = anisotropyDict
    def make_functions(self,param):
        if self.solidSolution:
            comp = param
            func_dict = {}
            for k in self.anisotropyDict.keys():
                func_dict[k] = lambda T, Tc: self.anisotropyDict[k](comp, T, Tc)
                
            anis_type = type(self)
            return(anis_type(**func_dict))
        else:
            pass

class CubicAnisotropy(Anisotropy):
    def __init__(self,k1,k2):
        self.anisotropyDict = {'k1':k1, 'k2':k2}

TM_dict={'Tc':lambda comp: 3.7237e+02*comp**3 - 6.9152e+02*comp**2 - 4.1385e+02*comp**1 + 5.8000e+02,
'Ms': lambda comp,T,Tc: (-2.8106e+05*comp**3 + 5.2850e+05*comp**2 - 7.9381e+05*comp**1 + 4.9537e+05) * (1-T/Tc)**4.0025e-01,
'Aex': lambda comp,T,Tc: (1e-11 * 1.3838e+00 * ((Tc+273.15)/853.15) * (1-T/Tc)**6.7448e-01)}

TM_k1 = lambda comp,T,Tc: 1e4 * (-3.5725e+01*comp**3 + 5.0920e+01*comp**2 - \
     1.5257e+01*comp**1 - 1.3579e+00) * (1-T/Tc)**(-6.3643e+00*comp**2 + \
         2.3779e+00*comp**1 + 3.0318e+00)

TM_k2 = lambda comp, T, Tc: 1e4 * (1.5308e+02*comp**4 - 2.2600e+01*comp**3 - \
     4.9734e+01*comp**2 + 1.5822e+01*comp**1 - 5.5522e-01) * \
     (1-T/Tc)**7.2652e+00

TM_anis = CubicAnisotropy(TM_k1,TM_k2)
TM = SolidSolution(TM_dict,TM_anis)