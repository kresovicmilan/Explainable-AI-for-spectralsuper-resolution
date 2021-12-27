import numpy as np
import scipy.io
from colour import (MSDS_CMFS, SDS_ILLUMINANTS, SpectralDistribution, SpectralShape, sd_to_XYZ)
import colour
import os

def nrm(array):
    array = array - np.min(array)
    return array/np.max(array)

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3

def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, None) for itm in a]  # None can be replaced by any other "not in b" value

"""def ref2xyz(ref, ref_lam, ill, ill_lam, obs, obs_lam):
    cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    illuminant = SDS_ILLUMINANTS['D65']
    common_lam = intersection(ref_lam, intersection(obs_lam, ill_lam))
    #obs = obs[:,np.where(ismember(obs_lam,common_lam))[0]]
    #ill = ill[:,np.where(ismember(ill_lam,common_lam))[0]]
    #ref = ref[:,np.where(ismember(ref_lam,common_lam))[0]]
    #exm = np.where(ismember(obs_lam,common_lam))[0]
    #exm1 = np.where(np.not_equal(ismember(obs_lam,common_lam), None))[0]
    obs = obs[:,np.where(np.not_equal(ismember(obs_lam,common_lam), None))[0]]
    ill = ill[:,np.where(np.not_equal(ismember(ill_lam,common_lam), None))[0]]
    ref = ref[:,np.where(np.not_equal(ismember(ref_lam,common_lam), None))[0]]
    x_fun = np.multiply(obs[0,:],ill)
    y_fun = np.multiply(obs[1,:],ill)
    z_fun = np.multiply(obs[2,:],ill)
    M     = np.transpose(np.concatenate((x_fun,y_fun,z_fun),axis=0))
    xyz   = np.matmul(ref,M)
    xyz_new_list = []
    for idx, r in enumerate(ref):
        if idx % 1000 == 0:
            print(idx)
        sd = SpectralDistribution(r, common_lam)
        xyz_new_list.append(sd_to_XYZ(sd, cmfs, illuminant))
    return ref, ill, obs, xyz, M"""

def ref2xyz(ref, ref_lam, ill, ill_lam, obs, obs_lam):
    common_lam = intersection(ref_lam, intersection(obs_lam, ill_lam))
    #obs = obs[:,np.where(ismember(obs_lam,common_lam))[0]]
    #ill = ill[:,np.where(ismember(ill_lam,common_lam))[0]]
    #ref = ref[:,np.where(ismember(ref_lam,common_lam))[0]]
    obs = obs[:,np.where(np.not_equal(ismember(obs_lam,common_lam), None))[0]]
    ill = ill[:,np.where(np.not_equal(ismember(ill_lam,common_lam), None))[0]]
    ref = ref[:,np.where(np.not_equal(ismember(ref_lam,common_lam), None))[0]]
    x_fun = np.multiply(obs[0,:],ill)
    y_fun = np.multiply(obs[1,:],ill)
    z_fun = np.multiply(obs[2,:],ill)
    M     = np.transpose(np.concatenate((x_fun,y_fun,z_fun),axis=0))
    xyz   = np.matmul(ref,M)
    return ref, ill, obs, xyz, M

def xyz2lab(xyz):
    def f(t):
        delta = 6/29
        if t > (delta**3):
            f = np.cbrt(t)
        else:
            f = t/(3*delta**2) + 4/29
        return f
    x = xyz[:,0]/np.amax(xyz[:,1])
    y = xyz[:,1]/np.amax(xyz[:,1])
    z = xyz[:,2]/np.amax(xyz[:,1])
    # D65
    #xn = 0.950489
    #yn = 1
    #zn = 0.1088840
    l = np.zeros((x.shape[0],))
    a = np.zeros((x.shape[0],))
    b = np.zeros((x.shape[0],))
    for i in range(l.shape[0]-1):
        l[i] = 116*f(y[i])-16
        a[i] = 500*(f(x[i])-f(y[i]))
        b[i] = 200*(f(y[i])-f(z[i]))
    lab = np.concatenate((l,a,b))
    lab = np.reshape(lab,(-1,3))
    return lab

def load_ref(spectral_data_path = os.path.join("hyperspectral", "SOCS.mat")):
    gamut = scipy.io.loadmat(spectral_data_path)
    gamut = np.transpose(gamut["SOCS"])

    # adding 1: to not include the nanometers
    ref_data = gamut[np.where(np.sum(gamut,axis=1)>1)[0][1:],:]
    ref_lam = range(400,710,10)

    return ref_data, ref_lam

def load_obs(obs_path = os.path.join("hyperspectral","CIEobs.mat")):
    tmp     = scipy.io.loadmat(obs_path)
    obs     = np.transpose(tmp["bars"][:,1:4:])
    obs_lam = tmp["bars"][:,0]  
    return obs, obs_lam

def load_ill(ill_path):
    tmp = scipy.io.loadmat(ill_path)
    ill = np.transpose(tmp["Ill"][:,1:])    
    ill_lam = tmp["Ill"][:,0]
    return ill, ill_lam